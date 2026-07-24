"""Sidecar written by ``clean_subject_edf_files`` on successful completion.

``deidentify.json`` in the output directory serves three purposes:

1. **Completion marker.** Its presence means the pipeline finished
   successfully — re-invoking on the same directory can skip straight
   to the transfer step (unless ``--force`` is passed). An interrupted
   run leaves no manifest; the next invocation starts fresh.
2. **Preflight input for the transfer tool.** ``transfer.py`` reads
   ``subject_code`` + ``site_incoming_folder`` from here and refuses
   to run if the file is missing.
3. **Hash manifest for the post-transfer audit.** The audit's
   ``check_transfer_integrity`` can seed ``previous_hashes`` from this
   file on the very first audit run, so bit-rot between de-id and
   upload is detected.

Hashes are computed with :func:`clean_eeg.audit.hashes.sha256_fast_of_file`
so audit and pipeline agree byte-for-byte on what a "fast" digest
covers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from clean_eeg.provenance import _git_provenance, _package_version

# ``clean_eeg.audit.hashes`` is imported lazily inside ``compute_file_hashes``
# to avoid a circular import: ``audit.__init__`` pulls in ``audit.subject`` →
# ``audit.checks`` → ``clean_eeg.clean_subject_eeg`` (which imports THIS
# module at top level). Deferring the import until it's actually needed
# means ``clean_subject_eeg`` can import this module at parse time without
# forcing the audit chain to load simultaneously.


MANIFEST_FILENAME = "deidentify.json"
SCHEMA_VERSION = 1


@dataclass
class ReviewEvent:
    """One item worth surfacing in the end-of-run 'Human review needed'
    block. Serialized verbatim into ``review_events`` in the manifest.

    ``kind`` values used today:
      - ``annotation_redaction``: Presidio flagged an annotation. Extra
        fields: ``file``, ``redacted_value``, ``boilerplate`` (bool —
        matched the site's boilerplate whitelist; can be suppressed in
        the review print).
      - ``header_truncation``: pyedflib truncated a header field on
        write. Extra fields: ``file``, ``message`` (verbatim pyedflib
        warning; ``field`` is not parsed out because the warning text
        is unstructured).
    """
    kind: str
    file: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "file": self.file, **self.details}


class ManifestSchemaError(ValueError):
    """Raised when a loaded manifest has an unsupported schema version.
    The transfer tool and audit both refuse to consume such a file
    rather than silently misinterpret its fields."""


def compute_file_hashes(paths: Iterable[Path]) -> tuple[dict, dict, dict]:
    """Return ``(hashes, modes, details)`` for a set of files, all keyed
    by ``path.name``. Uses the audit's fast-hash function so the audit
    can verify byte-identity later without agreeing separately on a
    coverage window.
    """
    from clean_eeg.audit.hashes import sha256_fast_of_file  # lazy: see module docstring
    hashes: dict[str, str] = {}
    modes: dict[str, str] = {}
    details: dict[str, dict] = {}
    for p in paths:
        p = Path(p)
        digest, mode_used, det = sha256_fast_of_file(p)
        hashes[p.name] = digest
        modes[p.name] = mode_used
        details[p.name] = det
    return hashes, modes, details


def build_manifest(*,
                   subject_code: str,
                   site_code: str,
                   site_incoming_folder: str,
                   input_path: str,
                   output_path: str,
                   inplace: bool,
                   output_edf_paths: Iterable[Path],
                   n_files_deidentified: int,
                   n_files_failed: int,
                   n_files_quarantined: int,
                   review_events: Iterable[ReviewEvent] = (),
                   ) -> dict:
    """Assemble the manifest dict. Pure — no I/O side effects other than
    reading the EDF files (via fast-hash). Callers wrap this in
    :func:`write_manifest` to persist it.
    """
    sha, dirty = _git_provenance()
    hashes, modes, details = compute_file_hashes(output_edf_paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clean_eeg_version": _package_version("clean_eeg"),
        "git_commit": sha,
        "git_dirty": dirty,
        "subject_code": subject_code,
        "site_code": site_code,
        "site_incoming_folder": site_incoming_folder,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "inplace": inplace,
        "n_files_deidentified": n_files_deidentified,
        "n_files_failed": n_files_failed,
        "n_files_quarantined": n_files_quarantined,
        "hash_mode": "fast",
        "file_hashes": hashes,
        "hash_mode_by_file": modes,
        "hash_details_by_file": details,
        "review_events": [e.to_dict() for e in review_events],
    }


def write_manifest(output_path: str | Path, manifest: dict) -> Path:
    """Persist the manifest to ``<output_path>/deidentify.json`` and
    return the written path."""
    out = Path(output_path) / MANIFEST_FILENAME
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return out


def read_manifest(output_path: str | Path) -> dict | None:
    """Return the parsed manifest dict, or ``None`` if the file is not
    present. Raises :class:`ManifestSchemaError` on an unsupported
    schema version — silently returning stale/incompatible data would
    let a stale manifest bypass the transfer tool's preflight."""
    p = Path(output_path) / MANIFEST_FILENAME
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    v = data.get("schema_version")
    if v != SCHEMA_VERSION:
        raise ManifestSchemaError(
            f"{p}: schema_version={v!r} but this build only understands "
            f"schema_version={SCHEMA_VERSION}. Regenerate with the current "
            "clean_eeg, or upgrade the tool that reads the manifest."
        )
    return data


def manifest_exists(output_path: str | Path) -> bool:
    """Cheap presence check — does not open or parse the file."""
    return (Path(output_path) / MANIFEST_FILENAME).exists()
