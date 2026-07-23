"""Per-subject audit orchestrator.

Runs every check, assembles the results dict, writes ``edf_audit.json``
to the subject directory, and (unless suppressed) renders the audit
notebook + HTML alongside. Also handles idempotent-skip, ``--force``
re-run, and ``--annotation-only`` fast-path semantics.

The transfer-integrity (hash-manifest) check runs even under
``--force`` because that's the operator's always-on guarantee that
subsequent audits catch bit rot on disk — see
[`hashes.py`](hashes.py).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from clean_eeg.audit.annotations import check_annotation_phi_scan
from clean_eeg.audit.checks import (
    check_annotation_pairing,
    check_byte_geometry,
    check_header_phi_residue,
    check_recording_gaps,
    check_signal_header_uniformity,
    check_subject_code_consistency,
)
from clean_eeg.audit.hashes import check_transfer_integrity
from clean_eeg.audit.logs import LOG_FILENAME, check_log_file
from clean_eeg.print_edf_header import ANNOTATION_STUB_SUFFIX


AUDIT_JSON_FILENAME = "edf_audit.json"


def _discover_edf_files(subject_dir: Path) -> list[Path]:
    """Return all *.edf files in a subject dir (recordings + stubs)."""
    return sorted(p for p in subject_dir.iterdir()
                  if p.is_file() and p.suffix.lower() == ".edf")


def _load_previous_audit(subject_dir: Path) -> dict | None:
    p = subject_dir / AUDIT_JSON_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def audit_subject(subject_dir: str | Path,
                  *,
                  force: bool = False,
                  annotation_only: bool = False,
                  skip_hashes: bool = False,
                  name_dictionary=None,
                  vocab_whitelist: set[str] | None = None,
                  ) -> dict:
    """Run the full audit on a single subject directory.

    Returns the audit-results dict (also written to
    ``edf_audit.json``). Under ``force=False``, if an audit already
    exists the transfer-integrity hash check still runs against the
    prior manifest and the result is returned without redoing the rest
    of the checks — this catches on-disk changes cheaply.
    """
    subject_dir = Path(subject_dir)
    if not subject_dir.is_dir():
        raise NotADirectoryError(f"{subject_dir} is not a directory")

    edf_files = _discover_edf_files(subject_dir)
    previous = _load_previous_audit(subject_dir)
    previous_hashes = None
    if previous is not None:
        prev_hash_check = previous.get("checks", {}).get("transfer_integrity", {})
        previous_hashes = prev_hash_check.get("file_hashes")

    checks: dict[str, dict] = {}
    if not skip_hashes:
        checks["transfer_integrity"] = check_transfer_integrity(
            edf_files, previous_hashes=previous_hashes)

    if previous is not None and not force:
        # Idempotent skip: keep prior check results, only refresh the
        # hash step to catch on-disk changes.
        merged = dict(previous)
        merged.setdefault("checks", {})
        if not skip_hashes:
            merged["checks"]["transfer_integrity"] = checks["transfer_integrity"]
        merged["skipped"] = True
        merged["generated_at"] = previous.get("generated_at")
        merged["rechecked_at"] = datetime.now(timezone.utc).isoformat()
        _write_audit_json(subject_dir, merged)
        return merged

    stubs = [p for p in edf_files if p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    recordings = [p for p in edf_files if not p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    # In stub-pair mode annotations live in the sidecars; in inline mode
    # they're embedded in the recordings themselves.
    annotation_carriers = stubs if stubs else recordings

    if annotation_only:
        checks["annotation_phi_scan"] = check_annotation_phi_scan(
            annotation_carriers, name_dictionary=name_dictionary,
            vocab_whitelist=vocab_whitelist)
    else:
        checks["subject_code_consistency"] = check_subject_code_consistency(edf_files)
        checks["header_phi_residue"] = check_header_phi_residue(edf_files)
        checks["recording_gaps"] = check_recording_gaps(recordings)
        checks["byte_geometry"] = check_byte_geometry(edf_files)
        checks["annotation_pairing"] = check_annotation_pairing(edf_files)
        checks["signal_header_uniformity"] = check_signal_header_uniformity(recordings)
        checks["annotation_phi_scan"] = check_annotation_phi_scan(
            annotation_carriers, name_dictionary=name_dictionary,
            vocab_whitelist=vocab_whitelist)
        checks["log_file"] = check_log_file(subject_dir / LOG_FILENAME
                                            if (subject_dir / LOG_FILENAME).exists()
                                            else None)

    subject_code = checks.get("subject_code_consistency", {}).get("subject_code")
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject_dir": str(subject_dir),
        "subject_code": subject_code,
        "n_files": len(edf_files),
        "mode": "annotation_only" if annotation_only else "full",
        "checks": checks,
        "overall_status": _overall_status(checks),
    }
    _write_audit_json(subject_dir, audit)
    return audit


def _overall_status(checks: dict[str, dict]) -> str:
    statuses = {r.get("status", "fail") for r in checks.values()}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _write_audit_json(subject_dir: Path, audit: dict) -> None:
    (subject_dir / AUDIT_JSON_FILENAME).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str)
    )
