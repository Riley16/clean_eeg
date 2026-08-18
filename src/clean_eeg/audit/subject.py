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
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
from clean_eeg.audit.annotations import check_annotation_phi_scan
from clean_eeg.audit.checks import (
    check_annotation_pairing,
    check_byte_geometry,
    check_filename_convention,
    check_header_phi_residue,
    check_recording_gaps,
    check_signal_header_uniformity,
    check_subject_code_consistency,
)
from clean_eeg.audit.hashes import check_transfer_integrity
from clean_eeg.audit.logs import LOG_FILENAME, check_log_file
from clean_eeg.print_edf_header import ANNOTATION_STUB_SUFFIX


AUDIT_JSON_FILENAME = "edf_audit.json"
IN_PROGRESS_FILENAME = "edf_audit.in_progress"


class AuditInterruptedError(RuntimeError):
    """Raised when ``audit_subject`` finds an in-progress sentinel from a
    prior run that didn't complete (Ctrl-C, crash, SIGKILL, cluster
    preemption). The prior run left no ``edf_audit.json`` — silently
    starting over would hide from the operator that anything went
    wrong. Requires an explicit ``force=True`` to clear the sentinel
    and re-run.

    ``sentinel_path`` is the file to inspect (or delete manually); the
    remaining attributes come from the sentinel content and describe
    when/where the prior run started so operators can correlate with
    cluster logs.
    """

    def __init__(self, sentinel_path: Path, *, started_at: str | None,
                 hostname: str | None, pid: int | None):
        self.sentinel_path = sentinel_path
        self.started_at = started_at
        self.hostname = hostname
        self.pid = pid
        detail = f"started {started_at or '?'} on {hostname or '?'} (pid={pid or '?'})"
        super().__init__(
            f"Previous audit was interrupted ({detail}); left "
            f"{sentinel_path}. Re-run with force=True to wipe the "
            f"sentinel and audit from scratch, or delete the sentinel "
            f"file manually."
        )


def _read_sentinel(sentinel_path: Path) -> dict:
    """Return the sentinel's recorded metadata, or an empty dict if the
    file is missing or unreadable. Never raises — a corrupted sentinel
    still signals interruption; we just can't tell the operator when."""
    try:
        return json.loads(sentinel_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_sentinel(sentinel_path: Path) -> None:
    """Record the start of an audit run to disk. Content is a small
    self-describing JSON so a stale sentinel from days-ago is still
    diagnostic (timestamp + host + pid)."""
    sentinel_path.write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }))


# ``phase`` is 'start' when a check begins and 'end' when it finishes.
# On 'end' the callback also receives the elapsed seconds and the check's
# status string ('pass'/'warn'/'fail') so a CLI can stream one-line
# updates as the audit progresses.
ProgressCallback = Callable[..., None]


def _run_check(checks: dict, timings: dict, name: str, fn,
               progress: ProgressCallback | None) -> dict:
    """Run one check under a stopwatch, wire progress events, store the
    result + elapsed time under ``name``. Returns the check's result dict
    so callers can chain further logic on it (e.g. subject-code extraction).
    """
    if progress is not None:
        progress(name=name, phase="start")
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    checks[name] = result
    timings[name] = dt
    if progress is not None:
        progress(name=name, phase="end", elapsed_s=dt,
                 status=result.get("status", "?"))
    return result


def _discover_edf_files(subject_dir: Path) -> list[Path]:
    """Return all *.edf files in a subject dir (recordings + stubs)."""
    return sorted(p for p in subject_dir.iterdir()
                  if p.is_file() and p.suffix.lower() == ".edf")


def _load_previous_audit(output_dir: Path) -> dict | None:
    p = output_dir / AUDIT_JSON_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _seed_hashes_from_deidentify_manifest(subject_dir: Path
                                           ) -> tuple[dict | None, str | None]:
    """Return ``(file_hashes, hash_mode)`` from ``deidentify.json`` in
    ``subject_dir`` if present, else ``(None, None)``.

    The pipeline writes this sidecar on successful completion, so if
    it's there we can compare the audit's fresh hashes against the
    de-identification-time hashes. Any drift indicates on-disk change
    between de-id and the first audit — usually transfer damage, less
    often disk-level corruption. Lazy-imported to avoid a hard
    dependency on the top-level ``deidentify_manifest`` module (which
    itself imports from ``audit.hashes`` — deferring here keeps the
    audit importable in environments that haven't installed the whole
    pipeline)."""
    try:
        from clean_eeg.deidentify_manifest import (
            ManifestSchemaError,
            read_manifest,
        )
    except ImportError:
        return None, None
    try:
        manifest = read_manifest(subject_dir)
    except (ManifestSchemaError, OSError, json.JSONDecodeError):
        # Malformed manifest — better to skip the comparison than
        # crash the audit. The audit still records fresh hashes.
        return None, None
    if manifest is None:
        return None, None
    return manifest.get("file_hashes"), manifest.get("hash_mode")


def audit_subject(subject_dir: str | Path,
                  *,
                  output_dir: str | Path | None = None,
                  force: bool = False,
                  annotation_only: bool = False,
                  skip_hashes: bool = False,
                  hash_mode: str = "fast",
                  name_dictionary=None,
                  vocab_whitelist: set[str] | None = None,
                  boilerplate_whitelist: BoilerplateWhitelist | None = None,
                  progress: ProgressCallback | None = None,
                  ) -> dict:
    """Run the full audit on a single subject directory.

    Returns the audit-results dict (also written to
    ``<output_dir>/edf_audit.json``). ``output_dir`` defaults to
    ``subject_dir`` — override to avoid writing into read-only fixture
    dirs or shared archive locations.

    Under ``force=False``, if a prior audit exists in ``output_dir``
    the transfer-integrity hash check still runs against the prior
    manifest and the result is returned without redoing the rest of
    the checks — this catches on-disk changes cheaply.
    """
    subject_dir = Path(subject_dir)
    if not subject_dir.is_dir():
        raise NotADirectoryError(f"{subject_dir} is not a directory")
    output_dir = Path(output_dir) if output_dir is not None else subject_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Interruption sentinel: a prior run that didn't complete leaves
    # this file behind (we only remove it on successful completion,
    # below). Refusing to silently start over is the whole point — the
    # operator must acknowledge via ``force=True``.
    sentinel_path = output_dir / IN_PROGRESS_FILENAME
    if sentinel_path.exists():
        if not force:
            meta = _read_sentinel(sentinel_path)
            raise AuditInterruptedError(
                sentinel_path,
                started_at=meta.get("started_at"),
                hostname=meta.get("hostname"),
                pid=meta.get("pid"),
            )
        sentinel_path.unlink()
    _write_sentinel(sentinel_path)

    edf_files = _discover_edf_files(subject_dir)
    previous = _load_previous_audit(output_dir)
    previous_hashes = None
    previous_hash_mode = None
    if previous is not None:
        prev_hash_check = previous.get("checks", {}).get("transfer_integrity", {})
        previous_hashes = prev_hash_check.get("file_hashes")
        previous_hash_mode = prev_hash_check.get("hash_mode")
    else:
        # First audit run on this subject — seed previous_hashes from
        # the de-identification manifest (deidentify.json) if the
        # pipeline wrote one. This catches bit-rot between de-id and
        # the first audit (transfer over slow/unreliable link, disk
        # corruption on the ingest server, etc.). Absent manifest is
        # fine — the check reverts to "first run, record only".
        seed_hashes, seed_mode = _seed_hashes_from_deidentify_manifest(subject_dir)
        if seed_hashes is not None:
            previous_hashes = seed_hashes
            previous_hash_mode = seed_mode

    checks: dict[str, dict] = {}
    timings: dict[str, float] = {}
    # ``skip_hashes`` is the legacy flag; ``hash_mode='none'`` is the
    # new spelling of the same intent. Either one suppresses hashing.
    effective_hash_mode = "none" if skip_hashes else hash_mode
    if effective_hash_mode != "none":
        _run_check(checks, timings, "transfer_integrity",
                   lambda: check_transfer_integrity(
                       edf_files, previous_hashes=previous_hashes,
                       previous_hash_mode=previous_hash_mode,
                       hash_mode=effective_hash_mode),
                   progress)

    if previous is not None and not force:
        # Idempotent skip: keep prior check results, only refresh the
        # hash step to catch on-disk changes.
        merged = dict(previous)
        merged.setdefault("checks", {})
        if effective_hash_mode != "none":
            merged["checks"]["transfer_integrity"] = checks["transfer_integrity"]
        merged["skipped"] = True
        merged["generated_at"] = previous.get("generated_at")
        merged["rechecked_at"] = datetime.now(timezone.utc).isoformat()
        merged["_timings_by_check_s"] = timings
        _write_audit_json(output_dir, merged)
        sentinel_path.unlink(missing_ok=True)
        return merged

    stubs = [p for p in edf_files if p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    recordings = [p for p in edf_files if not p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    # In stub-pair mode annotations live in the sidecars; in inline mode
    # they're embedded in the recordings themselves.
    annotation_carriers = stubs if stubs else recordings

    if annotation_only:
        # In annotation-only mode we don't run subject_code_consistency,
        # so site_code stays None → only the shared boilerplate bucket
        # applies. That's the safe default when we don't know the site.
        _run_check(checks, timings, "annotation_phi_scan",
                   lambda: check_annotation_phi_scan(
                       annotation_carriers, name_dictionary=name_dictionary,
                       vocab_whitelist=vocab_whitelist,
                       boilerplate_whitelist=boilerplate_whitelist,
                       site_code=None),
                   progress)
    else:
        _run_check(checks, timings, "subject_code_consistency",
                   lambda: check_subject_code_consistency(edf_files), progress)
        _run_check(checks, timings, "filename_convention",
                   lambda: check_filename_convention(edf_files), progress)
        _run_check(checks, timings, "header_phi_residue",
                   lambda: check_header_phi_residue(edf_files), progress)
        _run_check(checks, timings, "recording_gaps",
                   lambda: check_recording_gaps(recordings), progress)
        _run_check(checks, timings, "byte_geometry",
                   lambda: check_byte_geometry(edf_files), progress)
        _run_check(checks, timings, "annotation_pairing",
                   lambda: check_annotation_pairing(edf_files), progress)
        _run_check(checks, timings, "signal_header_uniformity",
                   lambda: check_signal_header_uniformity(recordings), progress)
        # Derive site_code from the subject_code that
        # subject_code_consistency parsed off the patient_id field —
        # last letter is the site by convention (see SITE_CODE_TO_INCOMING_FOLDER).
        _sc = checks.get("subject_code_consistency", {}).get("subject_code")
        _site_code = _sc[-1] if _sc else None
        _run_check(checks, timings, "annotation_phi_scan",
                   lambda: check_annotation_phi_scan(
                       annotation_carriers, name_dictionary=name_dictionary,
                       vocab_whitelist=vocab_whitelist,
                       boilerplate_whitelist=boilerplate_whitelist,
                       site_code=_site_code),
                   progress)
        _run_check(checks, timings, "log_file",
                   lambda: check_log_file(
                       subject_dir / LOG_FILENAME
                       if (subject_dir / LOG_FILENAME).exists() else None),
                   progress)

    subject_code = checks.get("subject_code_consistency", {}).get("subject_code")
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject_dir": str(subject_dir),
        "output_dir": str(output_dir),
        "subject_code": subject_code,
        "n_files": len(edf_files),
        "mode": "annotation_only" if annotation_only else "full",
        "checks": checks,
        "overall_status": _overall_status(checks),
        "_timings_by_check_s": timings,
    }
    _write_audit_json(output_dir, audit)
    sentinel_path.unlink(missing_ok=True)
    return audit


def _overall_status(checks: dict[str, dict]) -> str:
    statuses = {r.get("status", "fail") for r in checks.values()}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _write_audit_json(output_dir: Path, audit: dict) -> None:
    (output_dir / AUDIT_JSON_FILENAME).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str)
    )
