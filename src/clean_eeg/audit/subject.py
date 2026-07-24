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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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


def audit_subject(subject_dir: str | Path,
                  *,
                  output_dir: str | Path | None = None,
                  force: bool = False,
                  annotation_only: bool = False,
                  skip_hashes: bool = False,
                  hash_mode: str = "fast",
                  name_dictionary=None,
                  vocab_whitelist: set[str] | None = None,
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

    edf_files = _discover_edf_files(subject_dir)
    previous = _load_previous_audit(output_dir)
    previous_hashes = None
    previous_hash_mode = None
    if previous is not None:
        prev_hash_check = previous.get("checks", {}).get("transfer_integrity", {})
        previous_hashes = prev_hash_check.get("file_hashes")
        previous_hash_mode = prev_hash_check.get("hash_mode")

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
        return merged

    stubs = [p for p in edf_files if p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    recordings = [p for p in edf_files if not p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    # In stub-pair mode annotations live in the sidecars; in inline mode
    # they're embedded in the recordings themselves.
    annotation_carriers = stubs if stubs else recordings

    if annotation_only:
        _run_check(checks, timings, "annotation_phi_scan",
                   lambda: check_annotation_phi_scan(
                       annotation_carriers, name_dictionary=name_dictionary,
                       vocab_whitelist=vocab_whitelist),
                   progress)
    else:
        _run_check(checks, timings, "subject_code_consistency",
                   lambda: check_subject_code_consistency(edf_files), progress)
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
        _run_check(checks, timings, "annotation_phi_scan",
                   lambda: check_annotation_phi_scan(
                       annotation_carriers, name_dictionary=name_dictionary,
                       vocab_whitelist=vocab_whitelist),
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
