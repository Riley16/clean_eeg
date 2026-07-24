"""SHA-256 manifest + transfer-integrity check.

Isolated from the other checks because hashing multi-GB EDF files is
expensive and the operator may want a cheaper spot-check on slow
filesystems. Three modes:

  - ``full``   — hash every byte. Definitive but O(filesize). At ~500
                 MB/s SHA-256 this dominates the audit on multi-GB files.
  - ``fast``   — hash the header + a short window at the start, middle,
                 and end of the data body (default 2 s each). Catches
                 header tampering, truncation, and endpoint bit-rot at
                 O(header + 3 * window) cost. Falls back to ``full``
                 automatically when the file is short enough that the
                 windows would overlap the entire body.
  - ``none``   — skip hashing entirely (``--skip-hashes`` legacy path).

The orchestrator calls ``check_transfer_integrity`` even under
``--force`` (the always-on integrity check) unless the operator opts
into ``none`` mode.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from clean_eeg.print_edf_header import (
    MAIN_HEADER_BYTES,
    SIGNAL_HEADER_BYTES_PER_SIGNAL,
    read_main_header,
    read_signal_headers,
)


HASH_BLOCK_SIZE = 1 << 20  # 1 MiB streaming reads

FAST_HASH_WINDOW_SECONDS = 2.0

VALID_HASH_MODES = ("full", "fast", "none")


def sha256_of_file(path: str | Path, *, block_size: int = HASH_BLOCK_SIZE) -> str:
    """Streaming SHA-256 of a file. Returns lowercase hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_fast_of_file(path: str | Path,
                        *,
                        window_seconds: float = FAST_HASH_WINDOW_SECONDS,
                        ) -> tuple[str, str, dict]:
    """Hash the EDF header + a short window at the start, middle, and
    end of the data records.

    Returns ``(hex_digest, mode_used, details)`` where:
      - ``mode_used`` is ``'fast'`` when the head/mid/tail sampling was
        applied, or ``'full'`` if the file is short enough that all
        three windows would overlap and we hashed the entire file
        instead (which gives the same coverage at lower complexity).
      - ``details`` records the header/window byte ranges actually
        hashed, so the JSON manifest is self-describing (an auditor
        reading ``edf_audit.json`` can verify what "fast" meant on a
        given file).

    Falls back to ``sha256_of_file`` (mode ``full``) when the header is
    unparseable — the audit still gets *a* stable hash, just not one
    that skips reading the whole file.
    """
    path = Path(path)
    file_size = path.stat().st_size

    header = read_main_header(str(path))
    n_signals = header.get("n_signals")
    n_records = header.get("n_records")
    rec_dur = header.get("record_duration")
    if (not isinstance(n_signals, int) or not isinstance(n_records, int)
            or not isinstance(rec_dur, float)
            or n_signals <= 0 or n_records <= 0 or rec_dur <= 0):
        return sha256_of_file(path), "full", {
            "reason": "unparseable header — fell back to full-file hash",
            "file_size": file_size,
        }

    sigs = read_signal_headers(str(path), n_signals)
    spr = [s.get("samples_per_record") for s in sigs]
    if not all(isinstance(v, int) and v > 0 for v in spr):
        return sha256_of_file(path), "full", {
            "reason": "unparseable samples_per_record — fell back to full-file hash",
            "file_size": file_size,
        }

    header_bytes = MAIN_HEADER_BYTES + n_signals * SIGNAL_HEADER_BYTES_PER_SIGNAL
    record_bytes = sum(spr) * 2  # int16 samples
    data_bytes = file_size - header_bytes
    if record_bytes <= 0 or data_bytes <= 0:
        return sha256_of_file(path), "full", {
            "reason": "no data records on disk — fell back to full-file hash",
            "file_size": file_size,
        }

    records_per_window = max(1, int(round(window_seconds / rec_dur)))
    window_bytes = records_per_window * record_bytes
    total_data_records = data_bytes // record_bytes

    # If the three windows would cover the whole body, hash the whole
    # thing — same coverage, simpler / no overlap accounting.
    if records_per_window * 3 >= total_data_records:
        return sha256_of_file(path), "full", {
            "reason": "file too short for windowed sampling — hashed in full",
            "file_size": file_size,
            "records_per_window": records_per_window,
            "total_data_records": total_data_records,
        }

    start_offset = header_bytes
    mid_record = (total_data_records - records_per_window) // 2
    mid_offset = header_bytes + mid_record * record_bytes
    end_offset = header_bytes + (total_data_records - records_per_window) * record_bytes

    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(0)
        h.update(f.read(header_bytes))
        for offset in (start_offset, mid_offset, end_offset):
            f.seek(offset)
            h.update(f.read(window_bytes))
    return h.hexdigest(), "fast", {
        "file_size": file_size,
        "header_bytes": header_bytes,
        "record_bytes": record_bytes,
        "records_per_window": records_per_window,
        "window_bytes": window_bytes,
        "window_offsets": {
            "start": start_offset,
            "middle": mid_offset,
            "end": end_offset,
        },
    }


def check_transfer_integrity(edf_paths: Iterable[str | Path],
                             *,
                             previous_hashes: dict[str, str] | None = None,
                             previous_hash_mode: str | None = None,
                             hash_mode: str = "fast",
                             window_seconds: float = FAST_HASH_WINDOW_SECONDS,
                             ) -> dict:
    """Compute SHA-256 of every provided file, compare against
    ``previous_hashes`` if given.

    ``previous_hashes`` maps ``file.name`` (basename) → hex digest, as
    recovered from a prior ``edf_audit.json``. On first run, pass
    ``None`` — every file is recorded with no comparison. On subsequent
    runs, mismatches, additions, and removals are all surfaced.

    ``status``:
      - ``pass`` — first run, or every file present and matching.
      - ``fail`` — any hash mismatch OR any file listed in
        ``previous_hashes`` is now missing.
    New files that weren't in ``previous_hashes`` are additive and do
    not fail — they're recorded under ``new_files``.

    ``hash_mode`` selects the hashing strategy (see module docstring).
    A per-file mode is recorded because ``fast`` can fall back to
    ``full`` on short files; a comparison against ``previous_hashes``
    only holds when both runs used the same effective mode.
    """
    if hash_mode not in VALID_HASH_MODES:
        raise ValueError(
            f"hash_mode must be one of {VALID_HASH_MODES!r}, got {hash_mode!r}")

    current: dict[str, str] = {}
    per_file_mode: dict[str, str] = {}
    per_file_details: dict[str, dict] = {}

    if hash_mode == "none":
        # No hashing — but still return a well-formed result so the
        # orchestrator can pass this into the audit JSON alongside the
        # rest of the checks.
        return {
            "check": "transfer_integrity",
            "status": "warn",
            "hash_mode": "none",
            "n_files": 0,
            "first_run": previous_hashes is None,
            "file_hashes": {},
            "hash_mode_by_file": {},
            "hash_details_by_file": {},
            "mismatches": {},
            "new_files": [],
            "missing_files": [],
            "issues": ["hash_mode='none' — file-integrity check was not run"],
        }

    for p in edf_paths:
        p = Path(p)
        if hash_mode == "full":
            current[p.name] = sha256_of_file(p)
            per_file_mode[p.name] = "full"
        else:  # fast
            digest, mode_used, details = sha256_fast_of_file(
                p, window_seconds=window_seconds)
            current[p.name] = digest
            per_file_mode[p.name] = mode_used
            per_file_details[p.name] = details

    mismatches: dict[str, dict[str, str]] = {}
    new_files: list[str] = []
    missing_files: list[str] = []

    # A hash-mode change makes the stored digests uncomparable — full and
    # fast digests have no relationship. Rather than fire a spurious
    # mismatch for every file, we skip the comparison, treat this as a
    # first run (record new hashes) and surface a warning so the operator
    # knows the integrity check reset.
    mode_changed = (
        previous_hash_mode is not None
        and previous_hash_mode != hash_mode
    )
    if previous_hashes is not None and not mode_changed:
        for name, digest in current.items():
            if name not in previous_hashes:
                new_files.append(name)
            elif previous_hashes[name] != digest:
                mismatches[name] = {"stored": previous_hashes[name], "current": digest}
        for name in previous_hashes:
            if name not in current:
                missing_files.append(name)

    issues: list[str] = []
    if not current:
        status = "fail"
        issues.append("No EDF files were provided")
    elif mismatches or missing_files:
        status = "fail"
        for name, pair in mismatches.items():
            issues.append(
                f"{name}: hash changed since prior audit "
                f"(stored {pair['stored'][:12]}…, current {pair['current'][:12]}…)"
            )
        for name in missing_files:
            issues.append(f"{name}: file listed in prior audit but not present now")
    elif mode_changed:
        status = "warn"
        issues.append(
            f"hash_mode changed since prior audit "
            f"({previous_hash_mode!r} → {hash_mode!r}); recorded new hashes "
            "and skipped comparison against the prior manifest"
        )
    else:
        status = "pass"

    return {
        "check": "transfer_integrity",
        "status": status,
        "hash_mode": hash_mode,
        "n_files": len(current),
        "first_run": previous_hashes is None,
        "file_hashes": current,
        "hash_mode_by_file": per_file_mode,
        "hash_details_by_file": per_file_details,
        "mismatches": mismatches,
        "new_files": sorted(new_files),
        "missing_files": sorted(missing_files),
        "issues": issues,
    }
