"""SHA-256 manifest + transfer-integrity check.

Isolated from the other checks because hashing multi-GB EDF files is
expensive and the operator may want to disable it on slow filesystems.
The orchestrator calls ``check_transfer_integrity`` even under
``--force`` (the always-on integrity check) unless ``--skip-hashes``
is passed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


HASH_BLOCK_SIZE = 1 << 20  # 1 MiB streaming reads


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


def check_transfer_integrity(edf_paths: Iterable[str | Path],
                             *,
                             previous_hashes: dict[str, str] | None = None
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
    """
    current: dict[str, str] = {}
    for p in edf_paths:
        p = Path(p)
        current[p.name] = sha256_of_file(p)

    mismatches: dict[str, dict[str, str]] = {}
    new_files: list[str] = []
    missing_files: list[str] = []

    if previous_hashes is not None:
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
    else:
        status = "pass"

    return {
        "check": "transfer_integrity",
        "status": status,
        "n_files": len(current),
        "first_run": previous_hashes is None,
        "file_hashes": current,
        "mismatches": mismatches,
        "new_files": sorted(new_files),
        "missing_files": sorted(missing_files),
        "issues": issues,
    }
