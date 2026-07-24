"""Audit-performance benchmark tests.

Gated behind the ``audit_benchmark`` marker so the ~1.8 GB dataset is
only generated when someone explicitly opts in. Run with::

    pytest -m audit_benchmark tests/test_audit_benchmark.py -s

``-s`` disables pytest output capture so per-check timings stream to
the terminal as the audit runs.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from clean_eeg.audit.subject import audit_subject
from clean_eeg.paths import TEST_DATA_DIR
from tests.generate_audit_benchmark import (
    BENCHMARK_SUBJECT_CODE,
    build_benchmark_subject,
)


BENCHMARK_DIR = TEST_DATA_DIR / "benchmark_audit"
SUBJECT_DIR = BENCHMARK_DIR / BENCHMARK_SUBJECT_CODE


@pytest.fixture(scope="session")
def benchmark_subject_dir() -> Path:
    """Lazily materialize the benchmark subject dir. Only runs when a
    test using this fixture is selected — the ``audit_benchmark`` marker
    is the gate."""
    print(f"\n[benchmark] ensuring dataset at {SUBJECT_DIR} …")
    t0 = time.perf_counter()
    build_benchmark_subject(SUBJECT_DIR)
    dt = time.perf_counter() - t0
    total_mb = sum(p.stat().st_size for p in SUBJECT_DIR.glob("*.edf")) / (1024 ** 2)
    print(f"[benchmark] dataset ready ({total_mb:.0f} MB) in {dt:.1f} s")
    return SUBJECT_DIR


@pytest.mark.audit_benchmark
def test_audit_subject_timing(benchmark_subject_dir: Path, tmp_path: Path) -> None:
    """Run a full audit and print per-check + total timings.

    Not an assertion test — the purpose is to eyeball where time goes.
    Uses an empty vocab whitelist and an in-memory dictionary of one
    non-matching token so the name-dictionary load doesn't dominate."""
    t0 = time.perf_counter()
    audit = audit_subject(
        benchmark_subject_dir,
        output_dir=tmp_path,
        name_dictionary={"placeholder_nonmatch"},   # skip real dict load
        vocab_whitelist=set(),
    )
    total_s = time.perf_counter() - t0

    per_check = audit.get("_timings_by_check_s", {})
    print(f"\n=== Audit timing ({audit['n_files']} files) ===")
    print(f"Total: {total_s:.3f} s")
    if per_check:
        print("Per-check (slowest first):")
        for name, dt in sorted(per_check.items(), key=lambda kv: -kv[1]):
            print(f"  {dt:7.3f} s  {name}")
    else:
        print("(per-check timings not recorded — audit_subject did not populate "
              "_timings_by_check_s)")
    assert audit["overall_status"] in ("pass", "warn"), (
        f"benchmark fixture unexpectedly failed audit: {audit['overall_status']}"
    )
