"""End-to-end pipeline benchmark test.

Marked `benchmark` and excluded from default pytest runs via
`addopts = "-m 'not benchmark'"` in pyproject.toml. To execute explicitly:

    pytest -m benchmark
    pytest tests/test_benchmark_pipeline.py -m benchmark -s

The `-s` flag keeps pytest from swallowing stdout so the per-step
[bench] lines and final benchmark report are visible live.
"""

import os

import pytest

from clean_eeg.anonymize import PersonalName
from clean_eeg.clean_subject_eeg import clean_subject_edf_files
from tests.generate_edf import generate_large_benchmark_edf


@pytest.mark.benchmark
def test_pipeline_benchmark_large_edf(tmp_path, capsys):
    """Generate a ~30 MB synthetic EDF and run the full pipeline with
    benchmark=True. Asserts the benchmark report is emitted and contains
    the expected step names. The printed timing/memory rows are the real
    product of this test; the assertions just guard that instrumentation
    stays wired up."""
    subject_code = "R1BENCHS"
    subject_dir = tmp_path / subject_code / "raw"
    subject_dir.mkdir(parents=True)
    edf_path = subject_dir / "bench_large.edf"

    generate_large_benchmark_edf(
        str(edf_path),
        n_channels=178,
        sample_rate_hz=500,
        duration_s=180,
        patient_name="Test Patient",
        subject_code=subject_code,
    )
    size_mb = os.path.getsize(edf_path) / (1024 ** 2)
    assert size_mb >= 25, f"fixture smaller than expected: {size_mb:.1f} MB"

    output_dir = tmp_path / "deidentified"
    output_dir.mkdir()

    clean_subject_edf_files(
        input_path=str(subject_dir),
        output_path=str(output_dir),
        subject_code=subject_code,
        subject_name=PersonalName(first_name="Test", middle_names=[], last_name="Patient"),
        inplace=False,
        raise_errors=True,
        verbosity=1,
        benchmark=True,
    )

    out = capsys.readouterr().out
    assert "Benchmark report" in out
    for step in ("load_preload_signals", "deidentify_edf",
                 "write_edf_pyedflib", "Totals per file"):
        assert step in out, f"expected '{step}' in benchmark output"


@pytest.mark.benchmark_heavy
def test_pipeline_benchmark_heavy_edf(tmp_path, capsys):
    """Heavy benchmark: ~150 MB fixture with 100 annotations (copy mode).

    Modelled on real Nihon Kohden recordings that take minutes and many GB
    of RAM on operator laptops. File is sized by:
        178 ch * 500 Hz * 2 bytes * 900 s ~= 153 MB
    and seeded with 100 annotations, ~40% of which contain the subject's
    name or a gendered pronoun so annotation-redaction cost is exercised.

    Opt in with: pytest -m benchmark_heavy -s
    """
    subject_code = "R1BENCHH"
    subject_dir = tmp_path / subject_code / "raw"
    subject_dir.mkdir(parents=True)
    edf_path = subject_dir / "bench_heavy.edf"

    generate_large_benchmark_edf(
        str(edf_path),
        n_channels=178,
        sample_rate_hz=500,
        duration_s=900,
        patient_name="Test Patient",
        subject_code=subject_code,
        n_annotations=100,
    )
    size_mb = os.path.getsize(edf_path) / (1024 ** 2)
    assert size_mb >= 140, f"fixture smaller than expected: {size_mb:.1f} MB"

    output_dir = tmp_path / "deidentified"
    output_dir.mkdir()

    clean_subject_edf_files(
        input_path=str(subject_dir),
        output_path=str(output_dir),
        subject_code=subject_code,
        subject_name=PersonalName(first_name="Test", middle_names=[], last_name="Patient"),
        inplace=False,
        raise_errors=True,
        verbosity=1,
        benchmark=True,
    )

    out = capsys.readouterr().out
    assert "Benchmark report" in out
    assert "deidentify_edf" in out


@pytest.mark.benchmark_heavy
def test_pipeline_benchmark_heavy_edf_inplace_no_audit(tmp_path, capsys):
    """Heavy benchmark variant: inplace mode with skip_audit=True.

    This is the performance-critical configuration on multi-GB NK files.
    The audit is the only reason inplace-mode loads signals (the
    de-identification itself only touches header/signal_headers/
    annotations), so skipping it avoids pyedflib's per-channel
    interleaved read entirely — effectively the ``load_preload_signals``
    step should be gone from the benchmark output and peak RSS growth
    should be orders of magnitude smaller than the copy-mode variant.

    Opt in with: pytest -m benchmark_heavy -s
    """
    subject_code = "R1BENCHI"
    subject_dir = tmp_path / subject_code / "raw"
    subject_dir.mkdir(parents=True)
    edf_path = subject_dir / "bench_heavy_inplace.edf"

    generate_large_benchmark_edf(
        str(edf_path),
        n_channels=178,
        sample_rate_hz=500,
        duration_s=900,
        patient_name="Test Patient",
        subject_code=subject_code,
        n_annotations=100,
    )
    size_mb = os.path.getsize(edf_path) / (1024 ** 2)
    assert size_mb >= 140, f"fixture smaller than expected: {size_mb:.1f} MB"

    clean_subject_edf_files(
        input_path=str(subject_dir),
        output_path=str(subject_dir),   # inplace: input == output
        subject_code=subject_code,
        subject_name=PersonalName(first_name="Test", middle_names=[], last_name="Patient"),
        inplace=True,
        raise_errors=True,
        verbosity=1,
        benchmark=True,
        skip_audit=True,
    )

    out = capsys.readouterr().out
    assert "Benchmark report" in out
    # With skip_audit=True in inplace mode, no file should have preloaded
    # signals — every file should go through load_metadata_only instead.
    assert "load_metadata_only" in out
    assert "load_preload_signals" not in out
    # Audit step should not appear either.
    assert "audit_signal_integrity" not in out
