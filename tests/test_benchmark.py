import time

from clean_eeg.benchmark import BenchmarkCollector


def test_disabled_collector_is_noop():
    """enabled=False must yield without recording or printing bench rows."""
    bench = BenchmarkCollector(enabled=False)
    with bench.step("nothing_to_see", file="f.edf"):
        pass
    assert bench.steps == []
    assert bench.report() == ""


def test_enabled_collector_records_one_row(capsys):
    """A single timed step should land one row; stdout should include a
    [bench] line with the step name."""
    bench = BenchmarkCollector(enabled=True)
    with bench.step("work", file="a.edf"):
        time.sleep(0.01)
    assert len(bench.steps) == 1
    row = bench.steps[0]
    assert row.step == "work"
    assert row.file == "a.edf"
    assert row.elapsed_s >= 0.0  # perf_counter is monotonic; small durations OK
    out = capsys.readouterr().out
    assert "[bench]" in out
    assert "work" in out


def test_report_contains_totals_per_file(capsys):
    """The summary should include one 'Totals per file' line per distinct file."""
    bench = BenchmarkCollector(enabled=True)
    with bench.step("step1", file="a.edf"):
        pass
    with bench.step("step2", file="a.edf"):
        pass
    with bench.step("step1", file="b.edf"):
        pass
    # Discard the per-step [bench] prints
    capsys.readouterr()

    report = bench.report()
    assert "Benchmark report" in report
    assert "Totals per file" in report
    assert "a.edf" in report
    assert "b.edf" in report


def test_step_records_even_on_exception():
    """A step that raises inside the `with` block should still record a row."""
    bench = BenchmarkCollector(enabled=True)
    with pytest_raises_value_error():
        with bench.step("boom", file="x.edf"):
            raise ValueError("boom")
    assert len(bench.steps) == 1
    assert bench.steps[0].step == "boom"


# Tiny helper to avoid pulling in pytest's importlib machinery
# at the top of the module (keeps the failure mode clear).
def pytest_raises_value_error():
    import pytest
    return pytest.raises(ValueError)
