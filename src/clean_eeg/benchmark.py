"""Lightweight per-step timing + memory accounting for the de-identification
pipeline.

Usage
-----
    bench = BenchmarkCollector(enabled=True)
    with bench.step("load_preload_signals", file=fname):
        edf = load_edf(...)
    ...
    print(bench.report())

When ``enabled=False`` the ``step`` context manager is a no-op, so
instrumentation sites can stay in place with zero observable cost.

Memory accounting
-----------------
- ``rss_start_mb``, ``rss_end_mb`` — current RSS (Resident Set Size) sampled
  via ``psutil`` if available, otherwise via ``resource.getrusage`` as a
  coarse fallback.
- ``rss_delta_mb`` — net change across the step.
- ``peak_growth_mb`` — how far peak RSS moved during the step
  (``ru_maxrss`` after minus before). Useful for catching transient spikes
  that disappear before the step returns.

Linux reports ``ru_maxrss`` in kilobytes; macOS in bytes. The helper below
normalises both to bytes.
"""

import contextlib
import resource
import sys
import time
from dataclasses import dataclass


def _peak_rss_bytes() -> int:
    """Peak RSS since process start, in bytes (normalised across OSes)."""
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return val
    return val * 1024  # Linux/BSD report kilobytes


def _current_rss_bytes() -> int:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        # Fall back to peak as a coarse lower bound. Better than nothing
        # when psutil isn't available; the delta column will under-report.
        return _peak_rss_bytes()


@dataclass
class Step:
    file: str
    step: str
    elapsed_s: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    peak_growth_mb: float


class BenchmarkCollector:
    """Accumulates per-step timing + memory rows for a pipeline run."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.steps: list[Step] = []

    @contextlib.contextmanager
    def step(self, name: str, file: str = ""):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        rss0 = _current_rss_bytes()
        peak0 = _peak_rss_bytes()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            rss1 = _current_rss_bytes()
            peak1 = _peak_rss_bytes()
            row = Step(
                file=file,
                step=name,
                elapsed_s=dt,
                rss_start_mb=rss0 / 1024 / 1024,
                rss_end_mb=rss1 / 1024 / 1024,
                rss_delta_mb=(rss1 - rss0) / 1024 / 1024,
                peak_growth_mb=(peak1 - peak0) / 1024 / 1024,
            )
            self.steps.append(row)
            print(
                f"[bench] {file or '-':<30}  {name:<28}  "
                f"t={dt:6.2f}s  RSSΔ={row.rss_delta_mb:+7.1f} MB  "
                f"peak↑={row.peak_growth_mb:+7.1f} MB  "
                f"RSS now={row.rss_end_mb:7.0f} MB"
            )

    def report(self) -> str:
        """Return a multi-line summary: per-step table + per-file totals."""
        if not self.enabled or not self.steps:
            return ""
        lines = []
        lines.append("\n=== Benchmark report ===")
        header = (f"{'file':<40} {'step':<28} {'elapsed (s)':>11} "
                  f"{'RSS Δ (MB)':>12} {'peak ↑ (MB)':>12} {'RSS now (MB)':>13}")
        lines.append(header)
        lines.append("-" * len(header))
        for s in self.steps:
            fname = (s.file[:37] + "...") if len(s.file) > 40 else s.file
            lines.append(
                f"{fname:<40} {s.step:<28} {s.elapsed_s:>11.2f} "
                f"{s.rss_delta_mb:>12.1f} {s.peak_growth_mb:>12.1f} "
                f"{s.rss_end_mb:>13.0f}"
            )

        by_file: dict[str, list[Step]] = {}
        for s in self.steps:
            by_file.setdefault(s.file, []).append(s)
        lines.append("\n--- Totals per file ---")
        for fname, ss in by_file.items():
            total_t = sum(s.elapsed_s for s in ss)
            peak_rss_end = max(s.rss_end_mb for s in ss)
            total_peak_growth = sum(max(s.peak_growth_mb, 0.0) for s in ss)
            lines.append(
                f"  {fname}: {total_t:.1f}s total, "
                f"max RSS seen {peak_rss_end:.0f} MB, "
                f"cumulative peak growth {total_peak_growth:.0f} MB"
            )
        return "\n".join(lines)
