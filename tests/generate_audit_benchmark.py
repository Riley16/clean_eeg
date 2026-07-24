"""Generate a benchmark subject directory for audit performance testing.

Writes 5 large EDF+C files (~360 MB each; 100 channels @ 1 kHz for 30
minutes) into a subject directory. Signal data is random int16 reused
across records — the audit doesn't inspect signal statistics, so this
gives us realistic file sizes without paying for real signal generation.

The files are shaped to *pass* every audit check with default flags,
so the audit's cost is exercised without noise from failing checks:

  - patient_id has the post-cleaning sentinel layout
  - startdate is 01.01.85 (matches ``BASE_START_DATE``)
  - signal headers are identical across files
  - annotation channel carries only timekeeping TALs (no PHI)
  - files are perfectly adjacent (30-min chunks starting on the half hour)

Invoked from ``tests/test_audit_benchmark.py`` via a session-scoped
fixture — generation is lazy and only runs when the ``audit_benchmark``
marker is selected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


MAIN_HEADER_BYTES = 256
SIGNAL_HEADER_BYTES_PER_SIGNAL = 256

BENCHMARK_SUBJECT_CODE = "R1000A"
BENCHMARK_N_CHANNELS = 100
BENCHMARK_SAMPLE_RATE_HZ = 1000
BENCHMARK_DURATION_S = 30 * 60         # 30 minutes per file
BENCHMARK_N_FILES = 5


def _fixed(s: str | int, width: int) -> bytes:
    """Right-pad or truncate ``s`` to exactly ``width`` ASCII bytes."""
    b = str(s).encode("ascii")[:width]
    return b + b" " * (width - len(b))


def _build_main_header(*, n_total_signals: int,
                       patient_id: str, recording_id: str,
                       startdate: str, starttime: str,
                       n_records: int, record_duration_s: float) -> bytes:
    header_bytes = MAIN_HEADER_BYTES + n_total_signals * SIGNAL_HEADER_BYTES_PER_SIGNAL
    parts = [
        _fixed("0", 8),                          # version
        _fixed(patient_id, 80),                  # patient_id
        _fixed(recording_id, 80),                # recording_id
        _fixed(startdate, 8),                    # startdate DD.MM.YY
        _fixed(starttime, 8),                    # starttime HH.MM.SS
        _fixed(header_bytes, 8),                 # bytes_in_header
        _fixed("EDF+C", 44),                     # reserved (continuous)
        _fixed(n_records, 8),                    # n_records
        _fixed(f"{record_duration_s:g}", 8),     # record_duration
        _fixed(n_total_signals, 4),              # n_signals
    ]
    out = b"".join(parts)
    assert len(out) == MAIN_HEADER_BYTES
    return out


def _build_signal_headers(*, n_signals: int, samples_per_record: int,
                          ann_samples_per_record: int) -> bytes:
    """Signal headers for ``n_signals`` regular channels + 1 annotation
    channel. Every field-block spans all N signals, then the next field
    block, per EDF+ spec.
    """
    n_total = n_signals + 1
    labels = [f"CH{i:03d}" for i in range(n_signals)] + ["EDF Annotations"]

    def _field(values, width):
        return b"".join(_fixed(v, width) for v in values)

    parts = [
        _field(labels, 16),                              # label
        _field(["EEG"] * n_total, 80),                   # transducer
        _field(["uV"] * n_total, 8),                     # phys_dim
        _field(["-3200"] * n_total, 8),                  # phys_min
        _field(["3200"] * n_total, 8),                   # phys_max
        _field(["-32768"] * n_total, 8),                 # dig_min
        _field(["32767"] * n_total, 8),                  # dig_max
        _field([""] * n_total, 80),                      # prefilter
        _field([samples_per_record] * n_signals
               + [ann_samples_per_record], 8),           # samples_per_record
        _field([""] * n_total, 32),                      # reserved
    ]
    out = b"".join(parts)
    assert len(out) == SIGNAL_HEADER_BYTES_PER_SIGNAL * n_total
    return out


def _build_timekeeping_ann_block(onset_s: float, ann_bytes: int) -> bytes:
    """Timekeeping-only TAL, null-padded to the annotation channel's
    per-record byte budget.

    Format: ``+onset\\x14\\x14\\x00 <null padding>``. The trailing
    ``\\x14\\x14`` marks an empty text list (per EDF+ spec 2.2.4) so the
    PHI scanner sees no annotation text and cannot flag PHI.
    """
    tk = f"+{onset_s:.6f}".encode("ascii") + b"\x14\x14"
    block = tk + b"\x00"
    if len(block) > ann_bytes:
        raise ValueError(f"timekeeping TAL {len(block)} > record budget {ann_bytes}")
    return block + b"\x00" * (ann_bytes - len(block))


def write_benchmark_edf(out_path: str | Path, *,
                        n_signals: int = BENCHMARK_N_CHANNELS,
                        sample_rate_hz: int = BENCHMARK_SAMPLE_RATE_HZ,
                        duration_s: int = BENCHMARK_DURATION_S,
                        subject_code: str = BENCHMARK_SUBJECT_CODE,
                        startdate: str = "01.01.85",
                        starttime: str = "10.00.00",
                        rng_seed: int = 0) -> Path:
    """Write a valid EDF+C file directly to disk (bypassing pyedflib).

    Signal data is reused int16 noise — one record's worth is generated
    once and written ``n_records`` times. This produces a realistic
    on-disk size at negligible memory cost (~200 KB working set instead
    of the ~360 MB the file will occupy).
    """
    out_path = Path(out_path)
    record_duration_s = 1.0
    n_records = int(duration_s / record_duration_s)
    ann_samples_per_record = 32  # → 64 bytes; enough for any timekeeping TAL
    ann_bytes_per_record = ann_samples_per_record * 2
    record_reg_bytes = n_signals * sample_rate_hz * 2

    patient_id = f"{subject_code} X 01-JAN-1900 unknown unknown"
    recording_id = "Startdate 01-JAN-1985 X X X"
    main_header = _build_main_header(
        n_total_signals=n_signals + 1,
        patient_id=patient_id, recording_id=recording_id,
        startdate=startdate, starttime=starttime,
        n_records=n_records, record_duration_s=record_duration_s,
    )
    signal_headers = _build_signal_headers(
        n_signals=n_signals,
        samples_per_record=sample_rate_hz,
        ann_samples_per_record=ann_samples_per_record,
    )

    rng = np.random.default_rng(seed=rng_seed)
    reg_bytes = rng.integers(
        low=-1000, high=1000, size=n_signals * sample_rate_hz, dtype=np.int16
    ).tobytes()
    assert len(reg_bytes) == record_reg_bytes

    with open(out_path, "wb") as f:
        f.write(main_header)
        f.write(signal_headers)
        for r in range(n_records):
            f.write(reg_bytes)
            f.write(_build_timekeeping_ann_block(
                float(r * record_duration_s), ann_bytes_per_record))
    return out_path


def _starttime_str(hh: int, mm: int, ss: int) -> str:
    return f"{hh:02d}.{mm:02d}.{ss:02d}"


def build_benchmark_subject(subject_dir: str | Path, *,
                            n_files: int = BENCHMARK_N_FILES,
                            duration_s: int = BENCHMARK_DURATION_S,
                            subject_code: str = BENCHMARK_SUBJECT_CODE,
                            ) -> list[Path]:
    """Populate ``subject_dir`` with ``n_files`` back-to-back EDF+C files.

    File k starts ``k * duration_s`` seconds after the first, so
    ``check_recording_gaps`` sees zero gap. Skips files that already
    exist (idempotent — safe to call after a partial run).
    """
    subject_dir = Path(subject_dir)
    subject_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i in range(n_files):
        offset_s = i * duration_s
        hh = 10 + offset_s // 3600
        mm = (offset_s % 3600) // 60
        ss = offset_s % 60
        starttime = _starttime_str(hh, mm, ss)
        filename = f"{subject_code}_01.01__{starttime}.edf"
        path = subject_dir / filename
        if not path.exists():
            write_benchmark_edf(path, duration_s=duration_s,
                                subject_code=subject_code,
                                starttime=starttime, rng_seed=i)
        written.append(path)
    return written


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate the audit benchmark subject dataset."
    )
    parser.add_argument("subject_dir", type=Path,
                        help="Directory to write the 5 benchmark EDFs into.")
    parser.add_argument("--n-files", type=int, default=BENCHMARK_N_FILES)
    parser.add_argument("--duration-s", type=int, default=BENCHMARK_DURATION_S,
                        help="Recording length per file (default: 30 minutes).")
    args = parser.parse_args()
    paths = build_benchmark_subject(args.subject_dir,
                                    n_files=args.n_files,
                                    duration_s=args.duration_s)
    for p in paths:
        print(f"{p} ({p.stat().st_size / (1024 ** 2):.1f} MB)")
