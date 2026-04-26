import os
import shutil

import pyedflib
import pytest

from clean_eeg.repair_edf import (
    _read_header_fields,
    is_edf_truncated,
    repair_truncated_edf_header,
    repair_main_header_numeric_fields,
    repair_degenerate_signal_ranges,
    MAIN_HEADER_BYTES,
    SIGNAL_HEADER_BYTES_PER_SIGNAL,
    SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH,
    SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH,
    SIG_PHYS_DIM_OFFSET, SIG_PHYS_DIM_WIDTH,
    SIG_DIG_MIN_OFFSET, SIG_DIG_MIN_WIDTH,
    SIG_DIG_MAX_OFFSET, SIG_DIG_MAX_WIDTH,
    BYTES_IN_HEADER_OFFSET,
    BYTES_IN_HEADER_WIDTH,
    N_RECORDS_OFFSET,
    N_RECORDS_WIDTH,
    RECORD_DURATION_OFFSET,
    RECORD_DURATION_WIDTH,
    N_SIGNALS_OFFSET,
    N_SIGNALS_WIDTH,
    SAMPLES_PER_RECORD_FIELD_OFFSET,
    SAMPLES_PER_RECORD_FIELD_WIDTH,
    DEFAULT_RECORD_DURATION_S,
)
from tests.generate_edf import generate_partial_record_edf


def _zero_bytes(edf_path: str, offset: int, width: int) -> None:
    """Overwrite ``width`` bytes at ``offset`` with ASCII spaces (the EDF
    convention for blank field-slots, as observed in problem NK files)."""
    with open(edf_path, "r+b") as f:
        f.seek(offset)
        f.write(b" " * width)


def test_detects_truncation(tmp_path):
    """A file with a missing half-record should be flagged as truncated."""
    _, partial_path = generate_partial_record_edf(tmp_path / "truncated.edf",
                                                  n_channels=2,
                                                  sample_rate=100,
                                                  duration_sec=10)
    assert is_edf_truncated(partial_path)


def test_reports_intact_file_as_not_truncated(tmp_path):
    """A well-formed EDF should not be flagged as truncated."""
    full_path, _ = generate_partial_record_edf(tmp_path / "intact.edf",
                                               n_channels=2,
                                               sample_rate=100,
                                               duration_sec=10)
    assert not is_edf_truncated(full_path)


def test_repair_allows_pyedflib_to_open(tmp_path):
    """After repair, pyedflib should open the file without OSError."""
    _, partial_path = generate_partial_record_edf(tmp_path / "broken.edf",
                                                  n_channels=2,
                                                  sample_rate=100,
                                                  duration_sec=10)
    # Sanity: pyedflib rejects before repair
    with pytest.raises(OSError, match="(?i)filesize"):
        pyedflib.EdfReader(partial_path).close()

    new_n_records = repair_truncated_edf_header(partial_path, verbosity=0)
    assert new_n_records >= 1

    reader = pyedflib.EdfReader(partial_path)
    try:
        assert reader.datarecords_in_file == new_n_records
        # reading signal data should also succeed
        sig = reader.readSignal(0)
        assert len(sig) == new_n_records * reader.getSampleFrequency(0) * reader.datarecord_duration
    finally:
        reader.close()


def test_repair_is_noop_on_intact_file(tmp_path):
    """Repair on a non-truncated file should not modify it."""
    full_path, _ = generate_partial_record_edf(tmp_path / "intact2.edf",
                                               n_channels=2,
                                               sample_rate=100,
                                               duration_sec=10)
    size_before = os.path.getsize(full_path)
    with open(full_path, "rb") as f:
        bytes_before = f.read()

    original_n_records = _read_header_fields(full_path)["n_records"]
    returned = repair_truncated_edf_header(full_path, verbosity=0)
    assert returned == original_n_records

    with open(full_path, "rb") as f:
        bytes_after = f.read()
    assert os.path.getsize(full_path) == size_before
    assert bytes_before == bytes_after


def test_repair_writes_correct_n_records(tmp_path):
    """After repair, header's n_records must equal (data_bytes // record_bytes)."""
    _, partial_path = generate_partial_record_edf(tmp_path / "broken2.edf",
                                                  n_channels=2,
                                                  sample_rate=100,
                                                  duration_sec=10)
    fields_before = _read_header_fields(partial_path)
    data_bytes = os.path.getsize(partial_path) - fields_before["header_bytes"]
    expected_new = data_bytes // fields_before["record_bytes"]

    repair_truncated_edf_header(partial_path, verbosity=0)

    fields_after = _read_header_fields(partial_path)
    assert fields_after["n_records"] == expected_new
    # Signal headers untouched; only bytes 236-244 should differ.
    with open(partial_path, "rb") as f:
        main_after = f.read(256)
    assert int(main_after[N_RECORDS_OFFSET:N_RECORDS_OFFSET + N_RECORDS_WIDTH]
               .decode().strip()) == expected_new


def test_repair_refuses_file_with_no_complete_records(tmp_path):
    """If not even one complete data record is present, repair should raise."""
    full_path, _ = generate_partial_record_edf(tmp_path / "totaled.edf",
                                               n_channels=2,
                                               sample_rate=100,
                                               duration_sec=10)
    # Wipe almost all data, leaving strictly less than one complete record.
    totaled_path = str(tmp_path / "totaled_fatal.edf")
    shutil.copy2(full_path, totaled_path)
    fields = _read_header_fields(totaled_path)
    # Truncate to header + half of one record
    half_record = fields["record_bytes"] // 2
    with open(totaled_path, "r+b") as f:
        f.truncate(fields["header_bytes"] + half_record)

    with pytest.raises(ValueError, match="not even one complete data record"):
        repair_truncated_edf_header(totaled_path, verbosity=0)


def test_repair_prints_one_liner_report(tmp_path, capsys):
    """verbosity >= 1 should print a short report of the repair."""
    _, partial_path = generate_partial_record_edf(tmp_path / "chatty.edf",
                                                  n_channels=2,
                                                  sample_rate=100,
                                                  duration_sec=10)
    repair_truncated_edf_header(partial_path, verbosity=1)
    out = capsys.readouterr().out
    assert "Repairing truncated EDF header" in out


# ------------------------------------------------------------
# repair_degenerate_signal_ranges tests
# ------------------------------------------------------------

def _write_sig_field(path: str, signal_idx: int,
                     field_offset: int, field_width: int,
                     value: str) -> None:
    with open(path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
    n_signals = int(main[252:256].decode().strip())
    abs_offset = (MAIN_HEADER_BYTES
                  + field_offset * n_signals
                  + signal_idx * field_width)
    with open(path, "r+b") as f:
        f.seek(abs_offset)
        f.write(value.ljust(field_width).encode("ascii"))


def _corrupt_phys_fields(path: str, signal_idx: int,
                         new_min: str, new_max: str,
                         new_dim: str = None) -> None:
    """Directly write ASCII bytes to a signal's physical-range header fields."""
    _write_sig_field(path, signal_idx,
                     SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH, new_min)
    _write_sig_field(path, signal_idx,
                     SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH, new_max)
    if new_dim is not None:
        _write_sig_field(path, signal_idx,
                         SIG_PHYS_DIM_OFFSET, SIG_PHYS_DIM_WIDTH, new_dim)


def _corrupt_dig_fields(path: str, signal_idx: int,
                        new_min: str, new_max: str) -> None:
    """Directly write ASCII bytes to a signal's digital-range header fields."""
    _write_sig_field(path, signal_idx,
                     SIG_DIG_MIN_OFFSET, SIG_DIG_MIN_WIDTH, new_min)
    _write_sig_field(path, signal_idx,
                     SIG_DIG_MAX_OFFSET, SIG_DIG_MAX_WIDTH, new_max)


def _read_signal_phys_bytes(path: str, signal_idx: int) -> dict:
    """Read the 8-byte phys_min, phys_max, and phys_dim fields for a signal."""
    with open(path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
    n_signals = int(main[252:256].decode().strip())
    out = {}
    with open(path, "rb") as f:
        for field_name, field_offset, field_width in (
            ("phys_min", SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH),
            ("phys_max", SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH),
            ("phys_dim", SIG_PHYS_DIM_OFFSET, SIG_PHYS_DIM_WIDTH),
        ):
            f.seek(MAIN_HEADER_BYTES + field_offset * n_signals
                   + signal_idx * field_width)
            out[field_name] = f.read(field_width)
    return out


def _read_signal_dig_bytes(path: str, signal_idx: int) -> dict:
    """Read the 8-byte dig_min and dig_max fields for a signal."""
    with open(path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
    n_signals = int(main[252:256].decode().strip())
    out = {}
    with open(path, "rb") as f:
        for field_name, field_offset, field_width in (
            ("dig_min", SIG_DIG_MIN_OFFSET, SIG_DIG_MIN_WIDTH),
            ("dig_max", SIG_DIG_MAX_OFFSET, SIG_DIG_MAX_WIDTH),
        ):
            f.seek(MAIN_HEADER_BYTES + field_offset * n_signals
                   + signal_idx * field_width)
            out[field_name] = f.read(field_width)
    return out


def test_detects_and_repairs_exact_degenerate_range(tmp_path):
    """phys_max == phys_min should be detected and repaired to -1/1 with
    blank dimension."""
    full_path, _ = generate_partial_record_edf(tmp_path / "degen.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    # Corrupt signal 1's phys range so phys_max == phys_min.
    _corrupt_phys_fields(full_path, signal_idx=1,
                         new_min="0.00000", new_max="0.00000",
                         new_dim="uV")

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    r = repairs[0]
    assert r["signal_idx"] == 1
    assert r["original_phys_min"] == "0.00000"
    assert r["original_phys_max"] == "0.00000"
    assert "phys_min == phys_max" in r["phys_issue"]
    assert r["dig_issue"] is None

    post = _read_signal_phys_bytes(full_path, 1)
    assert post["phys_min"].decode().rstrip() == "-1"
    assert post["phys_max"].decode().rstrip() == "1"
    assert post["phys_dim"].decode() == " " * 8  # 8-space spec canonical

    # Dig range is also normalized to full int16 because any repair
    # uncalibrates the signal on both sides.
    post_dig = _read_signal_dig_bytes(full_path, 1)
    assert post_dig["dig_min"].decode().rstrip() == "-32768"
    assert post_dig["dig_max"].decode().rstrip() == "32767"


def test_detects_negative_zero_degenerate_range(tmp_path):
    """The NK "-0.00000" / "0.00000" pair (float equal, string-different) must
    also be caught, because pyedflib does the numeric comparison in C."""
    full_path, _ = generate_partial_record_edf(tmp_path / "neg_zero.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_phys_fields(full_path, signal_idx=2,
                         new_min="0.00000", new_max="-0.00000",
                         new_dim="V")

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    assert repairs[0]["signal_idx"] == 2
    assert repairs[0]["original_phys_max"] == "-0.00000"

    # File must now open in pyedflib (the "Physical Maximum" check passes).
    with pyedflib.EdfReader(full_path) as r:
        assert r.signals_in_file >= 1


def test_preserves_valid_signals_and_file_opens(tmp_path):
    """Well-formed signals must not be touched; file must still be pyedflib-openable."""
    full_path, _ = generate_partial_record_edf(tmp_path / "valid.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    before = [_read_signal_phys_bytes(full_path, i) for i in range(3)]

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)
    assert repairs == []

    after = [_read_signal_phys_bytes(full_path, i) for i in range(3)]
    assert before == after

    with pyedflib.EdfReader(full_path) as r:
        assert r.signals_in_file == 3


def test_unparseable_phys_field_is_repaired(tmp_path):
    """A phys field that isn't a valid number should also trigger repair."""
    full_path, _ = generate_partial_record_edf(tmp_path / "garbled.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_phys_fields(full_path, signal_idx=0,
                         new_min="notanum", new_max="0.5",
                         new_dim="uV")

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    assert repairs[0]["signal_idx"] == 0
    assert "unparseable" in repairs[0]["phys_issue"]

    post = _read_signal_phys_bytes(full_path, 0)
    assert post["phys_min"].decode().rstrip() == "-1"
    assert post["phys_max"].decode().rstrip() == "1"
    assert post["phys_dim"].decode() == " " * 8


def test_repair_is_idempotent(tmp_path):
    """Running repair twice on the same file should find nothing new on the
    second pass."""
    full_path, _ = generate_partial_record_edf(tmp_path / "idemp.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_phys_fields(full_path, signal_idx=1,
                         new_min="5", new_max="5")

    first = repair_degenerate_signal_ranges(full_path, verbosity=0)
    second = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(first) == 1
    assert second == []


def test_repair_warning_includes_original_values(tmp_path, capsys):
    """verbosity >= 1 must print the signal label + original values so the
    operator can see exactly what was rewritten."""
    full_path, _ = generate_partial_record_edf(tmp_path / "chatty_phys.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_phys_fields(full_path, signal_idx=0,
                         new_min="0.00000", new_max="-0.00000",
                         new_dim="raw")
    repair_degenerate_signal_ranges(full_path, verbosity=1)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "invalid EDF+" in out
    assert "0.00000" in out
    assert "-0.00000" in out
    assert "raw" in out


# ------------------------------------------------------------
# Digital-range: detect and repair (uncalibrates the whole signal, so
# BOTH phys and dig get rewritten together — see the repair policy
# docstring for the "all-or-nothing per signal" rationale).
# ------------------------------------------------------------

def test_dig_degeneracy_triggers_full_uncalibration(tmp_path):
    """A signal with dig_max == dig_min (and otherwise-valid phys) must
    get BOTH sides rewritten: phys to the uncalibrated convention (-1 / 1 /
    8-space units) AND dig to the full int16 range (-32768 / 32767). This
    keeps the header's scaling mapping internally consistent."""
    import pyedflib
    full_path, _ = generate_partial_record_edf(tmp_path / "dig_degen.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_dig_fields(full_path, signal_idx=0,
                        new_min="0", new_max="0")

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    r = repairs[0]
    assert r["signal_idx"] == 0
    assert r["phys_issue"] is None           # phys was fine, dig was not
    assert r["dig_issue"] is not None
    assert "dig_max" in r["dig_issue"]

    # Phys must have been uncalibrated too, even though phys was valid —
    # we refuse to leave a valid phys paired with a synthesised dig.
    phys_after = _read_signal_phys_bytes(full_path, 0)
    assert phys_after["phys_min"].decode().rstrip() == "-1"
    assert phys_after["phys_max"].decode().rstrip() == "1"
    assert phys_after["phys_dim"].decode() == " " * 8

    # Dig range rewritten to full int16.
    dig_after = _read_signal_dig_bytes(full_path, 0)
    assert dig_after["dig_min"].decode().rstrip() == "-32768"
    assert dig_after["dig_max"].decode().rstrip() == "32767"

    # File must now open cleanly in pyedflib (both checks pass).
    with pyedflib.EdfReader(full_path) as r_:
        assert r_.signals_in_file == 3


def test_dig_strict_inequality_triggers_repair(tmp_path):
    """EDF+ requires strict dig_max > dig_min. dig_max < dig_min (which is
    < dig_min + 1) must fire the same repair."""
    full_path, _ = generate_partial_record_edf(tmp_path / "dig_strict.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_dig_fields(full_path, signal_idx=1,
                        new_min="100", new_max="50")  # dig_max < dig_min
    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)
    assert len(repairs) == 1
    assert repairs[0]["signal_idx"] == 1
    assert repairs[0]["dig_issue"] is not None

    dig_after = _read_signal_dig_bytes(full_path, 1)
    assert dig_after["dig_min"].decode().rstrip() == "-32768"
    assert dig_after["dig_max"].decode().rstrip() == "32767"


def test_unparseable_digital_field_triggers_repair(tmp_path):
    """Garbage bytes in the dig fields must also trigger the full
    uncalibration repair."""
    full_path, _ = generate_partial_record_edf(tmp_path / "dig_garble.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_dig_fields(full_path, signal_idx=2,
                        new_min="abc", new_max="32767")

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    assert "unparseable" in repairs[0]["dig_issue"]

    dig_after = _read_signal_dig_bytes(full_path, 2)
    assert dig_after["dig_min"].decode().rstrip() == "-32768"
    assert dig_after["dig_max"].decode().rstrip() == "32767"


def test_combined_phys_and_dig_issues_both_rewritten(tmp_path):
    """A signal with BOTH phys and dig issues: both sides get rewritten
    to the uncalibrated defaults in a single pass."""
    full_path, _ = generate_partial_record_edf(tmp_path / "combined.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_phys_fields(full_path, signal_idx=1,
                         new_min="0.00000", new_max="-0.00000")
    _corrupt_dig_fields(full_path, signal_idx=1,
                        new_min="0", new_max="0")

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    r = repairs[0]
    assert r["phys_issue"] is not None
    assert r["dig_issue"] is not None

    phys_after = _read_signal_phys_bytes(full_path, 1)
    assert phys_after["phys_min"].decode().rstrip() == "-1"
    assert phys_after["phys_max"].decode().rstrip() == "1"
    assert phys_after["phys_dim"].decode() == " " * 8

    dig_after = _read_signal_dig_bytes(full_path, 1)
    assert dig_after["dig_min"].decode().rstrip() == "-32768"
    assert dig_after["dig_max"].decode().rstrip() == "32767"


def test_phys_only_issue_also_rewrites_dig(tmp_path):
    """Even when only phys is broken, dig gets rewritten too — the
    all-or-nothing-per-signal rule. Keeps header scaling self-consistent."""
    full_path, _ = generate_partial_record_edf(tmp_path / "phys_only.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_phys_fields(full_path, signal_idx=2,
                         new_min="0.00000", new_max="-0.00000")

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)
    assert len(repairs) == 1
    assert repairs[0]["phys_issue"] is not None
    assert repairs[0]["dig_issue"] is None

    dig_after = _read_signal_dig_bytes(full_path, 2)
    assert dig_after["dig_min"].decode().rstrip() == "-32768"
    assert dig_after["dig_max"].decode().rstrip() == "32767"


def test_dig_warning_message_reports_original_values(tmp_path, capsys):
    """Warning must include the signal label and the original dig values
    so operators can see which channel triggered the uncalibration."""
    full_path, _ = generate_partial_record_edf(tmp_path / "dig_msg.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_dig_fields(full_path, signal_idx=0,
                        new_min="42", new_max="42")
    repair_degenerate_signal_ranges(full_path, verbosity=1)
    out = capsys.readouterr().out
    assert "digital range issue" in out
    assert "uncalibrated" in out
    assert "dig_min='42'" in out
    assert "dig_max='42'" in out


# ---------------------------------------------------------------------------
# Empty numeric header fields (Nihon Kohden anomaly).
#
# These tests synthesize the failure mode reported in the field
# (``ValueError: invalid literal for int() with base 10: ''``) by zeroing
# out specific byte ranges of a freshly-generated EDF and then exercising
# the repair pass. We DO NOT have the offending source file (it contains
# PHI), so reproducing via byte-edits is the only avenue.
# ---------------------------------------------------------------------------

def _make_clean_edf(tmp_path, n_channels=3, sample_rate=100, duration_sec=5):
    """Helper: generate a small intact EDF and return its path."""
    full_path, _ = generate_partial_record_edf(tmp_path / "src.edf",
                                                n_channels=n_channels,
                                                sample_rate=sample_rate,
                                                duration_sec=duration_sec)
    return full_path


def _on_disk_n_signals(edf_path: str) -> int:
    """Read the actual on-disk n_signals (pyedflib's writer auto-adds an
    'EDF Annotations' channel for FILETYPE_EDFPLUS, so this is one more
    than the n_channels passed to the generator)."""
    with open(edf_path, "rb") as f:
        f.seek(N_SIGNALS_OFFSET)
        return int(f.read(N_SIGNALS_WIDTH).decode().strip())


def test_empty_n_records_repairs_from_filesize(tmp_path, capsys):
    """Empty bytes 236-244 (n_records) must not crash. The repair pass
    should compute n_records from the actual file size and write it back."""
    edf = _make_clean_edf(tmp_path, n_channels=3, sample_rate=100, duration_sec=5)
    expected = _read_header_fields(edf)["n_records"]
    assert expected is not None and expected > 0

    _zero_bytes(edf, N_RECORDS_OFFSET, N_RECORDS_WIDTH)

    new_n = repair_truncated_edf_header(edf, verbosity=1)
    assert new_n == expected
    out = capsys.readouterr().out
    assert "empty" in out.lower()
    # File now opens normally.
    fields = _read_header_fields(edf)
    assert fields["n_records"] == expected


def test_empty_n_records_sentinel_is_resolved(tmp_path, capsys):
    """EDF+ spec sentinel n_records=-1 ('recording in progress') must
    also be resolved to the actual count from filesize."""
    edf = _make_clean_edf(tmp_path)
    expected = _read_header_fields(edf)["n_records"]
    # Write '-1' padded to 8 bytes.
    with open(edf, "r+b") as f:
        f.seek(N_RECORDS_OFFSET)
        f.write(b"-1      ")
    new_n = repair_truncated_edf_header(edf, verbosity=1)
    assert new_n == expected
    out = capsys.readouterr().out
    assert "sentinel" in out.lower() or "recording in progress" in out.lower()


def test_empty_n_signals_raises_clear_error(tmp_path):
    """Empty bytes 252-256 (n_signals) cannot be safely defaulted —
    every per-signal field-slice depends on knowing N. Must raise with
    field name and value in the message."""
    edf = _make_clean_edf(tmp_path)
    _zero_bytes(edf, N_SIGNALS_OFFSET, N_SIGNALS_WIDTH)
    with pytest.raises(ValueError) as excinfo:
        repair_truncated_edf_header(edf)
    msg = str(excinfo.value)
    assert "'n_signals'" in msg
    assert "''" in msg, f"error must show the empty value: {msg!r}"
    assert "Cannot proceed" in msg or "unrecoverable" in msg.lower() \
        or "no safe" in msg.lower()


def test_unparseable_n_signals_error_shows_field_and_value(tmp_path):
    """A non-empty but garbage n_signals value must produce a message
    that names the field AND shows the offending value."""
    edf = _make_clean_edf(tmp_path)
    with open(edf, "r+b") as f:
        f.seek(N_SIGNALS_OFFSET)
        f.write(b"abcd")
    with pytest.raises(ValueError) as excinfo:
        repair_truncated_edf_header(edf)
    msg = str(excinfo.value)
    assert "'n_signals'" in msg
    assert "'abcd'" in msg


def test_empty_samples_per_record_one_signal_raises(tmp_path):
    """Empty samples_per_record slot for ANY signal is unrecoverable —
    record geometry is undefined. Error must name the field, the signal
    index, and show the empty value."""
    edf = _make_clean_edf(tmp_path, n_channels=3)
    # pyedflib auto-adds an 'EDF Annotations' channel for FILETYPE_EDFPLUS,
    # so on-disk n_signals is 4 (not 3). Field-block offsets depend on N.
    n_signals_disk = _on_disk_n_signals(edf)
    block_start = (MAIN_HEADER_BYTES
                   + SAMPLES_PER_RECORD_FIELD_OFFSET * n_signals_disk)
    _zero_bytes(edf, block_start + 0 * SAMPLES_PER_RECORD_FIELD_WIDTH,
                SAMPLES_PER_RECORD_FIELD_WIDTH)
    with pytest.raises(ValueError) as excinfo:
        repair_truncated_edf_header(edf)
    msg = str(excinfo.value)
    assert "'samples_per_record'" in msg
    assert "signal=0" in msg
    assert "''" in msg
    assert "byte layout" in msg.lower() or "stride" in msg.lower() \
        or "no safe" in msg.lower()


def test_empty_samples_per_record_middle_signal_double_space_pattern(tmp_path):
    """When only the middle signal's slot is blank, the on-disk pattern
    is the populated value, then 8 spaces, then the next populated
    value — the 'double space' case the field operator described.
    Repair must still raise with the correct signal index."""
    edf = _make_clean_edf(tmp_path, n_channels=3)
    n_signals_disk = _on_disk_n_signals(edf)
    block_start = (MAIN_HEADER_BYTES
                   + SAMPLES_PER_RECORD_FIELD_OFFSET * n_signals_disk)
    # Blank signal 1 (a middle data signal), leaving signals 0/2/3 intact.
    _zero_bytes(edf, block_start + 1 * SAMPLES_PER_RECORD_FIELD_WIDTH,
                SAMPLES_PER_RECORD_FIELD_WIDTH)
    with pytest.raises(ValueError) as excinfo:
        repair_truncated_edf_header(edf)
    msg = str(excinfo.value)
    assert "'samples_per_record'" in msg
    assert "signal=1" in msg


def test_empty_samples_per_record_all_signals_raises(tmp_path):
    """Sanity: blanking every slot still raises (on the first signal)
    rather than silently accepting an all-empty array."""
    edf = _make_clean_edf(tmp_path, n_channels=3)
    n_signals_disk = _on_disk_n_signals(edf)
    block_start = (MAIN_HEADER_BYTES
                   + SAMPLES_PER_RECORD_FIELD_OFFSET * n_signals_disk)
    _zero_bytes(edf, block_start,
                SAMPLES_PER_RECORD_FIELD_WIDTH * n_signals_disk)
    with pytest.raises(ValueError) as excinfo:
        repair_truncated_edf_header(edf)
    msg = str(excinfo.value)
    assert "'samples_per_record'" in msg
    # First failing signal is reported.
    assert "signal=0" in msg


def test_empty_bytes_in_header_repaired_to_derived_value(tmp_path, capsys):
    """Empty bytes 184-192 (bytes_in_header) are repaired in place by
    writing 256 * (1 + n_signals). Verify both the side-effect on disk
    and the warning text."""
    edf = _make_clean_edf(tmp_path, n_channels=3)
    n_signals_disk = _on_disk_n_signals(edf)
    expected = MAIN_HEADER_BYTES * (1 + n_signals_disk)
    _zero_bytes(edf, BYTES_IN_HEADER_OFFSET, BYTES_IN_HEADER_WIDTH)

    repaired = repair_main_header_numeric_fields(edf, verbosity=1)
    assert repaired.get("bytes_in_header") == expected

    # Field is now populated on disk.
    with open(edf, "rb") as f:
        f.seek(BYTES_IN_HEADER_OFFSET)
        assert int(f.read(BYTES_IN_HEADER_WIDTH).decode().strip()) == expected
    out = capsys.readouterr().out
    assert "bytes_in_header" in out
    assert str(expected) in out


def test_empty_record_duration_repaired_with_implied_rate_warning(tmp_path, capsys):
    """Empty record_duration is repaired to the 1.0s default with a
    WARNING that includes the implied sample rate (samples_per_record
    signal 0 / record_duration). The 'implied rate' line is the user
    visibility check that flags wrong defaults."""
    sample_rate = 100   # samples_per_record will equal this for spr=rate*1s
    edf = _make_clean_edf(tmp_path, n_channels=3, sample_rate=sample_rate)
    _zero_bytes(edf, RECORD_DURATION_OFFSET, RECORD_DURATION_WIDTH)

    repaired = repair_main_header_numeric_fields(edf, verbosity=1)
    assert repaired.get("record_duration") == DEFAULT_RECORD_DURATION_S

    # Field on disk now reads as 1.
    with open(edf, "rb") as f:
        f.seek(RECORD_DURATION_OFFSET)
        assert float(f.read(RECORD_DURATION_WIDTH).decode().strip()) == 1.0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "record_duration" in out
    # Implied-rate line: samples_per_record signal 0 (= sample_rate * 1s
    # = sample_rate) divided by 1.0 = sample_rate Hz.
    assert f"{sample_rate} Hz" in out


def test_empty_record_duration_implied_rate_unknown_when_spr_also_empty(tmp_path):
    """If samples_per_record signal 0 is also empty, record_duration
    repair never runs — _read_header_fields raises on the empty SPR
    first. Locks in the dependency order."""
    edf = _make_clean_edf(tmp_path, n_channels=3)
    n_signals_disk = _on_disk_n_signals(edf)
    # Blank record_duration AND samples_per_record signal 0.
    _zero_bytes(edf, RECORD_DURATION_OFFSET, RECORD_DURATION_WIDTH)
    block_start = (MAIN_HEADER_BYTES
                   + SAMPLES_PER_RECORD_FIELD_OFFSET * n_signals_disk)
    _zero_bytes(edf, block_start, SAMPLES_PER_RECORD_FIELD_WIDTH)
    with pytest.raises(ValueError) as excinfo:
        repair_main_header_numeric_fields(edf)
    msg = str(excinfo.value)
    assert "'samples_per_record'" in msg
    assert "signal=0" in msg


def test_repair_main_header_is_idempotent_on_clean_file(tmp_path, capsys):
    """Calling the merged repair on a healthy file should be a no-op:
    no fields reported, no warnings printed."""
    edf = _make_clean_edf(tmp_path, n_channels=3)
    repaired = repair_main_header_numeric_fields(edf, verbosity=1)
    assert repaired == {}
    out = capsys.readouterr().out
    assert "Repairing" not in out
    assert "WARNING" not in out


def test_legacy_repair_truncated_alias_still_works(tmp_path):
    """The old function name is preserved as a thin alias. Smoke test
    that the existing import path still functions and returns the
    n_records value (matching the legacy contract)."""
    edf = _make_clean_edf(tmp_path, n_channels=3)
    expected = _read_header_fields(edf)["n_records"]
    _zero_bytes(edf, N_RECORDS_OFFSET, N_RECORDS_WIDTH)
    n = repair_truncated_edf_header(edf, verbosity=0)
    assert n == expected


def test_empty_phys_min_already_handled_by_existing_repair(tmp_path, capsys):
    """Empty phys_min (8 spaces) was already handled implicitly because
    ``float('')`` raises ValueError. Locking in the contract: blank
    phys/dig fields trigger the existing 'mark uncalibrated' repair."""
    edf = _make_clean_edf(tmp_path, n_channels=3)
    n_signals_disk = _on_disk_n_signals(edf)
    # Blank phys_min for signal 0.
    pmin_offset = (MAIN_HEADER_BYTES
                   + SIG_PHYS_MIN_OFFSET * n_signals_disk
                   + 0 * SIG_PHYS_MIN_WIDTH)
    _zero_bytes(edf, pmin_offset, SIG_PHYS_MIN_WIDTH)
    repairs = repair_degenerate_signal_ranges(edf, verbosity=1)
    assert any(r["signal_idx"] == 0 for r in repairs)
    out = capsys.readouterr().out
    assert "uncalibrated" in out
