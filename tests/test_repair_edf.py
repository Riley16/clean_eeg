import os
import shutil

import pyedflib
import pytest

from clean_eeg.repair_edf import (
    _read_header_fields,
    is_edf_truncated,
    repair_truncated_edf_header,
    repair_degenerate_signal_ranges,
    MAIN_HEADER_BYTES,
    SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH,
    SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH,
    SIG_PHYS_DIM_OFFSET, SIG_PHYS_DIM_WIDTH,
    SIG_DIG_MIN_OFFSET, SIG_DIG_MIN_WIDTH,
    SIG_DIG_MAX_OFFSET, SIG_DIG_MAX_WIDTH,
    N_RECORDS_OFFSET,
    N_RECORDS_WIDTH,
)
from tests.generate_edf import generate_partial_record_edf


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
    assert r["phys_repaired"] is True
    assert r["dig_repaired"] is False

    post = _read_signal_phys_bytes(full_path, 1)
    assert post["phys_min"].decode().rstrip() == "-1"
    assert post["phys_max"].decode().rstrip() == "1"
    assert post["phys_dim"].decode() == " " * 8  # 8-space spec canonical


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
# Digital-range: detect-and-warn only (no byte rewriting)
# ------------------------------------------------------------

def test_detects_digital_range_degeneracy_without_repair(tmp_path):
    """dig_max <= dig_min must be detected and warned about, but the dig
    bytes must NOT be rewritten — the caller should let pyedflib reject
    the file so it surfaces via the skipped-files path."""
    full_path, _ = generate_partial_record_edf(tmp_path / "dig_degen.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    # Corrupt signal 0's digital range so dig_max == dig_min
    _corrupt_dig_fields(full_path, signal_idx=0,
                        new_min="0", new_max="0")
    before = _read_signal_dig_bytes(full_path, 0)

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    r = repairs[0]
    assert r["signal_idx"] == 0
    assert r["phys_issue"] is None
    assert r["dig_issue"] is not None
    assert "dig_max" in r["dig_issue"]
    assert r["phys_repaired"] is False
    assert r["dig_repaired"] is False

    # Dig bytes must be unchanged on disk.
    after = _read_signal_dig_bytes(full_path, 0)
    assert before == after


def test_detects_digital_strict_inequality(tmp_path):
    """EDF+ requires strict dig_max > dig_min. dig_max < dig_min + 1 must
    fire (captures the dig_max == dig_min case and also dig_max < dig_min
    if somehow written that way)."""
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


def test_detects_unparseable_digital_field(tmp_path):
    """Garbage bytes in the dig fields must be flagged, not repaired."""
    full_path, _ = generate_partial_record_edf(tmp_path / "dig_garble.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_dig_fields(full_path, signal_idx=2,
                        new_min="abc", new_max="32767")
    before = _read_signal_dig_bytes(full_path, 2)

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    assert "unparseable" in repairs[0]["dig_issue"]
    assert repairs[0]["dig_repaired"] is False

    after = _read_signal_dig_bytes(full_path, 2)
    assert before == after


def test_combined_phys_and_dig_issues_phys_repaired_dig_flagged(tmp_path):
    """A signal with BOTH phys and dig problems: phys gets repaired,
    dig gets flagged-only (dig bytes unchanged)."""
    full_path, _ = generate_partial_record_edf(tmp_path / "combined.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_phys_fields(full_path, signal_idx=1,
                         new_min="0.00000", new_max="-0.00000")
    _corrupt_dig_fields(full_path, signal_idx=1,
                        new_min="0", new_max="0")
    dig_before = _read_signal_dig_bytes(full_path, 1)

    repairs = repair_degenerate_signal_ranges(full_path, verbosity=0)

    assert len(repairs) == 1
    r = repairs[0]
    assert r["phys_issue"] is not None
    assert r["dig_issue"] is not None
    assert r["phys_repaired"] is True
    assert r["dig_repaired"] is False

    # Phys bytes should now be the uncalibrated convention.
    phys_after = _read_signal_phys_bytes(full_path, 1)
    assert phys_after["phys_min"].decode().rstrip() == "-1"
    assert phys_after["phys_max"].decode().rstrip() == "1"
    assert phys_after["phys_dim"].decode() == " " * 8

    # Dig bytes must be unchanged.
    dig_after = _read_signal_dig_bytes(full_path, 1)
    assert dig_before == dig_after


def test_dig_warning_message_includes_not_repaired(tmp_path, capsys):
    """The dig warning must make it obvious to the operator that the dig
    bytes were NOT rewritten, so downstream pyedflib rejection is expected."""
    full_path, _ = generate_partial_record_edf(tmp_path / "dig_msg.edf",
                                               n_channels=3,
                                               sample_rate=100,
                                               duration_sec=5)
    _corrupt_dig_fields(full_path, signal_idx=0,
                        new_min="42", new_max="42")
    repair_degenerate_signal_ranges(full_path, verbosity=1)
    out = capsys.readouterr().out
    assert "digital range issue" in out
    assert "NOT REPAIRED" in out
    # Original values must appear so the operator knows which signal.
    assert "dig_min='42'" in out
    assert "dig_max='42'" in out
