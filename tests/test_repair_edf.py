import os
import shutil

import pyedflib
import pytest

from clean_eeg.repair_edf import (
    _read_header_fields,
    is_edf_truncated,
    repair_truncated_edf_header,
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
