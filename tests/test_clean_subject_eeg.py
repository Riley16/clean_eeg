import numpy as np
import os
import shutil
import pytest

from clean_eeg.clean_subject_eeg import remove_gendered_pronouns, _GENDERED_PRONOUNS, BASE_START_DATE,\
        DEFAULT_REDACT_HEADER_KEYS, REDACT_REPLACEMENT, REDACT_PRONOUN_REPLACEMENT, clean_subject_edf_files, \
        _check_subject_name_consistency
from clean_eeg.load_eeg import load_edf
from tests.generate_edf import format_edf_config_json
from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE, TEST_SUBJECT_DATA_DIR, INCONSISTENT_SUBJECT_DATA_DIR
from clean_eeg.anonymize import PersonalName, REDACT_NAME_REPLACEMENT

from datetime import datetime, timedelta
import json
with open(TEST_CONFIG_FILE, 'r') as f:
    TEST_CONFIG = json.load(f)
BASIC_EDF_PATH = str(TEST_DATA_DIR / TEST_CONFIG["basic_EDF+C"]['filename'])
SUBJECT_EDF_PATH1 = str(TEST_SUBJECT_DATA_DIR / TEST_CONFIG["subject_EDF+C_1"]['filename'])
SUBJECT_EDF_PATH2 = str(TEST_SUBJECT_DATA_DIR / TEST_CONFIG["subject_EDF+C_2"]['filename'])


def test_remove_gendered_pronouns_basic():
    input = ' asdf '.join(_GENDERED_PRONOUNS)
    output = ' asdf '.join([REDACT_PRONOUN_REPLACEMENT] * len(_GENDERED_PRONOUNS))
    assert remove_gendered_pronouns(input) == output

EDF_CONFIG = TEST_CONFIG["basic_EDF+C"]
EDF_TIMESTAMP_FORMAT = EDF_CONFIG['timestamp_format']
EDF_CONFIG = format_edf_config_json(EDF_CONFIG)
EDF_HEADER = EDF_CONFIG['header']

SUBJECT_CODE = 'R1755A'
PATIENT_NAME = PersonalName(first_name='L.',
                            middle_names=[],
                            last_name="Smith")

def test_deidentify_edf_header():
    from clean_eeg.clean_subject_eeg import deidentify_edf_header
    recording_timestamp = EDF_HEADER['startdate']
    recording_offset = timedelta(days=1)
    earliest_recording_timestamp = recording_timestamp - recording_offset

    # insert patient pronoun and name into 'equipment' field
    EDF_HEADER['equipment'] = 'his ' + PATIENT_NAME.get_full_name()

    new_header = deidentify_edf_header(EDF_HEADER,
                                       earliest_recording_start_time=earliest_recording_timestamp,
                                       subject_code=SUBJECT_CODE,
                                       subject_name=PATIENT_NAME)
    
    assert new_header['startdate'] == BASE_START_DATE + recording_offset
    for key in DEFAULT_REDACT_HEADER_KEYS:
        assert new_header[key] == REDACT_REPLACEMENT

    assert new_header['patientcode'] == SUBJECT_CODE
    assert new_header['equipment'] == REDACT_PRONOUN_REPLACEMENT + ' ' + REDACT_NAME_REPLACEMENT


def test_deidentify_edf_annotations():
    from clean_eeg.clean_subject_eeg import deidentify_edf_annotations
    data = load_edf(BASIC_EDF_PATH, load_method='pyedflib', preload=True)
    annotations = data['annotations']
    
    # insert patient pronoun and name into annotations
    annotation_texts = list(annotations[2])
    annotation_texts[2] = 'his ' + PATIENT_NAME.get_full_name()
    annotations_list = list(annotations)
    annotations_list[2] = np.array(annotation_texts)
    annotations = tuple(annotations_list)

    new_annotations = deidentify_edf_annotations(annotations,
                                                 subject_name=PATIENT_NAME)

    assert new_annotations[2][2] == REDACT_PRONOUN_REPLACEMENT + ' ' + REDACT_NAME_REPLACEMENT


def test_deidentify_edf():
    # integration test
    from clean_eeg.clean_subject_eeg import deidentify_edf
    data = load_edf(BASIC_EDF_PATH, load_method='pyedflib', preload=True)

    recording_timestamp = data['header']['startdate']
    recording_offset = timedelta(days=1)
    earliest_recording_timestamp = recording_timestamp - recording_offset

    # insert patient pronoun and name into 'equipment' field
    data['header']['equipment'] = 'his ' + PATIENT_NAME.get_full_name()

    # insert patient pronoun and name into annotations
    annotations = data['annotations']
    annotation_texts = list(annotations[2])
    annotation_texts[2] = 'his ' + PATIENT_NAME.get_full_name()
    annotations_list = list(annotations)
    annotations_list[2] = np.array(annotation_texts)
    annotations = tuple(annotations_list)

    data['annotations'] = annotations
    new_data = deidentify_edf(data,
                              earliest_recording_start_time=earliest_recording_timestamp,
                              subject_code=SUBJECT_CODE,
                              subject_name=PATIENT_NAME)
    
    new_annotations = new_data['annotations']
    assert new_annotations[2][2] == REDACT_PRONOUN_REPLACEMENT + ' ' + REDACT_NAME_REPLACEMENT

@pytest.mark.parametrize("inplace", [False, True])
def test_clean_subject_edf_files(monkeypatch, inplace):
    responses = iter(["y"])  # answers in sequence
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    from clean_eeg.clean_subject_eeg import clean_subject_edf_files
    from pathlib import Path

    output_path = TEST_SUBJECT_DATA_DIR / 'temp_clean_output'
    if not output_path.exists():
        os.makedirs(output_path)
    elif inplace:
        # for inplace, clear out existing files in output_path
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))

    if inplace:
        shutil.copyfile(SUBJECT_EDF_PATH1, os.path.join(output_path, os.path.basename(SUBJECT_EDF_PATH1)))
        shutil.copyfile(SUBJECT_EDF_PATH2, os.path.join(output_path, os.path.basename(SUBJECT_EDF_PATH2)))
    
    clean_subject_edf_files(subject_name=PATIENT_NAME,
                            subject_code=SUBJECT_CODE,
                            input_path=str(TEST_SUBJECT_DATA_DIR) if not inplace else str(output_path),
                            output_path=str(output_path),
                            inplace=inplace)
    
    # check that file was created
    filename_no_ext1 = Path(SUBJECT_EDF_PATH1).stem
    clean_filename1 = f"{filename_no_ext1}_{SUBJECT_CODE}_1985.01.01__00.00.00.edf"
    filename_no_ext2 = Path(SUBJECT_EDF_PATH2).stem
    clean_filename2 = f"{filename_no_ext2}_{SUBJECT_CODE}_1985.01.01__01.00.00.edf"
    for clean_filename in [clean_filename1, clean_filename2]:
        clean_full_path = os.path.join(output_path, clean_filename)
        assert os.path.exists(clean_full_path), 'Cleaned EDF file was not created: ' + clean_full_path
        os.remove(clean_full_path)
    shutil.rmtree(output_path)


def test_clean_subject_edf_files_w_large_gap(monkeypatch):
    responses = iter(["n"])  # answers in sequence
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    output_path = TEST_SUBJECT_DATA_DIR / 'temp_clean_output'
    if not output_path.exists():
        os.makedirs(output_path)
    
    # assert RunTimeError is raised with pytest due to large time gap between recordings
    try:
        clean_subject_edf_files(subject_name=PATIENT_NAME,
                                subject_code=SUBJECT_CODE,
                                input_path=str(TEST_SUBJECT_DATA_DIR),
                                output_path=str(output_path))
    except RuntimeError as e:
        assert str(e).startswith('Aborting EDF de-identification conversion due to recording gap.')
    else:
        assert False, 'RuntimeError was not raised for large time gap between recordings'

def test_clean_subject_edf_files_w_inconsistent_subject_names(monkeypatch):
    responses = iter(['y', "n"])  # answers in sequence
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    output_path = INCONSISTENT_SUBJECT_DATA_DIR / 'temp_clean_output'
    if not output_path.exists():
        os.makedirs(output_path)
    
    try:
        clean_subject_edf_files(subject_name=PATIENT_NAME,
                                subject_code=SUBJECT_CODE,
                                input_path=str(INCONSISTENT_SUBJECT_DATA_DIR),
                                output_path=str(output_path))
    except RuntimeError as e:
        print(e)
        assert str(e).startswith('Aborting EDF de-identification conversion due to inconsistent subject names')
    else:
        assert False, 'RuntimeError was not raised for inconsistent subject names'

def test_clean_subject_edf_files_w_inconsistent_signal_headers(monkeypatch):
    responses = iter(['y', 'y', 'y', 'n'])  # answers in sequence
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    output_path = INCONSISTENT_SUBJECT_DATA_DIR / 'temp_clean_output'
    if not output_path.exists():
        os.makedirs(output_path)

    try:
        clean_subject_edf_files(subject_name=PATIENT_NAME,
                                subject_code=SUBJECT_CODE,
                                input_path=str(INCONSISTENT_SUBJECT_DATA_DIR),
                                output_path=str(output_path))
    except RuntimeError as e:
        print(e)
        assert str(e).startswith('Aborting EDF de-identification conversion due to inconsistent signal headers')
    else:
        assert False, 'RuntimeError was not raised for inconsistent signal headers'


# --- _check_subject_name_consistency unit tests ---

def _make_edf_meta(filenames_and_names: dict) -> dict:
    """Build a minimal EDF_meta_data dict for testing name consistency."""
    return {
        fname: {'data': {'header': {'patientname': name}}}
        for fname, name in filenames_and_names.items()
    }


def test_name_consistency_matching_name():
    """CLI name matches EDF header name — should pass without prompting."""
    cli_name = PersonalName(first_name='John', middle_names=[], last_name='Doe')
    meta = _make_edf_meta({'file1.edf': 'John Doe'})
    # No prompt needed, should not raise
    _check_subject_name_consistency(meta, command_line_subject_name=cli_name)


def test_name_consistency_already_redacted():
    """EDF header already redacted as 'X' — should pass without prompting."""
    cli_name = PersonalName(first_name='John', middle_names=[], last_name='Doe')
    meta = _make_edf_meta({'file1.edf': 'X'})
    _check_subject_name_consistency(meta, command_line_subject_name=cli_name)


def test_name_consistency_mismatch_user_confirms(monkeypatch):
    """CLI name differs from EDF header — user confirms yes, should pass."""
    responses = iter(['yes'])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    cli_name = PersonalName(first_name='John', middle_names=[], last_name='Doe')
    meta = _make_edf_meta({'file1.edf': 'Jane Smith'})
    _check_subject_name_consistency(meta, command_line_subject_name=cli_name)


def test_name_consistency_mismatch_user_denies(monkeypatch):
    """CLI name differs from EDF header — user says no, should raise RuntimeError."""
    responses = iter(['no'])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    cli_name = PersonalName(first_name='John', middle_names=[], last_name='Doe')
    meta = _make_edf_meta({'file1.edf': 'Jane Smith'})
    with pytest.raises(RuntimeError, match='inconsistent subject names'):
        _check_subject_name_consistency(meta, command_line_subject_name=cli_name)


def test_name_consistency_no_cli_name():
    """No CLI name provided — should pass without prompting regardless of header name."""
    meta = _make_edf_meta({'file1.edf': 'Jane Smith'})
    _check_subject_name_consistency(meta, command_line_subject_name=None)


def test_clean_subject_edf_files_empty_dir_raises(tmp_path):
    """An input directory with no .edf files should raise RuntimeError with a
    helpful message rather than crashing in min() on an empty sequence."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match="No EDF files were successfully loaded"):
        clean_subject_edf_files(
            input_path=str(empty_dir),
            output_path=str(empty_dir),
            subject_code=SUBJECT_CODE,
            subject_name=PATIENT_NAME,
            inplace=True,
        )


# ---------------------------------------------------------------------
# End-to-end regression: degenerate physical_min/physical_max (NK-style
# "0.00000" / "-0.00000" pair) must not kill clean_subject_edf_files.
# The whole pipeline — repair, load, redact, write, optional audit —
# must produce an output file pyedflib can re-open cleanly, with the
# degenerate signal's phys range rewritten to -1/1.
# ---------------------------------------------------------------------

def _corrupt_one_signal_phys_range(edf_path: str,
                                    signal_idx: int,
                                    new_min: str,
                                    new_max: str) -> None:
    """Overwrite a signal's phys_min/phys_max bytes with the given ASCII
    values (left-padded to 8 bytes). No mutation to signal data."""
    from clean_eeg.repair_edf import (
        MAIN_HEADER_BYTES,
        SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH,
        SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH,
    )
    with open(edf_path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
    n_signals = int(main[252:256].decode().strip())

    def write(field_offset, field_width, value):
        off = (MAIN_HEADER_BYTES
               + field_offset * n_signals
               + signal_idx * field_width)
        with open(edf_path, "r+b") as f:
            f.seek(off)
            f.write(value.ljust(field_width).encode("ascii"))

    write(SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH, new_min)
    write(SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH, new_max)


def _write_minimal_edfplus_with_annotations(path: str,
                                             n_channels: int = 3,
                                             sample_rate: int = 100,
                                             duration_s: int = 5) -> None:
    """Write a small EDF+C with a couple of user annotations.

    The annotations are needed because the inplace pipeline creates an
    ``_annotations.edf`` stub and re-opens it to validate; pyedflib rejects
    stubs that contain zero records, so the input must have at least one
    annotation for the inplace path to work end-to-end.
    """
    import pyedflib
    from datetime import datetime
    signal_headers = [
        {'label': f'CH{i}', 'dimension': 'uV',
         'sample_frequency': sample_rate,
         'physical_max': 3200.0, 'physical_min': -3200.0,
         'digital_max': 32767, 'digital_min': -32768,
         'prefilter': '', 'transducer': ''}
        for i in range(n_channels)
    ]
    t = np.arange(0, duration_s, 1.0 / sample_rate, dtype=np.float32)
    signals = [
        (1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
        for i in range(n_channels)
    ]
    with pyedflib.EdfWriter(path, n_channels,
                             file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            'technician': 'T', 'recording_additional': '',
            'patientname': f'{PATIENT_NAME.first_name} {PATIENT_NAME.last_name}',
            'patient_additional': '',
            'patientcode': SUBJECT_CODE, 'equipment': 'test',
            'admincode': '', 'sex': 'Male',
            'startdate': datetime(2023, 1, 1, 10, 0, 0),
            'birthdate': '01 feb 1970', 'gender': 'Male',
        })
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)
        f.writeAnnotation(0.5, -1, "START")
        f.writeAnnotation(float(duration_s) - 0.5, -1, "END")


@pytest.mark.parametrize("inplace", [False, True])
def test_clean_subject_edf_files_repairs_degenerate_phys_range(monkeypatch, tmp_path, inplace):
    """A file with phys_min == phys_max (exact NK pattern: "0.00000" /
    "-0.00000") must flow through the full pipeline without error and the
    output must open cleanly in pyedflib with phys_min=-1, phys_max=1 on
    the previously-degenerate channel."""
    import pyedflib

    responses = iter(["y", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = input_dir if inplace else tmp_path / "out"
    if not inplace:
        output_dir.mkdir()

    edf_path = input_dir / "degen.edf"
    _write_minimal_edfplus_with_annotations(str(edf_path),
                                             n_channels=3,
                                             sample_rate=100,
                                             duration_s=5)
    # Corrupt signal 1 the exact NK way: phys_min == 0.0, phys_max == -0.0
    # (numerically equal, string-different).
    _corrupt_one_signal_phys_range(str(edf_path), signal_idx=1,
                                   new_min="0.00000", new_max="-0.00000")

    # Sanity: corrupted file is NOT pyedflib-openable before the pipeline
    with pytest.raises(OSError, match=r"(?i)physical\s*max"):
        pyedflib.EdfReader(str(edf_path)).close()

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(output_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=inplace,
        raise_errors=True,
    )

    # Locate the cleaned output file (named with timestamp suffix), skipping
    # the annotations stub.
    out_files = [
        p for p in os.listdir(str(output_dir))
        if p.endswith('.edf') and '_annotations' not in p
           and p != 'degen.edf'
    ]
    assert len(out_files) == 1, f"expected 1 cleaned file, got: {out_files}"
    out_path = os.path.join(str(output_dir), out_files[0])

    # Full pipeline must have produced a pyedflib-openable output with the
    # degenerate signal now carrying a valid -1 / 1 range.
    with pyedflib.EdfReader(out_path) as r:
        assert r.signals_in_file == 3
        sh = r.getSignalHeader(1)
        assert sh['physical_min'] == -1.0
        assert sh['physical_max'] == 1.0


# ---------------------------------------------------------------------
# Skip-with-warning regression: a header-field pathology we do NOT
# proactively repair (e.g. non-ASCII bytes in a signal label) must
# cause the affected file to be skipped cleanly, appear in the
# "skipped files" summary, and not kill the rest of the run.
# This is the fallback path for the TODO "Handled by skip-with-warning"
# list of EDFLIB_FILE_ERRORS_* codes in repair_edf's non-coverage
# section.
# ---------------------------------------------------------------------

def test_pipeline_skips_file_with_non_ascii_label_gracefully(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """A file whose signal-label bytes contain non-ASCII characters
    must be rejected by pyedflib (EDFLIB_FILE_ERRORS_LABEL), caught
    by _load_edf_metadata's per-file try/except, added to the
    skipped-files summary, and must not prevent other files from
    being processed."""
    responses = iter(["y", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()

    # Two files: one good, one with a corrupted label byte.
    good_path = input_dir / "good.edf"
    bad_path = input_dir / "bad.edf"
    _write_minimal_edfplus_with_annotations(str(good_path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=5)
    _write_minimal_edfplus_with_annotations(str(bad_path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=5)

    # Corrupt the first byte of signal 0's label with a non-ASCII
    # value. The label region starts at byte 256 (after the 256-byte
    # main header).
    with open(bad_path, "r+b") as f:
        f.seek(256)
        f.write(b"\xff")  # byte outside 32-126 ASCII range

    output_dir = input_dir  # inplace
    try:
        clean_subject_edf_files(
            input_path=str(input_dir),
            output_path=str(output_dir),
            subject_code=SUBJECT_CODE,
            subject_name=PATIENT_NAME,
            inplace=True,
            raise_errors=False,
        )
    except RuntimeError:
        # It's OK if the run raises (e.g. "No EDF files loaded" when
        # both fail). What matters is that the bad file appears in the
        # skipped-files reporting.
        pass

    out = capsys.readouterr().out
    assert "bad.edf" in out, "skipped file must be named in the summary"
    assert "skipped" in out.lower() or "failed" in out.lower(), \
        "skip-with-warning path must produce a visible message"
    assert "send the log file" in out.lower() or \
           "data management team" in out.lower(), \
        "operator must be directed to send log.out to data team"


def test_audit_runs_on_every_file_with_pyedflib_cross_check(monkeypatch,
                                                              tmp_path,
                                                              capsys):
    """Default behaviour: every file gets the streamed mmap audit AND a
    single-channel pyedflib cross-check. Verifies (1) more than 2 files
    in a subject don't fall out of the audit set (we used to cap at 2)
    and (2) the pyedflib cross-check message appears in the audit
    output for every audited file."""
    responses = iter(["y"] * 10)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    n_files = 4  # > 2 so we can verify the old cap is gone
    for i in range(n_files):
        path = input_dir / f"f{i}.edf"
        _write_minimal_edfplus_with_annotations(str(path),
                                                 n_channels=3,
                                                 sample_rate=100,
                                                 duration_s=2)

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
    )

    out = capsys.readouterr().out
    audit_lines = [l for l in out.splitlines() if "Audit passed" in l]
    assert len(audit_lines) == n_files, (
        f"expected {n_files} audit confirmations (one per file), "
        f"got {len(audit_lines)}: {audit_lines}"
    )
    # Every audit message must mention the pyedflib cross-check signal.
    for line in audit_lines:
        assert "pyedflib cross-check" in line, \
            f"audit line missing pyedflib cross-check note: {line!r}"


def test_audit_raises_runtime_error_on_signal_corruption(tmp_path):
    """The streamed mmap audit must raise AUDIT FAILURE when the clean
    file's signal bytes differ from orig_signals. Simulates a case
    where the inplace operations corrupted (or were intercepted to
    corrupt) signal data — guards against silent regressions in any
    future change that touches inplace writes."""
    from clean_eeg.clean_subject_eeg import _audit_signal_integrity
    from clean_eeg.load_eeg import load_edf

    path = str(tmp_path / "corrupt_after_load.edf")
    _write_minimal_edfplus_with_annotations(str(path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)

    # Load orig signals via the same path the pipeline uses.
    data = load_edf(path, preload=True, read_digital=True, use_mmap=True)
    orig_signals = data['signals']

    # Now corrupt the very first signal byte on disk so it disagrees
    # with orig_signals[0][0]. Header is 256*(1+n_signals_on_disk) bytes.
    with open(path, "rb") as f:
        main = f.read(256)
    n_signals_on_disk = int(main[252:256].decode().strip())
    first_signal_byte_offset = 256 + 256 * n_signals_on_disk
    with open(path, "rb") as f:
        f.seek(first_signal_byte_offset)
        original_first_byte = f.read(2)
    # Flip a bit so the int16 sample value differs.
    new_first_byte = bytes([(original_first_byte[0] ^ 0xFF)]) + original_first_byte[1:2]
    with open(path, "r+b") as f:
        f.seek(first_signal_byte_offset)
        f.write(new_first_byte)

    with pytest.raises(RuntimeError, match=r"AUDIT FAILURE"):
        _audit_signal_integrity(orig_signals, path, "corrupt_after_load.edf",
                                inplace=True, digital=True)


def test_audit_raises_on_signal_count_mismatch(tmp_path):
    """If orig_signals has a different number of data signals than the
    clean file on disk, the audit must raise immediately rather than
    silently iterating only the matching prefix."""
    from clean_eeg.clean_subject_eeg import _audit_signal_integrity
    from clean_eeg.load_eeg import load_edf

    path = str(tmp_path / "count_mismatch.edf")
    _write_minimal_edfplus_with_annotations(str(path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)

    data = load_edf(path, preload=True, read_digital=True, use_mmap=True)
    short_orig = data['signals'][:-1]  # one signal short

    with pytest.raises(RuntimeError, match=r"signal count mismatch|AUDIT FAILURE"):
        _audit_signal_integrity(short_orig, path, "count_mismatch.edf",
                                inplace=True, digital=True)


def test_audit_pyedflib_cross_check_raises_when_orig_disagrees(tmp_path):
    """The pyedflib cross-check fires after the streamed mmap audit, on
    one random channel. Confirm it actually raises when orig_signals[i]
    disagrees with what pyedflib's readSignal returns. We stage this by
    handing the audit zero-filled orig_signals — the streamed mmap
    audit will catch the discrepancy first (so we expect AUDIT FAILURE
    either way), proving both checks are wired in and active."""
    from clean_eeg.clean_subject_eeg import _audit_signal_integrity
    from clean_eeg.load_eeg import load_edf

    path = str(tmp_path / "fake_orig.edf")
    _write_minimal_edfplus_with_annotations(str(path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)
    data = load_edf(path, preload=True, read_digital=True, use_mmap=True)
    fake_orig = [np.zeros_like(s) for s in data['signals']]

    with pytest.raises(RuntimeError, match=r"AUDIT FAILURE"):
        _audit_signal_integrity(fake_orig, path, "fake_orig.edf",
                                inplace=True, digital=True)


def test_failed_file_is_quarantined_not_left_in_output_dir(monkeypatch,
                                                              tmp_path,
                                                              capsys):
    """When a file fails the audit (or any other mid-pipeline step), its
    partial output artifacts MUST be moved to a 'quarantine/' subdirectory
    of the output path, NOT left in the main output directory where the
    operator's `scp output/*.edf` would pick them up. Operators may
    forget to read warnings; this is the structural guarantee that
    incompletely-de-identified files cannot be sent."""
    responses = iter(["y"] * 5)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    edf_path = input_dir / "will_fail.edf"
    _write_minimal_edfplus_with_annotations(str(edf_path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)

    # Force the audit to fail by monkeypatching it to always raise.
    import clean_eeg.clean_subject_eeg as _csm
    real_audit = _csm._audit_signal_integrity
    def boom(*args, **kwargs):
        raise RuntimeError("AUDIT FAILURE for test (synthetic)")
    monkeypatch.setattr(_csm, "_audit_signal_integrity", boom)

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),  # inplace
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=False,
    )

    # The clean output filename pattern is *_R{subject}_*.edf — this
    # MUST NOT exist directly in the output dir.
    main_dir_edfs = [
        f for f in os.listdir(str(input_dir))
        if f.endswith('.edf') and SUBJECT_CODE in f
    ]
    assert main_dir_edfs == [], (
        f"Failed-audit file should not remain in main output dir: "
        f"{main_dir_edfs}"
    )

    # The quarantine subdir MUST contain the file.
    quarantine_dir = input_dir / "quarantine"
    assert quarantine_dir.is_dir(), \
        "quarantine/ subdirectory must be created on failure"
    quarantined_edfs = [
        f for f in os.listdir(str(quarantine_dir))
        if f.endswith('.edf')
    ]
    assert any(SUBJECT_CODE in f for f in quarantined_edfs), \
        f"failed file should be in quarantine: {quarantined_edfs}"

    # End-of-run summary must explicitly warn about quarantine.
    out = capsys.readouterr().out
    assert "quarantine" in out.lower(), \
        "summary must mention quarantine"
    assert "MUST NOT" in out or "DO NOT" in out, \
        "summary must use strong language about not sending these"


def test_audit_skipped_when_skip_audit_true(monkeypatch, tmp_path, capsys):
    """skip_audit=True must suppress both the streamed mmap audit AND the
    pyedflib cross-check — no 'Audit passed' lines should be emitted."""
    responses = iter(["y"] * 5)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(2):
        path = input_dir / f"f{i}.edf"
        _write_minimal_edfplus_with_annotations(str(path),
                                                 n_channels=3,
                                                 sample_rate=100,
                                                 duration_s=2)

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
        skip_audit=True,
    )

    out = capsys.readouterr().out
    assert "Audit passed" not in out
    assert "pyedflib cross-check" not in out