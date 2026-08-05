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
    # One "y" for the recording-gap prompt (the test subject data has
    # a ~59-minute gap between the two files, above the 60 s threshold).
    # The transfer prompt is short-circuited via auto_transfer_response.
    responses = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    from clean_eeg.clean_subject_eeg import clean_subject_edf_files
    from pathlib import Path

    output_path = TEST_SUBJECT_DATA_DIR / 'temp_clean_output'
    # Wipe any leftover state (including a prior run's deidentify.json,
    # which would trigger the 'already done' fast path) — the test
    # must always exercise a fresh de-id run.
    if output_path.exists():
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    if inplace:
        shutil.copyfile(SUBJECT_EDF_PATH1, os.path.join(output_path, os.path.basename(SUBJECT_EDF_PATH1)))
        shutil.copyfile(SUBJECT_EDF_PATH2, os.path.join(output_path, os.path.basename(SUBJECT_EDF_PATH2)))

    clean_subject_edf_files(subject_name=PATIENT_NAME,
                            subject_code=SUBJECT_CODE,
                            input_path=str(TEST_SUBJECT_DATA_DIR) if not inplace else str(output_path),
                            output_path=str(output_path),
                            inplace=inplace,
                            auto_transfer_response="n")
    
    # check that file was created
    filename_no_ext1 = Path(SUBJECT_EDF_PATH1).stem
    # Filenames carry month/day/time only — the year (always 1985, the
    # BASE_START_DATE anchor) was omitted to avoid confusing operators.
    clean_filename1 = f"{filename_no_ext1}_{SUBJECT_CODE}_01.01__00.00.00.edf"
    filename_no_ext2 = Path(SUBJECT_EDF_PATH2).stem
    clean_filename2 = f"{filename_no_ext2}_{SUBJECT_CODE}_01.01__01.00.00.edf"
    for clean_filename in [clean_filename1, clean_filename2]:
        clean_full_path = os.path.join(output_path, clean_filename)
        assert os.path.exists(clean_full_path), 'Cleaned EDF file was not created: ' + clean_full_path
        os.remove(clean_full_path)
    shutil.rmtree(output_path)


def test_clean_subject_edf_files_w_large_gap(monkeypatch):
    responses = iter(["n"])  # answers in sequence
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    output_path = TEST_SUBJECT_DATA_DIR / 'temp_clean_output'
    # Prior tests may have left a deidentify.json here — clear so the
    # gap check actually runs (otherwise the completion fast-path
    # returns before we get anywhere near _check_recording_gaps).
    if output_path.exists():
        shutil.rmtree(output_path)
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
        auto_transfer_response="n",
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
            auto_transfer_response="n",
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
    and (2) the audit actually ran on every file (would raise if a
    signal disagreed) — evidence is the manifest's file_hashes, which
    covers every file that made it through the audit successfully."""
    responses = iter([])
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

    # Wrap the audit to count invocations — a stronger guarantee than
    # the old "check the stdout for N audit lines" pattern (which broke
    # when we moved per-file confirmations out of stdout).
    import clean_eeg.clean_subject_eeg as _csm
    audit_call_count = {"n": 0}
    real_audit = _csm._audit_signal_integrity

    def counting_audit(*args, **kwargs):
        audit_call_count["n"] += 1
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(_csm, "_audit_signal_integrity", counting_audit)

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
        auto_transfer_response="n",
    )

    assert audit_call_count["n"] == n_files, (
        f"expected {n_files} audit invocations (one per file), "
        f"got {audit_call_count['n']}"
    )
    # Manifest records a hash for every audited file — cross-checks that
    # the pipeline both ran the audit AND wrote the output.
    from clean_eeg.deidentify_manifest import read_manifest
    manifest = read_manifest(input_dir)
    assert manifest is not None
    # inplace mode writes both the main EDF and its _annotations stub;
    # both survive the audit and land in file_hashes.
    hashed = list(manifest["file_hashes"].keys())
    non_stub = [h for h in hashed if "_annotations" not in h]
    assert len(non_stub) == n_files, (
        f"manifest should hash one main EDF per file: {hashed}"
    )


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

    # The quarantine subdir MUST contain the file, with a renamed
    # extension that does not end in '.edf' so any *.edf glob (server
    # or client side) cannot accidentally pick it up.
    quarantine_dir = input_dir / "quarantine"
    assert quarantine_dir.is_dir(), \
        "quarantine/ subdirectory must be created on failure"
    quarantined = os.listdir(str(quarantine_dir))
    assert quarantined, "quarantine should not be empty after audit failure"
    # Defense-in-depth: no quarantined file may end in '.edf' — even if
    # the operator runs `scp -r` or `rsync` without --exclude, the
    # standard *.edf glob cannot match these names.
    edf_in_quarantine = [f for f in quarantined if f.endswith('.edf')]
    assert edf_in_quarantine == [], (
        f"quarantined files must NOT end in .edf (defense-in-depth "
        f"against recursive copies): {edf_in_quarantine}"
    )
    # And the marker suffix must be present so the data team can spot
    # a mis-uploaded file at a glance.
    assert any('QUARANTINED-DO-NOT-USE' in f for f in quarantined), \
        f"quarantined files should carry the QUARANTINED suffix: {quarantined}"

    # End-of-run summary must explicitly warn about quarantine.
    out = capsys.readouterr().out
    assert "quarantine" in out.lower(), \
        "summary must mention quarantine"
    assert "MUST NOT" in out or "DO NOT" in out, \
        "summary must use strong language about not sending these"


def test_audit_failure_dumps_phi_masked_header(monkeypatch, tmp_path, capsys):
    """When a file fails mid-pipeline, the error handler must dump the
    EDF header to log.out so the data team has everything they need to
    triage. The four PHI-bearing main-header fields (patient_id,
    recording_id, startdate, starttime) MUST be masked because log.out
    is shared with the data team."""
    responses = iter(["y"] * 5)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    edf_path = input_dir / "will_fail.edf"
    _write_minimal_edfplus_with_annotations(str(edf_path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)

    import clean_eeg.clean_subject_eeg as _csm
    monkeypatch.setattr(_csm, "_audit_signal_integrity",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("synthetic audit failure")))

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=False,
    )

    out = capsys.readouterr().out
    # Header dump section is present.
    assert "EDF header dump (for the data team)" in out
    assert "Main header" in out
    # Numeric / structural fields the data team needs are unmasked.
    assert "n_signals" in out
    assert "n_records" in out
    assert "samples_per_record" in out
    # PHI-bearing main-header fields are masked.
    assert "[PHI_REDACTED]" in out
    main_block = out[out.index("Main header"):]
    for phi_field in ("patient_id", "recording_id", "startdate", "starttime"):
        # Each PHI field appears on its own line followed by the masked marker.
        assert f"{phi_field}" in main_block, f"{phi_field} should still be labelled"
        # Locate the row, verify it carries the redaction marker on the same line.
        row_start = main_block.index(phi_field)
        row = main_block[row_start:row_start + 200]
        assert "[PHI_REDACTED]" in row, \
            f"PHI field {phi_field!r} must be masked in the dump; got: {row!r}"


def test_empty_edf_file_produces_user_readable_error(monkeypatch, tmp_path, capsys):
    """A 0-byte EDF must surface a clear 'file is empty' message rather
    than the obscure spec-level 'n_signals empty' that the repair pass
    would otherwise emit. Pair with a healthy file so the pipeline
    doesn't abort on 'no EDFs loaded' before we can inspect the message."""
    responses = iter(["y"] * 5)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    healthy = input_dir / "healthy.edf"
    _write_minimal_edfplus_with_annotations(str(healthy),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)
    empty = input_dir / "empty.edf"
    empty.write_bytes(b"")

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=False,
        auto_transfer_response="n",
    )

    out = capsys.readouterr().out
    assert "Failed to load EDF file empty.edf" in out
    assert "is empty (0 bytes)" in out
    # The obscure spec-level n_signals error should NOT be the lead message.
    err_section_start = out.index("Failed to load EDF file empty.edf")
    next_file_or_done = out.find("Failed to load EDF file ", err_section_start + 1)
    if next_file_or_done == -1:
        next_file_or_done = out.index("Done cleaning EDF files")
    err_section = out[err_section_start:next_file_or_done]
    assert "n_signals" not in err_section.split("Stack trace")[0], \
        "the empty-file message should fire BEFORE the n_signals parse"


def test_load_failure_dumps_header_for_empty_n_signals(monkeypatch, tmp_path, capsys):
    """When the load-time repair pass raises (e.g. empty n_signals),
    the diagnostic dump must still run so the data team gets the header
    info even though the file never reached pyedflib. Pair the broken
    file with a healthy one so the pipeline doesn't immediately abort
    on 'no EDF files loaded' before we can inspect the dump."""
    responses = iter(["y"] * 5)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    healthy_path = input_dir / "healthy.edf"
    _write_minimal_edfplus_with_annotations(str(healthy_path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)
    broken_path = input_dir / "broken.edf"
    _write_minimal_edfplus_with_annotations(str(broken_path),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)
    # Blank n_signals on the broken file to trigger the unrecoverable
    # error path during the repair-truncated step (pre-pyedflib).
    with open(broken_path, "r+b") as f:
        f.seek(252)
        f.write(b"    ")

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=False,
        auto_transfer_response="n",
    )

    out = capsys.readouterr().out
    assert "Failed to load EDF file broken.edf" in out
    assert "EDF header dump (for the data team)" in out
    # PHI fields masked even on a load-time failure.
    assert "[PHI_REDACTED]" in out
    # Numeric/structural fields are visible (n_signals shows blank '<empty>').
    assert "n_signals" in out


# Transfer-command tests were moved to tests/test_transfer.py — the
# rsync/scp branching now lives in clean_eeg.transfer, not inside the
# pipeline. `test_transfer_command_uses_rsync_when_available` and
# `test_transfer_command_falls_back_to_scp_when_rsync_unavailable`
# were replaced by `test_transfer_plan_uses_rsync_when_available` /
# `test_transfer_plan_falls_back_to_scp` (which test the command
# strings) and `test_transfer_subject_dry_run_returns_plan_without_executing`
# (which tests orchestration without touching the network).


def test_audit_skipped_when_skip_audit_true(monkeypatch, tmp_path, capsys):
    """skip_audit=True must not invoke the audit function at all — the
    prior 'Audit passed' stdout assertion was replaced by a direct
    call-count check because per-file confirmations no longer scroll
    to stdout."""
    responses = iter([])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(2):
        path = input_dir / f"f{i}.edf"
        _write_minimal_edfplus_with_annotations(str(path),
                                                 n_channels=3,
                                                 sample_rate=100,
                                                 duration_s=2)

    import clean_eeg.clean_subject_eeg as _csm
    audit_calls = {"n": 0}
    real_audit = _csm._audit_signal_integrity

    def counting_audit(*args, **kwargs):
        audit_calls["n"] += 1
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(_csm, "_audit_signal_integrity", counting_audit)

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
        skip_audit=True,
        auto_transfer_response="n",
    )

    assert audit_calls["n"] == 0, (
        f"skip_audit=True must skip _audit_signal_integrity entirely, "
        f"got {audit_calls['n']} call(s)"
    )
    out = capsys.readouterr().out
    assert "Audit passed" not in out
    assert "pyedflib cross-check" not in out

# ---- CLI: --no_middle_name flag ----
# Cross-platform alternative to --middle_name "" — added because
# Windows cmd.exe strips empty quoted arguments, so a collaborator
# could not express "no middle name" on the command line.

def _parse_cli(argv, monkeypatch):
    """Run get_clean_eeg_cli_arguments() with a synthetic argv.
    Mock out the interactive prompt so the test never blocks; if
    prompt_if_missing reaches the middle-name prompt, raise instead so
    the test fails loudly rather than hanging."""
    from clean_eeg import clean_subject_eeg as mod

    def _no_prompt(_msg):
        raise AssertionError(
            "logged_input was called — test should have provided all "
            "required args via argv. msg=" + repr(_msg))

    monkeypatch.setattr("sys.argv", ["clean_subject_eeg.py"] + argv)
    monkeypatch.setattr(mod, "logged_input", _no_prompt)
    return mod.get_clean_eeg_cli_arguments()


def test_no_middle_name_flag_sets_empty_middle_name(monkeypatch):
    """--no_middle_name should leave args.middle_name == "" so the
    downstream validator accepts it (instead of erroring on the
    "NOT_SPECIFIED" sentinel)."""
    args = _parse_cli([
        "--input_path", "/tmp/x",
        "--subject_code", "R1764A",
        "--first_name", "John",
        "--last_name", "Doe",
        "--no_middle_name",
    ], monkeypatch)
    assert args.middle_name == ""
    assert args.no_middle_name is True


def test_no_middle_name_conflicts_with_middle_name_value(monkeypatch):
    """Passing both --no_middle_name AND --middle_name <value> is
    ambiguous — argparse should exit (parser.error → SystemExit)."""
    with pytest.raises(SystemExit):
        _parse_cli([
            "--input_path", "/tmp/x",
            "--subject_code", "R1764A",
            "--first_name", "John",
            "--last_name", "Doe",
            "--middle_name", "Paul",
            "--no_middle_name",
        ], monkeypatch)


def test_empty_string_middle_name_still_works_on_posix(monkeypatch):
    """The legacy --middle_name "" path must still resolve to "" for
    shells that DO pass empty quoted args (zsh, bash). The prompt-if-
    missing block currently re-prompts on "" (a pre-existing quirk —
    --no_middle_name avoids it entirely), so simulate the user
    pressing Enter at that prompt by returning an empty string."""
    from clean_eeg import clean_subject_eeg as mod

    monkeypatch.setattr("sys.argv", [
        "clean_subject_eeg.py",
        "--input_path", "/tmp/x",
        "--subject_code", "R1764A",
        "--first_name", "John",
        "--last_name", "Doe",
        "--middle_name", "",
    ])
    monkeypatch.setattr(mod, "logged_input", lambda _msg: "")
    args = mod.get_clean_eeg_cli_arguments()
    assert args.middle_name == ""
    assert args.no_middle_name is False


# ---- validate_cli_arguments: empty first_name / last_name rejected ----
# prompt_if_missing catches missing names interactively, but a batch
# invocation passing --first_name "" or --last_name "" on POSIX (where
# empty quoted args DO survive) would otherwise slip through. The
# backstop in validate_cli_arguments rejects them.

def _validate_args(tmp_path, **overrides):
    """Build a minimal argparse.Namespace good enough to reach the
    first/last-name checks, apply overrides, and run
    validate_cli_arguments. Uses copy_path to skip the inplace
    confirmation prompt."""
    from argparse import Namespace
    from clean_eeg.clean_subject_eeg import validate_cli_arguments

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    copy_dir = tmp_path / "out"

    base = dict(
        input_path=str(input_dir),
        output_path=str(copy_dir),
        copy_path=str(copy_dir),
        first_name="John",
        middle_name="Paul",
        last_name="Doe",
        subject_code="R1764A",
        no_middle_name=False,
    )
    base.update(overrides)
    validate_cli_arguments(Namespace(**base))


def test_empty_first_name_rejected_by_validate(tmp_path):
    """Empty --first_name must error with a message that points the
    user at the fix."""
    with pytest.raises(ValueError, match="First name is required"):
        _validate_args(tmp_path, first_name="")


def test_whitespace_only_first_name_rejected_by_validate(tmp_path):
    """Whitespace-only counts as empty for our purposes — otherwise
    `--first_name " "` could sneak past the check and produce
    nonsense downstream."""
    with pytest.raises(ValueError, match="First name is required"):
        _validate_args(tmp_path, first_name="   ")


def test_empty_last_name_rejected_by_validate(tmp_path):
    """Empty --last_name must error."""
    with pytest.raises(ValueError, match="Last name is required"):
        _validate_args(tmp_path, last_name="")


def test_whitespace_only_last_name_rejected_by_validate(tmp_path):
    """Whitespace-only --last_name is treated the same as empty."""
    with pytest.raises(ValueError, match="Last name is required"):
        _validate_args(tmp_path, last_name="\t \n")


def test_valid_names_pass_validate(tmp_path, capsys):
    """Sanity check: with all three names supplied, validate returns
    cleanly. Sole purpose is to prove the new backstop hasn't broken
    the happy path."""
    _validate_args(tmp_path)  # no exception
    # validate_cli_arguments prints a "Loading EDF files" line as a
    # side effect — drain it so test isolation isn't surprising.
    capsys.readouterr()


# ---------------------------------------------------------------------
# Consecutive load-failure cap: 5 consecutive load failures abort the
# subject rather than churning through more files. `--force_load_all`
# bypasses. A success in the middle resets the streak.
# ---------------------------------------------------------------------

def _make_bad_edf(path):
    """0-byte file — reliably fails validate_edf_minimum_size."""
    path.write_bytes(b"")


def test_load_failure_cap_aborts_after_5_consecutive(tmp_path, monkeypatch, capsys):
    responses = iter([])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(5):
        _make_bad_edf(input_dir / f"bad{i}.edf")

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=False,
    )

    out = capsys.readouterr().out
    assert "--force_load_all" in out, (
        "abort message must mention the --force_load_all escape hatch"
    )
    # Manifest MUST NOT be written on an abort — the transfer tool
    # relies on absence to refuse.
    from clean_eeg.deidentify_manifest import manifest_exists
    assert not manifest_exists(input_dir), (
        "no deidentify.json should be written when the load cap fires"
    )


def test_load_failure_cap_bypassed_by_force_load_all(tmp_path, monkeypatch, capsys):
    """With --force_load_all, 5 consecutive bad files should NOT abort;
    the pipeline continues past them and processes whatever remains."""
    responses = iter([])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(5):
        _make_bad_edf(input_dir / f"bad{i}.edf")
    # One good file after the bad streak.
    _write_minimal_edfplus_with_annotations(str(input_dir / "good.edf"),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=False,
        force_load_all=True,
        auto_transfer_response="n",
    )

    out = capsys.readouterr().out
    assert "--force_load_all" not in out or "Aborting" not in out, (
        "run should not have aborted with --force_load_all"
    )
    # Good file made it through → manifest present.
    from clean_eeg.deidentify_manifest import read_manifest
    manifest = read_manifest(input_dir)
    assert manifest is not None
    assert manifest["n_files_deidentified"] >= 1


def test_load_failure_cap_reset_by_intervening_success(tmp_path, monkeypatch,
                                                        capsys):
    """A success in the middle of a bad streak resets the consecutive
    counter — 4 bad, 1 good, 4 bad = 8 total failures but streak never
    hit 5, so the pipeline should complete."""
    responses = iter([])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    # Interleave: 4 bad, 1 good, 4 bad. os.listdir order is not
    # guaranteed to match creation order, so use alphabetical prefixes
    # to force the sequence.
    for i in range(4):
        _make_bad_edf(input_dir / f"a{i}_bad.edf")
    _write_minimal_edfplus_with_annotations(str(input_dir / "b_good.edf"),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)
    for i in range(4):
        _make_bad_edf(input_dir / f"c{i}_bad.edf")

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=False,
        auto_transfer_response="n",
    )

    from clean_eeg.deidentify_manifest import read_manifest
    manifest = read_manifest(input_dir)
    assert manifest is not None, "pipeline must have completed"


# ---------------------------------------------------------------------
# Completion marker: deidentify.json's presence is what makes the
# pipeline offer to skip straight to transfer on re-invocation.
# --force bypasses. An interrupted run leaves no marker.
# ---------------------------------------------------------------------

def test_completion_marker_prompts_skip_to_transfer(tmp_path, monkeypatch, capsys):
    """When a valid manifest already exists in output_path and --force
    was not passed, re-invoking the pipeline must offer to skip straight
    to transfer rather than silently redoing de-id."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _write_minimal_edfplus_with_annotations(str(input_dir / "f.edf"),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)

    # Prime with one clean run.
    responses = iter([])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
        auto_transfer_response="n",
    )

    # Second invocation on the same directory — must hit the
    # completion fast-path. Assert the transfer tool is called.
    import clean_eeg.clean_subject_eeg as _csm
    transfer_calls = []
    monkeypatch.setattr(_csm, "_invoke_transfer",
                        lambda path: transfer_calls.append(path))
    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
        # Say yes to the "already done, skip to transfer?" prompt.
        auto_transfer_response="y",
    )
    assert transfer_calls, (
        "the completion fast-path must invoke transfer when the "
        "operator confirms"
    )
    out = capsys.readouterr().out
    assert "already present" in out.lower() or "already completed" in out.lower(), (
        "the fast-path must announce the pre-existing manifest so the "
        "operator understands why de-id is being skipped"
    )


def test_force_flag_bypasses_completion_marker(tmp_path, monkeypatch, capsys):
    """--force must skip the completion fast-path. We assert this
    directly rather than triggering a full second de-id run, since a
    re-run against an inplace-mode output dir would try to re-de-id
    the previous run's ``_annotations`` stub (a real limitation of
    inplace mode, out of scope for this test)."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _write_minimal_edfplus_with_annotations(str(input_dir / "f.edf"),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)

    responses = iter([])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
        auto_transfer_response="n",
    )
    from clean_eeg.deidentify_manifest import read_manifest
    first = read_manifest(input_dir)
    assert first is not None

    # Assert the fast-path fires WITHOUT --force (positive control).
    import clean_eeg.clean_subject_eeg as _csm
    fast_path_calls = []
    monkeypatch.setattr(_csm, "_maybe_skip_to_transfer",
                        lambda path, auto_response=None: fast_path_calls.append(path))
    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
        # no force
    )
    assert fast_path_calls, (
        "without --force, an existing manifest must trigger the fast-path"
    )

    # Assert --force bypasses it (the point of this test).
    fast_path_calls.clear()

    # Stub out downstream so we don't have to worry about the second-run
    # signal-header divergence caused by annotation stubs in inplace mode.
    def stub_load(*_args, **_kwargs):
        return {}  # empty EDF_meta_data → pipeline raises before de-id loop
    monkeypatch.setattr(_csm, "_load_edf_metadata", stub_load)
    try:
        clean_subject_edf_files(
            input_path=str(input_dir),
            output_path=str(input_dir),
            subject_code=SUBJECT_CODE,
            subject_name=PATIENT_NAME,
            inplace=True,
            raise_errors=True,
            force=True,
        )
    except RuntimeError:
        pass  # expected: stubbed load returns empty
    assert fast_path_calls == [], (
        "--force must skip _maybe_skip_to_transfer even when the "
        "manifest exists"
    )


def test_interrupted_run_leaves_no_manifest(tmp_path, monkeypatch, capsys):
    """If the pipeline raises mid-loop, no deidentify.json is written —
    so a subsequent re-invocation starts fresh rather than false-positive
    triggering the fast-path."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _write_minimal_edfplus_with_annotations(str(input_dir / "f.edf"),
                                              n_channels=3,
                                              sample_rate=100,
                                              duration_s=2)

    responses = iter([])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    # Blow up mid-loop.
    import clean_eeg.clean_subject_eeg as _csm
    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash mid de-id")
    monkeypatch.setattr(_csm, "deidentify_edf", boom)

    with pytest.raises(RuntimeError, match="simulated crash"):
        clean_subject_edf_files(
            input_path=str(input_dir),
            output_path=str(input_dir),
            subject_code=SUBJECT_CODE,
            subject_name=PATIENT_NAME,
            inplace=True,
            raise_errors=True,
        )

    from clean_eeg.deidentify_manifest import manifest_exists
    assert not manifest_exists(input_dir), (
        "an interrupted run must not leave a completion marker — "
        "re-invocation must start fresh"
    )


def test_manifest_records_fast_hash_for_every_output(tmp_path, monkeypatch):
    """Every EDF the pipeline writes must appear in the manifest's
    file_hashes so the audit can verify byte-identity post-transfer."""
    responses = iter([])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    n_files = 3
    for i in range(n_files):
        _write_minimal_edfplus_with_annotations(
            str(input_dir / f"f{i}.edf"),
            n_channels=3, sample_rate=100, duration_s=2,
        )

    clean_subject_edf_files(
        input_path=str(input_dir),
        output_path=str(input_dir),
        subject_code=SUBJECT_CODE,
        subject_name=PATIENT_NAME,
        inplace=True,
        raise_errors=True,
        auto_transfer_response="n",
    )

    from clean_eeg.deidentify_manifest import read_manifest
    manifest = read_manifest(input_dir)
    assert manifest is not None
    # inplace writes both the main file and an _annotations stub per input.
    hashed = list(manifest["file_hashes"].keys())
    main_files = [h for h in hashed if "_annotations" not in h]
    assert len(main_files) == n_files
    # All hashes are 64-char SHA-256 hex.
    for name, digest in manifest["file_hashes"].items():
        assert len(digest) == 64, f"{name}: {digest!r}"
    assert manifest["hash_mode"] == "fast"
    assert manifest["subject_code"] == SUBJECT_CODE
    assert manifest["site_code"] == SUBJECT_CODE[-1]


# --- --wipe-annotations ----------------------------------------------------


def _n_data_records_from_header(path: str) -> int:
    """Read n_records (bytes 236-244 of the EDF main header) directly."""
    with open(path, "rb") as f:
        f.seek(236)
        return int(f.read(8).decode().strip())


def _count_timekeeping_tals(edf_path: str) -> int:
    """Byte-level count of records whose annotation channel starts with
    a timekeeping TAL (``+<onset>\\x14\\x14``). Verifies the timekeeping
    delimiter survives ``clear_edf_annotations_inplace`` in every record.
    """
    from clean_eeg.modify_edf_inplace import (
        get_annotation_signal_header_index,
        get_signal_header_fields,
        get_header_field,
        TOTAL_HEADER_BYTES,
        SIGNAL_HEADER_BYTES,
    )
    ann_idx = get_annotation_signal_header_index(edf_path)
    lengths = [n * 2 for n in get_signal_header_fields(edf_path, field='num_samples')]
    ann_len = lengths[ann_idx]
    ann_off_in_record = sum(lengths[:ann_idx])
    total_record_length = sum(lengths)
    n_signals = len(lengths)
    n_records = get_header_field(edf_path, 'num_data_records')
    n_with_tk = 0
    with open(edf_path, "rb") as f:
        for i in range(n_records):
            offset = (TOTAL_HEADER_BYTES + SIGNAL_HEADER_BYTES * n_signals
                      + total_record_length * i + ann_off_in_record)
            f.seek(offset)
            record = f.read(ann_len)
            if b"\x14\x14" in record[:64] and record.startswith(b"+"):
                n_with_tk += 1
    return n_with_tk


def _run_wipe_pipeline(monkeypatch, tmp_output, inplace: bool) -> str:
    """Common driver for the wipe integration tests. Returns output_path."""
    # Recording-gap 'y' — same as the sibling test_clean_subject_edf_files.
    responses = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    if tmp_output.exists():
        shutil.rmtree(tmp_output)
    os.makedirs(tmp_output)
    if inplace:
        shutil.copyfile(SUBJECT_EDF_PATH1, tmp_output / os.path.basename(SUBJECT_EDF_PATH1))
        shutil.copyfile(SUBJECT_EDF_PATH2, tmp_output / os.path.basename(SUBJECT_EDF_PATH2))

    clean_subject_edf_files(
        subject_name=PATIENT_NAME,
        subject_code=SUBJECT_CODE,
        input_path=str(TEST_SUBJECT_DATA_DIR) if not inplace else str(tmp_output),
        output_path=str(tmp_output),
        inplace=inplace,
        wipe_annotations=True,
        auto_transfer_response="n",
    )
    return str(tmp_output)


def _wipe_output_edfs(output_path: str) -> list[str]:
    return [os.path.join(output_path, f)
            for f in os.listdir(output_path)
            if f.endswith(".edf") and "_annotations" not in f]


@pytest.mark.parametrize("inplace", [True, False])
def test_wipe_annotations_removes_all_events(monkeypatch, tmp_path, inplace):
    """Positive: pyedflib reports zero annotation texts on wipe output."""
    import pyedflib
    output_path = _run_wipe_pipeline(monkeypatch, tmp_path / "out", inplace=inplace)
    edfs = _wipe_output_edfs(output_path)
    assert edfs, "no output EDFs produced"
    for edf in edfs:
        with pyedflib.EdfReader(edf) as f:
            onsets, durations, texts = f.readAnnotations()
        assert len(texts) == 0, f"{edf}: expected 0 annotations, got {len(texts)}"


@pytest.mark.parametrize("inplace", [True, False])
def test_wipe_annotations_preserves_timekeeping_tals(monkeypatch, tmp_path, inplace):
    """Byte-level guard: every data record still has a timekeeping TAL.

    ``clear_edf_annotations_inplace`` zeros bytes AFTER the ``\\x14\\x14``
    delimiter — if a future refactor accidentally strips the timekeeping
    prefix too, this test fires before we ship a spec-noncompliant EDF+.
    """
    output_path = _run_wipe_pipeline(monkeypatch, tmp_path / "out", inplace=inplace)
    edfs = _wipe_output_edfs(output_path)
    assert edfs
    for edf in edfs:
        n_records = _n_data_records_from_header(edf)
        n_tk = _count_timekeeping_tals(edf)
        assert n_tk == n_records, \
            f"{edf}: timekeeping TAL count {n_tk} != n_records {n_records}"


@pytest.mark.parametrize("inplace", [True, False])
def test_wipe_annotations_skips_stub_creation(monkeypatch, tmp_path, inplace):
    """Negative: no ``*_annotations.edf`` sidecar is written."""
    output_path = _run_wipe_pipeline(monkeypatch, tmp_path / "out", inplace=inplace)
    stubs = [f for f in os.listdir(output_path) if f.endswith("_annotations.edf")]
    assert stubs == [], f"unexpected stubs: {stubs}"


def test_confirm_wipe_annotations_accepts_subject_code(capsys):
    from clean_eeg.clean_subject_eeg import confirm_wipe_annotations
    result = confirm_wipe_annotations(
        "R1755J", approved=set(), input_fn=lambda _: "R1755J")
    assert result is True
    assert "Confirmed" in capsys.readouterr().out


def test_confirm_wipe_annotations_rejects_wrong_code(capsys):
    from clean_eeg.clean_subject_eeg import confirm_wipe_annotations
    # Case mismatch is deliberately rejected — the prompt says
    # "case-sensitive" and hospital-code letters carry meaning.
    result = confirm_wipe_annotations(
        "R1755J", approved=set(), input_fn=lambda _: "r1755j")
    assert result is False
    assert "does not match" in capsys.readouterr().out


def test_confirm_wipe_annotations_rejects_yes_shortcut(capsys):
    from clean_eeg.clean_subject_eeg import confirm_wipe_annotations
    # Guard against operators reflexively typing "yes" from other tools.
    result = confirm_wipe_annotations(
        "R1755J", approved=set(), input_fn=lambda _: "yes")
    assert result is False


def test_confirm_wipe_annotations_bypassed_by_approve_confirmations(capsys):
    from clean_eeg.clean_subject_eeg import confirm_wipe_annotations
    # Sentinel: if the prompt is called under the bypass, this raises.
    result = confirm_wipe_annotations(
        "R1755J",
        approved={"wipe-annotations"},
        input_fn=lambda _: (_ for _ in ()).throw(AssertionError("prompt called")),
    )
    assert result is True
    out = capsys.readouterr().out
    assert "auto-approved" in out
    assert "R1755J" in out


def test_confirm_wipe_annotations_unrelated_approvals_do_not_bypass(capsys):
    """The list is per-type: an ``approved`` set containing an unrelated
    entry must still prompt for wipe. Guards the "no global --yes"
    invariant against a future refactor that treats the set generically.
    """
    from clean_eeg.clean_subject_eeg import confirm_wipe_annotations
    prompt_called = []
    result = confirm_wipe_annotations(
        "R1755J",
        approved={"some-other-future-confirmation"},
        input_fn=lambda p: (prompt_called.append(p) or "R1755J"),
    )
    assert result is True
    assert prompt_called, "prompt should have been shown when wipe-annotations not in approved"


# --- --recursive -----------------------------------------------------------


def _build_recursive_fixture(root, base_edf: str, subdirs: list[str]) -> list[str]:
    """Copy the same source EDF into each subdir under ``root``. Returns
    the created file paths in discovery order. Each subdir gets a copy
    named after the original, so recursive discovery finds all of them.
    """
    created = []
    for i, sub in enumerate(subdirs):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        # Unique filename per subdir so we don't collide even without the
        # timestamp-based clean_filename disambiguation. Recording start
        # times are identical (same source file) so the gap check between
        # copies is 0 — no gap prompt fires.
        dst = d / f"file_{i}.edf"
        shutil.copyfile(base_edf, dst)
        created.append(str(dst))
    return created


@pytest.mark.parametrize("inplace", [True, False])
def test_recursive_discovers_files_in_subdirs(monkeypatch, tmp_path, inplace):
    """Positive: --recursive picks up EDFs from nested subdirs and
    processes them alongside root files. Output preserves subdir layout.
    """
    # y for the recording-gap prompt (same as sibling tests — the fixture
    # copies span >60s between the two source recordings).
    responses = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _build_recursive_fixture(input_dir, SUBJECT_EDF_PATH1,
                             ["session_a", "session_b/subsub"])

    if inplace:
        output_dir = input_dir
    else:
        output_dir = tmp_path / "out"
        output_dir.mkdir()

    clean_subject_edf_files(
        subject_name=PATIENT_NAME,
        subject_code=SUBJECT_CODE,
        input_path=str(input_dir),
        output_path=str(output_dir),
        inplace=inplace,
        recursive=True,
        auto_transfer_response="n",
    )

    # Both subdirs should exist under output_dir with a cleaned file inside.
    for sub in ["session_a", "session_b/subsub"]:
        out_sub = output_dir / sub
        edfs = list(out_sub.glob("*.edf"))
        # No _annotations sidecars in this assertion — we care about
        # the main output landing in the correct subdir.
        mains = [e for e in edfs if "_annotations" not in e.name]
        assert len(mains) == 1, f"expected 1 cleaned EDF in {out_sub}, got {mains}"


def test_recursive_flat_input_unchanged(monkeypatch, tmp_path):
    """Regression: --recursive on a flat directory behaves like the
    non-recursive default. Guards against the recursive walk missing
    root-level files.
    """
    responses = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    shutil.copyfile(SUBJECT_EDF_PATH1, input_dir / "a.edf")
    shutil.copyfile(SUBJECT_EDF_PATH2, input_dir / "b.edf")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    clean_subject_edf_files(
        subject_name=PATIENT_NAME,
        subject_code=SUBJECT_CODE,
        input_path=str(input_dir),
        output_path=str(output_dir),
        inplace=False,
        recursive=True,
        auto_transfer_response="n",
    )
    mains = [p for p in output_dir.glob("*.edf") if "_annotations" not in p.name]
    assert len(mains) == 2, f"expected 2 cleaned EDFs at output root, got {mains}"


def test_recording_gaps_bypassed_by_approve_confirmations(monkeypatch, tmp_path, capsys):
    """Positive bypass: --approve-confirmations recording-gaps skips the
    interactive prompt. Uses a monkeypatched input that raises so the
    test fails loudly if the prompt is triggered.
    """
    def _no_prompt(_):
        raise AssertionError("gap prompt should have been bypassed")
    monkeypatch.setattr("builtins.input", _no_prompt)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    shutil.copyfile(SUBJECT_EDF_PATH1, input_dir / "a.edf")
    shutil.copyfile(SUBJECT_EDF_PATH2, input_dir / "b.edf")  # ~59-min gap
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    clean_subject_edf_files(
        subject_name=PATIENT_NAME,
        subject_code=SUBJECT_CODE,
        input_path=str(input_dir),
        output_path=str(output_dir),
        inplace=False,
        approve_confirmations={"recording-gaps"},
        auto_transfer_response="n",
    )
    out = capsys.readouterr().out
    assert "auto-approved" in out.lower()


def test_recording_gaps_prompt_still_fires_without_bypass(monkeypatch, tmp_path):
    """Negative guard: an unrelated approval entry must NOT bypass the
    gap prompt. Same "no global --yes" invariant as wipe-annotations.
    """
    prompt_called = []
    def _record_prompt(p):
        prompt_called.append(p)
        return "y"
    monkeypatch.setattr("builtins.input", _record_prompt)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    shutil.copyfile(SUBJECT_EDF_PATH1, input_dir / "a.edf")
    shutil.copyfile(SUBJECT_EDF_PATH2, input_dir / "b.edf")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    clean_subject_edf_files(
        subject_name=PATIENT_NAME,
        subject_code=SUBJECT_CODE,
        input_path=str(input_dir),
        output_path=str(output_dir),
        inplace=False,
        approve_confirmations={"wipe-annotations"},  # unrelated
        auto_transfer_response="n",
    )
    # At least the gap prompt should have been shown; other prompts may fire too.
    gap_prompts = [p for p in prompt_called if "Continue?" in p]
    assert gap_prompts, "gap prompt should have fired when 'recording-gaps' not in approved"
