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