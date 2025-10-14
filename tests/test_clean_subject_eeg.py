import numpy as np

from clean_eeg.clean_subject_eeg import remove_gendered_pronouns, _GENDERED_PRONOUNS, BASE_START_DATE,\
        DEFAULT_REDACT_HEADER_KEYS, REDACT_REPLACEMENT, REDACT_PRONOUN_REPLACEMENT
from clean_eeg.load_eeg import load_edf
from tests.generate_edf import format_edf_config_json
from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE, TEST_SUBJECT_DATA_DIR
from clean_eeg.anonymize import PersonalName, REDACT_NAME_REPLACEMENT

from datetime import datetime, timedelta
import json
with open(TEST_CONFIG_FILE, 'r') as f:
    TEST_CONFIG = json.load(f)
BASIC_EDF_PATH = str(TEST_DATA_DIR / TEST_CONFIG["basic_EDF+C"]['filename'])


def test_remove_gendered_pronouns_basic():
    input = ' asdf '.join(_GENDERED_PRONOUNS)
    output = ' asdf '.join([REDACT_PRONOUN_REPLACEMENT] * len(_GENDERED_PRONOUNS))
    assert remove_gendered_pronouns(input) == output

EDF_CONFIG = TEST_CONFIG["basic_EDF+C"]
EDF_TIMESTAMP_FORMAT = EDF_CONFIG['timestamp_format']
EDF_CONFIG = format_edf_config_json(EDF_CONFIG)
EDF_HEADER = EDF_CONFIG['header']

SUBJECT_CODE = 'R1755A'
PATIENT_NAME = PersonalName(first_name='John',
                            middle_names=['P.'],
                            last_name="O'Connor")

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


def test_clean_subject_edf_files():
    from clean_eeg.clean_subject_eeg import clean_subject_edf_files
    from pathlib import Path
    import os

    output_path = TEST_SUBJECT_DATA_DIR / 'temp_clean_output'
    if not output_path.exists():
        os.makedirs(output_path)
    
    clean_subject_edf_files(subject_name=PATIENT_NAME,
                            subject_code=SUBJECT_CODE,
                            input_path=str(TEST_SUBJECT_DATA_DIR),
                            output_path=str(output_path))
    
    # check that file was created
    filename_no_ext = Path(BASIC_EDF_PATH).stem
    clean_filename = f"{filename_no_ext}_{SUBJECT_CODE}_2023.01.01__00:00:00.edf"
    clean_full_path = os.path.join(output_path, clean_filename)
    assert os.path.exists(clean_full_path)

    # cleanup
    os.remove(clean_full_path)
    os.rmdir(output_path)
