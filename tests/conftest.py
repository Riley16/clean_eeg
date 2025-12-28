import json
import pytest

from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE, TEST_SUBJECT_DATA_DIR, INCONSISTENT_SUBJECT_DATA_DIR


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    TEST_DATA_DIR.mkdir(exist_ok=True)
    TEST_SUBJECT_DATA_DIR.mkdir(exist_ok=True)
    # Only regenerate if missing
    with open(TEST_CONFIG_FILE, 'r') as f:
        test_config = json.load(f)
    edf_config = test_config.get('basic_EDF+C')

    for config_key in test_config.keys():
        edf_config = test_config.get(config_key)
        if edf_config['filename'].startswith('inconsistent_subject_'):
            path = INCONSISTENT_SUBJECT_DATA_DIR / edf_config['filename']
        elif edf_config['filename'].startswith('subject_'):
            path = TEST_SUBJECT_DATA_DIR / edf_config['filename']
        else:
            path = TEST_DATA_DIR / edf_config['filename']
        if not path.exists():
            from .generate_edf import run_generate_test_edf
            print('Generating test EDF data files on first test run...')
            run_generate_test_edf()
            print('Finished generating test EDF data files on first test run.')
            break
