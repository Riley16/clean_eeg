from clean_eeg.split_discontinuous_edf import overwrite_edfD_to_edfC
from clean_eeg.load_eeg import is_edfC, is_edfD, is_edf_continuous
from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE

import pytest

import json
with open(TEST_CONFIG_FILE, 'r') as f:
    TEST_CONFIG = json.load(f)
BASIC_EDFC = str(TEST_DATA_DIR / TEST_CONFIG["basic_EDF+C"]['filename'])
BASIC_EDFD = str(TEST_DATA_DIR / TEST_CONFIG["basic_EDF+D"]['filename'])
CONTINUOUS_EDFD_FILE = str(TEST_DATA_DIR / TEST_CONFIG["continuous_EDF+D"]['filename'])


def test_overwrite_edfD_to_edfC():
    import shutil
    temp_file = "tests/test_data/continuous_edfD_converted_to_edfC.edf"
    shutil.copyfile(CONTINUOUS_EDFD_FILE, temp_file)
    
    assert is_edfD(temp_file) and is_edf_continuous(temp_file)
    overwrite_edfD_to_edfC(temp_file, require_continuous_data=True)
    assert is_edfC(temp_file) and is_edf_continuous(temp_file)

    # clean up
    import os
    os.remove(temp_file)


def test_overwrite_edfD_to_edfC_discontinuous_failsafe():
    import shutil
    temp_file = "tests/test_data/edfD_temp.edf"
    shutil.copyfile(BASIC_EDFD, temp_file)
    
    assert is_edfD(temp_file) and not is_edf_continuous(temp_file)
    with pytest.raises(ValueError):
        overwrite_edfD_to_edfC(temp_file, require_continuous_data=True)

    # clean up
    import os
    os.remove(temp_file)
