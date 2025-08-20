from clean_eeg.load_eeg import load_edf
from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE

import json
with open(TEST_CONFIG_FILE, 'r') as f:
    TEST_CONFIG = json.load(f)
BASIC_EDF = str(TEST_DATA_DIR / TEST_CONFIG["basic_EDF+C"]['filename'])

def test_load_edf_pyedflib():
    data = load_edf(BASIC_EDF, load_method='pyedflib', preload=True)
    assert isinstance(data, dict)
    assert 'signals' in data
    assert 'header' in data
    assert 'signal_headers' in data
    assert 'annotations' in data


def test_load_edf_edfio():
    data = load_edf(BASIC_EDF, load_method='edfio', preload=True)
    assert hasattr(data, 'signals')
    assert hasattr(data, 'patient')
    assert hasattr(data, 'recording')
    assert hasattr(data, 'annotations')


def test_load_edf_mne():
    import mne
    data = load_edf(BASIC_EDF, load_method='mne', preload=True)
    assert isinstance(data, mne.io.edf.edf.RawEDF)
    assert data.ch_names is not None
    assert data.info is not None


def test_roundtrip_io_pyedflib():
    import os
    import pyedflib
    from clean_eeg.compare_eeg import compare_edf_files
    data = load_edf(BASIC_EDF, load_method='pyedflib', preload=True)
    
    new_edf_file = str(TEST_DATA_DIR / 'test_roundtrip.edf')
    with pyedflib.EdfWriter(new_edf_file, len(data['signals']), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setSignalHeaders(data['signal_headers'])
        f.writeSamples(data['signals'])
        f.setHeader(data['header'])
        for time, duration, text in zip(*data['annotations']):
            f.writeAnnotation(time, duration, text)

    compare_edf_files(BASIC_EDF, new_edf_file)

    # delete roudtrip file
    os.remove(new_edf_file)
