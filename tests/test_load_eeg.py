import clean_eeg.load_eeg as load_eeg
from clean_eeg.load_eeg import load_edf
from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE

import json
with open(TEST_CONFIG_FILE, 'r') as f:
    TEST_CONFIG = json.load(f)
BASIC_EDFC = str(TEST_DATA_DIR / TEST_CONFIG["basic_EDF+C"]['filename'])
BASIC_EDFD = str(TEST_DATA_DIR / TEST_CONFIG["basic_EDF+D"]['filename'])
CONTINUOUS_EDFD_FILE = str(TEST_DATA_DIR / TEST_CONFIG["continuous_EDF+D"]['filename'])

def test_load_edf_pyedflib():
    data = load_edf(BASIC_EDFC, load_method='pyedflib', preload=True)
    assert isinstance(data, dict)
    assert 'signals' in data
    assert 'header' in data
    assert 'signal_headers' in data
    assert 'annotations' in data


def test_load_edf_edfio():
    data = load_edf(BASIC_EDFC, load_method='edfio', preload=True)
    assert hasattr(data, 'signals')
    assert hasattr(data, 'patient')
    assert hasattr(data, 'recording')
    assert hasattr(data, 'annotations')


def test_load_edf_mne():
    import mne
    data = load_edf(BASIC_EDFC, load_method='mne', preload=True)
    assert isinstance(data, mne.io.edf.edf.RawEDF)
    assert data.ch_names is not None
    assert data.info is not None


def test_load_edf_lunapi():
    data = load_edf(BASIC_EDFC, load_method='lunapi', preload=True)
    import lunapi as lp
    assert isinstance(data, lp.inst)


def test_is_edf_format():
    assert load_eeg.is_edf_plus(BASIC_EDFC)
    assert load_eeg.is_edf_plus(CONTINUOUS_EDFD_FILE)
    assert load_eeg.is_edfC(BASIC_EDFC)
    assert not load_eeg.is_edfC(CONTINUOUS_EDFD_FILE)
    assert not load_eeg.is_edfD(BASIC_EDFC)
    assert load_eeg.is_edfD(CONTINUOUS_EDFD_FILE)
    assert load_eeg.is_edf_continuous(BASIC_EDFC)
    assert load_eeg.is_edf_continuous(CONTINUOUS_EDFD_FILE)
    assert not load_eeg.is_edf_continuous(BASIC_EDFD)


def test_load_edf_discontinuous_lunapi():
    data = load_edf(CONTINUOUS_EDFD_FILE, load_method='lunapi', preload=True)
    import lunapi as lp
    assert isinstance(data, lp.inst)
    print(data)


def test_roundtrip_io_pyedflib():
    import os
    import pyedflib
    from clean_eeg.compare_eeg import compare_edf_files
    data = load_edf(BASIC_EDFC, load_method='pyedflib', preload=True)
    
    new_edf_file = str(TEST_DATA_DIR / 'test_roundtrip.edf')
    with pyedflib.EdfWriter(new_edf_file, len(data['signals']), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setSignalHeaders(data['signal_headers'])
        f.writeSamples(data['signals'])
        f.setHeader(data['header'])
        for time, duration, text in zip(*data['annotations']):
            f.writeAnnotation(time, duration, text)

    compare_edf_files(BASIC_EDFC, new_edf_file)

    # delete roudtrip file
    os.remove(new_edf_file)
