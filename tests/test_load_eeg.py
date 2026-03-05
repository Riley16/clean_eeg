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


# TODO: pyedflib should open EDF files with partial final records when properly
# generated (NK exports produce these and we have opened them with pyedflib).
# The current test generates a partial record by crude file truncation, which
# causes a filesize mismatch that pyedflib rejects. Update generate_partial_record_edf()
# to produce a properly-formatted partial record (matching real NK export structure)
# and update this test to confirm pyedflib zero-pads the partial record on read.
def test_pyedflib_rejects_partial_final_record():
    """pyedflib rejects our synthetically truncated partial-record EDF due to
    filesize mismatch. See TODO above — a properly-formatted partial record
    (as produced by NK exports) should be accepted and zero-padded."""
    import os
    import pyedflib
    from tests.generate_edf import generate_partial_record_edf

    full_path, partial_path = generate_partial_record_edf(
        output_path=str(TEST_DATA_DIR / 'partial_record_test.edf'),
        file_type=pyedflib.FILETYPE_EDFPLUS,
    )

    try:
        # Verify full reference file opens fine
        f = pyedflib.EdfReader(full_path)
        n_samples_full = len(f.readSignal(0))
        assert n_samples_full > 0
        f.close()

        # Verify partial file is actually truncated
        with open(full_path, 'rb') as fh:
            fh.seek(184)
            header_size = int(fh.read(8).decode('ascii').strip())
        assert os.path.getsize(partial_path) < os.path.getsize(full_path)
        assert (os.path.getsize(partial_path) - header_size) % \
               ((os.path.getsize(full_path) - header_size) //
                pyedflib.EdfReader(full_path).datarecords_in_file) != 0, \
            "Partial file should have incomplete final record"

        # pyedflib should reject the partial-record file
        try:
            pyedflib.EdfReader(partial_path)
            assert False, "pyedflib should have rejected partial-record EDF"
        except OSError as e:
            assert "Filesize" in str(e) or "format errors" in str(e)
    finally:
        os.remove(full_path)
        os.remove(partial_path)


def test_pyedflib_zero_pads_partial_records_on_write():
    """When pyedflib writes an EDF and the signal data doesn't evenly fill the
    last data record, it zero-pads the remainder rather than dropping samples.
    This guarantees that our annotation stub merge (which relies on pyedflib for
    the rewrite path) never loses signal data."""
    import os
    import numpy as np
    import pyedflib

    n_channels = 2
    sample_rate = 100  # samples per second per channel
    # 950 samples = 9.5 seconds; with 1s data records, the last record is half-full
    n_samples = 950

    path = str(TEST_DATA_DIR / 'zero_pad_write_test.edf')
    writer = pyedflib.EdfWriter(path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    for i in range(n_channels):
        writer.setSignalHeader(i, {
            'label': f'CH{i}', 'dimension': 'uV',
            'sample_frequency': sample_rate,
            'physical_max': 3200, 'physical_min': -3200,
            'digital_max': 32767, 'digital_min': -32768,
        })
    # Values must stay within [-3200, 3200] physical range
    original_signals = [np.arange(1, n_samples + 1, dtype=float) + i * 1000
                        for i in range(n_channels)]
    writer.writeSamples(original_signals)
    writer.close()

    try:
        reader = pyedflib.EdfReader(path)
        assert reader.datarecords_in_file == 10, \
            "Should have 10 data records (9 full + 1 zero-padded)"

        for ch in range(n_channels):
            sig = reader.readSignal(ch)

            # All 950 original samples preserved, padded to 1000 (not dropped)
            assert len(sig) == 1000, \
                f"CH{ch}: expected 1000 samples (10 records), got {len(sig)}"

            # First 950 samples match the original data (digital-to-physical
            # conversion introduces small rounding; atol covers the quantization
            # step: (pmax-pmin)/(dmax-dmin) = 6400/65535 ≈ 0.098)
            np.testing.assert_allclose(sig[:n_samples], original_signals[ch],
                                       atol=0.1,
                                       err_msg=f"CH{ch}: original samples not preserved")

            # Last 50 samples are zero-padded (near-zero from digital 0)
            padded = sig[n_samples:]
            assert len(padded) == 50
            assert np.all(np.abs(padded) < 0.5), \
                f"CH{ch}: padded samples should be near-zero, got {padded}"
        reader.close()
    finally:
        os.remove(path)
