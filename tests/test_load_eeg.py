import clean_eeg.load_eeg as load_eeg
from clean_eeg.load_eeg import load_edf
from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE

import os
import json
import numpy as np
import pyedflib
import pytest
from datetime import datetime

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


# =======================================================================
# mmap-based signal loader (_read_signals_via_mmap / use_mmap=True path)
# =======================================================================
#
# These tests build synthetic EDFs with various geometries and verify that
# the mmap path produces byte-identical int32 arrays to pyedflib's
# readSignal(digital=True). The mmap path is opt-in via
# load_edf(..., preload=True, read_digital=True, use_mmap=True) so default
# users are never affected.


def _write_synthetic_edf_with_signals(path: str,
                                      n_channels: int,
                                      sample_rates: list,
                                      record_duration_s: float,
                                      n_records: int,
                                      n_annotations: int = 1,
                                      seed: int = 0) -> list:
    """Write an EDF+ with the caller-specified geometry.

    Each channel gets ``sample_rates[i] * record_duration_s * n_records``
    deterministic pseudo-random int16 samples. Returns the list of int16
    arrays that were written (for direct comparison by callers).

    setDatarecordDuration pins the record duration so
    samples_per_record per signal is exactly ``rate * record_duration``.
    """
    import warnings as _w
    rng = np.random.default_rng(seed)
    signal_headers = []
    signals = []
    for i, sr in enumerate(sample_rates):
        samples_per_record = int(sr * record_duration_s)
        n_samples = samples_per_record * n_records
        # Generate int16 data directly so we control the digital bytes exactly.
        samples_int16 = rng.integers(-30000, 30000, size=n_samples, dtype=np.int16)
        signals.append(samples_int16)
        signal_headers.append({
            'label': f'CH{i:03d}',
            'dimension': 'uV',
            'sample_frequency': sr,
            'physical_max': 3200.0,
            'physical_min': -3200.0,
            'digital_max': 32767,
            'digital_min': -32768,
            'prefilter': '',
            'transducer': '',
        })

    with pyedflib.EdfWriter(path, n_channels,
                             file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            'technician': 'T', 'recording_additional': '',
            'patientname': 'Test Subject', 'patient_additional': '',
            'patientcode': 'R1TESTS', 'equipment': 'test',
            'admincode': '', 'sex': 'Male',
            'startdate': datetime(2023, 1, 1, 10, 0, 0),
            'birthdate': '01 jan 1970', 'gender': 'Male',
        })
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            f.setDatarecordDuration(record_duration_s)
        f.setSignalHeaders(signal_headers)
        # writeSamples with digital=True expects the same int16 values we
        # stored above — round-trip is exact.
        f.writeSamples([s.astype(np.int32) for s in signals], digital=True)
        for k in range(n_annotations):
            f.writeAnnotation(0.5 + k * 0.1, -1, f"marker_{k}")
    return signals


def _compare_pyedflib_vs_mmap(path: str):
    """Load once via pyedflib's readSignal and once via our mmap helper,
    and assert value-identical output across every channel.

    pyedflib's readSignal(digital=True) returns int32; the mmap helper
    returns int16 (matching the on-disk width, halving RAM). Both are
    integer types so np.array_equal compares values correctly — we
    assert value equality + integer dtype, not dtype equality."""
    pyedflib_data = load_edf(path, load_method='pyedflib',
                             preload=True, read_digital=True,
                             use_mmap=False)
    mmap_data = load_edf(path, load_method='pyedflib',
                         preload=True, read_digital=True,
                         use_mmap=True)

    assert len(pyedflib_data['signals']) == len(mmap_data['signals'])
    for i, (pe, mm) in enumerate(zip(pyedflib_data['signals'],
                                     mmap_data['signals'])):
        assert pe.dtype == np.int32, f"signal {i}: pyedflib dtype {pe.dtype}"
        assert np.issubdtype(mm.dtype, np.integer), \
            f"signal {i}: mmap dtype {mm.dtype} not integer"
        assert pe.shape == mm.shape, \
            f"signal {i}: shapes pe={pe.shape} mm={mm.shape}"
        assert np.array_equal(pe, mm), \
            f"signal {i}: values differ (pyedflib vs mmap)"


@pytest.mark.parametrize(
    "n_channels,sample_rate,record_duration_s,n_records",
    [
        # --- channel-count sweep ---
        (1,    100, 1.0,    10),  # single channel
        (2,    100, 1.0,    10),  # basic multi-channel
        (8,    200, 1.0,     5),  # 8 channels
        (16,   100, 0.5,   100),  # 16 channels, many short records
        (50,   250, 1.0,     3),  # many channels
        (178,  100, 1.0,     2),  # full NK-style channel count
        # --- record-duration sweep (NK short, standard, long) ---
        (4,    500, 0.086,  30),  # NK-style short record duration
        (4,    500, 0.5,    10),  # sub-second record
        (4,    500, 2.0,     4),  # 2s records
        (4,    500, 5.0,     2),  # 5s records
        # --- sample-rate sweep ---
        (4,   1000, 1.0,     3),  # 1 kHz
        (3,   2000, 0.086,  12),  # NK-style 2 kHz + 86 ms record
        # --- record-count sweep ---
        (2,    100, 1.0,     1),  # minimum case: 1 record
        (2,    100, 1.0,  1000),  # long file: 1000 records
    ],
    ids=[
        "1ch_100Hz_1s_10rec",
        "2ch_100Hz_1s_10rec",
        "8ch_200Hz_1s_5rec",
        "16ch_100Hz_500ms_100rec",
        "50ch_250Hz_1s_3rec",
        "178ch_100Hz_1s_2rec_NK_geometry",
        "4ch_500Hz_NK086ms_30rec",
        "4ch_500Hz_500ms_10rec",
        "4ch_500Hz_2s_4rec",
        "4ch_500Hz_5s_2rec",
        "4ch_1kHz_1s_3rec",
        "3ch_2kHz_NK086ms_12rec",
        "2ch_100Hz_1s_1rec_minimum",
        "2ch_100Hz_1s_1000rec_long",
    ],
)
def test_mmap_loader_matches_pyedflib_various_geometries(tmp_path,
                                                         n_channels,
                                                         sample_rate,
                                                         record_duration_s,
                                                         n_records):
    """The mmap path must produce byte-identical int32 signals to
    pyedflib's readSignal(digital=True) across a wide range of EDF
    geometries — channel count, sample rate, record duration, and
    number of records all systematically varied."""
    path = str(tmp_path / f"mmap_{n_channels}_{sample_rate}_{int(record_duration_s*1000)}ms_{n_records}rec.edf")
    _write_synthetic_edf_with_signals(
        path,
        n_channels=n_channels,
        sample_rates=[sample_rate] * n_channels,
        record_duration_s=record_duration_s,
        n_records=n_records,
    )
    _compare_pyedflib_vs_mmap(path)


def test_mmap_loader_heterogeneous_sample_rates(tmp_path):
    """EDF+ allows different sample rates per channel (provided they all
    fit into the same record_duration). The mmap path must handle that
    case — each signal has a different samples_per_record."""
    path = str(tmp_path / "heterogeneous.edf")
    _write_synthetic_edf_with_signals(
        path,
        n_channels=4,
        sample_rates=[100, 200, 500, 1000],  # distinct per channel
        record_duration_s=1.0,
        n_records=5,
    )
    _compare_pyedflib_vs_mmap(path)


def test_mmap_loader_preserves_sample_values_exactly(tmp_path):
    """Ground-truth check: the int32 output of the mmap loader must equal
    the int16 samples we originally wrote (cast up to int32)."""
    path = str(tmp_path / "ground_truth.edf")
    written = _write_synthetic_edf_with_signals(
        path,
        n_channels=3,
        sample_rates=[100, 100, 100],
        record_duration_s=1.0,
        n_records=5,
    )
    data = load_edf(path, load_method='pyedflib',
                    preload=True, read_digital=True, use_mmap=True)
    # data['signals'] has 3 data signals + 1 Annotations signal
    # (pyedflib reports signals_in_file == 3 for EDF+ because it hides
    # the Annotations channel; but our mmap iterates signals_in_file
    # from pyedflib which also returns 3 — so the loader should also
    # produce 3 arrays.)
    assert len(data['signals']) == 3
    for i in range(3):
        assert np.array_equal(data['signals'][i], written[i].astype(np.int32)), \
            f"signal {i} not preserved"


def test_mmap_loader_returns_empty_when_preload_false(tmp_path):
    """use_mmap is ignored when preload=False; signals must be None."""
    path = str(tmp_path / "nosig.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=3,
    )
    data = load_edf(path, preload=False, read_digital=True, use_mmap=True)
    assert data['signals'] is None


def test_mmap_loader_falls_back_to_pyedflib_when_read_digital_false(tmp_path):
    """The mmap path only supports digital reads. When read_digital=False
    the caller still passes use_mmap=True by mistake; behavior should be
    identical to the pyedflib path (same float64 physical output)."""
    path = str(tmp_path / "physical.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=3,
    )
    phys_via_pyedflib = load_edf(path, preload=True, read_digital=False,
                                  use_mmap=False)['signals']
    phys_via_mmap_flag = load_edf(path, preload=True, read_digital=False,
                                   use_mmap=True)['signals']
    assert len(phys_via_pyedflib) == len(phys_via_mmap_flag)
    for a, b in zip(phys_via_pyedflib, phys_via_mmap_flag):
        assert a.dtype == np.float64
        assert b.dtype == np.float64
        np.testing.assert_array_equal(a, b)


def test_mmap_loader_detects_truncated_file(tmp_path):
    """If the data region is shorter than the header claims, the mmap
    helper must raise rather than reading garbage. This guards the
    helper's own defensive check — when driven through ``load_edf``,
    pyedflib's own compliance check rejects the file first (with its
    own ``(Filesize)`` error), which is the expected behaviour, but the
    helper must still refuse if called directly in future code paths."""
    from clean_eeg.load_eeg import _read_signals_via_mmap
    path = str(tmp_path / "short.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=5,
    )
    full_size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.truncate(full_size - 500)

    with pytest.raises(ValueError, match=r"(?i)smaller than expected"):
        _read_signals_via_mmap(path)


def test_load_edf_falls_back_to_pyedflib_when_mmap_helper_raises(tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
    """When the mmap helper throws, load_edf must catch the error, emit a
    warning, and transparently fall back to pyedflib's per-channel
    readSignal loop — so no caller regresses below the baseline
    behaviour for any pathological file."""
    path = str(tmp_path / "fallback.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=3,
    )

    # Force the mmap helper to fail.
    import clean_eeg.load_eeg as _le
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic mmap failure")
    monkeypatch.setattr(_le, "_read_signals_via_mmap", boom)

    data = load_edf(path, preload=True, read_digital=True, use_mmap=True)

    out = capsys.readouterr().out
    assert "mmap signal loader failed" in out
    assert "Falling back" in out

    # Signals must still be present, same shape and dtype as the pyedflib
    # path — i.e. the fallback actually produced a usable result.
    assert data['signals'] is not None
    assert len(data['signals']) == 2
    for sig in data['signals']:
        assert sig.dtype == np.int32
        assert len(sig) == 300  # 100 samples/record * 3 records


def test_mmap_loader_rejects_invalid_samples_per_record(tmp_path):
    """A header whose samples_per_record bytes for some signal contain 0
    must raise a clear error instead of silently producing an empty
    signal. We corrupt the on-disk samples_per_record bytes directly."""
    from clean_eeg import load_eeg as _le
    path = str(tmp_path / "any.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=2,
    )
    # Overwrite signal 1's samples_per_record field with "0" to force the
    # failure. Layout: main(256) + field-by-field signal headers; the
    # samples_per_record region is at offset (256 + 216 * n_signals) within
    # the signal-header area. For n_signals=3 (2 data + 1 annotations):
    #   abs_offset = 256 + 216*3 + 1*8 = 256 + 648 + 8 = 912
    with open(path, "rb") as f:
        main = f.read(256)
    n_signals = int(main[252:256].decode().strip())
    abs_offset = 256 + 216 * n_signals + 1 * 8
    with open(path, "r+b") as f:
        f.seek(abs_offset)
        f.write(b"0       ")  # 8-byte ASCII "0"

    with pytest.raises(ValueError, match=r"non-positive"):
        _le._read_signals_via_mmap(path)


def test_mmap_loader_rejects_unparseable_samples_per_record(tmp_path):
    """samples_per_record byte-field full of ASCII garbage must raise a
    clear error naming the offending signal — not silently return junk."""
    from clean_eeg.load_eeg import _read_signals_via_mmap
    path = str(tmp_path / "garbled_spr.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=2,
    )
    with open(path, "rb") as f:
        main = f.read(256)
    n_signals = int(main[252:256].decode().strip())
    abs_offset = 256 + 216 * n_signals + 0 * 8  # signal 0's spr
    with open(path, "r+b") as f:
        f.seek(abs_offset)
        f.write(b"abcdefgh")  # 8 non-numeric bytes

    with pytest.raises(ValueError, match=r"Unparseable samples_per_record"):
        _read_signals_via_mmap(path)


def test_mmap_loader_tolerates_file_larger_than_expected(tmp_path):
    """File with trailing garbage past the expected data region must still
    load cleanly — pyedflib's ``EDFLIB_FILE_ERRORS_FILESIZE`` only
    rejects under-size (not over-size), and our helper should match."""
    path = str(tmp_path / "trailing.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=3,
    )
    # Append 1 KB of trailing garbage.
    with open(path, "ab") as f:
        f.write(b"\x00" * 1024)

    _compare_pyedflib_vs_mmap(path)


def test_mmap_loader_returns_empty_on_zero_records(tmp_path):
    """A header claiming n_records=0 must produce an empty signals list
    rather than raising — the helper short-circuits the mmap entirely."""
    from clean_eeg.load_eeg import _read_signals_via_mmap
    path = str(tmp_path / "zero_records.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=3,
    )
    # Overwrite n_records (bytes 236..244) with "0".
    with open(path, "r+b") as f:
        f.seek(236)
        f.write(b"0       ")  # left-aligned ASCII "0" in 8 bytes

    signals = _read_signals_via_mmap(path)
    assert signals == []


def test_mmap_loader_returns_empty_on_zero_signals(tmp_path):
    """A header claiming n_signals=0 must short-circuit to an empty list
    rather than trying to parse any signal-header bytes."""
    from clean_eeg.load_eeg import _read_signals_via_mmap
    path = str(tmp_path / "zero_signals.edf")
    _write_synthetic_edf_with_signals(
        path, n_channels=2, sample_rates=[100, 100],
        record_duration_s=1.0, n_records=3,
    )
    # Overwrite n_signals (bytes 252..256) with "   0" (4-char field).
    with open(path, "r+b") as f:
        f.seek(252)
        f.write(b"0   ")

    signals = _read_signals_via_mmap(path)
    assert signals == []
