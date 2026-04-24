import shutil
import numpy as np

import pyedflib
import pytest

from clean_eeg.modify_edf_inplace import (
    update_edf_header_inplace,
    clear_edf_annotations_inplace,
    create_annotations_only_edf,
    merge_annotation_stub_edf,
    validate_header_roundtrip,
)
from clean_eeg.paths import TEST_DATA_DIR
from .generate_edf import format_edf_config_json, DEFAULT_NUMBER_SIGNALS


@pytest.fixture
def base_edf(tmp_path):
    """Copy an existing test EDF file to a temporary location for testing."""
    source_path = TEST_DATA_DIR / "basic_EDF_C.edf"
    test_path = tmp_path / "test.edf"
    shutil.copyfile(str(source_path), str(test_path))
    return str(test_path)

ORIGINAL_MODIFIED_EDF_PATH = str(TEST_DATA_DIR / "basic_EDF_C_modified.edf")

# ======================
# Test 1: no-op rewrite
# ======================

def test_inplace_noop_header_and_annotations(base_edf, tmp_path):
    """
    Copy an EDF, rewrite header and annotations with *identical* content,
    and verify the files are byte-for-byte identical.
    """
    orig_path = base_edf
    copy_path = str(tmp_path / "copy_noop.edf")
    shutil.copyfile(orig_path, copy_path)

    # Use original header strings verbatim so our formatting
    # is a strict no-op.
    with pyedflib.EdfReader(orig_path) as f:
        orig_hdr = f.getHeader()
        orig_signal_headers = [f.getSignalHeader(i) for i in range(f.signals_in_file)]

    # Apply in-place header rewrite (no-op content-wise)
    update_edf_header_inplace(copy_path,
                              orig_hdr,
                              signal_header_updates=orig_signal_headers)


    with open(orig_path, "rb") as f1, open(copy_path, "rb") as f2:
        orig_bytes = f1.read()
        copy_bytes = f2.read()

    assert orig_bytes == copy_bytes, "No-op in-place update should not change file bytes"


def load_edf_test_config(config_key):
    import json
    from clean_eeg.paths import TEST_CONFIG_FILE
    with open(TEST_CONFIG_FILE, 'r') as f:
        test_config = json.load(f)
    edf_config = test_config[config_key]
    return edf_config


@pytest.fixture
def header_updates():
    """Test fixture providing input-output mappings for field modifications."""
    modified_edf_config = load_edf_test_config("basic_EDF+C_modified")
    modified_edf_config = format_edf_config_json(modified_edf_config)
    return modified_edf_config['header']

@pytest.fixture
def signal_header_updates():
    modified_edf_config = load_edf_test_config("basic_EDF+C_modified")
    modified_edf_config = format_edf_config_json(modified_edf_config)
    signal_header = modified_edf_config['signal_headers']
    signal_header_updates = list()
    for i in range(DEFAULT_NUMBER_SIGNALS):
        updated_signal_header = dict(signal_header)
        updated_signal_header['label'] += f'_{i + 1}'
        signal_header_updates.append(updated_signal_header)
    return signal_header_updates


def _update_annotation_text(text: str) -> str:
    return text.strip() + "_update"


# ======================
# Helper: rewrite EDF from scratch with pyedflib
# ======================

def rewrite_edf_with_updates(orig_path: str, new_path: str, header_updates: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Read an EDF, apply simple transformations to header string fields
    and annotation texts, and write a new EDF from scratch using pyedflib.

    This is the "gold standard" we compare our in-place updates against.
    """
    r = pyedflib.EdfReader(orig_path)
    try:
        n_signals = r.signals_in_file
        sig_headers = [r.getSignalHeader(i) for i in range(n_signals)]
        main_header = r.getHeader()
        signals = [r.readSignal(i) for i in range(n_signals)]
        ann_onsets, ann_durations, ann_texts = r.readAnnotations()
    finally:
        r.close()

    updated_header = dict(main_header)
    for field, value in header_updates.items():
        if value is not None:
            updated_header[field] = value

    updated_ann_texts = [_update_annotation_text(t) for t in ann_texts]

    w = pyedflib.EdfWriter(
        new_path,
        n_signals,
        file_type=pyedflib.FILETYPE_EDFPLUS,
    )
    try:
        w.setHeader(updated_header)
        w.setSignalHeaders(sig_headers)
        w.writeSamples(signals)
        for onset, dur, txt in zip(ann_onsets, ann_durations, updated_ann_texts):
            w.writeAnnotation(onset, dur, txt)
    finally:
        w.close()
    
    return ann_onsets, ann_durations, updated_ann_texts


# ======================
# Test 2: real updates
# ======================

def test_inplace_vs_rewrite_semantic_equality(base_edf, tmp_path, header_updates, signal_header_updates):
    """
    Copy an EDF, update all main-header string fields and all annotation texts
    in two ways:

      1) In-place, using update_edf_header_inplace + update_edf_annotations_inplace
      2) By rewriting a new EDF from scratch with pyedflib

    Then compare:
      - Main-header dicts
      - Annotations (onset, duration, text)
      - Signals

    for semantic equality.
    """
    orig = base_edf
    inplace_path = str(tmp_path / "inplace.edf")
    annotation_edf_path = str(tmp_path / "annotations.edf")
    rewrite_path = str(tmp_path / "rewrite.edf")

    shutil.copyfile(orig, inplace_path)

    # ---- Rewrite-with-updates path ----
    updated_annotations = rewrite_edf_with_updates(orig, rewrite_path, header_updates)

    update_edf_header_inplace(inplace_path,
                              header_updates,
                              signal_header_updates=signal_header_updates,
                              confirm_signals_unchanged=True)
    
    create_annotations_only_edf(annotation_edf_path,
                                header_updates,
                                updated_annotations)
    clear_edf_annotations_inplace(inplace_path)

    # ---- Compare semantic equality (header, signals, annotations) ----
    rin = pyedflib.EdfReader(inplace_path)
    r_annotation = pyedflib.EdfReader(annotation_edf_path)
    rre = pyedflib.EdfReader(rewrite_path)
    rmodified = pyedflib.EdfReader(ORIGINAL_MODIFIED_EDF_PATH)
    try:
        # Compare number of signals and records
        assert rin.signals_in_file == rre.signals_in_file
        assert rin.datarecords_in_file == rre.datarecords_in_file

        # Compare main header relevant string fields
        hin = rin.getHeader()
        hre = rre.getHeader()
        for key in hin:
            assert hin.get(key) == hre.get(key)

        # compare signal headers
        for i in range(rin.signals_in_file):
            sh_in = rin.getSignalHeader(i)
            sh_modified = rmodified.getSignalHeader(i)
            for key in sh_in:
                assert sh_in.get(key) == sh_modified.get(key)

        # Compare signals
        # for an unknown reason the rewritten EDF has slightly different signal values,
        # so we instead compare against modified EDF file generated directly from config
        for i in range(rin.signals_in_file):
            sig_in = rin.readSignal(i)
            sig_modified = rmodified.readSignal(i)
            assert all(sig_in == sig_modified)

        # Compare annotations (onset, duration, text)
        o_ann, d_ann, t_ann = r_annotation.readAnnotations()
        o_re, d_re, t_re = rre.readAnnotations()

        o_ann = list(o_ann)
        d_ann = list(d_ann)
        t_ann = list(t_ann)
        o_re = list(o_re)
        d_re = list(d_re)
        t_re = list(t_re)

        assert o_ann == o_re
        assert d_ann == d_re
        assert t_ann == t_re

    finally:
        rin.close()
        rre.close()
        r_annotation.close()
    # confirm signals of inplace and original (not rewritten) EDF are identical
    rin_orig = pyedflib.EdfReader(orig)
    rin_inplace = pyedflib.EdfReader(inplace_path)
    try:
        for i in range(rin_orig.signals_in_file):
            sig_orig = rin_orig.readSignal(i)
            sig_inplace = rin_inplace.readSignal(i)
            assert all(sig_orig == sig_inplace)
    finally:
        rin_orig.close()
        rin_inplace.close()


# ======================
# Test 3: clear annotations standalone
# ======================

def test_clear_annotations_inplace(base_edf):
    """Verify clear_edf_annotations_inplace removes all annotation texts
    while preserving signals and header."""
    # Read original data
    with pyedflib.EdfReader(base_edf) as f:
        orig_header = f.getHeader()
        orig_signals = [f.readSignal(i) for i in range(f.signals_in_file)]
        _, _, orig_texts = f.readAnnotations()

    # Confirm annotations exist before clearing
    assert len(orig_texts) > 0, "Test EDF should have annotations to clear"

    clear_edf_annotations_inplace(base_edf)

    # Verify annotations are gone, signals and header preserved
    with pyedflib.EdfReader(base_edf) as f:
        post_header = f.getHeader()
        post_signals = [f.readSignal(i) for i in range(f.signals_in_file)]
        _, _, post_texts = f.readAnnotations()

    assert len(post_texts) == 0, "Annotations should be empty after clearing"
    for key in orig_header:
        assert orig_header[key] == post_header[key], f"Header field '{key}' changed after clearing annotations"
    for i, (orig_sig, post_sig) in enumerate(zip(orig_signals, post_signals)):
        assert np.array_equal(orig_sig, post_sig), f"Signal {i} changed after clearing annotations"


# ======================
# Test 4: create annotations-only EDF standalone
# ======================

def test_create_annotations_only_edf(base_edf, tmp_path):
    """Verify create_annotations_only_edf produces a readable EDF with correct annotations."""
    with pyedflib.EdfReader(base_edf) as f:
        header = f.getHeader()
        annotations = f.readAnnotations()

    stub_path = str(tmp_path / "annotations_only.edf")
    create_annotations_only_edf(stub_path, header, annotations)

    with pyedflib.EdfReader(stub_path) as f:
        stub_annotations = f.readAnnotations()
        assert f.signals_in_file == 0, "Annotations-only EDF should have no signals"

    assert np.array_equal(stub_annotations[0], annotations[0]), "Annotation onsets mismatch"
    assert np.array_equal(stub_annotations[1], annotations[1]), "Annotation durations mismatch"
    assert np.array_equal(stub_annotations[2], annotations[2]), "Annotation texts mismatch"


# ======================
# Test 5: header-only update (no signal header changes)
# ======================

def test_inplace_header_only_update(base_edf):
    """Verify updating only main header (no signal_header_updates) preserves everything else."""
    with pyedflib.EdfReader(base_edf) as f:
        orig_signal_headers = [f.getSignalHeader(i) for i in range(f.signals_in_file)]
        orig_signals = [f.readSignal(i) for i in range(f.signals_in_file)]

    update_edf_header_inplace(base_edf,
                              header_updates={'technician': 'UpdatedTech'},
                              signal_header_updates=None)

    with pyedflib.EdfReader(base_edf) as f:
        new_header = f.getHeader()
        new_signal_headers = [f.getSignalHeader(i) for i in range(f.signals_in_file)]
        new_signals = [f.readSignal(i) for i in range(f.signals_in_file)]

    assert new_header['technician'] == 'UpdatedTech'
    for i, (orig_sh, new_sh) in enumerate(zip(orig_signal_headers, new_signal_headers)):
        for key in orig_sh:
            assert orig_sh[key] == new_sh[key], f"Signal header {i} field '{key}' changed"
    for i, (orig_sig, new_sig) in enumerate(zip(orig_signals, new_signals)):
        assert np.array_equal(orig_sig, new_sig), f"Signal {i} changed after header-only update"


# ======================
# Test 6: file size unchanged after in-place operations
# ======================

def test_inplace_preserves_file_size(base_edf, header_updates, signal_header_updates):
    """In-place header update must not change file size."""
    import os
    orig_size = os.path.getsize(base_edf)

    update_edf_header_inplace(base_edf,
                              header_updates,
                              signal_header_updates=signal_header_updates)

    assert os.path.getsize(base_edf) == orig_size, "File size changed after in-place update"

    clear_edf_annotations_inplace(base_edf)

    assert os.path.getsize(base_edf) == orig_size, "File size changed after clearing annotations"


# ======================
# Test 7: validate_header_roundtrip detects truncation
# ======================

def test_validate_header_roundtrip_no_warnings(base_edf):
    """Normal-length header fields should produce no warnings."""
    with pyedflib.EdfReader(base_edf) as f:
        header = f.getHeader()
        signal_headers = [f.getSignalHeader(i) for i in range(f.signals_in_file)]
    result = validate_header_roundtrip(header, signal_headers)
    assert result == [], f"Unexpected warnings: {result}"


def test_validate_header_roundtrip_truncation_warning(base_edf):
    """Overly long header field should produce a truncation warning."""
    with pyedflib.EdfReader(base_edf) as f:
        header = f.getHeader()
    # technician is packed into recording_id (80 bytes total with equipment, admincode, etc.)
    header['technician'] = 'A' * 200
    result = validate_header_roundtrip(header)
    assert len(result) > 0, "Expected truncation warning for oversized field"
    assert any('80 chars' in w for w in result)


# ======================
# Test 8: merge annotation stub roundtrip
# ======================

def test_merge_annotation_stub_roundtrip(base_edf, tmp_path):
    """Clear annotations, create stub, merge back — annotations should match original."""
    # Read original annotations and signals
    with pyedflib.EdfReader(base_edf) as f:
        orig_annotations = f.readAnnotations()
        orig_signals = [f.readSignal(i) for i in range(f.signals_in_file)]
        orig_header = f.getHeader()
    orig_onsets, orig_durations, orig_texts = orig_annotations

    assert len(orig_texts) > 0, "Test EDF should have annotations"

    # Create annotation stub from original annotations
    stub_path = str(tmp_path / "annotations.edf")
    create_annotations_only_edf(stub_path, orig_header, orig_annotations)

    # Clear annotations from data EDF
    clear_edf_annotations_inplace(base_edf)
    with pyedflib.EdfReader(base_edf) as f:
        _, _, cleared_texts = f.readAnnotations()
    assert len(cleared_texts) == 0, "Annotations should be cleared"

    # Merge stub back into data EDF
    merge_annotation_stub_edf(base_edf, stub_path)

    # Verify annotations match original
    with pyedflib.EdfReader(base_edf) as f:
        merged_onsets, merged_durations, merged_texts = f.readAnnotations()
        merged_signals = [f.readSignal(i) for i in range(f.signals_in_file)]

    assert list(merged_texts) == list(orig_texts), (
        f"Texts mismatch: {list(merged_texts)} != {list(orig_texts)}")
    assert np.allclose(merged_onsets, orig_onsets), "Onsets mismatch after merge"
    assert np.allclose(merged_durations, orig_durations), "Durations mismatch after merge"

    # Verify signals unchanged
    for i, (orig_sig, merged_sig) in enumerate(zip(orig_signals, merged_signals)):
        assert np.array_equal(orig_sig, merged_sig), f"Signal {i} changed after merge"


# ======================
# Test 9: merge preserves file size
# ======================

def test_merge_annotation_stub_preserves_file_size(base_edf, tmp_path):
    """Merging annotations back should not change file size."""
    import os as _os
    orig_size = _os.path.getsize(base_edf)

    with pyedflib.EdfReader(base_edf) as f:
        header = f.getHeader()
        annotations = f.readAnnotations()

    stub_path = str(tmp_path / "annotations.edf")
    create_annotations_only_edf(stub_path, header, annotations)
    clear_edf_annotations_inplace(base_edf)
    merge_annotation_stub_edf(base_edf, stub_path)

    assert _os.path.getsize(base_edf) == orig_size, "File size changed after merge"


# ======================
# Test 10: atomic merge — original untouched on failure
# ======================

def test_merge_annotation_stub_atomic_on_failure(base_edf, tmp_path):
    """If the integrity check fails, the original data EDF should be
    untouched and the temp file preserved for debugging."""
    import os as _os
    from unittest.mock import patch

    with pyedflib.EdfReader(base_edf) as f:
        header = f.getHeader()
        annotations = f.readAnnotations()

    # Create a valid stub, then clear the data EDF
    stub_path = str(tmp_path / "annotations.edf")
    create_annotations_only_edf(stub_path, header, annotations)
    clear_edf_annotations_inplace(base_edf)

    with open(base_edf, "rb") as f:
        original_bytes = f.read()

    # Patch _verify_merge_integrity to simulate a post-write integrity failure
    with patch("clean_eeg.modify_edf_inplace._verify_merge_integrity",
               side_effect=ValueError("fake integrity failure")):
        temp_path = base_edf + ".merge_tmp"
        with pytest.raises(ValueError, match="temp file preserved"):
            merge_annotation_stub_edf(base_edf, stub_path)

    # Original should be unchanged
    with open(base_edf, "rb") as f:
        after_bytes = f.read()
    assert original_bytes == after_bytes, "Data EDF was modified despite merge failure"

    # Temp file should be preserved for debugging
    assert _os.path.exists(temp_path), "Temp file should be preserved on failure"


# ======================
# Test 11: merge with empty stub (no annotations)
# ======================

def test_merge_empty_stub_is_noop(base_edf, tmp_path):
    """Merging a stub with no annotations should leave the data EDF unchanged."""
    # Create a stub with annotations, then clear them — produces a valid EDF
    # that pyedflib can read but which has 0 user annotations.
    with pyedflib.EdfReader(base_edf) as f:
        header = f.getHeader()
        annotations = f.readAnnotations()
    stub_path = str(tmp_path / "empty_stub.edf")
    create_annotations_only_edf(stub_path, header, annotations)
    clear_edf_annotations_inplace(stub_path, validate=False)

    clear_edf_annotations_inplace(base_edf)
    with open(base_edf, "rb") as f:
        before_bytes = f.read()

    merge_annotation_stub_edf(base_edf, stub_path)

    with open(base_edf, "rb") as f:
        after_bytes = f.read()
    assert before_bytes == after_bytes, "Data EDF changed after merging empty stub"


# =================================================================
# Regressions for the NK truncated-file / non-default record-duration
# inplace-header-update path (see: "filesize / contains format errors"
# bugs hit on Nihon Kohden exports with record_duration = 0.086 and an
# atypically-sized EDF Annotations channel).
# =================================================================

import numpy as _np
import pyedflib as _pyedflib


def _write_synthetic_edf(path: str,
                         *,
                         n_data_channels: int,
                         sample_frequency: int,
                         record_duration_s: float,
                         duration_s: float,
                         n_annotations: int = 4) -> None:
    """Write an EDF+ with the caller-specified record_duration_s.

    setDatarecordDuration disables pyedflib's auto-calc so
    samples_per_record = sample_frequency * record_duration_s exactly.
    """
    signal_headers = []
    for i in range(n_data_channels):
        signal_headers.append({
            'label': f'CH{i:02d}',
            'dimension': 'uV',
            'sample_frequency': sample_frequency,
            'physical_max': 3200.0,
            'physical_min': -3200.0,
            'digital_max': 32767,
            'digital_min': -32768,
            'prefilter': '',
            'transducer': '',
        })

    n_samples = int(sample_frequency * duration_s)
    t = _np.arange(n_samples, dtype=_np.float32) / sample_frequency
    signals = [
        (1000.0 * _np.sin(2 * _np.pi * ((i % 5) + 1) * t)).astype(_np.float64)
        for i in range(n_data_channels)
    ]

    import warnings
    with _pyedflib.EdfWriter(path, n_data_channels,
                              file_type=_pyedflib.FILETYPE_EDFPLUS) as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.setDatarecordDuration(record_duration_s)
        f.setSignalHeaders(signal_headers)
        # Inline header so the writer doesn't look up undefined fields.
        from datetime import datetime
        f.setHeader({
            'technician': 'T', 'recording_additional': '',
            'patientname': 'Synthetic Subject', 'patient_additional': '',
            'patientcode': 'R1SYNTH', 'equipment': 'test',
            'admincode': '', 'sex': 'Male',
            'startdate': datetime(2023, 1, 1, 10, 0, 0),
            'birthdate': '01 jan 1970', 'gender': 'Male',
        })
        f.writeSamples(signals)
        for i in range(n_annotations):
            f.writeAnnotation(i * (duration_s / max(n_annotations, 1)), -1,
                              f"marker {i}")


def _read_on_disk_numeric_signal_fields(path: str) -> dict:
    """Return the raw bytes of every numeric signal-header field
    (phys_min/max, dig_min/max, samples_per_record) from the file."""
    from clean_eeg.modify_edf_inplace import (
        TOTAL_HEADER_BYTES,
        EDF_SIGNAL_HEADER_FIELD_OFFSETS_LENGTHS as FIELD_TABLE,
    )
    with open(path, "rb") as f:
        main = f.read(TOTAL_HEADER_BYTES)
        n_signals = int(main[252:256].decode().strip())
        f.seek(TOTAL_HEADER_BYTES)
        block = f.read(n_signals * 256)

    out = {}
    for field in ('physical_min', 'physical_max',
                  'digital_min', 'digital_max',
                  'num_samples'):
        field_offset, field_width, _ = FIELD_TABLE[field]
        start = field_offset * n_signals
        end = start + field_width * n_signals
        out[field] = block[start:end]
    return out


@pytest.mark.parametrize("record_duration_s", [0.086, 0.5, 2.0])
def test_update_inplace_preserves_record_duration(tmp_path, record_duration_s):
    """After update_edf_header_inplace the file's data_record_duration must
    survive unchanged. Exposes the bug where pyedflib's writer auto-derived
    a (typically 1.0 s) record_duration, producing wrong samples_per_record
    in the signal headers the copy-back step then wrote into the orig."""
    path = str(tmp_path / f"rec_{record_duration_s}.edf")
    _write_synthetic_edf(path,
                         n_data_channels=4,
                         sample_frequency=500,
                         record_duration_s=record_duration_s,
                         duration_s=10.0)

    with _pyedflib.EdfReader(path) as f:
        orig_header = f.getHeader()
        orig_signal_headers = [f.getSignalHeader(i) for i in range(f.signals_in_file)]
        expected_record_duration = f.datarecord_duration
        expected_samples_per_record = [
            sh['sample_frequency'] * expected_record_duration
            for sh in orig_signal_headers
        ]

    update_edf_header_inplace(path,
                              header_updates=orig_header,
                              signal_header_updates=orig_signal_headers)

    # File must still be readable and structural fields must be untouched.
    with _pyedflib.EdfReader(path) as f:
        assert f.datarecord_duration == pytest.approx(expected_record_duration), (
            f"record_duration changed: {f.datarecord_duration} "
            f"vs {expected_record_duration}"
        )
        for i in range(f.signals_in_file):
            sh = f.getSignalHeader(i)
            got = sh['sample_frequency'] * f.datarecord_duration
            assert got == pytest.approx(expected_samples_per_record[i]), (
                f"signal {i} samples_per_record implied by sample_frequency * "
                f"record_duration changed: {got} vs {expected_samples_per_record[i]}"
            )


def test_update_inplace_preserves_numeric_signal_header_bytes(tmp_path):
    """Numeric signal-header fields (phys_min/max, dig_min/max, samples_per_record)
    must be byte-identical after a no-op inplace update — they describe the
    on-disk data layout and can't be silently rewritten by pyedflib's
    trailing-zero formatting or empty-temp heuristics.

    Regression for the NK case where the 'EDF Annotations' signal's
    samples_per_record got overwritten from 172 to 57 because pyedflib's
    temp writer had no annotations to size against."""
    path = str(tmp_path / "numeric_fields.edf")
    # Use an NK-style record_duration so pyedflib's auto-calc would diverge.
    _write_synthetic_edf(path,
                         n_data_channels=6,
                         sample_frequency=500,
                         record_duration_s=0.086,
                         duration_s=5.0,
                         # Many annotations so the annotations-channel size is
                         # larger than what an empty temp would allocate.
                         n_annotations=40)

    before = _read_on_disk_numeric_signal_fields(path)

    with _pyedflib.EdfReader(path) as f:
        hdr = f.getHeader()
        sigs = [f.getSignalHeader(i) for i in range(f.signals_in_file)]
    update_edf_header_inplace(path, header_updates=hdr, signal_header_updates=sigs)

    after = _read_on_disk_numeric_signal_fields(path)
    for field in before:
        assert after[field] == before[field], (
            f"Numeric signal-header field {field!r} changed bytes during "
            f"update_edf_header_inplace; this breaks on-disk record layout."
        )

    # File must remain pyedflib-openable.
    with _pyedflib.EdfReader(path) as f:
        assert f.signals_in_file == 6
        _ = f.readSignal(0)
