import shutil

import pyedflib
import pytest

from clean_eeg.modify_edf_inplace import (
    update_edf_annotations_inplace,
    update_edf_header_inplace,
    read_header_raw_bytes,
    # redact_edf_annotations_inplace,
    # _EDF_HEADER_FIELDS,  # used for length checks
)
from clean_eeg.paths import TEST_DATA_DIR
from .generate_edf import format_edf_config_json


@pytest.fixture
def base_edf(tmp_path):
    """Copy an existing test EDF file to a temporary location for testing."""
    source_path = TEST_DATA_DIR / "basic_EDF_C.edf"
    test_path = tmp_path / "test.edf"
    shutil.copyfile(str(source_path), str(test_path))
    return str(test_path)


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

    # Apply in-place header rewrite (no-op content-wise)
    update_edf_header_inplace(copy_path, orig_hdr)

    # Apply in-place annotation redaction with identity function
    # redact_edf_annotations_inplace(copy_path, redact_fn=lambda s: s, verbose=False)

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
def field_mappings():
    """Test fixture providing input-output mappings for field modifications."""
    modified_edf_config = load_edf_test_config("basic_EDF+C_modified")
    modified_edf_config = format_edf_config_json(modified_edf_config)
    return modified_edf_config['header']


def _update_annotation_text(text: str) -> str:
    return text.strip() + "_update"


# ======================
# Helper: rewrite EDF from scratch with pyedflib
# ======================

def rewrite_edf_with_updates(orig_path: str, new_path: str, field_mappings: dict) -> None:
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
    for field, value in field_mappings.items():
        if value is not None:
            updated_header[field] = value

    updated_ann_texts = ann_texts
    # updated_ann_texts = [_update_annotation_text(t) for t in ann_texts]

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


# ======================
# Test 2: real updates
# ======================

def test_inplace_vs_rewrite_semantic_equality(base_edf, tmp_path, field_mappings):
    """
    Copy an EDF, update all main-header string fields and all annotation texts
    in two ways:

      1) In-place, using update_edf_main_header_inplace + redact_edf_annotations_inplace
      2) By rewriting a new EDF from scratch with pyedflib

    Then compare:
      - Main-header dicts
      - Annotations (onset, duration, text)
      - Signals

    for semantic equality.
    """
    orig = base_edf
    inplace_path = str(tmp_path / "inplace.edf")
    rewrite_path = str(tmp_path / "rewrite.edf")

    shutil.copyfile(orig, inplace_path)

    # print('Original header in EDF config:')
    # print(load_edf_test_config("basic_EDF+C")['pyedflib_header'])

    # ---- Rewrite-with-updates path ----
    rewrite_edf_with_updates(orig, rewrite_path, field_mappings)

    # Compare signals just before inplace update
    # TODO: fix differences in rewritten signals
    # rin = pyedflib.EdfReader(inplace_path)
    # rre = pyedflib.EdfReader(rewrite_path)
    # for i in range(rin.signals_in_file):
    #     sig_in = rin.readSignal(i)
    #     sig_re = rre.readSignal(i)
    #     print(f"Comparing signal {i} before inplace update")
    #     # assert all(sig_in == sig_re)
    #     if not all(sig_in == sig_re):
    #         print(f"Signal {i} differs before inplace update")

    update_edf_header_inplace(inplace_path,
                              field_mappings,
                              confirm_signals_unchanged=True)

    # update_edf_annotations_inplace(inplace_path,
    #                                updated_annotations=(1, 2, 3),
    #                                verbosity=0)

    # ---- Compare semantic equality (header, signals, annotations) ----
    rin = pyedflib.EdfReader(inplace_path)
    rre = pyedflib.EdfReader(rewrite_path)
    try:
        # Compare number of signals and records
        assert rin.signals_in_file == rre.signals_in_file
        assert rin.datarecords_in_file == rre.datarecords_in_file

        # Compare main header relevant string fields
        hin = rin.getHeader()
        hre = rre.getHeader()
        for key in hin:
            assert hin.get(key) == hre.get(key)

        # Compare signals
        # TODO: fix rewritten signals, which differ from original 
        # for i in range(rin.signals_in_file):
        #     sig_in = rin.readSignal(i)
        #     sig_re = rre.readSignal(i)
        #     assert all(sig_in == sig_re)

        # Compare annotations (onset, duration, text)
        o_in, d_in, t_in = rin.readAnnotations()
        o_re, d_re, t_re = rre.readAnnotations()

        o_in = list(o_in)
        d_in = list(d_in)
        t_in = list(t_in)
        o_re = list(o_re)
        d_re = list(d_re)
        t_re = list(t_re)

        assert o_in == o_re
        assert d_in == d_re
        assert t_in == t_re

    finally:
        rin.close()
        rre.close()

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
