"""Tests for scripts/count_annotations.py.

Small ops tool -- coverage is minimal but proves:
    1. The counter reads real EDF annotations correctly
    2. Word-tokenization is whitespace-based (matches WPM assumption)
    3. Sidecar '_annotations.edf' files are NOT double-counted
    4. Files pyedflib refuses are surfaced as skipped, not silently dropped
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib


_SCRIPT = (Path(__file__).parent.parent
           / "scripts" / "count_annotations.py")
_spec = importlib.util.spec_from_file_location("count_annotations", _SCRIPT)
assert _spec is not None and _spec.loader is not None
count_annotations = importlib.util.module_from_spec(_spec)
sys.modules["count_annotations"] = count_annotations
_spec.loader.exec_module(count_annotations)


def _write_edf_with_annotations(path: Path, annotation_texts: list[str]
                                 ) -> None:
    """Write a minimal EDF+ with the given annotation texts (one per
    annotation, onset spaced 1 s apart)."""
    n_ch = 2
    sr = 100
    dur = max(2, len(annotation_texts) + 1)
    signal_headers = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sr,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_ch)
    ]
    t = np.arange(0, dur, 1.0 / sr, dtype=np.float32)
    signals = [(1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
               for i in range(n_ch)]
    with pyedflib.EdfWriter(str(path), n_ch,
                             file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            "technician": "T", "recording_additional": "",
            "patientname": "X", "patient_additional": "",
            "patientcode": "R1TEST", "equipment": "X", "admincode": "",
            "sex": "X",
            "startdate": datetime(2023, 1, 1, 10, 0, 0),
            "birthdate": "01 jan 1970", "gender": "X",
        })
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)
        for i, text in enumerate(annotation_texts):
            f.writeAnnotation(float(i + 0.5), -1, text)


def test_count_edf_annotations_counts_words_whitespace_tokenized(tmp_path):
    """Word count is whitespace-tokenized -- matches the assumption
    behind the WPM reading-time estimate. 'PAT REF EEG' -> 3 words."""
    edf = tmp_path / "ann.edf"
    _write_edf_with_annotations(edf, [
        "PAT REF EEG",       # 3 words
        "Seizure",           # 1 word
        "eyes closed",       # 2 words
        "",                  # empty -- excluded from BOTH counts
    ])
    n_ann, n_words = count_annotations.count_edf_annotations(edf)
    assert n_ann == 3
    assert n_words == 6


def test_scan_parent_reports_per_subject_totals(tmp_path):
    """Per-subject stats bucketed by subject folder name; the CSV-style
    report uses these bins for its table."""
    for code, texts in [
        ("R1A", ["one two", "three"]),           # 2 ann, 3 words
        ("R1B", ["four five six seven"]),        # 1 ann, 4 words
    ]:
        (tmp_path / code / "clinical_eeg").mkdir(parents=True)
        _write_edf_with_annotations(
            tmp_path / code / "clinical_eeg" / f"{code}_file.edf", texts)

    result = count_annotations.scan_parent(tmp_path)
    assert result["R1A"] == (2, 3, 1, 0)   # (n_ann, n_words, ok, skipped)
    assert result["R1B"] == (1, 4, 1, 0)


def test_scan_parent_skips_annotation_sidecars(tmp_path):
    """NEGATIVE regression: inplace-mode '*_annotations.edf' stubs are
    copies of what's already inline in the main EDF. Counting both
    would inflate the estimate by 2x and mislead the operator into
    scheduling more time than needed.
    """
    inner = tmp_path / "R1SUBJ" / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "R1SUBJ.edf",
                                 ["one two", "three"])
    _write_edf_with_annotations(inner / "R1SUBJ_annotations.edf",
                                 ["one two", "three"])   # sidecar dup

    result = count_annotations.scan_parent(tmp_path)
    # Only the main EDF counted -- sidecar filtered
    assert result["R1SUBJ"] == (2, 3, 1, 0)


def test_scan_parent_reports_skipped_files_separately(tmp_path):
    """Files pyedflib refuses (raw NK EDF+D, corrupt bytes) must land
    in the 'skipped' column, NOT be silently dropped -- an operator
    scoping review time deserves to know coverage is incomplete."""
    inner = tmp_path / "R1SUBJ" / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "ok.edf", ["one two"])
    (inner / "garbage.edf").write_bytes(b"not an EDF at all")

    result = count_annotations.scan_parent(tmp_path)
    n_ann, n_words, n_ok, n_skipped = result["R1SUBJ"]
    assert n_ok == 1
    assert n_skipped == 1
    assert n_ann == 1 and n_words == 2   # only the readable file
