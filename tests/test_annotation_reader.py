"""Tests for the fast mmap-based annotation reader.

Priorities:
    1. Correctness: same texts as pyedflib for a valid EDF+
    2. Speed / independence: works on files pyedflib refuses (raw NK
       exports, EDF+D not yet split) -- the fast reader must work
       even when pyedflib doesn't
    3. Edge cases: multi-TAL-per-record, TALs with durations
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from clean_eeg.annotation_reader import (
    Annotation,
    count_words_in_annotations,
    iter_annotations,
)


def _write_edf_with_annotations(path: Path,
                                 annotations: list[tuple[float, float, str]],
                                 duration_s: int = 10,
                                 ) -> None:
    """Write a minimal EDF+ with the given annotations.
    Each item is ``(onset_s, duration_s, text)``."""
    n_ch = 2
    sr = 100
    signal_headers = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sr,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_ch)
    ]
    t = np.arange(0, duration_s, 1.0 / sr, dtype=np.float32)
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
        for onset, duration, text in annotations:
            f.writeAnnotation(onset, duration if duration > 0 else -1, text)


def _pyedflib_annotations(path: Path) -> list[str]:
    """Ground-truth annotation texts via pyedflib. Skips empties (which
    pyedflib sometimes emits for the timekeeping TAL)."""
    with pyedflib.EdfReader(str(path)) as f:
        _, _, texts = f.readAnnotations()
    return [t for t in texts if t and t.strip()]


# ---------------------------------------------------------------------------
# Correctness vs pyedflib
# ---------------------------------------------------------------------------

def test_iter_annotations_matches_pyedflib_texts(tmp_path):
    """The fast reader must return the same annotation texts pyedflib
    would -- otherwise we can't safely swap it in as a drop-in."""
    edf = tmp_path / "ann.edf"
    _write_edf_with_annotations(edf, [
        (0.5, -1, "START"),
        (1.2, -1, "eyes closed"),
        (3.7, 2.0, "seizure"),   # with duration
        (5.0, -1, "eyes open"),
        (8.0, -1, "END"),
    ])

    fast = iter_annotations(edf)
    fast_texts = [a.text for a in fast if a.text.strip()]
    pyedflib_texts = _pyedflib_annotations(edf)

    assert fast_texts == pyedflib_texts, (
        f"fast={fast_texts}  pyedflib={pyedflib_texts}")


def test_iter_annotations_preserves_onset_and_duration(tmp_path):
    """Onset + duration round-trip through the byte-level parse. The
    upcoming review TUI needs both to display timestamps and to locate
    edits back to the source record."""
    edf = tmp_path / "ann.edf"
    _write_edf_with_annotations(edf, [
        (1.2, -1, "no-duration"),
        (3.7, 2.5, "with-duration"),
    ])

    anns = iter_annotations(edf)
    by_text = {a.text: a for a in anns if a.text.strip()}
    assert "no-duration" in by_text
    assert "with-duration" in by_text
    assert by_text["no-duration"].onset_s == pytest.approx(1.2, abs=1e-3)
    assert by_text["no-duration"].duration_s == pytest.approx(0.0)
    assert by_text["with-duration"].onset_s == pytest.approx(3.7, abs=1e-3)
    assert by_text["with-duration"].duration_s == pytest.approx(2.5, abs=1e-3)


def test_iter_annotations_zero_annotations_returns_empty(tmp_path):
    """Files with only the timekeeping TALs (which the pipeline creates
    for wiped-annotations mode) must return an empty list, not error
    or emit spurious entries from parsing the timekeeping."""
    edf = tmp_path / "empty.edf"
    _write_edf_with_annotations(edf, annotations=[])
    anns = iter_annotations(edf)
    assert [a for a in anns if a.text.strip()] == []


# ---------------------------------------------------------------------------
# Multiple TALs per data record (dense clinical annotations)
# ---------------------------------------------------------------------------

def test_iter_annotations_handles_multiple_tals_in_same_record(tmp_path):
    """When several annotations land in the same 1-second data record,
    all of them must be surfaced. Clinical recordings routinely have
    dense annotations near seizure onset -- missing any would corrupt
    both the count and the review UX."""
    edf = tmp_path / "dense.edf"
    # 5 annotations within the same 1-s record
    _write_edf_with_annotations(edf, [
        (0.1, -1, "A"),
        (0.2, -1, "B"),
        (0.3, -1, "C"),
        (0.4, -1, "D"),
        (0.5, -1, "E"),
    ])
    fast = [a.text for a in iter_annotations(edf) if a.text.strip()]
    assert set(fast) >= {"A", "B", "C", "D", "E"}


# ---------------------------------------------------------------------------
# Works on files pyedflib refuses
# ---------------------------------------------------------------------------

def _corrupt_num_data_records_to_break_pyedflib(edf_path: Path) -> None:
    """Overwrite num_data_records (bytes 236..244) with a value that
    doesn't match the on-disk record count. pyedflib validates
    file_size == header + records * record_size and refuses when it
    disagrees. The main header fields (num_signals, per-signal
    num_samples) that the fast reader uses to compute the annotation
    offset stay untouched, so the fast reader can still read.
    """
    data = bytearray(edf_path.read_bytes())
    # num_data_records lives at offset 236, 8 ASCII bytes. Bump it
    # up so pyedflib expects far more data than the file contains.
    data[236:244] = b"99999999"
    edf_path.write_bytes(bytes(data))


def test_iter_annotations_works_when_pyedflib_would_refuse(tmp_path):
    """CORE VALUE PROP: byte-level annotation read succeeds on files
    pyedflib rejects. Proven by: (1) build a valid EDF+, (2) verify
    fast reader picks up an annotation, (3) corrupt the reserved
    field so pyedflib refuses, (4) prove pyedflib really does refuse,
    (5) prove the fast reader STILL surfaces the annotation.

    This is why we can point count_annotations at raw pre-clean data
    (which is often marked EDF+D and rejected by pyedflib).
    """
    edf = tmp_path / "will_be_corrupted.edf"
    _write_edf_with_annotations(edf, [(0.5, -1, "MARKER_TEXT")])

    # Baseline: fast reader sees the annotation on the valid file
    baseline = [a.text for a in iter_annotations(edf) if a.text.strip()]
    assert "MARKER_TEXT" in baseline

    # Break num_data_records so pyedflib refuses
    _corrupt_num_data_records_to_break_pyedflib(edf)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        with pyedflib.EdfReader(str(edf)):
            pass

    # Fast reader still surfaces the annotation
    after_corruption = [a.text for a in iter_annotations(edf)
                         if a.text.strip()]
    assert "MARKER_TEXT" in after_corruption, (
        f"fast reader must survive pyedflib-refused files; got "
        f"{after_corruption}")


# ---------------------------------------------------------------------------
# Helper: word count matches whitespace tokenization
# ---------------------------------------------------------------------------

def test_count_words_matches_whitespace_tokenization():
    anns = [
        Annotation(record_index=0, byte_offset_in_record=0,
                   onset_s=0.0, duration_s=0.0, text="PAT REF EEG"),
        Annotation(record_index=1, byte_offset_in_record=0,
                   onset_s=1.0, duration_s=0.0, text="Seizure"),
        Annotation(record_index=2, byte_offset_in_record=0,
                   onset_s=2.0, duration_s=0.0, text="   "),  # skipped
    ]
    assert count_words_in_annotations(anns) == 4   # 3 + 1
