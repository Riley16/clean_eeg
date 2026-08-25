"""Tests for the corruption-safe apply-edits pass.

Coverage priorities:
    1. HAPPY PATH: an edit lands in the file's annotations after
       apply, readable via pyedflib.
    2. SIGNAL SAFETY: signal bytes byte-identical before/after
       (guaranteed by construction because only annotation-channel
       bytes are ever mutated; test enforces it).
    3. CORRUPTION SAFETY: verify failure -> original untouched, temp
       kept for inspection.
    4. STALE EDIT: unmatched EditRecord aborts with a clear reason
       rather than silently skipping.
    5. NO-EDITS: empty pending list is a no-op, not a crash.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from clean_eeg.annotation_review.apply_edits import (
    APPLY_TEMP_SUFFIX,
    ApplyEditsError,
    _apply_edits_in_memory,
    apply_pending_edits,
)
from clean_eeg.annotation_review.models import EditRecord
from clean_eeg.annotation_reader import iter_annotations


def _write_edf(path: Path, annotations: list[tuple[float, str]],
                duration_s: int = 10) -> None:
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
        for onset, text in annotations:
            f.writeAnnotation(onset, -1, text)


def _annotation_texts(path: Path) -> list[str]:
    with pyedflib.EdfReader(str(path)) as f:
        _, _, texts = f.readAnnotations()
    return [str(t) for t in texts if str(t).strip()]


def _signal_hash(path: Path) -> str:
    """SHA256 of the raw int16 signal bytes (excluding annotation
    channel). Used to prove signals are byte-identical after apply."""
    with pyedflib.EdfReader(str(path)) as f:
        sigs = [f.readSignal(i, digital=True)
                for i in range(f.signals_in_file)]
    h = hashlib.sha256()
    for s in sigs:
        h.update(np.ascontiguousarray(s).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_apply_edits_replaces_annotation_text_on_disk(tmp_path):
    """POSITIVE end-to-end: one edit, one file, readable via pyedflib
    after apply. This is the core value: the operator's edit actually
    made it to disk in a format the pipeline (and downstream tools)
    can read.
    """
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [
        (0.5, "SEIZURE at Dr. Smith clinic"),
        (2.0, "eyes closed"),
    ])
    ann = iter_annotations(edf)
    dirty = next(a for a in ann if "Dr. Smith" in a.text)

    edit = EditRecord.new(
        file_path=str(edf),
        record_index=dirty.record_index,
        byte_offset_in_record=dirty.byte_offset_in_record,
        onset_s=dirty.onset_s,
        orig_text=dirty.text,
        new_text="SEIZURE at XXXXXXX clinic")

    results = apply_pending_edits([edit])

    assert len(results) == 1
    assert results[0].succeeded
    assert results[0].n_edits_applied == 1

    texts_after = _annotation_texts(edf)
    assert "SEIZURE at XXXXXXX clinic" in texts_after
    assert "SEIZURE at Dr. Smith clinic" not in texts_after
    # Untouched annotation survived
    assert "eyes closed" in texts_after


def test_apply_edits_multiple_edits_same_file(tmp_path):
    """Multiple edits on one file all land in a single apply pass.
    Regression guard against a bug where only the last edit sticks
    (would trivially happen if the merge overwrote instead of
    building the full text list)."""
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [
        (0.5, "a"), (1.5, "b"), (2.5, "c"),
    ])

    ann = iter_annotations(edf)
    edits = [
        EditRecord.new(
            file_path=str(edf),
            record_index=a.record_index,
            byte_offset_in_record=a.byte_offset_in_record,
            onset_s=a.onset_s, orig_text=a.text,
            new_text=a.text.upper() * 2)  # "AA", "BB", "CC"
        for a in ann
    ]
    results = apply_pending_edits(edits)
    assert results[0].succeeded

    texts = _annotation_texts(edf)
    assert set(texts) == {"AA", "BB", "CC"}


def test_apply_edits_across_multiple_files(tmp_path):
    """Grouping-by-file: edits split across two files are applied
    separately and both succeed."""
    edf_a = tmp_path / "A.edf"
    edf_b = tmp_path / "B.edf"
    _write_edf(edf_a, [(0.5, "keep_a")])
    _write_edf(edf_b, [(0.5, "orig_b")])

    ann_a = iter_annotations(edf_a)[0]
    ann_b = iter_annotations(edf_b)[0]
    edits = [
        EditRecord.new(file_path=str(edf_a),
                       record_index=ann_a.record_index,
                       byte_offset_in_record=ann_a.byte_offset_in_record,
                       onset_s=ann_a.onset_s, orig_text=ann_a.text,
                       new_text="new_a"),
        EditRecord.new(file_path=str(edf_b),
                       record_index=ann_b.record_index,
                       byte_offset_in_record=ann_b.byte_offset_in_record,
                       onset_s=ann_b.onset_s, orig_text=ann_b.text,
                       new_text="new_b"),
    ]
    results = apply_pending_edits(edits)
    assert len(results) == 2
    assert all(r.succeeded for r in results)
    assert "new_a" in _annotation_texts(edf_a)
    assert "new_b" in _annotation_texts(edf_b)


# ---------------------------------------------------------------------------
# Signal safety
# ---------------------------------------------------------------------------

def test_signal_bytes_are_byte_identical_after_apply(tmp_path):
    """HARD REQUIREMENT: annotation-only edits MUST NOT change any
    signal sample. Enforced by SHA256 of the concatenated signal-
    channel bytes before and after apply. If this ever regresses,
    the manual review would silently corrupt data."""
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [(0.5, "orig")], duration_s=20)

    hash_before = _signal_hash(edf)

    ann = iter_annotations(edf)[0]
    edit = EditRecord.new(
        file_path=str(edf),
        record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s, orig_text=ann.text,
        new_text="edited longer text with more content")

    results = apply_pending_edits([edit])
    assert results[0].succeeded

    hash_after = _signal_hash(edf)
    assert hash_after == hash_before, (
        "signal bytes changed after annotation-only edit -- "
        "corruption in the merge path")


# ---------------------------------------------------------------------------
# Corruption safety: unmatched edit
# ---------------------------------------------------------------------------

def test_apply_edits_aborts_on_unmatched_edit_leaving_original_intact(
        tmp_path):
    """SAFETY: an EditRecord that doesn't match any current annotation
    (file was mutated between review and apply) must abort the pass
    with ApplyEditsError, leaving the original untouched. Silently
    skipping would mean the operator's session log claims edits that
    were never applied.
    """
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [(0.5, "hello")])
    ann = iter_annotations(edf)[0]

    # Craft a stale edit that won't match: pretend the annotation
    # originally said something different.
    stale = EditRecord.new(
        file_path=str(edf),
        record_index=ann.record_index,
        byte_offset_in_record=99999,   # bogus offset
        onset_s=99.9,                   # bogus onset
        orig_text="never existed",
        new_text="ghost")

    original_bytes = edf.read_bytes()
    results = apply_pending_edits([stale])

    assert not results[0].succeeded
    # Original file MUST be byte-identical
    assert edf.read_bytes() == original_bytes


def test_apply_edits_matches_via_onset_and_orig_text_fallback(tmp_path):
    """POSITIVE regression: when byte_offset_in_record doesn't match
    (e.g. after a benign re-parse), the fallback (onset_s + orig_text)
    still lets the edit land. Guards against a brittle
    identify-by-offset-only design that would refuse any edit whose
    file was re-read between sessions."""
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [(0.5, "original")])
    ann = iter_annotations(edf)[0]

    edit = EditRecord.new(
        file_path=str(edf),
        record_index=999,               # doesn't match current
        byte_offset_in_record=999,      # doesn't match current
        onset_s=ann.onset_s,            # DOES match
        orig_text=ann.text,             # DOES match
        new_text="redacted")

    results = apply_pending_edits([edit])
    assert results[0].succeeded
    assert "redacted" in _annotation_texts(edf)


# ---------------------------------------------------------------------------
# No-edits + edge cases
# ---------------------------------------------------------------------------

def test_apply_pending_edits_empty_list_is_noop(tmp_path):
    """Empty pending list -> no results, no crash. Reached when the
    operator quits a review after only navigating (no edits made)."""
    results = apply_pending_edits([])
    assert results == []


def test_apply_edits_refuses_leftover_temp_file(tmp_path):
    """DEFENSIVE: if a prior apply crashed mid-write leaving a
    <path>.review_apply.tmp behind, refuse to proceed. Operator
    must inspect and remove manually -- silently overwriting the
    leftover would destroy evidence of the earlier failure.
    """
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [(0.5, "hello")])
    # Simulate leftover from a prior crashed run
    (tmp_path / f"R1TEST.edf{APPLY_TEMP_SUFFIX}").write_bytes(b"stale")

    ann = iter_annotations(edf)[0]
    edit = EditRecord.new(
        file_path=str(edf),
        record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s, orig_text=ann.text, new_text="new")

    results = apply_pending_edits([edit])
    assert not results[0].succeeded
    assert "leftover" in (results[0].error_message or "").lower()


# ---------------------------------------------------------------------------
# _apply_edits_in_memory unit tests (isolated from disk)
# ---------------------------------------------------------------------------

def test_apply_edits_in_memory_replaces_by_key():
    from clean_eeg.annotation_reader import Annotation
    current = [
        Annotation(record_index=0, byte_offset_in_record=10,
                   onset_s=0.5, duration_s=0, text="a"),
        Annotation(record_index=1, byte_offset_in_record=10,
                   onset_s=1.5, duration_s=0, text="b"),
    ]
    edits = [EditRecord.new(
        file_path="/x.edf", record_index=1, byte_offset_in_record=10,
        onset_s=1.5, orig_text="b", new_text="B_EDITED")]
    result = _apply_edits_in_memory(current, edits)
    assert result == ["a", "B_EDITED"]


def test_apply_edits_in_memory_raises_on_unmatched():
    from clean_eeg.annotation_reader import Annotation
    current = [Annotation(record_index=0, byte_offset_in_record=10,
                            onset_s=0.5, duration_s=0, text="a")]
    edits = [EditRecord.new(
        file_path="/x.edf", record_index=99, byte_offset_in_record=99,
        onset_s=99.0, orig_text="not_a", new_text="ghost")]
    with pytest.raises(ApplyEditsError, match="could not be matched"):
        _apply_edits_in_memory(current, edits)
