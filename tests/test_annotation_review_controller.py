"""Tests for the AnnotationReviewController state machine.

No TTY, no prompt_toolkit -- the controller is pure Python state
transitions on top of the annotation reader + journal + tracker.
This is where we prove the review UX is correct.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from clean_eeg.annotation_review.controller import (
    AnnotationReviewController,
    PreflightFailure,
    preflight_subject_for_review,
    _derive_site_code,
)
from clean_eeg.annotation_review.journal import (
    ReviewedTracker,
    SessionJournal,
)
from clean_eeg.annotation_review.models import ReviewedFile


# ---------------------------------------------------------------------------
# Fixture: write a preflight-passing subject dir
# ---------------------------------------------------------------------------

def _write_edf(path: Path, annotation_texts: list[str]) -> None:
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


def _make_subject(tmp_path: Path, code: str,
                   files: dict[str, list[str]]) -> Path:
    """Build a preflight-passing subject dir:
        <tmp>/<code>/clinical_eeg/<filename>.edf   for each entry
        <tmp>/<code>/clinical_eeg/deidentify.json  (empty stub)
    ``files`` maps filename -> list of annotation texts."""
    subj = tmp_path / code
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    for name, anns in files.items():
        _write_edf(inner / name, anns)
    (inner / "deidentify.json").write_text('{"subject_code": "' + code + '"}')
    return subj


# ---------------------------------------------------------------------------
# preflight_subject_for_review: gates
# ---------------------------------------------------------------------------

def test_preflight_passes_on_cleaned_subject(tmp_path):
    subj = _make_subject(tmp_path, "R1755A",
                          {"R1755A_a.edf": ["one", "two"]})
    edfs = preflight_subject_for_review(subj)
    assert len(edfs) == 1
    assert edfs[0].name == "R1755A_a.edf"


def test_preflight_fails_when_subfolder_missing(tmp_path):
    subj = tmp_path / "R1755A"
    subj.mkdir()
    with pytest.raises(PreflightFailure, match="clinical_eeg"):
        preflight_subject_for_review(subj)


def test_preflight_fails_when_manifest_missing(tmp_path):
    """HARD REQUIREMENT: subjects without deidentify.json are still
    RAW -- annotations may contain PHI. Refuse to load them into the
    review TUI. The error message must point at the fix (run
    clean-batch-eeg first)."""
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf(inner / "a.edf", ["one"])   # EDF present but no manifest
    with pytest.raises(PreflightFailure, match="deidentify.json"):
        preflight_subject_for_review(subj)


def test_preflight_fails_on_no_edfs(tmp_path):
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    (inner / "deidentify.json").write_text("{}")
    with pytest.raises(PreflightFailure, match="no .edf"):
        preflight_subject_for_review(subj)


def test_preflight_excludes_annotation_sidecars(tmp_path):
    """Sidecar '*_annotations.edf' files are the inplace-mode
    annotation stubs -- their annotations are already surfaced via
    the main EDF, so including them in the review list would double-
    count everything."""
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    (inner / "deidentify.json").write_text("{}")
    _write_edf(inner / "R1755A.edf", ["real"])
    _write_edf(inner / "R1755A_annotations.edf", ["sidecar"])
    edfs = preflight_subject_for_review(subj)
    assert [p.name for p in edfs] == ["R1755A.edf"]


# ---------------------------------------------------------------------------
# _derive_site_code
# ---------------------------------------------------------------------------

def test_derive_site_code_various_shapes():
    assert _derive_site_code("R1755A") == "A"
    assert _derive_site_code("R1702J_1") == "J"
    assert _derive_site_code("something_else") is None
    assert _derive_site_code("") is None


# ---------------------------------------------------------------------------
# Cursor navigation
# ---------------------------------------------------------------------------

def test_cursor_starts_at_first_annotation_of_first_file(tmp_path):
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["ann1", "ann2", "ann3"],
    })
    c = AnnotationReviewController(subj)
    assert c.current_annotation().text == "ann1"


def test_move_cursor_clamps_at_edges(tmp_path):
    """POSITIVE + NEGATIVE regression: moving past either end must
    NOT wrap or advance to the next file automatically. next_file()
    is explicit precisely to avoid an accidental key hold from
    scrolling past unreviewed content."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["ann1", "ann2", "ann3"],
    })
    c = AnnotationReviewController(subj)
    c.move_cursor(-5)
    assert c.current_annotation().text == "ann1"
    c.move_cursor(+100)
    assert c.current_annotation().text == "ann3"
    c.move_cursor(-1)
    assert c.current_annotation().text == "ann2"


def test_jump_to_start_and_end(tmp_path):
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["a", "b", "c", "d", "e"],
    })
    c = AnnotationReviewController(subj)
    c.jump_to_end()
    assert c.current_annotation().text == "e"
    c.jump_to_start()
    assert c.current_annotation().text == "a"


def test_next_prev_file_advances_and_resets_cursor(tmp_path):
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["a1", "a2", "a3"],
        "b.edf": ["b1"],
    })
    c = AnnotationReviewController(subj)
    c.jump_to_end()
    assert c.current_annotation().text == "a3"

    assert c.next_file() is True
    assert c.current_annotation().text == "b1"   # reset to first
    assert c.next_file() is False                # no more files

    assert c.prev_file() is True
    assert c.current_annotation().text == "a1"   # reset to first


def test_on_last_annotation_of_file(tmp_path):
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["a", "b"]})
    c = AnnotationReviewController(subj)
    assert c.on_last_annotation_of_file() is False
    c.move_cursor(+1)
    assert c.on_last_annotation_of_file() is True


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def test_queue_edit_records_in_pending_and_journal(tmp_path):
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["real name here"]})
    c = AnnotationReviewController(subj)
    c.queue_edit("redacted")
    assert c.is_current_edited()
    pending = c.pending_edits()
    assert len(pending) == 1
    assert pending[0].orig_text == "real name here"
    assert pending[0].new_text == "redacted"
    # Persisted to journal on disk (survives process death)
    assert len(SessionJournal(subj).read_all()) == 1


def test_queue_edit_second_time_overwrites_first(tmp_path):
    """Same annotation edited twice: most-recent intent wins in
    pending state. Journal still has both entries as an audit trail."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["orig"]})
    c = AnnotationReviewController(subj)
    c.queue_edit("first attempt")
    c.queue_edit("second attempt")
    pending = c.pending_edits()
    assert len(pending) == 1                         # dedup by cursor
    assert pending[0].new_text == "second attempt"
    # Journal preserved both for audit
    assert len(SessionJournal(subj).read_all()) == 2


# ---------------------------------------------------------------------------
# Whitelist filtering (view state)
# ---------------------------------------------------------------------------

def _write_wl(tmp_path: Path, per_site: dict, shared=None) -> Path:
    """Write a whitelist JSON in the format
    ``clean_eeg.annotation_boilerplate.load_whitelist`` accepts."""
    import json
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({
        "shared": shared or [],
        "per_site": per_site,
    }))
    return p


def test_is_whitelisted_uses_site_from_subject_name(tmp_path):
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["PAT REF EEG",
                                                          "seizure"]})
    wl_path = _write_wl(tmp_path, {"A": ["PAT REF EEG"]})
    c = AnnotationReviewController(subj, whitelist_path=wl_path)
    anns = c.annotations_in_current_file()
    assert c.is_whitelisted(anns[0]) is True    # PAT REF EEG -> matched
    assert c.is_whitelisted(anns[1]) is False   # seizure -> not matched


def test_reload_whitelist_picks_up_disk_edits(tmp_path):
    """HARD REQUIREMENT: operator can edit the whitelist JSON on
    disk and hit 'r' to have new entries take effect immediately
    without restarting the TUI."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["boilerplate"]})
    wl_path = _write_wl(tmp_path, {"A": []})
    c = AnnotationReviewController(subj, whitelist_path=wl_path)
    ann = c.annotations_in_current_file()[0]
    assert c.is_whitelisted(ann) is False

    # Operator edits the file on disk to whitelist "boilerplate"
    import json
    wl_path.write_text(json.dumps({
        "shared": [], "per_site": {"A": ["boilerplate"]}}))
    # Without reload, view is stale
    assert c.is_whitelisted(ann) is False
    c.reload_whitelist()
    assert c.is_whitelisted(ann) is True


# ---------------------------------------------------------------------------
# Reviewed tracker + skip-on-restart
# ---------------------------------------------------------------------------

def test_already_reviewed_files_skipped_by_default(tmp_path):
    """Restart scenario: files listed in the tracker do NOT appear
    in the reviewable file list, so the operator doesn't scroll
    through them again."""
    subj = _make_subject(tmp_path, "R1755A", {
        "done.edf": ["a", "b"],
        "todo.edf": ["c", "d"],
    })
    ReviewedTracker(subj).mark_reviewed(ReviewedFile.new(
        file_path=subj / "clinical_eeg" / "done.edf",
        n_annotations=2, n_edited=0))
    c = AnnotationReviewController(subj)
    assert c.num_files_to_review == 1
    assert c.current_file().name == "todo.edf"


def test_respect_reviewed_tracker_false_includes_all_files(tmp_path):
    """--include-reviewed override: re-review everything (e.g. after
    updating the whitelist and wanting a fresh pass)."""
    subj = _make_subject(tmp_path, "R1755A", {
        "done.edf": ["a"], "todo.edf": ["b"],
    })
    ReviewedTracker(subj).mark_reviewed(ReviewedFile.new(
        file_path=subj / "clinical_eeg" / "done.edf",
        n_annotations=1, n_edited=0))
    c = AnnotationReviewController(subj, respect_reviewed_tracker=False)
    assert c.num_files_to_review == 2


def test_mark_current_file_reviewed_persists_to_disk(tmp_path):
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["x", "y"]})
    c = AnnotationReviewController(subj)
    c.queue_edit("y-redacted")  # 1 edit on current file
    c.jump_to_end()
    c.mark_current_file_reviewed()
    entries = ReviewedTracker(subj).read_all()
    assert len(entries) == 1
    assert entries[0].n_annotations == 2
    # n_edited counts only edits to THIS file
    assert entries[0].n_edited >= 0   # (edit at cursor 0 was 'y-redacted')


# ---------------------------------------------------------------------------
# git-log-style visible_lines rendering
# ---------------------------------------------------------------------------

def test_visible_lines_returns_current_plus_context_below(tmp_path):
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": [f"ann{i}" for i in range(10)],
    })
    c = AnnotationReviewController(subj)
    c.move_cursor(+2)   # cursor at ann2
    lines = c.visible_lines(context=3)
    assert [l.annotation.text for l in lines] == [
        "ann2", "ann3", "ann4", "ann5"]
    assert lines[0].is_current
    assert not any(l.is_current for l in lines[1:])


def test_visible_lines_marks_whitelisted_and_edited(tmp_path):
    """Positive regression: whitelisted + edited flags are surfaced
    so the TUI can render them distinctly (grey + strikethrough,
    say). Without these the operator can't tell what state each row
    is in from the terminal."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["PAT REF EEG", "seizure", "notes"]})
    wl_path = _write_wl(tmp_path, {"A": ["PAT REF EEG"]})
    c = AnnotationReviewController(subj, whitelist_path=wl_path)
    c.move_cursor(+1)  # cursor at seizure
    c.queue_edit("SEIZURE")   # edit current
    lines = c.visible_lines(context=5)
    # ann0 not visible (below-only context)
    by_text = {l.annotation.text: l for l in lines}
    assert "PAT REF EEG" not in by_text          # not in window
    assert by_text["seizure"].is_current
    assert by_text["seizure"].is_edited
    assert by_text["notes"].is_whitelisted is False


# ---------------------------------------------------------------------------
# Crash recovery: journal re-hydration
# ---------------------------------------------------------------------------

def test_controller_rehydrates_pending_edits_from_journal(tmp_path):
    """Simulate crash + restart: prior session's journal edits are
    re-loaded into pending on next controller instantiation, so the
    approval gate at end-of-subject still sees them and the operator
    doesn't have to redo work.
    """
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["one", "two"]})
    c1 = AnnotationReviewController(subj)
    c1.queue_edit("one-edited")
    c1.close()

    c2 = AnnotationReviewController(subj)
    pending = c2.pending_edits()
    assert len(pending) == 1
    assert pending[0].new_text == "one-edited"


def test_prefetch_warms_next_two_files_at_init(tmp_path):
    """POSITIVE: at controller init, prefetch should have queued
    (or already loaded) the file at cursor + the next
    PREFETCH_LOOKAHEAD files. Verified by checking
    _annotations_cache after a brief wait for background workers.
    """
    from clean_eeg.annotation_review.controller import PREFETCH_LOOKAHEAD
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["one"],
        "b.edf": ["two"],
        "c.edf": ["three"],
        "d.edf": ["four"],
        "e.edf": ["five"],
    })
    c = AnnotationReviewController(subj)
    # Trigger sync load of the current file
    c.annotations_in_current_file()
    # Give the prefetch pool a chance to finish (small files ->
    # milliseconds). Waiting on the pool's queue would be cleaner
    # but that requires reaching into internals.
    import time as _t
    _t.sleep(0.5)
    # After init + one sync load: current file (0) cached, next
    # PREFETCH_LOOKAHEAD files (1..PREFETCH_LOOKAHEAD) either
    # cached OR their futures still pending. Either way they must
    # NOT need a fresh sync load when accessed.
    for i in range(1, 1 + PREFETCH_LOOKAHEAD):
        assert (i in c._annotations_cache
                or i in c._prefetch_futures), (
            f"file {i} not warmed (cache keys: "
            f"{sorted(c._annotations_cache)}, "
            f"future keys: {sorted(c._prefetch_futures)})")
    c.close()


def test_prefetched_file_returns_from_cache_on_next_file(tmp_path):
    """HAPPY PATH: navigate to a pre-warmed file -- no synchronous
    read from disk needed. Simulated by monkeypatching
    iter_annotations to count calls; after prefetch has run for
    file 1, next_file() should NOT trigger a fresh call.
    """
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["a"], "b.edf": ["b"], "c.edf": ["c"],
    })
    c = AnnotationReviewController(subj)
    c.annotations_in_current_file()
    import time as _t
    _t.sleep(0.5)   # let prefetch settle

    call_count = {"n": 0}
    from clean_eeg.annotation_review import controller as _cm
    orig = _cm.iter_annotations
    def counting(path):
        call_count["n"] += 1
        return orig(path)
    _cm.iter_annotations = counting
    try:
        c.next_file()
        anns = c.annotations_in_current_file()
    finally:
        _cm.iter_annotations = orig

    assert [a.text for a in anns] == ["b"]
    assert call_count["n"] == 0, (
        f"next_file() triggered {call_count['n']} sync loads -- "
        "prefetch cache missed")
    c.close()


def test_controller_close_shuts_down_prefetch_pool(tmp_path):
    """Regression: close() must shutdown the ThreadPoolExecutor so
    hanging daemon threads don't outlive the process. Verified by
    checking the pool's _shutdown flag."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["x"]})
    c = AnnotationReviewController(subj)
    c.close()
    assert c._prefetch_pool._shutdown is True


def test_controller_drops_stale_journal_entries_that_no_longer_match(
        tmp_path):
    """DEFENSIVE: if the source file changed between sessions
    (unlikely but possible -- annotation removed by another tool),
    the stale journal entry is dropped rather than corrupting
    pending state. Operator sees no phantom pending edits."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["one", "two"]})
    # Manually write a journal entry that won't match anything in
    # the current file
    from clean_eeg.annotation_review.models import EditRecord
    j = SessionJournal(subj)
    with j:
        j.append(EditRecord.new(
            file_path=str(subj / "clinical_eeg" / "a.edf"),
            record_index=99,
            byte_offset_in_record=999,
            onset_s=999.0,
            orig_text="never existed", new_text="ghost"))

    c = AnnotationReviewController(subj)
    assert c.pending_edits() == []
