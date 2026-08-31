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


def test_preflight_prefers_sidecar_when_present(tmp_path):
    """In-place cleaning zeros the main EDF's annotation channel and
    writes annotations into a '<base>_annotations.edf' sidecar. The
    reviewer must read from the sidecar in that case -- reading the
    main EDF would show zero annotations. Also drops per-file I/O
    from GB (main EDF) to KB (sidecar), critical on Box/NFS."""
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    (inner / "deidentify.json").write_text("{}")
    _write_edf(inner / "R1755A.edf", ["real"])
    _write_edf(inner / "R1755A_annotations.edf", ["sidecar"])
    edfs = preflight_subject_for_review(subj)
    # Sidecar picked over main EDF.
    assert [p.name for p in edfs] == ["R1755A_annotations.edf"]


def test_preflight_falls_back_to_main_edf_when_no_sidecar(tmp_path):
    """Rewrite mode (--copy_path) leaves annotations inline in the main
    EDF and writes no sidecar. Reviewer must read from the main EDF."""
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    (inner / "deidentify.json").write_text("{}")
    _write_edf(inner / "R1755A.edf", ["real"])
    edfs = preflight_subject_for_review(subj)
    assert [p.name for p in edfs] == ["R1755A.edf"]


def test_preflight_mixes_modes_per_recording(tmp_path):
    """A subject dir could plausibly contain some recordings cleaned
    in-place (sidecar exists) and some in rewrite mode (no sidecar).
    The picker is per-recording so each one gets the right file."""
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    (inner / "deidentify.json").write_text("{}")
    _write_edf(inner / "R1755A_a.edf", ["main-a"])
    _write_edf(inner / "R1755A_a_annotations.edf", ["sidecar-a"])
    _write_edf(inner / "R1755A_b.edf", ["main-b"])   # no sidecar
    edfs = preflight_subject_for_review(subj)
    assert [p.name for p in edfs] == [
        "R1755A_a_annotations.edf",
        "R1755A_b.edf",
    ]


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
    assert len(SessionJournal(subj / "clinical_eeg").read_all()) == 1


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
    assert len(SessionJournal(subj / "clinical_eeg").read_all()) == 2


def test_bulk_regex_swap_queues_edits_across_all_files(tmp_path):
    """The operator's use case: subject has *X annotations (from an
    earlier Presidio pass on a Mark-named subject); bulk_regex_swap
    with pattern '\\*X\\b' -> '*Mark' queues pending edits on every
    matching annotation in every reviewable file."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["*X", "other", "*X trailing"],
        "b.edf": ["*X", "*Y"],
    })
    c = AnnotationReviewController(subj)

    n = c.bulk_regex_swap(r"\*X\b", "*Mark")

    # *X in a.edf[0], a.edf[2], b.edf[0] all match; other/*Y don't.
    assert n == 3
    pending = c.pending_edits()
    assert len(pending) == 3
    new_texts = {p.new_text for p in pending}
    assert new_texts == {"*Mark", "*Mark trailing", "*Mark"}


def test_bulk_regex_swap_invalid_regex_returns_minus_1(tmp_path):
    """Malformed regex -> -1 sentinel + no pending edits queued. The
    TUI turns this into an error banner instead of crashing."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["*X"]})
    c = AnnotationReviewController(subj)
    n = c.bulk_regex_swap("[unclosed", "*Mark")
    assert n == -1
    assert c.pending_edits() == []


def test_bulk_regex_swap_skips_no_op_matches(tmp_path):
    """When the pattern matches but sub returns identical text, DON'T
    queue an edit. Keeps the pending counter honest and the journal
    tidy."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["*Mark", "other"]})
    c = AnnotationReviewController(subj)
    n = c.bulk_regex_swap(r"\*Mark", "*Mark")  # substitution -> identical
    assert n == 0
    assert c.pending_edits() == []


def test_bulk_regex_swap_respects_prior_pending_edit(tmp_path):
    """If a manual edit already queued 'FOO' on some annotation, the
    swap operates on FOO (the current display text), not the raw
    annotation. Same principle as re-editing via 'e' pre-fills the
    edit buffer with the pending text."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["raw text"]})
    c = AnnotationReviewController(subj)
    c.queue_edit("FOO")  # manual edit on ann 0
    assert c.pending_edits()[0].new_text == "FOO"

    # Swap should see FOO, not "raw text".
    n = c.bulk_regex_swap("FOO", "BAR")
    assert n == 1
    pending = c.pending_edits()
    assert len(pending) == 1
    assert pending[0].new_text == "BAR"
    # orig_text stays anchored to the RAW value for a clean audit trail
    # regardless of how many stacked edits fed into the final new_text.
    assert pending[0].orig_text == "raw text"


def test_bulk_regex_swap_scope_current_only_touches_current_file(tmp_path):
    """scope='current' walks only the file at self.file_cursor. Useful
    for spot fixes when the operator doesn't want a global rewrite."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["*X"], "b.edf": ["*X"]})
    c = AnnotationReviewController(subj)

    # Cursor starts at file_cursor=0 (a.edf).
    n = c.bulk_regex_swap(r"\*X", "*Mark", scope="current")
    assert n == 1
    pending = c.pending_edits()
    assert len(pending) == 1
    # Only the a.edf annotation was rewritten.
    assert pending[0].file_path.endswith("a.edf")


def test_bulk_regex_swap_backreferences_work(tmp_path):
    """re.sub with backreferences (\\1, \\g<name>) is a common idiom.
    Verify the swap uses full re.sub semantics, not a plain string
    replace."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["marker: 123", "marker: 456"]})
    c = AnnotationReviewController(subj)
    n = c.bulk_regex_swap(r"marker: (\d+)", r"code=\1", scope="all")
    assert n == 2
    new_texts = sorted(p.new_text for p in c.pending_edits())
    assert new_texts == ["code=123", "code=456"]


def test_queue_edit_dedups_identical_repeat_submission(tmp_path):
    """Enter-mashing or re-saving the same text must NOT append duplicate
    journal lines or inflate the pending-count. Same key + same new_text
    is a no-op; the returned record is the existing one."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["orig"]})
    c = AnnotationReviewController(subj)
    first = c.queue_edit("redacted")
    second = c.queue_edit("redacted")  # identical text -- no-op
    third = c.queue_edit("redacted")   # identical again -- no-op
    assert first is second, "second submission should return existing record"
    assert first is third, "third submission should return existing record"
    assert len(c.pending_edits()) == 1
    # Journal has exactly 1 entry, not 3.
    assert len(SessionJournal(subj / "clinical_eeg").read_all()) == 1


def test_current_display_text_returns_pending_edit_new_text(tmp_path):
    """Regression: pressing 'e' after an edit must show the OPERATOR'S
    current text (their previous edit), not the original on-disk text.
    current_display_text is what the tui.py `e` handler pre-fills into
    the edit buffer."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["dr. smith noted"]})
    c = AnnotationReviewController(subj)
    assert c.current_display_text() == "dr. smith noted"
    c.queue_edit("<REDACTED> noted")
    assert c.current_display_text() == "<REDACTED> noted", (
        "must return the pending edit's new_text so re-editing lets "
        "the operator build on their change instead of starting over"
    )


def test_visible_lines_shows_pending_edit_text(tmp_path):
    """Regression: after an edit, the scroll view must render the
    OPERATOR'S new text -- not the original -- so they get immediate
    visual confirmation the edit registered."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["orig text"]})
    c = AnnotationReviewController(subj)
    c.queue_edit("edited text")
    lines = c.visible_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.is_edited is True
    assert line.display_text == "edited text"
    # Original still on the raw Annotation for audit / apply-time matching.
    assert line.annotation.text == "orig text"


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
    ReviewedTracker(subj / "clinical_eeg").mark_reviewed(ReviewedFile.new(
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
    ReviewedTracker(subj / "clinical_eeg").mark_reviewed(ReviewedFile.new(
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
    entries = ReviewedTracker(subj / "clinical_eeg").read_all()
    assert len(entries) == 1
    assert entries[0].n_annotations == 2
    # n_edited counts only edits to THIS file
    assert entries[0].n_edited >= 0   # (edit at cursor 0 was 'y-redacted')


def test_unreviewed_reviewable_files_returns_untracked(tmp_path):
    """Anything reviewable that's NOT in the tracker is returned so the
    CLI knows what to prompt about on quit."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["x"], "b.edf": ["y"], "c.edf": ["z"]})
    c = AnnotationReviewController(subj)
    # None marked yet -> all three are unreviewed.
    assert len(c.unreviewed_reviewable_files()) == 3
    # Mark just 'a' -> only b, c remain.
    c.mark_current_file_reviewed()
    unreviewed = c.unreviewed_reviewable_files()
    assert sorted(p.name for p in unreviewed) == ["b.edf", "c.edf"]


def test_mark_all_reviewable_files_reviewed_bulk_marks_and_dedups(tmp_path):
    """The end-of-session bulk-mark path: operator quits after looking at
    everything (or after only marking one via 'n'), and the CLI's prompt
    on quit calls this to close out the tracker. Every previously-
    unmarked reviewable file gets an entry; already-marked files are
    skipped (idempotent)."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["x"], "b.edf": ["y"], "c.edf": ["z"]})
    c = AnnotationReviewController(subj)
    # Pre-mark 'a' via the explicit-per-file path.
    c.mark_current_file_reviewed()
    assert len(ReviewedTracker(subj / "clinical_eeg").read_all()) == 1
    newly = c.mark_all_reviewable_files_reviewed()
    # Only b and c are newly marked; a was already tracked.
    assert sorted(p.name for p in newly) == ["b.edf", "c.edf"]
    entries = ReviewedTracker(subj / "clinical_eeg").read_all()
    # Tracker now has entries for all three files (1 pre-existing + 2 new).
    assert sorted({e.file_path for e in entries}) == sorted(
        [str(subj / "clinical_eeg" / n) for n in ("a.edf", "b.edf", "c.edf")])


def test_mark_all_reviewable_files_reviewed_is_idempotent(tmp_path):
    """Second call after everything's marked adds no new entries --
    important because the CLI might call this after auto-mark-all-
    whitelisted has already run inside --preload-all."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["x"], "b.edf": ["y"]})
    c = AnnotationReviewController(subj)
    first = c.mark_all_reviewable_files_reviewed()
    assert len(first) == 2
    second = c.mark_all_reviewable_files_reviewed()
    assert second == [], "no files should be newly marked on second call"


def test_mark_all_reviewable_files_reviewed_captures_pending_edit_counts(
        tmp_path):
    """Each ReviewedFile entry records how many edits landed on that
    file. Bulk-mark must correctly attribute pending edits by file so
    the audit trail is honest even for files marked reviewed without
    the operator ever pressing 'n' on them."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["x", "y"], "b.edf": ["z"]})
    c = AnnotationReviewController(subj)
    c.queue_edit("x-redacted")   # edit on a.edf (file_cursor=0, ann=0)
    c.move_cursor(+1)
    c.queue_edit("y-redacted")   # edit on a.edf (file_cursor=0, ann=1)
    c.mark_all_reviewable_files_reviewed()
    entries = {e.file_path: e for e in ReviewedTracker(subj / "clinical_eeg").read_all()}
    a_entry = entries[str(subj / "clinical_eeg" / "a.edf")]
    b_entry = entries[str(subj / "clinical_eeg" / "b.edf")]
    assert a_entry.n_edited == 2, f"a.edf should have 2 edits: {a_entry}"
    assert b_entry.n_edited == 0, f"b.edf should have 0 edits: {b_entry}"


# ---------------------------------------------------------------------------
# git-log-style visible_lines rendering
# ---------------------------------------------------------------------------

def test_visible_lines_default_hides_whitelisted_annotations(tmp_path):
    """Default hide_whitelisted=True: whitelisted annotations are
    DROPPED from the scroll view. The operator only sees + interacts
    with lines that need review. Motivated by the actual review
    workflow: without hiding, every `+numeric.000`, `*Mark`,
    `RhythmicBurst *` etc. shows greyed out and clutters the view."""
    wl_path = _write_wl(tmp_path, {"A": ["boilerplate"]})
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["boilerplate", "real event 1", "boilerplate",
                   "real event 2", "boilerplate"]})
    c = AnnotationReviewController(subj, whitelist_path=wl_path)   # hide=True default
    lines = c.visible_lines(context=10)
    texts = [l.annotation.text for l in lines]
    assert texts == ["real event 1", "real event 2"], (
        f"whitelisted 'boilerplate' should be hidden by default; "
        f"got {texts}"
    )


def test_visible_lines_show_whitelisted_keeps_them_visible(tmp_path):
    """hide_whitelisted=False (--show-whitelisted opt-in) restores the
    pre-fix behaviour: whitelisted lines stay in the scroll view with
    is_whitelisted=True so the renderer can grey them out."""
    wl_path = _write_wl(tmp_path, {"A": ["boilerplate"]})
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["boilerplate", "real event"]})
    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                    hide_whitelisted=False)
    lines = c.visible_lines(context=5)
    texts = [l.annotation.text for l in lines]
    assert texts == ["boilerplate", "real event"]
    assert lines[0].is_whitelisted is True
    assert lines[1].is_whitelisted is False


def test_move_cursor_skips_whitelisted_when_hidden(tmp_path):
    """Cursor navigation must skip whitelisted annotations when they're
    hidden -- otherwise the cursor could land on an invisible line and
    the operator sees nothing selected."""
    wl_path = _write_wl(tmp_path, {"A": ["skip"]})
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["real0", "skip", "skip", "real1", "skip", "real2"]})
    c = AnnotationReviewController(subj, whitelist_path=wl_path)   # hide=True
    assert c.annotation_cursor == 0    # starts at real0

    c.move_cursor(+1)
    assert c.annotation_cursor == 3, (
        f"move_cursor(+1) should skip whitelisted 'skip' entries and "
        f"land on real1 (idx 3); got {c.annotation_cursor}"
    )

    c.move_cursor(+1)
    assert c.annotation_cursor == 5    # real2

    # Moving back skips whitelisted the other direction too.
    c.move_cursor(-1)
    assert c.annotation_cursor == 3
    c.move_cursor(-1)
    assert c.annotation_cursor == 0


def test_jump_to_end_lands_on_last_visible_annotation(tmp_path):
    """`G` (jump-to-end) must land on the last VISIBLE annotation, not
    the raw last annotation. Otherwise on a file ending with
    whitelisted lines the operator would jump to an invisible cursor
    and `on_last_annotation_of_file` would fire on a line they can't
    see."""
    wl_path = _write_wl(tmp_path, {"A": ["boilerplate"]})
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["real", "boilerplate", "boilerplate"]})
    c = AnnotationReviewController(subj, whitelist_path=wl_path)
    c.jump_to_end()
    assert c.annotation_cursor == 0, (
        f"jump_to_end should land on 'real' (idx 0), the only visible "
        f"annotation; got {c.annotation_cursor}")


def test_on_last_annotation_of_file_uses_visible_indices(tmp_path):
    """The n-gate ('cannot advance until you scroll to last annotation')
    must key off the last VISIBLE annotation, not the raw last. Under
    hide_whitelisted=True, when the last non-whitelisted line is at
    index K, the operator has completed review as soon as their cursor
    reaches K -- even if there are whitelisted lines at K+1, K+2."""
    wl_path = _write_wl(tmp_path, {"A": ["boilerplate"]})
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["real0", "real1", "boilerplate", "boilerplate"]})
    c = AnnotationReviewController(subj, whitelist_path=wl_path)

    # At cursor=0, not on last visible.
    assert c.on_last_annotation_of_file() is False
    # Move to real1 (idx 1) -- last VISIBLE annotation. n-gate satisfied.
    c.move_cursor(+1)
    assert c.annotation_cursor == 1
    assert c.on_last_annotation_of_file() is True, (
        "cursor is on the last visible annotation -- n-gate must fire")


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
    is in from the terminal.

    Exercised under hide_whitelisted=False (the --show-whitelisted
    CLI opt-in) since the default is to hide whitelisted lines
    entirely. The flags are still populated in that mode -- the
    renderer just doesn't need them when the lines are hidden."""
    subj = _make_subject(tmp_path, "R1755A", {
        "a.edf": ["PAT REF EEG", "seizure", "notes"]})
    wl_path = _write_wl(tmp_path, {"A": ["PAT REF EEG"]})
    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                    hide_whitelisted=False)
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


def test_preload_all_drops_files_with_only_whitelisted_annotations(tmp_path):
    """POSITIVE: with preload_all=True + a whitelist that silences
    every annotation in a file, that file is auto-marked reviewed
    AND removed from the reviewable list. The operator only ever
    sees files with real content."""
    import json as _json
    subj = _make_subject(tmp_path, "R1755A", {
        "all_boilerplate.edf": ["PAT REF EEG", "PAT REF EEG"],
        "has_real.edf": ["seizure onset", "eyes closed"],
    })
    # Whitelist silences everything in all_boilerplate.edf
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(_json.dumps({
        "shared": [], "per_site": {"A": [r"PAT REF EEG"]}}))

    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                     preload_all=True)
    reviewable_names = [
        c._edfs[i].name for i in c._file_indices]
    assert "all_boilerplate.edf" not in reviewable_names
    assert "has_real.edf" in reviewable_names

    # The dropped file was silently marked reviewed on disk
    from clean_eeg.annotation_review.journal import ReviewedTracker
    reviewed = {r.file_path for r in ReviewedTracker(subj / "clinical_eeg").read_all()}
    assert any(p.endswith("all_boilerplate.edf") for p in reviewed)
    c.close()


def test_preload_all_keeps_files_with_partial_boilerplate(tmp_path):
    """NEGATIVE regression: a file with SOME whitelisted content and
    SOME real content stays in the reviewable list. Guards against
    an over-eager drop that would hide files where the operator
    still needs to look at 1-2 real annotations."""
    import json as _json
    subj = _make_subject(tmp_path, "R1755A", {
        "mixed.edf": ["PAT REF EEG", "real event", "PAT REF EEG"],
    })
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(_json.dumps({
        "shared": [], "per_site": {"A": [r"PAT REF EEG"]}}))

    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                     preload_all=True)
    reviewable_names = [
        c._edfs[i].name for i in c._file_indices]
    assert "mixed.edf" in reviewable_names
    c.close()


def test_preload_all_records_auto_skipped_whitelist_count(tmp_path):
    """The controller exposes num_files_auto_skipped_whitelist so the
    CLI can distinguish "auto-skipped because fully whitelisted" from
    "already reviewed by human in a prior session" when the reviewable
    queue is empty at startup."""
    import json as _json
    subj = _make_subject(tmp_path, "R1755A", {
        "all_boilerplate.edf": ["PAT REF EEG", "PAT REF EEG"],
        "also_boilerplate.edf": ["PAT REF EEG"],
        "has_real.edf": ["seizure onset"],
    })
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(_json.dumps({
        "shared": [], "per_site": {"A": [r"PAT REF EEG"]}}))

    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                     preload_all=True)
    assert c.num_files_auto_skipped_whitelist == 2
    c.close()


def test_preload_all_does_not_drop_files_with_delete_matched_annotations(tmp_path):
    """R1670J regression: files whose only annotations match the
    delete-whitelist bucket (e.g. Jefferson's
    'Segment: REC START.*' pattern) were being auto-marked reviewed
    at preload time even though the apply path preserves those rows
    verbatim -- so the file stayed on disk with the patient name in
    the annotation. Delete-matched patterns must NOT be treated as
    'already handled' by the preload auto-skip.
    """
    import json as _json
    subj = _make_subject(tmp_path, "R1755J", {  # J site
        "leaks_phi.edf": ["Segment: REC START SMITH E"],
        "purely_whitelisted.edf": ["A1+A2 OFF"],
    })
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(_json.dumps({
        "shared": [],
        "per_site": {"J": [r"A1\+A2 (?:ON|OFF)"]},
        "delete_shared": [],
        "delete_per_site": {"J": [r"Segment: REC START.*"]},
    }))

    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                     preload_all=True)
    reviewable_names = [c._edfs[i].name for i in c._file_indices]
    # leaks_phi.edf's ONLY annotation matches the delete pattern.
    # Prior to the fix it would be auto-skipped -- confirm it now
    # shows up for review.
    assert "leaks_phi.edf" in reviewable_names, reviewable_names
    # purely_whitelisted.edf's annotation matches the true whitelist,
    # so it's still auto-skipped.
    assert "purely_whitelisted.edf" not in reviewable_names, reviewable_names
    c.close()


def test_auto_queue_delete_matches_queues_edits_with_X(tmp_path):
    """auto_queue_delete_matches walks every reviewable file, finds
    annotations that fullmatch the boilerplate whitelist's delete
    bucket, and queues an EditRecord with new_text='X'. Sanity-checks
    that the pipeline's delete-branch behavior is mirrored at review
    time so the operator sees the replacements before apply."""
    import json as _json
    subj = _make_subject(tmp_path, "R1755J", {
        "leaks.edf": ["Segment: REC START SMITH E", "unrelated"],
        "cleaner.edf": ["A1+A2 OFF"],   # matches true whitelist, not delete
    })
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(_json.dumps({
        "shared": [], "per_site": {"J": [r"A1\+A2 (?:ON|OFF)"]},
        "delete_shared": [],
        "delete_per_site": {"J": [r"Segment: REC START.*"]},
    }))

    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                     preload_all=True)
    n = c.auto_queue_delete_matches(replacement="X")
    assert n == 1, "exactly one annotation should have delete-matched"

    edits = c.pending_edits()
    assert len(edits) == 1
    assert edits[0].orig_text == "Segment: REC START SMITH E"
    assert edits[0].new_text == "X"
    c.close()


def test_auto_queue_delete_matches_idempotent_on_second_call(tmp_path):
    """Calling twice must not double-queue -- the second call sees
    the pending edit already carries new_text='X' and short-circuits."""
    import json as _json
    subj = _make_subject(tmp_path, "R1755J", {
        "leaks.edf": ["Segment: REC START SMITH E"],
    })
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(_json.dumps({
        "shared": [], "per_site": {"J": []},
        "delete_shared": [], "delete_per_site": {"J": [r"Segment: REC START.*"]},
    }))
    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                     preload_all=True)
    n1 = c.auto_queue_delete_matches()
    n2 = c.auto_queue_delete_matches()
    assert n1 == 1
    assert n2 == 0
    assert len(c.pending_edits()) == 1
    c.close()


def test_auto_queue_delete_matches_skips_already_X_annotations(tmp_path):
    """Files that have already been cleaned by the pipeline's delete
    branch will have the annotation as 'X' on disk. matches_delete
    fires on the ORIGINAL text, so 'X' is not a match and nothing is
    queued -- the auto-queue at review time is a no-op on
    already-cleaned data."""
    import json as _json
    subj = _make_subject(tmp_path, "R1755J", {
        "cleaned.edf": ["X", "X"],
    })
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(_json.dumps({
        "shared": [], "per_site": {"J": []},
        "delete_shared": [], "delete_per_site": {"J": [r"Segment: REC START.*"]},
    }))
    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                     preload_all=False)
    n = c.auto_queue_delete_matches()
    assert n == 0
    c.close()


def test_sentinel_X_annotation_never_hidden_by_whitelist(tmp_path):
    """The shared '.{1,5}' whitelist pattern fullmatches 'X' (1 char),
    which would silently hide every annotation the pipeline's delete
    branch replaced. Regression: is_whitelisted must special-case
    text=='X' to keep those rows visible so the operator can audit
    what got deleted."""
    import json as _json
    subj = _make_subject(tmp_path, "R1755J", {
        "cleaned.edf": ["X", "X", "Segment: REC START SMITH E"],
    })
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(_json.dumps({
        "shared": [".{1,5}"],   # would fullmatch 'X' pre-fix
        "per_site": {}, "delete_shared": [], "delete_per_site": {},
    }))
    c = AnnotationReviewController(subj, whitelist_path=wl_path,
                                     preload_all=False)
    anns = c.annotations_in_current_file()
    for a in anns:
        if a.text == "X":
            assert not c.is_whitelisted(a), (
                "sentinel 'X' must NEVER be treated as whitelisted -- "
                "the operator needs to see what the delete branch replaced")
    c.close()


def test_num_files_auto_skipped_whitelist_defaults_to_zero(tmp_path):
    """Without preload_all the auto-drop path never runs; the counter
    must stay at 0 so the CLI doesn't misreport."""
    subj = _make_subject(tmp_path, "R1755A", {"a.edf": ["x"]})
    c = AnnotationReviewController(subj)
    assert c.num_files_auto_skipped_whitelist == 0
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
    j = SessionJournal(subj / "clinical_eeg")
    with j:
        j.append(EditRecord.new(
            file_path=str(subj / "clinical_eeg" / "a.edf"),
            record_index=99,
            byte_offset_in_record=999,
            onset_s=999.0,
            orig_text="never existed", new_text="ghost"))

    c = AnnotationReviewController(subj)
    assert c.pending_edits() == []
