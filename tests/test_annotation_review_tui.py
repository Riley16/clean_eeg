"""Tests for the annotation-review TUI + CLI.

Coverage priorities (things that DON'T require a real event loop):
    1. Whitelist append: atomic writer, re.escape semantics, per-site
       bucket, dedup, crash safety.
    2. TUI builds against a real controller (no crash on layout /
       key-binding wire-up).
    3. CLI: preflight failure exit code, no-files-to-review path,
       approval-gate y/N prompt.

The interactive event-loop behavior (arrow-key nav, edit mode
transitions, whitelist-on-'w') is out of scope for automated
tests -- driving a prompt_toolkit app headlessly requires a full
async event loop and reliably deadlocks without one. Visual QA
covers that via the manual test commands in the session summary.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyedflib
import pytest

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from clean_eeg.annotation_review.controller import (
    AnnotationReviewController,
)
from clean_eeg.annotation_review.tui import (
    append_annotation_to_whitelist,
    build_review_app,
)
from clean_eeg.annotation_review_cli import main as cli_main


# ---------------------------------------------------------------------------
# Headless TUI runner (proven asyncio pattern)
# ---------------------------------------------------------------------------

def _drive_tui(controller, keys: str, timeout: float = 5.0):
    """Run the review TUI headlessly with pre-formed keystrokes.

    Feeds ``keys`` into a prompt_toolkit pipe input, runs the app
    via ``app.run_async()`` inside ``asyncio.wait_for(..., timeout)``
    so a stuck test kills itself instead of hanging pytest. Returns
    when the app exits (typically via 'q' in the fed sequence).

    Every headless test must end its ``keys`` with 'q' to exit REVIEW
    mode; without it the app waits forever for more input and the
    timeout fires.
    """
    async def _run():
        with create_pipe_input() as inp:
            with create_app_session(input=inp, output=DummyOutput()):
                app = build_review_app(controller)
                inp.send_text(keys)
                try:
                    await asyncio.wait_for(app.run_async(),
                                            timeout=timeout)
                except asyncio.TimeoutError:
                    pytest.fail(
                        f"TUI headless run did not exit within "
                        f"{timeout}s. Key sequence: {keys!r}")
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Fixture: subject dir that passes preflight
# ---------------------------------------------------------------------------

def _write_edf(path: Path, annotations: list[tuple[float, str]],
                duration_s: int = 10) -> None:
    n_ch = 2
    sr = 100
    sh = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sr,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_ch)
    ]
    t = np.arange(0, duration_s, 1.0 / sr, dtype=np.float32)
    sigs = [(1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
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
        f.setSignalHeaders(sh)
        f.writeSamples(sigs)
        for onset, text in annotations:
            f.writeAnnotation(onset, -1, text)


def _make_subject(tmp_path: Path, code: str = "R1755A",
                   files: dict[str, list[tuple[float, str]]] | None = None
                   ) -> Path:
    files = files or {"a.edf": [(0.5, "hello"), (1.5, "world")]}
    subj = tmp_path / code
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    for name, anns in files.items():
        _write_edf(inner / name, anns)
    (inner / "deidentify.json").write_text(
        json.dumps({"subject_code": code}))
    return subj


# ---------------------------------------------------------------------------
# append_annotation_to_whitelist
# ---------------------------------------------------------------------------

def test_append_creates_file_when_missing(tmp_path):
    wl = tmp_path / "wl.json"
    append_annotation_to_whitelist(wl, "PAT REF EEG", site_code="A")
    data = json.loads(wl.read_text())
    assert data["per_site"]["A"] == [re.escape("PAT REF EEG")]
    assert data["shared"] == []


def test_append_uses_re_escape_semantics(tmp_path):
    """POSITIVE regression: special regex chars in the annotation
    text must be escaped so future annotations that happen to look
    like the pattern aren't silenced. Example: an annotation
    'seizure at 3.5' should whitelist 'seizure at 3\\.5' literally,
    not the more permissive 'seizure at 3.5' (which would silence
    'seizure at 305' too)."""
    wl = tmp_path / "wl.json"
    append_annotation_to_whitelist(wl, "seizure at 3.5", site_code="A")
    data = json.loads(wl.read_text())
    pattern = data["per_site"]["A"][0]
    # Escaped period
    assert r"3\.5" in pattern
    # Compiles to a regex that only matches the exact literal text
    assert re.fullmatch(pattern, "seizure at 3.5")
    assert not re.fullmatch(pattern, "seizure at 305")


def test_append_per_site_bucket_matches_site(tmp_path):
    wl = tmp_path / "wl.json"
    append_annotation_to_whitelist(wl, "text_A", site_code="A")
    append_annotation_to_whitelist(wl, "text_J", site_code="J")
    data = json.loads(wl.read_text())
    assert data["per_site"]["A"] == [re.escape("text_A")]
    assert data["per_site"]["J"] == [re.escape("text_J")]


def test_append_none_site_goes_to_shared(tmp_path):
    """When site_code is None (subject folder didn't match R1XXXY),
    the whitelist entry lands in the shared bucket -- applied
    across every site's future reviews."""
    wl = tmp_path / "wl.json"
    append_annotation_to_whitelist(wl, "global text", site_code=None)
    data = json.loads(wl.read_text())
    assert data["shared"] == [re.escape("global text")]
    assert data["per_site"] == {}


def test_append_dedupes_repeated_entries(tmp_path):
    """NEGATIVE regression: pressing 'w' twice on the same annotation
    must NOT append a duplicate. Keeps the whitelist tidy and avoids
    misleading count metrics."""
    wl = tmp_path / "wl.json"
    append_annotation_to_whitelist(wl, "boilerplate", site_code="A")
    append_annotation_to_whitelist(wl, "boilerplate", site_code="A")
    data = json.loads(wl.read_text())
    assert data["per_site"]["A"] == [re.escape("boilerplate")]


def test_append_atomic_write_preserves_original_on_crash(tmp_path,
                                                           monkeypatch):
    """CRASH SAFETY: if os.replace fails after the temp is written,
    the original whitelist is unchanged. Verified by monkeypatching
    os.replace to raise; the on-disk JSON should still equal the
    pre-call content, and no leaked .tmp file should survive."""
    import os
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"shared": ["existing"],
                                "per_site": {}}))
    original = wl.read_text()

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace"):
        append_annotation_to_whitelist(wl, "new_entry", site_code="A")

    # Original untouched
    assert wl.read_text() == original
    # No .tmp litter
    leftovers = [p for p in tmp_path.glob("wl.json.*.tmp")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# TUI: builds against a real controller (no crash on layout)
# ---------------------------------------------------------------------------

def test_build_review_app_smoke(tmp_path):
    """SMOKE: the layout + key bindings wire up against a real
    controller without raising. Guards against typos in
    ConditionalContainer / Filter / KeyBindings wiring that would
    otherwise only surface when a user launches the tool."""
    subj = _make_subject(tmp_path)
    controller = AnnotationReviewController(subj)
    app = build_review_app(controller)
    assert app.layout is not None
    assert app.key_bindings is not None


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def test_cli_preflight_failure_returns_2(tmp_path, capsys):
    """A subject without deidentify.json is not ready for review;
    CLI must fail hard with exit code 2 and a targeted message that
    points at the fix (run clean-batch-eeg first)."""
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf(inner / "a.edf", [(0.5, "x")])
    # NO deidentify.json -> preflight fails

    rc = cli_main(["--subject-dir", str(subj)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "deidentify" in err.lower()


def test_cli_no_files_to_review_returns_0_with_hint(tmp_path, capsys):
    """A subject where every file is already in the reviewed tracker
    should exit 0 with a hint about --include-reviewed. NOT launch
    the TUI on nothing."""
    from clean_eeg.annotation_review.journal import ReviewedTracker
    from clean_eeg.annotation_review.models import ReviewedFile
    subj = _make_subject(tmp_path, files={"a.edf": [(0.5, "x")]})
    ReviewedTracker(subj).mark_reviewed(ReviewedFile.new(
        file_path=subj / "clinical_eeg" / "a.edf",
        n_annotations=1, n_edited=0))

    rc = cli_main(["--subject-dir", str(subj)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "already reviewed" in out or "--include-reviewed" in out


def test_cli_all_files_whitelisted_says_whitelisted_not_reviewed(
        tmp_path, capsys):
    """When --preload-all auto-drops every file because 100% of its
    annotations match the whitelist, the operator should see a message
    that says the files were whitelisted -- not "already reviewed" --
    so they don't chase a phantom prior session that never happened.
    """
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(json.dumps({
        "shared": [], "per_site": {"A": [r"PAT REF EEG"]}}))
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "PAT REF EEG")],
        "b.edf": [(0.5, "PAT REF EEG")]})

    rc = cli_main(["--subject-dir", str(subj),
                    "--whitelist-path", str(wl_path),
                    "--preload-all"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "whitelisted" in out
    assert "nothing to review" in out.lower()
    # Must NOT misdirect the operator to the tracker.
    assert ".annotation_reviewed_tracker" not in out


def test_cli_approval_prompt_only_y_applies():
    """SAFETY: only exact 'y' / 'yes' triggers the apply pass. A
    fat-fingered 'n' or 'no' must discard, not silently apply.
    Guards against a semantics change that would flip the default."""
    from clean_eeg.annotation_review_cli import _prompt_apply
    from clean_eeg.annotation_review.models import EditRecord
    pending = [EditRecord.new(
        file_path="/x.edf", record_index=0, byte_offset_in_record=0,
        onset_s=0.0, orig_text="a", new_text="b")]
    with patch("builtins.input", return_value="y"):
        assert _prompt_apply(pending) is True
    with patch("builtins.input", return_value="Yes"):
        assert _prompt_apply(pending) is True
    with patch("builtins.input", return_value="n"):
        assert _prompt_apply(pending) is False
    with patch("builtins.input", return_value=""):
        assert _prompt_apply(pending) is False
    with patch("builtins.input", return_value="apply"):
        assert _prompt_apply(pending) is False


def test_cli_prompt_apply_empty_pending_returns_false():
    """No pending edits -> False (don't run apply, don't prompt).
    The CLI already has an earlier 'no pending edits' early exit,
    but the helper must be a safe standalone too."""
    from clean_eeg.annotation_review_cli import _prompt_apply
    assert _prompt_apply([]) is False


# ---------------------------------------------------------------------------
# Headless TUI behavioral tests (pre-formed inputs via pipe + asyncio)
# ---------------------------------------------------------------------------

def test_tui_j_advances_cursor(tmp_path):
    """Pressing 'j' twice moves the cursor down two annotations.
    Baseline that the REVIEW-mode key bindings actually fire."""
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "a1"), (1.5, "a2"), (2.5, "a3")]})
    controller = AnnotationReviewController(subj)
    _drive_tui(controller, "jjq")
    assert controller.annotation_cursor == 2


def test_tui_k_moves_cursor_up_after_j(tmp_path):
    """'j' then 'k' returns to the previous annotation. Guards
    against 'k' silently being no-op after a REVIEW-mode refactor."""
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "a1"), (1.5, "a2"), (2.5, "a3")]})
    controller = AnnotationReviewController(subj)
    _drive_tui(controller, "jjkq")
    assert controller.annotation_cursor == 1


def test_tui_edit_enter_saves_pending_edit(tmp_path):
    """Press 'e' -> EDIT mode with the annotation pre-filled ->
    backspace over it -> type new text -> Enter to save. Pending
    edit lands in the controller's buffer AND on disk (via journal
    flush inside queue_edit).
    """
    subj = _make_subject(tmp_path, files={"a.edf": [(0.5, "orig")]})
    controller = AnnotationReviewController(subj)

    # \b = backspace; \n = Enter. Sequence: e (enter EDIT) ->
    # 4 backspaces to clear "orig" -> type "REDACTED" -> Enter
    # (save) -> q (quit REVIEW).
    _drive_tui(controller, "e" + "\b" * 4 + "REDACTED" + "\nq")

    pending = controller.pending_edits()
    assert len(pending) == 1
    assert pending[0].orig_text == "orig"
    assert pending[0].new_text == "REDACTED"


def test_tui_w_appends_current_to_whitelist_and_reloads(tmp_path):
    """Press 'w' on the current annotation: whitelist JSON gains the
    re.escape'd literal entry AND the controller's live matcher
    reflects it immediately (via reload_whitelist called on 'w').
    This is the 'immediate effect' hard requirement.
    """
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "boilerplate"), (1.5, "real event")]})
    wl_path = tmp_path / "wl.json"
    wl_path.write_text(json.dumps({"shared": [], "per_site": {"A": []}}))
    controller = AnnotationReviewController(subj, whitelist_path=wl_path)
    ann0 = controller.annotations_in_current_file()[0]
    assert not controller.is_whitelisted(ann0)   # baseline

    _drive_tui(controller, "wq")

    # File: literal-escaped entry appended
    data = json.loads(wl_path.read_text())
    assert re.escape("boilerplate") in data["per_site"]["A"]
    # Live matcher: same annotation now returns True
    assert controller.is_whitelisted(
        controller.annotations_in_current_file()[0])


def test_tui_s_key_two_stage_swap_queues_edits(tmp_path):
    """End-to-end TUI drive of the regex-swap flow: press 's' (enter
    swap-pattern mode) -> type `\\*X` -> Enter (transition to
    swap-replace mode) -> type `*Mark` -> Enter (apply). Pending
    edits land in the controller; display would show the substituted
    text via the existing display_text mechanism.

    Guards against three regressions at once:
      * 's' keybind wired to swap_pattern mode transition.
      * Enter in swap_pattern advances to swap_replace (not review).
      * Enter in swap_replace calls bulk_regex_swap correctly.
    """
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "*X"), (1.5, "other"), (2.5, "*X trailing")]})
    controller = AnnotationReviewController(subj)

    # Sequence: s (open swap) -> `\*X` (pattern) -> Enter (advance)
    # -> `*Mark` (replacement) -> Enter (apply) -> q (quit)
    _drive_tui(controller, "s" + r"\*X" + "\n" + "*Mark" + "\nq")

    pending = controller.pending_edits()
    # Two *X occurrences in a.edf; "other" is unchanged.
    assert len(pending) == 2, f"expected 2 pending edits, got {pending}"
    new_texts = {p.new_text for p in pending}
    assert new_texts == {"*Mark", "*Mark trailing"}


def test_tui_s_key_swap_escape_aborts(tmp_path):
    """Pressing Esc during either stage of the swap prompt returns
    to review mode WITHOUT queuing edits. Same escape-hatch UX as
    the manual-edit abort path."""
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "*X"), (1.5, "other")]})
    controller = AnnotationReviewController(subj)

    # s (open swap) -> `\*X` (pattern) -> Esc (abort) -> q (quit)
    # \x1b is Escape.
    _drive_tui(controller, "s" + r"\*X" + "\x1b" + "q")

    assert controller.pending_edits() == [], (
        f"escape from swap must NOT queue edits: {controller.pending_edits()}"
    )


def test_tui_n_marks_current_file_reviewed(tmp_path):
    """Press 'n': current file's entry lands in
    .annotation_reviewed_tracker on disk. Regression guard against
    'n' advancing without persisting the mark (would break the
    restart-skips-reviewed-files behavior).

    Single-annotation file: cursor is at position 0 which IS the last
    (and only) annotation, so the n-gate is satisfied immediately."""
    subj = _make_subject(tmp_path, files={"a.edf": [(0.5, "x")]})
    controller = AnnotationReviewController(subj)

    _drive_tui(controller, "nq")

    from clean_eeg.annotation_review.journal import ReviewedTracker
    entries = ReviewedTracker(subj).read_all()
    assert len(entries) == 1
    assert entries[0].file_path.endswith("a.edf")


def test_reset_review_state_deletes_tracker_and_pending_session(tmp_path):
    """reset_review_state removes the tracker + pending session.jsonl
    (the two artifacts that carry 'aborted-mid-review' state) so the
    next TUI launch treats every file as fresh."""
    from clean_eeg.annotation_review.journal import (
        reset_review_state,
        REVIEWED_TRACKER_NAME, SESSION_SUBDIR, SESSION_JSONL_NAME,
    )
    inner = tmp_path / "R1755J" / "clinical_eeg"
    inner.mkdir(parents=True)
    tracker = inner / REVIEWED_TRACKER_NAME
    tracker.write_text('{"file_path":"/tmp/a.edf","reviewed_at":"2026","n_annotations":1,"n_edited":0}\n')
    session_dir = inner / SESSION_SUBDIR
    session_dir.mkdir()
    session = session_dir / SESSION_JSONL_NAME
    session.write_text('{"file_path":"/tmp/a.edf","record_index":0,"byte_offset_in_record":0,"onset_s":0.5,"orig_text":"x","new_text":"y","edited_at":"2026"}\n')

    deleted = reset_review_state(inner)

    assert not tracker.exists()
    assert not session.exists()
    assert "tracker" in deleted
    assert "pending_session" in deleted


def test_reset_review_state_preserves_applied_audit_trail(tmp_path):
    """Applied-session archive files (edits that already landed on
    disk) MUST survive reset -- they're the compliance record of what
    changed and when. Same for discarded/. Deleting either would
    silently lose PHI-review provenance."""
    from clean_eeg.annotation_review.journal import (
        reset_review_state,
        SESSION_SUBDIR, APPLIED_SUBDIR, DISCARDED_SUBDIR,
    )
    inner = tmp_path / "R1755J" / "clinical_eeg"
    inner.mkdir(parents=True)
    (inner / ".annotation_reviewed_tracker").write_text("{}\n")
    session_dir = inner / SESSION_SUBDIR
    (session_dir / APPLIED_SUBDIR).mkdir(parents=True)
    applied_archive = session_dir / APPLIED_SUBDIR / "session_20260101T000000Z.jsonl"
    applied_archive.write_text('{"edit":"landed"}\n')
    (session_dir / DISCARDED_SUBDIR).mkdir()
    discarded_archive = session_dir / DISCARDED_SUBDIR / "session_20260102T000000Z.jsonl"
    discarded_archive.write_text('{"edit":"rejected"}\n')

    reset_review_state(inner)

    assert applied_archive.exists(), (
        "applied-session archive must survive reset (compliance record)")
    assert discarded_archive.exists(), (
        "discarded-session archive must survive reset (compliance record)")


def test_reset_review_state_noop_when_nothing_to_reset(tmp_path):
    """Fresh subject dir (never reviewed) -> reset is a safe no-op,
    returns empty dict. Regression against a reset that crashes on
    missing files instead of just skipping them."""
    from clean_eeg.annotation_review.journal import reset_review_state
    inner = tmp_path / "R1755J" / "clinical_eeg"
    inner.mkdir(parents=True)

    deleted = reset_review_state(inner)
    assert deleted == {}


def test_cli_rerun_annot_review_calls_reset_before_controller(monkeypatch,
                                                                tmp_path,
                                                                capsys):
    """--rerun-annot-review must run reset_review_state BEFORE the
    controller is instantiated -- otherwise the controller would
    read the stale tracker + register the un-applied edits as
    'pending', defeating the reset."""
    from clean_eeg import annotation_review_cli as _cli

    # Set up subject with prior review state.
    subj = tmp_path / "R1755J"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    tracker = inner / ".annotation_reviewed_tracker"
    tracker.write_text('{"file_path":"/tmp/a.edf","reviewed_at":"2026","n_annotations":1,"n_edited":0}\n')

    # Capture what the controller sees when it's instantiated: the
    # tracker must be gone by then.
    tracker_exists_at_controller_init: dict[str, bool] = {}

    class _StubController:
        def __init__(self, subject_dir, *, subfolder=None, whitelist_path=None,
                      respect_reviewed_tracker=True, preload_all=False,
                      **_ignored):
            tracker_exists_at_controller_init["value"] = tracker.exists()
            self.num_files_to_review = 0
            self.num_files = 0
            self.num_files_auto_skipped_whitelist = 0

        def close(self):
            pass

    monkeypatch.setattr(_cli, "AnnotationReviewController", _StubController)
    _cli.main(["--subject-dir", str(subj), "--rerun-annot-review"])

    assert tracker_exists_at_controller_init.get("value") is False, (
        "tracker must be deleted BEFORE controller init when "
        "--rerun-annot-review is passed")
    err = capsys.readouterr().err
    assert "[rerun]" in err
    assert "reset" in err


def test_cli_rerun_annot_review_noop_message_when_nothing_to_reset(
        monkeypatch, tmp_path, capsys):
    """Fresh subject with --rerun-annot-review -> prints 'nothing to
    reset' hint but STILL launches (doesn't error out). This is the
    ergonomic default: the flag is safe to include unconditionally
    in scripts even for subjects that haven't been reviewed yet."""
    from clean_eeg import annotation_review_cli as _cli

    subj = tmp_path / "R1755J"
    (subj / "clinical_eeg").mkdir(parents=True)

    class _StubController:
        def __init__(self, subject_dir, *, subfolder=None, whitelist_path=None,
                      respect_reviewed_tracker=True, preload_all=False,
                      **_ignored):
            self.num_files_to_review = 0
            self.num_files = 0
            self.num_files_auto_skipped_whitelist = 0

        def close(self):
            pass

    monkeypatch.setattr(_cli, "AnnotationReviewController", _StubController)
    rc = _cli.main(["--subject-dir", str(subj), "--rerun-annot-review"])
    err = capsys.readouterr().err
    assert "nothing to reset" in err
    # Non-fatal: the CLI still launches (falls through to normal path,
    # which the stub short-circuits via num_files_to_review == 0).
    assert rc == 0


def test_cli_auto_locates_standard_whitelist_when_not_specified(
        tmp_path, monkeypatch, capsys):
    """Regression: `annotation-review-eeg` without --whitelist-path
    must auto-locate data/annotation_boilerplate_whitelist.json and
    load it. Previously the TUI ran with an EMPTY whitelist by default,
    silently showing every '*Mark' / boilerplate annotation.

    Approach: patch AnnotationReviewController to capture the
    whitelist_path it was constructed with, run main() through argv,
    assert the path points at the tracked standard whitelist."""
    from clean_eeg import annotation_review_cli as _cli
    from clean_eeg.paths import ANNOTATION_BOILERPLATE_WHITELIST_PATH

    captured: dict = {}

    class _StubController:
        def __init__(self, subject_dir, *, subfolder=None, whitelist_path=None,
                      respect_reviewed_tracker=True, preload_all=False,
                      **_ignored):
            captured["whitelist_path"] = whitelist_path
            self.num_files_to_review = 0    # short-circuits main() early
            self.num_files = 0
            self.num_files_auto_skipped_whitelist = 0

        def close(self):
            pass

    monkeypatch.setattr(_cli, "AnnotationReviewController", _StubController)
    subj = tmp_path / "R1651J"
    (subj / "clinical_eeg").mkdir(parents=True)

    # Args: no --whitelist-path, no --no-whitelist. Must auto-locate.
    _cli.main(["--subject-dir", str(subj), "--subfolder", "clinical_eeg"])

    assert captured.get("whitelist_path") == ANNOTATION_BOILERPLATE_WHITELIST_PATH, (
        f"expected auto-located standard whitelist path, got "
        f"{captured.get('whitelist_path')!r}"
    )
    err = capsys.readouterr().err
    assert "applying whitelist" in err, (
        f"expected loud stderr banner confirming auto-locate: {err!r}")


def test_cli_no_whitelist_flag_disables_auto_load(tmp_path, monkeypatch):
    """--no-whitelist explicitly opts out of the auto-load. Controller
    receives whitelist_path=None (i.e. empty BoilerplateWhitelist)."""
    from clean_eeg import annotation_review_cli as _cli

    captured: dict = {}

    class _StubController:
        def __init__(self, subject_dir, *, subfolder=None, whitelist_path=None,
                      respect_reviewed_tracker=True, preload_all=False,
                      **_ignored):
            captured["whitelist_path"] = whitelist_path
            self.num_files_to_review = 0
            self.num_files = 0
            self.num_files_auto_skipped_whitelist = 0

        def close(self):
            pass

    monkeypatch.setattr(_cli, "AnnotationReviewController", _StubController)
    subj = tmp_path / "R1651J"
    (subj / "clinical_eeg").mkdir(parents=True)

    _cli.main(["--subject-dir", str(subj), "--no-whitelist"])
    assert captured.get("whitelist_path") is None


def test_tui_n_refuses_when_cursor_not_at_last_annotation(tmp_path):
    """Press 'n' before scrolling to the last annotation: the mark
    must be REFUSED. The file must stay UNMARKED in the tracker AND
    the cursor stays put -- neither an advance nor a silent skip.

    Guards against the mistake the operator called out: hitting 'n'
    partway through a file and marking un-scrolled annotations as
    'reviewed' when they were never actually seen."""
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "first"), (1.5, "middle"), (2.5, "last")]})
    controller = AnnotationReviewController(subj)
    assert controller.annotation_cursor == 0

    _drive_tui(controller, "nq")   # 'n' immediately -> should REFUSE

    # File NOT marked reviewed
    from clean_eeg.annotation_review.journal import ReviewedTracker
    entries = ReviewedTracker(subj).read_all()
    assert entries == [], (
        f"'n' before reaching end of file must NOT mark reviewed; "
        f"got tracker entries {entries}")
    # Cursor did NOT advance (n neither marked nor moved to next file)
    assert controller.file_cursor == 0


def test_tui_n_succeeds_after_G_jumps_to_end(tmp_path):
    """Press 'G' (jump to end), then 'n': the gate lets us through
    because the cursor is now at the last annotation. This is the
    intended workaround for 'I've decided to skip the rest without
    scrolling one-by-one' -- a deliberate two-keystroke gesture."""
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "first"), (1.5, "middle"), (2.5, "last")]})
    controller = AnnotationReviewController(subj)

    _drive_tui(controller, "Gnq")

    from clean_eeg.annotation_review.journal import ReviewedTracker
    entries = ReviewedTracker(subj).read_all()
    assert len(entries) == 1, (
        f"'G' then 'n' must mark reviewed; got {entries}")


def test_tui_n_succeeds_after_scrolling_j_to_last(tmp_path):
    """Same as above but reached via arrow-key scrolling (`jjj` walks
    the cursor to position 2 which is the last of the 3 annotations).
    The gate cares about the cursor position, not HOW we got there."""
    subj = _make_subject(tmp_path, files={
        "a.edf": [(0.5, "first"), (1.5, "middle"), (2.5, "last")]})
    controller = AnnotationReviewController(subj)

    _drive_tui(controller, "jjnq")  # 'j'*2 -> cursor at position 2 (last)

    from clean_eeg.annotation_review.journal import ReviewedTracker
    entries = ReviewedTracker(subj).read_all()
    assert len(entries) == 1
