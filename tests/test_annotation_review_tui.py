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


def test_tui_n_marks_current_file_reviewed(tmp_path):
    """Press 'n': current file's entry lands in
    .annotation_reviewed_tracker on disk. Regression guard against
    'n' advancing without persisting the mark (would break the
    restart-skips-reviewed-files behavior)."""
    subj = _make_subject(tmp_path, files={"a.edf": [(0.5, "x")]})
    controller = AnnotationReviewController(subj)

    _drive_tui(controller, "nq")

    from clean_eeg.annotation_review.journal import ReviewedTracker
    entries = ReviewedTracker(subj).read_all()
    assert len(entries) == 1
    assert entries[0].file_path.endswith("a.edf")
