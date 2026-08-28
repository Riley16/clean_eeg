"""prompt_toolkit shell wrapping :class:`AnnotationReviewController`.

Thin layer: turns key events into method calls on the controller.
The controller owns all state -- this module owns only screen
rendering and key bindings, so the entire review UX can be
unit-tested via the controller without a TTY.

Key bindings (REVIEW mode):
    up / k / ↑         cursor up one annotation
    down / j / ↓       cursor down one annotation
    PgUp / PgDn        cursor by ten annotations
    g                  jump to first annotation of current file
    G                  jump to last annotation of current file
    n                  advance to next reviewable file (marks
                       current file as reviewed)
    p                  back to previous reviewable file
    e                  enter EDIT mode on current annotation
    w                  append current annotation text (literal, re-
                       escaped) to the per-site whitelist and reload
    r                  reload whitelist from disk
    ?                  help overlay
    q                  quit (session pending edits stay journaled;
                       apply pass runs after quit)

Key bindings (EDIT mode):
    all printable keys / arrows / home / end / backspace / del
                       standard line editing on the annotation text
    Enter              save edit + return to REVIEW mode
    Esc                abort edit (y/n confirm; y returns to REVIEW
                       without saving; n stays in EDIT)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import (
    BufferControl,
    FormattedTextControl,
)
from prompt_toolkit.layout.dimension import Dimension

from clean_eeg.annotation_review.controller import (
    AnnotationReviewController,
)


# ---------------------------------------------------------------------------
# UI state (mode toggles) kept separate from the controller
# ---------------------------------------------------------------------------

@dataclass
class UIState:
    """Ephemeral UI-only state. The controller owns everything
    persistent (cursor, pending edits, tracker). Values in here
    describe what MODE the UI is currently in and any transient
    input buffer."""
    mode: str = "review"          # 'review' | 'edit' | 'confirm_abort' |
                                  # 'help' | 'quit_prompt'
    status_message: str = ""      # transient one-line message
    quit_requested: bool = False  # set true when the operator confirms
                                  # quit; app.exit() runs on next tick


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_scroll(controller: AnnotationReviewController) -> FormattedText:
    """git-log style: current annotation + ~15 below, tagged with
    visual markers for current/edited/whitelisted status."""
    lines = controller.visible_lines(context=15)
    if not lines:
        return FormattedText([("", "  (no annotations to review in this "
                                    "file -- press 'n' for next file)\n")])
    frags: list[tuple[str, str]] = []
    for line in lines:
        # Marker column: * = current, + = edited, ~ = whitelisted,
        # blank = normal
        markers = []
        if line.is_current:
            markers.append("*")
        if line.is_edited:
            markers.append("+")
        if line.is_whitelisted:
            markers.append("~")
        marker = "".join(markers) or " "
        marker = f"{marker:<3s}"

        onset = f"{line.annotation.onset_s:>8.2f}s"
        # display_text reflects the pending edit's new_text when
        # is_edited, else the raw on-disk annotation. The renderer must
        # NOT read line.annotation.text directly -- that would show the
        # pre-edit value and mask the operator's in-flight change.
        text = line.display_text
        # Truncate long texts to keep the scroll compact; full text
        # visible in EDIT mode
        if len(text) > 200:
            text = text[:197] + "..."

        # Style: current = reverse; edited = green; whitelisted = grey
        # Multiple can stack (current + edited = reverse over green;
        # prompt_toolkit shows the first).
        if line.is_current:
            style = "reverse bold"
        elif line.is_edited:
            style = "fg:ansigreen"
        elif line.is_whitelisted:
            style = "fg:ansibrightblack"
        else:
            style = ""

        frags.append((style,
                       f"{marker}{onset}   {text}\n"))
    return FormattedText(frags)


def _render_status(controller: AnnotationReviewController,
                    ui: UIState) -> FormattedText:
    """Bottom status bar: subject / file M-of-N / annotation X-of-Y /
    mode / transient message."""
    anns = controller.annotations_in_current_file()
    n_ann = len(anns)
    n_pending = len(controller.pending_edits())
    subject = controller.subject_dir.name
    file_name = controller.current_file().name
    # File-progress: how many reviewable files done vs total
    reviewable_files = controller.num_files_to_review
    subject_pending_msg = (f"  [{n_pending} pending edit(s)]"
                            if n_pending else "")

    top = (f" {subject}  |  {file_name}  |  "
           f"ann {controller.annotation_cursor + 1}/{n_ann}  |  "
           f"reviewable files: {reviewable_files}"
           f"{subject_pending_msg}\n")
    bottom = (f" mode: {ui.mode}   ?=help  q=quit  "
              f"e=edit  n=next-file  w=whitelist  r=reload-wl\n")
    msg = f" {ui.status_message}\n" if ui.status_message else ""
    return FormattedText([("reverse", top),
                            ("", bottom),
                            ("fg:ansiyellow", msg)])


def _render_edit_prompt(edit_buf: Buffer,
                         controller: AnnotationReviewController
                         ) -> FormattedText:
    ann = controller.current_annotation()
    orig = ann.text if ann is not None else ""
    return FormattedText([
        ("fg:ansiyellow bold", " EDIT MODE — Enter to save, Esc to abort\n"),
        ("", f" original:  {orig}\n"),
        ("", " new:       "),
    ])


def _render_help(_controller: AnnotationReviewController) -> FormattedText:
    return FormattedText([
        ("bold", " Annotation Review — key reference\n\n"),
        ("", "  REVIEW MODE\n"),
        ("", "    up / down / j / k       cursor by 1\n"),
        ("", "    PgUp / PgDn             cursor by 10\n"),
        ("", "    g / G                   first / last of current file\n"),
        ("", "    n / p                   next / prev file (marks current reviewed)\n"),
        ("", "    e                       edit current annotation\n"),
        ("", "    w                       whitelist current annotation text\n"),
        ("", "    r                       reload whitelist from disk\n"),
        ("", "    q                       quit\n"),
        ("", "    ?                       toggle this help\n\n"),
        ("", "  EDIT MODE\n"),
        ("", "    Enter                   save edit\n"),
        ("", "    Esc                     abort edit (y/n confirm)\n\n"),
        ("fg:ansiyellow", " press any key to close\n"),
    ])


# ---------------------------------------------------------------------------
# Whitelist editing
# ---------------------------------------------------------------------------

def append_annotation_to_whitelist(whitelist_path: Path,
                                    text: str,
                                    site_code: str | None) -> None:
    """Append the operator's current annotation text as a re.escape'd
    literal regex to the per-site whitelist JSON, atomically.

    Uses ``re.escape`` so special characters (periods, backslashes,
    etc.) in the annotation don't accidentally match unrelated
    annotations. Operator can hand-edit the file later to relax the
    pattern into a real regex if they want.
    """
    import json
    import os
    import tempfile

    if not whitelist_path.exists():
        data = {"shared": [], "per_site": {}}
    else:
        data = json.loads(whitelist_path.read_text())
    data.setdefault("shared", [])
    data.setdefault("per_site", {})

    escaped = re.escape(text)
    bucket = "per_site"
    key = site_code
    if not key:
        bucket = "shared"

    if bucket == "shared":
        if escaped not in data["shared"]:
            data["shared"].append(escaped)
    else:
        data["per_site"].setdefault(key, [])
        if escaped not in data["per_site"][key]:
            data["per_site"][key].append(escaped)

    # Atomic write: temp + rename so a mid-write crash doesn't leave
    # a truncated JSON that the pipeline would fail to load.
    fd, tmp_path = tempfile.mkstemp(
        prefix=whitelist_path.name + ".", suffix=".tmp",
        dir=str(whitelist_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(whitelist_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Application builder
# ---------------------------------------------------------------------------

def build_review_app(controller: AnnotationReviewController,
                      *,
                      on_bell: Callable[[], None] | None = None,
                      ) -> Application:
    """Build the prompt_toolkit ``Application`` bound to ``controller``.

    ``on_bell``: optional callback fired when the operator finishes
    the last annotation of a file. Defaults to a no-op; the TUI also
    emits a visible banner via the status message so a silent bell
    isn't the only signal.
    """
    on_bell = on_bell or (lambda: None)
    ui = UIState()
    edit_buf = Buffer(multiline=False)

    kb = KeyBindings()

    # --- global (any mode) ---
    @kb.add("c-c")
    def _(event):
        # Ctrl-C treated as quit
        ui.quit_requested = True
        event.app.exit()

    # --- REVIEW mode key bindings ---
    in_review = Condition(lambda: ui.mode == "review")

    @kb.add("up", filter=in_review)
    @kb.add("k", filter=in_review)
    def _(event):
        controller.move_cursor(-1)
        ui.status_message = ""

    @kb.add("down", filter=in_review)
    @kb.add("j", filter=in_review)
    def _(event):
        controller.move_cursor(+1)
        ui.status_message = ""
        if controller.on_last_annotation_of_file():
            ui.status_message = ("end of file -- press 'n' to advance "
                                  "(marks file reviewed)")

    @kb.add("pageup", filter=in_review)
    def _(event):
        controller.move_cursor(-10)
        ui.status_message = ""

    @kb.add("pagedown", filter=in_review)
    def _(event):
        controller.move_cursor(+10)
        ui.status_message = ""

    @kb.add("g", filter=in_review)
    def _(event):
        controller.jump_to_start()
        ui.status_message = ""

    @kb.add("G", filter=in_review)
    def _(event):
        controller.jump_to_end()
        ui.status_message = ""

    @kb.add("n", filter=in_review)
    def _(event):
        controller.mark_current_file_reviewed()
        moved = controller.next_file()
        if moved:
            ui.status_message = f"advanced to {controller.current_file().name}"
            on_bell()
        else:
            ui.status_message = ("no more reviewable files -- press 'q' "
                                  "to quit and apply pending edits")
            on_bell()

    @kb.add("p", filter=in_review)
    def _(event):
        moved = controller.prev_file()
        ui.status_message = (
            f"back to {controller.current_file().name}" if moved
            else "already at first reviewable file")

    @kb.add("e", filter=in_review)
    def _(event):
        # Pre-fill the edit buffer with what the operator SEES in the
        # scroll view, not the raw on-disk text. If a pending edit
        # exists on this annotation, current_display_text returns its
        # new_text so re-editing lets the operator build on the last
        # change instead of starting over from the original.
        current_text = controller.current_display_text()
        if current_text is None:
            ui.status_message = "no annotation to edit"
            return
        edit_buf.text = current_text
        edit_buf.cursor_position = len(edit_buf.text)
        ui.mode = "edit"
        ui.status_message = ""
        event.app.layout.focus(edit_control)

    @kb.add("w", filter=in_review)
    def _(event):
        ann = controller.current_annotation()
        if ann is None:
            ui.status_message = "no annotation to whitelist"
            return
        wp = controller.whitelist_path
        if wp is None:
            ui.status_message = ("no --whitelist-path configured; can't "
                                  "append")
            return
        append_annotation_to_whitelist(wp, ann.text, controller.site_code)
        controller.reload_whitelist()
        ui.status_message = f"whitelisted: {ann.text[:60]!r}"

    @kb.add("r", filter=in_review)
    def _(event):
        controller.reload_whitelist()
        ui.status_message = "whitelist reloaded from disk"

    @kb.add("q", filter=in_review)
    def _(event):
        ui.quit_requested = True
        event.app.exit()

    @kb.add("?", filter=in_review)
    def _(event):
        ui.mode = "help"

    # --- HELP mode: any key returns to review ---
    in_help = Condition(lambda: ui.mode == "help")

    @kb.add("<any>", filter=in_help)
    def _(event):
        ui.mode = "review"

    # --- EDIT mode key bindings ---
    in_edit = Condition(lambda: ui.mode == "edit")

    @kb.add("enter", filter=in_edit)
    def _(event):
        new_text = edit_buf.text
        record = controller.queue_edit(new_text)
        if record is None:
            ui.status_message = "edit failed: no current annotation"
        else:
            ui.status_message = f"saved: {record.new_text[:60]!r}"
        ui.mode = "review"
        event.app.layout.focus(scroll_window)

    @kb.add("escape", filter=in_edit)
    def _(event):
        ui.mode = "confirm_abort"
        ui.status_message = ""

    # --- CONFIRM_ABORT sub-mode: y/n prompt on Esc-in-edit ---
    in_confirm_abort = Condition(lambda: ui.mode == "confirm_abort")

    @kb.add("y", filter=in_confirm_abort)
    @kb.add("Y", filter=in_confirm_abort)
    def _(event):
        ui.mode = "review"
        ui.status_message = "edit aborted"
        event.app.layout.focus(scroll_window)

    @kb.add("n", filter=in_confirm_abort)
    @kb.add("N", filter=in_confirm_abort)
    def _(event):
        ui.mode = "edit"
        ui.status_message = ""

    # ---- layout ----
    scroll_control = FormattedTextControl(
        lambda: _render_scroll(controller))
    scroll_window = Window(content=scroll_control, wrap_lines=True,
                            always_hide_cursor=True)

    edit_control = BufferControl(buffer=edit_buf)
    edit_prompt_window = Window(
        content=FormattedTextControl(
            lambda: _render_edit_prompt(edit_buf, controller)),
        height=Dimension(min=3, max=3, preferred=3),
        wrap_lines=True)
    edit_input_window = Window(
        content=edit_control,
        height=Dimension(min=1, max=1, preferred=1))

    confirm_window = Window(content=FormattedTextControl(
        lambda: FormattedText([
            ("fg:ansiyellow bold",
             " Discard edit? y = discard, n = keep editing\n")])),
        height=Dimension(min=1, max=1, preferred=1))

    help_window = Window(content=FormattedTextControl(
        lambda: _render_help(controller)), wrap_lines=True)

    status_window = Window(content=FormattedTextControl(
        lambda: _render_status(controller, ui)),
        height=Dimension(min=3, max=4))

    def _current_body():
        # HSplit is fixed at build time; use ConditionalContainer to
        # swap the body window based on mode.
        pass  # (placeholder -- ConditionalContainer wired below)

    from prompt_toolkit.layout.containers import ConditionalContainer

    body = HSplit([
        ConditionalContainer(scroll_window,
                              filter=Condition(
                                  lambda: ui.mode == "review")),
        ConditionalContainer(HSplit([edit_prompt_window,
                                      edit_input_window]),
                              filter=Condition(
                                  lambda: ui.mode == "edit")),
        ConditionalContainer(confirm_window,
                              filter=Condition(
                                  lambda: ui.mode == "confirm_abort")),
        ConditionalContainer(help_window,
                              filter=Condition(
                                  lambda: ui.mode == "help")),
        status_window,
    ])
    layout = Layout(body, focused_element=scroll_window)
    app = Application(layout=layout, key_bindings=kb,
                       full_screen=True, mouse_support=False)
    # Expose the UI state so callers (tests, approval gate) can
    # inspect it after the app exits.
    app._review_ui_state = ui  # type: ignore[attr-defined]
    return app
