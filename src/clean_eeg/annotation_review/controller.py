"""State machine for the manual annotation review TUI.

Pure Python -- no terminal I/O -- so the entire review workflow can
be unit-tested without a TTY. The prompt_toolkit layer (upcoming
tui.py) is a thin wrapper that turns key events into method calls
on this controller.

Ownership:
    * loads the annotation list per EDF via the fast mmap reader
    * owns the cursor position (which file, which annotation)
    * owns the pending-edits buffer (mirrored to the on-disk journal)
    * owns the reviewed-files tracker
    * owns the boilerplate whitelist (reloadable from disk)

Explicitly OUT of scope for this module:
    * terminal drawing (tui.py)
    * applying edits to EDF headers (apply_edits.py, next commit)
    * launching / arg parsing (annotation_review_cli.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clean_eeg.annotation_boilerplate import (
    BoilerplateWhitelist,
    load_whitelist,
)
from clean_eeg.annotation_reader import Annotation, iter_annotations
from clean_eeg.annotation_review.journal import (
    ReviewedTracker,
    SessionJournal,
)
from clean_eeg.annotation_review.models import EditRecord, ReviewedFile


class PreflightFailure(RuntimeError):
    """Raised by :func:`preflight_subject_for_review` when the subject
    dir isn't ready for manual review. Distinct type so the CLI can
    print a targeted error instead of a generic traceback."""


def preflight_subject_for_review(subject_dir: Path,
                                  subfolder: str = "clinical_eeg"
                                  ) -> list[Path]:
    """Confirm the subject dir is ready for manual annotation review
    and return the list of EDF files to review.

    Gates:
      1. ``<subject_dir>/<subfolder>`` exists (otherwise there's no
         cleaned data to review).
      2. ``<subject_dir>/<subfolder>/deidentify.json`` exists (proves
         the cleaning pipeline finished on this subject; annotations
         should be redacted, so manually reading them for review
         won't surface PHI on the screen).
      3. At least one .edf file exists to review.

    Fails LOUDLY on each gate rather than silently skipping -- the
    operator should never start reviewing something that isn't ready.
    Sidecar ``*_annotations.edf`` files are excluded from the returned
    list; the inline annotations in the main EDFs are canonical.
    """
    subject_dir = Path(subject_dir)
    inner = subject_dir / subfolder
    if not inner.exists():
        raise PreflightFailure(
            f"{subject_dir}: missing '{subfolder}/' subdir; nothing "
            f"to review here.")

    manifest = inner / "deidentify.json"
    if not manifest.exists():
        raise PreflightFailure(
            f"{subject_dir}: no {manifest.name} in {subfolder}/. This "
            f"tool ONLY runs on already-cleaned data (raw annotations "
            f"may still contain PHI). Run clean-batch-eeg on this "
            f"subject first, then re-launch review.")

    edfs = sorted(p for p in inner.rglob("*.edf")
                  if not p.name.endswith("_annotations.edf"))
    if not edfs:
        raise PreflightFailure(
            f"{subject_dir}: no .edf files under {inner}. Nothing "
            f"to review.")
    return edfs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

@dataclass
class DisplayLine:
    """One rendered row for the git-log-style scroll view.

    ``is_current`` is the operator's cursor position. ``is_whitelisted``
    marks annotations the current site's whitelist matches; the TUI
    typically greys these out or hides them entirely depending on a
    toggle. ``is_edited`` marks annotations that have a pending edit
    queued in this session -- shown so the operator sees their own
    in-flight changes distinctly from the on-disk state.
    """
    file_index: int
    annotation_index: int
    file_path: Path
    annotation: Annotation
    is_current: bool
    is_whitelisted: bool
    is_edited: bool


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class AnnotationReviewController:
    """State machine for reviewing one subject's annotations.

    Instantiate with a preflighted subject dir; the constructor
    loads every EDF's annotations via the fast mmap reader (peaks
    at ~KB per file, not GB). Cursor starts at the first
    non-reviewed file's first annotation.

    Whitelist matches are computed with the site code derived from
    the subject dir name (R1XXXY[_M]).
    """

    def __init__(self, subject_dir: Path, *,
                 subfolder: str = "clinical_eeg",
                 whitelist_path: Path | None = None,
                 respect_reviewed_tracker: bool = True):
        self.subject_dir = Path(subject_dir)
        self.subfolder = subfolder
        self.whitelist_path = whitelist_path
        self.respect_reviewed_tracker = respect_reviewed_tracker

        self._edfs: list[Path] = preflight_subject_for_review(
            self.subject_dir, subfolder=subfolder)
        self._tracker = ReviewedTracker(self.subject_dir)
        self._journal = SessionJournal(self.subject_dir)
        self.site_code = _derive_site_code(self.subject_dir.name)
        self._whitelist = self._load_whitelist()

        # File filtering: skip already-reviewed on start unless the
        # operator explicitly asked for a full re-review.
        reviewed_set = (self._tracker.reviewed_paths()
                        if respect_reviewed_tracker else set())
        self._file_indices: list[int] = [
            i for i, p in enumerate(self._edfs)
            if str(p) not in reviewed_set]
        # Load annotations per file. Empty entry means an already-
        # reviewed file (kept in _annotations to keep index alignment
        # with self._edfs, but never displayed).
        self._annotations: list[list[Annotation]] = []
        for i, p in enumerate(self._edfs):
            if i in self._file_indices:
                self._annotations.append(list(iter_annotations(p)))
            else:
                self._annotations.append([])

        self.file_cursor: int = (self._file_indices[0]
                                 if self._file_indices else 0)
        self.annotation_cursor: int = 0

        # Pending edits keyed by (file_index, annotation_index) so a
        # second edit to the same annotation overwrites the first --
        # the operator's most-recent intent wins.
        self._pending: dict[tuple[int, int], EditRecord] = {}
        # Re-hydrate from journal on restart (crash recovery).
        for e in self._journal.read_all():
            key = self._locate_edit_in_current_state(e)
            if key is not None:
                self._pending[key] = e

    # ---- whitelist ----

    def _load_whitelist(self) -> BoilerplateWhitelist:
        if self.whitelist_path is None:
            return BoilerplateWhitelist()
        return load_whitelist(self.whitelist_path)

    def reload_whitelist(self) -> None:
        """Reread the whitelist file from disk. Used by the 'r' key
        after the operator edits the JSON directly (their
        immediately-in-effect requirement)."""
        self._whitelist = self._load_whitelist()

    def is_whitelisted(self, ann: Annotation) -> bool:
        return self._whitelist.matches(ann.text, site_code=self.site_code)

    # ---- introspection ----

    @property
    def num_files(self) -> int:
        return len(self._edfs)

    @property
    def num_files_to_review(self) -> int:
        return len(self._file_indices)

    def annotations_in_current_file(self) -> list[Annotation]:
        return self._annotations[self.file_cursor]

    def current_annotation(self) -> Annotation | None:
        anns = self.annotations_in_current_file()
        if not anns:
            return None
        return anns[min(self.annotation_cursor, len(anns) - 1)]

    def current_file(self) -> Path:
        return self._edfs[self.file_cursor]

    def pending_edits(self) -> list[EditRecord]:
        """All queued edits across the whole subject, ordered by
        (file_index, annotation_index). Used by the approval gate."""
        return [self._pending[k] for k in sorted(self._pending)]

    # ---- cursor movement ----

    def move_cursor(self, delta: int) -> None:
        """Move the annotation cursor by ``delta`` within the current
        file. Clamped -- moving past the end doesn't wrap or advance
        to the next file (the operator uses next_file() explicitly)."""
        n = len(self.annotations_in_current_file())
        if n == 0:
            self.annotation_cursor = 0
            return
        self.annotation_cursor = max(0, min(n - 1,
                                             self.annotation_cursor + delta))

    def jump_to_start(self) -> None:
        self.annotation_cursor = 0

    def jump_to_end(self) -> None:
        n = len(self.annotations_in_current_file())
        self.annotation_cursor = max(0, n - 1)

    def next_file(self) -> bool:
        """Advance to the next reviewable file. Returns True if the
        cursor moved, False if already on the last reviewable file."""
        pos = self._file_indices.index(self.file_cursor) \
            if self.file_cursor in self._file_indices else -1
        if pos + 1 >= len(self._file_indices):
            return False
        self.file_cursor = self._file_indices[pos + 1]
        self.annotation_cursor = 0
        return True

    def prev_file(self) -> bool:
        pos = self._file_indices.index(self.file_cursor) \
            if self.file_cursor in self._file_indices else 0
        if pos <= 0:
            return False
        self.file_cursor = self._file_indices[pos - 1]
        self.annotation_cursor = 0
        return True

    def on_last_annotation_of_file(self) -> bool:
        anns = self.annotations_in_current_file()
        return bool(anns) and self.annotation_cursor >= len(anns) - 1

    # ---- editing ----

    def queue_edit(self, new_text: str) -> EditRecord | None:
        """Persist an edit for the current annotation. Overwrites
        any previous pending edit for the same annotation. Returns
        the created record, or None if there's no current annotation
        (empty file). Same-text 'edits' still record so the operator
        can see confirmed no-ops in the approval gate."""
        ann = self.current_annotation()
        if ann is None:
            return None
        record = EditRecord.new(
            file_path=str(self.current_file()),
            record_index=ann.record_index,
            byte_offset_in_record=ann.byte_offset_in_record,
            onset_s=ann.onset_s,
            orig_text=ann.text, new_text=new_text)
        key = (self.file_cursor, self.annotation_cursor)
        self._pending[key] = record
        self._journal.append(record)
        return record

    def is_current_edited(self) -> bool:
        return (self.file_cursor, self.annotation_cursor) in self._pending

    # ---- rendering ----

    def visible_lines(self, context: int = 15) -> list[DisplayLine]:
        """Return the annotations to render for a git-log-style scroll
        view: ``context`` lines below the current one (plus the
        current line itself). Callers filter or grey-out whitelisted
        entries based on operator preference.
        """
        anns = self.annotations_in_current_file()
        if not anns:
            return []
        end = min(len(anns), self.annotation_cursor + context + 1)
        out: list[DisplayLine] = []
        for i in range(self.annotation_cursor, end):
            a = anns[i]
            out.append(DisplayLine(
                file_index=self.file_cursor,
                annotation_index=i,
                file_path=self.current_file(),
                annotation=a,
                is_current=(i == self.annotation_cursor),
                is_whitelisted=self.is_whitelisted(a),
                is_edited=(self.file_cursor, i) in self._pending,
            ))
        return out

    # ---- reviewed-file tracker ----

    def mark_current_file_reviewed(self) -> None:
        """Append a ReviewedFile entry for the current file to the
        tracker. Idempotent from the tracker's perspective (it stores
        every mark, but reviewed_paths() dedupes)."""
        anns = self.annotations_in_current_file()
        n_edited = sum(1 for k in self._pending
                       if k[0] == self.file_cursor)
        self._tracker.mark_reviewed(ReviewedFile.new(
            file_path=self.current_file(),
            n_annotations=len(anns),
            n_edited=n_edited))

    # ---- close-out ----

    def rotate_applied(self) -> Path | None:
        return self._journal.rotate_applied()

    def rotate_discarded(self) -> Path | None:
        return self._journal.rotate_discarded()

    def close(self) -> None:
        self._journal.close()

    # ---- crash-recovery helper ----

    def _locate_edit_in_current_state(self, edit: EditRecord
                                       ) -> tuple[int, int] | None:
        """Given a journal entry from a prior session, find the
        (file_index, annotation_index) it now maps to. Match by
        file_path + onset + orig_text so a benign re-parse can still
        line them up. Returns None if no plausible match (e.g. file
        was renamed / annotation removed by another tool) so the
        stale entry is dropped rather than corrupting current state.
        """
        for fi, p in enumerate(self._edfs):
            if str(p) != edit.file_path:
                continue
            for ai, ann in enumerate(self._annotations[fi]):
                if (ann.onset_s == edit.onset_s
                        and ann.text == edit.orig_text):
                    return (fi, ai)
        return None


# ---------------------------------------------------------------------------
# Small utility, kept out of the class for testability
# ---------------------------------------------------------------------------

def _derive_site_code(subject_dir_name: str) -> str | None:
    """R1XXXY[_M] -> Y (single site letter). None if the folder name
    doesn't match the R-code shape -- callers fall back to shared-
    whitelist-only matching."""
    import re
    m = re.match(r"^R1\d{3}([ACDEFHJMNPST])(?:_\d+)?$", subject_dir_name)
    return m.group(1) if m else None
