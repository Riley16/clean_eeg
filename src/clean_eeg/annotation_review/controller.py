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

import re
from concurrent.futures import Future, ThreadPoolExecutor
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


PREFETCH_LOOKAHEAD = 2   # how many files ahead of cursor to pre-load


def _prefetch_one(edf_path: Path) -> list[Annotation]:
    """Worker function for the prefetch thread pool. Errors are
    swallowed to an empty list -- the caller decides how to surface
    the failure (typically: leave the cached entry empty so the
    operator can navigate past the bad file without hanging on it).
    """
    try:
        return list(iter_annotations(edf_path))
    except Exception:
        return []


class PreflightFailure(RuntimeError):
    """Raised by :func:`preflight_subject_for_review` when the subject
    dir isn't ready for manual review. Distinct type so the CLI can
    print a targeted error instead of a generic traceback."""


ANNOTATION_SIDECAR_SUFFIX = "_annotations.edf"


def _pick_annotation_carrier(main_edf: Path) -> Path:
    """Return the file that actually carries this recording's redacted
    annotations. In-place cleaning ([clean_subject_eeg.py:498-513](
    ../clean_subject_eeg.py#L498-L513)) writes annotations into a
    ``<base>_annotations.edf`` sidecar (~KB) and ZEROES the main EDF's
    annotation channel, so on in-place output the main EDF is empty
    and the sidecar is canonical. Rewrite mode leaves annotations
    inline in the main EDF (no sidecar exists). Prefer the sidecar
    when present; else fall back to the main file.

    Reading the sidecar in in-place mode also drops per-file review
    I/O from GB to KB — the entire file is annotation channel plus
    a few header bytes, so on network storage (Box FS, NFS) it's
    orders of magnitude faster than mmap-scanning the full recording.
    """
    sidecar = main_edf.parent / (main_edf.stem + ANNOTATION_SIDECAR_SUFFIX)
    return sidecar if sidecar.exists() else main_edf


def preflight_subject_for_review(subject_dir: Path,
                                  subfolder: str = "clinical_eeg"
                                  ) -> list[Path]:
    """Confirm the subject dir is ready for manual annotation review
    and return the list of files to review (one per recording).

    Gates:
      1. ``<subject_dir>/<subfolder>`` exists (otherwise there's no
         cleaned data to review).
      2. ``<subject_dir>/<subfolder>/deidentify.json`` exists (proves
         the cleaning pipeline finished on this subject; annotations
         should be redacted, so manually reading them for review
         won't surface PHI on the screen).
      3. At least one recording (non-sidecar .edf) exists to review.

    Fails LOUDLY on each gate rather than silently skipping -- the
    operator should never start reviewing something that isn't ready.

    For each recording, returns the annotation carrier -- the sidecar
    ``<base>_annotations.edf`` if it exists (in-place cleaning
    output), else the main EDF (rewrite mode). Downstream code
    (``iter_annotations``, ``apply_pending_edits``) targets whatever
    path is returned here, so edits land in the right file.
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

    main_edfs = sorted(p for p in inner.rglob("*.edf")
                       if not p.name.endswith(ANNOTATION_SIDECAR_SUFFIX))
    if not main_edfs:
        raise PreflightFailure(
            f"{subject_dir}: no .edf files under {inner}. Nothing "
            f"to review.")
    return [_pick_annotation_carrier(p) for p in main_edfs]


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

    ``display_text`` is what the renderer should print. When an edit is
    pending, this is the edit's ``new_text``; otherwise it's the raw
    annotation text. Kept as a separate field (rather than making the
    renderer look up ``_pending`` itself) so the render is a pure
    function of the DisplayLine list -- easy to snapshot / test.
    """
    file_index: int
    annotation_index: int
    file_path: Path
    annotation: Annotation
    is_current: bool
    is_whitelisted: bool
    is_edited: bool
    display_text: str


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
                 respect_reviewed_tracker: bool = True,
                 external_prefetch_paths: list[Path] | None = None,
                 preload_all: bool = False,
                 hide_whitelisted: bool = True):
        """``external_prefetch_paths``: optional additional EDF paths
        (e.g. the FIRST N files of the NEXT subject) to warm the
        prefetch queue after this subject's own files are exhausted.
        Lets a caller iterating parent-dir subjects avoid the
        subject-transition pause. None -> intra-subject prefetch only.

        ``preload_all``: when True, load EVERY reviewable file's
        annotations at __init__ time with a tqdm progress bar (one-
        time startup cost). Files whose annotations are entirely
        matched by the whitelist or delete bucket are auto-marked
        reviewed and dropped from the reviewable list -- the
        operator never scrolls into a file with nothing to look at.
        Default (False) uses the lazy per-file load + 2-worker
        prefetch pattern, which is better when startup latency
        matters more than steady-state throughput.
        """
        self.subject_dir = Path(subject_dir)
        self.subfolder = subfolder
        self.whitelist_path = whitelist_path
        self.respect_reviewed_tracker = respect_reviewed_tracker
        # hide_whitelisted=True (default): whitelisted annotations are
        # removed from the scroll view AND skipped by cursor navigation
        # -- the operator only sees + interacts with annotations that
        # actually need review. False: whitelisted annotations stay in
        # the view greyed out (previous default). Toggled at CLI via
        # --show-whitelisted.
        self.hide_whitelisted = hide_whitelisted
        # Populated by _preload_all_and_drop_empty: count of files
        # auto-marked reviewed because 100% of their annotations matched
        # the whitelist / delete bucket. Distinct from files already in
        # the reviewed tracker from a prior session -- the CLI uses this
        # to give the operator an accurate "why is there nothing to do"
        # message.
        self.num_files_auto_skipped_whitelist = 0

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

        # ---- Lazy per-file annotation loading + background prefetch ----
        # Previously loaded ALL files' annotations synchronously in
        # the constructor -- a 50-file subject on /oceanus took ~2 min
        # of startup pause. Now: load the current file synchronously
        # (blocking is fine, operator hasn't seen anything yet), and
        # let a 2-worker thread pool warm the next PREFETCH_LOOKAHEAD
        # files in the background. Moving between files hits the
        # cache and returns instantly.
        #
        # ``_annotations_cache``: file_index -> list[Annotation],
        # populated on demand.
        # ``_prefetch_futures``: file_index -> Future waiting to
        # populate the cache. Consulted before triggering a fresh
        # sync load.
        # ``_prefetch_pool``: 2 workers -- fits the "next 2 files"
        # ask. Larger pool = more concurrent I/O but no meaningful
        # UX win.
        self._annotations_cache: dict[int, list[Annotation]] = {}
        self._prefetch_futures: dict[int, "Future[list[Annotation]]"] = {}
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=PREFETCH_LOOKAHEAD,
            thread_name_prefix="ann-prefetch")
        # Already-reviewed files: cache empty list (never shown, but
        # keeps annotations_by_file_index() total-consistent).
        for i, _ in enumerate(self._edfs):
            if i not in self._file_indices:
                self._annotations_cache[i] = []

        # External prefetch queue (e.g. next-subject files). Loaded
        # via a separate _external_prefetch_futures so they don't
        # collide with per-subject index keys.
        self._external_prefetch_paths = list(external_prefetch_paths or [])
        self._external_prefetch_futures: dict[str, "Future[list[Annotation]]"] = {}

        self.file_cursor: int = (self._file_indices[0]
                                 if self._file_indices else 0)
        self.annotation_cursor: int = 0

        # Pending edits keyed by (file_index, annotation_index) so a
        # second edit to the same annotation overwrites the first --
        # the operator's most-recent intent wins.
        self._pending: dict[tuple[int, int], EditRecord] = {}
        # Re-hydrate from journal on restart (crash recovery). Requires
        # some file annotations loaded -- do sync loads for any files
        # referenced by the journal.
        for e in self._journal.read_all():
            key = self._locate_edit_in_current_state(e)
            if key is not None:
                self._pending[key] = e

        # Preload-all path: eagerly load every reviewable file
        # (with tqdm progress) so subsequent navigation is instant
        # AND we can auto-drop files with nothing left to review.
        if preload_all:
            self._preload_all_and_drop_empty()

        # Warm the prefetch queue for the initial cursor position.
        self._schedule_prefetch()

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
        return self._load_annotations_for_index(self.file_cursor)

    # ---- lazy loading + background prefetch ----

    def _load_annotations_for_index(self, file_index: int
                                     ) -> list[Annotation]:
        """Get annotations for ``file_index``. If a prefetch future is
        already in flight for this file, WAIT for it (don't spawn a
        duplicate load). If neither cache nor future is available,
        load synchronously here.

        Always schedules prefetch for the next PREFETCH_LOOKAHEAD
        files after this call returns, so subsequent moves are
        instantaneous.
        """
        if file_index in self._annotations_cache:
            self._schedule_prefetch()
            return self._annotations_cache[file_index]
        if file_index in self._prefetch_futures:
            # Prefetch already started -- wait for it rather than
            # racing a duplicate load. .result() will re-raise any
            # exception; we swallow to an empty list so navigation
            # still works (a bad file's annotations are gone but the
            # cursor isn't stuck).
            try:
                anns = self._prefetch_futures[file_index].result()
            except Exception:
                anns = []
            self._annotations_cache[file_index] = anns
            del self._prefetch_futures[file_index]
            self._schedule_prefetch()
            return anns
        # Cold: load synchronously here (blocks current op).
        try:
            anns = list(iter_annotations(self._edfs[file_index]))
        except Exception:
            anns = []
        self._annotations_cache[file_index] = anns
        self._schedule_prefetch()
        return anns

    # ---- eager preload with tqdm + auto-drop empty files ----

    def _preload_all_and_drop_empty(self) -> None:
        """Eagerly load every reviewable file's annotations with a
        tqdm progress bar. Files whose annotations are entirely
        matched by the whitelist or delete bucket are auto-marked
        reviewed and dropped from ``self._file_indices`` so the
        operator never scrolls into a file with nothing to look at.
        """
        from tqdm import tqdm

        to_load = list(self._file_indices)
        # Load each file synchronously; the pool would only add
        # scheduling overhead here since we're going to wait for
        # everything anyway.
        for fi in tqdm(to_load, desc=f"preloading {self.subject_dir.name}",
                        unit="file", leave=False):
            if fi in self._annotations_cache:
                continue
            self._annotations_cache[fi] = _prefetch_one(self._edfs[fi])

        # After all loaded, drop files whose EVERY annotation is
        # whitelisted or delete-marked. Empty files also drop.
        keep: list[int] = []
        dropped_paths: list[Path] = []
        for fi in self._file_indices:
            anns = self._annotations_cache.get(fi, [])
            non_boilerplate = [
                a for a in anns
                if a.text.strip()
                and not self._whitelist.matches(
                    a.text, site_code=self.site_code)
                and not self._whitelist.matches_delete(
                    a.text, site_code=self.site_code)]
            if non_boilerplate:
                keep.append(fi)
            else:
                # Auto-record as reviewed. n_edited=0 because we're
                # skipping without visiting.
                self._tracker.mark_reviewed(ReviewedFile.new(
                    file_path=self._edfs[fi],
                    n_annotations=len(anns),
                    n_edited=0))
                dropped_paths.append(self._edfs[fi])
        self._file_indices = keep
        if self._file_indices:
            self.file_cursor = self._file_indices[0]
        self.annotation_cursor = 0
        self.num_files_auto_skipped_whitelist = len(dropped_paths)

        if dropped_paths:
            print(f"[review] auto-skipped {len(dropped_paths)} file(s) "
                  f"with no non-boilerplate annotations "
                  f"(all whitelisted; marked reviewed).")

    def _schedule_prefetch(self) -> None:
        """Ensure the next PREFETCH_LOOKAHEAD unreviewed files after
        the cursor are being loaded in the background. Idempotent:
        already-cached and already-scheduled files are left alone.
        When intra-subject files are exhausted, warms the external
        prefetch queue (cross-subject) instead.
        """
        # Find the next PREFETCH_LOOKAHEAD reviewable file indices
        # after the current cursor (not INCLUDING the cursor itself
        # -- that one is either cached or being loaded right now).
        try:
            cursor_pos = self._file_indices.index(self.file_cursor)
        except ValueError:
            cursor_pos = -1
        next_indices = self._file_indices[cursor_pos + 1:
                                            cursor_pos + 1
                                            + PREFETCH_LOOKAHEAD]
        for fi in next_indices:
            if (fi in self._annotations_cache
                    or fi in self._prefetch_futures):
                continue
            self._prefetch_futures[fi] = self._prefetch_pool.submit(
                _prefetch_one, self._edfs[fi])

        # Cross-subject warmup: if our own file queue is exhausted
        # (fewer than PREFETCH_LOOKAHEAD files remaining), fill the
        # rest from the external queue.
        room = PREFETCH_LOOKAHEAD - len(next_indices)
        for path in self._external_prefetch_paths[:room]:
            key = str(path)
            if key in self._external_prefetch_futures:
                continue
            self._external_prefetch_futures[key] = \
                self._prefetch_pool.submit(_prefetch_one, path)

    def annotations_for_external_prefetch_path(self, path: Path
                                                 ) -> list[Annotation]:
        """Retrieve pre-warmed annotations for a cross-subject file
        listed in ``external_prefetch_paths``. If the future finished,
        returns its result; if still running, blocks. Used by a
        parent-dir iterator (e.g. an upcoming multi-subject CLI) to
        hand off the pre-warmed cache to the next subject's
        controller without re-reading."""
        key = str(path)
        fut = self._external_prefetch_futures.get(key)
        if fut is None:
            return list(iter_annotations(path))
        try:
            return fut.result()
        except Exception:
            return []

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

    def _visible_indices_in_current_file(self) -> list[int]:
        """Indices into ``annotations_in_current_file()`` that the
        cursor is allowed to land on. When ``hide_whitelisted`` is
        True (default), whitelisted annotations are excluded so the
        operator's cursor never rests on an invisible line. When
        False, every annotation index is visible (matches the
        pre-hide-mode behaviour).

        Falls back to the full range when no annotation would be
        visible under the filter -- keeps navigation methods
        well-defined on 100%-whitelisted files (though such files
        are auto-skipped by --preload-all in practice).
        """
        anns = self.annotations_in_current_file()
        if not self.hide_whitelisted:
            return list(range(len(anns)))
        visible = [i for i, a in enumerate(anns)
                   if not self.is_whitelisted(a)]
        # Safety: on an all-whitelisted file the operator might still
        # navigate in (e.g. without --preload-all). Fall back to full
        # index range so move_cursor / jump_to_end don't lock up.
        return visible if visible else list(range(len(anns)))

    def move_cursor(self, delta: int) -> None:
        """Move the annotation cursor by ``delta`` within the current
        file's VISIBLE annotations (whitelisted lines are skipped when
        ``hide_whitelisted`` is True). Clamped -- moving past the end
        doesn't wrap or advance to the next file."""
        anns = self.annotations_in_current_file()
        if not anns:
            self.annotation_cursor = 0
            return
        visible = self._visible_indices_in_current_file()
        # Find the visible-slot the cursor is currently on (or nearest
        # after it if the cursor happens to be on a whitelisted index).
        current_slot = 0
        for i, idx in enumerate(visible):
            if idx >= self.annotation_cursor:
                current_slot = i
                break
        else:
            current_slot = len(visible) - 1
        new_slot = max(0, min(len(visible) - 1, current_slot + delta))
        self.annotation_cursor = visible[new_slot]

    def jump_to_start(self) -> None:
        visible = self._visible_indices_in_current_file()
        self.annotation_cursor = visible[0] if visible else 0

    def jump_to_end(self) -> None:
        visible = self._visible_indices_in_current_file()
        self.annotation_cursor = visible[-1] if visible else 0

    def next_file(self) -> bool:
        """Advance to the next reviewable file. Returns True if the
        cursor moved, False if already on the last reviewable file."""
        pos = self._file_indices.index(self.file_cursor) \
            if self.file_cursor in self._file_indices else -1
        if pos + 1 >= len(self._file_indices):
            return False
        self.file_cursor = self._file_indices[pos + 1]
        self.annotation_cursor = 0
        # Warm the queue for the NEW cursor position -- if the
        # operator has been advancing steadily, the file they just
        # arrived on was almost certainly already prefetched.
        self._schedule_prefetch()
        return True

    def prev_file(self) -> bool:
        pos = self._file_indices.index(self.file_cursor) \
            if self.file_cursor in self._file_indices else 0
        if pos <= 0:
            return False
        self.file_cursor = self._file_indices[pos - 1]
        self.annotation_cursor = 0
        self._schedule_prefetch()
        return True

    def on_last_annotation_of_file(self) -> bool:
        """True iff the cursor is on the LAST visible annotation of
        this file. When hide_whitelisted is True the 'last visible'
        may be well before the raw last annotation (which could be
        whitelisted); the 'n'-gate should still fire correctly for
        the operator's actual view of the file."""
        anns = self.annotations_in_current_file()
        if not anns:
            return False
        visible = self._visible_indices_in_current_file()
        return bool(visible) and self.annotation_cursor >= visible[-1]

    # ---- editing ----

    def queue_edit(self, new_text: str) -> EditRecord | None:
        """Persist an edit for the current annotation. Overwrites
        any previous pending edit for the same annotation. Returns
        the created record, or None if there's no current annotation
        (empty file), or None if the submission is a no-op re-save of
        an already-pending edit with identical ``new_text`` (avoids
        flooding the journal + status counter when the operator
        accidentally hits Enter twice or re-enters the same edit)."""
        ann = self.current_annotation()
        if ann is None:
            return None
        key = (self.file_cursor, self.annotation_cursor)
        # No-op detection: if a pending edit already covers this key
        # with the identical new_text, do NOT append to the journal.
        # Prevents duplicate journal lines from Enter-mashing and
        # keeps the [N pending edit(s)] status counter honest.
        existing = self._pending.get(key)
        if existing is not None and existing.new_text == new_text:
            return existing
        record = EditRecord.new(
            file_path=str(self.current_file()),
            record_index=ann.record_index,
            byte_offset_in_record=ann.byte_offset_in_record,
            onset_s=ann.onset_s,
            orig_text=ann.text, new_text=new_text)
        self._pending[key] = record
        self._journal.append(record)
        return record

    def bulk_regex_swap(self, pattern_str: str, replacement: str,
                         scope: str = "all") -> int:
        """Apply a regex substitution to every annotation in scope and
        queue each result as a pending edit. Returns the number of
        annotations changed; -1 signals an invalid regex.

        Semantics:
          - ``re.sub`` under the hood -- backreferences (\\1, \\g<name>)
            and character-class escapes work as expected.
          - Scope: ``'all'`` (default) walks every reviewable file
            in the subject; ``'current'`` walks only the currently-
            open file. Bulk fix-ups (e.g. '*X' -> '*Mark' across
            hundreds of annotations for a Mark-named subject) use
            'all'; single-file spot fixes use 'current'.
          - Respects prior pending edits: the pattern is applied to
            each annotation's CURRENT displayed text (i.e. an earlier
            pending edit's new_text takes precedence over the raw
            annotation text). Same principle as the ``e`` re-edit
            behaviour: build on existing state, don't reset it.
          - No-op swaps (pattern matches but sub returns identical
            text) are NOT queued -- keeps the pending counter honest.
          - Nothing lands on disk here. The pending edits flow through
            the standard end-of-session ``apply_pending_edits`` gate
            just like manual edits.
        """
        try:
            pat = re.compile(pattern_str)
        except re.error:
            return -1
        if scope not in ("all", "current"):
            raise ValueError(f"scope must be 'all' or 'current', got {scope!r}")
        target_indices = (list(self._file_indices) if scope == "all"
                          else [self.file_cursor])
        count = 0
        for file_idx in target_indices:
            anns = self._load_annotations_for_index(file_idx)
            for ann_idx, ann in enumerate(anns):
                key = (file_idx, ann_idx)
                existing = self._pending.get(key)
                current_text = existing.new_text if existing else ann.text
                new_text = pat.sub(replacement, current_text)
                if new_text == current_text:
                    continue    # regex didn't change anything -> skip
                record = EditRecord.new(
                    file_path=str(self._edfs[file_idx]),
                    record_index=ann.record_index,
                    byte_offset_in_record=ann.byte_offset_in_record,
                    onset_s=ann.onset_s,
                    # orig_text is the RAW on-disk value so the audit
                    # trail always shows what was actually mutated,
                    # regardless of how many stacked edits fed into it.
                    orig_text=ann.text,
                    new_text=new_text,
                )
                self._pending[key] = record
                self._journal.append(record)
                count += 1
        return count

    def is_current_edited(self) -> bool:
        return (self.file_cursor, self.annotation_cursor) in self._pending

    def current_display_text(self) -> str | None:
        """Text the operator should see when opening the current
        annotation for edit. Returns the pending edit's ``new_text``
        if one exists (so the operator can build on their previous
        change), else the raw annotation text. Returns None if there
        is no current annotation (empty file)."""
        ann = self.current_annotation()
        if ann is None:
            return None
        key = (self.file_cursor, self.annotation_cursor)
        pending = self._pending.get(key)
        return pending.new_text if pending is not None else ann.text

    # ---- rendering ----

    def visible_lines(self, context: int = 15) -> list[DisplayLine]:
        """Return the annotations to render for a git-log-style scroll
        view: ``context`` VISIBLE lines below the current cursor (plus
        the current line).

        When ``hide_whitelisted`` is True (default), whitelisted
        annotations are excluded from the returned list -- the
        operator only sees annotations that need review. When False,
        every annotation is included; the renderer greys out
        whitelisted ones (previous default behaviour).
        """
        anns = self.annotations_in_current_file()
        if not anns:
            return []
        # Walk forward from the cursor collecting up to context+1
        # VISIBLE lines. Whitelisted lines are skipped when
        # hide_whitelisted is True, so 'context' means what the
        # operator actually sees rather than raw-index distance.
        out: list[DisplayLine] = []
        want = context + 1
        for i in range(self.annotation_cursor, len(anns)):
            a = anns[i]
            is_whitelisted = self.is_whitelisted(a)
            if self.hide_whitelisted and is_whitelisted:
                continue
            key = (self.file_cursor, i)
            pending = self._pending.get(key)
            display_text = pending.new_text if pending is not None else a.text
            out.append(DisplayLine(
                file_index=self.file_cursor,
                annotation_index=i,
                file_path=self.current_file(),
                annotation=a,
                is_current=(i == self.annotation_cursor),
                is_whitelisted=is_whitelisted,
                is_edited=pending is not None,
                display_text=display_text,
            ))
            if len(out) >= want:
                break
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

    def unreviewed_reviewable_files(self) -> list[Path]:
        """Reviewable files (i.e. files the TUI would let the operator
        cursor into) that are NOT yet in the tracker. Used by the CLI's
        end-of-session prompt to ask the operator "mark these as
        reviewed?"; if empty, nothing to prompt about."""
        reviewed = self._tracker.reviewed_paths()
        return [self._edfs[i] for i in self._file_indices
                if str(self._edfs[i]) not in reviewed]

    def mark_all_reviewable_files_reviewed(self) -> list[Path]:
        """Bulk-mark every reviewable file as reviewed. Returns the
        list of files newly marked (i.e. files that weren't already
        in the tracker). Used by the CLI's end-of-session "operator
        looked at everything and quit" flow -- see the design note in
        annotation_review_cli.main().

        Iterates in file_index order so the tracker entries land in
        a deterministic order (helps operators reading the tracker
        file by hand)."""
        reviewed = self._tracker.reviewed_paths()
        newly_marked: list[Path] = []
        for i in self._file_indices:
            path = self._edfs[i]
            if str(path) in reviewed:
                continue
            anns = self._load_annotations_for_index(i)
            n_edited = sum(1 for k in self._pending if k[0] == i)
            self._tracker.mark_reviewed(ReviewedFile.new(
                file_path=path, n_annotations=len(anns),
                n_edited=n_edited))
            newly_marked.append(path)
        return newly_marked

    # ---- close-out ----

    def rotate_applied(self) -> Path | None:
        return self._journal.rotate_applied()

    def rotate_discarded(self) -> Path | None:
        return self._journal.rotate_discarded()

    def close(self) -> None:
        self._journal.close()
        # Shutdown the prefetch pool without waiting for in-flight
        # loads (their results are about to be discarded anyway).
        # cancel_futures=True skips any not-yet-started tasks.
        self._prefetch_pool.shutdown(wait=False, cancel_futures=True)

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
            # Journal replay needs annotations loaded to look them
            # up. Force a sync load; the prefetch pool would be
            # overkill here (one-time cost at controller init).
            anns_for_file = self._load_annotations_for_index(fi)
            for ai, ann in enumerate(anns_for_file):
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
