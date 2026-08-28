"""On-disk state for a manual-annotation-review session.

Two files per subject dir, both JSONL append-only:

    <subject_dir>/.annotation_review/session.jsonl
        Every accepted edit while the review is in progress. Cleared
        when the operator applies (edits then land in the EDFs) or
        discards (edits are moved aside as an audit trail). Append-
        only within a session so a crash never loses previously-
        confirmed work.

    <subject_dir>/.annotation_reviewed_tracker
        One line per fully-reviewed EDF. Read at TUI startup to skip
        already-reviewed files by default; override with a CLI flag.

Both files live under the subject dir (not a central location) so
they travel with the data and never get orphaned by a subject-code
rename.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from clean_eeg.annotation_review.models import EditRecord, ReviewedFile


SESSION_SUBDIR = ".annotation_review"
SESSION_JSONL_NAME = "session.jsonl"
APPLIED_SUBDIR = "applied"
DISCARDED_SUBDIR = "discarded"
REVIEWED_TRACKER_NAME = ".annotation_reviewed_tracker"


# ---------------------------------------------------------------------------
# SessionJournal: pending edits for the current review pass
# ---------------------------------------------------------------------------

class SessionJournal:
    """Append-only JSONL of edits the operator has confirmed but not
    yet applied to the EDF headers. Instantiate once per subject
    review pass; ``append`` on each accepted edit; call ``apply``
    or ``discard`` at the end of the pass to close it out.

    File-open policy: opened lazily on first append so instantiating
    to just query pending edits doesn't create the file.
    """

    def __init__(self, subject_dir: Path):
        self.subject_dir = Path(subject_dir)
        self.session_dir = self.subject_dir / SESSION_SUBDIR
        self.path = self.session_dir / SESSION_JSONL_NAME
        self._fh = None

    # ---- write ----

    def append(self, edit: EditRecord) -> None:
        """Persist one edit immediately (flush before returning). If
        the process is killed on the very next instruction the edit
        is safe on disk."""
        if self._fh is None:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(edit.to_json_dict()) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ---- read ----

    def read_all(self) -> list[EditRecord]:
        """Return every appended edit in insertion order. Missing
        file -> empty list (no session started yet)."""
        if not self.path.exists():
            return []
        edits: list[EditRecord] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            edits.append(EditRecord.from_json_dict(json.loads(line)))
        return edits

    # ---- close-out ----

    def rotate_to(self, subdir_name: str) -> Path | None:
        """Move the session file into ``subject_dir/.annotation_review/
        <subdir_name>/session_<UTC-timestamp>.jsonl`` for audit. Returns
        the new path, or None if there was nothing to rotate (no
        session started or file already rotated).
        """
        self.close()
        if not self.path.exists():
            return None
        archive_dir = self.session_dir / subdir_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = archive_dir / f"session_{ts}.jsonl"
        shutil.move(str(self.path), str(dest))
        return dest

    def rotate_applied(self) -> Path | None:
        """Call after successfully applying edits to the EDFs."""
        return self.rotate_to(APPLIED_SUBDIR)

    def rotate_discarded(self) -> Path | None:
        """Call when the operator declines to apply (or aborts)."""
        return self.rotate_to(DISCARDED_SUBDIR)

    # ---- context manager sugar so callers don't leak file handles ----

    def __enter__(self) -> "SessionJournal":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# ReviewedTracker: per-file "fully seen" bookkeeping
# ---------------------------------------------------------------------------

class ReviewedTracker:
    """One-line-per-file JSONL of every EDF whose annotations have
    been fully reviewed in some prior session.

    Read at TUI startup to compute the default file-skip list.
    Multiple entries per path are allowed (a file re-reviewed in a
    later session appends a new line); ``reviewed_paths()`` returns
    the deduplicated set.
    """

    def __init__(self, subject_dir: Path):
        self.subject_dir = Path(subject_dir)
        self.path = self.subject_dir / REVIEWED_TRACKER_NAME

    def mark_reviewed(self, entry: ReviewedFile) -> None:
        """Append + flush a single reviewed-file entry."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_json_dict()) + "\n")

    def read_all(self) -> list[ReviewedFile]:
        """Every entry in insertion order. Missing file -> []."""
        if not self.path.exists():
            return []
        out: list[ReviewedFile] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(ReviewedFile.from_json_dict(json.loads(line)))
        return out

    def reviewed_paths(self) -> set[str]:
        """Deduplicated set of paths that have been reviewed at least
        once. Callers filter their file list against this set."""
        return {e.file_path for e in self.read_all()}


# ---------------------------------------------------------------------------
# reset_review_state: full-reset helper for --rerun-annot-review
# ---------------------------------------------------------------------------

def reset_review_state(subject_inner: Path) -> dict:
    """Reset the per-subject review state so the TUI treats every file
    as fresh on the next launch. Called by annotation-review-eeg's
    --rerun-annot-review flag.

    Deletes:
      - ``.annotation_reviewed_tracker``: the per-file "seen" record
        (drops files from the reviewable set on next launch).
      - ``.annotation_review/session.jsonl``: the pending-edit buffer
        (any un-applied edits from the aborted session).

    PRESERVES:
      - ``.annotation_review/applied/session_*.jsonl``: audit trail of
        edits that already landed on disk in a prior session. Deleting
        this would lose the compliance record even though the actual
        edited text is already in the sidecar EDFs.
      - ``.annotation_review/discarded/session_*.jsonl``: audit trail
        of edits the operator explicitly discarded. Same reasoning.

    Returns a dict describing what was deleted (empty when nothing to
    reset) so the CLI can print a clear "reset X and Y" message.
    ``subject_inner`` is the directory that CONTAINS the tracker file
    and .annotation_review/ dir (typically ``<subject>/clinical_eeg/``).
    """
    deleted: dict[str, str] = {}
    tracker = subject_inner / REVIEWED_TRACKER_NAME
    if tracker.exists():
        tracker.unlink()
        deleted["tracker"] = str(tracker)
    session_jsonl = subject_inner / SESSION_SUBDIR / SESSION_JSONL_NAME
    if session_jsonl.exists():
        session_jsonl.unlink()
        deleted["pending_session"] = str(session_jsonl)
    return deleted
