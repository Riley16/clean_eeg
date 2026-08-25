"""Tests for the annotation-review journal + tracker on-disk state.

Coverage priorities:
    1. Round-trip: what you write, you read back (JSONL correctness)
    2. Append-only + immediate flush: crash safety
    3. Rotation: applied and discarded sessions preserved for audit
    4. Skip-list semantics: reviewed_paths deduplicates re-reviews
"""

from __future__ import annotations

import json

from clean_eeg.annotation_review.journal import (
    APPLIED_SUBDIR,
    DISCARDED_SUBDIR,
    REVIEWED_TRACKER_NAME,
    SESSION_JSONL_NAME,
    SESSION_SUBDIR,
    ReviewedTracker,
    SessionJournal,
)
from clean_eeg.annotation_review.models import EditRecord, ReviewedFile


# ---------------------------------------------------------------------------
# EditRecord round-trip
# ---------------------------------------------------------------------------

def _edit(file_path: str = "/data/R1XXXY_a.edf",
          record_index: int = 3,
          byte_offset_in_record: int = 42,
          onset_s: float = 12.5,
          orig_text: str = "seizure at Dr. Smith clinic",
          new_text: str = "seizure at XXXXXXXXXXXX clinic") -> EditRecord:
    return EditRecord.new(
        file_path=file_path, record_index=record_index,
        byte_offset_in_record=byte_offset_in_record,
        onset_s=onset_s, orig_text=orig_text, new_text=new_text)


def test_edit_record_roundtrips_through_json():
    """dataclass -> json_dict -> dataclass is lossless. Regression
    guard against a future field addition that forgets one direction."""
    original = _edit()
    dumped = original.to_json_dict()
    reloaded = EditRecord.from_json_dict(dumped)
    assert reloaded == original


def test_edit_record_new_stamps_current_time():
    """``EditRecord.new`` populates edited_at at creation. The exact
    timestamp isn't tested (racy); we just confirm it's non-empty
    and parses as ISO 8601 with a UTC marker."""
    e = _edit()
    assert e.edited_at
    # Ends with '+00:00' or 'Z' -- our helper uses '+00:00' via
    # datetime.isoformat on a UTC-aware datetime
    assert e.edited_at.endswith("+00:00")


# ---------------------------------------------------------------------------
# SessionJournal: write + read round-trip
# ---------------------------------------------------------------------------

def test_session_journal_appends_and_reads_back_in_order(tmp_path):
    """Preserving order matters: the approval gate at end-of-subject
    presents edits chronologically so the operator can undo the last
    one intuitively."""
    j = SessionJournal(tmp_path)
    e1 = _edit(orig_text="first", new_text="firstX")
    e2 = _edit(orig_text="second", new_text="secondX")
    e3 = _edit(orig_text="third", new_text="thirdX")
    with j:
        j.append(e1)
        j.append(e2)
        j.append(e3)

    reread = SessionJournal(tmp_path).read_all()
    assert [r.orig_text for r in reread] == ["first", "second", "third"]
    assert reread == [e1, e2, e3]


def test_session_journal_creates_hidden_subdir_lazily(tmp_path):
    """Instantiating a journal must NOT create files on disk. Only
    the first append creates the session subdir. Otherwise a query-
    only 'just show me pending edits' call would pollute the tree
    with an empty directory."""
    j = SessionJournal(tmp_path)
    assert not (tmp_path / SESSION_SUBDIR).exists()

    j.append(_edit())
    j.close()

    assert (tmp_path / SESSION_SUBDIR).exists()
    assert (tmp_path / SESSION_SUBDIR / SESSION_JSONL_NAME).exists()


def test_session_journal_survives_reopen(tmp_path):
    """Simulate the process dying and restarting: existing lines must
    remain readable and the next append goes on the end. Otherwise
    a crash mid-session would silently lose prior confirmed edits."""
    j = SessionJournal(tmp_path)
    with j:
        j.append(_edit(orig_text="a"))
        j.append(_edit(orig_text="b"))

    # Fresh instance, same dir
    j2 = SessionJournal(tmp_path)
    with j2:
        j2.append(_edit(orig_text="c"))

    reread = SessionJournal(tmp_path).read_all()
    assert [r.orig_text for r in reread] == ["a", "b", "c"]


def test_session_journal_flushes_after_each_append(tmp_path):
    """CRASH SAFETY: an accepted edit must be on disk before the next
    keystroke. Verified by reading the file directly (bypassing the
    write handle) between appends."""
    j = SessionJournal(tmp_path)
    j.append(_edit(orig_text="only-one-so-far"))

    # Read via a fresh handle -- doesn't share buffers with j._fh
    on_disk = (tmp_path / SESSION_SUBDIR / SESSION_JSONL_NAME).read_text()
    parsed = json.loads(on_disk.splitlines()[0])
    assert parsed["orig_text"] == "only-one-so-far"
    j.close()


def test_session_journal_read_all_returns_empty_when_no_file(tmp_path):
    """No session started yet -> [] not error. Callers use this to
    decide whether an approval gate is even needed."""
    assert SessionJournal(tmp_path).read_all() == []


# ---------------------------------------------------------------------------
# SessionJournal: rotate on apply / discard
# ---------------------------------------------------------------------------

def test_session_journal_rotate_applied_moves_file_and_leaves_empty(tmp_path):
    """After apply, the pending session.jsonl is gone (next session
    starts fresh) but the audit trail is preserved under applied/."""
    j = SessionJournal(tmp_path)
    with j:
        j.append(_edit(orig_text="was applied"))

    dest = j.rotate_applied()
    assert dest is not None
    assert dest.parent.name == APPLIED_SUBDIR
    assert dest.exists()
    # Live session file gone -> next session starts fresh
    assert not (tmp_path / SESSION_SUBDIR / SESSION_JSONL_NAME).exists()
    assert SessionJournal(tmp_path).read_all() == []
    # Audit trail retains the applied edit
    audit_lines = dest.read_text().splitlines()
    assert len(audit_lines) == 1
    assert json.loads(audit_lines[0])["orig_text"] == "was applied"


def test_session_journal_rotate_discarded_uses_separate_subdir(tmp_path):
    """discarded/ and applied/ are separate so an operator scanning
    for 'what did I actually change' isn't drowned in aborted
    sessions."""
    j = SessionJournal(tmp_path)
    with j:
        j.append(_edit(orig_text="never applied"))

    dest = j.rotate_discarded()
    assert dest is not None
    assert dest.parent.name == DISCARDED_SUBDIR
    assert dest.exists()
    assert not (tmp_path / SESSION_SUBDIR / APPLIED_SUBDIR).exists()


def test_session_journal_rotate_when_no_session_returns_none(tmp_path):
    """Rotating an empty session is a noop, not an error -- an
    operator who quits without editing anything shouldn't have to
    worry about triggering an exception."""
    j = SessionJournal(tmp_path)
    assert j.rotate_applied() is None
    assert not (tmp_path / SESSION_SUBDIR / APPLIED_SUBDIR).exists()


# ---------------------------------------------------------------------------
# ReviewedTracker: skip-list semantics
# ---------------------------------------------------------------------------

def _reviewed(path: str, n_ann: int = 5, n_edited: int = 0) -> ReviewedFile:
    return ReviewedFile.new(file_path=path, n_annotations=n_ann,
                             n_edited=n_edited)


def test_reviewed_tracker_appends_and_reads_back(tmp_path):
    t = ReviewedTracker(tmp_path)
    e1 = _reviewed("/data/a.edf")
    e2 = _reviewed("/data/b.edf", n_edited=2)
    t.mark_reviewed(e1)
    t.mark_reviewed(e2)

    all_ = t.read_all()
    assert [r.file_path for r in all_] == ["/data/a.edf", "/data/b.edf"]
    assert all_[1].n_edited == 2


def test_reviewed_tracker_reviewed_paths_dedupes_re_reviews(tmp_path):
    """A file re-reviewed in a later session appends a new line, but
    reviewed_paths() must return one entry per path. Otherwise the
    skip-set could balloon and slow down the startup filter."""
    t = ReviewedTracker(tmp_path)
    t.mark_reviewed(_reviewed("/data/a.edf"))
    t.mark_reviewed(_reviewed("/data/b.edf"))
    t.mark_reviewed(_reviewed("/data/a.edf"))    # re-review

    assert t.reviewed_paths() == {"/data/a.edf", "/data/b.edf"}
    # But the audit log still has all three entries
    assert len(t.read_all()) == 3


def test_reviewed_tracker_missing_file_returns_empty(tmp_path):
    """First-ever run: no tracker file yet. read_all() and
    reviewed_paths() both return empty, not raise."""
    t = ReviewedTracker(tmp_path)
    assert t.read_all() == []
    assert t.reviewed_paths() == set()


def test_reviewed_tracker_lives_at_expected_path(tmp_path):
    """The tracker file's location + name is a documented interface
    for operators who want to inspect / edit / delete it manually.
    Regression guard against a rename that would break their
    workflow."""
    t = ReviewedTracker(tmp_path)
    t.mark_reviewed(_reviewed("/data/x.edf"))
    assert (tmp_path / REVIEWED_TRACKER_NAME).exists()
    assert REVIEWED_TRACKER_NAME == ".annotation_reviewed_tracker"
