"""Pure dataclasses shared across the annotation-review layers.

No I/O, no side effects -- just typed records. All layers that need
to describe an edit or a review-progress entry consume these so the
journal, controller, apply-edits pass, and TUI can't drift apart on
schema.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


def _iso_utc_now() -> str:
    """Timestamp string used in every journal entry. UTC + ISO 8601
    so entries sort lexicographically = chronologically without
    timezone gymnastics on read-back."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class EditRecord:
    """One accepted edit, appended to the session journal on Enter.

    ``file_path`` + ``record_index`` + ``byte_offset_in_record``
    together uniquely locate the TAL inside the EDF, so the batch
    apply step can find the exact bytes to overwrite without re-
    parsing the whole annotation channel. ``onset_s`` is redundant
    with the byte offset for locating but kept for human-readable
    audit trails.
    """
    file_path: str            # absolute path to the EDF
    record_index: int         # data-record index within the file
    byte_offset_in_record: int
    onset_s: float
    orig_text: str
    new_text: str
    edited_at: str            # ISO 8601 UTC

    @classmethod
    def new(cls, *, file_path: str, record_index: int,
            byte_offset_in_record: int, onset_s: float,
            orig_text: str, new_text: str) -> "EditRecord":
        """Convenience constructor that stamps ``edited_at`` now."""
        return cls(
            file_path=file_path, record_index=record_index,
            byte_offset_in_record=byte_offset_in_record,
            onset_s=onset_s, orig_text=orig_text, new_text=new_text,
            edited_at=_iso_utc_now())

    def to_json_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict) -> "EditRecord":
        return cls(**d)


@dataclass
class ReviewedFile:
    """One append-only entry in ``.annotation_reviewed_tracker``.

    Presence of ``file_path`` in the tracker means every annotation
    in that file was seen at least once during a review session.
    Whether or not any were edited is captured for audit but doesn't
    change the skip-on-restart behavior.
    """
    file_path: str            # absolute path to the EDF
    reviewed_at: str          # ISO 8601 UTC
    n_annotations: int
    n_edited: int

    @classmethod
    def new(cls, *, file_path: str | Path,
            n_annotations: int, n_edited: int) -> "ReviewedFile":
        return cls(
            file_path=str(file_path), reviewed_at=_iso_utc_now(),
            n_annotations=n_annotations, n_edited=n_edited)

    def to_json_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict) -> "ReviewedFile":
        return cls(**d)
