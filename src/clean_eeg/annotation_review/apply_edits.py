"""Corruption-safe batch application of pending edits to EDF files.

Takes the list of :class:`EditRecord` accumulated by the controller
and mutates each affected EDF's annotation channel to reflect them.
Reuses the existing corruption-safe primitives in
:mod:`clean_eeg.modify_edf_inplace` rather than reinventing:

    1. Load current annotations via the fast reader
       (:func:`clean_eeg.annotation_reader.iter_annotations`).
    2. Apply the operator's edits in-memory: match each EditRecord
       by ``(record_index, byte_offset_in_record)`` and replace its
       text.
    3. Write a fresh annotations-only stub EDF containing the FULL
       modified annotation list, via
       :func:`create_annotations_only_edf`.
    4. Copy the original data EDF to ``<path>.review_apply.tmp``.
    5. Blank the temp's annotation channel via
       :func:`clear_edf_annotations_inplace`.
    6. Merge the stub back into the temp via
       :func:`merge_annotation_stub_edf` (which atomic-swaps the
       merge_tmp file it creates internally).
    7. Verify the temp loads via pyedflib AND every edit's ``new_text``
       is present.
    8. ``os.replace`` the temp over the original -- atomic on POSIX.
    9. On ANY failure between steps 4-7: the original is untouched,
       the temp is kept for inspection, and the whole apply pass
       aborts (does NOT continue to other files).

Signal-byte identity is guaranteed by construction: neither
``clear_edf_annotations_inplace`` nor ``merge_annotation_stub_edf``
touch any bytes outside the annotation channel slices.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyedflib

from clean_eeg.annotation_reader import iter_annotations
from clean_eeg.annotation_review.models import EditRecord
from clean_eeg.modify_edf_inplace import (
    clear_edf_annotations_inplace,
    create_annotations_only_edf,
    merge_annotation_stub_edf,
)


APPLY_TEMP_SUFFIX = ".review_apply.tmp"
STUB_TEMP_SUFFIX = ".review_stub.edf"


class ApplyEditsError(RuntimeError):
    """Raised when the apply pass cannot safely proceed. The original
    files are always left untouched when this fires."""


@dataclass
class ApplyResult:
    """One entry per affected EDF file after the pass finishes.
    ``succeeded=True`` means the file was atomically replaced with
    the edited version. Signal data is byte-identical to the original
    by construction (only annotation-channel bytes were mutated)."""
    file_path: Path
    n_edits_applied: int
    succeeded: bool
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_pending_edits(pending_edits: list[EditRecord]
                         ) -> list[ApplyResult]:
    """Apply every pending edit to the EDFs on disk. Groups edits by
    ``file_path`` and processes each file atomically. Aborts the
    whole pass on the first file failure (does not proceed to other
    files with the tree in a partially-applied state).

    Returns one :class:`ApplyResult` per file the pass ATTEMPTED. On
    abort, the file that failed has ``succeeded=False`` and later
    files have no result entry (they were never touched).
    """
    if not pending_edits:
        return []

    grouped: dict[Path, list[EditRecord]] = {}
    for e in pending_edits:
        grouped.setdefault(Path(e.file_path), []).append(e)

    results: list[ApplyResult] = []
    for path, file_edits in grouped.items():
        try:
            _apply_edits_to_one_file(path, file_edits)
            results.append(ApplyResult(
                file_path=path,
                n_edits_applied=len(file_edits),
                succeeded=True))
        except Exception as e:
            results.append(ApplyResult(
                file_path=path,
                n_edits_applied=0,
                succeeded=False,
                error_message=f"{type(e).__name__}: {e}"))
            # Abort the whole pass -- don't cascade damage. The
            # operator inspects the failed file and either fixes the
            # underlying issue and reruns, or discards the edits.
            break
    return results


# ---------------------------------------------------------------------------
# Per-file apply with backup + verify + restore
# ---------------------------------------------------------------------------

def _apply_edits_to_one_file(edf_path: Path,
                              edits: list[EditRecord]) -> None:
    """Apply ``edits`` to ``edf_path`` atomically. Raises
    ApplyEditsError with a specific reason on failure; the original
    is guaranteed untouched in that case."""
    current = iter_annotations(edf_path)
    modified_texts = _apply_edits_in_memory(current, edits)

    # Snapshot annotation onsets + durations + modified texts as
    # arrays for the pyedflib writer.
    #
    # Duration convention: our byte-level reader returns 0.0 for TALs
    # that carry no duration (no \x15 in the TAL). pyedflib's
    # readAnnotations returns -1.0 for the same case. The merge's
    # post-write integrity check compares 'durations we wrote' with
    # 'durations pyedflib reads back from the merged file', so if we
    # write 0.0 into the stub, pyedflib rounds-trips it as -1.0 and
    # the check fires 'durations mismatch'. Convert at the writer
    # boundary so the stub carries pyedflib's sentinel and the check
    # passes without confusing our reader's semantics elsewhere.
    onsets = np.array([a.onset_s for a in current], dtype=np.float64)
    durations = np.array(
        [(a.duration_s if a.duration_s > 0 else -1.0) for a in current],
        dtype=np.float64)
    texts = np.array(modified_texts, dtype=object)

    stub_path = Path(str(edf_path) + STUB_TEMP_SUFFIX)
    temp_data_path = Path(str(edf_path) + APPLY_TEMP_SUFFIX)

    if temp_data_path.exists() or stub_path.exists():
        raise ApplyEditsError(
            f"leftover temp file(s) exist next to {edf_path.name} -- "
            f"an earlier apply may have crashed. Inspect and remove "
            f"{temp_data_path.name} / {stub_path.name} manually.")

    # Header for the stub: reuse the original's header via pyedflib.
    # This preserves patient_id, startdate, etc. so the merge doesn't
    # inject inconsistent metadata.
    with pyedflib.EdfReader(str(edf_path)) as f:
        stub_header = f.getHeader()

    try:
        # Write stub with the FULL modified annotation list.
        create_annotations_only_edf(
            str(stub_path), stub_header,
            (onsets, durations, texts), validate=True)

        # Copy the original data EDF to a temp; all mutation happens
        # on the temp so the original is untouched until the final
        # os.replace.
        shutil.copy2(str(edf_path), str(temp_data_path))
        clear_edf_annotations_inplace(str(temp_data_path), validate=True)
        merge_annotation_stub_edf(
            str(temp_data_path), str(stub_path), validate=True)

        _verify_edits_present(temp_data_path, edits)

        # Atomic swap -- last mutation of the original.
        os.replace(str(temp_data_path), str(edf_path))
    except Exception as e:
        # Ensure the original is untouched on any failure between
        # here and the os.replace. We NEVER wrote to edf_path
        # directly; the temp is what we've been mutating. Leave the
        # temp on disk for post-mortem inspection.
        raise ApplyEditsError(
            f"apply aborted for {edf_path.name} at "
            f"{type(e).__name__}: {e}. Original untouched. "
            f"Inspect {temp_data_path.name} / {stub_path.name}."
        ) from e
    finally:
        # Cleanup stub in the success path. Keep the temp on failure
        # (for inspection); on success it's already been renamed.
        if stub_path.exists():
            try:
                stub_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# In-memory edit application
# ---------------------------------------------------------------------------

def _apply_edits_in_memory(current, edits: list[EditRecord]) -> list[str]:
    """Return the annotation text list with edits applied. Matches
    each EditRecord to an Annotation by
    ``(record_index, byte_offset_in_record)``, falling back to
    ``(onset_s, orig_text)`` for robustness against a benign
    re-parse.

    Raises ApplyEditsError if any edit doesn't match a current
    annotation -- unexpected state deserving operator attention,
    NOT a silent skip.
    """
    texts = [a.text for a in current]
    by_key = {(a.record_index, a.byte_offset_in_record): i
              for i, a in enumerate(current)}
    for e in edits:
        key = (e.record_index, e.byte_offset_in_record)
        idx = by_key.get(key)
        if idx is None:
            # Fallback: match by (onset, orig_text). Handles the case
            # where the file was benign-re-parsed and byte offsets
            # shifted, but the annotation content matches.
            candidates = [i for i, a in enumerate(current)
                          if (a.onset_s == e.onset_s
                              and a.text == e.orig_text)]
            if len(candidates) != 1:
                raise ApplyEditsError(
                    f"edit for onset={e.onset_s} orig={e.orig_text!r} "
                    f"could not be matched to a unique current "
                    f"annotation ({len(candidates)} candidates). File "
                    f"may have been mutated between review and apply.")
            idx = candidates[0]
        texts[idx] = e.new_text
    return texts


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_edits_present(edf_path: Path,
                           edits: list[EditRecord]) -> None:
    """After merge, reopen with pyedflib and confirm every edit's
    ``new_text`` appears in the file's annotations. Raises
    ApplyEditsError if any is missing -- the merge silently dropped
    something and we do NOT want to atomic-swap that over the
    original.
    """
    with pyedflib.EdfReader(str(edf_path)) as f:
        _, _, texts_after = f.readAnnotations()
    seen = set(texts_after)
    missing = [e for e in edits if e.new_text not in seen]
    if missing:
        raise ApplyEditsError(
            f"post-merge verify: {len(missing)} edit(s) not found in "
            f"the merged file (first: onset={missing[0].onset_s} "
            f"new={missing[0].new_text!r})")
