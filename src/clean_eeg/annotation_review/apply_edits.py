"""Corruption-safe batch application of pending edits to EDF files.

Takes the list of :class:`EditRecord` accumulated by the controller
and mutates each affected EDF's annotation channel to reflect them.

Two code paths, chosen per file at apply time:

* **Annotation-only sidecar** (``pyedflib.EdfReader.signals_in_file == 0``):
  no signal data to preserve, so the whole file is just annotations.
  Write a fresh sidecar via :func:`create_annotations_only_edf` with
  the modified annotation list, then atomic-swap. The pipeline emits
  these as the ``_annotations.edf`` sidecars in in-place mode, and
  they are what annotation-review normally operates on.
* **Data EDF** (signals present): copy → clear the annotation channel →
  merge a fresh stub back into the temp via
  :func:`merge_annotation_stub_edf`, then atomic-swap. Signal bytes are
  byte-identical by construction because only annotation-channel byte
  ranges are ever mutated.

Sidecars must NOT use the merge path. ``merge_annotation_stub_edf``
redistributes annotations across the target's records using its
``record_duration`` and ``n_records`` -- for pipeline-written sidecars
(``record_duration = 1.0``, one annotation per record) that math sends
all onsets past the last record straight into the tail record's
114-byte slot, which overflows and aborts the whole pass.

On ANY failure the original is untouched and the temp is kept for
inspection.
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

    # Duration convention: our byte-level reader returns 0.0 for TALs
    # that carry no duration (no \x15 in the TAL). pyedflib's
    # readAnnotations returns -1.0 for the same case. Convert at the
    # writer boundary so the on-disk stub carries pyedflib's sentinel
    # and roundtrip checks pass.
    onsets = np.array([a.onset_s for a in current], dtype=np.float64)
    durations = np.array(
        [(a.duration_s if a.duration_s > 0 else -1.0) for a in current],
        dtype=np.float64)
    texts = np.array(modified_texts, dtype=object)

    with pyedflib.EdfReader(str(edf_path)) as f:
        n_signals = f.signals_in_file
        stub_header = f.getHeader()

    if n_signals == 0:
        _apply_edits_sidecar(edf_path, stub_header,
                             (onsets, durations, texts), edits)
    else:
        _apply_edits_data_edf(edf_path, stub_header,
                              (onsets, durations, texts), edits)


def _apply_edits_sidecar(edf_path: Path,
                          header: dict,
                          annotations: tuple,
                          edits: list[EditRecord]) -> None:
    """Sidecar path: rewrite the whole file with the modified
    annotation list. No merge dance -- there are no signal bytes to
    preserve, so the safest thing is to let pyedflib pick its own
    record layout for the modified list and atomic-swap the result
    over the original.

    Before the atomic swap, cross-check the temp against the ORIGINAL
    file (both read via pyedflib) so a corrupted or misordered
    replacement cannot overwrite the source. Verifies:
      * pyedflib can load the temp (loadability guarantee).
      * Headers match field-by-field.
      * Every edited-slot text equals the corresponding
        ``EditRecord.new_text``.
      * Every UNEDITED-slot text equals the original file's text at
        the same onset/duration.
    """
    temp_path = Path(str(edf_path) + APPLY_TEMP_SUFFIX)
    if temp_path.exists():
        raise ApplyEditsError(
            f"leftover temp file next to {edf_path.name} -- "
            f"an earlier apply may have crashed. Inspect and remove "
            f"{temp_path.name} manually.")
    try:
        create_annotations_only_edf(
            str(temp_path), header, annotations, validate=True)
        _verify_edits_present(temp_path, edits)
        _verify_sidecar_against_original(
            original_path=edf_path, temp_path=temp_path, edits=edits)
        os.replace(str(temp_path), str(edf_path))
    except Exception as e:
        raise ApplyEditsError(
            f"apply aborted for {edf_path.name} at "
            f"{type(e).__name__}: {e}. Original untouched. "
            f"Inspect {temp_path.name}."
        ) from e


def _verify_sidecar_against_original(*, original_path: Path,
                                       temp_path: Path,
                                       edits: list[EditRecord]) -> None:
    """Read both files via pyedflib and verify the temp is a safe
    replacement: headers identical, edited slots carry the requested
    new text, and every UNEDITED slot matches the original verbatim.

    Raises ApplyEditsError with a specific reason on the first
    mismatch (halts the swap, leaves the original untouched).
    """
    with pyedflib.EdfReader(str(original_path)) as f:
        orig_header = f.getHeader()
        o_on, o_dur, o_txt = f.readAnnotations()
    with pyedflib.EdfReader(str(temp_path)) as f:
        new_header = f.getHeader()
        n_on, n_dur, n_txt = f.readAnnotations()

    for key in orig_header:
        if orig_header[key] != new_header.get(key):
            raise ApplyEditsError(
                f"header field {key!r} changed during rewrite: "
                f"{orig_header[key]!r} -> {new_header.get(key)!r}")

    if len(o_txt) != len(n_txt):
        raise ApplyEditsError(
            f"annotation count changed: original had {len(o_txt)}, "
            f"replacement has {len(n_txt)}")

    # pyedflib returns annotations in written order for both files.
    # The temp was written with the same onset ordering the original
    # had (because we read the original via iter_annotations, applied
    # in-memory edits, and wrote the list back in the same order).
    # Anchor edits by onset for O(1) lookup; use rounded key to survive
    # float roundtrip noise.
    edited_by_onset = {round(e.onset_s, 6): e.new_text for e in edits}

    for i, (o_ons, o_du, o_tx, n_ons, n_du, n_tx) in enumerate(
            zip(o_on, o_dur, o_txt, n_on, n_dur, n_txt)):
        if not np.isclose(o_ons, n_ons):
            raise ApplyEditsError(
                f"annotation {i} onset moved: {o_ons} -> {n_ons}")
        if not np.isclose(o_du, n_du):
            raise ApplyEditsError(
                f"annotation {i} duration moved: {o_du} -> {n_du}")
        expected = edited_by_onset.get(round(float(o_ons), 6))
        if expected is not None:
            if str(n_tx) != expected:
                raise ApplyEditsError(
                    f"edited annotation at onset={o_ons} has text "
                    f"{str(n_tx)!r}, expected {expected!r}")
        else:
            if str(n_tx) != str(o_tx):
                raise ApplyEditsError(
                    f"UNEDITED annotation at onset={o_ons} changed "
                    f"during rewrite: {str(o_tx)!r} -> {str(n_tx)!r}")


def _apply_edits_data_edf(edf_path: Path,
                           header: dict,
                           annotations: tuple,
                           edits: list[EditRecord]) -> None:
    """Data-EDF path: preserve every signal byte. Uses the byte-level
    clear+merge primitives so the signal channels stay untouched."""
    stub_path = Path(str(edf_path) + STUB_TEMP_SUFFIX)
    temp_data_path = Path(str(edf_path) + APPLY_TEMP_SUFFIX)

    if temp_data_path.exists() or stub_path.exists():
        raise ApplyEditsError(
            f"leftover temp file(s) exist next to {edf_path.name} -- "
            f"an earlier apply may have crashed. Inspect and remove "
            f"{temp_data_path.name} / {stub_path.name} manually.")

    try:
        create_annotations_only_edf(
            str(stub_path), header, annotations, validate=True)
        shutil.copy2(str(edf_path), str(temp_data_path))
        clear_edf_annotations_inplace(str(temp_data_path), validate=True)
        merge_annotation_stub_edf(
            str(temp_data_path), str(stub_path), validate=True)
        _verify_edits_present(temp_data_path, edits)
        os.replace(str(temp_data_path), str(edf_path))
    except Exception as e:
        raise ApplyEditsError(
            f"apply aborted for {edf_path.name} at "
            f"{type(e).__name__}: {e}. Original untouched. "
            f"Inspect {temp_data_path.name} / {stub_path.name}."
        ) from e
    finally:
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
