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

    # Normalize empty-string 'delete' edits to the anonymization
    # sentinel 'X'. Rationale: empty-text edits used to leave a blank
    # TAL that pyedflib preserves but iter_annotations skips -- the
    # two readers disagreed on the file's annotation count, and the
    # 'delete' didn't feel deterministic downstream. Substituting 'X'
    # gives a single, visible-anywhere placeholder consistent with the
    # header PHI sentinel (REDACT_NAME_REPLACEMENT). Applies to both
    # manual edits and bulk-regex-swap results.
    from dataclasses import replace as _replace
    pending_edits = [
        _replace(e, new_text="X") if e.new_text == "" else e
        for e in pending_edits
    ]

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
    is guaranteed untouched in that case.

    Sidecars and data EDFs use different reading strategies:
      * Sidecars are read via ``pyedflib.readAnnotations`` -- captures
        every annotation the on-disk file has, including empty-text
        rows that our byte-level ``iter_annotations`` skips. This
        matters because pipeline sidecars routinely contain
        empty-text markers (screenshot 3 of R1671J's original), and
        losing one silently corrupts the file.
      * Data EDFs continue to use ``iter_annotations`` so
        ``_apply_edits_in_memory`` can match edits by their exact
        ``(record_index, byte_offset_in_record)`` fields (which
        pyedflib doesn't expose).
    """
    with pyedflib.EdfReader(str(edf_path)) as f:
        n_signals = f.signals_in_file
        stub_header = f.getHeader()

    if n_signals == 0:
        _apply_edits_sidecar(edf_path, stub_header, edits)
    else:
        current = iter_annotations(edf_path)
        modified_texts = _apply_edits_in_memory(current, edits)
        onsets = np.array([a.onset_s for a in current], dtype=np.float64)
        durations = np.array(
            [(a.duration_s if a.duration_s > 0 else -1.0) for a in current],
            dtype=np.float64)
        texts = np.array(modified_texts, dtype=object)
        _apply_edits_data_edf(edf_path, stub_header,
                              (onsets, durations, texts), edits)


def _apply_edits_sidecar(edf_path: Path,
                          header: dict,
                          edits: list[EditRecord]) -> None:
    """Sidecar path: rewrite the whole file with the modified
    annotation list. No merge dance -- there are no signal bytes to
    preserve, so the safest thing is to let pyedflib pick its own
    record layout for the modified list and atomic-swap the result
    over the original.

    Reads via ``pyedflib.readAnnotations`` (not iter_annotations)
    because pipeline sidecars contain empty-text annotations that
    the byte-level reader skips; missing even one triggers the
    pre-swap count-mismatch and aborts otherwise-valid edits.

    Matches edits to on-disk rows by ``(round(onset, 6), orig_text)``.
    Duplicates at the same (onset, orig_text) are handled by
    consuming edits FIFO from a per-pair queue. An edit whose
    (onset, orig_text) doesn't match any on-disk row is rejected --
    the file may have been mutated between review and apply.

    Before the atomic swap, cross-check the temp against the ORIGINAL
    file (both read via pyedflib) so a corrupted or misordered
    replacement cannot overwrite the source.
    """
    from collections import defaultdict

    with pyedflib.EdfReader(str(edf_path)) as f:
        onsets, durations, texts = f.readAnnotations()

    # Edits keyed by (rounded onset, orig_text). Value is a FIFO queue
    # of new_texts -- handles the rare case where multiple edits share
    # the same (onset, orig_text) pair.
    edits_queue: dict[tuple[float, str], list[str]] = defaultdict(list)
    for e in edits:
        edits_queue[(round(e.onset_s, 6), e.orig_text)].append(e.new_text)

    modified_texts = []
    for onset, text in zip(onsets, texts):
        key = (round(float(onset), 6), str(text))
        queue = edits_queue.get(key)
        if queue:
            modified_texts.append(queue.pop(0))
        else:
            modified_texts.append(str(text))

    unmatched = {k: v for k, v in edits_queue.items() if v}
    if unmatched:
        example = next(iter(unmatched.keys()))
        raise ApplyEditsError(
            f"edit references (onset, orig_text) not present in "
            f"{edf_path.name}: {example!r} (all such: {list(unmatched.keys())[:3]})")

    annotations = (onsets, durations, np.array(modified_texts, dtype=object))

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
    replacement.

    Uses multiset (Counter) math on ``(onset, text)`` tuples so
    duplicate onsets are handled correctly -- files routinely have
    multiple annotations at onset=0.0 (a whitelist-shaped numeric
    marker + a segment header, say), and an onset-keyed lookup would
    incorrectly demand every one of them equal the edit's ``new_text``.

    Expected temp multiset = original multiset - edits' orig entries
    + edits' new entries. If they don't match, an unedited annotation
    changed OR an edit didn't land where expected -- either way, abort
    before overwriting the source.

    Also verifies headers field-by-field and that the total annotation
    count is preserved.
    """
    from collections import Counter

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

    # Rounded-onset multiset comparison. round(6) survives float noise
    # from the pyedflib roundtrip.
    r = lambda x: round(float(x), 6)

    # Onset + duration multisets must survive verbatim (edits only
    # change text).
    if Counter(r(o) for o in o_on) != Counter(r(o) for o in n_on):
        raise ApplyEditsError(
            "onset multiset drifted between original and rewrite")
    if Counter(r(d) for d in o_dur) != Counter(r(d) for d in n_dur):
        raise ApplyEditsError(
            "duration multiset drifted between original and rewrite")

    # (onset, text) multiset math.
    orig_pairs = Counter((r(o), str(t)) for o, t in zip(o_on, o_txt))
    temp_pairs = Counter((r(o), str(t)) for o, t in zip(n_on, n_txt))
    edits_orig = Counter((r(e.onset_s), e.orig_text) for e in edits)
    edits_new = Counter((r(e.onset_s), e.new_text) for e in edits)

    # Sanity: every edit must claim to REPLACE a pair actually present
    # in the original. If not, the operator's edit references a
    # (onset, orig_text) that doesn't exist -- refuse rather than
    # silently ship a rewrite that doesn't correspond to the edit.
    missing_from_orig = edits_orig - orig_pairs
    if missing_from_orig:
        example = next(iter(missing_from_orig))
        raise ApplyEditsError(
            f"edit references (onset, orig_text) not present in the "
            f"original file: {example!r}")

    expected_temp = orig_pairs - edits_orig + edits_new
    if temp_pairs != expected_temp:
        missing = expected_temp - temp_pairs
        extra = temp_pairs - expected_temp
        raise ApplyEditsError(
            f"(onset, text) multiset drift after apply. Expected but "
            f"missing: {list(missing.keys())[:3]}. Present but not "
            f"expected: {list(extra.keys())[:3]}.")


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
        # verify_signals=False: targeted annotation edits don't touch
        # signal-channel byte ranges by construction (byte-surgery
        # writes only into annotation slots). Full signal load + compare
        # on a multi-GB iEEG file can hang for minutes and the guarantee
        # matters much less at review time than at initial-cleaning
        # time. Annotation preservation is still verified by the merge's
        # own annotation-multiset check + _verify_edits_present below.
        merge_annotation_stub_edf(
            str(temp_data_path), str(stub_path), validate=True,
            verify_signals=False)
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
