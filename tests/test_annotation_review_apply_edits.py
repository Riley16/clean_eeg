"""Tests for the corruption-safe apply-edits pass.

Coverage priorities:
    1. HAPPY PATH: an edit lands in the file's annotations after
       apply, readable via pyedflib.
    2. SIGNAL SAFETY: signal bytes byte-identical before/after
       (guaranteed by construction because only annotation-channel
       bytes are ever mutated; test enforces it).
    3. CORRUPTION SAFETY: verify failure -> original untouched, temp
       kept for inspection.
    4. STALE EDIT: unmatched EditRecord aborts with a clear reason
       rather than silently skipping.
    5. NO-EDITS: empty pending list is a no-op, not a crash.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from clean_eeg.annotation_review.apply_edits import (
    APPLY_TEMP_SUFFIX,
    ApplyEditsError,
    _apply_edits_in_memory,
    apply_pending_edits,
)
from clean_eeg.annotation_review.models import EditRecord
from clean_eeg.annotation_reader import iter_annotations
from clean_eeg.modify_edf_inplace import create_annotations_only_edf


def _write_sidecar(path: Path, annotations: list[tuple[float, str]]) -> None:
    """Write an annotation-only EDF the same way the pipeline does in
    in-place mode. `create_annotations_only_edf` calls pyedflib with
    `n_channels=0`, and pyedflib packs annotations one per record --
    which is what makes the merge path fail on real sidecars.
    """
    header = {
        "technician": "T", "recording_additional": "",
        "patientname": "X", "patient_additional": "",
        "patientcode": "R1TEST", "equipment": "X", "admincode": "",
        "sex": "X",
        "startdate": datetime(1985, 1, 1, 10, 0, 0),
        "birthdate": "01 jan 1985", "gender": "X",
    }
    onsets = np.array([o for o, _ in annotations], dtype=np.float64)
    durations = np.array([-1.0] * len(annotations), dtype=np.float64)
    texts = np.array([t for _, t in annotations], dtype=object)
    create_annotations_only_edf(str(path), header,
                                 (onsets, durations, texts), validate=True)


def _write_edf(path: Path, annotations: list[tuple[float, str]],
                duration_s: int = 10) -> None:
    n_ch = 2
    sr = 100
    signal_headers = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sr,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_ch)
    ]
    t = np.arange(0, duration_s, 1.0 / sr, dtype=np.float32)
    signals = [(1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
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
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)
        for onset, text in annotations:
            f.writeAnnotation(onset, -1, text)


def _annotation_texts(path: Path) -> list[str]:
    with pyedflib.EdfReader(str(path)) as f:
        _, _, texts = f.readAnnotations()
    return [str(t) for t in texts if str(t).strip()]


def _signal_hash(path: Path) -> str:
    """SHA256 of the raw int16 signal bytes (excluding annotation
    channel). Used to prove signals are byte-identical after apply."""
    with pyedflib.EdfReader(str(path)) as f:
        sigs = [f.readSignal(i, digital=True)
                for i in range(f.signals_in_file)]
    h = hashlib.sha256()
    for s in sigs:
        h.update(np.ascontiguousarray(s).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_apply_edits_replaces_annotation_text_on_disk(tmp_path):
    """POSITIVE end-to-end: one edit, one file, readable via pyedflib
    after apply. This is the core value: the operator's edit actually
    made it to disk in a format the pipeline (and downstream tools)
    can read.
    """
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [
        (0.5, "SEIZURE at Dr. Smith clinic"),
        (2.0, "eyes closed"),
    ])
    ann = iter_annotations(edf)
    dirty = next(a for a in ann if "Dr. Smith" in a.text)

    edit = EditRecord.new(
        file_path=str(edf),
        record_index=dirty.record_index,
        byte_offset_in_record=dirty.byte_offset_in_record,
        onset_s=dirty.onset_s,
        orig_text=dirty.text,
        new_text="SEIZURE at XXXXXXX clinic")

    results = apply_pending_edits([edit])

    assert len(results) == 1
    assert results[0].succeeded
    assert results[0].n_edits_applied == 1

    texts_after = _annotation_texts(edf)
    assert "SEIZURE at XXXXXXX clinic" in texts_after
    assert "SEIZURE at Dr. Smith clinic" not in texts_after
    # Untouched annotation survived
    assert "eyes closed" in texts_after


def test_apply_edits_multiple_edits_same_file(tmp_path):
    """Multiple edits on one file all land in a single apply pass.
    Regression guard against a bug where only the last edit sticks
    (would trivially happen if the merge overwrote instead of
    building the full text list)."""
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [
        (0.5, "a"), (1.5, "b"), (2.5, "c"),
    ])

    ann = iter_annotations(edf)
    edits = [
        EditRecord.new(
            file_path=str(edf),
            record_index=a.record_index,
            byte_offset_in_record=a.byte_offset_in_record,
            onset_s=a.onset_s, orig_text=a.text,
            new_text=a.text.upper() * 2)  # "AA", "BB", "CC"
        for a in ann
    ]
    results = apply_pending_edits(edits)
    assert results[0].succeeded

    texts = _annotation_texts(edf)
    assert set(texts) == {"AA", "BB", "CC"}


def test_apply_edits_across_multiple_files(tmp_path):
    """Grouping-by-file: edits split across two files are applied
    separately and both succeed."""
    edf_a = tmp_path / "A.edf"
    edf_b = tmp_path / "B.edf"
    _write_edf(edf_a, [(0.5, "keep_a")])
    _write_edf(edf_b, [(0.5, "orig_b")])

    ann_a = iter_annotations(edf_a)[0]
    ann_b = iter_annotations(edf_b)[0]
    edits = [
        EditRecord.new(file_path=str(edf_a),
                       record_index=ann_a.record_index,
                       byte_offset_in_record=ann_a.byte_offset_in_record,
                       onset_s=ann_a.onset_s, orig_text=ann_a.text,
                       new_text="new_a"),
        EditRecord.new(file_path=str(edf_b),
                       record_index=ann_b.record_index,
                       byte_offset_in_record=ann_b.byte_offset_in_record,
                       onset_s=ann_b.onset_s, orig_text=ann_b.text,
                       new_text="new_b"),
    ]
    results = apply_pending_edits(edits)
    assert len(results) == 2
    assert all(r.succeeded for r in results)
    assert "new_a" in _annotation_texts(edf_a)
    assert "new_b" in _annotation_texts(edf_b)


# ---------------------------------------------------------------------------
# Signal safety
# ---------------------------------------------------------------------------

def test_signal_bytes_are_byte_identical_after_apply(tmp_path):
    """HARD REQUIREMENT: annotation-only edits MUST NOT change any
    signal sample. Enforced by SHA256 of the concatenated signal-
    channel bytes before and after apply. If this ever regresses,
    the manual review would silently corrupt data."""
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [(0.5, "orig")], duration_s=20)

    hash_before = _signal_hash(edf)

    ann = iter_annotations(edf)[0]
    edit = EditRecord.new(
        file_path=str(edf),
        record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s, orig_text=ann.text,
        new_text="edited longer text with more content")

    results = apply_pending_edits([edit])
    assert results[0].succeeded

    hash_after = _signal_hash(edf)
    assert hash_after == hash_before, (
        "signal bytes changed after annotation-only edit -- "
        "corruption in the merge path")


# ---------------------------------------------------------------------------
# Annotation-only sidecar (the pipeline's in-place `_annotations.edf` output)
# ---------------------------------------------------------------------------


def test_apply_edits_on_sidecar_across_many_records(tmp_path):
    """Reproduces the real-world failure the operator hit: a sidecar
    written by ``create_annotations_only_edf`` (as the pipeline does)
    with annotations spread across many records. The merge path used
    to overflow record 14's 114-byte slot because every onset >=
    record_duration got clamped to the last record; the sidecar branch
    just rewrites the file.
    """
    sidecar = tmp_path / "R1670J_annotations.edf"
    ann_list = [(0.0, "Segment: REC START SMITH E"),
                (15.0, "A1+A2 OFF")] + [
                (30.0 + 15 * i, f"RhythmicBurst RB{i+2}-RB") for i in range(13)]
    _write_sidecar(sidecar, ann_list)
    # Sanity: this really is annotation-only.
    with pyedflib.EdfReader(str(sidecar)) as f:
        assert f.signals_in_file == 0, (
            "pipeline sidecar must have 0 signal channels")

    ann = iter_annotations(sidecar)
    assert len({a.record_index for a in ann}) > 1, (
        "test fixture is degenerate: annotations must span multiple "
        "records to exercise the failure mode")

    dirty = next(a for a in ann if "SMITH" in a.text)
    edit = EditRecord.new(
        file_path=str(sidecar),
        record_index=dirty.record_index,
        byte_offset_in_record=dirty.byte_offset_in_record,
        onset_s=dirty.onset_s, orig_text=dirty.text,
        new_text="Segment: REC START X E")

    results = apply_pending_edits([edit])
    assert len(results) == 1 and results[0].succeeded, (
        results[0].error_message)

    texts_after = _annotation_texts(sidecar)
    assert "Segment: REC START X E" in texts_after
    assert "Segment: REC START SMITH E" not in texts_after
    # Every other annotation must survive verbatim.
    for _, orig in ann_list[1:]:
        assert orig in texts_after, (
            f"annotation {orig!r} lost during apply -- merge path is "
            f"corrupting the sidecar")


def test_sidecar_apply_no_leftover_temp_on_success(tmp_path):
    """On success, no `.review_apply.tmp` file remains next to the sidecar."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [(0.0, "a"), (5.0, "b"), (10.0, "c")])
    ann = iter_annotations(sidecar)[0]
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s, orig_text=ann.text, new_text="A_EDITED")
    apply_pending_edits([edit])
    leftovers = sorted(p.name for p in tmp_path.iterdir())
    assert leftovers == ["R1670J_annotations.edf"], leftovers


def test_sidecar_still_annotation_only_after_apply(tmp_path):
    """Invariant: apply must not silently convert a sidecar into a
    data EDF (would break downstream tools that expect 0 signals)."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [(0.0, "orig"), (5.0, "keep")])
    ann = iter_annotations(sidecar)[0]
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s, orig_text=ann.text, new_text="EDITED")
    apply_pending_edits([edit])
    with pyedflib.EdfReader(str(sidecar)) as f:
        assert f.signals_in_file == 0


def test_sidecar_apply_verifies_unedited_annotations_against_original(tmp_path):
    """Pre-swap check: every UNEDITED annotation in the replacement
    must equal the corresponding annotation in the ORIGINAL file
    (read via pyedflib). Guards against a chain-of-trust break where
    iter_annotations misreads and the pipeline silently overwrites
    the source with the misread text."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    ann_list = [(0.0, "orig_zero"), (5.0, "REMOVE_ME"), (10.0, "orig_ten"),
                (15.0, "orig_fifteen")]
    _write_sidecar(sidecar, ann_list)

    ann = iter_annotations(sidecar)
    target = next(a for a in ann if a.text == "REMOVE_ME")
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=target.record_index,
        byte_offset_in_record=target.byte_offset_in_record,
        onset_s=target.onset_s, orig_text=target.text,
        new_text="clean_annotation")
    results = apply_pending_edits([edit])
    assert results[0].succeeded, results[0].error_message

    # Read back via pyedflib and compare each unedited slot verbatim.
    with pyedflib.EdfReader(str(sidecar)) as f:
        onsets, _, texts = f.readAnnotations()
    by_onset = {round(float(o), 6): str(t) for o, t in zip(onsets, texts)}
    assert by_onset[0.0] == "orig_zero"
    assert by_onset[5.0] == "clean_annotation"   # edited
    assert by_onset[10.0] == "orig_ten"
    assert by_onset[15.0] == "orig_fifteen"


def test_sidecar_apply_verifies_headers_identical(tmp_path):
    """Pre-swap check: pyedflib.getHeader() must be field-by-field
    identical between original and replacement. Guards against a
    silent header drift (e.g., pyedflib normalizing patientname, or
    a future refactor tacking on admincode)."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [(0.0, "a"), (5.0, "orig"), (10.0, "c")])
    with pyedflib.EdfReader(str(sidecar)) as f:
        header_before = f.getHeader()

    ann = iter_annotations(sidecar)[1]
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s, orig_text=ann.text, new_text="edited")
    results = apply_pending_edits([edit])
    assert results[0].succeeded, results[0].error_message

    with pyedflib.EdfReader(str(sidecar)) as f:
        header_after = f.getHeader()
    for key in header_before:
        assert header_before[key] == header_after.get(key), (
            f"header field {key!r} drifted: {header_before[key]!r} -> "
            f"{header_after.get(key)!r}")


def test_sidecar_apply_aborts_when_temp_diverges_from_original(tmp_path,
                                                                monkeypatch):
    """If the write step somehow produces a temp that mangles an
    UNEDITED annotation, the pre-swap check MUST catch it and abort
    before os.replace. Simulate by making pyedflib produce a temp with
    a mangled non-edited slot (monkeypatch the writer stage to
    corrupt one text)."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [(0.0, "keep_zero"), (5.0, "REMOVE_ME"),
                              (10.0, "keep_ten")])
    ann = iter_annotations(sidecar)
    target = next(a for a in ann if a.text == "REMOVE_ME")

    # Patch create_annotations_only_edf to silently mangle an unedited
    # slot in the temp file. Simulates a chain-of-trust failure.
    from clean_eeg.modify_edf_inplace import create_annotations_only_edf as real_write
    from clean_eeg.annotation_review import apply_edits as ae

    def corrupt_write(path, header, annotations, validate=True):
        onsets, durations, texts = annotations
        texts = list(texts)
        # Mangle the FIRST unedited annotation ("keep_zero").
        for i, t in enumerate(texts):
            if t == "keep_zero":
                texts[i] = "SILENTLY_MANGLED"
                break
        real_write(str(path), header,
                   (onsets, durations, np.array(texts, dtype=object)),
                   validate=validate)

    monkeypatch.setattr(ae, "create_annotations_only_edf", corrupt_write)

    edit = EditRecord.new(
        file_path=str(sidecar), record_index=target.record_index,
        byte_offset_in_record=target.byte_offset_in_record,
        onset_s=target.onset_s, orig_text=target.text, new_text="clean")
    results = apply_pending_edits([edit])
    assert not results[0].succeeded
    err = (results[0].error_message or "").lower()
    # New detection is multiset-based; the error names the missing
    # expected pair and the unexpected present pair.
    assert "silently_mangled" in err and "keep_zero" in err, err

    # Original untouched: the mangled string is NOT in the file.
    texts_after = _annotation_texts(sidecar)
    assert "SILENTLY_MANGLED" not in texts_after
    assert "keep_zero" in texts_after


# ---------------------------------------------------------------------------
# Sidecar edge cases the pre-swap verifier must handle correctly.
# The verifier uses (onset, text) multiset math so duplicate onsets,
# duplicate texts, whitelist-shaped annotations, and other real-world
# quirks don't false-positive-abort the apply.
# ---------------------------------------------------------------------------


def test_sidecar_apply_multi_annotation_same_onset(tmp_path):
    """Real-world case: sidecar has TWO annotations at onset=0.0 (a
    whitelist-shaped '+0.000000' marker + a segment header). The
    operator edits only the segment header. Apply must succeed with
    the numeric marker preserved verbatim. Regression guard for the
    onset-keyed lookup that used to abort with 'edited annotation at
    onset=0.0 has text "+0.000000", expected ...'."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [
        (0.0, "+0.000000"),
        (0.0, "Segment: REC START SMITH E"),
        (15.0, "A1+A2 OFF"),
        (30.0, "RhythmicBurst RB1-RB"),
    ])
    ann = iter_annotations(sidecar)
    target = next(a for a in ann if "SMITH" in a.text)
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=target.record_index,
        byte_offset_in_record=target.byte_offset_in_record,
        onset_s=target.onset_s, orig_text=target.text,
        new_text="Segment: REC START X E")
    results = apply_pending_edits([edit])
    assert results[0].succeeded, results[0].error_message

    texts_after = _annotation_texts(sidecar)
    assert "+0.000000" in texts_after           # whitelist-shaped preserved
    assert "Segment: REC START X E" in texts_after
    assert "Segment: REC START SMITH E" not in texts_after
    assert "A1+A2 OFF" in texts_after
    assert "RhythmicBurst RB1-RB" in texts_after


def test_sidecar_apply_duplicate_orig_text_different_onsets(tmp_path):
    """Two annotations share the SAME text at different onsets.
    Operator edits only one. The unedited copy at the other onset
    must survive verbatim (multiset math on (onset, text) tuples is
    what makes this correct)."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [
        (0.0, "SMITH_LEAK"),
        (10.0, "unrelated"),
        (20.0, "SMITH_LEAK"),   # duplicate text, different onset
    ])
    ann = iter_annotations(sidecar)
    # Edit the LATER copy (onset=20.0). The onset=0.0 copy must survive.
    target = next(a for a in ann if a.onset_s == 20.0)
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=target.record_index,
        byte_offset_in_record=target.byte_offset_in_record,
        onset_s=target.onset_s, orig_text=target.text,
        new_text="clean_20")
    results = apply_pending_edits([edit])
    assert results[0].succeeded, results[0].error_message

    with pyedflib.EdfReader(str(sidecar)) as f:
        onsets, _, texts = f.readAnnotations()
    by_onset = [(round(float(o), 6), str(t)) for o, t in zip(onsets, texts)]
    assert (0.0, "SMITH_LEAK") in by_onset      # unedited copy survives
    assert (20.0, "clean_20") in by_onset       # edit landed
    assert (10.0, "unrelated") in by_onset


def test_sidecar_apply_multiple_edits_in_one_file(tmp_path):
    """Sidecar with 5 annotations, 3 edited in one apply pass. All
    edits land, both unedited annotations survive verbatim."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [
        (0.0, "SMITH_a"), (5.0, "keep_a"), (10.0, "SMITH_b"),
        (15.0, "keep_b"), (20.0, "SMITH_c"),
    ])
    ann = iter_annotations(sidecar)
    edits = [
        EditRecord.new(file_path=str(sidecar),
                        record_index=a.record_index,
                        byte_offset_in_record=a.byte_offset_in_record,
                        onset_s=a.onset_s, orig_text=a.text,
                        new_text=a.text.replace("SMITH_", "X_"))
        for a in ann if a.text.startswith("SMITH_")
    ]
    assert len(edits) == 3
    results = apply_pending_edits(edits)
    assert results[0].succeeded, results[0].error_message

    texts_after = set(_annotation_texts(sidecar))
    assert texts_after == {"X_a", "keep_a", "X_b", "keep_b", "X_c"}


def test_sidecar_apply_edits_a_whitelist_shaped_annotation(tmp_path):
    """The user's earlier concern was that whitelist behavior interferes
    with apply. It doesn't -- the boilerplate whitelist only hides
    annotations from the review VIEW, but the apply path reads every
    annotation (including whitelisted ones) and treats them all as
    ordinary. Confirm by directly editing a '+0.000000' annotation
    (whitelist-shaped)."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [
        (0.0, "+0.000000"),
        (5.0, "SMITH_target"),
    ])
    ann = iter_annotations(sidecar)
    target = next(a for a in ann if a.text == "+0.000000")
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=target.record_index,
        byte_offset_in_record=target.byte_offset_in_record,
        onset_s=target.onset_s, orig_text=target.text,
        new_text="cleaned_boilerplate")
    results = apply_pending_edits([edit])
    assert results[0].succeeded, results[0].error_message

    texts_after = _annotation_texts(sidecar)
    assert "cleaned_boilerplate" in texts_after
    assert "+0.000000" not in texts_after
    assert "SMITH_target" in texts_after         # untouched


def test_sidecar_apply_aborts_when_edit_references_missing_pair(tmp_path):
    """If an EditRecord names an (onset, orig_text) that isn't in the
    current file (e.g. the file was mutated between review and apply),
    the pre-swap check must refuse to swap. Better to fail loudly than
    ship a temp whose annotation multiset doesn't correspond to what
    the operator authorized."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [(0.0, "hello"), (10.0, "world")])
    ann = iter_annotations(sidecar)[0]

    bogus = EditRecord.new(
        file_path=str(sidecar), record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s,
        orig_text="STALE_TEXT_NOT_ON_DISK",   # doesn't match anything
        new_text="new")
    results = apply_pending_edits([bogus])
    # _apply_edits_in_memory catches this before write (unique candidate
    # check fails), so error mentions matching not multiset. Either way:
    # apply must fail, original must survive.
    assert not results[0].succeeded
    texts_after = _annotation_texts(sidecar)
    assert texts_after == ["hello", "world"]


def test_sidecar_apply_preserves_numeric_shaped_texts(tmp_path):
    """Explicit coverage: annotation texts like '+0.5', '-1.234', or
    '0.000000' must round-trip verbatim through the write path even
    though they superficially resemble EDF+ onset/duration byte
    sequences."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [
        (0.0, "+0.000000"),
        (5.0, "-1.234"),
        (10.0, "0.5"),
        (15.0, "edit_me"),
    ])
    ann = iter_annotations(sidecar)
    target = next(a for a in ann if a.text == "edit_me")
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=target.record_index,
        byte_offset_in_record=target.byte_offset_in_record,
        onset_s=target.onset_s, orig_text=target.text,
        new_text="edited")
    results = apply_pending_edits([edit])
    assert results[0].succeeded, results[0].error_message

    texts_after = _annotation_texts(sidecar)
    for expected in ("+0.000000", "-1.234", "0.5", "edited"):
        assert expected in texts_after, (expected, texts_after)


# ---------------------------------------------------------------------------
# Composed workflows: multiple categories of annotations coexisting.
# The apply path is whitelist-agnostic (whitelist only affects the TUI
# view), but we compose the actual runtime state to prove that.
# ---------------------------------------------------------------------------


def _make_whitelist(patterns: list[str]):
    """Construct an in-memory BoilerplateWhitelist matching the given
    patterns as shared entries. Lets us verify controller-level whitelist
    interactions without touching the on-disk boilerplate JSON."""
    from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
    import re as _re
    return BoilerplateWhitelist(
        shared=[_re.compile(p) for p in patterns])


def test_sidecar_apply_5_annotations_1_whitelisted_1_unedited_3_edited(tmp_path):
    """The composed case the operator asked about:
      pos 0: whitelisted (boilerplate) -- unedited
      pos 1: NOT whitelisted, unedited (operator saw it, chose not to edit)
      pos 2, 3, 4: edited by the operator
    After apply, pos 0/1 survive verbatim, pos 2/3/4 carry the edits.
    Whitelist status doesn't reach the apply path -- this is a
    positive-control that the operator's mental model of the pipeline
    (whitelisted rows are preserved, unedited rows are preserved,
    edited rows land) matches what actually happens on disk."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [
        (0.0, "+0.000000"),          # pos 0: whitelist-shaped, unedited
        (5.0, "eyes closed"),        # pos 1: reviewed, deliberately not edited
        (10.0, "SMITH_a"),           # pos 2: edited
        (15.0, "SMITH_b"),           # pos 3: edited
        (20.0, "SMITH_c"),           # pos 4: edited
    ])
    # Sanity: the whitelist WOULD match pos 0 if the controller were
    # running. Apply doesn't care, but assert the fixture reflects the
    # scenario the operator described.
    wl = _make_whitelist([r"\+\d+\.\d+"])
    assert wl.matches("+0.000000")
    assert not wl.matches("eyes closed")

    ann = iter_annotations(sidecar)
    edits = [
        EditRecord.new(file_path=str(sidecar),
                        record_index=a.record_index,
                        byte_offset_in_record=a.byte_offset_in_record,
                        onset_s=a.onset_s, orig_text=a.text,
                        new_text=a.text.replace("SMITH_", "X_"))
        for a in ann if a.text.startswith("SMITH_")
    ]
    assert len(edits) == 3, [a.text for a in ann]
    results = apply_pending_edits(edits)
    assert results[0].succeeded, results[0].error_message

    with pyedflib.EdfReader(str(sidecar)) as f:
        onsets, _, texts = f.readAnnotations()
    pairs = [(round(float(o), 6), str(t)) for o, t in zip(onsets, texts)]
    # Position-by-position (pyedflib preserves write order):
    assert pairs == [
        (0.0, "+0.000000"),      # whitelisted, unedited
        (5.0, "eyes closed"),    # not whitelisted, unedited
        (10.0, "X_a"),           # edited
        (15.0, "X_b"),           # edited
        (20.0, "X_c"),           # edited
    ], pairs


def test_sidecar_apply_all_edge_cases_composed(tmp_path):
    """The kitchen-sink test: one sidecar containing every annotation
    category that can coexist in the wild, one apply pass:

      A. whitelist-shaped ('+0.000000') at a shared onset with the
         edited row (duplicate-onset case)
      B. plain unedited non-whitelisted rows (two of them)
      C. numeric-shaped text at a distinct onset ('-1.5')
      D. duplicate ORIG-TEXT at different onsets ('SMITH_dup' twice)
      E. manual edit on the segment header
      F. bulk-regex-style edit (SMITH -> X)
      G. regex-swap-to-empty-string 'delete' edit (new_text='')

    Verifies:
      - every unedited row lands verbatim
      - every edit lands verbatim
      - the multiset check passes on this mixed shape
      - duplicate-onset rows both survive with correct texts
      - duplicate-text-at-different-onset rows only the edited one
        changes
      - empty-text edit is preserved as empty text in the pyedflib
        readback (apply succeeds; downstream iter_annotations will
        skip the empty-text row -- that's expected behavior)
    """
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [
        (0.0, "+0.000000"),                # A
        (0.0, "Segment: REC START SMITH E"),  # A (same onset), edited by E
        (5.0, "eyes closed"),              # B
        (10.0, "SMITH_dup"),               # D-1 (unedited copy of duplicate)
        (15.0, "-1.5"),                    # C
        (20.0, "SMITH_dup"),               # D-2 (edited copy of duplicate)
        (25.0, "SMITH_regex"),             # F
        (30.0, "REMOVE_ME"),               # G (delete via empty-string)
        (35.0, "system boot"),             # B
    ])

    ann = iter_annotations(sidecar)

    def find(pred):
        return next(a for a in ann if pred(a))

    edits = [
        # E: manual edit on the SMITH segment header at onset=0.0 (duplicate onset)
        EditRecord.new(file_path=str(sidecar),
                        record_index=(seg := find(lambda a: "REC START" in a.text)).record_index,
                        byte_offset_in_record=seg.byte_offset_in_record,
                        onset_s=seg.onset_s, orig_text=seg.text,
                        new_text="Segment: REC START X E"),
        # D-2: edit the LATER duplicate (onset=20.0) only
        EditRecord.new(file_path=str(sidecar),
                        record_index=(dup2 := find(lambda a: a.onset_s == 20.0 and a.text == "SMITH_dup")).record_index,
                        byte_offset_in_record=dup2.byte_offset_in_record,
                        onset_s=dup2.onset_s, orig_text=dup2.text,
                        new_text="X_dup_20"),
        # F: bulk-regex-style edit
        EditRecord.new(file_path=str(sidecar),
                        record_index=(reg := find(lambda a: a.text == "SMITH_regex")).record_index,
                        byte_offset_in_record=reg.byte_offset_in_record,
                        onset_s=reg.onset_s, orig_text=reg.text,
                        new_text="X_regex"),
        # G: empty-string 'delete' edit
        EditRecord.new(file_path=str(sidecar),
                        record_index=(rm := find(lambda a: a.text == "REMOVE_ME")).record_index,
                        byte_offset_in_record=rm.byte_offset_in_record,
                        onset_s=rm.onset_s, orig_text=rm.text,
                        new_text=""),
    ]
    results = apply_pending_edits(edits)
    assert results[0].succeeded, results[0].error_message

    with pyedflib.EdfReader(str(sidecar)) as f:
        onsets, _, texts = f.readAnnotations()
    pairs = sorted(
        (round(float(o), 6), str(t)) for o, t in zip(onsets, texts))
    expected = sorted([
        (0.0, "+0.000000"),
        (0.0, "Segment: REC START X E"),
        (5.0, "eyes closed"),
        (10.0, "SMITH_dup"),          # UNedited duplicate copy
        (15.0, "-1.5"),
        (20.0, "X_dup_20"),           # edited duplicate copy
        (25.0, "X_regex"),
        (30.0, ""),                   # empty-string delete preserved by pyedflib
        (35.0, "system boot"),
    ])
    assert pairs == expected, (
        f"multiset mismatch.\n  got:    {pairs}\n  wanted: {expected}")

    # Downstream sanity: our byte-level reader skips empty-text rows.
    # Not a bug in apply -- documents the current behavior explicitly.
    from clean_eeg.annotation_reader import iter_annotations as _iter
    ann_after = _iter(sidecar)
    assert len(ann_after) == 8, [(a.onset_s, a.text) for a in ann_after]
    assert all(a.text != "" for a in ann_after)
    assert all(a.text != "REMOVE_ME" for a in ann_after)


def test_sidecar_apply_refuses_leftover_temp(tmp_path):
    """Refuses to apply if a `.review_apply.tmp` leftover from a prior
    crash is still on disk -- matches the data-EDF path's safety."""
    sidecar = tmp_path / "R1670J_annotations.edf"
    _write_sidecar(sidecar, [(0.0, "orig")])
    (Path(str(sidecar) + APPLY_TEMP_SUFFIX)).write_bytes(b"leftover")
    ann = iter_annotations(sidecar)[0]
    edit = EditRecord.new(
        file_path=str(sidecar), record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s, orig_text=ann.text, new_text="X")
    results = apply_pending_edits([edit])
    assert not results[0].succeeded
    assert "leftover" in (results[0].error_message or "").lower()


# ---------------------------------------------------------------------------
# Corruption safety: unmatched edit
# ---------------------------------------------------------------------------

def test_apply_edits_aborts_on_unmatched_edit_leaving_original_intact(
        tmp_path):
    """SAFETY: an EditRecord that doesn't match any current annotation
    (file was mutated between review and apply) must abort the pass
    with ApplyEditsError, leaving the original untouched. Silently
    skipping would mean the operator's session log claims edits that
    were never applied.
    """
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [(0.5, "hello")])
    ann = iter_annotations(edf)[0]

    # Craft a stale edit that won't match: pretend the annotation
    # originally said something different.
    stale = EditRecord.new(
        file_path=str(edf),
        record_index=ann.record_index,
        byte_offset_in_record=99999,   # bogus offset
        onset_s=99.9,                   # bogus onset
        orig_text="never existed",
        new_text="ghost")

    original_bytes = edf.read_bytes()
    results = apply_pending_edits([stale])

    assert not results[0].succeeded
    # Original file MUST be byte-identical
    assert edf.read_bytes() == original_bytes


def test_apply_edits_matches_via_onset_and_orig_text_fallback(tmp_path):
    """POSITIVE regression: when byte_offset_in_record doesn't match
    (e.g. after a benign re-parse), the fallback (onset_s + orig_text)
    still lets the edit land. Guards against a brittle
    identify-by-offset-only design that would refuse any edit whose
    file was re-read between sessions."""
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [(0.5, "original")])
    ann = iter_annotations(edf)[0]

    edit = EditRecord.new(
        file_path=str(edf),
        record_index=999,               # doesn't match current
        byte_offset_in_record=999,      # doesn't match current
        onset_s=ann.onset_s,            # DOES match
        orig_text=ann.text,             # DOES match
        new_text="redacted")

    results = apply_pending_edits([edit])
    assert results[0].succeeded
    assert "redacted" in _annotation_texts(edf)


# ---------------------------------------------------------------------------
# No-edits + edge cases
# ---------------------------------------------------------------------------

def test_apply_pending_edits_empty_list_is_noop(tmp_path):
    """Empty pending list -> no results, no crash. Reached when the
    operator quits a review after only navigating (no edits made)."""
    results = apply_pending_edits([])
    assert results == []


def test_apply_edits_refuses_leftover_temp_file(tmp_path):
    """DEFENSIVE: if a prior apply crashed mid-write leaving a
    <path>.review_apply.tmp behind, refuse to proceed. Operator
    must inspect and remove manually -- silently overwriting the
    leftover would destroy evidence of the earlier failure.
    """
    edf = tmp_path / "R1TEST.edf"
    _write_edf(edf, [(0.5, "hello")])
    # Simulate leftover from a prior crashed run
    (tmp_path / f"R1TEST.edf{APPLY_TEMP_SUFFIX}").write_bytes(b"stale")

    ann = iter_annotations(edf)[0]
    edit = EditRecord.new(
        file_path=str(edf),
        record_index=ann.record_index,
        byte_offset_in_record=ann.byte_offset_in_record,
        onset_s=ann.onset_s, orig_text=ann.text, new_text="new")

    results = apply_pending_edits([edit])
    assert not results[0].succeeded
    assert "leftover" in (results[0].error_message or "").lower()


# ---------------------------------------------------------------------------
# _apply_edits_in_memory unit tests (isolated from disk)
# ---------------------------------------------------------------------------

def test_apply_edits_in_memory_replaces_by_key():
    from clean_eeg.annotation_reader import Annotation
    current = [
        Annotation(record_index=0, byte_offset_in_record=10,
                   onset_s=0.5, duration_s=0, text="a"),
        Annotation(record_index=1, byte_offset_in_record=10,
                   onset_s=1.5, duration_s=0, text="b"),
    ]
    edits = [EditRecord.new(
        file_path="/x.edf", record_index=1, byte_offset_in_record=10,
        onset_s=1.5, orig_text="b", new_text="B_EDITED")]
    result = _apply_edits_in_memory(current, edits)
    assert result == ["a", "B_EDITED"]


def test_apply_edits_in_memory_raises_on_unmatched():
    from clean_eeg.annotation_reader import Annotation
    current = [Annotation(record_index=0, byte_offset_in_record=10,
                            onset_s=0.5, duration_s=0, text="a")]
    edits = [EditRecord.new(
        file_path="/x.edf", record_index=99, byte_offset_in_record=99,
        onset_s=99.0, orig_text="not_a", new_text="ghost")]
    with pytest.raises(ApplyEditsError, match="could not be matched"):
        _apply_edits_in_memory(current, edits)
