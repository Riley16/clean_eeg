"""Tests for scripts/count_annotations.py.

Small ops tool -- coverage is minimal but proves:
    1. The counter reads real EDF annotations correctly
    2. Word-tokenization is whitespace-based (matches WPM assumption)
    3. Sidecar '_annotations.edf' files are NOT double-counted
    4. Files pyedflib refuses are surfaced as skipped, not silently dropped
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import pytest


_SCRIPT = (Path(__file__).parent.parent
           / "scripts" / "count_annotations.py")
_spec = importlib.util.spec_from_file_location("count_annotations", _SCRIPT)
assert _spec is not None and _spec.loader is not None
count_annotations = importlib.util.module_from_spec(_spec)
sys.modules["count_annotations"] = count_annotations
_spec.loader.exec_module(count_annotations)


def _write_edf_with_annotations(path: Path, annotation_texts: list[str]
                                 ) -> None:
    """Write a minimal EDF+ with the given annotation texts (one per
    annotation, onset spaced 1 s apart)."""
    n_ch = 2
    sr = 100
    dur = max(2, len(annotation_texts) + 1)
    signal_headers = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sr,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_ch)
    ]
    t = np.arange(0, dur, 1.0 / sr, dtype=np.float32)
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
        for i, text in enumerate(annotation_texts):
            f.writeAnnotation(float(i + 0.5), -1, text)


def test_count_edf_annotations_counts_words_whitespace_tokenized(tmp_path):
    """Word count is whitespace-tokenized -- matches the assumption
    behind the WPM reading-time estimate. 'PAT REF EEG' -> 3 words."""
    edf = tmp_path / "ann.edf"
    _write_edf_with_annotations(edf, [
        "PAT REF EEG",       # 3 words
        "Seizure",           # 1 word
        "eyes closed",       # 2 words
        "",                  # empty -- excluded from BOTH counts
    ])
    n_ann, n_words, n_wl, n_del = count_annotations.count_edf_annotations(edf)
    assert n_ann == 3
    assert n_words == 6
    assert n_wl == 0


def test_scan_parent_reports_per_subject_totals(tmp_path):
    """Per-subject stats bucketed by subject folder name. Tuple shape:
    (n_ann, n_words, n_files_ok, n_files_skipped, n_files_reviewed,
    n_whitelisted)."""
    for code, texts in [
        ("R1A", ["one two", "three"]),           # 2 ann, 3 words
        ("R1B", ["four five six seven"]),        # 1 ann, 4 words
    ]:
        (tmp_path / code / "clinical_eeg").mkdir(parents=True)
        _write_edf_with_annotations(
            tmp_path / code / "clinical_eeg" / f"{code}_file.edf", texts)

    result, _ = count_annotations.scan_parent(tmp_path)
    assert result["R1A"] == (2, 3, 1, 0, 0, 0, 0)
    assert result["R1B"] == (1, 4, 1, 0, 0, 0, 0)


def test_scan_parent_skips_annotation_sidecars(tmp_path):
    """NEGATIVE regression: inplace-mode '*_annotations.edf' stubs are
    copies of what's already inline in the main EDF. Counting both
    would inflate the estimate by 2x and mislead the operator into
    scheduling more time than needed.
    """
    inner = tmp_path / "R1SUBJ" / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "R1SUBJ.edf",
                                 ["one two", "three"])
    _write_edf_with_annotations(inner / "R1SUBJ_annotations.edf",
                                 ["one two", "three"])   # sidecar dup

    result, _ = count_annotations.scan_parent(tmp_path)
    assert result["R1SUBJ"] == (2, 3, 1, 0, 0, 0, 0)


def test_scan_parent_reports_skipped_files_separately(tmp_path):
    """Files pyedflib refuses (raw NK EDF+D, corrupt bytes) must land
    in the 'skipped' column, NOT be silently dropped -- an operator
    scoping review time deserves to know coverage is incomplete."""
    inner = tmp_path / "R1SUBJ" / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "ok.edf", ["one two"])
    (inner / "garbage.edf").write_bytes(b"not an EDF at all")

    result, _ = count_annotations.scan_parent(tmp_path)
    n_ann, n_words, n_ok, n_skipped, n_reviewed, n_wl, n_del = result["R1SUBJ"]
    assert n_ok == 1
    assert n_skipped == 1
    assert n_reviewed == 0
    assert n_ann == 1 and n_words == 2   # only the readable file


# ---------------------------------------------------------------------------
# Whitelist filtering: matched annotations excluded from review count
# ---------------------------------------------------------------------------

def test_scan_parent_excludes_whitelist_matched_annotations(tmp_path):
    """Positive: annotations that match a per-site whitelist regex
    are moved from the review-count bucket to the whitelisted bucket.
    Lets the estimate shrink as the operator's whitelist grows during
    review.
    """
    from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
    import re
    # 'A' site letter -- matches folder R1755A below
    wl = BoilerplateWhitelist(
        shared=[],
        per_site={"A": [re.compile(r"PAT REF EEG")]})

    inner = tmp_path / "R1755A" / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "R1755A_file.edf", [
        "PAT REF EEG",            # whitelisted -> excluded
        "seizure",                # counted
        "eyes closed",            # counted (2 words)
    ])

    result, _ = count_annotations.scan_parent(tmp_path, whitelist=wl)
    n_ann, n_words, n_ok, n_skip, n_rev, n_wl, n_del = result["R1755A"]
    assert n_ann == 2           # PAT REF EEG excluded
    assert n_words == 3         # 'seizure' + 'eyes closed' = 1 + 2
    assert n_wl == 1


def test_scan_parent_whitelist_uses_correct_site_bucket(tmp_path):
    """NEGATIVE regression: a whitelist entry under site 'S' must NOT
    silence the same text under site 'A'. Otherwise sites would
    cross-contaminate each other's review counts.
    """
    from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
    import re
    # Entry only under 'S' -- must NOT apply to 'A' below
    wl = BoilerplateWhitelist(
        shared=[],
        per_site={"S": [re.compile(r"PAT REF EEG")]})

    inner = tmp_path / "R1755A" / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "R1755A.edf", ["PAT REF EEG"])

    result, _ = count_annotations.scan_parent(tmp_path, whitelist=wl)
    n_ann, n_words, _, _, _, n_wl, n_del = result["R1755A"]
    assert n_ann == 1   # NOT whitelisted for site A
    assert n_wl == 0


def test_scan_parent_whitelist_matches_full_text_only(tmp_path):
    """NEGATIVE regression: fullmatch semantics are load-bearing.
    A permissive pattern like 'PAT REF' must NOT silence
    'PAT REF EEG CAROL LOOK AT THIS' -- the operator would miss
    the real content."""
    from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
    import re
    wl = BoilerplateWhitelist(shared=[re.compile(r"PAT REF")],
                                per_site={})

    inner = tmp_path / "R1755A" / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "R1755A.edf",
                                 ["PAT REF EEG CAROL LOOK AT THIS"])

    result, _ = count_annotations.scan_parent(tmp_path, whitelist=wl)
    n_ann, n_words, _, _, _, n_wl, n_del = result["R1755A"]
    assert n_ann == 1
    assert n_wl == 0     # NOT matched (partial)
    assert n_words == 7  # PAT REF EEG CAROL LOOK AT THIS -> 7 words


# ---------------------------------------------------------------------------
# Reviewed tracker: already-reviewed files skipped by default
# ---------------------------------------------------------------------------

def test_scan_parent_skips_files_in_reviewed_tracker(tmp_path):
    """POSITIVE integration with the annotation-review tracker: files
    marked reviewed are excluded from the count so the operator's
    remaining-work estimate shrinks as they progress."""
    from clean_eeg.annotation_review.journal import ReviewedTracker
    from clean_eeg.annotation_review.models import ReviewedFile
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    done = inner / "R1755A_done.edf"
    todo = inner / "R1755A_todo.edf"
    _write_edf_with_annotations(done, ["one", "two", "three"])
    _write_edf_with_annotations(todo, ["four five"])

    ReviewedTracker(subj).mark_reviewed(
        ReviewedFile.new(file_path=done, n_annotations=3, n_edited=0))

    result, _ = count_annotations.scan_parent(tmp_path)
    n_ann, n_words, n_ok, n_skip, n_rev, n_wl, n_del = result["R1755A"]
    assert n_rev == 1
    assert n_ok == 1
    assert n_ann == 1        # only 'four five' -> 1 annotation
    assert n_words == 2      # 'four five' -> 2 words


def test_scan_parent_include_reviewed_disables_tracker_filter(tmp_path):
    """--include-reviewed override: still count files listed in the
    tracker. Useful when redoing a full pass after updating the
    whitelist, so the operator can compare 'total' vs 'remaining'."""
    from clean_eeg.annotation_review.journal import ReviewedTracker
    from clean_eeg.annotation_review.models import ReviewedFile
    subj = tmp_path / "R1755A"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "R1755A.edf", ["one two three"])
    ReviewedTracker(subj).mark_reviewed(
        ReviewedFile.new(file_path=inner / "R1755A.edf",
                         n_annotations=1, n_edited=0))

    result, _ = count_annotations.scan_parent(
        tmp_path, respect_reviewed_tracker=False)
    n_ann, n_words, n_ok, n_skip, n_rev, n_wl, n_del = result["R1755A"]
    assert n_rev == 0        # tracker ignored
    assert n_ok == 1
    assert n_ann == 1
    assert n_words == 3


# ---------------------------------------------------------------------------
# Skip subjects without the expected subfolder / with permission errors
# ---------------------------------------------------------------------------

def test_scan_parent_silently_drops_subjects_without_subfolder(tmp_path):
    """A folder without <subfolder>/ is 'just empty' from this tool's
    perspective -- subject not ingested yet, wrong layout, or stray
    dir. MUST be silently dropped: not counted in per_subject (would
    fabricate a 0-annotation row and inflate the subject count) and
    not surfaced in the skipped list (would be pure noise across a
    large parent dir where most folders don't yet have data).
    """
    # Layout A: expected -- gets counted
    (tmp_path / "R1755A" / "clinical_eeg").mkdir(parents=True)
    _write_edf_with_annotations(
        tmp_path / "R1755A" / "clinical_eeg" / "R1755A.edf",
        ["one two"])
    # Layout B: no clinical_eeg subdir -- silently dropped
    (tmp_path / "R1000Z").mkdir()
    _write_edf_with_annotations(
        tmp_path / "R1000Z" / "loose.edf", ["should not count"])

    result, skipped = count_annotations.scan_parent(tmp_path)
    assert "R1755A" in result
    assert "R1000Z" not in result
    # NEGATIVE: R1000Z is NOT in the skipped list (silent drop)
    assert not any(name == "R1000Z" for name, _ in skipped)


@pytest.mark.skipif(os.geteuid() == 0,
                     reason="root bypasses chmod-based permission checks")
def test_scan_parent_skips_permission_denied_subject_and_continues(
        tmp_path):
    """POSITIVE robustness test: an unreadable subject dir does NOT
    halt the whole scan. Simulated by chmod'ing one subject's
    clinical_eeg dir to 000 so iterdir/rglob raise PermissionError.
    The other subject still gets counted, and the offender is
    reported in the skipped list with a clear reason.
    """
    import stat
    # Good subject
    good = tmp_path / "R1755A" / "clinical_eeg"
    good.mkdir(parents=True)
    _write_edf_with_annotations(good / "R1755A.edf", ["one two"])
    # Bad subject: clinical_eeg exists but is unreadable
    bad = tmp_path / "R1666A" / "clinical_eeg"
    bad.mkdir(parents=True)
    _write_edf_with_annotations(bad / "R1666A.edf", ["cant read this"])
    # Chmod to 000. Restore on the way out so pytest can clean up.
    original_mode = bad.stat().st_mode
    os.chmod(bad, 0)
    try:
        result, skipped = count_annotations.scan_parent(tmp_path)
    finally:
        os.chmod(bad, original_mode | stat.S_IRWXU)

    # Good subject counted
    assert "R1755A" in result
    assert result["R1755A"][:2] == (1, 2)
    # Bad subject skipped with a permission-denied reason
    assert "R1666A" not in result
    bad_entries = [(n, r) for n, r in skipped if n == "R1666A"]
    assert len(bad_entries) == 1
    assert "permission" in bad_entries[0][1].lower()


# ---------------------------------------------------------------------------
# Delete bucket: excluded from review count separately from whitelist
# ---------------------------------------------------------------------------

def test_scan_parent_excludes_delete_bucket_annotations(tmp_path):
    """HARD REQUIREMENT: annotations matched by the DELETE bucket
    are excluded from the review count separately from whitelist
    matches. Was a bug: earlier count_annotations only applied
    whitelist.matches, not matches_delete -- delete-marked
    annotations still inflated the count.
    """
    from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
    import re
    # J site: 'Segment: REC START.*' in the delete bucket,
    # 'PAT REF EEG' in the whitelist bucket.
    wl = BoilerplateWhitelist(
        shared=[], per_site={"J": [re.compile(r"PAT REF EEG")]},
        delete_shared=[], delete_per_site={
            "J": [re.compile(r"Segment: REC START.*")]})

    inner = tmp_path / "R1755J" / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf_with_annotations(inner / "R1755J.edf", [
        "Segment: REC START at 10:00",   # DELETE bucket
        "PAT REF EEG",                    # WHITELIST bucket
        "seizure onset",                  # kept (real content)
    ])

    result, _ = count_annotations.scan_parent(tmp_path, whitelist=wl)
    n_ann, n_words, n_ok, n_skip, n_rev, n_wl, n_del = result["R1755J"]
    # Only 'seizure onset' remains
    assert n_ann == 1
    assert n_words == 2
    assert n_wl == 1      # PAT REF EEG
    assert n_del == 1     # Segment: REC START ...
