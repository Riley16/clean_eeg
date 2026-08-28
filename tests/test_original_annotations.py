"""Tests for the raw-annotations backup writer.

Design contract: the pipeline writes PRE-Presidio annotation text to
a sibling of the cleaned-EDF subdir so operators can audit what got
touched without re-running. The sibling contains PHI and MUST NOT
transfer (that's enforced by clean_eeg.transfer's preflight, tested
in test_transfer.py).
"""

from __future__ import annotations

import json

import numpy as np

from clean_eeg.original_annotations import (
    ORIGINAL_ANNOTATIONS_SUFFIX,
    save_raw_annotations,
    sibling_dir_for,
    sibling_dir_inside,
)


def test_sibling_dir_for_appends_suffix_to_parent_dir_name(tmp_path):
    """Sibling path is derived by appending the suffix to the LAST
    component of output_path. For <subject>/clinical_eeg/ the sibling
    is <subject>/clinical_eeg_original_annotations/."""
    output_path = tmp_path / "R1755J" / "clinical_eeg"
    output_path.mkdir(parents=True)
    got = sibling_dir_for(output_path)
    assert got == (tmp_path / "R1755J" /
                    f"clinical_eeg{ORIGINAL_ANNOTATIONS_SUFFIX}")


def test_sibling_dir_for_works_with_arbitrary_subfolder_names(tmp_path):
    """The suffix is parametric so any subfolder convention works,
    not just 'clinical_eeg'. Batch runs on other subfolders should
    get correctly-named siblings."""
    for subfolder in ("clinical_eeg", "raw_eeg", "some_other_name"):
        p = tmp_path / f"subj_{subfolder}" / subfolder
        p.mkdir(parents=True)
        got = sibling_dir_for(p)
        assert got.name == f"{subfolder}{ORIGINAL_ANNOTATIONS_SUFFIX}"
        assert got.parent == p.parent


def test_save_raw_annotations_writes_json_next_to_output_path(tmp_path):
    """The core save operation: annotations for one source EDF land in
    a JSON file inside the sibling directory."""
    output_path = tmp_path / "R1755J" / "clinical_eeg"
    output_path.mkdir(parents=True)
    onsets = np.array([0.5, 1.5, 2.5])
    durations = np.array([-1.0, -1.0, -1.0])
    texts = np.array(["*Mark", "seizure onset", "dr. smith noted"],
                     dtype=object)

    dest = save_raw_annotations(output_path, "f01.edf",
                                  (onsets, durations, texts))

    expected_sibling = tmp_path / "R1755J" / f"clinical_eeg{ORIGINAL_ANNOTATIONS_SUFFIX}"
    assert dest == expected_sibling / "f01.json"
    assert dest.exists()

    payload = json.loads(dest.read_text())
    assert payload["source_edf"] == "f01.edf"
    assert payload["n_annotations"] == 3
    assert len(payload["annotations"]) == 3
    # Contents preserved verbatim -- raw PHI text expected.
    assert payload["annotations"][0]["text"] == "*Mark"
    assert payload["annotations"][2]["text"] == "dr. smith noted"
    assert payload["annotations"][0]["onset"] == 0.5


def test_save_raw_annotations_creates_sibling_dir_if_missing(tmp_path):
    """First-file-of-subject case: the sibling directory doesn't exist
    yet. save_raw_annotations must create it (mkdir with parents),
    otherwise the batch would crash on subject #1."""
    output_path = tmp_path / "R1755J" / "clinical_eeg"
    output_path.mkdir(parents=True)
    sibling = sibling_dir_for(output_path)
    assert not sibling.exists()

    save_raw_annotations(output_path, "f01.edf",
                          (np.array([0.5]), np.array([-1.0]),
                           np.array(["x"], dtype=object)))
    assert sibling.is_dir()


def test_save_raw_annotations_overwrites_on_second_call(tmp_path):
    """On --force re-clean, the same source EDF is processed again.
    The fresh raw dump must replace (not append to) the stale one --
    otherwise the on-disk record diverges from what got fed into
    Presidio this run."""
    output_path = tmp_path / "R1755J" / "clinical_eeg"
    output_path.mkdir(parents=True)

    save_raw_annotations(output_path, "f01.edf",
                          (np.array([0.5]), np.array([-1.0]),
                           np.array(["FIRST"], dtype=object)))
    dest = save_raw_annotations(output_path, "f01.edf",
                                  (np.array([1.5]), np.array([-1.0]),
                                   np.array(["SECOND"], dtype=object)))
    payload = json.loads(dest.read_text())
    assert payload["annotations"][0]["text"] == "SECOND"
    assert payload["n_annotations"] == 1


def test_sibling_dir_inside_returns_none_when_absent(tmp_path):
    """Positive control: transfer source with no misplaced sibling
    dir returns None (safe state)."""
    src = tmp_path / "clinical_eeg"
    src.mkdir()
    (src / "some.edf").touch()
    assert sibling_dir_inside(src) is None


def test_sibling_dir_inside_flags_offender_when_present(tmp_path):
    """If a directory ending in '_original_annotations' lands INSIDE
    the transfer source, sibling_dir_inside returns the offending
    path so preflight can refuse to proceed."""
    src = tmp_path / "clinical_eeg"
    src.mkdir()
    (src / "some.edf").touch()
    bad = src / "clinical_eeg_original_annotations"
    bad.mkdir()
    got = sibling_dir_inside(src)
    assert got == bad


def test_sibling_dir_inside_recursive_catches_deeply_nested(tmp_path):
    """The check must be recursive -- a subdir like
    src/deep/nesting/clinical_eeg_original_annotations/ is still
    a PHI leak. Prevents attackers of the code review process from
    hiding the offender behind one level of nesting."""
    src = tmp_path / "clinical_eeg"
    nested = src / "deep" / "nesting" / "clinical_eeg_original_annotations"
    nested.mkdir(parents=True)
    (nested / "phi.json").write_text('{"text": "raw"}')
    got = sibling_dir_inside(src)
    assert got == nested
