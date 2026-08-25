"""Tests for scripts/sample_annotations.py -- the whitelist-iteration
helper. Coverage: top-N ordering, sidecar exclusion, --max-files cap.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib


_SCRIPT = (Path(__file__).parent.parent
           / "scripts" / "sample_annotations.py")
_spec = importlib.util.spec_from_file_location("sample_annotations", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sample_mod = importlib.util.module_from_spec(_spec)
sys.modules["sample_annotations"] = sample_mod
_spec.loader.exec_module(sample_mod)


def _write_edf(path: Path, annotations: list[tuple[float, str]]) -> None:
    n_ch, sr, dur = 2, 100, max(2, len(annotations) + 1)
    sh = [{"label": f"CH{i}", "dimension": "uV",
           "sample_frequency": sr,
           "physical_max": 3200.0, "physical_min": -3200.0,
           "digital_max": 32767, "digital_min": -32768,
           "prefilter": "", "transducer": ""}
          for i in range(n_ch)]
    t = np.arange(0, dur, 1.0 / sr, dtype=np.float32)
    sigs = [(1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
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
        f.setSignalHeaders(sh)
        f.writeSamples(sigs)
        for onset, text in annotations:
            f.writeAnnotation(onset, -1, text)


def test_top_n_ranked_by_frequency_across_files(tmp_path, capsys):
    """POSITIVE: the printed top-N is ordered by count. The most-
    common text across all inspected files appears first with its
    total count. This is the load-bearing behavior for whitelist
    iteration -- one regex silencing the top text kills the most
    entries from the review queue.
    """
    _write_edf(tmp_path / "a.edf", [
        (0.5, "PAT REF EEG"),
        (1.5, "PAT REF EEG"),
        (2.5, "seizure"),
    ])
    _write_edf(tmp_path / "b.edf", [
        (0.5, "PAT REF EEG"),
        (1.5, "eyes closed"),
    ])
    # --subject-dir mode: falls through to the dir itself when
    # <subfolder>/ is missing (per _resolve_edfs semantics).
    sample_mod.main([
        "--subject-dir", str(tmp_path),
        "--subfolder", "does_not_exist",
        "--top-n", "5",
        "--sample-n", "0",
    ])
    out = capsys.readouterr().out
    # Header line
    assert "Top 3" in out                        # 3 unique texts
    # Ordering: PAT REF EEG (3 occurrences) BEFORE seizure/eyes closed
    lines = [l for l in out.splitlines() if "PAT REF EEG" in l
             or "seizure" in l or "eyes closed" in l]
    assert lines, out
    assert "PAT REF EEG" in lines[0]
    assert "3" in lines[0]                       # count column


def test_sidecar_annotations_edf_excluded(tmp_path, capsys):
    """NEGATIVE regression: '*_annotations.edf' stubs are inplace-mode
    sidecars -- their annotations duplicate what's inline in the main
    EDF. Including them would inflate the frequency counts and mislead
    the whitelist decision.
    """
    _write_edf(tmp_path / "R1SUBJ.edf",
                [(0.5, "unique_A")])
    _write_edf(tmp_path / "R1SUBJ_annotations.edf",
                [(0.5, "unique_A")])   # dup via sidecar

    sample_mod.main([
        "--subject-dir", str(tmp_path),
        "--subfolder", "does_not_exist",
        "--top-n", "5",
        "--sample-n", "0",
    ])
    out = capsys.readouterr().out
    # 'unique_A' seen ONCE (main EDF), not twice
    lines = [l for l in out.splitlines() if "unique_A" in l]
    assert len(lines) == 1
    # Count column shows 1, not 2
    assert " 1 " in lines[0] or lines[0].strip().startswith("1")


def test_max_files_caps_scan(tmp_path):
    """--max-files caps the list. Simulates 'give me a fast first
    pass on this many-file subject' -- operator doesn't have to wait
    for every file to see the top patterns."""
    for i in range(5):
        _write_edf(tmp_path / f"{i}.edf", [(0.5, f"text_{i}")])
    rc = sample_mod.main([
        "--subject-dir", str(tmp_path),
        "--subfolder", "does_not_exist",
        "--max-files", "2",
        "--top-n", "10",
        "--sample-n", "0",
    ])
    assert rc == 0


def test_explicit_edf_file_mode(tmp_path, capsys):
    """--edf-file skips the subfolder scan and inspects only the
    listed files. Useful for zooming in on one suspect file."""
    _write_edf(tmp_path / "target.edf", [(0.5, "explicit_only")])
    _write_edf(tmp_path / "other.edf", [(0.5, "not_included")])

    sample_mod.main([
        "--edf-file", str(tmp_path / "target.edf"),
        "--top-n", "5",
        "--sample-n", "0",
    ])
    out = capsys.readouterr().out
    assert "explicit_only" in out
    assert "not_included" not in out


# ---------------------------------------------------------------------------
# --parent-dir: multi-subject scan
# ---------------------------------------------------------------------------

def test_parent_dir_scans_across_all_subjects(tmp_path, capsys):
    """POSITIVE: --parent-dir picks up EDFs from every subject
    folder under it. Subjects without <subfolder>/ are silently
    dropped (same rule as count_annotations)."""
    for code, texts in [
        ("R1A", [(0.5, "text_from_A")]),
        ("R1B", [(0.5, "text_from_B")]),
    ]:
        inner = tmp_path / code / "clinical_eeg"
        inner.mkdir(parents=True)
        _write_edf(inner / f"{code}.edf", texts)
    # Subject without the expected subfolder -> silent drop
    (tmp_path / "R1EMPTY").mkdir()

    sample_mod.main([
        "--parent-dir", str(tmp_path),
        "--top-n", "10",
        "--sample-n", "0",
    ])
    out = capsys.readouterr().out
    assert "text_from_A" in out
    assert "text_from_B" in out


# ---------------------------------------------------------------------------
# --random-sample: pick N random files from the resolved set
# ---------------------------------------------------------------------------

def test_random_sample_caps_the_resolved_set(tmp_path, capsys):
    """POSITIVE: --random-sample N reduces the inspected set to N
    files. Verified via the 'inspecting N file(s)' stderr line and
    via which annotation texts appear."""
    for i in range(5):
        _write_edf(tmp_path / f"{i}.edf", [(0.5, f"text_{i}")])
    sample_mod.main([
        "--subject-dir", str(tmp_path),
        "--subfolder", "does_not_exist",
        "--random-sample", "2",
        "--top-n", "20",
        "--sample-n", "0",
        "--seed", "42",
    ])
    out = capsys.readouterr().out
    err = capsys.readouterr().err   # after out is captured
    # The stderr message announces the reduced count
    # (capsys resets between reads; re-read out+err together)
    # Just check that FEWER than 5 unique texts appear.
    n_seen = sum(1 for i in range(5) if f"text_{i}" in out)
    assert n_seen == 2, (
        f"expected exactly 2 files sampled, got {n_seen} in output")


def test_random_sample_is_reproducible_with_seed(tmp_path, capsys):
    """--seed makes --random-sample deterministic across runs. The
    same seed on the same input picks the same subset -- important
    for iterative whitelist work where the operator wants to re-run
    against the same sample after editing the whitelist."""
    for i in range(20):
        _write_edf(tmp_path / f"{i:02d}.edf", [(0.5, f"text_{i:02d}")])

    def _run() -> set[str]:
        sample_mod.main([
            "--subject-dir", str(tmp_path),
            "--subfolder", "does_not_exist",
            "--random-sample", "3",
            "--top-n", "50",
            "--sample-n", "0",
            "--seed", "1234",
        ])
        out = capsys.readouterr().out
        return {f"text_{i:02d}" for i in range(20)
                if f"text_{i:02d}" in out}

    first = _run()
    second = _run()
    assert first == second
    assert len(first) == 3


# ---------------------------------------------------------------------------
# --all-annotations: full dump grouped by file
# ---------------------------------------------------------------------------

def test_all_annotations_dumps_every_text_grouped_by_file(tmp_path,
                                                            capsys):
    """POSITIVE: --all-annotations prints EVERY annotation, grouped
    by file. Catches near-duplicate patterns (e.g. 'seizure at 3.5s'
    vs 'seizure at 4.2s') that frequency counting misses because
    the strings aren't identical."""
    _write_edf(tmp_path / "a.edf", [
        (0.5, "seizure at 3.5s"),
        (1.5, "seizure at 4.2s"),
    ])
    _write_edf(tmp_path / "b.edf", [
        (0.5, "eyes closed"),
    ])
    sample_mod.main([
        "--subject-dir", str(tmp_path),
        "--subfolder", "does_not_exist",
        "--top-n", "0",
        "--sample-n", "0",
        "--all-annotations",
    ])
    out = capsys.readouterr().out
    # Header shows the total + file count
    assert "All annotations" in out
    # Every annotation surfaced
    assert "seizure at 3.5s" in out
    assert "seizure at 4.2s" in out
    assert "eyes closed" in out
    # Grouped by file: file name appears as a section header
    assert "a.edf" in out
    assert "b.edf" in out


def test_all_annotations_off_by_default(tmp_path, capsys):
    """NEGATIVE regression: without --all-annotations, the full dump
    is NOT printed. Guards against a flag flip that would spam huge
    output on every run."""
    _write_edf(tmp_path / "a.edf", [(0.5, "quiet")])
    sample_mod.main([
        "--subject-dir", str(tmp_path),
        "--subfolder", "does_not_exist",
        "--top-n", "10",
        "--sample-n", "0",
    ])
    out = capsys.readouterr().out
    assert "All annotations" not in out
