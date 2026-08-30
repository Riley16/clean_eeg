"""Tests for clean_eeg.quarantine_verify.

Construct a fake subject_inner/ + quarantine/ layout with real minimal
EDFs and exercise every result outcome (missing counterpart, byte-
equal counterpart, header drift, size mismatch, unreadable file).
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from clean_eeg.clean_subject_eeg import QUARANTINE_SUFFIX
from clean_eeg.quarantine_verify import (
    QuarantineFileResult,
    QuarantineReport,
    format_report,
    recover_original_name,
    verify_quarantine_matches_originals,
)


# ---------- recover_original_name unit tests ----------

def test_recover_original_name_strips_double_deid_and_qmarker():
    assert recover_original_name(
        "GA_R1665J_01.01__18.08.10_R1665J_01.01__18.08.10.edf"
        + QUARANTINE_SUFFIX
    ) == "GA_R1665J_01.01__18.08.10.edf"


def test_recover_original_name_handles_annotation_sidecar():
    assert recover_original_name(
        "GA_R1665J_01.01__18.08.10_R1665J_01.01__18.08.10_annotations.edf"
        + QUARANTINE_SUFFIX
    ) == "GA_R1665J_01.01__18.08.10_annotations.edf"


def test_recover_original_name_leaves_single_stamp_names_alone():
    assert recover_original_name(
        "GA_R1665J_01.01__18.08.10.edf" + QUARANTINE_SUFFIX
    ) == "GA_R1665J_01.01__18.08.10.edf"


def test_recover_original_name_leaves_unstamped_names_alone():
    assert recover_original_name(
        "raw_input.edf" + QUARANTINE_SUFFIX
    ) == "raw_input.edf"


# ---------- Helpers ----------

def _write_minimal_edf(path: Path, *, seed: int = 0) -> None:
    """Write a minimal EDF+C. ``seed`` perturbs the samples so two
    different-seed writes produce byte-different files (for the
    hash-mismatch test)."""
    n_channels = 2
    sample_rate = 100
    duration_s = 2
    signal_headers = [{
        "label": f"CH{i}", "dimension": "uV",
        "sample_frequency": sample_rate,
        "physical_max": 3200.0, "physical_min": -3200.0,
        "digital_max": 32767, "digital_min": -32768,
        "prefilter": "", "transducer": "",
    } for i in range(n_channels)]
    rng = np.random.default_rng(seed)
    signals = [rng.uniform(-1000, 1000, sample_rate * duration_s)
               for _ in range(n_channels)]
    with pyedflib.EdfWriter(str(path), n_channels,
                              file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            "technician": "T", "recording_additional": "",
            "patientname": "X", "patient_additional": "",
            "patientcode": "R1755A", "equipment": "X",
            "admincode": "", "sex": "X",
            "startdate": datetime(1985, 1, 1, 10, 0, 0),
            "birthdate": "01 jan 1900", "gender": "X",
        })
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)


# ---------- verify_quarantine_matches_originals ----------

def test_empty_quarantine_returns_empty_report(tmp_path):
    """No quarantine dir at all -> report with n_total=0, not an error."""
    subj = tmp_path / "s"; subj.mkdir()
    r = verify_quarantine_matches_originals(subj)
    assert r.n_total == 0
    assert r.all_safe_to_delete is False   # nothing to be "safe" about
    assert r.files == []


def test_quarantine_dir_exists_but_empty(tmp_path):
    """quarantine/ exists but empty -> same as no quarantine at all."""
    subj = tmp_path / "s"; subj.mkdir()
    (subj / "quarantine").mkdir()
    r = verify_quarantine_matches_originals(subj)
    assert r.n_total == 0


def test_quarantine_file_with_matching_original_marked_fully_equivalent(
        tmp_path):
    """Standard case: quarantined file has a byte-equal parent
    counterpart -> flagged fully_equivalent, hash matches."""
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()

    orig = subj / "ok_R1755A_01.01__10.00.00.edf"
    _write_minimal_edf(orig, seed=42)

    # Quarantine copy is byte-identical; the double-stamp in the name
    # is what recover_original_name will strip.
    q_name = ("ok_R1755A_01.01__10.00.00_R1755A_01.01__10.00.00.edf"
              + QUARANTINE_SUFFIX)
    shutil.copyfile(orig, q_dir / q_name)

    r = verify_quarantine_matches_originals(subj)
    assert r.n_total == 1
    assert r.n_fully_equivalent == 1
    assert r.all_safe_to_delete is True
    f0 = r.files[0]
    assert f0.orig_exists
    assert f0.quarantine_loads
    assert f0.orig_loads
    assert f0.header_match
    assert f0.signal_header_match
    assert f0.fast_hash_match


def test_quarantine_file_without_counterpart_flagged(tmp_path):
    """Quarantine has a file whose recovered parent name doesn't exist
    in parent dir -> flagged as missing counterpart, all comparison
    flags stay False."""
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()

    q_name = ("orphan_R1755A_01.01__10.00.00_R1755A_01.01__10.00.00.edf"
              + QUARANTINE_SUFFIX)
    _write_minimal_edf(q_dir / q_name)

    r = verify_quarantine_matches_originals(subj)
    assert r.n_total == 1
    assert r.n_fully_equivalent == 0
    assert r.all_safe_to_delete is False
    f0 = r.files[0]
    assert f0.orig_exists is False
    assert any("no counterpart" in n for n in f0.notes)


def test_size_mismatch_short_circuits_and_flags(tmp_path):
    """Files that differ in size can't be duplicates; the check fast-
    fails without reading headers or hashing. Header/hash flags stay
    False, note surfaces the mismatch."""
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()

    orig = subj / "small_R1755A_01.01__10.00.00.edf"
    _write_minimal_edf(orig, seed=0)

    # Write a larger quarantine file with the "double-stamp" name.
    q_name = ("small_R1755A_01.01__10.00.00_R1755A_01.01__10.00.00.edf"
              + QUARANTINE_SUFFIX)
    q_path = q_dir / q_name
    _write_minimal_edf(q_path, seed=0)   # start from same content...
    # ...then append junk bytes so sizes differ.
    with open(q_path, "ab") as f:
        f.write(b"\x00" * 1024)

    r = verify_quarantine_matches_originals(subj)
    assert r.n_total == 1
    assert r.n_fully_equivalent == 0
    f0 = r.files[0]
    assert f0.orig_exists is True
    assert f0.header_match is False
    assert f0.fast_hash_match is False
    assert any("file size differs" in n for n in f0.notes)


def test_content_drift_same_size_flagged_by_hash(tmp_path):
    """Same file size but different content (different seed) --
    caught by the sampled fast hash even though the sizes match."""
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()

    orig = subj / "same_size_R1755A_01.01__10.00.00.edf"
    _write_minimal_edf(orig, seed=0)

    q_name = ("same_size_R1755A_01.01__10.00.00_R1755A_01.01__10.00.00.edf"
              + QUARANTINE_SUFFIX)
    _write_minimal_edf(q_dir / q_name, seed=999)  # different signal payload

    r = verify_quarantine_matches_originals(subj)
    assert r.n_total == 1
    assert r.n_fully_equivalent == 0
    f0 = r.files[0]
    assert f0.orig_exists is True
    # Headers might match (same header shape) but data differs, so
    # fast_hash_match must be False.
    assert f0.fast_hash_match is False


def test_only_QUARANTINED_files_considered(tmp_path):
    """Non-QUARANTINED files sitting in quarantine/ (e.g. stray junk)
    are silently ignored. Only files with the canonical suffix are
    checked."""
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()
    (q_dir / "junk.txt").write_text("stray")

    r = verify_quarantine_matches_originals(subj)
    assert r.n_total == 0


def test_all_safe_to_delete_only_true_when_all_files_pass(tmp_path):
    """Mixed dir: one file safe, one file missing counterpart ->
    all_safe_to_delete is False."""
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()

    # Safe pair.
    good_orig = subj / "good_R1755A_01.01__10.00.00.edf"
    _write_minimal_edf(good_orig, seed=1)
    shutil.copyfile(good_orig,
                    q_dir / ("good_R1755A_01.01__10.00.00_R1755A_01.01"
                              "__10.00.00.edf" + QUARANTINE_SUFFIX))

    # Orphan (no counterpart).
    orphan = ("orphan_R1755A_01.01__10.00.00_R1755A_01.01__10.00.00.edf"
              + QUARANTINE_SUFFIX)
    _write_minimal_edf(q_dir / orphan)

    r = verify_quarantine_matches_originals(subj)
    assert r.n_total == 2
    assert r.n_fully_equivalent == 1
    assert r.all_safe_to_delete is False


# ---------- format_report ----------

def test_format_report_ok_path_recommends_rm(tmp_path):
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()

    orig = subj / "ok_R1755A_01.01__10.00.00.edf"
    _write_minimal_edf(orig)
    shutil.copyfile(orig, q_dir / ("ok_R1755A_01.01__10.00.00_R1755A_01"
                                    ".01__10.00.00.edf"
                                    + QUARANTINE_SUFFIX))

    r = verify_quarantine_matches_originals(subj)
    out = format_report(r)
    assert "[OK]" in out
    assert f"rm -rf {q_dir}" in out


def test_format_report_flagged_path_warns(tmp_path):
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()
    _write_minimal_edf(q_dir / ("orphan_R1755A_01.01__10.00.00_R1755A_01"
                                  ".01__10.00.00.edf"
                                  + QUARANTINE_SUFFIX))

    r = verify_quarantine_matches_originals(subj)
    out = format_report(r)
    assert "[!]" in out
    assert "Manually inspect" in out


# ---------- CLI ----------

def test_main_returns_0_when_all_safe(tmp_path, capsys):
    from clean_eeg.quarantine_verify import main
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()
    orig = subj / "ok_R1755A_01.01__10.00.00.edf"
    _write_minimal_edf(orig)
    shutil.copyfile(orig, q_dir / ("ok_R1755A_01.01__10.00.00_R1755A_01"
                                    ".01__10.00.00.edf"
                                    + QUARANTINE_SUFFIX))
    rc = main([str(subj)])
    assert rc == 0
    assert "[OK]" in capsys.readouterr().out


def test_main_returns_1_when_any_flagged(tmp_path, capsys):
    from clean_eeg.quarantine_verify import main
    subj = tmp_path / "s"; subj.mkdir()
    q_dir = subj / "quarantine"; q_dir.mkdir()
    _write_minimal_edf(q_dir / ("orphan_R1755A_01.01__10.00.00_R1755A_01"
                                  ".01__10.00.00.edf"
                                  + QUARANTINE_SUFFIX))
    rc = main([str(subj)])
    assert rc == 1


def test_main_returns_0_when_no_quarantine_dir(tmp_path, capsys):
    """Nothing to check -> exit 0 (no failure to report)."""
    from clean_eeg.quarantine_verify import main
    subj = tmp_path / "s"; subj.mkdir()
    rc = main([str(subj)])
    assert rc == 0
