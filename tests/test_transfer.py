"""Tests for the transfer tool.

Preflight is exercised in isolation from the full pipeline by crafting
de-identified subject dirs directly — a minimal EDF with the expected
post-de-id header shape, plus a matching manifest. Failure modes are
covered by mutating one field or fixture at a time.

Command composition (rsync vs scp fallback) is tested via dry_run so no
network or subprocess is invoked.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from clean_eeg.audit.hashes import sha256_fast_of_file
from clean_eeg.deidentify_manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    write_manifest,
)
from clean_eeg.transfer import (
    build_transfer_plan,
    preflight_deidentified_output,
    transfer_subject,
)


SUBJECT_CODE = "R1755A"
SITE_CODE = "A"
SITE_INCOMING_FOLDER = "CUDA"


def _write_deidentified_edf(path: Path, *,
                             patientname: str = "X",
                             patientcode: str = SUBJECT_CODE,
                             birthdate: str = "01 jan 1900",
                             startdate: datetime = datetime(1985, 1, 1, 10, 0, 0),
                             ) -> None:
    """Write a minimal EDF+C that looks like it's been through the
    de-identification pipeline. Every kwarg is a field preflight will
    inspect — tests override to fabricate specific failure modes."""
    n_channels = 2
    sample_rate = 100
    duration_s = 2
    signal_headers = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sample_rate,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_channels)
    ]
    t = np.arange(0, duration_s, 1.0 / sample_rate, dtype=np.float32)
    signals = [(1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
               for i in range(n_channels)]
    with pyedflib.EdfWriter(str(path), n_channels,
                             file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            "technician": "T", "recording_additional": "",
            "patientname": patientname, "patient_additional": "",
            "patientcode": patientcode, "equipment": "X",
            "admincode": "", "sex": "X",
            "startdate": startdate,
            "birthdate": birthdate, "gender": "X",
        })
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)
        f.writeAnnotation(0.5, -1, "START")


def _make_subject_dir(tmp_path: Path,
                       filename: str = "ok_R1755A_01.01__10.00.00.edf",
                       **edf_kwargs) -> Path:
    """Populate a subject dir with one de-identified EDF and a matching
    manifest. Returns the dir."""
    out = tmp_path / "out"
    out.mkdir()
    edf_path = out / filename
    _write_deidentified_edf(edf_path, **edf_kwargs)
    manifest = build_manifest(
        subject_code=SUBJECT_CODE,
        site_code=SITE_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER,
        input_path=str(tmp_path / "in"),
        output_path=str(out),
        inplace=True,
        output_edf_paths=[edf_path],
        n_files_deidentified=1,
        n_files_failed=0,
        n_files_quarantined=0,
    )
    write_manifest(out, manifest)
    return out


# ---------- preflight: happy path ----------

def test_preflight_passes_on_valid_output(tmp_path):
    out = _make_subject_dir(tmp_path)
    result = preflight_deidentified_output(out)
    assert result.passed, result.summary()
    assert result.manifest is not None
    assert result.failures == []


# ---------- preflight: failure modes ----------

def test_preflight_fails_when_manifest_missing(tmp_path):
    out = _make_subject_dir(tmp_path)
    (out / MANIFEST_FILENAME).unlink()
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("is missing" in f for f in result.failures)


def test_preflight_fails_on_non_empty_quarantine(tmp_path):
    out = _make_subject_dir(tmp_path)
    (out / "quarantine").mkdir()
    (out / "quarantine" / "junk.edf.QUARANTINED-DO-NOT-USE").write_bytes(b"x")
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("quarantine" in f for f in result.failures)


def test_preflight_passes_on_empty_quarantine(tmp_path):
    out = _make_subject_dir(tmp_path)
    (out / "quarantine").mkdir()  # empty
    result = preflight_deidentified_output(out)
    assert result.passed, result.summary()


def test_preflight_fails_when_patientname_not_x(tmp_path):
    out = _make_subject_dir(tmp_path, patientname="John")
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("patientname" in f for f in result.failures)


def test_preflight_fails_when_patientcode_mismatches(tmp_path):
    out = _make_subject_dir(tmp_path, patientcode="R9999X")
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("patientcode" in f for f in result.failures)


def test_preflight_fails_when_birthdate_wrong(tmp_path):
    out = _make_subject_dir(tmp_path, birthdate="15 mar 1970")
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("birthdate" in f for f in result.failures)


def test_preflight_fails_when_startdate_year_wrong(tmp_path):
    out = _make_subject_dir(tmp_path, startdate=datetime(2023, 1, 1, 10, 0, 0))
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("startdate" in f for f in result.failures)


def test_preflight_fails_when_filename_pattern_wrong(tmp_path):
    out = _make_subject_dir(tmp_path, filename="not_de_identified.edf")
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("does not match the de-identified pattern" in f
               for f in result.failures)


def test_preflight_accepts_annotations_stub_filename(tmp_path):
    """The inplace path writes both `..._R1XXXY_MM.DD__HH.MM.SS.edf`
    and `..._annotations.edf` sidecars. The naming regex must accept
    the sidecar too."""
    out = _make_subject_dir(tmp_path,
                             filename="ok_R1755A_01.01__10.00.00_annotations.edf")
    result = preflight_deidentified_output(out)
    assert result.passed, result.summary()


def test_preflight_fails_when_site_code_unknown(tmp_path):
    """Manifest says site_code='Z' but no known site letter matches Z.
    Even if downstream mapping would fail loudly, we want the transfer
    tool to refuse before touching the network."""
    out = _make_subject_dir(tmp_path)
    manifest = json.loads((out / MANIFEST_FILENAME).read_text())
    manifest["site_code"] = "Z"
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("SITE_CODE_TO_INCOMING_FOLDER" in f for f in result.failures)


def test_preflight_fails_when_hash_disagrees(tmp_path):
    out = _make_subject_dir(tmp_path)
    manifest = json.loads((out / MANIFEST_FILENAME).read_text())
    # Overwrite the hash with a different valid-looking one so the
    # spot-check fails.
    edf_name = next(iter(manifest["file_hashes"]))
    manifest["file_hashes"][edf_name] = "0" * 64
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("hash on disk" in f for f in result.failures)


def test_preflight_fails_when_directory_empty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    manifest = build_manifest(
        subject_code=SUBJECT_CODE,
        site_code=SITE_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER,
        input_path=str(tmp_path / "in"),
        output_path=str(out),
        inplace=True,
        output_edf_paths=[],
        n_files_deidentified=0,
        n_files_failed=0,
        n_files_quarantined=0,
    )
    write_manifest(out, manifest)
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("no .edf files" in f for f in result.failures)


# ---------- plan composition (dry_run) ----------

def test_transfer_plan_uses_rsync_when_available(tmp_path):
    out = _make_subject_dir(tmp_path)
    plan = build_transfer_plan(out,
                                subject_code=SUBJECT_CODE,
                                site_incoming_folder=SITE_INCOMING_FOLDER,
                                ssh_user="testuser",
                                use_rsync=True)
    assert plan.transport == "rsync"
    assert plan.upload_argv[0] == "rsync"
    assert "--partial" in plan.upload_argv
    assert "--exclude=quarantine/" in plan.upload_argv
    # Source path must end with '/' so rsync copies dir contents.
    src_arg = next(a for a in plan.upload_argv if a.endswith("/") and str(out) in a)
    assert src_arg.rstrip("/") == str(out)


def test_transfer_plan_falls_back_to_scp(tmp_path):
    out = _make_subject_dir(tmp_path)
    (out / "log.out").write_text("log")
    plan = build_transfer_plan(out,
                                subject_code=SUBJECT_CODE,
                                site_incoming_folder=SITE_INCOMING_FOLDER,
                                ssh_user="testuser",
                                use_rsync=False)
    assert plan.transport == "scp"
    assert plan.upload_argv[0] == "scp"
    # log.out and deidentify.json must be listed explicitly since
    # *.edf misses both.
    joined = " ".join(plan.upload_argv)
    assert "log.out" in joined
    assert MANIFEST_FILENAME in joined


def test_transfer_plan_remote_dir_matches_site_folder(tmp_path):
    out = _make_subject_dir(tmp_path)
    plan = build_transfer_plan(out,
                                subject_code=SUBJECT_CODE,
                                site_incoming_folder=SITE_INCOMING_FOLDER,
                                ssh_user="testuser",
                                use_rsync=True)
    assert plan.remote_dir.endswith("/CUDA/R1755A/all_clinical_eeg")


def test_transfer_plan_mkdir_uses_umask_007(tmp_path):
    out = _make_subject_dir(tmp_path)
    plan = build_transfer_plan(out,
                                subject_code=SUBJECT_CODE,
                                site_incoming_folder=SITE_INCOMING_FOLDER,
                                ssh_user="testuser",
                                use_rsync=True)
    joined = " ".join(plan.mkdir_argv)
    assert "umask 007" in joined
    assert plan.mkdir_argv[0] == "ssh"


def test_transfer_plan_perms_uses_chgrp_reference(tmp_path):
    out = _make_subject_dir(tmp_path)
    plan = build_transfer_plan(out,
                                subject_code=SUBJECT_CODE,
                                site_incoming_folder=SITE_INCOMING_FOLDER,
                                ssh_user="testuser",
                                use_rsync=True)
    joined = " ".join(plan.perms_argv)
    assert "chgrp -R --reference=" in joined
    assert "chmod -R g+rwX,o-rwx" in joined


# ---------- transfer_subject orchestration ----------

def test_transfer_subject_dry_run_returns_plan_without_executing(tmp_path,
                                                                   monkeypatch):
    out = _make_subject_dir(tmp_path)
    called = []
    monkeypatch.setattr("clean_eeg.transfer.execute_plan",
                        lambda plan: called.append(plan))
    plan = transfer_subject(out, ssh_user="testuser", dry_run=True)
    assert plan.transport in ("rsync", "scp")
    assert called == []  # dry_run must not execute


def test_transfer_subject_raises_on_preflight_failure(tmp_path):
    out = _make_subject_dir(tmp_path)
    (out / MANIFEST_FILENAME).unlink()
    with pytest.raises(RuntimeError, match="Preflight failed"):
        transfer_subject(out, ssh_user="testuser", dry_run=True)
