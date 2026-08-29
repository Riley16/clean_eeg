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
import os
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


def _mark_all_reviewed(out: Path) -> None:
    """Write a tracker entry for every EDF in ``out`` so the transfer
    preflight's "review complete" gate passes. Isolated helper so tests
    that want to exercise the un-reviewed failure mode can skip it."""
    from clean_eeg.annotation_review.journal import (
        REVIEWED_TRACKER_NAME,
        ReviewedTracker,
    )
    from clean_eeg.annotation_review.models import ReviewedFile
    from clean_eeg.print_edf_header import ANNOTATION_STUB_SUFFIX
    edfs = sorted(out.glob("*.edf"))
    stubs = [p for p in edfs if p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    carriers = stubs if stubs else [
        p for p in edfs if not p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    tracker = ReviewedTracker(out)
    for c in carriers:
        tracker.mark_reviewed(ReviewedFile.new(
            file_path=c, n_annotations=1, n_edited=0))
    assert (out / REVIEWED_TRACKER_NAME).exists()


def _make_subject_dir(tmp_path: Path,
                       filename: str = "ok_R1755A_01.01__10.00.00.edf",
                       mark_reviewed: bool = True,
                       **edf_kwargs) -> Path:
    """Populate a subject dir with one de-identified EDF and a matching
    manifest. ``mark_reviewed=True`` (default) also writes a valid
    ``.annotation_reviewed_tracker`` so preflight's review-complete gate
    passes; tests that need the un-reviewed failure mode pass False."""
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
    if mark_reviewed:
        _mark_all_reviewed(out)
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


# ---------- manifest.failed_files: exclusion behaviour ----------

def _add_failed_file_to_manifest(out: Path, failed_name: str,
                                 err: str = "OSError: test-only") -> None:
    """Append one failed-cleaning entry to the manifest at ``out``."""
    m = json.loads((out / MANIFEST_FILENAME).read_text())
    m.setdefault("failed_files", []).append({
        "filename": failed_name, "error_message": err,
        "stage": "load", "quarantined_paths": [],
    })
    (out / MANIFEST_FILENAME).write_text(json.dumps(m))


def test_preflight_excludes_manifest_failed_files_from_checks(tmp_path):
    """Positive: a file listed in manifest.failed_files should NOT be
    checked by preflight (no rename check, no header check). Warning
    is surfaced instead of a failure.
    """
    out = _make_subject_dir(tmp_path)
    # Drop a wrong-name file in the dir and register it as failed. If
    # preflight didn't respect the exclusion, its filename check would
    # fail on the wrong-name file.
    (out / "raw_but_failed.edf").write_bytes(b"not a real edf, doesnt matter")
    _add_failed_file_to_manifest(out, "raw_but_failed.edf")
    result = preflight_deidentified_output(out)
    # NEGATIVE regression guard: the KNOWN-GOOD sibling still shows,
    # preflight still passes overall.
    assert result.passed, result.summary()
    assert any("SKIPPING 1" in w and "raw_but_failed.edf" in w
               for w in result.warnings), result.warnings


def test_preflight_still_fails_when_unlisted_bad_file_present(tmp_path):
    """Negative: a bad file that ISN'T listed in manifest.failed_files
    still trips preflight. Guards against the exclusion path becoming
    a security bypass — files must be explicitly acknowledged to be
    skipped.
    """
    out = _make_subject_dir(tmp_path)
    # Bad-name file, NOT registered as failed in manifest
    _write_deidentified_edf(out / "sneaky_wrong_name.edf")
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("sneaky_wrong_name.edf" in f and "does not match" in f
               for f in result.failures)


def test_preflight_all_files_transferred_when_no_failed_files(tmp_path):
    """Regression guard: baseline behavior with empty failed_files list
    is unchanged — no warnings, no exclusions, everything passes.
    """
    out = _make_subject_dir(tmp_path)
    result = preflight_deidentified_output(out)
    assert result.passed, result.summary()
    assert result.warnings == []


def test_transfer_plan_rsync_excludes_failed_files(tmp_path):
    """rsync mode must add --exclude=<name> for each failed file."""
    plan = build_transfer_plan(
        tmp_path / "out", subject_code=SUBJECT_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER, ssh_user="alice",
        use_rsync=True, remote_dir_override="/tmp/target",
        excluded_names={"bad1.edf", "bad2.edf"},
    )
    assert "--exclude=bad1.edf" in plan.upload_argv
    assert "--exclude=bad2.edf" in plan.upload_argv
    assert "--exclude=quarantine/" in plan.upload_argv  # regression: unchanged


def test_transfer_plan_rsync_no_exclusions_when_no_failed_files(tmp_path):
    """Negative regression: empty excluded_names produces NO extra
    --exclude=<name> flags. Guards against accidentally excluding
    clean files.
    """
    plan = build_transfer_plan(
        tmp_path / "out", subject_code=SUBJECT_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER, ssh_user="alice",
        use_rsync=True, remote_dir_override="/tmp/target",
        excluded_names=None,
    )
    # Only the built-in exclusions (quarantine + the PHI-carrying
    # raw-annotations sibling); no per-file failed-file names.
    _BUILTIN_EXCLUDES = {
        "--exclude=quarantine/",
        "--exclude=*_original_annotations/",
    }
    per_file = [a for a in plan.upload_argv
                if a.startswith("--exclude=") and a not in _BUILTIN_EXCLUDES]
    assert per_file == [], per_file


def test_transfer_plan_scp_filters_glob_by_excluded_names(tmp_path):
    """scp fallback mode must filter the *.edf glob by excluded_names."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "ok_R1755A_01.01__10.00.00.edf").write_bytes(b"x")
    (out / "bad_file.edf").write_bytes(b"x")
    plan = build_transfer_plan(
        out, subject_code=SUBJECT_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER, ssh_user="alice",
        use_rsync=False, remote_dir_override="/tmp/target",
        excluded_names={"bad_file.edf"},
    )
    argv_str = " ".join(plan.upload_argv)
    assert "ok_R1755A_01.01__10.00.00.edf" in argv_str  # positive
    assert "bad_file.edf" not in argv_str               # negative


def test_transfer_plan_scp_no_filter_when_no_excluded(tmp_path):
    """Negative regression: without excluded_names, ALL *.edf files
    make it into the scp argv.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "a_R1755A_01.01__10.00.00.edf").write_bytes(b"x")
    (out / "b_R1755A_01.01__11.00.00.edf").write_bytes(b"x")
    plan = build_transfer_plan(
        out, subject_code=SUBJECT_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER, ssh_user="alice",
        use_rsync=False, remote_dir_override="/tmp/target",
        excluded_names=None,
    )
    argv_str = " ".join(plan.upload_argv)
    assert "a_R1755A_01.01__10.00.00.edf" in argv_str
    assert "b_R1755A_01.01__11.00.00.edf" in argv_str


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


# ---------- preflight: annotation-review-complete gate ----------

def test_preflight_fails_when_review_not_complete(tmp_path):
    """A cleaned subject whose annotations have NOT been manually
    reviewed must not transfer. PHI is only proven redacted after the
    review pass; the operator has explicitly asked for this gate so
    an accidental rsync of a cleaned-but-not-reviewed subject can't
    happen."""
    out = _make_subject_dir(tmp_path, mark_reviewed=False)
    result = preflight_deidentified_output(out)
    assert not result.passed
    assert any("annotation review not complete" in f
               for f in result.failures), result.failures


def test_preflight_review_gate_names_the_subject_and_progress(tmp_path):
    """The failure message must include the subject code AND the
    reviewed/total count so an operator sifting through a batch
    log can prioritise: 'the R1653J entry is at 0/3, the R1654J
    entry is at 3/3 but somehow still failing, so I know which one
    to look at.'"""
    out = _make_subject_dir(tmp_path, mark_reviewed=False)
    result = preflight_deidentified_output(out)
    msg = next(f for f in result.failures
               if "annotation review not complete" in f)
    assert SUBJECT_CODE in msg
    assert "0/1" in msg


def test_preflight_review_gate_uses_sidecars_when_present(tmp_path):
    """In-place cleaning writes annotations to a `*_annotations.edf`
    sidecar. The review gate must count SIDECARS (not the paired main
    EDFs) — that's what the TUI reviews and what the tracker records."""
    from clean_eeg.annotation_review.journal import ReviewedTracker
    from clean_eeg.annotation_review.models import ReviewedFile

    # Two EDFs: one main + one sidecar. Manually construct so we can
    # control which one gets marked reviewed.
    out = tmp_path / "out"
    out.mkdir()
    main = out / "ok_R1755A_01.01__10.00.00.edf"
    sidecar = out / "ok_R1755A_01.01__10.00.00_annotations.edf"
    _write_deidentified_edf(main)
    _write_deidentified_edf(sidecar)
    manifest = build_manifest(
        subject_code=SUBJECT_CODE, site_code=SITE_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER,
        input_path=str(tmp_path / "in"), output_path=str(out),
        inplace=True, output_edf_paths=[main, sidecar],
        n_files_deidentified=1, n_files_failed=0, n_files_quarantined=0)
    write_manifest(out, manifest)

    # Mark ONLY the sidecar reviewed. Preflight should pass -- carrier
    # coverage is 1/1, not 1/2.
    ReviewedTracker(out).mark_reviewed(ReviewedFile.new(
        file_path=sidecar, n_annotations=1, n_edited=0))
    result = preflight_deidentified_output(out)
    assert result.passed, result.summary()


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


# ---------- background transfer ----------

def test_execute_plan_background_writes_script_and_log(tmp_path, monkeypatch):
    """Background launcher must write a shell script + log alongside
    the output dir, and return the child pid. Doesn't actually run
    the transfer (Popen is monkeypatched)."""
    import clean_eeg.transfer as _tr
    out = _make_subject_dir(tmp_path)
    plan = build_transfer_plan(
        out, subject_code=SUBJECT_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER,
        ssh_user="testuser", use_rsync=True,
    )

    captured = {}

    class _FakeProc:
        pid = 12345

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(_tr.subprocess, "Popen", fake_popen)

    pid, script_path, log_path = _tr.execute_plan_background(plan, out)
    assert pid == 12345
    assert script_path == out / "transfer.sh"
    assert log_path == out / "transfer.log"
    # Shell script must exist and be executable.
    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111  # executable bit
    script = script_path.read_text()
    assert "set -e" in script
    assert "rsync" in script
    assert "ssh" in script  # mkdir step
    # nohup + start_new_session detach the child from the terminal.
    assert captured["argv"][0] == "nohup"
    assert captured["kwargs"]["start_new_session"] is True


def test_transfer_subject_background_flag_launches_detached(tmp_path,
                                                              monkeypatch):
    """The background=True path must call execute_plan_background,
    not execute_plan, and stash the pid + paths on the returned plan."""
    import clean_eeg.transfer as _tr
    out = _make_subject_dir(tmp_path)

    fg_calls = []
    bg_calls = []
    monkeypatch.setattr(_tr, "execute_plan",
                        lambda plan: fg_calls.append(plan))
    monkeypatch.setattr(
        _tr, "execute_plan_background",
        lambda plan, output_path: bg_calls.append(plan) or
        (99999, output_path / "transfer.sh", output_path / "transfer.log"),
    )

    plan = transfer_subject(out, ssh_user="testuser", background=True)
    assert fg_calls == [], "background=True must skip execute_plan"
    assert len(bg_calls) == 1
    assert plan.background_pid == 99999
    assert plan.background_log == out / "transfer.log"


# ---------- raw-annotations dump: exclusion + defensive assertion ----------


def test_rsync_argv_excludes_original_annotations_sibling(tmp_path):
    """The raw-annotations dump lives at a SIBLING of the transfer
    source (<subject>/clinical_eeg_original_annotations/). Rsync's
    source is <subject>/clinical_eeg/, so the sibling is already
    outside the sync scope structurally. The explicit --exclude flag
    is belt-and-suspenders against a refactor that changes the source
    path to the subject root -- verify the flag is emitted."""
    out = _make_subject_dir(tmp_path)
    plan = build_transfer_plan(
        out, subject_code=SUBJECT_CODE,
        site_incoming_folder=SITE_INCOMING_FOLDER,
        ssh_user="testuser", use_rsync=True,
        remote_dir_override="/tmp/e2e",
    )
    assert plan.transport == "rsync"
    # Look for the wildcard exclude that catches any '*_original_annotations'
    # directory at the top level of a hypothetical broader source.
    exclude_flags = [a for a in plan.upload_argv if a.startswith("--exclude=")]
    assert any("_original_annotations" in f for f in exclude_flags), (
        f"rsync argv missing --exclude for original-annotations sibling: "
        f"{exclude_flags}"
    )


def test_preflight_fails_when_raw_annotations_dump_inside_source(tmp_path):
    """Defensive: if the raw-annotations dump ever lands INSIDE the
    transfer source (via bad refactor / misplaced write / etc.),
    preflight must fail LOUDLY. Simulate by creating the sibling
    directory inside the transfer source rather than beside it."""
    out = _make_subject_dir(tmp_path)
    # Simulate: raw-annotations dump misplaced INSIDE the transfer
    # source instead of at its sibling.
    bad_dir = out / "clinical_eeg_original_annotations"
    bad_dir.mkdir()
    (bad_dir / "leak.json").write_text('{"text": "PHI here"}')

    result = preflight_deidentified_output(out)
    assert not result.passed, (
        f"preflight must fail when raw-annotations dump is inside "
        f"transfer source; got failures={result.failures}"
    )
    assert any("raw-annotations dump found INSIDE" in f
                for f in result.failures), (
        f"expected specific failure message; got {result.failures}"
    )


def test_preflight_passes_when_raw_annotations_dump_is_sibling(tmp_path):
    """Positive control: raw-annotations dump BESIDE (not inside)
    the transfer source is the intended layout -- preflight must
    pass. Guards against the assertion being too aggressive."""
    from clean_eeg.original_annotations import sibling_dir_for
    out = _make_subject_dir(tmp_path)
    # Create the sibling as the pipeline would.
    sibling = sibling_dir_for(out)
    sibling.mkdir()
    (sibling / "raw.json").write_text('{"text": "PHI here"}')

    result = preflight_deidentified_output(out)
    assert result.passed, (
        f"preflight must pass when raw-annotations dump is a proper "
        f"sibling; got failures={result.failures}"
    )


# ---------- ssh-agent hint ----------


def _stub_subprocess_run_for_ssh_agent(monkeypatch, *,
                                         ssh_add_l_exit: int = 0,
                                         ssh_agent_stdout: str | None = None,
                                         ssh_add_exit: int = 0):
    """Helper: stub subprocess.run to fake ssh-agent-related calls
    deterministically. Returns a call-log list the test can inspect.

    - `ssh-add -l` returns exit=ssh_add_l_exit.
    - `ssh-agent -s` returns exit 0 with the stdout you pass (or a
      valid default); pass None to disable (raises).
    - `ssh-add <key>` returns exit=ssh_add_exit.

    Every OTHER subprocess.run (`git rev-parse HEAD` for manifest
    provenance, `rsync`, etc.) is forwarded to the real subprocess.run
    so the rest of the pipeline works normally. Only ssh-related
    calls are intercepted.
    """
    import clean_eeg.transfer as _tr
    real_run = _tr.subprocess.run
    call_log: list[list[str]] = []

    class _P:
        def __init__(self, rc: int, out: str = "", err: str = ""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def _run(argv, **kw):
        call_log.append(list(argv))
        if argv[:2] == ["ssh-add", "-l"]:
            return _P(ssh_add_l_exit)
        if argv[:2] == ["ssh-agent", "-s"]:
            if ssh_agent_stdout is None:
                raise FileNotFoundError("ssh-agent not on PATH")
            return _P(0, ssh_agent_stdout)
        if argv[:2] == ["ssh-agent", "-k"]:
            return _P(0)   # atexit cleanup call
        if argv[:1] == ["ssh-add"] and len(argv) >= 2:
            return _P(ssh_add_exit)
        # Forward every non-ssh subprocess (git, rsync, etc.) to the
        # real subprocess.run so upstream code paths (provenance,
        # execution) work unchanged.
        return real_run(argv, **kw)

    monkeypatch.setattr(_tr.subprocess, "run", _run)
    return call_log


def test_ensure_ssh_agent_noop_when_keys_already_loaded(monkeypatch, capsys,
                                                          tmp_path):
    """Positive control: `ssh-add -l` exit 0 means the operator already
    has an agent with keys. Auto-setup must be silent -- no hint, no
    spawn, no ssh-add -- just proceed."""
    # Guard against SSH_AUTH_SOCK leaking from an earlier test.
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.delenv("SSH_AGENT_PID", raising=False)
    call_log = _stub_subprocess_run_for_ssh_agent(
        monkeypatch, ssh_add_l_exit=0)

    out = _make_subject_dir(tmp_path)
    transfer_subject(out, ssh_user="testuser", dry_run=True,
                      remote_dir_override="/tmp/dry")
    combined = capsys.readouterr().out + capsys.readouterr().err
    assert "ssh-agent has no keys loaded" not in combined
    assert "started ssh-agent" not in combined
    # Only `ssh-add -l` was called; nothing spawned.
    ssh_related = [c for c in call_log if c[0] in ("ssh-add", "ssh-agent")]
    assert ssh_related == [["ssh-add", "-l"]], ssh_related


def test_ensure_ssh_agent_auto_spawns_and_adds_key(monkeypatch, capsys, tmp_path):
    """No agent, no keys -> auto-spawn ssh-agent, capture its env vars,
    then run `ssh-add <key>` (interactive one-time passphrase prompt).
    Verifies the full auto-setup happy path."""
    # Simulate: no agent env, ssh-add -l exit 1 (no keys), ssh-agent
    # spawn succeeds with realistic-looking output, ssh-add succeeds.
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.delenv("SSH_AGENT_PID", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    # Fake key file so ensure_ssh_agent doesn't bail on "not found".
    fake_key = tmp_path / "id_ed25519"
    fake_key.write_text("stub")

    agent_out = (
        "SSH_AUTH_SOCK=/tmp/ssh-XXX/agent.999; export SSH_AUTH_SOCK;\n"
        "SSH_AGENT_PID=999; export SSH_AGENT_PID;\n"
        "echo Agent pid 999;\n"
    )
    call_log = _stub_subprocess_run_for_ssh_agent(
        monkeypatch, ssh_add_l_exit=1, ssh_agent_stdout=agent_out,
        ssh_add_exit=0)

    out = _make_subject_dir(tmp_path)
    transfer_subject(out, ssh_user="testuser", dry_run=True,
                      remote_dir_override="/tmp/dry",
                      ssh_key=fake_key)

    # Env vars were extracted from the agent stdout and set.
    assert os.environ.get("SSH_AUTH_SOCK") == "/tmp/ssh-XXX/agent.999"
    assert os.environ.get("SSH_AGENT_PID") == "999"
    # Sequence: ssh-add -l (check), ssh-agent -s (spawn), ssh-add <key> (load).
    ssh_related = [c for c in call_log if c[0] in ("ssh-add", "ssh-agent")]
    assert ssh_related[0] == ["ssh-add", "-l"]
    assert ssh_related[1] == ["ssh-agent", "-s"]
    assert ssh_related[2][0] == "ssh-add"
    assert str(fake_key) in ssh_related[2][1]
    combined = capsys.readouterr().out + capsys.readouterr().err
    assert "started ssh-agent" in combined
    assert "loading SSH key" in combined


def test_ensure_ssh_agent_no_tty_prints_manual_hint(monkeypatch, capsys,
                                                     tmp_path):
    """Under nohup / cron / SSH-without-PTY, ssh-add can't prompt for
    the passphrase. Auto-setup must recognise this and print the
    manual-setup hint instead of trying to prompt on a dead stdin."""
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    fake_key = tmp_path / "id_ed25519"
    fake_key.write_text("stub")

    agent_out = ("SSH_AUTH_SOCK=/tmp/s; export SSH_AUTH_SOCK;\n"
                  "SSH_AGENT_PID=1; export SSH_AGENT_PID;\n")
    call_log = _stub_subprocess_run_for_ssh_agent(
        monkeypatch, ssh_add_l_exit=1, ssh_agent_stdout=agent_out)

    out = _make_subject_dir(tmp_path)
    transfer_subject(out, ssh_user="testuser", dry_run=True,
                      remote_dir_override="/tmp/dry",
                      ssh_key=fake_key)
    # ssh-add should NOT be called on the key (no TTY to prompt on),
    # only the `ssh-add -l` probe and the agent spawn.
    ssh_add_key_calls = [c for c in call_log
                          if c[:1] == ["ssh-add"] and len(c) >= 2
                          and c[1] != "-l"]
    assert ssh_add_key_calls == [], (
        f"ssh-add on key must not fire without a TTY: {ssh_add_key_calls}"
    )
    combined = capsys.readouterr().out + capsys.readouterr().err
    assert "no TTY" in combined


def test_ensure_ssh_agent_auto_false_bypasses_setup(monkeypatch, capsys,
                                                     tmp_path):
    """auto=False disables the auto-setup entirely. Useful for external
    agent-management setups (e.g. keychain, keychain-integrated shells).
    Just prints the manual hint if the agent is empty."""
    # Guard against SSH_AUTH_SOCK leaking from an earlier test.
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.delenv("SSH_AGENT_PID", raising=False)
    call_log = _stub_subprocess_run_for_ssh_agent(
        monkeypatch, ssh_add_l_exit=1)

    out = _make_subject_dir(tmp_path)
    transfer_subject(out, ssh_user="testuser", dry_run=True,
                      remote_dir_override="/tmp/dry",
                      auto_ssh_agent=False)
    # Only the probe fired; no spawn attempted.
    assert all(c[0] != "ssh-agent" for c in call_log)
    combined = capsys.readouterr().out + capsys.readouterr().err
    assert "ssh-agent has no keys loaded" in combined
    assert "eval $(ssh-agent)" in combined


def test_ensure_ssh_agent_hints_when_ssh_tooling_completely_missing(
        monkeypatch, capsys, tmp_path):
    """When both ssh-add AND ssh-agent are missing (no SSH tooling
    installed at all), auto-setup can't help. Print the manual-setup
    hint so the operator at least sees the recommended commands, then
    proceed non-fatally. Transfer will still work if the operator has
    alternative auth (host-based, ProxyJump)."""
    import clean_eeg.transfer as _tr
    real_run = _tr.subprocess.run
    # Guard against SSH_AUTH_SOCK leaking from an earlier test that
    # exercised _spawn_ssh_agent (which does a direct os.environ
    # assignment, not monkeypatch.setenv, so it doesn't auto-revert).
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.delenv("SSH_AGENT_PID", raising=False)

    def _run(argv, **kw):
        if argv[0] in ("ssh-add", "ssh-agent"):
            raise FileNotFoundError(f"{argv[0]} not found")
        return real_run(argv, **kw)

    out = _make_subject_dir(tmp_path)
    monkeypatch.setattr(_tr.subprocess, "run", _run)

    # Must not raise -- hint is a warning, not an error.
    transfer_subject(out, ssh_user="testuser", dry_run=True,
                      remote_dir_override="/tmp/dry")
    combined = capsys.readouterr().out + capsys.readouterr().err
    assert "ssh-agent has no keys loaded" in combined
    # Continued past the hint (didn't raise) -- transfer plan built.
