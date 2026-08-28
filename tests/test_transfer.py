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


def test_ssh_agent_check_prints_hint_when_no_keys_loaded(monkeypatch, capsys,
                                                          tmp_path):
    """When `ssh-add -l` exits 1 (no keys / no agent), transfer_subject
    must print the ssh-agent setup hint and continue (non-fatal). Bulk
    transfers otherwise prompt for the passphrase repeatedly."""
    import clean_eeg.transfer as _tr

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "The agent has no identities.\n"

    def _fake_run(argv, **kw):
        if argv[:2] == ["ssh-add", "-l"]:
            return _FakeProc()
        # any OTHER subprocess in transfer_subject (rsync itself) shouldn't
        # fire in dry_run mode, but stub just in case.
        raise AssertionError(f"unexpected subprocess call: {argv}")

    out = _make_subject_dir(tmp_path)
    monkeypatch.setattr(_tr.subprocess, "run", _fake_run)

    plan = _tr.transfer_subject(out, ssh_user="testuser", dry_run=True,
                                 remote_dir_override="/tmp/dry")
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "ssh-agent has no keys loaded" in combined
    assert "eval $(ssh-agent)" in combined
    assert "ssh-add" in combined
    # Non-fatal: plan was still returned.
    assert plan is not None


def test_ssh_agent_check_silent_when_keys_loaded(monkeypatch, capsys, tmp_path):
    """Positive control: `ssh-add -l` exit 0 means keys are loaded;
    the hint must NOT print (silent OK). Regression against a hint
    that spams every invocation regardless of state."""
    import clean_eeg.transfer as _tr

    class _FakeProc:
        returncode = 0
        stdout = "2048 SHA256:abcd... user@host (ED25519)\n"
        stderr = ""

    def _fake_run(argv, **kw):
        if argv[:2] == ["ssh-add", "-l"]:
            return _FakeProc()
        raise AssertionError(f"unexpected subprocess call: {argv}")

    out = _make_subject_dir(tmp_path)
    monkeypatch.setattr(_tr.subprocess, "run", _fake_run)

    _tr.transfer_subject(out, ssh_user="testuser", dry_run=True,
                          remote_dir_override="/tmp/dry")
    combined = capsys.readouterr().out + capsys.readouterr().err
    assert "ssh-agent has no keys loaded" not in combined, (
        f"hint should be silent when keys loaded; got: {combined!r}"
    )


def test_ssh_agent_check_silent_when_ssh_add_missing(monkeypatch, tmp_path):
    """If `ssh-add` isn't on PATH, don't hint -- we can't confidently
    say the operator's SSH auth is broken (they might be on a
    minimal system with different auth). Just proceed silently."""
    import clean_eeg.transfer as _tr

    def _fake_run(argv, **kw):
        if argv[:2] == ["ssh-add", "-l"]:
            raise FileNotFoundError("ssh-add not found")
        raise AssertionError(f"unexpected subprocess call: {argv}")

    out = _make_subject_dir(tmp_path)
    monkeypatch.setattr(_tr.subprocess, "run", _fake_run)

    # Must not raise -- silent fallthrough.
    _tr.transfer_subject(out, ssh_user="testuser", dry_run=True,
                          remote_dir_override="/tmp/dry")
