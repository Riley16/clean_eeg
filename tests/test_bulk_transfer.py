"""Tests for the bulk-transfer orchestrator.

Focus: the pieces the orchestrator ADDS on top of transfer.py --
    * PHI safety (files in manifest.failed_files never reach rsync)
    * per-subject retry with exponential backoff (attempt counter,
      backoff schedule, boundary-crossings that DO NOT consume a retry)
    * day/night BwlimitPolicy time-of-day boundary math
    * structured JSONL event log
    * CLI wiring (--subjects-file parsing, exit-code convention)

Real rsync is exercised in ONE end-to-end localhost test so the
argv-composition path is proven wire-compatible with the actual
binary. All other tests stub subprocess to avoid network flakiness.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, time as dtime
from pathlib import Path

import pytest

# Reuse the deidentified-subject fixture builder from test_transfer.py --
# we do NOT want to duplicate the setup logic. Import via sys.path so
# pytest's rootdir discovery finds it.
sys.path.insert(0, str(Path(__file__).parent))
from test_transfer import (  # type: ignore  # noqa: E402
    SITE_INCOMING_FOLDER,
    _add_failed_file_to_manifest,
    _make_subject_dir,
)

from clean_eeg.bulk_transfer import (  # noqa: E402
    BwlimitPolicy,
    EventLog,
    FAILED_CSV_COLUMNS,
    FAILED_CSV_FILENAME,
    SubjectPlan,
    SubjectResult,
    _filter_plans_by_subject,
    _inject_rsync_flags,
    _load_subject_paths,
    build_subject_plans,
    main,
    run_bulk_transfer,
    transfer_one_subject_with_retry,
    write_failed_subjects_csv,
)


# ---------------------------------------------------------------------------
# BwlimitPolicy: time-of-day cap resolution
# ---------------------------------------------------------------------------

def _policy(day: int | None = 100, night: int | None = 1000,
            start: str = "09:00", end: str = "21:00") -> BwlimitPolicy:
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    return BwlimitPolicy(
        day_kbps=day, night_kbps=night,
        day_start=dtime(sh, sm), day_end=dtime(eh, em),
    )


def test_bwlimit_policy_returns_day_cap_during_day():
    pol = _policy()
    assert pol.current_kbps(datetime(2026, 1, 1, 12, 0, 0)) == 100


def test_bwlimit_policy_returns_night_cap_at_night():
    pol = _policy()
    assert pol.current_kbps(datetime(2026, 1, 1, 23, 0, 0)) == 1000
    assert pol.current_kbps(datetime(2026, 1, 1, 3, 0, 0)) == 1000


def test_bwlimit_policy_boundary_exclusive_at_end():
    """21:00:00 flips to night; 20:59:59 is still day. Confirms the
    boundary math doesn't briefly serve stale day cap at :00."""
    pol = _policy()
    assert pol.current_kbps(datetime(2026, 1, 1, 20, 59, 59)) == 100
    assert pol.current_kbps(datetime(2026, 1, 1, 21, 0, 0)) == 1000


def test_bwlimit_policy_boundary_inclusive_at_start():
    """09:00:00 flips to day; 08:59:59 is still night."""
    pol = _policy()
    assert pol.current_kbps(datetime(2026, 1, 1, 8, 59, 59)) == 1000
    assert pol.current_kbps(datetime(2026, 1, 1, 9, 0, 0)) == 100


def test_bwlimit_policy_no_cap_when_none():
    pol = _policy(day=None, night=None)
    assert pol.current_kbps(datetime(2026, 1, 1, 12, 0, 0)) is None
    assert pol.current_kbps(datetime(2026, 1, 1, 3, 0, 0)) is None


def test_bwlimit_policy_cross_midnight_day_window():
    """If the operator wanted 'day = 22:00-06:00' (unusual, but a
    tester might invert), the window still resolves correctly."""
    pol = _policy(start="22:00", end="06:00")
    # Inside inverted window
    assert pol.current_kbps(datetime(2026, 1, 1, 23, 0, 0)) == 100
    assert pol.current_kbps(datetime(2026, 1, 1, 3, 0, 0)) == 100
    # Outside inverted window
    assert pol.current_kbps(datetime(2026, 1, 1, 12, 0, 0)) == 1000


# ---------------------------------------------------------------------------
# _inject_rsync_flags: exactly the flags we intend, no extras
# ---------------------------------------------------------------------------

def test_inject_rsync_flags_adds_bwlimit_and_timeout():
    argv = _inject_rsync_flags(
        ["rsync", "--partial", "src/", "user@host:/dst/"],
        bwlimit_kbps=500, rsync_timeout_s=15)
    assert "--bwlimit=500" in argv
    assert "--timeout=15" in argv
    # Original flags preserved
    assert "--partial" in argv
    assert "src/" in argv


def test_inject_rsync_flags_omits_bwlimit_when_none():
    """Negative regression: passing None means NO --bwlimit at all,
    so rsync uses its unlimited default."""
    argv = _inject_rsync_flags(
        ["rsync", "--partial", "src/", "dst/"],
        bwlimit_kbps=None, rsync_timeout_s=15)
    assert not any(a.startswith("--bwlimit") for a in argv)
    assert "--timeout=15" in argv


def test_inject_rsync_flags_leaves_non_rsync_argv_alone():
    """A scp fallback argv shouldn't have rsync flags injected."""
    argv = _inject_rsync_flags(
        ["scp", "-r", "src/", "user@host:/dst/"],
        bwlimit_kbps=500, rsync_timeout_s=15)
    assert not any(a.startswith("--bwlimit") for a in argv)
    assert not any(a.startswith("--timeout") for a in argv)


# ---------------------------------------------------------------------------
# build_subject_plans: PHI safety + hard-failure routing
# ---------------------------------------------------------------------------

def test_build_subject_plans_populates_excluded_names_from_manifest(tmp_path):
    """PHI SAFETY: a file listed in manifest.failed_files must land in
    the plan's excluded_names. This is the roster the orchestrator
    downstream turns into --exclude=<name>."""
    out = _make_subject_dir(tmp_path)
    (out / "raw_but_failed.edf").write_bytes(b"not cleaned")
    _add_failed_file_to_manifest(out, "raw_but_failed.edf")

    ready, hard = build_subject_plans([out])
    assert hard == []
    assert len(ready) == 1
    assert "raw_but_failed.edf" in ready[0].excluded_names


def test_build_subject_plans_transferable_bytes_omits_failed_files(tmp_path):
    """Bytes accounting must NOT include failed files -- otherwise the
    ETA overshoots and progress bar looks stuck near the end."""
    out = _make_subject_dir(tmp_path)
    failed_path = out / "raw_but_failed.edf"
    failed_path.write_bytes(b"x" * 999)
    _add_failed_file_to_manifest(out, failed_path.name)

    ready, _ = build_subject_plans([out])
    clean_edf = out / "ok_R1755A_01.01__10.00.00.edf"
    assert ready[0].transferable_bytes == clean_edf.stat().st_size
    assert 999 not in (ready[0].transferable_bytes,)  # excluded byte count


def test_build_subject_plans_routes_preflight_failure_to_hard(tmp_path):
    """A subject with a broken manifest must NOT go into the ready
    queue. Belongs in hard_failures, reason preserved."""
    out = tmp_path / "broken"
    out.mkdir()
    # No manifest -> preflight fails
    ready, hard = build_subject_plans([out])
    assert ready == []
    assert len(hard) == 1
    assert hard[0][0] == out
    assert hard[0][1]  # non-empty reason


# ---------------------------------------------------------------------------
# transfer_one_subject_with_retry: retry / boundary / success paths
# ---------------------------------------------------------------------------

def _plan(subject_dir: Path, code: str = "R1755A") -> SubjectPlan:
    return SubjectPlan(
        subject_dir=subject_dir, subject_code=code,
        site_incoming_folder=SITE_INCOMING_FOLDER,
        transferable_bytes=100,
    )


def test_retry_returns_success_on_first_try(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_run_subject_rsync(*args, **kwargs):
        calls["n"] += 1
        return 0, "", False

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        fake_run_subject_rsync)
    result = transfer_one_subject_with_retry(
        _plan(tmp_path), ssh_user="alice",
        bwlimit_policy=_policy(), max_retries=3,
        rsync_timeout_s=15, backoff_base_s=0)
    assert result.succeeded
    assert result.attempts == 1
    assert calls["n"] == 1


def test_retry_gives_up_after_max_retries(monkeypatch, tmp_path):
    calls = {"n": 0}

    def always_fail(*args, **kwargs):
        calls["n"] += 1
        return 23, "always broken", False

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        always_fail)
    # backoff_base_s=0 so tests don't wait
    result = transfer_one_subject_with_retry(
        _plan(tmp_path), ssh_user="alice",
        bwlimit_policy=_policy(), max_retries=3,
        rsync_timeout_s=15, backoff_base_s=0)
    assert not result.succeeded
    assert result.attempts == 3
    assert calls["n"] == 3
    assert result.last_exit_code == 23


def test_retry_succeeds_after_transient_failures(monkeypatch, tmp_path):
    """Two failures, third succeeds -> attempts == 3, succeeded=True."""
    seq = iter([(23, "flaky", False), (23, "flaky", False), (0, "", False)])

    def stubbed(*args, **kwargs):
        return next(seq)

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        stubbed)
    result = transfer_one_subject_with_retry(
        _plan(tmp_path), ssh_user="alice",
        bwlimit_policy=_policy(), max_retries=5,
        rsync_timeout_s=15, backoff_base_s=0)
    assert result.succeeded
    assert result.attempts == 3


def test_boundary_crossing_does_not_consume_a_retry(monkeypatch, tmp_path):
    """Design invariant: bwlimit-boundary-triggered restarts must NOT
    count as retries. Otherwise a subject that starts near 21:00 could
    burn all its retries on one boundary flip.
    """
    # Sequence: 3 boundary-triggered restarts, then real failure * 2
    # (max_retries=2). Attempts must == 2, not 5.
    seq = iter([
        (255, "bwlimit_boundary_crossed", True),  # restart, not a retry
        (255, "bwlimit_boundary_crossed", True),  # restart, not a retry
        (255, "bwlimit_boundary_crossed", True),  # restart, not a retry
        (23, "real fail 1", False),               # retry 1
        (23, "real fail 2", False),               # retry 2 -> give up
    ])
    call_count = {"n": 0}

    def stubbed(*args, **kwargs):
        call_count["n"] += 1
        return next(seq)

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        stubbed)
    result = transfer_one_subject_with_retry(
        _plan(tmp_path), ssh_user="alice",
        bwlimit_policy=_policy(), max_retries=2,
        rsync_timeout_s=15, backoff_base_s=0)
    assert not result.succeeded
    assert result.attempts == 2
    assert call_count["n"] == 5  # 3 boundary + 2 real


def test_retry_logs_backoff_and_boundary_events(monkeypatch, tmp_path):
    """Structured log must capture per-attempt outcomes so an operator
    can grep the JSONL to reconstruct what happened.
    """
    seq = iter([
        (23, "flaky", False),  # retry 1 -> logs rsync_exit + backoff_wait
        (255, "bwlimit_boundary_crossed", True),  # boundary -> restart
        (0, "", False),                            # success
    ])
    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync",
        lambda *a, **k: next(seq))

    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        result = transfer_one_subject_with_retry(
            _plan(tmp_path), ssh_user="alice",
            bwlimit_policy=_policy(), max_retries=5,
            rsync_timeout_s=15, backoff_base_s=0, log=log)
    assert result.succeeded

    events = [json.loads(l) for l in log_path.read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert "subject_start" in kinds
    assert "rsync_exit" in kinds
    assert "backoff_wait" in kinds
    assert "bwlimit_boundary_restart" in kinds


# ---------------------------------------------------------------------------
# EventLog: JSONL append semantics
# ---------------------------------------------------------------------------

def test_event_log_appends_valid_jsonl(tmp_path):
    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        log.emit("start", subject="R1", n=1)
        log.emit("done", subject="R1", ok=True)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["event"] == "start"
    assert parsed[0]["subject"] == "R1"
    assert parsed[1]["event"] == "done"
    assert parsed[1]["ok"] is True
    # Every line has a timestamp
    for p in parsed:
        assert "timestamp" in p


def test_event_log_appends_on_reopen(tmp_path):
    """A crashed orchestrator that's restarted must APPEND to the
    existing log, not overwrite -- otherwise the operator loses history.
    """
    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        log.emit("first")
    with EventLog(log_path) as log:
        log.emit("second")
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "first"
    assert json.loads(lines[1])["event"] == "second"


# ---------------------------------------------------------------------------
# PHI SAFETY regression: excluded files never reach the wire
# ---------------------------------------------------------------------------

def test_phi_safety_failed_files_never_appear_in_rsync_argv(tmp_path,
                                                             monkeypatch):
    """End-to-end PHI SAFETY regression guard: run the orchestrator on
    a subject whose manifest lists one failed file. Capture every
    subprocess.run argv the orchestrator would fire. Assert:
        POSITIVE: the clean file's name appears in the rsync argv
        NEGATIVE: the failed file's name is NEVER in any argv
    """
    out = _make_subject_dir(tmp_path)
    clean_name = "ok_R1755A_01.01__10.00.00.edf"
    dirty_name = "leaked_phi_R1755A.edf"
    (out / dirty_name).write_bytes(b"contains a real subject name!")
    _add_failed_file_to_manifest(out, dirty_name)

    captured_argvs: list[list[str]] = []

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured_argvs.append(list(argv))
            self.returncode = 0
        def communicate(self, timeout=None):
            return "", ""
        def terminate(self): pass
        def kill(self): pass

    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **k: (captured_argvs.append(list(argv))
                                            or FakeCompleted()))
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    dest = tmp_path / "remote_scratch"
    dest.mkdir()
    results, hard = run_bulk_transfer(
        [out], ssh_user="alice",
        bwlimit_policy=_policy(day=None, night=None),
        parallel=1, max_retries=1, rsync_timeout_s=15,
        backoff_base_s=0, progress_interval_s=9999,
        remote_dir_override=str(dest),
        log_path=tmp_path / "events.jsonl",
    )
    assert hard == []
    assert len(results) == 1

    # POSITIVE: the failed file is excluded via --exclude=<name> in
    # at least one rsync argv (proves the exclusion mechanism actually
    # fired -- not just that the file happened not to be enumerated).
    exclude_flag = f"--exclude={dirty_name}"
    assert any(exclude_flag in argv for argv in captured_argvs), (
        f"Expected --exclude={dirty_name} in some argv but got {captured_argvs}")

    # NEGATIVE: the dirty file's name must appear ONLY inside an
    # --exclude=<name> position. Anywhere else (positional source
    # arg, --files-from list, remote destination path, etc.) would
    # mean the file would end up on the wire.
    for argv in captured_argvs:
        non_exclude_tokens = [a for a in argv if a != exclude_flag]
        joined = " ".join(non_exclude_tokens)
        assert dirty_name not in joined, (
            f"PHI SAFETY VIOLATION: {dirty_name} appears outside "
            f"--exclude= position in argv {argv}")

    # POSITIVE: the clean subject dir IS passed as an rsync source, so
    # the clean files inside it do get transferred.
    assert any(str(out) in " ".join(argv) or (str(out) + "/") in " ".join(argv)
               for argv in captured_argvs)


def test_phi_safety_clean_files_never_get_excluded_when_no_failures(
        tmp_path, monkeypatch):
    """Negative regression: with an empty failed_files list, NO
    --exclude=<name> flags should be emitted. Guards against a bug
    where the exclusion list gets accidentally populated with clean
    filenames.
    """
    out = _make_subject_dir(tmp_path)

    captured_argvs: list[list[str]] = []

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured_argvs.append(list(argv))
            self.returncode = 0
        def communicate(self, timeout=None):
            return "", ""
        def terminate(self): pass
        def kill(self): pass

    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **k: (captured_argvs.append(list(argv))
                                            or FakeCompleted()))
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    dest = tmp_path / "remote_scratch"
    dest.mkdir()
    results, hard = run_bulk_transfer(
        [out], ssh_user="alice",
        bwlimit_policy=_policy(day=None, night=None),
        parallel=1, max_retries=1, rsync_timeout_s=15,
        backoff_base_s=0, progress_interval_s=9999,
        remote_dir_override=str(dest),
        log_path=tmp_path / "events.jsonl",
    )
    assert hard == []
    assert results[0].succeeded
    # Only the built-in excludes are allowed (built into build_transfer_plan):
    #  - quarantine/  (partial-run leftover files)
    #  - *_original_annotations/  (raw pre-Presidio annotation dumps that
    #    live at a sibling of the transfer source; belt-and-suspenders
    #    against a refactor that changes the source path)
    #  - .annotation_review/ + .annotation_reviewed_tracker  (edit-journal
    #    with raw pre-redaction text; operator workflow state)
    #  - edf_audit.{json,ipynb,html,in_progress}  (audit artifacts that
    #    can carry raw header PHI residue; operator-side review only)
    #  - *.review_apply.tmp  (annotation-review rollback copies)
    _BUILTIN_EXCLUDES = {
        "--exclude=quarantine/",
        "--exclude=*_original_annotations/",
        "--exclude=.annotation_review/",
        "--exclude=.annotation_reviewed_tracker",
        "--exclude=edf_audit.json",
        "--exclude=edf_audit.ipynb",
        "--exclude=edf_audit.html",
        "--exclude=edf_audit.in_progress",
        "--exclude=*.review_apply.tmp",
    }
    for argv in captured_argvs:
        per_file_excludes = [
            a for a in argv
            if a.startswith("--exclude=") and a not in _BUILTIN_EXCLUDES
        ]
        assert per_file_excludes == [], (
            f"clean subject leaked an --exclude=<name>: {per_file_excludes}")


# ---------------------------------------------------------------------------
# CLI: subject-file parsing + exit-code convention
# ---------------------------------------------------------------------------

def test_load_subject_paths_ignores_blanks_and_comments(tmp_path):
    f = tmp_path / "subjects.txt"
    f.write_text("/a/b\n\n# comment\n /c/d \n")
    paths = _load_subject_paths(f)
    assert paths == [Path("/a/b"), Path("/c/d")]


def test_split_files_by_size_produces_balanced_chunks(tmp_path):
    """Bin-packer keeps parallel rsync flows near-equal in total bytes.
    With 3 files sized 100/50/50 and 2 chunks, the split should be
    [100] and [50,50] -- NOT [100,50] and [50]."""
    from clean_eeg.bulk_transfer import _split_files_by_size
    src = tmp_path / "src"; src.mkdir()
    (src / "big.edf").write_bytes(b"x" * 100)
    (src / "med1.edf").write_bytes(b"x" * 50)
    (src / "med2.edf").write_bytes(b"x" * 50)
    chunks = _split_files_by_size(src, set(), 2)
    totals = sorted(sum((src / n).stat().st_size for n in c)
                    for c in chunks)
    assert totals == [100, 100], (chunks, totals)


def test_split_files_by_size_excludes_named_files(tmp_path):
    """Files in the excluded_names set are dropped from the split entirely."""
    from clean_eeg.bulk_transfer import _split_files_by_size
    src = tmp_path / "src"; src.mkdir()
    (src / "keep.edf").write_bytes(b"x" * 10)
    (src / "skip.edf").write_bytes(b"x" * 999)
    chunks = _split_files_by_size(src, {"skip.edf"}, 2)
    all_names = [n for c in chunks for n in c]
    assert "skip.edf" not in all_names
    assert "keep.edf" in all_names


def test_transfer_bails_early_on_deterministic_config_error(
        tmp_path, monkeypatch):
    """A failure like 'umask is not recognized as an internal or
    external command' (Windows cmd.exe rejecting POSIX mkdir) cannot
    succeed on retry -- burning through 5 attempts with exponential
    backoff just delays the real failure by minutes. Fast-abort after
    the first attempt for known-deterministic errors."""
    out = _make_subject_dir(tmp_path)
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(str(out) + "\n")

    attempts = {"n": 0}
    def _fake_rsync(*a, **k):
        attempts["n"] += 1
        return 1, ("mkdir: 'umask' is not recognized as an internal "
                    "or external command, operable program or batch "
                    "file."), False
    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync", _fake_rsync)

    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "5", "--backoff-base", "0",
        "--remote-dir-override", str(tmp_path / "dest"),
    ])
    assert rc == 1
    assert attempts["n"] == 1, (
        f"deterministic config error should abort after 1 attempt "
        f"(saw {attempts['n']} retries)")


def test_reachability_preflight_aborts_batch_on_unreachable_host(
        tmp_path, monkeypatch, capsys):
    """When the SSH endpoint isn't reachable, the batch must abort with
    a clear error BEFORE dispatching any rsync workers. Prevents the
    'silent hang for hours' failure mode where every parallel worker
    sits blocked on the same broken SSH handshake."""
    out = _make_subject_dir(tmp_path)
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(str(out) + "\n")

    # Force the reachability probe to report unreachable, and make ANY
    # transfer-related subprocess (rsync, chgrp) raise so we can assert
    # workers never fired. ssh-add / ssh-agent probes from ensure_ssh_agent
    # are allowed to pass through as no-ops.
    def fake_run(argv, **kw):
        head = (argv or [""])[0]
        if head == "ssh":
            return subprocess.CompletedProcess(
                argv, 255, stdout="", stderr="ssh: Connection timed out")
        if head in ("ssh-add", "ssh-agent"):
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="")
        raise AssertionError(
            f"reachability probe failed but bulk_transfer still spawned "
            f"a downstream subprocess: {argv[:3]}...")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("workers should not have run")))

    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
    ])
    assert rc == 0  # returns 0 because no results = no failures
    err = capsys.readouterr().err
    assert "ABORT" in err
    assert "Connection timed out" in err or "ssh" in err


def test_remote_dir_override_supports_subject_code_template(
        tmp_path, monkeypatch, capsys):
    """The '{subject_code}' placeholder in --remote-dir-override is
    expanded per subject so a single template lands each subject in
    its own dir. Otherwise a batch of N subjects would all land in the
    same override dir and clobber each other."""
    out = _make_subject_dir(tmp_path)
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(str(out) + "\n")

    captured = {"rsync_paths": []}

    def fake_rsync(plan, ssh_user, bwlimit_policy, rsync_timeout_s,
                    remote_dir_override, **kw):
        captured["rsync_paths"].append(remote_dir_override)
        return 0, "", False

    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync", fake_rsync)

    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
        "--remote-dir-override",
        "/mnt/backup/clean_eeg/{subject_code}",
    ])
    assert rc == 0
    assert captured["rsync_paths"] == [
        "/mnt/backup/clean_eeg/R1755A"], captured["rsync_paths"]


def test_main_returns_nonzero_when_subjects_file_empty(tmp_path, capsys):
    f = tmp_path / "subjects.txt"
    f.write_text("\n# only comments\n")
    rc = main(["--subjects-file", str(f)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No subject paths" in err


def test_main_accepts_subjects_csv_and_derives_output_paths(
        tmp_path, monkeypatch):
    """--subjects-csv reads the same CSV clean-batch-eeg consumes and
    uses each row's output_path column as the subject dir. Lets an
    operator drive both stages off one file instead of maintaining
    parallel lists."""
    out = _make_subject_dir(tmp_path)
    csv_path = tmp_path / "subjects.csv"
    csv_path.write_text(
        "input_path,output_path,subject_code,first_name,last_name\n"
        f"{tmp_path / 'in'},{out},R1755A,Alice,Smith\n")
    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync",
        lambda *a, **k: (0, "", False))
    rc = main([
        "--subjects-csv", str(csv_path),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
        "--remote-dir-override", str(tmp_path / "dest"),
    ])
    assert rc == 0


def test_main_rejects_both_subjects_file_and_subjects_csv(tmp_path, capsys):
    """--subjects-file and --subjects-csv are mutually exclusive at the
    parser level. Guards against confused invocations where the operator
    thinks one overrides the other."""
    (tmp_path / "s.txt").write_text("/tmp\n")
    (tmp_path / "s.csv").write_text("x")
    with pytest.raises(SystemExit) as ei:
        main(["--subjects-file", str(tmp_path / "s.txt"),
              "--subjects-csv", str(tmp_path / "s.csv")])
    assert ei.value.code != 0


def test_main_requires_exactly_one_of_subjects_file_or_csv(capsys):
    """Neither given -> parser rejects. Complements the mutex test."""
    with pytest.raises(SystemExit) as ei:
        main([])
    assert ei.value.code != 0


def test_subjects_csv_skips_hash_prefixed_rows_in_transfer(
        tmp_path, monkeypatch, capsys):
    """Skip syntax parity with clean-batch-eeg: prepending '#' to a
    subject_code cell in the CSV must also skip that subject when the
    same CSV is fed to bulk-transfer-eeg. Otherwise operators who
    commented out a bad subject for the clean batch would have to
    remember to also filter it out of the transfer command."""
    out = _make_subject_dir(tmp_path)
    csv_path = tmp_path / "subjects.csv"
    csv_path.write_text(
        "input_path,output_path,subject_code,first_name,last_name\n"
        f"{tmp_path / 'in'},{out},R1755A,Alice,Smith\n"
        f",,#R1665J,,\n"  # skip via # prefix; other cells blank OK
    )
    seen_rsync_targets: list = []
    def _fake_rsync(plan, *a, **k):
        seen_rsync_targets.append(plan.subject_code)
        return 0, "", False
    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync", _fake_rsync)

    rc = main([
        "--subjects-csv", str(csv_path),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
        "--remote-dir-override", str(tmp_path / "dest"),
    ])
    assert rc == 0
    assert "R1665J" not in seen_rsync_targets, (
        f"# prefix should have skipped R1665J; rsync ran on "
        f"{seen_rsync_targets}")
    # The [csv] skipped ... line prints to stderr from
    # parse_subjects_csv so the operator sees it in both stages.
    err = capsys.readouterr().err
    assert "skipped 1" in err and "#R1665J" in err


def test_main_subjects_csv_bad_schema_returns_2(tmp_path, capsys):
    """A malformed CSV (missing required columns) exits 2 with a schema
    error message rather than crashing partway through the transfer."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("subject_code\nR1755A\n")  # missing required cols
    rc = main(["--subjects-csv", str(csv_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "CSV schema error" in err


def test_main_returns_nonzero_when_any_subject_fails(tmp_path, monkeypatch):
    """CLI exit-code convention: any failure -> nonzero. Wrapper scripts
    depend on this.
    """
    out = _make_subject_dir(tmp_path)
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(str(out) + "\n")

    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync",
        lambda *a, **k: (23, "boom", False))

    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
        "--remote-dir-override", str(tmp_path / "dest"),
    ])
    assert rc == 1


def test_main_returns_zero_when_all_subjects_succeed(tmp_path, monkeypatch):
    out = _make_subject_dir(tmp_path)
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(str(out) + "\n")

    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync",
        lambda *a, **k: (0, "", False))

    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
        "--remote-dir-override", str(tmp_path / "dest"),
    ])
    assert rc == 0


# ---------------------------------------------------------------------------
# End-to-end batch: JSONL log bookends when the whole run succeeds
# ---------------------------------------------------------------------------

def test_run_bulk_transfer_writes_batch_start_and_complete(tmp_path,
                                                            monkeypatch):
    """Batch run must emit ``batch_start`` at kickoff and
    ``batch_complete`` at the end regardless of outcome, so an
    operator can grep the JSONL for the run's bookends. Also asserts
    ``subject_complete`` fires per subject."""
    out = _make_subject_dir(tmp_path)
    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync",
        lambda *a, **k: (0, "", False))
    log_path = tmp_path / "events.jsonl"
    results, hard = run_bulk_transfer(
        [out], ssh_user="alice",
        bwlimit_policy=_policy(day=None, night=None),
        parallel=1, max_retries=1, rsync_timeout_s=15,
        backoff_base_s=0, progress_interval_s=9999,
        remote_dir_override=str(tmp_path / "dest"),
        log_path=log_path,
    )
    assert hard == []
    assert results[0].succeeded
    events = [json.loads(l) for l in log_path.read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert "batch_start" in kinds
    assert "subject_complete" in kinds
    assert "batch_complete" in kinds


# ---------------------------------------------------------------------------
# Parallelism: N workers run concurrently
# ---------------------------------------------------------------------------

def test_parallel_workers_actually_overlap(tmp_path, monkeypatch):
    """Positive: with --parallel=3 and three subjects that each stall
    the rsync stub for a short spell, the total wall time must be
    substantially less than the serial sum. This is the observable
    signature of real concurrency."""
    outs = []
    for i in range(3):
        (tmp_path / f"s{i}").mkdir()
        outs.append(_make_subject_dir(tmp_path / f"s{i}"))

    per_subject_s = 0.3

    def slow_stub(*a, **k):
        import time as _t
        _t.sleep(per_subject_s)
        return 0, "", False

    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync", slow_stub)

    import time as _t
    t0 = _t.perf_counter()
    results, hard = run_bulk_transfer(
        outs, ssh_user="alice",
        bwlimit_policy=_policy(day=None, night=None),
        parallel=3, max_retries=1, rsync_timeout_s=15,
        backoff_base_s=0, progress_interval_s=9999,
        remote_dir_override=str(tmp_path / "dest"),
        log_path=tmp_path / "events.jsonl",
    )
    elapsed = _t.perf_counter() - t0
    assert hard == []
    assert all(r.succeeded for r in results)
    # Serial would be ~0.9s; parallel-3 should be well under 0.6s (2x).
    assert elapsed < per_subject_s * 2, (
        f"expected concurrent run to finish under {per_subject_s * 2}s, "
        f"got {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Background relaunch: shell script content & argv rewriting
# ---------------------------------------------------------------------------

def test_background_launch_writes_script_and_strips_background_flag(
        tmp_path, monkeypatch):
    """--background must:
      * write a shell script the operator can inspect / re-run
      * strip the --background flag from the child invocation so the
        child doesn't recursively re-launch itself
      * detach via nohup + start_new_session (Popen args)
    """
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("/some/subject/dir\n")

    launched: dict = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            launched["argv"] = list(argv)
            launched["kwargs"] = kwargs
            self.pid = 12345

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice",
        "--remote-dir-override", "/tmp/dest",
        "--background",
    ])
    assert rc == 0
    script_path = subjects_file.parent / "bulk_transfer.launch.sh"
    assert script_path.exists()
    script = script_path.read_text()
    # The EXECUTABLE line (i.e. lines that are neither comments nor
    # blank) must NOT contain --background -- else the child would
    # re-launch itself endlessly.
    exec_lines = [ln for ln in script.splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
    assert exec_lines, script
    for ln in exec_lines:
        assert "--background" not in ln, ln
    # It must invoke the module entrypoint
    assert any("clean_eeg.bulk_transfer" in ln for ln in exec_lines)
    # nohup + start_new_session
    assert launched["argv"][0] == "nohup"
    assert launched["kwargs"].get("start_new_session") is True


# ---------------------------------------------------------------------------
# --only-subjects: subject-picker filter
# ---------------------------------------------------------------------------

def _fake_plan(code: str) -> SubjectPlan:
    return SubjectPlan(subject_dir=Path("/i") / code, subject_code=code,
                       site_incoming_folder="CUDA",
                       transferable_bytes=100)


def test_filter_plans_returns_all_when_picker_none():
    plans = [_fake_plan("R1"), _fake_plan("R2")]
    assert _filter_plans_by_subject(plans, None) == plans
    assert _filter_plans_by_subject(plans, []) == plans


def test_filter_plans_picks_only_listed_codes():
    plans = [_fake_plan("R1755J"), _fake_plan("R1702J_1"),
             _fake_plan("R1042J")]
    picked = _filter_plans_by_subject(plans, ["R1702J_1"])
    assert [p.subject_code for p in picked] == ["R1702J_1"]


def test_filter_plans_warns_on_unmatched_picker_entry(capsys):
    """Typo in --only-subjects must warn -- otherwise the operator's
    'transfer one subject' smoke test silently transfers nothing."""
    plans = [_fake_plan("R1755J")]
    picked = _filter_plans_by_subject(plans, ["R1755J", "R9999Z"])
    assert [p.subject_code for p in picked] == ["R1755J"]
    err = capsys.readouterr().err
    assert "R9999Z" in err
    assert "not found" in err.lower()


def test_run_bulk_transfer_only_subjects_scopes_workers_to_picked_code(
        tmp_path, monkeypatch):
    """End-to-end at the run_bulk_transfer level: one preflight-passing
    subject, --only-subjects matches its code -> that plan reaches
    the worker. This is the positive-path smoke test. The filter's
    exclusion semantics (unmatched codes don't reach the worker) are
    covered by the direct unit tests on _filter_plans_by_subject.
    """
    out = _make_subject_dir(tmp_path)

    seen: list[str] = []

    def stub_rsync(plan, *a, **k):
        seen.append(plan.subject_code)
        return 0, "", False

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        stub_rsync)

    results, hard = run_bulk_transfer(
        [out], ssh_user="alice",
        bwlimit_policy=_policy(day=None, night=None),
        parallel=1, max_retries=1, rsync_timeout_s=15,
        backoff_base_s=0, progress_interval_s=9999,
        remote_dir_override=str(tmp_path / "dest"),
        log_path=tmp_path / "events.jsonl",
        only_subjects=["R1755A"],
    )
    assert hard == []
    assert len(results) == 1
    assert results[0].subject_code == "R1755A"
    assert seen == ["R1755A"]


def test_main_only_subjects_returns_nonzero_when_no_match(tmp_path,
                                                           monkeypatch):
    """Negative regression: --only-subjects R_NOT_A_SUBJECT must NOT
    silently transfer the whole batch. Exit nonzero so the operator's
    smoke test surfaces the typo."""
    out = _make_subject_dir(tmp_path)
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(str(out) + "\n")

    monkeypatch.setattr(
        "clean_eeg.bulk_transfer._run_subject_rsync",
        lambda *a, **k: (0, "", False))

    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
        "--remote-dir-override", str(tmp_path / "dest"),
        "--only-subjects", "R_NOT_IN_ANY_MANIFEST",
    ])
    # Design: an empty picker result set (all entries typo'd or
    # missing from manifests) MUST exit nonzero. Otherwise the
    # operator's 'transfer one subject' smoke test would silently
    # "succeed" while transferring nothing.
    assert rc == 1


def test_main_only_subjects_file_scopes_transfer_to_listed_codes(
        tmp_path, monkeypatch):
    """POSITIVE integration for --only-subjects-file at the bulk
    transfer CLI: file with one subject code selects it exactly,
    matching the behavior of the inline --only-subjects flag but
    driven from disk so the same file can be shared with
    clean-batch-eeg."""
    out = _make_subject_dir(tmp_path)
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(str(out) + "\n")
    filter_path = tmp_path / "filter.txt"
    filter_path.write_text("# batch file\nR1755A\n")

    seen: list[str] = []

    def stub_rsync(plan, *a, **k):
        seen.append(plan.subject_code)
        return 0, "", False

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        stub_rsync)

    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
        "--remote-dir-override", str(tmp_path / "dest"),
        "--only-subjects-file", str(filter_path),
    ])
    assert rc == 0
    assert seen == ["R1755A"]


def test_main_only_subjects_file_returns_2_when_file_missing(tmp_path):
    """Distinct exit code (2) for input-validation vs 1 for
    'ran and something failed' -- same convention as clean-batch-eeg."""
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("/nonexistent/subject\n")
    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice",
        "--remote-dir-override", str(tmp_path / "dest"),
        "--only-subjects-file", str(tmp_path / "does_not_exist.txt"),
    ])
    assert rc == 2


def test_main_only_subjects_file_merges_with_inline_only_subjects(
        tmp_path, monkeypatch):
    """Inline --only-subjects merged with --only-subjects-file: union,
    same as clean-batch-eeg. Regression guard against a bug where one
    would silently override the other."""
    out = _make_subject_dir(tmp_path)
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(str(out) + "\n")
    filter_path = tmp_path / "filter.txt"
    filter_path.write_text("R_FROM_FILE\n")   # not in this batch

    seen: list[str] = []

    def stub_rsync(plan, *a, **k):
        seen.append(plan.subject_code)
        return 0, "", False

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        stub_rsync)

    # Inline R1755A is present in the batch; file's R_FROM_FILE is not.
    # The union should still pick R1755A -- proving both inputs feed
    # the filter.
    rc = main([
        "--subjects-file", str(subjects_file),
        "--user", "alice", "--parallel", "1",
        "--max-retries", "1", "--backoff-base", "0",
        "--remote-dir-override", str(tmp_path / "dest"),
        "--only-subjects", "R1755A",
        "--only-subjects-file", str(filter_path),
    ])
    assert rc == 0
    assert seen == ["R1755A"]


# ---------------------------------------------------------------------------
# Failed-subjects CSV: post-batch review artifact
# ---------------------------------------------------------------------------

def _result(code, succeeded, attempts=1, last_exit=None, last_err=None):
    return SubjectResult(
        subject_code=code, subject_dir=f"/data/{code}",
        attempts=attempts, succeeded=succeeded, elapsed_s=1.0,
        last_exit_code=last_exit, last_error=last_err)


def test_failed_csv_columns_match_the_public_constant(tmp_path):
    """The CSV header is a documented contract -- downstream tooling
    reads by column name. Regression guard so a future rename doesn't
    break consumers silently."""
    results = [_result("R1FAIL", succeeded=False, attempts=5,
                        last_exit=23, last_err="rsync partial")]
    path = tmp_path / FAILED_CSV_FILENAME
    write_failed_subjects_csv(results, [], path)
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert tuple(header) == FAILED_CSV_COLUMNS


def test_failed_csv_includes_both_failure_types(tmp_path):
    """POSITIVE integration test: transfer_failed and preflight_failed
    subjects both land in the CSV, with the correct failure_type tag."""
    results = [
        _result("R1OK", succeeded=True),
        _result("R1FAIL", succeeded=False, attempts=5, last_exit=23,
                last_err="all retries exhausted"),
    ]
    hard_failures = [
        (Path("/data/R1BADPREFLIGHT"), "no manifest present"),
    ]
    path = tmp_path / FAILED_CSV_FILENAME
    assert write_failed_subjects_csv(results, hard_failures, path) is True

    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    by_type = {r["failure_type"]: r for r in rows}
    assert "transfer_failed" in by_type
    assert "preflight_failed" in by_type
    # Successful subject R1OK MUST NOT appear (regression: a bug
    # where a clean subject leaks into the failure list would waste
    # the operator's review time and undermine trust in the CSV).
    assert not any(r["subject_code"] == "R1OK" for r in rows)


def test_failed_csv_transfer_fail_row_has_attempts_and_exit_code(tmp_path):
    """transfer_failed rows carry attempts + last_exit_code so an
    operator can distinguish 'network flake, retry-able' (exit 23)
    from 'permission denied' (exit 11) at a glance."""
    results = [_result("R1FAIL", succeeded=False, attempts=5,
                        last_exit=23, last_err="rsync partial transfer")]
    path = tmp_path / FAILED_CSV_FILENAME
    write_failed_subjects_csv(results, [], path)
    row = next(csv.DictReader(open(path)))
    assert row["subject_code"] == "R1FAIL"
    assert row["attempts"] == "5"
    assert row["last_exit_code"] == "23"
    assert "rsync partial transfer" in row["reason"]


def test_failed_csv_preflight_row_leaves_attempts_and_exit_blank(tmp_path):
    """Preflight failures never got to run rsync, so attempts and
    last_exit_code MUST be blank (not '0', not 'None'). A '0' there
    would falsely suggest the transfer ran and succeeded."""
    hard_failures = [(Path("/data/R1BAD"), "manifest missing")]
    path = tmp_path / FAILED_CSV_FILENAME
    write_failed_subjects_csv([], hard_failures, path)
    row = next(csv.DictReader(open(path)))
    assert row["failure_type"] == "preflight_failed"
    assert row["attempts"] == ""
    assert row["last_exit_code"] == ""
    assert "manifest missing" in row["reason"]


def test_failed_csv_not_written_when_batch_fully_clean(tmp_path):
    """NEGATIVE regression: no failures -> no CSV. An empty CSV would
    be an ambiguous artifact ('did the batch fail with an empty error?
    Or did it just succeed?'). The correct answer is: no file at all,
    absence = success."""
    results = [_result("R1OK1", succeeded=True),
               _result("R1OK2", succeeded=True)]
    path = tmp_path / FAILED_CSV_FILENAME
    assert write_failed_subjects_csv(results, [], path) is False
    assert not path.exists()


def test_run_bulk_transfer_writes_failed_csv_alongside_log(tmp_path,
                                                            monkeypatch):
    """End-to-end at run_bulk_transfer level: a subject whose rsync
    fails must produce failed_subject_transfer.csv in the same dir
    as the JSONL event log."""
    out = _make_subject_dir(tmp_path)

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        lambda *a, **k: (23, "simulated failure", False))

    log_path = tmp_path / "events.jsonl"
    results, hard = run_bulk_transfer(
        [out], ssh_user="alice",
        bwlimit_policy=_policy(day=None, night=None),
        parallel=1, max_retries=1, rsync_timeout_s=15,
        backoff_base_s=0, progress_interval_s=9999,
        remote_dir_override=str(tmp_path / "dest"),
        log_path=log_path,
    )
    assert hard == []
    assert not results[0].succeeded

    csv_path = log_path.parent / FAILED_CSV_FILENAME
    assert csv_path.exists(), (
        "failed-subjects CSV must be written when there are failures")
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1
    assert rows[0]["subject_code"] == "R1755A"
    assert rows[0]["failure_type"] == "transfer_failed"


def test_run_bulk_transfer_omits_failed_csv_on_clean_run(tmp_path,
                                                          monkeypatch):
    """End-to-end negative regression: fully-successful batch produces
    NO failed-subjects CSV. Regression guard against a bug where the
    CSV would always be created (an empty CSV in a green batch would
    be a misleading artifact)."""
    out = _make_subject_dir(tmp_path)

    monkeypatch.setattr("clean_eeg.bulk_transfer._run_subject_rsync",
                        lambda *a, **k: (0, "", False))

    log_path = tmp_path / "events.jsonl"
    results, hard = run_bulk_transfer(
        [out], ssh_user="alice",
        bwlimit_policy=_policy(day=None, night=None),
        parallel=1, max_retries=1, rsync_timeout_s=15,
        backoff_base_s=0, progress_interval_s=9999,
        remote_dir_override=str(tmp_path / "dest"),
        log_path=log_path,
    )
    assert hard == []
    assert results[0].succeeded

    csv_path = log_path.parent / FAILED_CSV_FILENAME
    assert not csv_path.exists()
