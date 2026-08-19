"""Tests for the bandwidth-test script.

Real network tests are unnecessary here: the script's job is to
correctly time an rsync subprocess and average the results. We test
the plumbing (argv construction, rounds accounting, cleanup call)
with subprocess.run monkeypatched, plus one true localhost end-to-end
that exercises real rsync in-process.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from clean_eeg.bandwidth_test import (
    RoundResult,
    _rsync_argv,
    main,
    run_bandwidth_test,
)


def test_rsync_argv_uses_localhost_local_path(tmp_path):
    """Positive: when host='localhost', the remote is a bare path (no
    user@host: prefix) so rsync uses local-mode copy."""
    argv = _rsync_argv(
        tmp_path / "payload.bin", "alice", "localhost", "/tmp/scratch")
    remote_arg = argv[-1]
    assert remote_arg == "/tmp/scratch/"
    assert "alice@" not in remote_arg


def test_rsync_argv_uses_user_at_host_for_remote(tmp_path):
    """Positive: real hostname → user@host:path format for SSH."""
    argv = _rsync_argv(
        tmp_path / "payload.bin", "alice", "example.com", "/data/tmp")
    remote_arg = argv[-1]
    assert remote_arg == "alice@example.com:/data/tmp/"


def test_rsync_argv_includes_measurement_flags(tmp_path):
    """The three flags that make the timing meaningful:
    --partial     -> resume-safe if interrupted mid-round
    --whole-file  -> skip delta-xfer, measure raw transport
    a progress flag (--info=progress2 on real rsync, --progress on
                     openrsync/macOS default)
    """
    argv = _rsync_argv(
        tmp_path / "payload.bin", "alice", "example.com", "/data/tmp")
    assert "--partial" in argv
    assert "--whole-file" in argv
    assert "--info=progress2" in argv or "--progress" in argv


def test_run_bandwidth_test_averages_multiple_rounds(monkeypatch, tmp_path):
    """The report's summary should reflect ALL successful rounds --
    min/median/max computed across them. Monkeypatches subprocess.run
    to fake per-round timing without touching the network.
    """
    # Reduce random-byte generation cost — the actual timing loop is
    # what we're testing, not the file write.
    monkeypatch.setattr(
        "clean_eeg.bandwidth_test._write_random_bytes",
        lambda p, n: p.write_bytes(b"\x00" * n))

    # Simulate three rounds with different wall times by monkey-
    # patching subprocess.run and time.perf_counter in lockstep.
    call_count = {"n": 0}
    fake_elapsed = [1.0, 2.0, 4.0]  # seconds per round
    fake_time = [0.0]

    def fake_perf_counter():
        v = fake_time[0]
        return v

    def fake_run(argv, **kwargs):
        i = call_count["n"]
        call_count["n"] += 1
        fake_time[0] += fake_elapsed[i]
        # rsync exit 0
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "perf_counter", fake_perf_counter)

    report = run_bandwidth_test(
        size_mb=1, rounds=3, host="localhost", ssh_user="alice",
        remote_dir=str(tmp_path))

    assert len(report.rounds) == 3
    assert all(r.rsync_exit_code == 0 for r in report.rounds)
    # 1 MB / N s → 1.0, 0.5, 0.25 MBps
    rates = sorted(r.mbps for r in report.rounds)
    assert rates == pytest.approx([0.25, 0.5, 1.0])
    # Summary text mentions min/median/max
    summary = report.summary()
    assert "min=0.2" in summary or "min=0.3" in summary
    assert "max=1.0" in summary


def test_run_bandwidth_test_reports_all_rounds_even_when_all_fail(
        monkeypatch, tmp_path):
    """Negative regression: if rsync exits non-zero every round, we
    still get a report with per-round exit codes preserved. Nothing
    is silently swallowed.
    """
    monkeypatch.setattr(
        "clean_eeg.bandwidth_test._write_random_bytes",
        lambda p, n: p.write_bytes(b"\x00" * n))

    def failing_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, returncode=23)  # rsync partial

    monkeypatch.setattr(subprocess, "run", failing_run)

    report = run_bandwidth_test(
        size_mb=1, rounds=2, host="localhost", ssh_user="alice",
        remote_dir=str(tmp_path))
    assert len(report.rounds) == 2
    assert all(r.rsync_exit_code == 23 for r in report.rounds)
    assert "all rounds failed" in report.summary()


@pytest.mark.skipif(shutil.which("rsync") is None,
                    reason="rsync not installed")
def test_end_to_end_local_transfer_actually_moves_bytes(tmp_path):
    """Real localhost rsync integration: 1 MB payload transferred to
    a scratch dir on the same machine. Confirms the argv shape rsync
    accepts, the file lands on disk, and the report has non-zero
    throughput. Cleanup removes the remote copy.
    """
    dest = tmp_path / "remote_scratch"
    dest.mkdir()
    # Use a tiny size to keep the test fast (~ms even on slow disks)
    report = run_bandwidth_test(
        size_mb=1, rounds=1, host="localhost",
        ssh_user="ignored",  # localhost path skips the user@host format
        remote_dir=str(dest))
    assert report.rounds[0].rsync_exit_code == 0, report.rounds[0]
    assert report.rounds[0].mbps > 0
    # The test file should have been cleaned up
    assert not any(dest.glob("bwtest_*.bin"))


def test_main_writes_json_report(monkeypatch, tmp_path, capsys):
    """CLI: --json-out path writes a complete report JSON."""
    monkeypatch.setattr(
        "clean_eeg.bandwidth_test._write_random_bytes",
        lambda p, n: p.write_bytes(b"\x00" * n))
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **_: subprocess.CompletedProcess(argv, 0))
    out_json = tmp_path / "report.json"
    rc = main([
        "--size-mb", "1", "--rounds", "2",
        "--host", "localhost", "--user", "alice",
        "--remote-dir", str(tmp_path),
        "--json-out", str(out_json),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["host"] == "localhost"
    assert payload["size_mb"] == 1
    assert len(payload["rounds"]) == 2


def test_main_nonzero_exit_when_all_rounds_fail(monkeypatch, tmp_path):
    """CLI exit status conveys failure so a wrapper script can detect
    a bad network before proceeding to the transfer orchestrator.
    """
    monkeypatch.setattr(
        "clean_eeg.bandwidth_test._write_random_bytes",
        lambda p, n: p.write_bytes(b"\x00" * n))
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **_: subprocess.CompletedProcess(argv, 12))
    rc = main([
        "--size-mb", "1", "--rounds", "1",
        "--host", "localhost", "--user", "alice",
        "--remote-dir", str(tmp_path),
    ])
    assert rc == 1
