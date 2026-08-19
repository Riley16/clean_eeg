"""Measure sustained rsync throughput to the CML server.

Run this BEFORE sizing the bulk transfer orchestrator's
``--bwlimit-day`` / ``--bwlimit-night`` values. Two runs recommended:
one during the day (typical daytime interference), one overnight
(quiet network). The transfer orchestrator caps at half the measured
daytime max between 09:00 and 21:00, and at the full nighttime max
otherwise.

The test transfers a single file of ``--size-mb`` random bytes (default
1 GB) via rsync ``--partial --info=progress2 --whole-file``, waits for
completion, computes MBps, then deletes the remote copy. Repeats
``--rounds`` times (default 3) to average out startup / TCP-slow-start
overhead.

``--whole-file`` disables rsync's delta algorithm — we don't want to
measure "how fast can rsync figure out this identical file didn't
change", we want raw transport throughput.

Uses the same host + user resolution as ``transfer_subject``, so no
new config is needed. Test-mode alternative: ``--remote-dir-override``
lets you point at a scratch dir on the same host (or ``localhost``) for
end-to-end verification without touching the real CML incoming tree.
"""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HOST = "rhino2.psych.upenn.edu"
DEFAULT_REMOTE_TMPDIR = "/tmp"    # writable + local to the CML server
DEFAULT_SIZE_MB = 1000
DEFAULT_ROUNDS = 3


@dataclass
class RoundResult:
    round_index: int
    size_bytes: int
    wall_seconds: float
    mbps: float
    rsync_exit_code: int


@dataclass
class BandwidthTestReport:
    host: str
    remote_dir: str
    ssh_user: str
    size_mb: int
    rounds: list[RoundResult]
    started_at: str
    finished_at: str

    def summary(self) -> str:
        if not self.rounds:
            return "no rounds ran"
        successful = [r for r in self.rounds if r.rsync_exit_code == 0]
        if not successful:
            return "all rounds failed; see per-round log above"
        rates = [r.mbps for r in successful]
        return (
            f"host={self.host}  user={self.ssh_user}  remote_dir={self.remote_dir}\n"
            f"rounds={len(successful)}/{len(self.rounds)} successful  "
            f"size_per_round={self.size_mb} MB\n"
            f"MBps min={min(rates):.1f}  median={statistics.median(rates):.1f}  "
            f"max={max(rates):.1f}\n"
            f"Recommended orchestrator flags (rsync --bwlimit uses KBps):\n"
            f"  --bwlimit-night {int(min(rates) * 1024)}   "
            f"# conservative floor of measured throughput\n"
            f"  --bwlimit-day   {int(min(rates) * 1024 * 0.5)}   "
            f"# half the measured floor (daytime courtesy cap)\n"
        )


def _write_random_bytes(path: Path, size_bytes: int) -> None:
    """Fill ``path`` with random bytes via /dev/urandom to defeat any
    upstream compression / dedup. Streams in 4 MiB chunks so peak RAM
    stays constant regardless of ``size_bytes``."""
    remaining = size_bytes
    chunk = 4 * 1024 * 1024
    with open("/dev/urandom", "rb") as src, open(path, "wb") as dst:
        while remaining > 0:
            n = min(chunk, remaining)
            dst.write(src.read(n))
            remaining -= n


from functools import lru_cache


@lru_cache(maxsize=1)
def _rsync_supports_info_progress() -> bool:
    """rsync >=3 supports --info=progress2 (single-line overall).
    Openrsync (macOS default) does NOT. Falls back to --progress.
    Cached so timing-loop tests can monkeypatch subprocess.run without
    the version probe consuming a stubbed call.
    """
    try:
        v = subprocess.run(["rsync", "--version"], capture_output=True,
                           text=True, timeout=5)
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return False
    first = (v.stdout or v.stderr or "").splitlines()[:1]
    return bool(first) and "openrsync" not in first[0].lower() \
        and "protocol version 2" not in first[0].lower()


def _rsync_argv(local_path: Path, ssh_user: str, host: str, remote_dir: str
                ) -> list[str]:
    remote = (f"{ssh_user}@{host}:{remote_dir}/"
              if host != "localhost" else f"{remote_dir}/")
    progress_flag = ("--info=progress2" if _rsync_supports_info_progress()
                     else "--progress")
    return [
        "rsync",
        "--partial",             # keep partial on interrupt
        progress_flag,           # single-line overall or per-file
        "--whole-file",          # skip delta-xfer, measure raw throughput
        "--times",               # preserve mtime so back-to-back rounds
                                  # don't confuse rsync's identity check
        str(local_path),
        remote,
    ]


def _run_round(local_path: Path, ssh_user: str, host: str, remote_dir: str,
               size_bytes: int, round_index: int) -> RoundResult:
    argv = _rsync_argv(local_path, ssh_user, host, remote_dir)
    print(f"\n=== Round {round_index + 1}: {' '.join(argv)} ===", flush=True)
    start = time.perf_counter()
    proc = subprocess.run(argv)
    elapsed = time.perf_counter() - start
    mbps = (size_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
    print(f"    -> {elapsed:.1f} s   {mbps:.1f} MBps   exit {proc.returncode}",
          flush=True)
    return RoundResult(
        round_index=round_index,
        size_bytes=size_bytes,
        wall_seconds=elapsed,
        mbps=mbps,
        rsync_exit_code=proc.returncode,
    )


def _cleanup_remote(ssh_user: str, host: str, remote_dir: str,
                    filename: str) -> None:
    """Best-effort delete of the test file on the remote. Non-fatal
    if it fails — leaves a diagnostic message but doesn't block."""
    if host == "localhost":
        target = Path(remote_dir) / filename
        try:
            target.unlink(missing_ok=True)
        except OSError as e:
            print(f"    (cleanup: could not remove {target}: {e})")
        return
    cmd = ["ssh", f"{ssh_user}@{host}", f"rm -f {remote_dir}/{filename}"]
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"    (cleanup: rm on remote failed: {e})")


def run_bandwidth_test(size_mb: int = DEFAULT_SIZE_MB,
                       rounds: int = DEFAULT_ROUNDS,
                       host: str = DEFAULT_HOST,
                       ssh_user: str | None = None,
                       remote_dir: str = DEFAULT_REMOTE_TMPDIR,
                       ) -> BandwidthTestReport:
    """Programmatic entry point (also used by the CLI)."""
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync not on PATH; install rsync first")

    ssh_user = ssh_user or getpass.getuser()
    size_bytes = size_mb * 1024 * 1024

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with tempfile.NamedTemporaryFile(
        prefix="bwtest_", suffix=".bin", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"Generating {size_mb} MB of random bytes at {tmp_path} ...",
              flush=True)
        _write_random_bytes(tmp_path, size_bytes)
        print(f"    -> {tmp_path.stat().st_size} bytes on disk", flush=True)

        results: list[RoundResult] = []
        for i in range(rounds):
            r = _run_round(tmp_path, ssh_user, host, remote_dir,
                           size_bytes, i)
            results.append(r)
    finally:
        tmp_path.unlink(missing_ok=True)
        _cleanup_remote(ssh_user, host, remote_dir, tmp_path.name)

    return BandwidthTestReport(
        host=host, remote_dir=remote_dir, ssh_user=ssh_user,
        size_mb=size_mb, rounds=results,
        started_at=started_at,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bandwidth-test-cml",
        description=(
            "Measure sustained rsync throughput to the CML server. "
            "Recommended two runs (day + night) before sizing the bulk "
            "transfer orchestrator's --bwlimit-day / --bwlimit-night flags."
        ),
    )
    p.add_argument("--size-mb", type=int, default=DEFAULT_SIZE_MB,
                   help="Per-round transfer size in MB (default: 1000 = 1 GB).")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help="Number of rounds to average (default: 3).")
    p.add_argument("--host", type=str, default=DEFAULT_HOST,
                   help=f"SSH host (default: {DEFAULT_HOST}). "
                        "Pass 'localhost' for a same-machine smoke test.")
    p.add_argument("--user", type=str, default=None,
                   help="SSH user (default: $USER).")
    p.add_argument("--remote-dir", type=str, default=DEFAULT_REMOTE_TMPDIR,
                   help=f"Remote scratch dir (default: {DEFAULT_REMOTE_TMPDIR}).")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Write a full report as JSON to this path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_bandwidth_test(
        size_mb=args.size_mb, rounds=args.rounds,
        host=args.host, ssh_user=args.user,
        remote_dir=args.remote_dir,
    )
    print("\n=== SUMMARY ===")
    print(report.summary())
    if args.json_out:
        args.json_out.write_text(json.dumps({
            "host": report.host,
            "remote_dir": report.remote_dir,
            "ssh_user": report.ssh_user,
            "size_mb": report.size_mb,
            "started_at": report.started_at,
            "finished_at": report.finished_at,
            "rounds": [r.__dict__ for r in report.rounds],
        }, indent=2))
        print(f"JSON report written to {args.json_out}")
    return 0 if any(r.rsync_exit_code == 0 for r in report.rounds) else 1


if __name__ == "__main__":
    sys.exit(main())
