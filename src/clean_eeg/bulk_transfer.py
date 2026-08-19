"""Bulk multi-subject transfer to the CML server.

Orchestrates ``transfer_subject`` (from ``transfer.py``) across many
cleaned subject folders, with:
  - N-parallel rsync workers (default 4)
  - per-subject retry with exponential backoff (default 5 tries)
  - day/night bandwidth caps that restart running rsyncs when the
    time-of-day boundary crosses (rsync ``--partial`` makes restart
    cheap, so the operator gets the correct bwlimit for the current
    time slot even mid-large-transfer)
  - structured JSONL event log (one line per subject_start / retry /
    subject_complete / bwlimit_change / batch_summary)
  - overall progress + ETA (sum-of-non-failed bytes across subjects
    divided by observed 5-minute rate)
  - PHI-safety inherited from ``transfer_subject``:
    manifest.failed_files entries are excluded via rsync
    ``--exclude=<name>`` per file, and preflight fails hard on any
    unlisted bad file in the subject dir.

The orchestrator is meant to survive multi-week runs. Every unit of
work (one subject transfer) is idempotent -- rsync's ``--partial``
means an interrupted subject resumes from the last checkpoint on the
next attempt, so a killed / crashed / bwlimit-restarted worker never
needs to redo work already on the wire.
"""

from __future__ import annotations

import argparse
import getpass
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARALLEL = 4
DEFAULT_MAX_RETRIES = 5
DEFAULT_RSYNC_TIMEOUT_S = 15
DEFAULT_BACKOFF_BASE_S = 30
# Progress print cadence. 30 s is fast enough that an operator watching
# the log sees updates, slow enough not to spam a multi-week transfer.
DEFAULT_PROGRESS_INTERVAL_S = 30
# Day / night defaults. Operator can override via CLI flags.
DEFAULT_DAY_START = "09:00"
DEFAULT_DAY_END = "21:00"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SubjectPlan:
    """One subject's transferable metadata computed at preflight time."""
    subject_dir: Path
    subject_code: str
    site_incoming_folder: str
    transferable_bytes: int      # sum of non-failed cleaned files
    excluded_names: set[str] = field(default_factory=set)


@dataclass
class SubjectResult:
    """Outcome of a single subject's transfer, after all retries."""
    subject_code: str
    subject_dir: str
    attempts: int
    succeeded: bool
    elapsed_s: float
    last_exit_code: int | None = None
    last_error: str | None = None


# ---------------------------------------------------------------------------
# Bandwidth policy (time-of-day → --bwlimit)
# ---------------------------------------------------------------------------

def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))


@dataclass
class BwlimitPolicy:
    """Day/night rsync bandwidth caps. Both values are KBps (what
    rsync's ``--bwlimit`` accepts). ``None`` = no cap (rsync omits
    the flag).
    """
    day_kbps: int | None
    night_kbps: int | None
    day_start: dtime
    day_end: dtime

    def current_kbps(self, now: datetime | None = None) -> int | None:
        """Return the cap that applies at ``now`` (default: real wall
        clock). Non-inclusive at day_end so a boundary at 21:00 flips
        to night at exactly 21:00:00.
        """
        now = now or datetime.now()
        t = now.time()
        in_day_window = (
            (self.day_start <= t < self.day_end)
            if self.day_start <= self.day_end
            # Cross-midnight (day_start > day_end) window: day = t >= start OR t < end
            else (t >= self.day_start or t < self.day_end)
        )
        return self.day_kbps if in_day_window else self.night_kbps


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

class EventLog:
    """Append-only JSONL sink for orchestrator events. One dict per
    line, ``timestamp`` (ISO 8601) auto-injected. Process-safe append
    via O_APPEND (single writer per file assumed -- the orchestrator
    process is the only writer)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, event: str, **fields) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Preflight: build the work plan
# ---------------------------------------------------------------------------

def _sum_transferable_bytes(subject_dir: Path,
                            excluded_names: set[str]) -> int:
    """Sum sizes of every *.edf file in subject_dir that isn't in
    excluded_names. Used to seed the overall progress denominator."""
    total = 0
    for p in subject_dir.iterdir():
        if (p.is_file() and p.suffix.lower() == ".edf"
                and p.name not in excluded_names):
            total += p.stat().st_size
    return total


def build_subject_plans(subject_dirs: Iterable[Path]) -> tuple[list[SubjectPlan],
                                                                list[tuple[Path, str]]]:
    """Preflight every subject; return ``(ready_plans, hard_failures)``.

    ``hard_failures`` are subjects whose preflight refuses to run --
    they never get scheduled and appear in the batch summary as
    ``preflight_failed`` (distinct from "transfer attempted and failed
    all retries"). Reasons: no manifest, unlisted bad files, wrong
    filename pattern, header mismatch, etc.

    PHI safety: ``manifest.failed_files`` names populate each plan's
    ``excluded_names`` so rsync ``--exclude=<name>`` blocks them.
    """
    # Late import so `bulk_transfer` can be imported standalone in
    # tests without pulling the pyedflib chain.
    from clean_eeg.transfer import (
        _failed_names_from_manifest,
        preflight_deidentified_output,
    )

    ready: list[SubjectPlan] = []
    hard_failures: list[tuple[Path, str]] = []
    for subject_dir in subject_dirs:
        subject_dir = Path(subject_dir)
        result = preflight_deidentified_output(subject_dir)
        if not result.passed:
            hard_failures.append(
                (subject_dir, "; ".join(result.failures)))
            continue
        assert result.manifest is not None
        excluded = _failed_names_from_manifest(result.manifest)
        transferable = _sum_transferable_bytes(subject_dir, excluded)
        ready.append(SubjectPlan(
            subject_dir=subject_dir,
            subject_code=result.manifest["subject_code"],
            site_incoming_folder=result.manifest["site_incoming_folder"],
            transferable_bytes=transferable,
            excluded_names=excluded,
        ))
    return ready, hard_failures


# ---------------------------------------------------------------------------
# Worker: transfer one subject (with retry)
# ---------------------------------------------------------------------------

def _inject_rsync_flags(upload_argv: list[str], bwlimit_kbps: int | None,
                        rsync_timeout_s: int) -> list[str]:
    """Return a copy of upload_argv with ``--bwlimit`` and ``--timeout``
    injected right after the ``rsync`` binary. Both are safe to add to
    any rsync invocation (they compose cleanly with --partial /
    --exclude=... that build_transfer_plan already emits).
    """
    argv = list(upload_argv)
    if not argv or argv[0] != "rsync":
        return argv
    if bwlimit_kbps is not None:
        argv.insert(1, f"--bwlimit={bwlimit_kbps}")
    argv.insert(1, f"--timeout={rsync_timeout_s}")
    return argv


def _run_short(argv: list[str]) -> tuple[int, str]:
    """Run a quick SSH command (mkdir / chgrp / chmod). Returns
    ``(exit_code, tail_of_stderr_or_stdout)``. These commands should
    complete in well under a second on a healthy network; we don't
    need bwlimit monitoring around them."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except (subprocess.SubprocessError, OSError) as e:
        return 255, f"{type(e).__name__}: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return proc.returncode, " | ".join(tail)
    return 0, ""


def _run_rsync_with_bwlimit_monitor(argv: list[str],
                                    bwlimit_policy: "BwlimitPolicy",
                                    initial_bwlimit: int | None,
                                    poll_interval_s: float = 30.0,
                                    ) -> tuple[int, str, bool]:
    """Popen rsync and poll the bwlimit policy every ``poll_interval_s``.
    If the effective cap changes from ``initial_bwlimit``, SIGTERM the
    rsync process and return ``boundary_crossed=True`` so the caller
    can restart with the new cap (rsync ``--partial`` makes restart
    cheap -- resumes from the last checkpoint on disk).

    Returns ``(exit_code, error_tail, boundary_crossed)``. When
    boundary_crossed, ``exit_code`` reflects the SIGTERM'd process
    and should NOT count as a retry-consuming failure.
    """
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
    except (subprocess.SubprocessError, OSError) as e:
        return 255, f"popen: {type(e).__name__}: {e}", False

    while True:
        try:
            stdout, stderr = proc.communicate(timeout=poll_interval_s)
            if proc.returncode == 0:
                return 0, "", False
            tail = (stderr or stdout or "").strip().splitlines()[-5:]
            return proc.returncode, " | ".join(tail), False
        except subprocess.TimeoutExpired:
            if bwlimit_policy.current_kbps() != initial_bwlimit:
                proc.terminate()
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                tail = (stderr or stdout or "").strip().splitlines()[-3:]
                return (proc.returncode, "bwlimit_boundary_crossed: "
                        + " | ".join(tail), True)


def _run_subject_rsync(plan: SubjectPlan,
                       ssh_user: str,
                       bwlimit_policy: "BwlimitPolicy",
                       rsync_timeout_s: int,
                       remote_dir_override: str | None,
                       bwlimit_poll_s: float = 30.0,
                       ) -> tuple[int, str, bool]:
    """Compose the mkdir + rsync + perms sequence for ONE subject and
    execute them. Returns ``(exit_code, error_message, boundary_crossed)``.

    ``boundary_crossed=True`` means the rsync was interrupted because
    the day/night bwlimit boundary was crossed -- the caller should
    restart WITHOUT counting this as a retry-consuming failure. Only
    the rsync step is monitored for the boundary; mkdir/perms are quick
    SSH one-liners.
    """
    from clean_eeg.transfer import build_transfer_plan

    tplan = build_transfer_plan(
        plan.subject_dir,
        subject_code=plan.subject_code,
        site_incoming_folder=plan.site_incoming_folder,
        ssh_user=ssh_user,
        remote_dir_override=remote_dir_override,
        excluded_names=plan.excluded_names,
    )

    initial_bwlimit = bwlimit_policy.current_kbps()
    upload_argv = _inject_rsync_flags(
        tplan.upload_argv, initial_bwlimit, rsync_timeout_s)

    if tplan.mkdir_argv:
        code, err = _run_short(tplan.mkdir_argv)
        if code != 0:
            return code, f"mkdir: {err}", False

    code, err, boundary = _run_rsync_with_bwlimit_monitor(
        upload_argv, bwlimit_policy, initial_bwlimit,
        poll_interval_s=bwlimit_poll_s)
    if code != 0 or boundary:
        return code, f"rsync: {err}", boundary

    if tplan.perms_argv:
        code, err = _run_short(tplan.perms_argv)
        if code != 0:
            return code, f"perms: {err}", False

    return 0, "", False


def transfer_one_subject_with_retry(plan: SubjectPlan, *,
                                    ssh_user: str,
                                    bwlimit_policy: BwlimitPolicy,
                                    max_retries: int,
                                    rsync_timeout_s: int,
                                    backoff_base_s: int,
                                    remote_dir_override: str | None = None,
                                    log: EventLog | None = None,
                                    bwlimit_poll_s: float = 30.0,
                                    ) -> SubjectResult:
    """Transfer one subject with exponential-backoff retry. Returns
    a ``SubjectResult`` regardless of outcome (never raises).

    Bandwidth cap is monitored via ``bwlimit_policy``; when the
    day/night boundary crosses mid-rsync, the running rsync is
    terminated and restarted with the new cap. Boundary-triggered
    restarts do NOT consume a retry (rsync's ``--partial`` makes the
    restart cheap and it's not a "failure" in any meaningful sense).
    """
    start = time.perf_counter()
    last_exit: int | None = None
    last_err: str | None = None
    attempt = 0
    while attempt < max_retries:
        bwlimit = bwlimit_policy.current_kbps()
        if log:
            log.emit("subject_start", subject=plan.subject_code,
                     attempt=attempt + 1, bwlimit_kbps=bwlimit,
                     bytes_expected=plan.transferable_bytes)
        attempt_start = time.perf_counter()
        exit_code, err, boundary = _run_subject_rsync(
            plan, ssh_user, bwlimit_policy, rsync_timeout_s,
            remote_dir_override, bwlimit_poll_s=bwlimit_poll_s)
        attempt_elapsed = time.perf_counter() - attempt_start
        if log:
            log.emit("rsync_exit", subject=plan.subject_code,
                     attempt=attempt + 1, exit_code=exit_code,
                     boundary_crossed=boundary,
                     elapsed_s=round(attempt_elapsed, 2),
                     error_tail=err[:500] if err else None)
        if exit_code == 0:
            return SubjectResult(
                subject_code=plan.subject_code,
                subject_dir=str(plan.subject_dir),
                attempts=attempt + 1,
                succeeded=True,
                elapsed_s=time.perf_counter() - start,
            )
        if boundary:
            # Don't count as a retry, no backoff -- immediately re-enter
            # with the new bwlimit picked up at the top of the loop.
            if log:
                log.emit("bwlimit_boundary_restart",
                         subject=plan.subject_code,
                         old_bwlimit_kbps=bwlimit,
                         new_bwlimit_kbps=bwlimit_policy.current_kbps())
            continue
        last_exit, last_err = exit_code, err
        attempt += 1
        if attempt < max_retries:
            # Exponential backoff: 30, 60, 120, 240, ...
            wait_s = backoff_base_s * (2 ** (attempt - 1))
            if log:
                log.emit("backoff_wait", subject=plan.subject_code,
                         attempt=attempt, wait_s=wait_s)
            time.sleep(wait_s)
    return SubjectResult(
        subject_code=plan.subject_code,
        subject_dir=str(plan.subject_dir),
        attempts=max_retries,
        succeeded=False,
        elapsed_s=time.perf_counter() - start,
        last_exit_code=last_exit,
        last_error=last_err,
    )


# ---------------------------------------------------------------------------
# Batch runner (parallelism + day/night bwlimit boundary detection)
# ---------------------------------------------------------------------------

def _default_log_path(subjects_file: Path | None) -> Path:
    """Log lives alongside --subjects-file. Fallback to cwd if the
    caller drove the API without a subjects_file path."""
    if subjects_file:
        return subjects_file.parent / "bulk_transfer.jsonl"
    return Path.cwd() / "bulk_transfer.jsonl"


def run_bulk_transfer(subject_dirs: list[Path],
                      *,
                      ssh_user: str,
                      bwlimit_policy: BwlimitPolicy,
                      parallel: int = DEFAULT_PARALLEL,
                      max_retries: int = DEFAULT_MAX_RETRIES,
                      rsync_timeout_s: int = DEFAULT_RSYNC_TIMEOUT_S,
                      backoff_base_s: int = DEFAULT_BACKOFF_BASE_S,
                      progress_interval_s: int = DEFAULT_PROGRESS_INTERVAL_S,
                      remote_dir_override: str | None = None,
                      log_path: Path | None = None,
                      subjects_file: Path | None = None,
                      ) -> tuple[list[SubjectResult], list[tuple[Path, str]]]:
    """Drive the whole batch. Returns
    ``(subject_results, preflight_hard_failures)``.

    Prints periodic progress (subjects done / bytes done / ETA) and
    writes structured JSONL log entries alongside.
    """
    log_path = log_path or _default_log_path(subjects_file)
    ready, hard_failures = build_subject_plans(subject_dirs)

    with EventLog(log_path) as log:
        log.emit("batch_start",
                 n_subjects=len(subject_dirs),
                 n_ready=len(ready),
                 n_preflight_failed=len(hard_failures),
                 parallel=parallel, max_retries=max_retries,
                 total_bytes=sum(p.transferable_bytes for p in ready))
        for path, reason in hard_failures:
            log.emit("preflight_failed", subject_dir=str(path), reason=reason)

        # Dispatch via ThreadPoolExecutor. Rsync is a subprocess, so
        # the GIL is released for the entire duration of each worker's
        # rsync -- threads give real N-way parallelism for this
        # workload. Threads (vs ProcessPool) also let a hard rsync
        # error surface as a real Python exception instead of a
        # pickled cross-process traceback.
        results: list[SubjectResult] = []
        completed_bytes = 0
        batch_start = time.perf_counter()

        def _worker(plan: SubjectPlan) -> SubjectResult:
            return transfer_one_subject_with_retry(
                plan, ssh_user=ssh_user, bwlimit_policy=bwlimit_policy,
                max_retries=max_retries, rsync_timeout_s=rsync_timeout_s,
                backoff_base_s=backoff_base_s,
                remote_dir_override=remote_dir_override,
                log=log,
            )

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_worker, p): p for p in ready}
            last_progress = batch_start
            for fut in as_completed(futures):
                plan = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    # Worker crashed hard. Fabricate a SubjectResult so
                    # the batch summary still reflects the failure.
                    result = SubjectResult(
                        subject_code=plan.subject_code,
                        subject_dir=str(plan.subject_dir),
                        attempts=0, succeeded=False, elapsed_s=0,
                        last_exit_code=None,
                        last_error=f"worker crash: {type(e).__name__}: {e}",
                    )
                results.append(result)
                if result.succeeded:
                    completed_bytes += plan.transferable_bytes
                log.emit("subject_complete",
                         subject=result.subject_code,
                         succeeded=result.succeeded,
                         attempts=result.attempts,
                         elapsed_s=round(result.elapsed_s, 2),
                         last_exit_code=result.last_exit_code,
                         last_error=(result.last_error or "")[:500] or None)

                now = time.perf_counter()
                if now - last_progress >= progress_interval_s:
                    _print_progress(results, completed_bytes,
                                    total_bytes=sum(p.transferable_bytes for p in ready),
                                    n_total=len(ready), batch_start=batch_start)
                    last_progress = now

        elapsed = time.perf_counter() - batch_start
        n_success = sum(1 for r in results if r.succeeded)
        log.emit("batch_complete",
                 n_succeeded=n_success,
                 n_failed=len(results) - n_success,
                 n_preflight_failed=len(hard_failures),
                 elapsed_s=round(elapsed, 2))
    _print_summary(results, hard_failures, elapsed)
    return results, hard_failures


def _print_progress(results, completed_bytes, *, total_bytes, n_total,
                    batch_start) -> None:
    elapsed = time.perf_counter() - batch_start
    rate_mbps = (completed_bytes / (1024 ** 2)) / elapsed if elapsed > 0 else 0
    remaining = total_bytes - completed_bytes
    eta_s = remaining / (completed_bytes / elapsed) if completed_bytes > 0 else float("inf")
    eta_hms = _format_hms(eta_s)
    n_done = len(results)
    print(
        f"[{n_done}/{n_total} subjects, "
        f"{completed_bytes / (1024**3):.1f} / {total_bytes / (1024**3):.1f} GB, "
        f"ETA {eta_hms} at {rate_mbps:.1f} MBps observed]",
        flush=True,
    )


def _print_summary(results, hard_failures, elapsed) -> None:
    print(f"\n=== BULK TRANSFER SUMMARY ({_format_hms(elapsed)} elapsed) ===")
    n_success = sum(1 for r in results if r.succeeded)
    n_failed = len(results) - n_success
    print(f"  succeeded:        {n_success}")
    print(f"  failed:           {n_failed}")
    print(f"  preflight_failed: {len(hard_failures)}")
    for r in results:
        if not r.succeeded:
            print(f"  FAIL {r.subject_code}: {r.attempts} attempt(s), "
                  f"last_exit={r.last_exit_code}, err={r.last_error}")
    for path, reason in hard_failures:
        print(f"  PREFLIGHT_FAIL {path}: {reason}")


def _format_hms(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "∞"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _relaunch_in_background(argv: list[str], subjects_file: Path,
                             log_path: Path) -> tuple[int, Path, Path]:
    """Re-exec this CLI without ``--background`` under ``nohup`` so the
    batch survives an SSH disconnect / operator logout. Writes the
    parent-invocation shell script next to ``subjects_file`` so an
    operator can inspect (or re-run) exactly what was launched.

    Returns ``(pid, script_path, log_path)``. The parent process exits
    immediately after launch so a shell prompt returns.
    """
    script_path = subjects_file.parent / "bulk_transfer.launch.sh"
    child_argv = [a for a in argv if a != "--background"]
    lines = [
        "#!/usr/bin/env bash",
        "set -e",
        "",
        "# Auto-generated by bulk-transfer-eeg --background.",
        "# Re-run this file directly (no nohup) to resume in the foreground.",
        "",
        " ".join(shlex.quote(a) for a in child_argv),
        "",
    ]
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)

    log_fh = open(log_path.with_suffix(".stdout.log"), "a")
    proc = subprocess.Popen(
        ["nohup", "bash", str(script_path)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid, script_path, log_path.with_suffix(".stdout.log")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bulk-transfer-eeg",
        description="Transfer many cleaned subject folders to the CML server "
                    "with retries, parallelism, day/night bandwidth caps, "
                    "and PHI-safety enforcement.",
    )
    p.add_argument("--subjects-file", type=Path, required=True,
                   help="Text file with one absolute subject-dir path per line. "
                        "Blank lines and '#' comments are ignored.")
    p.add_argument("--user", type=str, default=None,
                   help="SSH user (default: $USER).")
    p.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                   help=f"Concurrent rsync workers (default: {DEFAULT_PARALLEL}).")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help=f"Per-subject retry count (default: {DEFAULT_MAX_RETRIES}).")
    p.add_argument("--rsync-timeout", type=int, default=DEFAULT_RSYNC_TIMEOUT_S,
                   help=f"rsync I/O timeout seconds (default: {DEFAULT_RSYNC_TIMEOUT_S}).")
    p.add_argument("--backoff-base", type=int, default=DEFAULT_BACKOFF_BASE_S,
                   help=f"Base seconds for exponential backoff between retries "
                        f"(default: {DEFAULT_BACKOFF_BASE_S}). Backoff doubles each retry.")
    p.add_argument("--bwlimit-day", type=int, default=None,
                   help="Daytime rsync --bwlimit KBps. Omit for no cap.")
    p.add_argument("--bwlimit-night", type=int, default=None,
                   help="Nighttime rsync --bwlimit KBps. Omit for no cap.")
    p.add_argument("--day-start", type=str, default=DEFAULT_DAY_START,
                   help=f"HH:MM start of the daytime window (default: {DEFAULT_DAY_START}).")
    p.add_argument("--day-end", type=str, default=DEFAULT_DAY_END,
                   help=f"HH:MM end of the daytime window (default: {DEFAULT_DAY_END}).")
    p.add_argument("--log-path", type=Path, default=None,
                   help="JSONL event log path (default: alongside --subjects-file, "
                        "named bulk_transfer.jsonl).")
    p.add_argument("--remote-dir-override", type=str, default=None,
                   help="Test/scratch mode: full remote destination path. "
                        "Overrides the site-map-driven derivation.")
    p.add_argument("--background", action="store_true",
                   help="Detach from the controlling terminal and run under "
                        "nohup so the batch survives SSH disconnect / logout. "
                        "Stdout+stderr stream to a .stdout.log alongside the "
                        "JSONL event log; tail -f it to watch progress.")
    return p


def _load_subject_paths(subjects_file: Path) -> list[Path]:
    paths = []
    for raw in subjects_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    return paths


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    args = _build_parser().parse_args(argv)
    subject_dirs = _load_subject_paths(args.subjects_file)
    if not subject_dirs:
        print(f"No subject paths found in {args.subjects_file}", file=sys.stderr)
        return 1

    log_path = args.log_path or _default_log_path(args.subjects_file)
    if args.background:
        # Re-exec self under nohup, minus --background, and exit.
        pid, script_path, stdout_log = _relaunch_in_background(
            [sys.executable, "-m", "clean_eeg.bulk_transfer", *argv],
            args.subjects_file, log_path)
        print(f"[background] launched pid={pid}")
        print(f"[background] script: {script_path}")
        print(f"[background] stdout log: tail -f {stdout_log}")
        print(f"[background] event log:  tail -f {log_path}")
        return 0

    bwlimit_policy = BwlimitPolicy(
        day_kbps=args.bwlimit_day, night_kbps=args.bwlimit_night,
        day_start=_parse_hhmm(args.day_start),
        day_end=_parse_hhmm(args.day_end),
    )

    results, hard_failures = run_bulk_transfer(
        subject_dirs,
        ssh_user=args.user or getpass.getuser(),
        bwlimit_policy=bwlimit_policy,
        parallel=args.parallel, max_retries=args.max_retries,
        rsync_timeout_s=args.rsync_timeout,
        backoff_base_s=args.backoff_base,
        remote_dir_override=args.remote_dir_override,
        log_path=args.log_path,
        subjects_file=args.subjects_file,
    )
    # Exit nonzero on any failure so a wrapping script (cron, batch
    # scheduler) can detect a batch that needs operator attention.
    all_ok = (all(r.succeeded for r in results) and not hard_failures)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
