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
import collections
import csv
import getpass
import json
import shlex
import subprocess
import sys
import threading
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


PREFLIGHT_MAX_WORKERS = 8


# Substrings that mark a failure as deterministic (retrying will not
# help). Case-insensitive match on the rsync/ssh stderr tail.
_FATAL_CONFIG_ERROR_SUBSTRINGS = (
    "is not recognized as an internal or external command",  # Windows cmd
    "command not found",                                     # POSIX sh
    "not found",                                             # PowerShell
    "rsync: connection unexpectedly closed",  # remote spawn failed
    "permission denied (publickey",                          # SSH auth
    "no such file or directory",                             # bad path
    "host key verification failed",                          # unknown host
    "unknown option",                                        # rsync flag typo / version mismatch
    "syntax or usage error",                                 # rsync bad argv
)


def _is_fatal_config_error(err: str | None) -> bool:
    if not err:
        return False
    low = err.lower()
    return any(s.lower() in low for s in _FATAL_CONFIG_ERROR_SUBSTRINGS)


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

    Preflight is parallelised across up to PREFLIGHT_MAX_WORKERS
    threads. On network storage (Oceanus / NFS) each subject's preflight
    is I/O-bound -- pyedflib header opens dominate wall time -- so
    threads (not processes) yield a near-linear speedup with no GIL
    contention. Serial-order results are preserved so batch output and
    the summary read in the order the operator supplied.
    """
    # Late import so `bulk_transfer` can be imported standalone in
    # tests without pulling the pyedflib chain.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from clean_eeg.transfer import (
        _failed_names_from_manifest,
        preflight_deidentified_output,
    )

    subject_list = [Path(d) for d in subject_dirs]

    def _one(subject_dir: Path):
        t0 = time.perf_counter()
        result = preflight_deidentified_output(subject_dir)
        excluded: set[str] = set()
        transferable = 0
        if result.passed and result.manifest is not None:
            excluded = _failed_names_from_manifest(result.manifest)
            transferable = _sum_transferable_bytes(subject_dir, excluded)
        return (subject_dir, result, excluded, transferable,
                time.perf_counter() - t0)

    ready_by_idx: dict[int, SubjectPlan] = {}
    fail_by_idx: dict[int, tuple[Path, str]] = {}
    if not subject_list:
        return [], []

    n = len(subject_list)
    n_workers = min(PREFLIGHT_MAX_WORKERS, n)
    # Live per-subject notifications so the operator sees preflight
    # progress instead of staring at a blank terminal for the multi-
    # minute stretch it takes to open pyedflib headers over network
    # storage on the 1-2 subjects that survive the review-complete
    # fast-fail. Prints as each subject completes (unordered).
    print(f"[preflight] checking {n} subject(s) (parallel={n_workers}, "
          f"fast-fail on unreviewed)...", flush=True)
    completed = 0
    with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="preflight") as pool:
        futures = {pool.submit(_one, sd): i
                   for i, sd in enumerate(subject_list)}
        for fut in as_completed(futures):
            idx = futures[fut]
            subject_dir, result, excluded, transferable, elapsed = fut.result()
            completed += 1
            if result.passed:
                assert result.manifest is not None
                code = result.manifest["subject_code"]
                ready_by_idx[idx] = SubjectPlan(
                    subject_dir=subject_dir,
                    subject_code=code,
                    site_incoming_folder=result.manifest["site_incoming_folder"],
                    transferable_bytes=transferable,
                    excluded_names=excluded,
                )
                print(f"[preflight {completed}/{n}] OK    "
                      f"{code}  ({elapsed:.1f}s, "
                      f"{transferable / 1e9:.2f} GB to ship)", flush=True)
            else:
                reason = "; ".join(result.failures)
                fail_by_idx[idx] = (subject_dir, reason)
                # Truncate the reason for the live line -- full reason
                # lands in the event log + end-of-batch summary.
                short = (reason[:120] + "…") if len(reason) > 120 else reason
                print(f"[preflight {completed}/{n}] SKIP  "
                      f"{subject_dir.name}  ({elapsed:.1f}s): {short}",
                      flush=True)

    # Sort back into input order so downstream code (event log, summary)
    # reads in the operator-supplied sequence.
    ready = [ready_by_idx[i] for i in sorted(ready_by_idx)]
    hard_failures = [fail_by_idx[i] for i in sorted(fail_by_idx)]
    total_gb = sum(p.transferable_bytes for p in ready) / 1e9
    print(f"[preflight] done: {len(ready)} ready to ship "
          f"({total_gb:.2f} GB total), {len(hard_failures)} skipped.",
          flush=True)
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


class _PrefixedStreamReader(threading.Thread):
    """Reader thread that pulls chunks from a subprocess pipe and
    echoes them to ``sys.stdout`` with a per-subject prefix, splitting
    on BOTH '\\n' and '\\r'.

    Why both: rsync ``--progress`` emits '\\r'-only mid-file updates
    (percent complete, MB/s) that a plain readline() will buffer until
    the file completes. On a 100+GB file that means silence for the
    hour+ the transfer runs. Splitting on '\\r' unblocks those live
    updates so the operator sees percent progress in near-real-time.
    """
    def __init__(self, fh, prefix: str, keep_tail: int = 20):
        super().__init__(daemon=True, name=f"stream-{prefix}")
        self.fh = fh
        self.prefix = prefix
        self.tail = collections.deque(maxlen=keep_tail)

    def run(self) -> None:
        buf = ""
        try:
            while True:
                # Read whatever the OS has for us. 1 char at a time is
                # slow but robust across line-separator conventions;
                # rsync's throughput here is trivial vs the network
                # transfer this is monitoring.
                ch = self.fh.read(1)
                if ch == "":
                    break  # EOF (process closed the pipe)
                if ch in ("\n", "\r"):
                    if buf:
                        sys.stdout.write(f"[{self.prefix}] {buf}\n")
                        sys.stdout.flush()
                        self.tail.append(buf)
                        buf = ""
                else:
                    buf += ch
            # Flush any trailing content without a terminator.
            if buf:
                sys.stdout.write(f"[{self.prefix}] {buf}\n")
                sys.stdout.flush()
                self.tail.append(buf)
        except (OSError, ValueError):
            # ValueError: I/O op on closed file (parent .close()'d it)
            pass


def _run_rsync_with_bwlimit_monitor(argv: list[str],
                                    bwlimit_policy: "BwlimitPolicy",
                                    initial_bwlimit: int | None,
                                    poll_interval_s: float = 30.0,
                                    stream_prefix: str | None = None,
                                    ) -> tuple[int, str, bool]:
    """Popen rsync and poll the bwlimit policy every ``poll_interval_s``.
    If the effective cap changes from ``initial_bwlimit``, SIGTERM the
    rsync process and return ``boundary_crossed=True`` so the caller
    can restart with the new cap (rsync ``--partial`` makes restart
    cheap -- resumes from the last checkpoint on disk).

    Returns ``(exit_code, error_tail, boundary_crossed)``. When
    boundary_crossed, ``exit_code`` reflects the SIGTERM'd process
    and should NOT count as a retry-consuming failure.

    ``stream_prefix``: when set, spawns reader threads that echo rsync's
    stdout+stderr line-buffered to ``sys.stdout`` prefixed with
    ``[stream_prefix]``. Lets the operator see per-file completions live
    on a multi-hour transfer instead of watching a blank terminal. When
    None, the prior behaviour is preserved (capture-then-tail on
    failure) -- callers that need clean stdout (tests) can opt out.
    """
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                bufsize=1)
    except (subprocess.SubprocessError, OSError) as e:
        return 255, f"popen: {type(e).__name__}: {e}", False

    # Live-stream mode: reader threads consume stdout/stderr as they
    # arrive AND keep the last N lines around for the failure tail.
    # Only engage when the process has real pipes (test fakes may not
    # supply them; fall back to capture mode in that case).
    stdout_reader: _PrefixedStreamReader | None = None
    stderr_reader: _PrefixedStreamReader | None = None
    if (stream_prefix is not None
            and getattr(proc, "stdout", None) is not None
            and getattr(proc, "stderr", None) is not None):
        stdout_reader = _PrefixedStreamReader(proc.stdout, stream_prefix)
        stderr_reader = _PrefixedStreamReader(proc.stderr,
                                                stream_prefix + "!")
        stdout_reader.start()
        stderr_reader.start()

    def _final_tail() -> str:
        if stdout_reader is None or stderr_reader is None:
            return ""
        lines = list(stderr_reader.tail) or list(stdout_reader.tail)
        return " | ".join(lines[-5:])

    stream_mode = stdout_reader is not None
    while True:
        try:
            if not stream_mode:
                stdout, stderr = proc.communicate(
                    timeout=poll_interval_s)
                if proc.returncode == 0:
                    return 0, "", False
                tail = (stderr or stdout or "").strip().splitlines()[-5:]
                return proc.returncode, " | ".join(tail), False
            # Stream mode: block on proc.wait with the same cadence.
            rc = proc.wait(timeout=poll_interval_s)
            assert stdout_reader is not None and stderr_reader is not None
            stdout_reader.join(timeout=1)
            stderr_reader.join(timeout=1)
            if rc == 0:
                return 0, "", False
            return rc, _final_tail() or "(no output captured)", False
        except subprocess.TimeoutExpired:
            if bwlimit_policy.current_kbps() != initial_bwlimit:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                if stream_mode:
                    assert (stdout_reader is not None
                            and stderr_reader is not None)
                    stdout_reader.join(timeout=1)
                    stderr_reader.join(timeout=1)
                    tail = _final_tail()
                else:
                    stdout, stderr = proc.communicate()
                    tail = " | ".join(
                        (stderr or stdout or "").strip().splitlines()[-3:])
                return (proc.returncode, "bwlimit_boundary_crossed: "
                        + tail, True)


def _run_subject_rsync(plan: SubjectPlan,
                       ssh_user: str | None,
                       bwlimit_policy: "BwlimitPolicy",
                       rsync_timeout_s: int,
                       remote_dir_override: str | None,
                       bwlimit_poll_s: float = 30.0,
                       ssh_host: str | None = None,
                       remote_base: str | None = None,
                       rsync_path: str | None = None,
                       skip_remote_mkdir: bool = False,
                       remote_mkdir_cmd: str | None = None,
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

    # In test/scratch mode (remote_dir_override supplied without an
    # explicit ssh_host), stand in a placeholder host -- the argv still
    # gets composed but the subprocess is monkeypatched in tests.
    effective_ssh_host = ssh_host or "test-scratch-host"
    tplan = build_transfer_plan(
        plan.subject_dir,
        subject_code=plan.subject_code,
        site_incoming_folder=plan.site_incoming_folder,
        ssh_user=ssh_user,
        remote_dir_override=remote_dir_override,
        excluded_names=plan.excluded_names,
        ssh_host=effective_ssh_host,
        remote_base=remote_base,
        rsync_path=rsync_path,
        skip_remote_mkdir=skip_remote_mkdir,
        remote_mkdir_cmd=remote_mkdir_cmd,
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
        poll_interval_s=bwlimit_poll_s,
        stream_prefix=plan.subject_code)
    if code != 0 or boundary:
        return code, f"rsync: {err}", boundary

    if tplan.perms_argv:
        code, err = _run_short(tplan.perms_argv)
        if code != 0:
            return code, f"perms: {err}", False

    return 0, "", False


def transfer_one_subject_with_retry(plan: SubjectPlan, *,
                                    ssh_user: str | None,
                                    bwlimit_policy: BwlimitPolicy,
                                    max_retries: int,
                                    rsync_timeout_s: int,
                                    backoff_base_s: int,
                                    remote_dir_override: str | None = None,
                                    log: EventLog | None = None,
                                    bwlimit_poll_s: float = 30.0,
                                    ssh_host: str | None = None,
                                    remote_base: str | None = None,
                                    rsync_path: str | None = None,
                                    skip_remote_mkdir: bool = False,
                                    remote_mkdir_cmd: str | None = None,
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
        # Per-subject remote_dir_override: if the caller supplied a
        # template like "/mnt/backup/clean_eeg/{subject_code}", expand
        # it here so each subject lands in its own destination dir.
        # Non-template strings pass through unchanged (backwards-
        # compatible).
        per_subject_override = (
            remote_dir_override.format(subject_code=plan.subject_code)
            if remote_dir_override is not None
            else None)
        exit_code, err, boundary = _run_subject_rsync(
            plan, ssh_user, bwlimit_policy, rsync_timeout_s,
            per_subject_override, bwlimit_poll_s=bwlimit_poll_s,
            ssh_host=ssh_host, remote_base=remote_base,
            rsync_path=rsync_path,
            skip_remote_mkdir=skip_remote_mkdir,
            remote_mkdir_cmd=remote_mkdir_cmd)
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
        # Fast-abort on obviously-deterministic errors -- retrying a
        # command that failed because it doesn't exist (Windows cmd
        # rejecting POSIX 'umask', missing 'rsync' on remote PATH, etc)
        # wastes the retry budget and multiplies wall time. Bail after
        # the first attempt with a clear failure.
        if _is_fatal_config_error(err):
            if log:
                log.emit("fatal_abort", subject=plan.subject_code,
                         reason=err[:500])
            return SubjectResult(
                subject_code=plan.subject_code,
                subject_dir=str(plan.subject_dir),
                attempts=attempt + 1,
                succeeded=False,
                elapsed_s=time.perf_counter() - start,
                last_exit_code=exit_code,
                last_error=(err + " [aborting retries: deterministic "
                            "config error, not a transient network fault]"),
            )
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


def _filter_plans_by_subject(plans: list[SubjectPlan],
                              only_subjects: list[str] | None,
                              ) -> list[SubjectPlan]:
    """Return the subset of ``plans`` whose ``subject_code`` matches
    any entry in ``only_subjects``. None/empty picker -> return
    unchanged. Warns for picker entries that matched zero plans."""
    if not only_subjects:
        return plans
    picker = set(only_subjects)
    kept = [p for p in plans if p.subject_code in picker]
    unmatched = picker - {p.subject_code for p in kept}
    if unmatched:
        print(f"[warn] --only-subjects entries not found: "
              f"{sorted(unmatched)}", file=sys.stderr)
    return kept


def run_bulk_transfer(subject_dirs: list[Path],
                      *,
                      ssh_user: str | None,
                      bwlimit_policy: BwlimitPolicy,
                      parallel: int = DEFAULT_PARALLEL,
                      max_retries: int = DEFAULT_MAX_RETRIES,
                      rsync_timeout_s: int = DEFAULT_RSYNC_TIMEOUT_S,
                      backoff_base_s: int = DEFAULT_BACKOFF_BASE_S,
                      progress_interval_s: int = DEFAULT_PROGRESS_INTERVAL_S,
                      remote_dir_override: str | None = None,
                      log_path: Path | None = None,
                      subjects_file: Path | None = None,
                      only_subjects: list[str] | None = None,
                      ssh_key: Path | None = None,
                      auto_ssh_agent: bool = True,
                      ssh_host: str | None = None,
                      remote_base: str | None = None,
                      rsync_path: str | None = None,
                      skip_remote_mkdir: bool = False,
                      remote_mkdir_cmd: str | None = None,
                      ) -> tuple[list[SubjectResult], list[tuple[Path, str]]]:
    """Drive the whole batch. Returns
    ``(subject_results, preflight_hard_failures)``.

    ``only_subjects``: if non-empty, only transfer plans whose
    subject_code matches. Preflight still runs against every path so
    invalid dirs stay visible in ``preflight_hard_failures``; the
    picker only filters the eligible-for-transfer queue.

    ``ssh_key`` + ``auto_ssh_agent``: forwarded to ensure_ssh_agent so
    the passphrase gets entered ONCE before the batch instead of
    dozens of times per subject. See clean_eeg.transfer.ensure_ssh_agent
    for the auto-setup semantics.

    Prints periodic progress (subjects done / bytes done / ETA) and
    writes structured JSONL log entries alongside.
    """
    # ssh_host is required for a real transfer. remote_dir_override
    # is the escape hatch for test/scratch destinations that skip the
    # SSH probe entirely, so it can stand in when ssh_host is None.
    if not ssh_host and remote_dir_override is None:
        print("[transfer] ABORT: --ssh-host is required (no code-level "
              "default endpoint). Pass an ssh_config alias or "
              "user-visible hostname.", file=sys.stderr, flush=True)
        return [], []

    # Set up ssh-agent ONCE before the batch starts (passphrase entered
    # once, not per-subject). Every subprocess spawned below (mkdir,
    # rsync, perms per subject) inherits SSH_AUTH_SOCK from this
    # process, so they all reuse the same agent.
    from clean_eeg.transfer import ensure_ssh_agent
    ensure_ssh_agent(key_path=ssh_key, auto=auto_ssh_agent)

    # Reachability preflight. Rsync's own timeout is per-I/O and fires
    # ONLY after a connection is up; if the SSH handshake itself hangs
    # (VPS down, tunnel not established, wrong hostname) the operator
    # otherwise sits on a silent terminal indefinitely. A 10-second
    # ConnectTimeout gives a fast, clear error before we fan out to N
    # parallel workers all hanging on the same broken host.
    #
    # Skipped when remote_dir_override is set: that flag signals a
    # test/scratch destination (local filesystem, mock, etc.) where the
    # SSH probe would falsely fail.
    if remote_dir_override is None:
        # ssh_host is guaranteed non-None here: the top-of-function
        # guard rejects (None ssh_host AND None remote_dir_override).
        assert ssh_host is not None
        effective_host = ssh_host
        # Same user-prefix rule as the plan composer: only prepend
        # `user@` when the operator explicitly passed --user. Otherwise
        # let ssh_config's User directive apply.
        reach_target = (f"{ssh_user}@{effective_host}"
                        if ssh_user else effective_host)
        reach_argv = [
            "ssh", "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",  # never prompt for a password
            reach_target, ":"]
        print(f"[transfer] checking reachability of "
              f"{reach_target} (10s timeout)...", flush=True)
        try:
            reach = subprocess.run(reach_argv, capture_output=True,
                                    text=True, timeout=15)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"[transfer] ABORT: could not reach "
                  f"{reach_target} "
                  f"({type(e).__name__}: {e}). "
                  f"Fix the endpoint or pass --ssh-host <alias>.",
                  file=sys.stderr, flush=True)
            return [], []
        if reach.returncode != 0:
            err_tail = (reach.stderr or reach.stdout or "").strip()
            print(f"[transfer] ABORT: ssh {reach_target} "
                  f"exited {reach.returncode}. Fix credentials/tunnel "
                  f"and re-run.\n    stderr: {err_tail}",
                  file=sys.stderr, flush=True)
            return [], []
        print(f"[transfer] reachability OK.", flush=True)

    log_path = log_path or _default_log_path(subjects_file)
    ready, hard_failures = build_subject_plans(subject_dirs)
    ready = _filter_plans_by_subject(ready, only_subjects)

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

        if not ready:
            print("[transfer] nothing to ship after preflight; "
                  "no rsyncs will run.", flush=True)
        else:
            total_gb = sum(p.transferable_bytes for p in ready) / 1e9
            print(f"[transfer] starting {len(ready)} rsync worker(s) "
                  f"(parallel={parallel}, {total_gb:.2f} GB total). "
                  f"Per-subject progress prints on completion.",
                  flush=True)

        def _worker(plan: SubjectPlan) -> SubjectResult:
            return transfer_one_subject_with_retry(
                plan, ssh_user=ssh_user, bwlimit_policy=bwlimit_policy,
                max_retries=max_retries, rsync_timeout_s=rsync_timeout_s,
                backoff_base_s=backoff_base_s,
                remote_dir_override=remote_dir_override,
                log=log,
                ssh_host=ssh_host, remote_base=remote_base,
                rsync_path=rsync_path,
                skip_remote_mkdir=skip_remote_mkdir,
                remote_mkdir_cmd=remote_mkdir_cmd,
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

                # Live per-subject line so the operator sees progress
                # even when the periodic aggregate hasn't fired yet.
                # Bytes shown are transferable-plan bytes (rsync's own
                # --progress prints the wire numbers).
                tag = "OK  " if result.succeeded else "FAIL"
                print(f"[transfer {len(results)}/{len(ready)}] {tag} "
                      f"{result.subject_code}  "
                      f"({result.elapsed_s:.1f}s, "
                      f"{plan.transferable_bytes / 1e9:.2f} GB)",
                      flush=True)

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

    # Write the review-friendly failed-subjects CSV alongside the
    # JSONL log. Only created when there ARE failures -- an empty CSV
    # in a fully-successful batch would be a misleading artifact.
    failed_csv_path = log_path.parent / FAILED_CSV_FILENAME
    csv_written = write_failed_subjects_csv(
        results, hard_failures, failed_csv_path)
    _print_summary(results, hard_failures, elapsed,
                   failed_csv_path=failed_csv_path if csv_written else None)
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


FAILED_CSV_FILENAME = "failed_subject_transfer.csv"

FAILED_CSV_COLUMNS = ("subject_code", "subject_dir", "failure_type",
                       "attempts", "last_exit_code", "reason")


def write_failed_subjects_csv(results: list[SubjectResult],
                               hard_failures: list[tuple[Path, str]],
                               path: Path) -> bool:
    """Write a review-friendly CSV of every subject the batch did NOT
    successfully transfer. Two failure types are recorded:

      * ``transfer_failed``: preflight passed but rsync exhausted all
        retries. ``attempts`` / ``last_exit_code`` populated.
      * ``preflight_failed``: never got scheduled (bad manifest,
        unlisted PHI file, header mismatch, ...). ``attempts`` /
        ``last_exit_code`` are blank; the reason string carries the
        preflight diagnostic.

    Returns True if the CSV was written (there were any failures),
    False if the batch was fully clean (nothing written -- an empty
    CSV would be misleading).
    """
    failed_transfers = [r for r in results if not r.succeeded]
    if not failed_transfers and not hard_failures:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(FAILED_CSV_COLUMNS))
        w.writeheader()
        for r in failed_transfers:
            w.writerow({
                "subject_code": r.subject_code,
                "subject_dir": r.subject_dir,
                "failure_type": "transfer_failed",
                "attempts": r.attempts,
                "last_exit_code": (r.last_exit_code
                                    if r.last_exit_code is not None else ""),
                "reason": (r.last_error or "").strip(),
            })
        for subject_dir, reason in hard_failures:
            w.writerow({
                "subject_code": "",   # unknown -- preflight never got
                                       # to read the manifest
                "subject_dir": str(subject_dir),
                "failure_type": "preflight_failed",
                "attempts": "",
                "last_exit_code": "",
                "reason": reason.strip(),
            })
    return True


def _print_summary(results, hard_failures, elapsed,
                   failed_csv_path: Path | None = None) -> None:
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
    if failed_csv_path is not None and (n_failed or hard_failures):
        print(f"\n  Full failed-subjects CSV for review: {failed_csv_path}")


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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--subjects-file", type=Path,
                     help="Text file with one absolute subject-dir path per line. "
                          "Blank lines and '#' comments are ignored.")
    src.add_argument("--subjects-csv", type=Path,
                     help="Same CSV clean-batch-eeg consumes; the "
                          "'output_path' column is treated as the "
                          "subject-dir list. Lets a single CSV drive "
                          "both stages so operators don't maintain two "
                          "parallel lists. Preflight's review-complete "
                          "gate silently holds back subjects that "
                          "haven't been reviewed yet.")
    p.add_argument("--user", type=str, default=None,
                   help="SSH user (default: $USER).")
    p.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                   help=f"Concurrent rsync workers (default: {DEFAULT_PARALLEL}).")
    p.add_argument("--sequential", action="store_true",
                   help="Force sequential transfer (one rsync at a "
                        "time). Overrides --parallel. Useful when "
                        "parallel workers interleave output badly or "
                        "when the bottleneck is per-connection bandwidth "
                        "and multiple streams thrash instead of adding "
                        "throughput. Equivalent to --parallel 1 but "
                        "explicit so it survives config copy-paste.")
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
                   help="Full remote destination path per subject. "
                        "Overrides the site-map-driven derivation. Supports "
                        "'{subject_code}' as a substitution placeholder so a "
                        "single template lands each subject in its own dir "
                        "(e.g. /mnt/backup/clean_eeg/{subject_code}).")
    p.add_argument("--ssh-host", type=str, default=None,
                   help="Override the default rhino SSH endpoint. Accepts "
                        "any ssh_config alias -- useful for personal "
                        "endpoints with ProxyJump (e.g. an alias defined in "
                        "~/.ssh/config that hops through a VPS to a home "
                        "backup box). Combines with --remote-dir-override "
                        "for full control of the destination.")
    p.add_argument("--remote-base", type=str, default=None,
                   help="Override REMOTE_BASE (default: rhino's incoming "
                        "dir). Per-subject dirs are still derived below "
                        "this via the site-map layout. Use --remote-dir-"
                        "override instead for flat / non-hierarchical "
                        "targets.")
    p.add_argument("--rsync-path", type=str, default=None,
                   help="Passed to rsync as --rsync-path=<cmd>. Needed "
                        "when the remote's default shell can't find "
                        "'rsync' on its PATH -- e.g. a Windows sshd "
                        "whose cmd.exe shell won't invoke rsync "
                        "directly; use --rsync-path='wsl -e rsync' "
                        "to route through WSL.")
    p.add_argument("--no-remote-mkdir", action="store_true",
                   help="Skip the pre-transfer 'ssh HOST umask && "
                        "mkdir -p ...' step entirely. Operator must "
                        "pre-create the destination dir. Use "
                        "--remote-mkdir if you'd rather have the tool "
                        "create it via a non-default shell.")
    p.add_argument("--remote-mkdir", type=str, default=None,
                   metavar="CMD",
                   help="Override the default POSIX 'umask 007 && "
                        "mkdir -p' with a custom mkdir command. The "
                        "destination path is appended as the last "
                        "argument. Use for endpoints whose default "
                        "shell can't handle POSIX -- e.g. Windows sshd "
                        "with --remote-mkdir=\"wsl -e mkdir -p\" "
                        "invokes mkdir via WSL.")
    p.add_argument("--background", action="store_true",
                   help="Detach from the controlling terminal and run under "
                        "nohup so the batch survives SSH disconnect / logout. "
                        "Stdout+stderr stream to a .stdout.log alongside the "
                        "JSONL event log; tail -f it to watch progress.")
    p.add_argument("--only-subjects", nargs="+", default=None,
                   metavar="SUBJECT_CODE",
                   help="Transfer only these subject codes (matched against "
                        "each subject's manifest.subject_code). Useful for "
                        "smoke-testing one subject before releasing the whole "
                        "batch. Warns for entries that don't match any "
                        "successful preflight.")
    p.add_argument("--only-subjects-file", type=Path, default=None,
                   metavar="FILE",
                   help="File with one subject code per line (blank lines and "
                        "'#' comments ignored). Merged with --only-subjects "
                        "if both are given. Same subjects-list file can "
                        "drive clean-batch-eeg and bulk-transfer-eeg so "
                        "the two stages stay in lockstep.")
    p.add_argument("--ssh-key", type=Path, default=None,
                   help="SSH private key path for auto-loading into "
                        "ssh-agent (default: ~/.ssh/id_ed25519). Only "
                        "used when ssh-agent isn't already running with "
                        "keys; the passphrase is entered ONCE per "
                        "invocation (not per subject).")
    p.add_argument("--no-auto-ssh-agent", action="store_true",
                   help="Disable the auto-spawn-ssh-agent + auto-add-key "
                        "behaviour. Use when you're managing the agent "
                        "externally (keychain integration, custom "
                        "setup) or already have SSH_AUTH_SOCK exported "
                        "in your shell. Prints the manual-setup hint "
                        "instead if the agent is empty.")
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
    # --subjects-file and --subjects-csv are mutually exclusive at the
    # arg-parser level; exactly one is populated. --subjects-csv reuses
    # the clean-batch CSV so operators run both stages off one file.
    if args.subjects_csv is not None:
        from clean_eeg.clean_batch import parse_subjects_csv, CsvSchemaError
        try:
            rows = parse_subjects_csv(args.subjects_csv)
        except CsvSchemaError as e:
            print(f"CSV schema error: {e}", file=sys.stderr)
            return 2
        subject_dirs = [Path(r.output_path) for r in rows]
        source_path = args.subjects_csv
    else:
        subject_dirs = _load_subject_paths(args.subjects_file)
        source_path = args.subjects_file
    if not subject_dirs:
        print(f"No subject paths found in {source_path}", file=sys.stderr)
        return 1

    log_path = args.log_path or _default_log_path(source_path)
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

    only_subjects: list[str] = list(args.only_subjects or [])
    if args.only_subjects_file is not None:
        # Reuse the loader from clean_batch so both wrappers accept
        # the same file format (single point of truth for the parse
        # rules: strip whitespace, ignore blank + '#' lines, dedupe).
        from clean_eeg.clean_batch import load_subject_codes_from_file
        try:
            only_subjects.extend(
                load_subject_codes_from_file(args.only_subjects_file))
        except OSError as e:
            print(f"Cannot read --only-subjects-file "
                  f"{args.only_subjects_file}: {e}", file=sys.stderr)
            return 2

    # --sequential wins over --parallel (explicit override so a config
    # file that hardcodes --parallel N can still be dialled back to 1).
    effective_parallel = 1 if args.sequential else args.parallel
    results, hard_failures = run_bulk_transfer(
        subject_dirs,
        # None -> defer to ssh_config's `User` directive for the given
        # host alias (or SSH's own default). Prepending `$USER@` would
        # OVERRIDE the config's User line, which is exactly what broke
        # multi-user endpoints where the local username differs from
        # the remote (e.g. rxd873 on Jefferson connecting as `dasha` on
        # the Windows tunnel).
        ssh_user=args.user,
        bwlimit_policy=bwlimit_policy,
        parallel=effective_parallel, max_retries=args.max_retries,
        rsync_timeout_s=args.rsync_timeout,
        backoff_base_s=args.backoff_base,
        remote_dir_override=args.remote_dir_override,
        log_path=args.log_path,
        subjects_file=source_path,
        only_subjects=only_subjects or None,
        ssh_key=args.ssh_key,
        auto_ssh_agent=not args.no_auto_ssh_agent,
        ssh_host=args.ssh_host,
        remote_base=args.remote_base,
        rsync_path=args.rsync_path,
        skip_remote_mkdir=args.no_remote_mkdir,
        remote_mkdir_cmd=args.remote_mkdir,
    )
    if only_subjects and not results and not hard_failures:
        # Picker matched nothing (--only-subjects[-file] entries all
        # typos or missing manifests). Don't silently exit 0 -- that
        # would mask a smoke-test typo. run_bulk_transfer already
        # warned on stderr.
        print("[error] subject filter matched zero subjects; "
              "check for typos.", file=sys.stderr)
        return 1
    # Exit nonzero on any failure so a wrapping script (cron, batch
    # scheduler) can detect a batch that needs operator attention.
    all_ok = (all(r.succeeded for r in results) and not hard_failures)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
