"""Multi-subject wrapper around :mod:`clean_subject_eeg`.

CSV-driven batch cleaner. One row per subject with these columns:

    input_path, output_path, subject_code, first_name, last_name

Optional column: ``middle_name`` (blank cell -> ``--no_middle_name``).

Iterates the rows and shells out to ``clean_subject_eeg`` per subject
(process isolation so one crashing subject can't corrupt another's
state). Continue-on-failure: a bad subject doesn't stop the batch --
the summary at the end lists which subjects failed and the exit code
is nonzero if any did.

Extra columns in the CSV are ignored so operators can keep their
own bookkeeping fields alongside the required ones.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = ("input_path", "output_path", "subject_code",
                    "first_name", "last_name")
OPTIONAL_COLUMNS = ("middle_name",)


@dataclass
class SubjectRow:
    """One CSV row after normalization. ``middle_name`` is ``None`` when
    the column is absent OR the cell is blank -- either way we pass
    ``--no_middle_name`` to the per-subject CLI."""
    input_path: str
    output_path: str
    subject_code: str
    first_name: str
    last_name: str
    middle_name: str | None = None
    row_index: int = 0   # 1-based, for error messages

    def to_clean_argv(self, extra_argv: list[str] | None = None
                       ) -> list[str]:
        """Compose the argv for one clean_subject_eeg invocation from
        this row. ``extra_argv`` are appended verbatim -- operator's
        --wipe-annotations / --recursive / --approve-confirmations
        pass through unchanged.

        The single-subject CLI does NOT accept ``--output_path``; it
        derives output from ``--input_path`` (in-place default) OR
        ``--copy_path`` (rewrite mode). Map the CSV's ``output_path``
        column accordingly:
          - ``output_path == input_path`` -> in-place (only --input_path)
          - ``output_path != input_path`` -> rewrite (--input_path + --copy_path)
        """
        argv = [
            "--input_path", self.input_path,
            "--subject_code", self.subject_code,
            "--first_name", self.first_name,
            "--last_name", self.last_name,
        ]
        if self.output_path and self.output_path != self.input_path:
            # Rewrite mode: single-subject CLI writes cleaned files to
            # --copy_path instead of mutating --input_path in-place.
            argv += ["--copy_path", self.output_path]
        if self.middle_name:
            argv += ["--middle_name", self.middle_name]
        else:
            argv += ["--no_middle_name"]
        if extra_argv:
            argv += list(extra_argv)
        return argv


@dataclass
class SubjectOutcome:
    """Per-subject outcome after cleaning (and optional audit) is
    attempted. ``audit_exit_code`` is None when audit was not requested;
    otherwise 0=pass, nonzero=findings/error. Both must be non-failure
    for :attr:`succeeded` to be True."""
    row_index: int
    subject_code: str
    input_path: str
    output_path: str
    exit_code: int
    elapsed_s: float
    error_message: str | None = None   # None on success
    audit_exit_code: int | None = None
    audit_elapsed_s: float | None = None
    audit_error_message: str | None = None

    @property
    def clean_succeeded(self) -> bool:
        return self.exit_code == 0

    @property
    def succeeded(self) -> bool:
        """Both clean and (if run) audit must succeed. If audit was
        not requested, only the clean exit code matters."""
        if not self.clean_succeeded:
            return False
        if self.audit_exit_code is None:
            return True
        return self.audit_exit_code == 0


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

class CsvSchemaError(ValueError):
    """Raised when the CSV is missing a required column. Kept as a
    distinct type so the CLI can print a helpful column-list message
    instead of a generic ValueError traceback."""


def parse_subjects_csv(path: Path) -> list[SubjectRow]:
    """Read + validate ``path`` into a list of :class:`SubjectRow`.

    Fails fast on missing required columns or blank required cells --
    silently skipping a bad row would let the operator run for hours
    before noticing a subject was dropped.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise CsvSchemaError(f"{path}: CSV has no header row")
        missing = [c for c in REQUIRED_COLUMNS
                   if c not in reader.fieldnames]
        if missing:
            raise CsvSchemaError(
                f"{path}: missing required column(s) {missing}. "
                f"Required: {list(REQUIRED_COLUMNS)}. "
                f"Optional: {list(OPTIONAL_COLUMNS)}.")

        rows: list[SubjectRow] = []
        for i, raw in enumerate(reader, start=1):
            missing_cells = [c for c in REQUIRED_COLUMNS
                             if not (raw.get(c) or "").strip()]
            if missing_cells:
                raise CsvSchemaError(
                    f"{path}:row {i}: blank required cell(s) {missing_cells}")
            mn = (raw.get("middle_name") or "").strip() or None
            rows.append(SubjectRow(
                input_path=raw["input_path"].strip(),
                output_path=raw["output_path"].strip(),
                subject_code=raw["subject_code"].strip(),
                first_name=raw["first_name"].strip(),
                last_name=raw["last_name"].strip(),
                middle_name=mn,
                row_index=i,
            ))
    return rows


# ---------------------------------------------------------------------------
# Per-subject dispatch
# ---------------------------------------------------------------------------

def _default_clean_argv_prefix() -> list[str]:
    """The command that runs one subject. Uses ``python -m`` so the
    wrapper works in any env where ``clean_eeg`` is importable, without
    depending on the ``clean-subject-eeg`` console script being on PATH.

    Forwards flags so the per-subject cleaner runs completely unattended:

      --no-launch-review        Suppress the per-subject audit + TUI
                                auto-launch; batch aggregates review
                                into a single post-batch pass.
      --quiet-gap-check         Silence + auto-approve the recording-gap
                                "Continue?" prompt (large inter-file gaps
                                trigger a warning + prompt in interactive
                                mode; batch can't stop for it).
      --approve-confirmations recording-gaps in-place signal-header-mismatch
                                Auto-approve the in-place de-identification
                                warning (in-place is the only mode used
                                for batch runs), the recording-gaps
                                confirmation (belt + suspenders with
                                --quiet-gap-check), and the multi-montage
                                signal-header-mismatch prompt (common on
                                EMU stays where a mid-stay montage change
                                shows up as two signal-header signatures).
    """
    return [sys.executable, "-m", "clean_eeg.clean_subject_eeg",
            "--no-launch-review",
            "--quiet-gap-check",
            "--approve-confirmations", "recording-gaps", "in-place",
            "signal-header-mismatch"]


def _stream_subprocess_with_prefix(argv: list[str], prefix: str
                                     ) -> tuple[int, str | None]:
    """Popen ``argv`` and echo every stdout+stderr line to sys.stdout
    prefixed with ``[<prefix>]``. Keeps the last 1000 chars of output
    for the failure-message tail. Returns ``(exit_code, err_tail)``.

    Used by parallel batch mode so multiple concurrent subject clean
    subprocesses don't produce interleaved unattributed output --
    the operator can grep by [SUBJECT_CODE] to trace any single one.
    """
    tail = collections.deque(maxlen=100)
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
    except (subprocess.SubprocessError, OSError) as e:
        return 255, f"{type(e).__name__}: {e}"
    try:
        # readline() returns line-by-line; when the child closes
        # stdout on exit, readline() returns "" and we break.
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip("\n")
            sys.stdout.write(f"[{prefix}] {line}\n")
            sys.stdout.flush()
            tail.append(line)
        proc.wait()
    except Exception as e:
        proc.kill(); proc.wait()
        return 255, f"{type(e).__name__}: {e}"
    if proc.returncode != 0:
        err_tail = "\n".join(list(tail)[-20:])[-1000:]
        return proc.returncode, (
            err_tail or f"clean_subject_eeg exited {proc.returncode}")
    return proc.returncode, None


def clean_one_subject(row: SubjectRow, *, extra_argv: list[str] | None = None,
                      argv_prefix: list[str] | None = None,
                      stream_output: bool = True,
                      prefix_output: str | None = None,
                      ) -> SubjectOutcome:
    """Shell out to ``clean_subject_eeg`` for one row. Streams the
    child's stdout+stderr to the parent's terminal by default so an
    operator watching the batch sees per-file progress live.

    ``prefix_output``: when set, streams the child's output line-by-
    line to sys.stdout with each line prefixed by ``[<prefix>]``. Used
    by parallel batch mode where multiple subject clean subprocesses
    would otherwise interleave anonymously.

    Never raises; a subprocess or OS-level error is captured into the
    ``error_message`` field and the exit code is fabricated as 255.
    """
    argv = (argv_prefix or _default_clean_argv_prefix()) + row.to_clean_argv(
        extra_argv=extra_argv)
    start = time.perf_counter()
    try:
        if prefix_output is not None:
            exit_code, error_message = _stream_subprocess_with_prefix(
                argv, prefix_output)
        elif stream_output:
            proc = subprocess.run(argv)
            error_message = None if proc.returncode == 0 else (
                f"clean_subject_eeg exited {proc.returncode}")
            exit_code = proc.returncode
        else:
            proc = subprocess.run(argv, capture_output=True, text=True)
            error_message = None if proc.returncode == 0 else (
                (proc.stderr or proc.stdout or "").strip()[-1000:]
                or f"clean_subject_eeg exited {proc.returncode}")
            exit_code = proc.returncode
    except (subprocess.SubprocessError, OSError) as e:
        exit_code = 255
        error_message = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - start
    return SubjectOutcome(
        row_index=row.row_index,
        subject_code=row.subject_code,
        input_path=row.input_path,
        output_path=row.output_path,
        exit_code=exit_code,
        elapsed_s=elapsed,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Subject filter (--only-subjects, --only-subjects-file)
# ---------------------------------------------------------------------------

def load_subject_codes_from_file(path: Path) -> list[str]:
    """One subject code per line; blank lines and '#' comments ignored.
    Whitespace stripped. Preserves order and duplicates removed."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def filter_rows_by_subject(rows: list[SubjectRow],
                            only_subjects: list[str] | None,
                            ) -> list[SubjectRow]:
    """Return the subset of ``rows`` whose ``subject_code`` matches any
    of ``only_subjects``. If ``only_subjects`` is None/empty, ``rows``
    is returned unchanged. Prints a warning for any picker entry that
    matched zero rows so the operator notices typos immediately."""
    if not only_subjects:
        return rows
    picker = set(only_subjects)
    kept = [r for r in rows if r.subject_code in picker]
    unmatched = picker - {r.subject_code for r in kept}
    if unmatched:
        print(f"[warn] --only-subjects entries not found in CSV: "
              f"{sorted(unmatched)}", file=sys.stderr)
    return kept


# ---------------------------------------------------------------------------
# Post-clean audit
# ---------------------------------------------------------------------------

def _default_audit_argv_prefix() -> list[str]:
    return [sys.executable, "-m", "clean_eeg.audit.cli"]


# Flags used for both --audit-after-clean AND the post-batch
# review-phase audit. Kept in one place so the two "post-clean audit"
# invocations produce identical, header-inclusive output. The operator
# always wants:
#   --no-notebook            skip the ~10s ipynb/html render
#   --hide-annotation-flags  the annotation-review TUI (about to run,
#                            or already run) shows every annotation --
#                            listing them in the audit is redundant
#                            and noisy for cleaned+reviewed subjects
#   -v                       every check status, not just failures
#   --print-edf-header       header residue is the load-bearing thing
#                            a human wants to see after each clean
#   --print-edf-signal-header  same for the per-channel headers
POST_CLEAN_AUDIT_FLAGS = [
    "--no-notebook", "--hide-annotation-flags",
    "-v", "--print-edf-header", "--print-edf-signal-header"]


def audit_one_subject(output_path: str, *,
                       argv_prefix: list[str] | None = None,
                       stream_output: bool = True,
                       ) -> tuple[int, float, str | None]:
    """Shell out to ``audit-subject-eeg <output_path>``. Returns
    ``(exit_code, elapsed_s, error_tail)``. The audit CLI exits 0 on
    'pass' and nonzero on 'fail' or any exception -- see
    :mod:`clean_eeg.audit.cli`.
    """
    argv = (argv_prefix or _default_audit_argv_prefix()) + [output_path]
    start = time.perf_counter()
    try:
        if stream_output:
            proc = subprocess.run(argv)
            err_tail = None
        else:
            proc = subprocess.run(argv, capture_output=True, text=True)
            err_tail = ((proc.stderr or proc.stdout or "").strip()[-1000:]
                        if proc.returncode != 0 else None)
        exit_code = proc.returncode
    except (subprocess.SubprocessError, OSError) as e:
        exit_code = 255
        err_tail = f"{type(e).__name__}: {e}"
    return exit_code, time.perf_counter() - start, err_tail


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(rows: list[SubjectRow], *,
              extra_argv: list[str] | None = None,
              argv_prefix: list[str] | None = None,
              stream_output: bool = True,
              audit_after_clean: bool = False,
              audit_argv_prefix: list[str] | None = None,
              parallel: int = 1,
              heartbeat_interval_s: int = 30,
              ) -> list[SubjectOutcome]:
    """Drive the batch. If ``parallel`` > 1, dispatch subjects via a
    ThreadPoolExecutor of that width; each worker's output is line-
    prefixed with ``[SUBJECT_CODE]`` so parallel logs stay grep-able.
    A heartbeat thread prints "N/M done, K in flight" every 30s so the
    operator can confirm work is happening even when individual
    subjects are quiet.

    If ``audit_after_clean``, runs ``audit-subject-eeg`` on the
    output_path after a successful clean; audit failures roll up into
    the SubjectOutcome and count against the batch exit code. Continues
    past per-subject failures; caller reads the returned list to
    decide the process exit code.
    """
    if parallel > 1:
        return _run_batch_parallel(
            rows, extra_argv=extra_argv, argv_prefix=argv_prefix,
            audit_after_clean=audit_after_clean,
            audit_argv_prefix=audit_argv_prefix,
            parallel=parallel,
            heartbeat_interval_s=heartbeat_interval_s)
    outcomes: list[SubjectOutcome] = []
    total = len(rows)
    for row in rows:
        print(f"\n{'=' * 72}", flush=True)
        print(f"[{row.row_index}/{total}] {row.subject_code}  "
              f"input={row.input_path}", flush=True)
        print(f"{'=' * 72}", flush=True)
        outcome = clean_one_subject(
            row, extra_argv=extra_argv, argv_prefix=argv_prefix,
            stream_output=stream_output)

        if audit_after_clean and outcome.clean_succeeded:
            print(f"\n--- audit {row.subject_code} ---", flush=True)
            # Same flags as the review-phase audit -- see
            # POST_CLEAN_AUDIT_FLAGS. Header info visible, annotation
            # matches suppressed (TUI or a prior review already showed
            # them). Callers overriding argv_prefix win.
            effective_prefix = (audit_argv_prefix
                                or _default_audit_argv_prefix()
                                + POST_CLEAN_AUDIT_FLAGS)
            code, elapsed, err = audit_one_subject(
                row.output_path,
                argv_prefix=effective_prefix,
                stream_output=stream_output)
            outcome.audit_exit_code = code
            outcome.audit_elapsed_s = elapsed
            outcome.audit_error_message = err

        outcomes.append(outcome)
        status = "OK " if outcome.succeeded else "FAIL"
        aud = ""
        if outcome.audit_exit_code is not None:
            aud = (f"  audit_exit={outcome.audit_exit_code} "
                   f"({outcome.audit_elapsed_s:.1f} s)")
        print(f"[{row.row_index}/{total}] {status} {row.subject_code}  "
              f"(clean {outcome.elapsed_s:.1f} s, "
              f"exit={outcome.exit_code}){aud}", flush=True)
    return outcomes


def _run_batch_parallel(rows: list[SubjectRow], *,
                          extra_argv: list[str] | None,
                          argv_prefix: list[str] | None,
                          audit_after_clean: bool,
                          audit_argv_prefix: list[str] | None,
                          parallel: int,
                          heartbeat_interval_s: int,
                          ) -> list[SubjectOutcome]:
    """Parallel batch worker. See run_batch() for semantics."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    outcomes: list[SubjectOutcome] = []
    total = len(rows)
    in_flight: set[str] = set()
    lock = threading.Lock()
    batch_start = time.perf_counter()
    hb_stop = threading.Event()

    def _heartbeat():
        while not hb_stop.wait(heartbeat_interval_s):
            elapsed = time.perf_counter() - batch_start
            with lock:
                done = len(outcomes)
                inflight_snapshot = sorted(in_flight)
            hrs, rem = divmod(int(elapsed), 3600)
            mins, secs = divmod(rem, 60)
            print(f"\n[heartbeat] {hrs:02d}:{mins:02d}:{secs:02d} elapsed, "
                  f"{done}/{total} subject(s) complete, "
                  f"{len(inflight_snapshot)} in flight: "
                  f"{', '.join(inflight_snapshot) or '(none)'}",
                  flush=True)
    hb_thread = threading.Thread(target=_heartbeat, daemon=True,
                                   name="clean-batch-heartbeat")
    hb_thread.start()

    def _work_one(row: SubjectRow) -> SubjectOutcome:
        with lock:
            in_flight.add(row.subject_code)
        try:
            outcome = clean_one_subject(
                row, extra_argv=extra_argv, argv_prefix=argv_prefix,
                # Prefix per-subject so parallel workers' output stays
                # attributable (grep [R1XXXA] for one subject's log).
                prefix_output=row.subject_code)
            if audit_after_clean and outcome.clean_succeeded:
                effective_prefix = (audit_argv_prefix
                                     or _default_audit_argv_prefix()
                                     + POST_CLEAN_AUDIT_FLAGS)
                # Audit also prefixed so its lines are attributable.
                argv = effective_prefix + [outcome.output_path]
                exit_code, err = _stream_subprocess_with_prefix(
                    argv, f"{row.subject_code}:audit")
                outcome.audit_exit_code = exit_code
                outcome.audit_elapsed_s = 0.0
                outcome.audit_error_message = err
        finally:
            with lock:
                in_flight.discard(row.subject_code)
        return outcome

    print(f"[batch] starting {total} subject(s) with parallel={parallel}. "
          f"Per-subject output prefixed [SUBJECT_CODE]. "
          f"Heartbeat every {heartbeat_interval_s}s.",
          flush=True)
    with ThreadPoolExecutor(max_workers=parallel,
                              thread_name_prefix="clean-worker") as pool:
        futures = {pool.submit(_work_one, row): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                outcome = fut.result()
            except Exception as e:
                outcome = SubjectOutcome(
                    row_index=row.row_index,
                    subject_code=row.subject_code,
                    input_path=row.input_path,
                    output_path=row.output_path,
                    exit_code=255, elapsed_s=0.0,
                    error_message=f"worker crash: {type(e).__name__}: {e}",
                )
            with lock:
                outcomes.append(outcome)
            status = "OK " if outcome.succeeded else "FAIL"
            aud = ""
            if outcome.audit_exit_code is not None:
                aud = f"  audit_exit={outcome.audit_exit_code}"
            print(f"[{row.row_index}/{total}] {status} "
                  f"{row.subject_code}  "
                  f"(clean {outcome.elapsed_s:.1f} s, "
                  f"exit={outcome.exit_code}){aud}", flush=True)

    hb_stop.set()
    # Preserve input order for the summary + downstream review phase.
    outcomes.sort(key=lambda o: o.row_index)
    return outcomes


def run_review_phase(outcomes: list[SubjectOutcome]) -> None:
    """Post-batch pass: for each successfully-cleaned subject, run the
    audit + launch the annotation-review TUI. Runs sequentially so the
    operator handles one subject at a time. Skipped in headless
    environments (no TTY) with a per-subject hint for manual re-run.

    Subjects that failed to clean are skipped (they have no cleaned
    output to review). Errors in audit/TUI are printed and the loop
    continues -- no single subject aborts the whole review phase.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(f"\n[!] Skipping post-batch audit + review-TUI phase — no TTY.",
              flush=True)
        print(f"    Re-run per subject manually, e.g.:")
        for o in outcomes:
            if o.succeeded:
                print(f"      audit-subject-eeg {o.output_path}  &&  "
                      f"annotation-review-eeg --subject-dir "
                      f"{os.path.dirname(o.output_path.rstrip('/'))} "
                      f"--subfolder {os.path.basename(o.output_path.rstrip('/'))} "
                      f"--preload-all")
        return

    reviewable = [o for o in outcomes if o.succeeded]
    n_skipped = len(outcomes) - len(reviewable)
    if not reviewable:
        print(f"\n[review-phase] No successfully-cleaned subjects to review.",
              flush=True)
        return

    print(f"\n{'=' * 72}", flush=True)
    print(f"[review-phase] audit + TUI for {len(reviewable)} subject(s) "
          f"({n_skipped} skipped due to clean failure)", flush=True)
    print(f"{'=' * 72}", flush=True)

    for i, o in enumerate(reviewable, 1):
        print(f"\n{'-' * 72}", flush=True)
        print(f"[review-phase {i}/{len(reviewable)}] {o.subject_code}  "
              f"({o.output_path})", flush=True)
        print(f"{'-' * 72}", flush=True)

        # ---- audit (subprocess so failures don't taint the batch process).
        # Same flag choices as the single-subject auto-launch (see
        # clean_subject_eeg._run_audit_and_launch_review):
        #   --no-notebook            skip ~10 s ipynb/html render
        #   --hide-annotation-flags  TUI shows every annotation
        #   -v                       show every check status, not just
        #                            fails -- operator wants to confirm
        #                            header_phi_residue etc. passed
        #   --print-edf-header       dump unique patient_id / startdate
        #                            so operator visually verifies the
        #                            header was cleaned + dates anchored
        try:
            audit_argv_prefix = (_default_audit_argv_prefix()
                                  + POST_CLEAN_AUDIT_FLAGS)
            audit_rc, _elapsed, _err = audit_one_subject(
                o.output_path,
                argv_prefix=audit_argv_prefix,
                stream_output=True)
            if audit_rc != 0:
                print(f"[!] audit for {o.subject_code} exited {audit_rc}; "
                      f"continuing to TUI anyway.", flush=True)
        except Exception as e:
            print(f"[!] audit for {o.subject_code} failed ({type(e).__name__}"
                  f": {e}); continuing to TUI anyway.", flush=True)

        # ---- TUI (subprocess inherits stdin/stdout so the terminal
        # actually reaches the reviewer). Preflight expects
        # <subject_dir>/<subfolder>/deidentify.json; the output_path IS
        # that inner dir, so subject_dir = its parent.
        out_norm = o.output_path.rstrip("/")
        subject_dir = os.path.dirname(out_norm) or "."
        subfolder = os.path.basename(out_norm)
        tui_argv = [sys.executable, "-m", "clean_eeg.annotation_review_cli",
                    "--subject-dir", subject_dir,
                    "--subfolder", subfolder,
                    "--preload-all"]
        try:
            proc = subprocess.run(tui_argv)
            if proc.returncode != 0:
                print(f"[!] TUI for {o.subject_code} exited "
                      f"{proc.returncode}. Re-run manually with: "
                      f"{' '.join(tui_argv)}", flush=True)
        except KeyboardInterrupt:
            # Ctrl-C at the batch-loop level (as opposed to inside the
            # TUI, which the TUI intercepts) means the operator wants
            # to bail on the whole review phase.
            print(f"\n[review-phase] aborted by operator at "
                  f"{o.subject_code}. Remaining subjects can be reviewed "
                  f"later per-subject.", flush=True)
            return
        except Exception as e:
            print(f"[!] TUI launch failed for {o.subject_code} "
                  f"({type(e).__name__}: {e}). Re-run manually with: "
                  f"{' '.join(tui_argv)}", flush=True)

    print(f"\n[review-phase] complete.", flush=True)


def _print_summary(outcomes: list[SubjectOutcome]) -> None:
    n_ok = sum(1 for o in outcomes if o.succeeded)
    n_fail = len(outcomes) - n_ok
    clean_only_s = sum(o.elapsed_s for o in outcomes)
    audit_s = sum(o.audit_elapsed_s or 0 for o in outcomes)
    total_s = clean_only_s + audit_s
    print(f"\n=== CLEAN BATCH SUMMARY  ({total_s:.1f} s total wall time) ===")
    print(f"  attempted:  {len(outcomes)}")
    print(f"  succeeded:  {n_ok}")
    print(f"  failed:     {n_fail}")
    if audit_s:
        print(f"  clean time: {clean_only_s:.1f} s   "
              f"audit time: {audit_s:.1f} s")
    non_interactive_aborts: list[SubjectOutcome] = []
    for o in outcomes:
        if o.succeeded:
            continue
        err = o.error_message or ""
        if "[non-interactive]" in err:
            # Track separately so the operator's morning-triage list
            # highlights "subjects that needed a human answer" versus
            # real errors. Non-interactive aborts are always fixable
            # (add a bypass flag, or re-run that subject with a TTY).
            non_interactive_aborts.append(o)
        if not o.clean_succeeded:
            err_tail = err.splitlines()[-1:]
            print(f"  FAIL row {o.row_index} {o.subject_code} [clean]: "
                  f"exit={o.exit_code} :: "
                  f"{err_tail[0] if err_tail else ''}")
        else:
            # clean passed, audit failed
            err_tail = (o.audit_error_message or "").splitlines()[-1:]
            print(f"  FAIL row {o.row_index} {o.subject_code} [audit]: "
                  f"exit={o.audit_exit_code} :: "
                  f"{err_tail[0] if err_tail else ''}")

    if non_interactive_aborts:
        print(f"\n  [!] {len(non_interactive_aborts)} subject(s) aborted "
              f"because the pipeline hit an interactive prompt in "
              f"non-interactive mode:")
        for o in non_interactive_aborts:
            # Show the offending prompt text so the operator knows what
            # bypass flag / CSV correction is needed.
            for line in (o.error_message or "").splitlines():
                if "[non-interactive]" in line and "refusing to prompt" in line:
                    print(f"    {o.subject_code}: "
                          f"{line.split('refusing to prompt:', 1)[-1].strip()}")
                    break
        print(f"  Each of these subjects can be re-run individually "
              f"with an appropriate bypass flag once the underlying "
              f"issue is confirmed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clean-batch-eeg",
        description=(
            "Batch-drive clean_subject_eeg from a CSV manifest. "
            "One row per subject; continue-on-failure with aggregated "
            "summary at the end. Extra positional args after '--' are "
            "forwarded verbatim to each per-subject invocation, so the "
            "same batch can e.g. --wipe-annotations for all rows."),
    )
    p.add_argument("--subjects-csv", type=Path, required=True,
                   help=("CSV path. Required columns: "
                         f"{', '.join(REQUIRED_COLUMNS)}. "
                         f"Optional: {', '.join(OPTIONAL_COLUMNS)}."))
    p.add_argument("--quiet-child-output", action="store_true",
                   help=("Capture per-subject stdout+stderr instead of "
                         "streaming it live. Only the summary + the last "
                         "1000 chars of any failure appear. Useful for "
                         "unattended runs where the terminal is not being "
                         "watched."))
    p.add_argument("--only-subjects", nargs="+", default=None,
                   metavar="SUBJECT_CODE",
                   help=("Run only the listed subject codes (matched "
                         "against the CSV's subject_code column). Useful "
                         "for smoke-testing one subject before releasing "
                         "the whole batch. Warns if any listed code did "
                         "not match a row."))
    p.add_argument("--only-subjects-file", type=Path, default=None,
                   metavar="FILE",
                   help=("File with one subject code per line (blank "
                         "lines and '#' comments ignored). Merged with "
                         "--only-subjects if both are given. Lets you "
                         "generate the CSV once (with every subject) "
                         "and drive successive runs by editing a filter "
                         "file instead of re-running the extractor."))
    p.add_argument("--audit-after-clean", action="store_true",
                   help=("After each successful clean, invoke "
                         "audit-subject-eeg against the output_path. "
                         "Audit failures roll up into the subject's "
                         "outcome and count against the batch exit code."))
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help=("Run N subject clean subprocesses concurrently "
                         "(default: 1, sequential). Each worker's output "
                         "is line-prefixed with [SUBJECT_CODE] so parallel "
                         "logs stay grep-able. A heartbeat line every 30s "
                         "shows how many subjects are complete and which "
                         "are still in flight. Watch memory / network I/O "
                         "-- N=4 on a 130-file-per-subject workload can "
                         "saturate NFS."))
    p.add_argument("--no-review-after-batch", "--no_review_after_batch",
                   dest="no_review_after_batch", action="store_true",
                   help=("Suppress the post-batch review phase (default: "
                         "on). Normally, after every subject has finished "
                         "cleaning, the batch iterates successfully-"
                         "cleaned subjects one at a time running "
                         "audit-subject-eeg then launching the "
                         "annotation-review TUI. Skipped automatically "
                         "when stdin/stdout aren't TTYs (nohup, cron, "
                         "SSH-without-PTY); this flag lets you also "
                         "suppress it interactively (e.g. if you want to "
                         "hand the reviews off to a colleague later)."))
    return p


def main(argv: list[str] | None = None) -> int:
    """Split argv at ``--``: left half is our CLI, right half is
    forwarded verbatim to each ``clean_subject_eeg`` invocation."""
    argv = list(argv) if argv is not None else sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        our_argv, extra = argv[:i], argv[i + 1:]
    else:
        our_argv, extra = argv, []
    args = _build_parser().parse_args(our_argv)

    try:
        rows = parse_subjects_csv(args.subjects_csv)
    except CsvSchemaError as e:
        print(f"CSV schema error: {e}", file=sys.stderr)
        return 2
    if not rows:
        print(f"No subject rows found in {args.subjects_csv}", file=sys.stderr)
        return 1

    only_subjects: list[str] = list(args.only_subjects or [])
    if args.only_subjects_file is not None:
        try:
            only_subjects.extend(
                load_subject_codes_from_file(args.only_subjects_file))
        except OSError as e:
            print(f"Cannot read --only-subjects-file "
                  f"{args.only_subjects_file}: {e}", file=sys.stderr)
            return 2
    rows = filter_rows_by_subject(rows, only_subjects or None)
    if not rows:
        print(f"No rows match the subject filter "
              f"(--only-subjects={args.only_subjects}, "
              f"--only-subjects-file={args.only_subjects_file})",
              file=sys.stderr)
        return 1

    outcomes = run_batch(
        rows, extra_argv=extra,
        stream_output=not args.quiet_child_output,
        audit_after_clean=args.audit_after_clean,
        parallel=args.parallel)
    _print_summary(outcomes)

    if not args.no_review_after_batch:
        run_review_phase(outcomes)

    return 0 if all(o.succeeded for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
