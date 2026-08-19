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
import csv
import subprocess
import sys
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
        """
        argv = [
            "--input_path", self.input_path,
            "--output_path", self.output_path,
            "--subject_code", self.subject_code,
            "--first_name", self.first_name,
            "--last_name", self.last_name,
        ]
        if self.middle_name:
            argv += ["--middle_name", self.middle_name]
        else:
            argv += ["--no_middle_name"]
        if extra_argv:
            argv += list(extra_argv)
        return argv


@dataclass
class SubjectOutcome:
    """Per-subject outcome after cleaning is attempted."""
    row_index: int
    subject_code: str
    input_path: str
    output_path: str
    exit_code: int
    elapsed_s: float
    error_message: str | None = None   # None on success

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


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
    depending on the ``clean-subject-eeg`` console script being on PATH."""
    return [sys.executable, "-m", "clean_eeg.clean_subject_eeg"]


def clean_one_subject(row: SubjectRow, *, extra_argv: list[str] | None = None,
                      argv_prefix: list[str] | None = None,
                      stream_output: bool = True,
                      ) -> SubjectOutcome:
    """Shell out to ``clean_subject_eeg`` for one row. Streams the
    child's stdout+stderr to the parent's terminal by default so an
    operator watching the batch sees per-file progress live.

    Never raises; a subprocess or OS-level error is captured into the
    ``error_message`` field and the exit code is fabricated as 255.
    """
    argv = (argv_prefix or _default_clean_argv_prefix()) + row.to_clean_argv(
        extra_argv=extra_argv)
    start = time.perf_counter()
    try:
        if stream_output:
            proc = subprocess.run(argv)
            error_message = None if proc.returncode == 0 else (
                f"clean_subject_eeg exited {proc.returncode}")
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
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(rows: list[SubjectRow], *,
              extra_argv: list[str] | None = None,
              argv_prefix: list[str] | None = None,
              stream_output: bool = True,
              ) -> list[SubjectOutcome]:
    """Iterate ``rows`` sequentially, calling :func:`clean_one_subject`
    on each. Prints a header before each subject so an operator can
    match streamed output to the row it belongs to. Continues past
    per-subject failures; caller reads the returned list to detect
    them and decide the process exit code.
    """
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
        outcomes.append(outcome)
        status = "OK " if outcome.succeeded else "FAIL"
        print(f"[{row.row_index}/{total}] {status} {row.subject_code}  "
              f"({outcome.elapsed_s:.1f} s, exit={outcome.exit_code})",
              flush=True)
    return outcomes


def _print_summary(outcomes: list[SubjectOutcome]) -> None:
    n_ok = sum(1 for o in outcomes if o.succeeded)
    n_fail = len(outcomes) - n_ok
    total_s = sum(o.elapsed_s for o in outcomes)
    print(f"\n=== CLEAN BATCH SUMMARY  ({total_s:.1f} s total wall time) ===")
    print(f"  attempted:  {len(outcomes)}")
    print(f"  succeeded:  {n_ok}")
    print(f"  failed:     {n_fail}")
    for o in outcomes:
        if not o.succeeded:
            err_tail = (o.error_message or "").splitlines()[-1:]
            print(f"  FAIL row {o.row_index} {o.subject_code}: "
                  f"exit={o.exit_code} :: {err_tail[0] if err_tail else ''}")


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

    outcomes = run_batch(
        rows, extra_argv=extra,
        stream_output=not args.quiet_child_output)
    _print_summary(outcomes)
    return 0 if all(o.succeeded for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
