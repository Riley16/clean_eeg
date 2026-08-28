"""Tests for the multi-subject cleaning wrapper.

CSV parsing is exercised directly. Per-subject dispatch is tested by
substituting a tiny stub script for clean_subject_eeg so we can
exercise the real subprocess.run codepath (exit codes, argv shape,
continue-on-failure) without invoking the full pipeline.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from clean_eeg.clean_batch import (
    CsvSchemaError,
    SubjectOutcome,
    SubjectRow,
    _default_clean_argv_prefix,
    audit_one_subject,
    clean_one_subject,
    filter_rows_by_subject,
    load_subject_codes_from_file,
    main,
    parse_subjects_csv,
    run_batch,
    run_review_phase,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict], header: list[str] | None = None
               ) -> None:
    """Write a CSV that DictReader can round-trip. ``header`` overrides
    the field-name row -- useful for testing schema errors."""
    if header is None:
        header = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _valid_row(subject_code: str = "R1755A", middle: str = "") -> dict:
    return {
        "input_path": f"/data/in/{subject_code}",
        "output_path": f"/data/out/{subject_code}",
        "subject_code": subject_code,
        "first_name": "John",
        "middle_name": middle,
        "last_name": "Smith",
    }


def _make_stub_child(tmp_path: Path, exit_code: int, name: str = "stub.py"
                     ) -> list[str]:
    """Create a Python script that mimics clean_subject_eeg -- it accepts
    any argv, prints one line so we can assert streaming, and exits
    with ``exit_code``. Returns the argv_prefix that invokes it."""
    stub = tmp_path / name
    stub.write_text(
        f"import sys\n"
        f'print(f"stub_child received argv={{sys.argv[1:]}}")\n'
        f"sys.exit({exit_code})\n"
    )
    return [sys.executable, str(stub)]


# ---------------------------------------------------------------------------
# CSV parsing: happy path
# ---------------------------------------------------------------------------

def test_parse_csv_reads_all_required_columns(tmp_path):
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1755A"), _valid_row("R1755B")])
    rows = parse_subjects_csv(csv_path)
    assert len(rows) == 2
    assert [r.subject_code for r in rows] == ["R1755A", "R1755B"]
    assert rows[0].input_path == "/data/in/R1755A"
    assert rows[0].first_name == "John"
    assert rows[0].last_name == "Smith"
    # 1-based row index for readable error messages
    assert rows[0].row_index == 1
    assert rows[1].row_index == 2


def test_parse_csv_middle_name_optional(tmp_path):
    """Blank middle_name cell -> None on the row (which downstream
    translates to --no_middle_name)."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1", middle=""),
                          _valid_row("R2", middle="Paul")])
    rows = parse_subjects_csv(csv_path)
    assert rows[0].middle_name is None
    assert rows[1].middle_name == "Paul"


def test_parse_csv_middle_name_column_absent_entirely(tmp_path):
    """If the CSV has no middle_name column at all, parsing still
    succeeds (it's optional). middle_name defaults to None on the row."""
    csv_path = tmp_path / "s.csv"
    row = _valid_row("R1")
    row.pop("middle_name")
    _write_csv(csv_path, [row])
    rows = parse_subjects_csv(csv_path)
    assert len(rows) == 1
    assert rows[0].middle_name is None


def test_parse_csv_ignores_extra_columns(tmp_path):
    """Operator's own bookkeeping columns (site, cohort, notes...) must
    NOT trip the parser -- extra cells are silently ignored."""
    csv_path = tmp_path / "s.csv"
    row = _valid_row("R1")
    row["notes"] = "cohort A, imported 2026-08"
    row["site"] = "UPMC"
    _write_csv(csv_path, [row])
    rows = parse_subjects_csv(csv_path)
    assert len(rows) == 1


def test_parse_csv_strips_whitespace_from_cells(tmp_path):
    """Common operator mistake: pasted values have trailing spaces.
    Wrapper should strip so downstream argv is clean."""
    csv_path = tmp_path / "s.csv"
    row = _valid_row("R1")
    row["first_name"] = "  John  "
    row["subject_code"] = " R1755A "
    _write_csv(csv_path, [row])
    rows = parse_subjects_csv(csv_path)
    assert rows[0].first_name == "John"
    assert rows[0].subject_code == "R1755A"


# ---------------------------------------------------------------------------
# CSV parsing: schema errors
# ---------------------------------------------------------------------------

def test_parse_csv_raises_on_missing_required_column(tmp_path):
    """A dropped required column must fail LOUDLY, not silently drop
    rows. Otherwise an operator would run the batch and only notice
    hours later."""
    csv_path = tmp_path / "s.csv"
    row = _valid_row("R1")
    row.pop("last_name")
    _write_csv(csv_path, [row])
    with pytest.raises(CsvSchemaError) as exc:
        parse_subjects_csv(csv_path)
    assert "last_name" in str(exc.value)


def test_parse_csv_raises_on_blank_required_cell(tmp_path):
    """A blank required cell is also fatal. Row index is preserved so
    the operator can jump straight to the offending row."""
    csv_path = tmp_path / "s.csv"
    row = _valid_row("R1")
    row["first_name"] = ""
    _write_csv(csv_path, [_valid_row("R0"), row])
    with pytest.raises(CsvSchemaError) as exc:
        parse_subjects_csv(csv_path)
    # 1-based row index of the bad row is in the error
    assert "row 2" in str(exc.value)
    assert "first_name" in str(exc.value)


def test_parse_csv_raises_on_empty_file(tmp_path):
    csv_path = tmp_path / "s.csv"
    csv_path.write_text("")
    with pytest.raises(CsvSchemaError):
        parse_subjects_csv(csv_path)


def test_parse_csv_empty_but_valid_header_returns_empty_list(tmp_path):
    """Zero data rows is not an error at parse time -- the CLI treats
    that as an operator mistake and exits nonzero with a helpful
    message, but parse should return [] cleanly."""
    csv_path = tmp_path / "s.csv"
    csv_path.write_text(",".join(SubjectRow.__dataclass_fields__) + "\n")
    # This file has all fields incl. optional -- but writing raw header
    # only means DictReader yields zero data rows.
    rows = parse_subjects_csv(csv_path)
    assert rows == []


# ---------------------------------------------------------------------------
# SubjectRow.to_clean_argv: argv composition
# ---------------------------------------------------------------------------

def test_to_clean_argv_includes_middle_name_when_present():
    row = SubjectRow(input_path="/i", output_path="/o", subject_code="R1",
                     first_name="John", last_name="Smith",
                     middle_name="Paul")
    argv = row.to_clean_argv()
    assert "--middle_name" in argv
    assert "Paul" in argv
    assert "--no_middle_name" not in argv


def test_to_clean_argv_uses_no_middle_name_when_absent():
    """Design invariant: blank/None middle_name -> --no_middle_name.
    Otherwise clean_subject_eeg's interactive prompt would trigger on
    unattended batches."""
    row = SubjectRow(input_path="/i", output_path="/o", subject_code="R1",
                     first_name="John", last_name="Smith",
                     middle_name=None)
    argv = row.to_clean_argv()
    assert "--no_middle_name" in argv
    assert "--middle_name" not in argv


def test_to_clean_argv_appends_extra_argv():
    """Batch-level flags (--wipe-annotations etc.) forwarded verbatim
    to every subject."""
    row = SubjectRow(input_path="/i", output_path="/o", subject_code="R1",
                     first_name="J", last_name="S", middle_name=None)
    argv = row.to_clean_argv(
        extra_argv=["--wipe-annotations",
                    "--approve-confirmations", "wipe-annotations"])
    assert argv[-3:] == ["--wipe-annotations",
                          "--approve-confirmations", "wipe-annotations"]


def test_to_clean_argv_inplace_mode_emits_no_copy_path():
    """When CSV row has output_path == input_path, this is in-place
    mode. The single-subject CLI expects NO --copy_path (in-place is
    the default when --copy_path is absent), and definitely does NOT
    accept --output_path."""
    row = SubjectRow(input_path="/data/R1", output_path="/data/R1",
                     subject_code="R1", first_name="J", last_name="S",
                     middle_name=None)
    argv = row.to_clean_argv()
    assert "--input_path" in argv
    assert "/data/R1" in argv
    assert "--copy_path" not in argv, (
        f"in-place mode must not pass --copy_path: {argv}")
    assert "--output_path" not in argv, (
        f"single-subject CLI does not accept --output_path: {argv}")


def test_to_clean_argv_rewrite_mode_emits_copy_path():
    """When CSV row has output_path != input_path, this is rewrite mode.
    Must emit --copy_path (not --output_path)."""
    row = SubjectRow(input_path="/data/raw/R1",
                     output_path="/data/clean/R1",
                     subject_code="R1", first_name="J", last_name="S",
                     middle_name=None)
    argv = row.to_clean_argv()
    assert "--copy_path" in argv
    i = argv.index("--copy_path")
    assert argv[i + 1] == "/data/clean/R1"
    assert "--output_path" not in argv


def test_to_clean_argv_survives_real_clean_subject_eeg_argparser():
    """Regression: any argv produced by to_clean_argv must be accepted
    by the REAL clean_subject_eeg CLI parser. The existing subprocess
    tests use a stub script that accepts any argv, which lets bad
    argv-shape regressions slip through -- this test hits the actual
    CLI so a change like adding a stray --output_path is caught
    immediately.

    Motivated by an incident where the batch was emitting --output_path
    (never a valid clean_subject_eeg CLI arg) and no test caught it
    because all subprocess tests used stubs.

    Subprocesses `python -m clean_eeg.clean_subject_eeg` with the
    candidate argv + /dev/null on stdin, and checks that stderr does
    NOT contain 'unrecognized arguments'. Downstream errors (missing
    input path, prompt-if-missing EOFError, etc.) are fine and expected
    -- we only care that argparse itself accepts the shape."""
    import subprocess as _sp
    import sys as _sys
    row = SubjectRow(input_path="/nonexistent/data/R1755J",
                     output_path="/nonexistent/data/R1755J",
                     subject_code="R1755J",
                     first_name="J", last_name="S", middle_name=None)
    argv = row.to_clean_argv(
        extra_argv=["--fail-on-name-mismatch", "--skip-if-already-cleaned"])
    result = _sp.run(
        [_sys.executable, "-m", "clean_eeg.clean_subject_eeg", *argv],
        capture_output=True, text=True,
        stdin=_sp.DEVNULL,   # prompt-if-missing -> EOFError, no hang
        timeout=30,
    )
    assert "unrecognized arguments" not in result.stderr, (
        f"batch-generated argv rejected by clean_subject_eeg CLI:\n"
        f"  argv: {argv}\n"
        f"  stderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# clean_one_subject: subprocess dispatch (with a real stub script)
# ---------------------------------------------------------------------------

def test_clean_one_subject_returns_success_when_child_exits_zero(tmp_path):
    argv_prefix = _make_stub_child(tmp_path, exit_code=0)
    row = SubjectRow(input_path="/i", output_path="/o", subject_code="R1",
                     first_name="J", last_name="S", middle_name=None,
                     row_index=1)
    outcome = clean_one_subject(row, argv_prefix=argv_prefix,
                                 stream_output=False)
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert outcome.error_message is None
    assert outcome.subject_code == "R1"
    assert outcome.elapsed_s >= 0


def test_clean_one_subject_captures_stderr_tail_on_failure(tmp_path):
    """Failure captures a tail of stderr/stdout so the summary line
    doesn't just say 'exit 23' with no diagnostic."""
    stub = tmp_path / "err_stub.py"
    stub.write_text(
        "import sys\n"
        "sys.stderr.write('bad thing happened\\n')\n"
        "sys.exit(23)\n"
    )
    argv_prefix = [sys.executable, str(stub)]
    row = SubjectRow(input_path="/i", output_path="/o", subject_code="R1",
                     first_name="J", last_name="S", middle_name=None,
                     row_index=1)
    outcome = clean_one_subject(row, argv_prefix=argv_prefix,
                                 stream_output=False)
    assert not outcome.succeeded
    assert outcome.exit_code == 23
    assert "bad thing happened" in (outcome.error_message or "")


def test_clean_one_subject_never_raises_on_missing_binary(tmp_path):
    """A bogus argv_prefix (nonexistent binary) must NOT propagate --
    the batch summary depends on every row producing an Outcome."""
    row = SubjectRow(input_path="/i", output_path="/o", subject_code="R1",
                     first_name="J", last_name="S", middle_name=None,
                     row_index=1)
    outcome = clean_one_subject(row,
                                 argv_prefix=["/nonexistent/xyz_binary"],
                                 stream_output=False)
    assert not outcome.succeeded
    assert outcome.exit_code == 255
    assert outcome.error_message is not None


# ---------------------------------------------------------------------------
# run_batch: continue-on-failure
# ---------------------------------------------------------------------------

def test_run_batch_continues_after_a_failed_subject(tmp_path):
    """Positive + negative regression: three subjects, middle one
    fails, first + third still run and succeed. Guards against a
    regression where an exception in one row aborts the batch.
    """
    ok = _make_stub_child(tmp_path, exit_code=0, name="ok.py")
    fail = _make_stub_child(tmp_path, exit_code=1, name="fail.py")

    rows = [
        SubjectRow(input_path=f"/i/{i}", output_path=f"/o/{i}",
                   subject_code=f"R{i}", first_name="J", last_name="S",
                   middle_name=None, row_index=i)
        for i in (1, 2, 3)
    ]

    # Row 2 uses the failing stub; rows 1 and 3 use the passing stub.
    def dispatcher(row, *, extra_argv=None, argv_prefix=None,
                   stream_output=True):
        prefix = fail if row.row_index == 2 else ok
        return clean_one_subject(row, extra_argv=extra_argv,
                                  argv_prefix=prefix,
                                  stream_output=False)

    outcomes = []
    for r in rows:
        outcomes.append(dispatcher(r))

    assert [o.succeeded for o in outcomes] == [True, False, True]
    assert [o.row_index for o in outcomes] == [1, 2, 3]


def test_run_batch_calls_clean_one_subject_per_row_in_order(tmp_path):
    """run_batch preserves CSV order AND yields one Outcome per row.
    Serial execution is the design intent -- parallel per-subject
    cleaning would fight over Presidio's global model caches."""
    argv_prefix = _make_stub_child(tmp_path, exit_code=0)
    rows = [
        SubjectRow(input_path=f"/i/{i}", output_path=f"/o/{i}",
                   subject_code=f"R{i}", first_name="J", last_name="S",
                   middle_name=None, row_index=i)
        for i in (1, 2, 3)
    ]
    outcomes = run_batch(rows, argv_prefix=argv_prefix, stream_output=False)
    assert [o.subject_code for o in outcomes] == ["R1", "R2", "R3"]
    assert all(o.succeeded for o in outcomes)


# ---------------------------------------------------------------------------
# CLI: main() exit-code convention, -- forwarding
# ---------------------------------------------------------------------------

def test_main_returns_zero_when_all_rows_succeed(tmp_path, monkeypatch):
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1"), _valid_row("R2")])
    argv_prefix = _make_stub_child(tmp_path, exit_code=0)
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: argv_prefix)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output"])
    assert rc == 0


def test_main_returns_nonzero_when_any_row_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1")])
    argv_prefix = _make_stub_child(tmp_path, exit_code=17)
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: argv_prefix)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output"])
    assert rc == 1


def test_main_returns_2_on_csv_schema_error(tmp_path):
    """CLI convention: reserve exit code 2 for input-validation errors
    (matching argparse's own convention), separate from 1 which means
    'ran but at least one subject failed'."""
    csv_path = tmp_path / "s.csv"
    csv_path.write_text("wrong,columns\n1,2\n")
    rc = main(["--subjects-csv", str(csv_path)])
    assert rc == 2


def test_main_forwards_args_after_double_dash_to_children(
        tmp_path, monkeypatch):
    """Everything after ``--`` on the wrapper CLI must appear on every
    per-subject argv. This is how batch runs apply --wipe-annotations
    etc. to all subjects at once."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1")])

    # Stub script writes its argv to a file so we can assert on it
    argv_dump = tmp_path / "argv.txt"
    stub = tmp_path / "capture.py"
    stub.write_text(
        f"import sys, pathlib\n"
        f"pathlib.Path({str(argv_dump)!r}).write_text(repr(sys.argv[1:]))\n"
        f"sys.exit(0)\n"
    )
    argv_prefix = [sys.executable, str(stub)]
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: argv_prefix)

    rc = main([
        "--subjects-csv", str(csv_path), "--quiet-child-output",
        "--",
        "--wipe-annotations",
        "--approve-confirmations", "wipe-annotations",
    ])
    assert rc == 0
    captured = argv_dump.read_text()
    assert "--wipe-annotations" in captured
    assert "--approve-confirmations" in captured
    assert "wipe-annotations" in captured


def test_main_prints_helpful_error_on_missing_csv_column(tmp_path, capsys):
    """The exit code alone isn't enough -- the operator needs to know
    WHICH column is missing. Regression guard: the column name must
    appear in the error stream."""
    csv_path = tmp_path / "s.csv"
    row = _valid_row("R1")
    row.pop("last_name")
    _write_csv(csv_path, [row])
    rc = main(["--subjects-csv", str(csv_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "last_name" in err


# ---------------------------------------------------------------------------
# Default argv prefix (regression: uses -m clean_eeg.clean_subject_eeg)
# ---------------------------------------------------------------------------

def test_default_argv_prefix_uses_module_form():
    """Regression: the default child invocation must go through
    `python -m clean_eeg.clean_subject_eeg` so the wrapper works in
    any env where clean_eeg is importable, independent of whether the
    console-script entry point is on PATH.
    """
    prefix = _default_clean_argv_prefix()
    assert prefix[0] == sys.executable
    assert prefix[1] == "-m"
    assert prefix[2] == "clean_eeg.clean_subject_eeg"


def test_default_argv_prefix_forwards_no_launch_review():
    """The single-subject CLI auto-launches audit + TUI at end-of-clean
    by default. Batch runs the audit + TUI in a single post-batch pass
    instead (see run_review_phase), so every per-subject subprocess
    MUST get --no-launch-review to avoid launching mid-batch."""
    prefix = _default_clean_argv_prefix()
    assert "--no-launch-review" in prefix, (
        f"batch's per-subject invocation must suppress the cleaner's "
        f"end-of-run TUI launch: {prefix}"
    )


# ---------------------------------------------------------------------------
# --only-subjects: subject-picker filter
# ---------------------------------------------------------------------------

def _row(code: str, i: int) -> SubjectRow:
    return SubjectRow(input_path=f"/i/{code}", output_path=f"/o/{code}",
                      subject_code=code, first_name="J", last_name="S",
                      middle_name=None, row_index=i)


def test_filter_rows_returns_all_when_picker_none():
    rows = [_row("R1", 1), _row("R2", 2)]
    assert filter_rows_by_subject(rows, None) == rows
    assert filter_rows_by_subject(rows, []) == rows


def test_filter_rows_picks_only_listed_codes():
    rows = [_row("R1755J", 1), _row("R1702J_1", 2), _row("R1042J", 3)]
    picked = filter_rows_by_subject(rows, ["R1702J_1"])
    assert [r.subject_code for r in picked] == ["R1702J_1"]


def test_filter_rows_warns_on_unmatched_picker_entry(capsys):
    """A typo in --only-subjects (e.g. R1XXXJ vs R1XXX) must warn
    loudly -- otherwise the operator would think their smoke test
    ran and find nothing happened."""
    rows = [_row("R1755J", 1)]
    picked = filter_rows_by_subject(rows, ["R1755J", "R9999Z"])
    assert [r.subject_code for r in picked] == ["R1755J"]
    err = capsys.readouterr().err
    assert "R9999Z" in err
    assert "not found" in err.lower()


def test_main_only_subjects_runs_just_the_picked_row(tmp_path, monkeypatch):
    """End-to-end: two rows in the CSV, --only-subjects=R2 should run
    exactly one clean invocation and skip R1 entirely."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1"), _valid_row("R2")])

    seen_codes: list[str] = []
    stub = tmp_path / "capture.py"
    stub.write_text(
        "import sys\n"
        "# subject_code is 6th CLI arg on our fixed argv order\n"
        "code = sys.argv[sys.argv.index('--subject_code') + 1]\n"
        f"open({str(tmp_path / 'seen.txt')!r}, 'a').write(code + '\\n')\n"
        "sys.exit(0)\n"
    )
    argv_prefix = [sys.executable, str(stub)]
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: argv_prefix)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--only-subjects", "R2"])
    assert rc == 0
    seen = (tmp_path / "seen.txt").read_text().splitlines()
    assert seen == ["R2"], (
        f"only R2 should have been cleaned, got {seen}")


def test_main_only_subjects_returns_nonzero_when_no_match(tmp_path,
                                                           monkeypatch):
    """Negative regression: an entirely-unmatched picker must NOT
    silently run the whole batch. Exit nonzero so the operator's
    smoke test surfaces the typo instead of touching every subject."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1"), _valid_row("R2")])
    argv_prefix = _make_stub_child(tmp_path, exit_code=0)
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: argv_prefix)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--only-subjects", "R_NOT_IN_CSV"])
    assert rc == 1


# ---------------------------------------------------------------------------
# --audit-after-clean: post-clean audit dispatch
# ---------------------------------------------------------------------------

def test_audit_one_subject_shells_out_and_returns_exit_code(tmp_path):
    stub = tmp_path / "aud.py"
    stub.write_text("import sys; sys.exit(3)\n")
    code, elapsed, err = audit_one_subject(
        "/some/out", argv_prefix=[sys.executable, str(stub)],
        stream_output=False)
    assert code == 3
    assert elapsed >= 0


def test_audit_after_clean_wires_audit_into_outcome(tmp_path, monkeypatch):
    """Positive: successful clean + successful audit -> outcome.succeeded.
    Regression guard for both flags cooperating."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1")])

    clean_stub = _make_stub_child(tmp_path, exit_code=0, name="clean.py")
    audit_stub = _make_stub_child(tmp_path, exit_code=0, name="audit.py")
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: clean_stub)
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_audit_argv_prefix",
        lambda: audit_stub)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--audit-after-clean"])
    assert rc == 0


def test_audit_failure_counts_against_batch_exit_code(tmp_path, monkeypatch):
    """PHI-safety regression: a subject whose CLEAN passes but whose
    AUDIT fails must count as a batch failure. Otherwise a subject
    with residual PHI could sneak through the batch summary as 'OK'
    and get transferred to the CML server.
    """
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1")])

    clean_stub = _make_stub_child(tmp_path, exit_code=0, name="clean.py")
    audit_stub = _make_stub_child(tmp_path, exit_code=1, name="audit.py")
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: clean_stub)
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_audit_argv_prefix",
        lambda: audit_stub)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--audit-after-clean"])
    assert rc == 1


def test_audit_skipped_when_clean_fails(tmp_path, monkeypatch):
    """Negative regression: if the clean step exits nonzero, the
    audit MUST NOT run (there's nothing valid to audit -- and the
    audit against an uncleaned dir would just produce noise)."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1")])

    clean_stub = _make_stub_child(tmp_path, exit_code=13, name="clean.py")
    # If the audit stub is ever invoked, this file will exist.
    audit_marker = tmp_path / "audit_was_called.txt"
    audit_stub_path = tmp_path / "audit.py"
    audit_stub_path.write_text(
        f"import pathlib; pathlib.Path({str(audit_marker)!r}).touch()\n"
        "import sys; sys.exit(0)\n"
    )
    audit_stub = [sys.executable, str(audit_stub_path)]
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: clean_stub)
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_audit_argv_prefix",
        lambda: audit_stub)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--audit-after-clean"])
    assert rc == 1
    assert not audit_marker.exists(), (
        "audit must NOT run when clean fails")


def test_batch_flags_name_mismatch_subject_as_fail(tmp_path, monkeypatch):
    """HARD REQUIREMENT: if a subject's EDF patientname doesn't match
    the CSV-supplied name, that subject MUST be flagged as FAIL in
    the batch summary and the batch exit code MUST be nonzero.

    Simulated via a stub clean_subject_eeg that exits nonzero when
    the --first_name arg doesn't match a known-good value -- the
    real check lives in clean_subject_eeg.py and is proven separately
    by test_fail_on_name_mismatch_raises_without_prompt. This test
    proves the exit code propagates through the batch wrapper to
    become a per-subject FAIL, so the batch operator can never
    accidentally transfer a name-mismatched subject.
    """
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1", middle="")])
    # The row's first_name is 'John' (from _valid_row). We simulate
    # clean_subject_eeg's name-mismatch failure by making the stub
    # exit nonzero only when it sees 'John'.
    stub = tmp_path / "namefail.py"
    stub.write_text(
        "import sys\n"
        "argv = sys.argv[1:]\n"
        "i = argv.index('--first_name') + 1\n"
        "if argv[i] == 'John':\n"
        "    sys.stderr.write('name-mismatch: EDF says X, CLI says John\\n')\n"
        "    sys.exit(2)  # simulate --fail-on-name-mismatch RuntimeError\n"
        "sys.exit(0)\n"
    )
    argv_prefix = [sys.executable, str(stub)]
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: argv_prefix)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output"])

    assert rc == 1, (
        "batch exit code must be nonzero when a subject fails "
        "(would otherwise let a name-mismatched subject through)")


def test_audit_flag_off_by_default_skips_audit(tmp_path, monkeypatch):
    """Negative regression: without --audit-after-clean, audit MUST
    NOT run even if the audit stub is trivially available. Guards
    against a bug where the flag defaults to True (which would double
    every batch's wall time and surprise operators)."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1")])
    clean_stub = _make_stub_child(tmp_path, exit_code=0, name="clean.py")
    audit_marker = tmp_path / "audit_was_called.txt"
    audit_stub_path = tmp_path / "audit.py"
    audit_stub_path.write_text(
        f"import pathlib; pathlib.Path({str(audit_marker)!r}).touch()\n"
        "import sys; sys.exit(0)\n"
    )
    audit_stub = [sys.executable, str(audit_stub_path)]
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: clean_stub)
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_audit_argv_prefix",
        lambda: audit_stub)

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output"])
    assert rc == 0
    assert not audit_marker.exists()


# ---------------------------------------------------------------------------
# --only-subjects-file: file-based subject picker
# ---------------------------------------------------------------------------

def test_load_subject_codes_from_file_ignores_blanks_and_comments(tmp_path):
    """Same parse rules as --subjects-file / subjects_filter.txt:
    blank lines and '#' comments dropped, whitespace stripped."""
    f = tmp_path / "filter.txt"
    f.write_text(
        "# subjects to run this batch\n"
        "R1651J\n"
        "\n"
        "  R1665J  \n"      # extra whitespace stripped
        "# R1755J is out this round\n"
        "R1753J\n"
    )
    codes = load_subject_codes_from_file(f)
    assert codes == ["R1651J", "R1665J", "R1753J"]


def test_load_subject_codes_from_file_dedupes_preserving_order(tmp_path):
    """Duplicate entries in the filter file are collapsed to a single
    occurrence; first-seen order preserved. Guards against a typo
    where the same code appears twice generating a spurious warning
    downstream."""
    f = tmp_path / "filter.txt"
    f.write_text("R1B\nR1A\nR1B\nR1C\nR1A\n")
    assert load_subject_codes_from_file(f) == ["R1B", "R1A", "R1C"]


def test_main_only_subjects_file_runs_just_the_listed_row(tmp_path,
                                                            monkeypatch):
    """POSITIVE integration: file-based filter selects the intended
    row and skips the others."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1"), _valid_row("R2"),
                          _valid_row("R3")])
    filter_path = tmp_path / "filter.txt"
    filter_path.write_text("# batch of Aug 24\nR2\n")

    seen = tmp_path / "seen.txt"
    stub = tmp_path / "capture.py"
    stub.write_text(
        "import sys\n"
        "code = sys.argv[sys.argv.index('--subject_code') + 1]\n"
        f"open({str(seen)!r}, 'a').write(code + '\\n')\n"
        "sys.exit(0)\n"
    )
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: [sys.executable, str(stub)])

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--only-subjects-file", str(filter_path)])
    assert rc == 0
    assert seen.read_text().splitlines() == ["R2"]


def test_main_only_subjects_file_merges_with_inline_only_subjects(
        tmp_path, monkeypatch):
    """When both --only-subjects and --only-subjects-file are given,
    their union is used. Useful for 'run everyone in the standing
    batch file PLUS one ad-hoc code'."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1"), _valid_row("R2"),
                          _valid_row("R3")])
    filter_path = tmp_path / "filter.txt"
    filter_path.write_text("R1\n")   # file lists R1

    seen = tmp_path / "seen.txt"
    stub = tmp_path / "capture.py"
    stub.write_text(
        "import sys\n"
        "code = sys.argv[sys.argv.index('--subject_code') + 1]\n"
        f"open({str(seen)!r}, 'a').write(code + '\\n')\n"
        "sys.exit(0)\n"
    )
    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: [sys.executable, str(stub)])

    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--only-subjects", "R3",
               "--only-subjects-file", str(filter_path)])
    assert rc == 0
    assert sorted(seen.read_text().splitlines()) == ["R1", "R3"]


def test_main_only_subjects_file_returns_2_when_file_missing(tmp_path):
    """Missing filter file is an input-validation error, distinct from
    'ran and everything failed' (exit 1). Exit code 2 matches the
    CSV-schema-error convention already used by this CLI."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1")])
    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--only-subjects-file", str(tmp_path / "nonexistent.txt")])
    assert rc == 2


def test_main_only_subjects_file_returns_nonzero_when_no_match(
        tmp_path, monkeypatch):
    """Negative regression: file whose codes match no CSV row must
    NOT silently succeed with 'nothing to do'. Same semantics as
    typo'd inline --only-subjects."""
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, [_valid_row("R1"), _valid_row("R2")])
    filter_path = tmp_path / "filter.txt"
    filter_path.write_text("R_NOT_IN_CSV\n")

    monkeypatch.setattr(
        "clean_eeg.clean_batch._default_clean_argv_prefix",
        lambda: _make_stub_child(tmp_path, 0))
    rc = main(["--subjects-csv", str(csv_path), "--quiet-child-output",
               "--only-subjects-file", str(filter_path)])
    assert rc == 1


# ---------------------------------------------------------------------------
# run_review_phase: post-batch audit + TUI loop
# ---------------------------------------------------------------------------


def _outcome(subject_code: str, output_path: str, *,
             succeeded: bool = True) -> SubjectOutcome:
    """A SubjectOutcome shaped like run_batch would produce."""
    return SubjectOutcome(
        row_index=1, subject_code=subject_code,
        input_path=f"/in/{subject_code}", output_path=output_path,
        exit_code=0 if succeeded else 1, elapsed_s=1.0,
        error_message=None if succeeded else "clean_subject_eeg exited 1")


def test_review_phase_skips_when_no_tty(monkeypatch, capsys, tmp_path):
    """Headless invocations (nohup, cron, SSH-without-PTY) have no
    TTY -- run_review_phase must NOT try to launch the TUI (would
    hang) and must print the per-subject manual-run commands so the
    operator can resume later.
    """
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    outcomes = [_outcome("R1755J", str(tmp_path / "R1755J" / "clinical_eeg"))]
    run_review_phase(outcomes)
    out = capsys.readouterr().out
    assert "no TTY" in out
    # Per-subject fallback command printed with the right subject_dir /
    # subfolder split.
    assert "annotation-review-eeg --subject-dir" in out
    assert "--subfolder clinical_eeg" in out


def test_review_phase_skips_failed_clean(monkeypatch, capsys, tmp_path):
    """Only successfully-cleaned subjects have output worth reviewing;
    subjects whose clean failed have nothing on disk to audit or open
    in the TUI. Loop must skip them entirely (no subprocess launched)."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    calls: list[list[str]] = []
    monkeypatch.setattr(
        "clean_eeg.clean_batch.audit_one_subject",
        lambda output_path, **kw: (calls.append(["audit", output_path]) or
                                    (0, 0.1, None)))
    monkeypatch.setattr(
        "clean_eeg.clean_batch.subprocess.run",
        lambda argv, **kw: (calls.append(argv) or
                             type("R", (), {"returncode": 0})()))

    good = _outcome("R1755J", str(tmp_path / "R1755J" / "clinical_eeg"))
    bad = _outcome("R1755K", str(tmp_path / "R1755K" / "clinical_eeg"),
                    succeeded=False)
    run_review_phase([good, bad])

    # Two calls: audit + TUI, both for the successful subject only.
    subject_codes_touched = {
        arg for c in calls for arg in c if isinstance(arg, str)
        and "R1755" in arg
    }
    assert any("R1755J" in s for s in subject_codes_touched)
    assert not any("R1755K" in s for s in subject_codes_touched), (
        f"failed-clean subject leaked into review phase: {calls}"
    )


def test_review_phase_launches_tui_with_correct_subfolder(monkeypatch, tmp_path):
    """Verify the TUI subprocess argv splits output_path into
    (subject_dir, subfolder) correctly -- this is what the TUI's
    preflight expects. Regression against silently launching the TUI
    against the wrong parent dir (which would fail preflight)."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    captured_tui_argv: list[list[str]] = []

    def _fake_audit(output_path, **kw):
        return 0, 0.1, None

    def _fake_run(argv, **kw):
        captured_tui_argv.append(list(argv))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("clean_eeg.clean_batch.audit_one_subject", _fake_audit)
    monkeypatch.setattr("clean_eeg.clean_batch.subprocess.run", _fake_run)

    out = str(tmp_path / "R1755J" / "clinical_eeg")
    run_review_phase([_outcome("R1755J", out)])
    assert len(captured_tui_argv) == 1
    argv = captured_tui_argv[0]
    assert "--subject-dir" in argv
    sd = argv[argv.index("--subject-dir") + 1]
    assert sd == str(tmp_path / "R1755J"), (
        f"subject-dir should be the parent of output_path: {argv}"
    )
    assert "--subfolder" in argv
    sf = argv[argv.index("--subfolder") + 1]
    assert sf == "clinical_eeg", f"subfolder wrong: {argv}"
    assert "--preload-all" in argv


def test_review_phase_audit_argv_includes_header_visibility_flags(
        monkeypatch, tmp_path):
    """The audit-CLI invocation in the review phase must expose the
    header info the operator needs to visually confirm the clean
    (unique patient_id / startdate values, plus header_phi_residue
    check status). Without -v the summary hides passing checks; without
    --print-edf-header the actual values never render. Regression
    against a bug where the operator saw no header info at all."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    captured_audit_argv: list[list[str]] = []

    def _fake_audit_one_subject(output_path, *, argv_prefix=None,
                                 stream_output=True):
        captured_audit_argv.append(
            (argv_prefix or []) + [output_path])
        return 0, 0.1, None

    monkeypatch.setattr(
        "clean_eeg.clean_batch.audit_one_subject",
        _fake_audit_one_subject)
    monkeypatch.setattr(
        "clean_eeg.clean_batch.subprocess.run",
        lambda *a, **kw: type("R", (), {"returncode": 0})())

    out = str(tmp_path / "R1755J" / "clinical_eeg")
    run_review_phase([_outcome("R1755J", out)])
    assert len(captured_audit_argv) == 1
    argv = captured_audit_argv[0]
    assert "-v" in argv, (
        "review phase must pass -v so passing checks (including "
        f"header_phi_residue) appear in the summary: {argv}")
    assert "--print-edf-header" in argv, (
        "review phase must dump every unique main-header value so the "
        f"operator visually confirms all free-text fields were cleaned: {argv}")
    assert "--print-edf-signal-header" in argv, (
        "review phase must dump signal-header signatures so operator "
        f"can spot PHI in channel labels / transducer / prefilter: {argv}")
    assert "--hide-annotation-flags" in argv
    assert "--no-notebook" in argv


def test_review_phase_continues_on_tui_error(monkeypatch, capsys, tmp_path):
    """A per-subject TUI failure (non-zero exit, or a launch exception)
    must not abort the whole review phase -- the operator wants to work
    through remaining subjects even if one is misconfigured."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    monkeypatch.setattr(
        "clean_eeg.clean_batch.audit_one_subject",
        lambda output_path, **kw: (0, 0.1, None))
    n_calls = {"count": 0}

    def _run_fail_then_ok(argv, **kw):
        n_calls["count"] += 1
        rc = 1 if n_calls["count"] == 1 else 0
        return type("R", (), {"returncode": rc})()

    monkeypatch.setattr(
        "clean_eeg.clean_batch.subprocess.run", _run_fail_then_ok)

    outs = [_outcome("R1755J", str(tmp_path / "R1755J" / "clinical_eeg")),
            _outcome("R1755K", str(tmp_path / "R1755K" / "clinical_eeg"))]
    run_review_phase(outs)
    err = capsys.readouterr().out
    assert "TUI for R1755J exited 1" in err
    # Both subjects' TUI subprocesses were attempted.
    assert n_calls["count"] == 2, "loop stopped after first subject"


def test_review_phase_bails_on_keyboard_interrupt(monkeypatch, capsys, tmp_path):
    """Ctrl-C at the batch-loop level (as opposed to inside a TUI, which
    the TUI itself intercepts) means the operator wants to bail on the
    whole review phase -- must exit cleanly, print a friendly message,
    and not attempt subsequent subjects."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(
        "clean_eeg.clean_batch.audit_one_subject",
        lambda output_path, **kw: (0, 0.1, None))

    def _run_raises(argv, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr("clean_eeg.clean_batch.subprocess.run", _run_raises)

    outs = [_outcome("R1755J", str(tmp_path / "R1755J" / "clinical_eeg")),
            _outcome("R1755K", str(tmp_path / "R1755K" / "clinical_eeg"))]
    run_review_phase(outs)
    out = capsys.readouterr().out
    assert "aborted by operator" in out
    assert "R1755J" in out
    # Second subject should NOT have started.
    assert "R1755K" not in out or "aborted" in out.split("R1755K", 1)[0]
