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
    SubjectRow,
    _default_clean_argv_prefix,
    audit_one_subject,
    clean_one_subject,
    filter_rows_by_subject,
    main,
    parse_subjects_csv,
    run_batch,
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
