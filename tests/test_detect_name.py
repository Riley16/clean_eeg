"""Tests for header-based subject-name detection (clean_eeg.detect_name).

EDF files here are fabricated as bare 256-byte main headers — detection
reads header bytes directly (never pyedflib), so no signal data is needed
and the tests stay fast.
"""

import csv
import os
import stat

import pytest

from clean_eeg.detect_name import (
    NAME_CSV_COLUMNS,
    STATUS_AMBIGUOUS,
    STATUS_DISAGREE,
    STATUS_INITIAL_ONLY,
    STATUS_NO_FILES,
    STATUS_OK,
    STATUS_PLACEHOLDER,
    detect_subject_name,
    extract_name_field,
    extract_patient_code,
    is_placeholder_name,
    list_edf_files,
    parse_patient_name,
    write_name_csv_row,
)


def write_fake_edf(path, patient_id: str):
    """Write a minimal EDF main header carrying ``patient_id`` (bytes 8-87)."""
    header = bytearray(b" " * 256)
    header[0:8] = b"0".ljust(8)
    header[8:88] = patient_id.encode("ascii", "replace").ljust(80)[:80]
    header[88:168] = b"Startdate 01-JAN-1985 X X X".ljust(80)
    header[168:176] = b"01.01.85"
    header[176:184] = b"10.00.00"
    header[184:192] = b"256".ljust(8)
    header[236:244] = b"0".ljust(8)
    header[244:252] = b"1".ljust(8)
    header[252:256] = b"0".ljust(4)
    with open(path, "wb") as f:
        f.write(bytes(header))
    return str(path)


# ---------------------------------------------------------------------------
# parse_patient_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,first,middles,last", [
    ("SMITH^JOHN^P", "John", ["P"], "Smith"),          # HL7 PID: Last^First^Middle
    ("Smith, John P", "John", ["P"], "Smith"),          # Last, First Middle
    ("John_Paul_Smith", "John", ["Paul"], "Smith"),     # EDF+ underscore form
    ("John Smith", "John", [], "Smith"),                # First Last
    ("MCDONALD^JOHN", "John", [], "Mcdonald"),          # all-caps normalized
    ("O'BRIEN^Mary^A", "Mary", ["A"], "O'Brien"),       # apostrophe preserved
    ("DOE^JANE^MARIE^^MS", "Jane", ["Marie"], "Doe"),   # title dropped
    ("Smith-Jones, Ann", "Ann", [], "Smith-Jones"),     # hyphen preserved
])
def test_parse_patient_name_formats(raw, first, middles, last):
    name, status = parse_patient_name(raw)
    assert status == STATUS_OK
    assert (name.first_name, name.middle_names, name.last_name) == (first, middles, last)


@pytest.mark.parametrize("raw,status", [
    ("Smith", STATUS_AMBIGUOUS),        # one token — can't split first/last
    ("L. Smith", STATUS_INITIAL_ONLY),  # first name is a bare initial
    ("Smith^J", STATUS_INITIAL_ONLY),
    ("X", STATUS_PLACEHOLDER),
    ("  X  X ", STATUS_PLACEHOLDER),
    ("", STATUS_PLACEHOLDER),
    ("R1755A", STATUS_PLACEHOLDER),     # already de-identified
    ("anonymized", STATUS_PLACEHOLDER),
])
def test_parse_patient_name_declines(raw, status):
    name, got = parse_patient_name(raw)
    assert name is None
    assert got == status


def test_is_placeholder_name():
    assert is_placeholder_name("X")
    assert is_placeholder_name("")
    assert not is_placeholder_name("John Smith")


# ---------------------------------------------------------------------------
# patient_id field extraction
# ---------------------------------------------------------------------------

def test_extract_name_field_edf_plus_layout():
    # EDF+: code sex birthdate name
    assert extract_name_field("1234567 M 09-APR-1955 John_Paul_Smith") == "John_Paul_Smith"
    assert extract_patient_code("1234567 M 09-APR-1955 John_Paul_Smith") == "1234567"


def test_extract_name_field_non_conforming():
    # Not EDF+ shaped — treat the whole field as the name.
    assert extract_name_field("SMITH^JOHN") == "SMITH^JOHN"


# ---------------------------------------------------------------------------
# detect_subject_name
# ---------------------------------------------------------------------------

def test_detect_subject_name_agreeing_files(tmp_path):
    for i in range(3):
        write_fake_edf(tmp_path / f"rec_{i}.edf", "1234567 M 09-APR-1955 John_Paul_Smith")
    result = detect_subject_name(str(tmp_path))
    assert result.ok
    assert result.status == STATUS_OK
    assert (result.name.first_name, result.name.last_name) == ("John", "Smith")
    assert result.name.middle_names == ["Paul"]
    assert result.n_files == 3
    assert result.patient_code == "1234567"


def test_detect_subject_name_disagreeing_files(tmp_path):
    write_fake_edf(tmp_path / "a.edf", "1234567 M 09-APR-1955 John_Smith")
    write_fake_edf(tmp_path / "b.edf", "7654321 F 09-APR-1960 Jane_Doe")
    result = detect_subject_name(str(tmp_path))
    assert not result.ok
    assert result.status == STATUS_DISAGREE


def test_detect_subject_name_ignores_placeholder_files(tmp_path):
    """A partially de-identified directory still detects from the real names."""
    write_fake_edf(tmp_path / "a.edf", "X X X X")
    write_fake_edf(tmp_path / "b.edf", "1234567 M 09-APR-1955 John_Smith")
    result = detect_subject_name(str(tmp_path))
    assert result.ok
    assert result.name.last_name == "Smith"


def test_detect_subject_name_all_placeholders(tmp_path):
    write_fake_edf(tmp_path / "a.edf", "X X X X")
    result = detect_subject_name(str(tmp_path))
    assert result.status == STATUS_PLACEHOLDER
    assert not result.ok


def test_detect_subject_name_no_edf_files(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing here")
    result = detect_subject_name(str(tmp_path))
    assert result.status == STATUS_NO_FILES


def test_annotation_stubs_are_skipped(tmp_path):
    """Stubs written by a previous in-place run hold the subject code, not a
    name — reading them would produce a spurious disagreement."""
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 John_Smith")
    write_fake_edf(tmp_path / "rec_annotations.edf", "R1755A X X R1755A")
    assert len(list_edf_files(str(tmp_path))) == 1
    result = detect_subject_name(str(tmp_path))
    assert result.ok
    assert result.n_files == 1


def test_detect_subject_name_single_file_path(tmp_path):
    path = write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 John_Smith")
    result = detect_subject_name(path)
    assert result.ok


def test_detect_subject_name_finds_nested_edfs(tmp_path):
    """EDFs buried in subdirectories are found and aggregated as one subject."""
    (tmp_path / "day1").mkdir()
    (tmp_path / "day2" / "session_a").mkdir(parents=True)
    write_fake_edf(tmp_path / "day1" / "a.edf", "1234567 M 09-APR-1955 John_Smith")
    write_fake_edf(tmp_path / "day2" / "session_a" / "b.edf",
                   "1234567 M 09-APR-1955 John_Smith")
    result = detect_subject_name(str(tmp_path))
    assert result.ok
    assert result.n_files == 2
    assert result.name.last_name == "Smith"


def test_list_edf_files_non_recursive(tmp_path):
    (tmp_path / "nested").mkdir()
    write_fake_edf(tmp_path / "top.edf", "1 M 09-APR-1955 John_Smith")
    write_fake_edf(tmp_path / "nested" / "deep.edf", "1 M 09-APR-1955 John_Smith")
    assert len(list_edf_files(str(tmp_path), recursive=False)) == 1
    assert len(list_edf_files(str(tmp_path), recursive=True)) == 2


def test_name_tokens_include_raw_header_text(tmp_path):
    """The tokens registered as PHI must cover the raw header form too, so
    the un-parsed string never survives in log.out."""
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 SMITH^JOHN")
    result = detect_subject_name(str(tmp_path))
    tokens = {t.lower() for t in result.name_tokens()}
    assert "john" in tokens and "smith" in tokens


# ---------------------------------------------------------------------------
# ID -> name CSV
# ---------------------------------------------------------------------------

def test_write_name_csv_row_creates_locked_down_file(tmp_path):
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 John_Smith")
    result = detect_subject_name(str(tmp_path))
    csv_path = str(tmp_path / "sens" / "detected_names.csv")

    write_name_csv_row(result, subject_code="R1755A", csv_path=csv_path)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert list(row.keys()) == NAME_CSV_COLUMNS
    assert row["subject_code"] == "R1755A"
    assert row["patient_code"] == "1234567"
    assert (row["detected_first"], row["detected_last"]) == ("John", "Smith")
    assert row["parse_status"] == STATUS_OK
    assert row["n_edf_files"] == "1"

    mode = stat.S_IMODE(os.stat(csv_path).st_mode)
    assert mode == 0o600
    dir_mode = stat.S_IMODE(os.stat(os.path.dirname(csv_path)).st_mode)
    assert dir_mode == 0o700


def test_write_name_csv_row_updates_in_place(tmp_path):
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 John_Smith")
    result = detect_subject_name(str(tmp_path))
    csv_path = str(tmp_path / "detected_names.csv")

    write_name_csv_row(result, subject_code="R1755A", csv_path=csv_path)
    write_name_csv_row(result, subject_code="R1755B", csv_path=csv_path)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, "re-scanning the same directory must not duplicate rows"
    assert rows[0]["subject_code"] == "R1755B"


def test_write_name_csv_row_appends_other_subjects(tmp_path):
    dir_a = tmp_path / "subj_a"
    dir_b = tmp_path / "subj_b"
    dir_a.mkdir()
    dir_b.mkdir()
    write_fake_edf(dir_a / "rec.edf", "1 M 09-APR-1955 John_Smith")
    write_fake_edf(dir_b / "rec.edf", "2 F 09-APR-1960 Jane_Doe")
    csv_path = str(tmp_path / "detected_names.csv")

    write_name_csv_row(detect_subject_name(str(dir_a)), csv_path=csv_path)
    write_name_csv_row(detect_subject_name(str(dir_b)), csv_path=csv_path)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert [r["detected_last"] for r in rows] == ["Smith", "Doe"]


def test_write_name_csv_row_records_declines(tmp_path):
    """Declined detections still get a row — the review pass needs to see
    which subjects had no usable header name."""
    write_fake_edf(tmp_path / "rec.edf", "X X X X")
    result = detect_subject_name(str(tmp_path))
    csv_path = str(tmp_path / "detected_names.csv")
    write_name_csv_row(result, csv_path=csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["parse_status"] == STATUS_PLACEHOLDER
    assert rows[0]["detected_first"] == ""


# ---------------------------------------------------------------------------
# CLI wiring in clean_subject_eeg
# ---------------------------------------------------------------------------

def _run_cli(monkeypatch, argv):
    import sys
    from clean_eeg.clean_subject_eeg import get_clean_eeg_cli_arguments
    monkeypatch.setattr(sys, "argv", ["clean_subject_eeg"] + argv)
    return get_clean_eeg_cli_arguments()


def test_auto_name_fills_name_from_headers(tmp_path, monkeypatch):
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 John_Paul_Smith")
    args = _run_cli(monkeypatch, [
        "--input_path", str(tmp_path), "--subject_code", "R1755A", "--auto_name",
    ])
    assert (args.first_name, args.middle_name, args.last_name) == ("John", "Paul", "Smith")


def test_auto_name_explicit_args_win(tmp_path, monkeypatch):
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 John_Smith")
    args = _run_cli(monkeypatch, [
        "--input_path", str(tmp_path), "--subject_code", "R1755A", "--auto_name",
        "--first_name", "Jonathan", "--no_middle_name",
    ])
    assert (args.first_name, args.last_name) == ("Jonathan", "Smith")
    assert args.middle_name == ""


def test_auto_name_aborts_when_header_name_unusable(tmp_path, monkeypatch):
    # "L. Smith" — first name is only an initial, which gives the redactor
    # nothing to match on.
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 L._Smith")
    with pytest.raises(ValueError, match="auto_name"):
        _run_cli(monkeypatch, [
            "--input_path", str(tmp_path), "--subject_code", "R1755A", "--auto_name",
        ])


def test_interactive_prompt_defaults_to_detected_name(tmp_path, monkeypatch):
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 John_Paul_Smith")
    # Operator presses Enter at every prompt -> detected values accepted.
    monkeypatch.setattr("clean_eeg.clean_subject_eeg.logged_input", lambda prompt="": "")
    args = _run_cli(monkeypatch, ["--input_path", str(tmp_path),
                                  "--subject_code", "R1755A"])
    assert (args.first_name, args.middle_name, args.last_name) == ("John", "Paul", "Smith")


def test_interactive_typed_name_overrides_detected(tmp_path, monkeypatch):
    write_fake_edf(tmp_path / "rec.edf", "1234567 M 09-APR-1955 John_Smith")
    responses = iter(["Jonathan", "Smythe", ""])
    monkeypatch.setattr("clean_eeg.clean_subject_eeg.logged_input",
                        lambda prompt="": next(responses))
    args = _run_cli(monkeypatch, ["--input_path", str(tmp_path),
                                  "--subject_code", "R1755A"])
    assert (args.first_name, args.last_name) == ("Jonathan", "Smythe")


def test_detect_edf_names_cli_recursive_single_subject(tmp_path, capsys):
    from clean_eeg.detect_name import main
    import sys
    (tmp_path / "day1").mkdir()
    write_fake_edf(tmp_path / "day1" / "a.edf", "1234567 M 09-APR-1955 John_Smith")
    csv_path = str(tmp_path / "sens" / "names.csv")
    sys.argv = ["detect-edf-names", str(tmp_path), "--subject_code", "R1001P",
                "--csv", csv_path]
    assert main() == 0
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["detected_last"] == "Smith"


def test_detect_edf_names_cli_per_subject_dir(tmp_path, capsys):
    from clean_eeg.detect_name import main
    import sys
    for subj, name in (("R1001P", "John_Smith"), ("R1002P", "Jane_Doe")):
        d = tmp_path / subj / "day1"
        d.mkdir(parents=True)
        write_fake_edf(d / "a.edf", f"1 M 09-APR-1955 {name}")
    csv_path = str(tmp_path / "names.csv")
    sys.argv = ["detect-edf-names", str(tmp_path), "--per_subject_dir",
                "--csv", csv_path]
    assert main() == 0
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert sorted(r["detected_last"] for r in rows) == ["Doe", "Smith"]


def test_name_csv_flag_writes_row(tmp_path, monkeypatch):
    edf_dir = tmp_path / "edfs"
    edf_dir.mkdir()
    write_fake_edf(edf_dir / "rec.edf", "1234567 M 09-APR-1955 John_Smith")
    csv_path = str(tmp_path / "sens" / "detected_names.csv")
    _run_cli(monkeypatch, [
        "--input_path", str(edf_dir), "--subject_code", "R1755A", "--auto_name",
        "--name_csv", csv_path,
    ])
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["detected_last"] == "Smith"
