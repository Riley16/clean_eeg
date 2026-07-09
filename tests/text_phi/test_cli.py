"""Tests for scripts/text_phi/cli.py (subcommand model)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.text_phi.cli import main
from scripts.text_phi.schema import Schema


# ---------- schema derive ----------

def test_schema_derive_csv(tmp_path):
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "notes.schema.json"
    pd.DataFrame([{"patient_name": "x", "note_text": "y", "channel": "z",
                   "admission_date": "2024-01-01"}]).to_csv(in_path, index=False)
    rc = main([
        "schema", "derive",
        "--input", str(in_path), "--output", str(out_path),
    ])
    assert rc == 0
    schema = Schema.load(out_path)
    assert schema.fields["patient_name"].dtype == "subject_name"
    assert schema.fields["admission_date"].dtype == "date"
    assert schema.fields["note_text"].dtype == "string"
    assert schema.fields["channel"].dtype == "string"


def test_schema_derive_txt(tmp_path):
    in_path = tmp_path / "in.txt"
    out_path = tmp_path / "notes.schema.json"
    in_path.write_text("hi\n")
    rc = main([
        "schema", "derive",
        "--input", str(in_path), "--output", str(out_path),
    ])
    assert rc == 0
    schema = Schema.load(out_path)
    assert list(schema.fields.keys()) == ["text"]


def test_schema_derive_unsupported_extension(tmp_path, capsys):
    in_path = tmp_path / "in.docx"
    in_path.write_text("x")
    rc = main([
        "schema", "derive",
        "--input", str(in_path), "--output", str(tmp_path / "s.json"),
    ])
    assert rc == 2
    assert "Unsupported" in capsys.readouterr().err


# ---------- redact ----------

def _basic_csv_schema(tmp_path) -> str:
    p = tmp_path / "s.json"
    Schema.from_dict({
        "schema_version": "1", "format": "csv",
        "fields": {
            "note": {"dtype": "string", "operations": "default"},
            "channel": {"dtype": "string", "operations": ["passthrough"]},
        },
    })  # validates the shape
    p.write_text(json.dumps({
        "schema_version": "1", "format": "csv",
        "fields": {
            "note": {"dtype": "string", "operations": "default"},
            "channel": {"dtype": "string", "operations": ["passthrough"]},
        },
    }))
    return str(p)


def test_redact_csv_generic_mode(tmp_path):
    schema_path = _basic_csv_schema(tmp_path)
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    pd.DataFrame([
        {"note": "email user@example.org", "channel": "LFPx1"},
    ]).to_csv(in_path, index=False)
    rc = main([
        "redact",
        "--input", str(in_path), "--output", str(out_path),
        "--schema", schema_path,
        "--mode", "generic",
    ])
    assert rc == 0
    out = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert "user@example.org" not in out.at[0, "note"]
    assert out.at[0, "channel"] == "LFPx1"


def test_redact_csv_subject_mode_with_flags(tmp_path):
    schema_path = _basic_csv_schema(tmp_path)
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    pd.DataFrame([
        {"note": "Dr. John O'Connor was here.", "channel": "LFPx1"},
    ]).to_csv(in_path, index=False)
    rc = main([
        "redact",
        "--input", str(in_path), "--output", str(out_path),
        "--schema", schema_path,
        "--mode", "subject",
        "--subject-first", "John", "--subject-last", "O'Connor",
    ])
    assert rc == 0
    out = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert "John" not in out.at[0, "note"]
    assert "O'Connor" not in out.at[0, "note"]
    assert out.at[0, "channel"] == "LFPx1"


def test_redact_txt_uses_default_schema(tmp_path):
    in_path = tmp_path / "in.txt"
    out_path = tmp_path / "out.txt"
    in_path.write_text("Contact user@ex.com for details.\n")
    rc = main([
        "redact",
        "--input", str(in_path), "--output", str(out_path),
        "--mode", "generic",
    ])
    assert rc == 0
    assert "user@ex.com" not in out_path.read_text()


def test_redact_csv_requires_schema(tmp_path, capsys):
    in_path = tmp_path / "in.csv"
    pd.DataFrame([{"a": "1"}]).to_csv(in_path, index=False)
    rc = main([
        "redact",
        "--input", str(in_path), "--output", str(tmp_path / "out.csv"),
        "--mode", "generic",
    ])
    assert rc == 2
    assert "schema is required" in capsys.readouterr().err


def test_redact_unsupported_extension(tmp_path, capsys):
    in_path = tmp_path / "in.docx"
    in_path.write_text("x")
    rc = main([
        "redact",
        "--input", str(in_path), "--output", str(tmp_path / "out.docx"),
        "--mode", "generic",
    ])
    assert rc == 2
    assert "Unsupported" in capsys.readouterr().err


def test_redact_audit_out(tmp_path):
    schema_path = _basic_csv_schema(tmp_path)
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    audit_path = tmp_path / "audit.json"
    pd.DataFrame([
        {"note": "email js@ex.com", "channel": "LFPx1"},
    ]).to_csv(in_path, index=False)
    rc = main([
        "redact",
        "--input", str(in_path), "--output", str(out_path),
        "--schema", schema_path,
        "--mode", "generic",
        "--audit-out", str(audit_path),
    ])
    assert rc == 0
    audit = json.loads(audit_path.read_text())
    assert audit["n_entries"] >= 1
    assert audit["schema_sha256"] is not None


def test_redact_csv_with_name_column_and_date_flow(tmp_path):
    """Full schema-driven flow: name field feeds subject-name scan on notes,
    admission_date anchors a note_date shift."""
    schema_path = tmp_path / "s.json"
    schema_path.write_text(json.dumps({
        "schema_version": "1", "format": "csv",
        "fields": {
            "patient_name": {
                "dtype": "subject_name",
                "operations": [
                    "parse_subject_name",
                    {"name": "constant_replace", "params": {"value": "[NAME]"}},
                ],
            },
            "note_text": {
                "dtype": "string", "operations": ["subject_name_scan"],
                "depends_on": {"subject_name_field": "patient_name"},
            },
            "admission_date": {
                "dtype": "date", "operations": ["date_shift_to_base"],
            },
            "note_date": {
                "dtype": "date",
                "operations": ["date_shift_relative_to_stay_start"],
                "depends_on": {"stay_start_field": "admission_date"},
            },
            "channel": {"dtype": "string", "operations": ["passthrough"]},
        },
    }))
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    pd.DataFrame([{
        "patient_name": "John O'Connor",
        "note_text": "Dr. John O'Connor visited today.",
        "admission_date": "2024-01-10",
        "note_date": "2024-01-15",
        "channel": "LFPx1",
    }]).to_csv(in_path, index=False)

    rc = main([
        "redact",
        "--input", str(in_path), "--output", str(out_path),
        "--schema", str(schema_path),
        "--mode", "subject",
    ])
    assert rc == 0
    out = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert out.at[0, "patient_name"] == "[NAME]"
    assert "John" not in out.at[0, "note_text"]
    assert "O'Connor" not in out.at[0, "note_text"]
    assert out.at[0, "admission_date"] == "1985-01-01"
    assert out.at[0, "note_date"] == "1985-01-06"  # 5-day interval preserved
    assert out.at[0, "channel"] == "LFPx1"
