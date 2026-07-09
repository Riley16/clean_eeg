"""Tests for scripts/text_phi/formats/csv.py (records API)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.text_phi.formats import CsvFormat
from scripts.text_phi.schema import Schema


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _schema(fields: dict, **kwargs) -> Schema:
    raw = {"schema_version": "1", "format": "csv", "fields": fields, **kwargs}
    return Schema.from_dict(raw)


def _basic_schema() -> Schema:
    return _schema({
        "note": {"dtype": "string", "operations": ["passthrough"]},
        "channel": {"dtype": "string", "operations": ["passthrough"]},
    })


# ---------- load ----------

def test_load_returns_one_record_per_row(tmp_path):
    p = tmp_path / "in.csv"
    _write_csv(p, [
        {"note": "hello", "channel": "LFPx1"},
        {"note": "world", "channel": "LFPx2"},
    ])
    recs = CsvFormat().load(p, _basic_schema())
    assert len(recs) == 2
    assert recs[0].fields == {"note": "hello", "channel": "LFPx1"}
    assert recs[0].location == {"row": 0}


def test_load_requires_schema(tmp_path):
    p = tmp_path / "in.csv"
    _write_csv(p, [{"a": "1"}])
    with pytest.raises(ValueError, match="schema is required"):
        CsvFormat().load(p, None)


def test_load_unknown_field_policy_error(tmp_path):
    p = tmp_path / "in.csv"
    _write_csv(p, [{"note": "x", "extra": "y"}])
    with pytest.raises(ValueError, match="columns not in schema"):
        CsvFormat().load(p, _basic_schema())


def test_load_unknown_field_policy_allow_unknown(tmp_path):
    p = tmp_path / "in.csv"
    _write_csv(p, [{"note": "x", "extra": "y", "channel": "LFPx1"}])
    recs = CsvFormat().load(p, _basic_schema(), allow_unknown=True)
    # Unknown field passes through in the record.
    assert recs[0].fields["extra"] == "y"


def test_load_missing_field_policy_error(tmp_path):
    p = tmp_path / "in.csv"
    _write_csv(p, [{"note": "x"}])  # 'channel' missing
    with pytest.raises(ValueError, match="fields missing from CSV"):
        CsvFormat().load(p, _basic_schema())


def test_load_missing_field_policy_ignore(tmp_path):
    schema = _schema(
        {
            "note": {"dtype": "string", "operations": ["passthrough"]},
            "channel": {"dtype": "string", "operations": ["passthrough"]},
        },
        missing_field_policy="ignore",
    )
    p = tmp_path / "in.csv"
    _write_csv(p, [{"note": "x"}])
    recs = CsvFormat().load(p, schema)
    assert "channel" not in recs[0].fields


# ---------- eager dtype coercion ----------

def test_load_dtype_error_by_default_raises(tmp_path):
    schema = _schema({
        "age": {"dtype": "integer", "operations": ["passthrough"]},
    })
    p = tmp_path / "in.csv"
    _write_csv(p, [{"age": "not_a_number"}])
    with pytest.raises(ValueError, match="not an integer"):
        CsvFormat().load(p, schema)


def test_load_dtype_error_allow_skips_row(tmp_path, caplog):
    schema = _schema({
        "age": {"dtype": "integer", "operations": ["passthrough"]},
    })
    p = tmp_path / "in.csv"
    _write_csv(p, [{"age": "not_a_number"}, {"age": "42"}])
    with caplog.at_level("WARNING"):
        recs = CsvFormat().load(p, schema, allow_parse_errors=True)
    assert [r.fields["age"] for r in recs] == ["42"]
    assert any("skipping row" in rec.message for rec in caplog.records)


def test_load_all_dtypes_valid(tmp_path):
    schema = _schema({
        "d": {"dtype": "date", "operations": ["passthrough"]},
        "n": {"dtype": "float", "operations": ["passthrough"]},
        "b": {"dtype": "boolean", "operations": ["passthrough"]},
    })
    p = tmp_path / "in.csv"
    _write_csv(p, [{"d": "2024-01-15", "n": "3.14", "b": "true"}])
    recs = CsvFormat().load(p, schema)
    assert recs[0].fields == {"d": "2024-01-15", "n": "3.14", "b": "true"}


# ---------- save ----------

def test_save_roundtrip_preserves_values(tmp_path):
    schema = _basic_schema()
    fmt = CsvFormat()
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    _write_csv(in_path, [
        {"note": "abc", "channel": "LFPx1"},
        {"note": "def", "channel": "LFPx2"},
    ])
    recs = fmt.load(in_path, schema)
    fmt.save(out_path, recs, schema)
    reread = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert reread.to_dict(orient="records") == [
        {"note": "abc", "channel": "LFPx1"},
        {"note": "def", "channel": "LFPx2"},
    ]


def test_save_empty_records_emits_header_only(tmp_path):
    fmt = CsvFormat()
    schema = _basic_schema()
    out_path = tmp_path / "out.csv"
    fmt.save(out_path, [], schema)
    assert out_path.read_text().strip() == "note,channel"


# ---------- validation ----------

def test_validate_output_row_count(tmp_path):
    fmt = CsvFormat()
    schema = _basic_schema()
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    _write_csv(in_path, [{"note": "x", "channel": "y"}])
    _write_csv(out_path, [{"note": "x", "channel": "y"},
                          {"note": "z", "channel": "w"}])
    with pytest.raises(IOError, match="row count changed"):
        fmt.validate_output(in_path, out_path, schema, n_records=1)


def test_validate_output_missing_file(tmp_path):
    fmt = CsvFormat()
    schema = _basic_schema()
    with pytest.raises(IOError, match="was not written"):
        fmt.validate_output(tmp_path / "in.csv", tmp_path / "nope.csv",
                            schema, n_records=0)
