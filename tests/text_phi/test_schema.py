"""Tests for scripts/text_phi/schema.py."""

from __future__ import annotations

import json

import pytest

from scripts.text_phi.schema import (
    DEFAULT_OPERATIONS_SENTINEL,
    OperationCall,
    Schema,
    SchemaError,
    derive_schema_from_columns,
)


# ---------- helpers ----------

def _min_schema() -> dict:
    return {
        "schema_version": "1",
        "format": "csv",
        "fields": {
            "note": {
                "dtype": "string",
                "description": "free text",
                "operations": "default",
            },
        },
    }


# ---------- happy paths ----------

def test_load_minimal_schema():
    s = Schema.from_dict(_min_schema())
    assert s.schema_version == "1"
    assert s.format == "csv"
    assert s.unknown_field_policy == "error"
    assert s.missing_field_policy == "error"
    assert list(s.fields) == ["note"]
    assert s.fields["note"].dtype == "string"


def test_operations_default_resolves_to_dtype_default():
    s = Schema.from_dict(_min_schema())
    ops = [o.name for o in s.fields["note"].operations]
    assert ops == ["subject_name_scan", "generic_phi_scan"]


def test_operations_explicit_list_replaces_default():
    raw = _min_schema()
    raw["fields"]["note"]["operations"] = ["passthrough"]
    s = Schema.from_dict(raw)
    assert [o.name for o in s.fields["note"].operations] == ["passthrough"]


def test_operations_dict_form_carries_params():
    raw = _min_schema()
    raw["fields"]["note"]["operations"] = [
        {"name": "constant_replace", "params": {"value": "[X]"}},
    ]
    s = Schema.from_dict(raw)
    op = s.fields["note"].operations[0]
    assert op.name == "constant_replace"
    assert op.params == {"value": "[X]"}


def test_from_json_file_roundtrip(tmp_path):
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(_min_schema()))
    s = Schema.load(p)
    assert list(s.fields) == ["note"]


# ---------- validation failures ----------

def test_missing_required_key_raises():
    raw = _min_schema()
    del raw["format"]
    with pytest.raises(SchemaError, match="missing required key 'format'"):
        Schema.from_dict(raw)


def test_wrong_schema_version_raises():
    raw = _min_schema()
    raw["schema_version"] = "2"
    with pytest.raises(SchemaError, match="unsupported schema_version"):
        Schema.from_dict(raw)


def test_null_operations_raises():
    raw = _min_schema()
    raw["fields"]["note"]["operations"] = None
    with pytest.raises(SchemaError, match="operations is null"):
        Schema.from_dict(raw)


def test_unknown_dtype_raises():
    raw = _min_schema()
    raw["fields"]["note"]["dtype"] = "float_matrix"
    with pytest.raises(SchemaError, match="unknown dtype"):
        Schema.from_dict(raw)


def test_unknown_operation_raises():
    raw = _min_schema()
    raw["fields"]["note"]["operations"] = ["quantum_redact"]
    with pytest.raises(SchemaError, match="unknown operation"):
        Schema.from_dict(raw)


def test_dtype_incompatible_operation_raises():
    """date_shift_to_base on a string field is a config error."""
    raw = _min_schema()
    raw["fields"]["note"]["operations"] = ["date_shift_to_base"]
    with pytest.raises(SchemaError, match="not compatible with dtype 'string'"):
        Schema.from_dict(raw)


def test_missing_dep_role_raises():
    raw = {
        "schema_version": "1", "format": "csv",
        "fields": {
            "note_date": {
                "dtype": "date",
                "operations": ["date_shift_relative_to_stay_start"],
                # depends_on is missing entirely
            }
        },
    }
    with pytest.raises(SchemaError, match="requires depends_on role"):
        Schema.from_dict(raw)


def test_dep_target_not_in_schema_raises():
    raw = {
        "schema_version": "1", "format": "csv",
        "fields": {
            "note_date": {
                "dtype": "date",
                "operations": ["date_shift_relative_to_stay_start"],
                "depends_on": {"stay_start_field": "no_such_field"},
            }
        },
    }
    with pytest.raises(SchemaError, match="no such field exists"):
        Schema.from_dict(raw)


def test_dep_cycle_raises():
    raw = {
        "schema_version": "1", "format": "csv",
        "fields": {
            "a": {
                "dtype": "string", "operations": ["passthrough"],
                "depends_on": {"subject_name_field": "b"},
            },
            "b": {
                "dtype": "string", "operations": ["passthrough"],
                "depends_on": {"subject_name_field": "a"},
            },
        },
    }
    with pytest.raises(SchemaError, match="cycle detected"):
        Schema.from_dict(raw)


def test_bad_unknown_field_policy_raises():
    raw = _min_schema()
    raw["unknown_field_policy"] = "whatever"
    with pytest.raises(SchemaError, match="unknown_field_policy"):
        Schema.from_dict(raw)


def test_operations_non_string_non_list_raises():
    raw = _min_schema()
    raw["fields"]["note"]["operations"] = 42
    with pytest.raises(SchemaError, match="operations must be a list"):
        Schema.from_dict(raw)


def test_operation_dict_missing_name_raises():
    raw = _min_schema()
    raw["fields"]["note"]["operations"] = [{"params": {}}]
    with pytest.raises(SchemaError, match="missing 'name'"):
        Schema.from_dict(raw)


# ---------- processing order ----------

def test_processing_order_puts_deps_first():
    raw = {
        "schema_version": "1", "format": "csv",
        "fields": {
            "note_date": {
                "dtype": "date",
                "operations": ["date_shift_relative_to_stay_start"],
                "depends_on": {"stay_start_field": "admission_date"},
            },
            "admission_date": {
                "dtype": "date",
                "operations": ["date_shift_to_base"],
            },
        },
    }
    s = Schema.from_dict(raw)
    order = s.processing_order()
    assert order.index("admission_date") < order.index("note_date")


# ---------- canonical hash ----------

def test_sha256_stable_under_key_reorder():
    raw1 = _min_schema()
    raw2 = {k: raw1[k] for k in reversed(list(raw1.keys()))}
    s1 = Schema.from_dict(raw1)
    s2 = Schema.from_dict(raw2)
    assert s1.sha256() == s2.sha256()


# ---------- derivation ----------

def test_derive_from_columns_marks_name_and_date():
    s = derive_schema_from_columns(
        ["patient_name", "note_text", "admission_date", "channel"], "csv",
    )
    assert s.fields["patient_name"].dtype == "subject_name"
    assert s.fields["admission_date"].dtype == "date"
    assert s.fields["note_text"].dtype == "string"
    assert s.fields["channel"].dtype == "string"
    # All descriptions carry TODO markers.
    for fs in s.fields.values():
        assert "TODO" in fs.description


def test_derive_uses_default_operations():
    s = derive_schema_from_columns(["note_text"], "csv")
    ops = [o.name for o in s.fields["note_text"].operations]
    assert ops == ["subject_name_scan", "generic_phi_scan"]
