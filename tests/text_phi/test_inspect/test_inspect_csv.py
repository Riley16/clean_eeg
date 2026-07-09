"""Tests for scripts/text_phi/inspect/inspect_csv.py."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.text_phi.inspect.inspect_csv import (
    _cardinality_class,
    _phi_hint_from_name,
    _redcap_to_our_dtype,
    _suggest_operations,
    analyze_column,
    inspect,
    value_shape,
)


# ---------- value_shape ----------

def test_shape_ascii_letters():
    assert value_shape("John") == "A{4}"


def test_shape_letters_and_digits_disambiguated():
    # Adjacent runs must not collide into a single number.
    assert value_shape("Test1") == "A{4}9{1}"
    assert value_shape("1Test") == "9{1}A{4}"


def test_shape_preserves_punctuation():
    assert value_shape("2024-01-15") == "9{4}-9{2}-9{2}"
    assert value_shape("js@ex.com") == "A{2}@A{2}.A{3}"


def test_shape_spaces_collapsed_to_single():
    assert value_shape("Fake Test") == "A{4} A{4}"


def test_shape_empty():
    assert value_shape("") == ""


# ---------- _redcap_to_our_dtype ----------

def test_dtype_subject_name():
    it = {"data_type": "text", "field_type": "text",
          "identifier": True, "question": "Subject Name"}
    assert _redcap_to_our_dtype(it) == "subject_name"


def test_dtype_identifier_but_no_name_keyword_falls_through():
    """An identifier field whose question isn't name-shaped shouldn't be
    mislabeled as subject_name — e.g. MRN or Consent Form ID."""
    it = {"data_type": "text", "field_type": "text",
          "identifier": True, "question": "Consent Form"}
    # Should NOT be subject_name; but should still be treated as PHI later
    # via the identifier flag.
    assert _redcap_to_our_dtype(it) == "string"


def test_dtype_from_validation_type():
    assert _redcap_to_our_dtype(
        {"data_type": "text", "field_type": "text",
         "validation_type": "email", "question": "Email"}
    ) == "email"
    assert _redcap_to_our_dtype(
        {"data_type": "text", "field_type": "text",
         "validation_type": "date_mdy", "question": "DOB"}
    ) == "date"
    assert _redcap_to_our_dtype(
        {"data_type": "text", "field_type": "text",
         "validation_type": "datetime_ymd", "question": "T"}
    ) == "datetime"
    assert _redcap_to_our_dtype(
        {"data_type": "text", "field_type": "text",
         "validation_type": "zipcode", "question": "Z"}
    ) == "zip_code"
    assert _redcap_to_our_dtype(
        {"data_type": "text", "field_type": "text",
         "validation_type": "int", "question": "N"}
    ) == "integer"


def test_dtype_from_field_type():
    assert _redcap_to_our_dtype(
        {"data_type": "text", "field_type": "radio", "question": "X"}
    ) == "enum"
    assert _redcap_to_our_dtype(
        {"data_type": "text", "field_type": "dropdown", "question": "X"}
    ) == "enum"
    assert _redcap_to_our_dtype(
        {"data_type": "boolean", "field_type": "checkbox", "question": "X"}
    ) == "boolean"


def test_dtype_from_data_type():
    assert _redcap_to_our_dtype(
        {"data_type": "integer", "field_type": "text", "question": "X"}
    ) == "integer"
    assert _redcap_to_our_dtype(
        {"data_type": "float", "field_type": "calc", "question": "Age"}
    ) == "float"
    assert _redcap_to_our_dtype(
        {"data_type": "date", "field_type": "text", "question": "X"}
    ) == "date"


def test_dtype_falls_through_to_string():
    assert _redcap_to_our_dtype(
        {"data_type": "text", "field_type": "textarea", "question": "Notes"}
    ) == "string"


def test_dtype_none_item_defaults_string():
    assert _redcap_to_our_dtype(None) == "string"


# ---------- _suggest_operations ----------

def test_ops_subject_name():
    assert _suggest_operations("subject_name", {}) == \
        ["parse_subject_name", "constant_replace"]


def test_ops_date_shift_to_base():
    assert _suggest_operations("date", {}) == ["date_shift_to_base"]


def test_ops_typed_phi():
    assert _suggest_operations("email", {}) == ["email_redact"]
    assert _suggest_operations("phone", {}) == ["phone_redact"]
    assert _suggest_operations("zip_code", {}) == ["zip_redact"]
    assert _suggest_operations("mrn", {}) == ["mrn_redact"]


def test_ops_identifier_string_becomes_constant_replace():
    it = {"identifier": True, "field_type": "text"}
    assert _suggest_operations("string", it) == ["constant_replace"]


def test_ops_file_field_becomes_constant_replace():
    it = {"identifier": False, "field_type": "file"}
    assert _suggest_operations("string", it) == ["constant_replace"]


def test_ops_notes_becomes_generic_scan():
    it = {"identifier": False, "field_type": "notes"}
    assert _suggest_operations("string", it) == \
        ["subject_name_scan", "generic_phi_scan"]


def test_ops_enum_passthrough():
    assert _suggest_operations("enum", {}) == ["passthrough"]


def test_ops_boolean_passthrough():
    assert _suggest_operations("boolean", {}) == ["passthrough"]


# ---------- _phi_hint_from_name ----------

@pytest.mark.parametrize("name", ["Subject Name", "Patient Name",
                                  "Emergency Contact Name"])
def test_phi_hint_name(name):
    assert _phi_hint_from_name(name) == "column name mentions 'name'"


@pytest.mark.parametrize("name", ["Date of Birth", "DOB", "birthday"])
def test_phi_hint_birth(name):
    assert _phi_hint_from_name(name) is not None
    assert "birth" in _phi_hint_from_name(name)


def test_phi_hint_free_text():
    assert _phi_hint_from_name("Additional Notes") == "free-text label"
    assert _phi_hint_from_name("Comments") == "free-text label"


def test_phi_hint_none_for_neutral():
    assert _phi_hint_from_name("Subject Number") is None
    assert _phi_hint_from_name("Study site") is None


# ---------- cardinality ----------

@pytest.mark.parametrize("n_unique,n_filled,expected", [
    (0, 0, "empty"),
    (1, 10, "constant"),
    (5, 100, "very_low"),
    (25, 100, "low"),
    (100, 100, "unique"),
    (50, 500, "medium"),
    (300, 500, "high"),
])
def test_cardinality_class(n_unique, n_filled, expected):
    assert _cardinality_class(n_unique, n_filled) == expected


# ---------- analyze_column ----------

def test_analyze_column_no_sample_values_ever(tmp_path):
    df = pd.DataFrame({"Gender": ["Male", "Female", "Nonbinary"]})
    xml_item = {"oid": "gender", "identifier": False,
                "field_type": "radio", "data_type": "text",
                "question": "Gender", "form": "f1"}
    result = analyze_column(df, "Gender", xml_item, shape_min_freq=1)
    # Raw values never emitted. Enum levels come from xml.enum_choices.
    assert "sample_values" not in result


def test_analyze_column_identifier_flagged(tmp_path):
    df = pd.DataFrame({"Subject Name": ["Alice", "Bob", "Cara"]})
    xml_item = {"oid": "subject_name", "identifier": True,
                "field_type": "text", "data_type": "text",
                "question": "Subject Name", "form": "f1"}
    result = analyze_column(df, "Subject Name", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "subject_name"
    assert result["suggested_operations"] == ["parse_subject_name",
                                              "constant_replace"]
    assert "REDCap Identifier" in result["phi_hint"]


def test_analyze_column_reports_dtype_parseability(tmp_path):
    df = pd.DataFrame({"Subject Number": ["1001", "1002", "1003", "abc"]})
    xml_item = {"oid": "subject_number", "identifier": False,
                "field_type": "text", "data_type": "integer",
                "validation_type": "int",
                "question": "Subject Number", "form": "f1"}
    result = analyze_column(df, "Subject Number", xml_item, shape_min_freq=1)
    assert result["empirical_dtype_parse_rates"]["integer"]["rate"] == 0.75


def test_analyze_column_shape_min_freq_filter(tmp_path):
    df = pd.DataFrame({"c": ["abc", "abc", "xyzq"]})
    result = analyze_column(df, "c", None, shape_min_freq=2)
    shapes = {s["shape"] for s in result["top_value_shapes"]}
    assert "A{3}" in shapes
    assert "A{4}" not in shapes


def test_analyze_column_free_text_flag_triggers_string_scan(tmp_path):
    # Many long, structurally-varied values → free text.
    values = [
        "This is a long note about the patient with variable content " + str(i)
        for i in range(20)
    ]
    df = pd.DataFrame({"note": values})
    xml_item = {"oid": "note", "identifier": False,
                "field_type": "textarea", "data_type": "text",
                "question": "Clinical Note", "form": "f1"}
    result = analyze_column(df, "note", xml_item, shape_min_freq=1)
    assert result["likely_free_text"] is True
    assert result["suggested_operations"] == \
        ["subject_name_scan", "generic_phi_scan"]


def test_free_text_override_skips_file_field(tmp_path):
    """A REDCap `file` field stores attachment IDs, not prose. Free-text
    heuristic should NOT downgrade the constant_replace suggestion."""
    values = ["file_id_" + str(i) * 10 for i in range(20)]  # long, varied
    df = pd.DataFrame({"consent_form": values})
    xml_item = {"oid": "consent_form", "identifier": True,
                "field_type": "file", "data_type": "text",
                "question": "Consent Form", "form": "f1"}
    result = analyze_column(df, "consent_form", xml_item, shape_min_freq=1)
    assert result["suggested_operations"] == ["constant_replace"]


def test_free_text_override_skips_identifier_field(tmp_path):
    """An identifier-flagged field should stay at constant_replace even
    if the shape distribution looks free-text-like."""
    values = ["long varied value " + str(i) * 10 for i in range(20)]
    df = pd.DataFrame({"c": values})
    xml_item = {"oid": "c", "identifier": True,
                "field_type": "text", "data_type": "text",
                "question": "Some Field", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["suggested_operations"] == ["constant_replace"]


# ---------- empirical numeric override ----------

def test_empirical_override_all_integer_text_field(tmp_path):
    """A REDCap `text` field with no validation but all-integer data
    should recover to integer + passthrough."""
    df = pd.DataFrame({"score": ["12", "45", "8", "99", "0"]})
    xml_item = {"oid": "score", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None,
                "question": "Test Score", "form": "f1"}
    result = analyze_column(df, "score", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "integer"
    assert result["dtype_empirical_override"] is True
    assert result["suggested_operations"] == ["passthrough"]


def test_empirical_override_all_float_text_field(tmp_path):
    df = pd.DataFrame({"score": ["12.5", "45.0", "8.75", "99.9"]})
    xml_item = {"oid": "score", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None,
                "question": "Score", "form": "f1"}
    result = analyze_column(df, "score", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "float"
    assert result["dtype_empirical_override"] is True
    assert result["suggested_operations"] == ["passthrough"]


def test_empirical_override_skipped_when_one_value_non_numeric(tmp_path):
    """Even one non-numeric value → keep string (that outlier might carry PHI)."""
    df = pd.DataFrame({"c": ["12", "45", "not a number", "99"]})
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "string"
    assert result["dtype_empirical_override"] is False


def test_empirical_override_skipped_when_validation_type_set(tmp_path):
    """If REDCap declared a validation, we trust it — don't second-guess."""
    df = pd.DataFrame({"c": ["12", "45", "8"]})
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": "email",
                "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    # email validation → dtype stays as email (not overridden).
    assert result["suggested_our_dtype"] == "email"
    assert result["dtype_empirical_override"] is False


def test_empirical_override_skipped_when_too_few_values(tmp_path):
    df = pd.DataFrame({"c": ["42"]})
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["dtype_empirical_override"] is False


def test_empirical_override_skipped_when_empty(tmp_path):
    df = pd.DataFrame({"c": ["", "", ""]})
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["dtype_empirical_override"] is False


def test_empirical_override_prefers_integer_over_float(tmp_path):
    """When both parse rates are 100%, prefer integer (stricter)."""
    df = pd.DataFrame({"c": ["1", "2", "3", "4"]})
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "integer"


def test_empirical_override_time_hhmm(tmp_path):
    """All-HH:MM values → time_string pseudo-dtype → passthrough."""
    df = pd.DataFrame({"start": ["09:30", "14:15", "23:59", "00:00"]})
    xml_item = {"oid": "start", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "Start Time", "form": "f1"}
    result = analyze_column(df, "start", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "time_string"
    assert result["dtype_empirical_override"] is True
    assert result["suggested_operations"] == ["passthrough"]


def test_empirical_override_time_hhmmss(tmp_path):
    df = pd.DataFrame({"start": ["09:30:00", "14:15:30", "23:59:59"]})
    xml_item = {"oid": "start", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "Start Time", "form": "f1"}
    result = analyze_column(df, "start", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "time_string"


def test_empirical_override_invalid_time_rejected(tmp_path):
    """Not-HH:MM values → stay as string."""
    df = pd.DataFrame({"c": ["25:00", "09:30", "14:15"]})  # 25:00 invalid
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "string"


def test_empirical_override_ints_with_surrounding_whitespace(tmp_path):
    """Values like ' 42 ' or '  100' should still count as integer once
    stripped. Users often paste values with copy-paste whitespace."""
    df = pd.DataFrame({"c": [" 42", "100 ", "  7  ", "0"]})
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "integer"
    assert result["dtype_empirical_override"] is True


def test_empirical_override_floats_with_leading_space(tmp_path):
    """Matches the real preop_ldfr_norm case: shape ' -9{1}.9{1}'."""
    df = pd.DataFrame({"c": [" -1.5", " 2.3", " -0.7", "4.1"]})
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "float"
    assert result["dtype_empirical_override"] is True


def test_empirical_override_still_rejects_non_numeric_outlier(tmp_path):
    """Whitespace tolerance doesn't help if there's a real outlier."""
    df = pd.DataFrame({"c": [" 42 ", " 100 ", "WNL", "  55  "]})
    xml_item = {"oid": "c", "identifier": False,
                "field_type": "text", "data_type": "text",
                "validation_type": None, "question": "C", "form": "f1"}
    result = analyze_column(df, "c", xml_item, shape_min_freq=1)
    assert result["suggested_our_dtype"] == "string"
    assert result["dtype_empirical_override"] is False


def test_analyze_column_short_uniform_not_free_text(tmp_path):
    # Short, uniform values → not free text (structured enum-like).
    df = pd.DataFrame({"c": ["Yes", "No", "Yes", "No", "Yes", "No"]})
    result = analyze_column(df, "c", None, shape_min_freq=1)
    assert result["likely_free_text"] is False


def test_analyze_column_shape_stats_present(tmp_path):
    df = pd.DataFrame({"c": ["abc", "de", "fghi", "jk"]})
    result = analyze_column(df, "c", None, shape_min_freq=1)
    assert result["stats"]["n_unique_shapes"] >= 1
    assert result["stats"]["max_shape_length"] >= 4


# ---------- inspect end-to-end ----------

def test_inspect_small_synthetic(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame([
        {"Subject Number": "1001", "Subject Name": "Alice",
         "Gender": "F", "Event Name": "patient_information"},
        {"Subject Number": "1002", "Subject Name": "Bob",
         "Gender": "M", "Event Name": "patient_information"},
    ]).to_csv(csv_path, index=False)

    xml_meta = {
        "items": {
            "subject_number": {"oid": "subject_number", "identifier": False,
                               "field_type": "text", "data_type": "integer",
                               "validation_type": "int",
                               "question": "Subject Number", "form": "f1"},
            "subject_name": {"oid": "subject_name", "identifier": True,
                             "field_type": "text", "data_type": "text",
                             "question": "Subject Name", "form": "f1"},
            "gender": {"oid": "gender", "identifier": False,
                       "field_type": "radio", "data_type": "text",
                       "question": "Gender", "form": "f1",
                       "code_list": {"items": [{"code": "F", "label": "F"},
                                               {"code": "M", "label": "M"}]}},
        },
    }
    result = inspect(csv_path, xml_meta)
    assert result["meta"]["n_rows"] == 2
    assert result["meta"]["n_columns"] == 4
    assert result["meta"]["raw_values_emitted"] is False
    assert "Event Name" in result["meta"]["system_columns"]
    # No raw values anywhere.
    for col_result in result["columns"].values():
        assert "sample_values" not in col_result
    # cofill: patient_info columns all filled on 2 rows → cluster of 3
    clusters = result["structural_analysis"]["cofill_clusters"]
    assert any(c["n_columns"] >= 3 for c in clusters)
