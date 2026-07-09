"""Tests for scripts/text_phi/inspect/column_mapping.py."""

from __future__ import annotations

import json

import pytest

from scripts.text_phi.inspect.column_mapping import (
    _normalize_label,
    base_label,
    duplicate_base_name_groups,
    enumerate_expected_labels,
    expected_label,
    load_column_list,
    map_expected_to_csv,
)


# ---------- _normalize_label ----------

def test_normalize_html_entity_apostrophe():
    assert _normalize_label("Rasmussen&#039;s encephalitis") == "Rasmussen's encephalitis"


def test_normalize_collapses_newlines_to_single_space():
    assert _normalize_label("line one\nline two\n\nline three") == "line one line two line three"


def test_normalize_collapses_double_spaces():
    assert _normalize_label("Depth coverage (check all that apply)  (choice=X)") == \
        "Depth coverage (check all that apply) (choice=X)"


def test_normalize_double_quote_to_single():
    assert _normalize_label('Was this a "typical" seizure?') == "Was this a 'typical' seizure?"


def test_normalize_none_returns_empty():
    assert _normalize_label(None) == ""


# ---------- expected_label ----------

def test_expected_label_non_checkbox():
    it = {"question": "Subject Name", "field_type": "text"}
    assert expected_label(it) == "Subject Name"


def test_expected_label_checkbox():
    it = {"question": "Race", "field_type": "checkbox", "checkbox_label": "Black"}
    assert expected_label(it) == "Race (choice=Black)"


def test_expected_label_checkbox_empty_choice():
    it = {"question": "Was this a bad session? (DCC use only)",
          "field_type": "checkbox", "checkbox_label": ""}
    assert expected_label(it) == "Was this a bad session? (DCC use only) (choice=)"


def test_expected_label_checkbox_no_choice_label():
    it = {"question": "Race", "field_type": "checkbox", "checkbox_label": None}
    assert expected_label(it) == "Race"


# ---------- base_label ----------

def test_base_label_strips_dedup_suffix():
    assert base_label("Start Time.2") == "Start Time"
    assert base_label("Start Time") == "Start Time"


def test_base_label_preserves_case():
    # User relies on this to expose "Start Time" vs "Start time" inconsistency.
    assert base_label("Start time.1") == "Start time"
    assert base_label("Start Time.1") == "Start Time"


def test_base_label_normalizes():
    assert base_label("Race  (choice=Black).1") == "Race (choice=Black)"


# ---------- load_column_list ----------

def test_load_json_array(tmp_path):
    p = tmp_path / "cols.json"
    p.write_text(json.dumps(["a", "b", "c"]))
    assert load_column_list(p) == ["a", "b", "c"]


def test_load_python_list_literal(tmp_path):
    p = tmp_path / "cols.txt"
    p.write_text("[\n 'Subject Name',\n 'Date of Birth',\n]")
    assert load_column_list(p) == ["Subject Name", "Date of Birth"]


# ---------- duplicate_base_name_groups ----------

def test_duplicate_groups_finds_stress_case():
    cols = [
        "Start Time", "Start Time.1", "Start Time.2",  # capital T group
        "Start time", "Start time.1",                   # lowercase t group
        "Other",
    ]
    groups = duplicate_base_name_groups(cols)
    assert "Start Time" in groups
    assert "Start time" in groups
    assert "Other" not in groups  # single member excluded


# ---------- map_expected_to_csv ----------

def _meta(items: list[dict]) -> dict:
    """Wrap a list of item dicts in the shape parse_metadata returns."""
    return {"items": {it["oid"]: it for it in items}}


def test_map_all_matched():
    meta = _meta([
        {"oid": "subject_number", "question": "Subject Number",
         "field_type": "text", "form": "f1"},
        {"oid": "subject_name", "question": "Subject Name",
         "field_type": "text", "form": "f1", "identifier": True},
    ])
    expected = enumerate_expected_labels(meta)
    result = map_expected_to_csv(expected, ["Subject Number", "Subject Name"])
    assert result["n_mapped"] == 2
    assert result["n_unmapped_xml"] == 0
    assert result["n_unmapped_csv"] == 0


def test_map_repeat_column_dedup_zips():
    meta = _meta([
        {"oid": "st1", "question": "Start Time", "field_type": "text", "form": "f"},
        {"oid": "st2", "question": "Start Time", "field_type": "text", "form": "f"},
        {"oid": "st3", "question": "Start Time", "field_type": "text", "form": "f"},
    ])
    expected = enumerate_expected_labels(meta)
    result = map_expected_to_csv(
        expected, ["Start Time", "Start Time.1", "Start Time.2"],
    )
    assert result["n_mapped"] == 3
    assert result["n_unmapped_xml"] == 0


def test_map_extra_xml_reported_unmapped():
    meta = _meta([
        {"oid": "a", "question": "A", "field_type": "text", "form": "f"},
        {"oid": "b", "question": "B", "field_type": "text", "form": "f"},
    ])
    expected = enumerate_expected_labels(meta)
    result = map_expected_to_csv(expected, ["A"])  # 'B' missing from CSV
    assert result["n_mapped"] == 1
    assert result["n_unmapped_xml"] == 1
    assert result["unmapped_xml"][0]["item_oid"] == "b"


def test_map_extra_csv_reported_unmapped():
    meta = _meta([
        {"oid": "a", "question": "A", "field_type": "text", "form": "f"},
    ])
    expected = enumerate_expected_labels(meta)
    result = map_expected_to_csv(expected, ["A", "Mystery Column"])
    assert result["n_mapped"] == 1
    assert result["n_unmapped_csv"] == 1
    assert result["unmapped_csv"] == ["Mystery Column"]


def test_map_system_columns_pulled_out():
    meta = _meta([
        {"oid": "a", "question": "A", "field_type": "text", "form": "f"},
    ])
    expected = enumerate_expected_labels(meta)
    result = map_expected_to_csv(
        expected, ["A", "Event Name", "Data Access Group"],
    )
    assert result["n_mapped"] == 1
    assert result["n_unmapped_csv"] == 0
    assert set(result["system_columns"]) == {"Event Name", "Data Access Group"}
