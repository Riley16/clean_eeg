"""Tests for scripts/text_phi/inspect/generate_schema.py (LLM integration)."""

from __future__ import annotations

from scripts.text_phi.inspect.generate_schema import build_schema


def _mock_inspection(cols: list[tuple[str, str]]) -> dict:
    """Build a minimal inspection dict from (col_name, dtype) tuples."""
    columns: dict = {}
    for name, dtype in cols:
        columns[name] = {
            "csv_column": name,
            "xml": {
                "item_oid": name,
                "identifier": False,
                "field_type": "text",
                "data_type": "text",
                "question": name,
                "form": "f1",
            },
            "stats": {},
            "empirical_dtype_parse_rates": {},
            "top_value_shapes": [],
            "likely_free_text": False,
            "suggested_our_dtype": dtype,
            "dtype_empirical_override": False,
            "suggested_operations": [],
            "phi_hint": None,
        }
    return {"columns": columns}


def test_enable_llm_appends_to_scan_pipeline():
    insp = _mock_inspection([
        ("subject_name", "subject_name"),
        ("implant_date", "date"),
        ("note_text", "string"),
    ])
    schema = build_schema(insp, anchor_date="implant_date",
                         name_field="subject_name", enable_llm=True)
    ops = schema["fields"]["note_text"]["operations"]
    assert ops[0] == "subject_name_scan"
    assert ops[1] == "generic_phi_scan"
    # Third op is the LLM scan.
    assert isinstance(ops[2], dict)
    assert ops[2]["name"] == "llm_scan"
    assert ops[2]["params"]["report_only"] is True
    assert ops[2]["params"]["prompt"] == "generic_phi"
    # Context references the name + anchor date fields.
    assert "subject_name" in ops[2]["params"]["context"]["patient_name"]
    assert "implant_date" in ops[2]["params"]["context"]["admission_date"]


def test_enable_llm_skipped_for_passthrough_fields():
    insp = _mock_inspection([
        ("subject_name", "subject_name"),
        ("implant_date", "date"),
        ("channel", "string"),
    ])
    # channel gets `passthrough` because it looks structural — verify
    # llm_scan is NOT appended in that case.
    insp["columns"]["channel"]["xml"]["field_type"] = "radio"  # → enum → passthrough
    insp["columns"]["channel"]["suggested_our_dtype"] = "enum"

    schema = build_schema(insp, anchor_date="implant_date",
                         name_field="subject_name", enable_llm=True)
    assert schema["fields"]["channel"]["operations"] == ["passthrough"]


def test_disable_llm_leaves_ops_alone():
    insp = _mock_inspection([
        ("subject_name", "subject_name"),
        ("implant_date", "date"),
        ("note_text", "string"),
    ])
    schema = build_schema(insp, anchor_date="implant_date",
                         name_field="subject_name", enable_llm=False)
    ops = schema["fields"]["note_text"]["operations"]
    assert ops == ["subject_name_scan", "generic_phi_scan"]


def test_enable_llm_skipped_for_constant_replace():
    """An identifier-flagged field goes to constant_replace — llm_scan
    would add no value on filename attachments."""
    insp = _mock_inspection([
        ("subject_name", "subject_name"),
        ("implant_date", "date"),
        ("consent_form", "string"),
    ])
    insp["columns"]["consent_form"]["xml"]["identifier"] = True
    insp["columns"]["consent_form"]["xml"]["field_type"] = "file"
    schema = build_schema(insp, anchor_date="implant_date",
                         name_field="subject_name", enable_llm=True)
    assert schema["fields"]["consent_form"]["operations"] == ["constant_replace"]
