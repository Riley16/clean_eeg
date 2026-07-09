"""Tests for scripts/text_phi/audit.py (event-based interface)."""

from __future__ import annotations

import hashlib
import json

from scripts.text_phi.audit import AuditWriter
from scripts.text_phi.records import FieldRedactionEvent
from scripts.text_phi.redactor import RedactionSpan
from scripts.text_phi.schema import Schema


def _minimal_schema() -> Schema:
    return Schema.from_dict({
        "schema_version": "1", "format": "csv",
        "fields": {
            "note": {"dtype": "string", "operations": ["passthrough"]},
        },
    })


def _event(field="note", location=None, original="orig", redacted="X",
           spans=None, ops=None):
    return FieldRedactionEvent(
        location=location or {"row": 0},
        field_name=field,
        original=original,
        redacted=redacted,
        spans=spans or [RedactionSpan(0, 4, "PERSON", 0.9, "denylist", "orig")],
        operations_applied=ops or ["constant_replace"],
    )


def test_top_level_carries_schema_hash(tmp_path):
    out = tmp_path / "audit.json"
    schema = _minimal_schema()
    with AuditWriter(out, "in.csv", "both", schema, "literal") as aw:
        aw.add_event(_event())
    data = json.loads(out.read_text())
    assert data["source"] == "in.csv"
    assert data["mode"] == "both"
    assert data["schema_version"] == "1"
    assert data["schema_sha256"] == schema.sha256()
    assert data["n_entries"] == 1


def test_entry_carries_field_and_operations(tmp_path):
    out = tmp_path / "audit.json"
    with AuditWriter(out, "in.csv", "subject", _minimal_schema(), "literal") as aw:
        aw.add_event(_event(field="note", ops=["subject_name_scan", "generic_phi_scan"]))
    entry = json.loads(out.read_text())["entries"][0]
    assert entry["field"] == "note"
    assert entry["operations_applied"] == ["subject_name_scan", "generic_phi_scan"]


def test_default_hides_original(tmp_path):
    out = tmp_path / "audit.json"
    with AuditWriter(out, "in.csv", "both", _minimal_schema(), "literal") as aw:
        aw.add_event(_event(original="sensitive"))
    entry = json.loads(out.read_text())["entries"][0]
    assert entry["original_sha256"] == hashlib.sha256(b"sensitive").hexdigest()
    assert "original_value" not in entry


def test_include_original_true(tmp_path):
    out = tmp_path / "audit.json"
    with AuditWriter(
        out, "in.csv", "both", _minimal_schema(), "literal",
        include_original=True,
    ) as aw:
        aw.add_event(_event(original="John"))
    entry = json.loads(out.read_text())["entries"][0]
    assert entry["original_value"] == "John"


def test_spans_serialized_with_all_fields(tmp_path):
    out = tmp_path / "audit.json"
    span = RedactionSpan(0, 4, "SUBJECT_NAME", 0.95, "denylist", "John")
    with AuditWriter(out, "in.csv", "subject", _minimal_schema(), "literal") as aw:
        aw.add_event(_event(spans=[span]))
    s = json.loads(out.read_text())["entries"][0]["spans"][0]
    assert s["entity_type"] == "SUBJECT_NAME"
    assert s["matched_text"] == "John"


def test_multiple_events(tmp_path):
    out = tmp_path / "audit.json"
    with AuditWriter(out, "in.csv", "both", _minimal_schema(), "literal") as aw:
        aw.add_event(_event(location={"row": 0}))
        aw.add_event(_event(location={"row": 1}))
    data = json.loads(out.read_text())
    assert data["n_entries"] == 2
    assert [e["location"] for e in data["entries"]] == [{"row": 0}, {"row": 1}]


def test_subject_names_recorded(tmp_path):
    out = tmp_path / "audit.json"
    with AuditWriter(
        out, "in.csv", "subject", _minimal_schema(), "literal",
        subject_names=["John O'Connor"],
    ) as aw:
        pass
    data = json.loads(out.read_text())
    assert data["subject_names"] == ["John O'Connor"]
