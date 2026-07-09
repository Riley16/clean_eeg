"""Tests for scripts/text_phi/records.py (the processing loop)."""

from __future__ import annotations

import pytest

from scripts.text_phi.records import (
    FieldRedactionEvent,
    PreservedFieldViolation,
    Record,
    RecordRedactor,
)
from scripts.text_phi.redactor import TextRedactor
from scripts.text_phi.schema import Schema


def _schema(fields: dict) -> Schema:
    return Schema.from_dict({
        "schema_version": "1",
        "format": "csv",
        "fields": fields,
    })


# ---------- basic processing ----------

def test_passthrough_only_produces_no_events():
    schema = _schema({
        "note": {"dtype": "string", "operations": ["passthrough"]},
    })
    rr = RecordRedactor(schema)
    recs, events = rr.process_records([
        Record(location={"row": 0}, fields={"note": "hi"}),
    ])
    assert recs[0].fields == {"note": "hi"}
    assert events == []


def test_constant_replace_emits_event():
    schema = _schema({
        "name": {
            "dtype": "string",
            "operations": [{"name": "constant_replace", "params": {"value": "[X]"}}],
        },
    })
    rr = RecordRedactor(schema)
    recs, events = rr.process_records([
        Record({"row": 0}, {"name": "John"}),
    ])
    assert recs[0].fields == {"name": "[X]"}
    assert len(events) == 1
    e = events[0]
    assert e.field_name == "name"
    assert e.original == "John"
    assert e.redacted == "[X]"
    assert e.operations_applied == ["constant_replace"]


def test_multiple_operations_chain():
    schema = _schema({
        "s": {
            "dtype": "string",
            "operations": [
                "passthrough",
                {"name": "constant_replace", "params": {"value": "[X]"}},
                "passthrough",
            ],
        },
    })
    rr = RecordRedactor(schema)
    recs, events = rr.process_records([
        Record({"row": 0}, {"s": "orig"}),
    ])
    assert recs[0].fields["s"] == "[X]"
    assert events[0].operations_applied == ["passthrough", "constant_replace", "passthrough"]


def test_multiple_records_processed_independently():
    schema = _schema({
        "s": {
            "dtype": "string",
            "operations": [{"name": "constant_replace", "params": {"value": "[X]"}}],
        },
    })
    rr = RecordRedactor(schema)
    recs, events = rr.process_records([
        Record({"row": 0}, {"s": "a"}),
        Record({"row": 1}, {"s": ""}),  # empty → op is no-op
        Record({"row": 2}, {"s": "c"}),
    ])
    assert [r.fields["s"] for r in recs] == ["[X]", "", "[X]"]
    # empty value produced no event
    assert [e.location for e in events] == [{"row": 0}, {"row": 2}]


# ---------- cross-field dependencies ----------

def test_date_shift_relative_preserves_interval():
    schema = _schema({
        "admission_date": {
            "dtype": "date", "operations": ["date_shift_to_base"],
        },
        "note_date": {
            "dtype": "date",
            "operations": ["date_shift_relative_to_stay_start"],
            "depends_on": {"stay_start_field": "admission_date"},
        },
    })
    rr = RecordRedactor(schema)
    recs, _ = rr.process_records([
        Record({"row": 0}, {"admission_date": "2024-01-10", "note_date": "2024-01-15"}),
    ])
    # Interval was 5 days → post-shift 1985-01-06 - 1985-01-01 = 5 days.
    assert recs[0].fields["admission_date"] == "1985-01-01"
    assert recs[0].fields["note_date"] == "1985-01-06"


def test_subject_name_flows_to_downstream_scan():
    schema = _schema({
        "patient_name": {
            "dtype": "subject_name",
            "operations": [
                "parse_subject_name",
                {"name": "constant_replace", "params": {"value": "[NAME]"}},
            ],
        },
        "note": {
            "dtype": "string",
            "operations": ["subject_name_scan"],
            "depends_on": {"subject_name_field": "patient_name"},
        },
    })
    rr = RecordRedactor(schema, text_redactor=TextRedactor(mode="subject"))
    recs, _ = rr.process_records([
        Record({"row": 0}, {
            "patient_name": "John O'Connor",
            "note": "Dr. John O'Connor saw the patient.",
        }),
    ])
    assert recs[0].fields["patient_name"] == "[NAME]"
    assert "John" not in recs[0].fields["note"]
    assert "O'Connor" not in recs[0].fields["note"]


# ---------- preserved-field enforcement ----------

def test_preserved_field_ok_when_unchanged():
    schema = _schema({
        "channel": {"dtype": "string", "operations": ["passthrough"]},
        "note": {"dtype": "string",
                 "operations": [{"name": "constant_replace", "params": {"value": "[X]"}}]},
    })
    rr = RecordRedactor(schema)
    recs, _ = rr.process_records([
        Record({"row": 0}, {"channel": "LFPx1", "note": "hi"}),
    ])
    assert recs[0].fields["channel"] == "LFPx1"


def test_preserved_field_violation_detected(monkeypatch):
    """Simulate a rogue passthrough that mutates the value → violation raised."""
    from scripts.text_phi.operations import OPERATIONS

    real_passthrough = OPERATIONS["passthrough"]

    class RoguePassthrough:
        name = "passthrough"
        allowed_dtypes = None
        required_roles = frozenset()
        optional_roles = frozenset()

        def apply(self, value, ctx):
            return value + "!", []

    monkeypatch.setitem(OPERATIONS, "passthrough", RoguePassthrough())
    try:
        schema = _schema({
            "channel": {"dtype": "string", "operations": ["passthrough"]},
        })
        rr = RecordRedactor(schema)
        with pytest.raises(PreservedFieldViolation, match="byte-identity violated"):
            rr.process_records([Record({"row": 0}, {"channel": "LFPx1"})])
    finally:
        OPERATIONS["passthrough"] = real_passthrough


# ---------- missing fields tolerated (policy enforced at load time) ----------

def test_missing_field_in_record_is_skipped():
    schema = _schema({
        "a": {"dtype": "string", "operations": ["passthrough"]},
        "b": {"dtype": "string",
              "operations": [{"name": "constant_replace", "params": {"value": "[X]"}}]},
    })
    rr = RecordRedactor(schema)
    # Record only has 'a'; 'b' absent (missing-field policy is handled by the
    # format loader in Phase C; here the loop just skips).
    recs, _ = rr.process_records([Record({"row": 0}, {"a": "hi"})])
    assert recs[0].fields == {"a": "hi"}


# ---------- event structure ----------

def test_event_carries_location_and_operations_applied():
    schema = _schema({
        "s": {
            "dtype": "string",
            "operations": [{"name": "constant_replace", "params": {"value": "[X]"}}],
        },
    })
    rr = RecordRedactor(schema)
    _, events = rr.process_records([Record({"row": 5, "column": "s"}, {"s": "orig"})])
    assert len(events) == 1
    assert events[0].location == {"row": 5, "column": "s"}
    assert events[0].operations_applied == ["constant_replace"]
    assert len(events[0].spans) == 1


def test_empty_records_list():
    schema = _schema({"s": {"dtype": "string", "operations": ["passthrough"]}})
    rr = RecordRedactor(schema)
    recs, events = rr.process_records([])
    assert recs == []
    assert events == []
