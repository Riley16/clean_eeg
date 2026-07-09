"""Tests for date/datetime operations."""

from __future__ import annotations

import datetime as _dt

from scripts.text_phi.operations.dates import (
    DEFAULT_BASE_DATE,
    DateRedactFullOperation,
    DateShiftRelativeToStayStartOperation,
    DateShiftToBaseOperation,
    DateYearOnlyOperation,
)
from scripts.text_phi.records import RecordContext


# ---------- date_shift_to_base ----------

def test_shift_to_base_lands_on_base(make_ctx):
    op = DateShiftToBaseOperation()
    ctx = make_ctx("admission_date", "date")
    new_val, spans = op.apply("2024-06-15", ctx)
    assert new_val == DEFAULT_BASE_DATE.isoformat()
    assert len(spans) == 1
    assert spans[0].entity_type == "DATE_SHIFTED"


def test_shift_to_base_publishes_offset(make_ctx):
    op = DateShiftToBaseOperation()
    rc = RecordContext()
    ctx = make_ctx("admission_date", "date", record_context=rc)
    op.apply("2024-06-15", ctx)
    assert "admission_date" in rc.date_offsets
    original = _dt.date.fromisoformat("2024-06-15")
    expected = DEFAULT_BASE_DATE - original
    assert rc.date_offsets["admission_date"] == expected


def test_shift_to_base_custom_base(make_ctx):
    op = DateShiftToBaseOperation()
    ctx = make_ctx("d", "date", params={"base_date": "2000-01-01"})
    new_val, _ = op.apply("2024-06-15", ctx)
    assert new_val == "2000-01-01"


def test_shift_to_base_empty_passthrough(make_ctx):
    op = DateShiftToBaseOperation()
    ctx = make_ctx("d", "date")
    assert op.apply("", ctx) == ("", [])


def test_shift_to_base_datetime(make_ctx):
    op = DateShiftToBaseOperation()
    ctx = make_ctx("d", "datetime")
    new_val, _ = op.apply("2024-06-15T10:30:00", ctx)
    assert new_val == "1985-01-01T00:00:00"


# ---------- date_shift_relative_to_stay_start ----------

def test_shift_relative_preserves_interval(make_ctx):
    """Same offset applied to both anchor and dependent → interval preserved."""
    rc = RecordContext()
    anchor_op = DateShiftToBaseOperation()
    anchor_ctx = make_ctx("admission_date", "date", record_context=rc)
    anchor_ctx.record["admission_date"] = "2024-01-10"
    anchor_op.apply("2024-01-10", anchor_ctx)

    dep_op = DateShiftRelativeToStayStartOperation()
    dep_ctx = make_ctx(
        "note_date", "date",
        depends_on={"stay_start_field": "admission_date"},
        record_context=rc,
    )
    new_val, spans = dep_op.apply("2024-01-15", dep_ctx)
    # Original interval was 5 days; expect 1985-01-06 after shift.
    assert new_val == "1985-01-06"
    assert len(spans) == 1


def test_shift_relative_without_anchor_processed_passes_through(make_ctx):
    rc = RecordContext()  # no offsets published
    dep_op = DateShiftRelativeToStayStartOperation()
    ctx = make_ctx(
        "note_date", "date",
        depends_on={"stay_start_field": "admission_date"},
        record_context=rc,
    )
    new_val, spans = dep_op.apply("2024-01-15", ctx)
    assert new_val == "2024-01-15"
    assert spans == []


def test_shift_relative_empty_passthrough(make_ctx):
    dep_op = DateShiftRelativeToStayStartOperation()
    ctx = make_ctx("d", "date",
                   depends_on={"stay_start_field": "anchor"},
                   record_context=RecordContext())
    assert dep_op.apply("", ctx) == ("", [])


# ---------- date_year_only ----------

def test_year_only_zeros_month_day(make_ctx):
    op = DateYearOnlyOperation()
    ctx = make_ctx("d", "date")
    new_val, spans = op.apply("2024-06-15", ctx)
    assert new_val == "2024-01-01"
    assert len(spans) == 1


def test_year_only_datetime_preserves_timezone(make_ctx):
    op = DateYearOnlyOperation()
    ctx = make_ctx("d", "datetime")
    new_val, _ = op.apply("2024-06-15T10:30:00+00:00", ctx)
    assert new_val.startswith("2024-01-01T00:00:00")
    assert "+00:00" in new_val


def test_year_only_empty(make_ctx):
    op = DateYearOnlyOperation()
    ctx = make_ctx("d", "date")
    assert op.apply("", ctx) == ("", [])


# ---------- date_redact_full ----------

def test_redact_full_default_placeholder(make_ctx):
    op = DateRedactFullOperation()
    ctx = make_ctx("d", "date")
    new_val, spans = op.apply("2024-06-15", ctx)
    assert new_val == DEFAULT_BASE_DATE.isoformat()
    assert len(spans) == 1


def test_redact_full_custom_placeholder(make_ctx):
    op = DateRedactFullOperation()
    ctx = make_ctx("d", "date", params={"placeholder": "1900-01-01"})
    new_val, _ = op.apply("2024-06-15", ctx)
    assert new_val == "1900-01-01"


def test_redact_full_datetime_placeholder(make_ctx):
    op = DateRedactFullOperation()
    ctx = make_ctx("d", "datetime")
    new_val, _ = op.apply("2024-06-15T10:30:00", ctx)
    assert new_val == "1985-01-01T00:00:00"


def test_redact_full_empty(make_ctx):
    op = DateRedactFullOperation()
    ctx = make_ctx("d", "date")
    assert op.apply("", ctx) == ("", [])
