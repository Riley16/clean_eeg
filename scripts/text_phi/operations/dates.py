"""Date and datetime redaction operations.

The stay-anchor pattern:
  * On the anchor field (e.g. `admission_date`) apply `date_shift_to_base`.
    That computes offset = BASE - anchor_value, sets the field to BASE, and
    publishes the offset on record_context.date_offsets[anchor_field].
  * On dependent date fields apply `date_shift_relative_to_stay_start` with
    `depends_on: {stay_start_field: "admission_date"}`. Each dependent
    shifts by that same offset, so intra-record intervals are preserved
    exactly and the absolute dates disappear.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from ..dtypes import get_dtype
from ..records import OperationContext
from ..redactor import RedactionSpan


# Same anchor the EDF pipeline uses (clean_subject_eeg.BASE_START_DATE).
DEFAULT_BASE_DATE = _dt.date(1985, 1, 1)


def _parse_date_like(source_str: str, dtype_name: str):
    if not source_str:
        return None
    return get_dtype(dtype_name).parse(source_str)


def _format_date_like(value, dtype_name: str) -> str:
    if value is None:
        return ""
    return get_dtype(dtype_name).format(value)


def _redaction_span(field_value: str, entity: str, recognizer: str) -> RedactionSpan:
    return RedactionSpan(
        start=0, end=len(field_value),
        entity_type=entity, score=1.0,
        recognizer=recognizer, matched_text=field_value,
    )


class DateShiftToBaseOperation:
    """Shift the anchor date so it lands on `base_date` (default 1985-01-01),
    and publish the shift on `record_context.date_offsets[field_name]` so
    dependent date fields in the same record can consume it."""
    name: ClassVar[str] = "date_shift_to_base"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["date", "datetime"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        dtype_name = ctx.field_spec.dtype
        base_str = ctx.params.get("base_date", DEFAULT_BASE_DATE.isoformat())
        base_value = _parse_date_like(base_str, dtype_name)
        original = _parse_date_like(value, dtype_name)
        if original is None or base_value is None:
            return value, []

        if dtype_name == "date":
            offset = base_value - original
            new_val = base_value
        else:  # datetime
            offset = base_value - original
            new_val = base_value

        ctx.record_context.date_offsets[ctx.field_name] = offset
        formatted = _format_date_like(new_val, dtype_name)
        return formatted, [_redaction_span(value, "DATE_SHIFTED", self.name)]


class DateShiftRelativeToStayStartOperation:
    """Shift this date by the offset published by the stay-start anchor
    field. Requires `depends_on: {stay_start_field: <anchor_field_name>}`.
    Intervals within the record are preserved."""
    name: ClassVar[str] = "date_shift_relative_to_stay_start"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["date", "datetime"])
    required_roles: ClassVar[frozenset[str]] = frozenset(["stay_start_field"])
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        anchor_field = ctx.depends_on["stay_start_field"]
        offset = ctx.record_context.date_offsets.get(anchor_field)
        if offset is None:
            # Anchor hasn't been processed yet or produced no offset; leave
            # this field untouched rather than emit a wrong shift. The
            # scheduler ordering should prevent this, but stay safe.
            return value, []
        dtype_name = ctx.field_spec.dtype
        original = _parse_date_like(value, dtype_name)
        if original is None:
            return value, []
        shifted = original + offset
        formatted = _format_date_like(shifted, dtype_name)
        return formatted, [_redaction_span(value, "DATE_SHIFTED", self.name)]


class DateYearOnlyOperation:
    """HIPAA Safe Harbor: keep the year, set month/day to 01-01."""
    name: ClassVar[str] = "date_year_only"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["date", "datetime"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        dtype_name = ctx.field_spec.dtype
        original = _parse_date_like(value, dtype_name)
        if original is None:
            return value, []
        if dtype_name == "date":
            new_val = _dt.date(original.year, 1, 1)
        else:  # datetime
            new_val = _dt.datetime(original.year, 1, 1, tzinfo=original.tzinfo)
        formatted = _format_date_like(new_val, dtype_name)
        return formatted, [_redaction_span(value, "DATE_YEAR_ONLY", self.name)]


class DateRedactFullOperation:
    """Replace the value with a fixed placeholder date."""
    name: ClassVar[str] = "date_redact_full"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["date", "datetime"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        dtype_name = ctx.field_spec.dtype
        placeholder_str = ctx.params.get(
            "placeholder",
            DEFAULT_BASE_DATE.isoformat()
            if dtype_name == "date"
            else "1985-01-01T00:00:00",
        )
        return placeholder_str, [_redaction_span(value, "DATE_REDACTED", self.name)]
