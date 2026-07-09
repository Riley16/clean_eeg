"""Shared fixtures for operation tests: a helper that builds a minimal
OperationContext for a given field/value/params configuration."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.text_phi.records import OperationContext, RecordContext
from scripts.text_phi.schema import FieldSpec, OperationCall, Schema


def _make_schema(field_name: str, dtype: str, depends_on: dict[str, str] | None = None) -> Schema:
    raw = {
        "schema_version": "1",
        "format": "csv",
        "fields": {
            field_name: {
                "dtype": dtype,
                "description": "",
                "operations": ["passthrough"],
                **({"depends_on": depends_on} if depends_on else {}),
            }
        },
    }
    # Add dep target fields as passthrough strings so the schema validates.
    if depends_on:
        for target in depends_on.values():
            if target not in raw["fields"]:
                raw["fields"][target] = {
                    "dtype": dtype if target.endswith("_date") else "string",
                    "description": "",
                    "operations": ["passthrough"],
                }
    return Schema.from_dict(raw)


@pytest.fixture
def make_ctx():
    """Return a factory that builds an OperationContext with the given
    field name, dtype, current record fields, per-call params, and a
    fresh RecordContext.

    Signature: make_ctx(field_name, dtype, *, record=None, params=None,
                        depends_on=None, record_context=None, text_redactor=None)
    """
    def _build(
        field_name: str,
        dtype: str,
        *,
        record: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        depends_on: dict[str, str] | None = None,
        record_context: RecordContext | None = None,
        text_redactor=None,
    ) -> OperationContext:
        schema = _make_schema(field_name, dtype, depends_on)
        field_spec = schema.fields[field_name]
        # Substitute a FieldSpec with the desired depends_on (schema's copy
        # is what got validated).
        return OperationContext(
            field_name=field_name,
            field_spec=field_spec,
            params=params or {},
            depends_on=depends_on or {},
            record=record or {},
            record_context=record_context or RecordContext(),
            schema=schema,
            text_redactor=text_redactor,
        )

    return _build
