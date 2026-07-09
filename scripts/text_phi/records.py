"""Records model: the internal data type carried between formats + redactor,
plus the processing loop that walks records through their schema-declared
operations.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clean_eeg.anonymize import PersonalName

    from .redactor import RedactionSpan, TextRedactor
    from .schema import FieldSpec, Schema


@dataclass
class Record:
    """One row/entry from a file. Values are the exact source strings the
    format's `.load` produced — no coercion. Location is format-specific
    (e.g. `{"row": 3}` for CSV, `{"line": 12}` for TXT)."""
    location: dict[str, Any]
    fields: dict[str, str]


@dataclass
class RecordContext:
    """Per-record shared state that operations read and write during
    processing. Lives for the duration of one record."""
    subject: "PersonalName | None" = None
    date_offsets: dict[str, _dt.timedelta] = field(default_factory=dict)
    parsed_values: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationContext:
    """The bundle of state an Operation.apply() call sees.

    Fields:
      field_name: the schema field being processed
      field_spec: its FieldSpec (from the schema)
      params: this specific operation call's params dict
      depends_on: role -> field-name map from the field spec
      record: the current record's mutable fields dict — earlier operations
              may have already updated other fields
      record_context: per-record shared state (subject, date offsets, ...)
      schema: the Schema, for cross-field lookups (e.g. dtype of a dep field)
      text_redactor: the TextRedactor instance (only used by phi_scan ops)
    """
    field_name: str
    field_spec: "FieldSpec"
    params: dict[str, Any]
    depends_on: dict[str, str]
    record: dict[str, str]
    record_context: RecordContext
    schema: "Schema"
    text_redactor: "TextRedactor | None"


@dataclass
class FieldRedactionEvent:
    """Per-field record of what happened during processing. Emitted for any
    field where either spans were produced or the value changed. Feeds
    directly into the audit report."""
    location: dict[str, Any]
    field_name: str
    original: str
    redacted: str
    spans: list["RedactionSpan"]
    operations_applied: list[str]


class PreservedFieldViolation(RuntimeError):
    """A field declared as pure-passthrough was mutated by processing."""


class RecordRedactor:
    """Apply a Schema's operations to a stream of Records.

    Processing order for fields within one record is a topological sort over
    the depends_on graph (see Schema.processing_order()). All records share
    that order; each record has its own RecordContext.
    """

    def __init__(
        self,
        schema: "Schema",
        text_redactor: "TextRedactor | None" = None,
        default_subject: "PersonalName | None" = None,
    ):
        self.schema = schema
        self.text_redactor = text_redactor
        self.default_subject = default_subject
        self._processing_order = schema.processing_order()

    def process_records(
        self, records: list[Record]
    ) -> tuple[list[Record], list[FieldRedactionEvent]]:
        out_records: list[Record] = []
        all_events: list[FieldRedactionEvent] = []
        for r in records:
            new_r, events = self.process_record(r)
            out_records.append(new_r)
            all_events.extend(events)
        self._verify_preserved_fields(records, out_records)
        return out_records, all_events

    def process_record(
        self, record: Record
    ) -> tuple[Record, list[FieldRedactionEvent]]:
        # Local import so records.py doesn't import operations at module
        # load time (which would create a cycle: operations imports records).
        from .operations import OPERATIONS

        ctx_state = RecordContext(subject=self.default_subject)
        new_fields = dict(record.fields)
        events: list[FieldRedactionEvent] = []

        for field_name in self._processing_order:
            if field_name not in new_fields:
                # Field declared in schema but absent from this record —
                # missing-field policy is enforced at load time; here we
                # just skip.
                continue
            spec = self.schema.fields[field_name]
            original_value = new_fields[field_name]
            current_value = original_value
            spans_for_field: list["RedactionSpan"] = []
            ops_applied: list[str] = []

            for op_call in spec.operations:
                op = OPERATIONS[op_call.name]
                op_ctx = OperationContext(
                    field_name=field_name,
                    field_spec=spec,
                    params=op_call.params,
                    depends_on=spec.depends_on,
                    record=new_fields,
                    record_context=ctx_state,
                    schema=self.schema,
                    text_redactor=self.text_redactor,
                )
                current_value, new_spans = op.apply(current_value, op_ctx)
                spans_for_field.extend(new_spans)
                ops_applied.append(op_call.name)

            new_fields[field_name] = current_value

            if spans_for_field or current_value != original_value:
                events.append(FieldRedactionEvent(
                    location=dict(record.location),
                    field_name=field_name,
                    original=original_value,
                    redacted=current_value,
                    spans=spans_for_field,
                    operations_applied=ops_applied,
                ))

        return Record(location=dict(record.location), fields=new_fields), events

    def _verify_preserved_fields(
        self, in_records: list[Record], out_records: list[Record]
    ) -> None:
        """Structural invariant: any field whose operations list is exactly
        `["passthrough"]` must be byte-identical between input and output.

        Catches operation bugs early (e.g. an op that accidentally mutates
        the passthrough case)."""
        passthrough_fields = [
            name for name, fs in self.schema.fields.items()
            if [op.name for op in fs.operations] == ["passthrough"]
        ]
        for i, (in_r, out_r) in enumerate(zip(in_records, out_records)):
            for name in passthrough_fields:
                in_val = in_r.fields.get(name)
                out_val = out_r.fields.get(name)
                if in_val != out_val:
                    raise PreservedFieldViolation(
                        f"preserved-field byte-identity violated: record {i} "
                        f"field {name!r} mutated from {in_val!r} to {out_val!r}"
                    )
