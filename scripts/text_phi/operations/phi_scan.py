"""PHI-scan operations bridging to the existing detector layer.

- `parse_subject_name`: for subject_name-typed fields, parses the value into
  a `PersonalName` and publishes it to `record_context.subject`. Does not
  replace the field; a downstream operation (typically `constant_replace`)
  handles that.
- `subject_name_scan`: runs `TextRedactor.redact_with_subject` if the record
  has a parsed subject; otherwise passes through.
- `generic_phi_scan`: runs `TextRedactor`'s generic layer. Whitelist filter
  on PERSON hits is applied inside the generic analyzer.
"""

from __future__ import annotations

from typing import ClassVar

from ..name_parse import parse_name
from ..records import OperationContext
from ..redactor import RedactionResult, RedactionSpan


class ParseSubjectNameOperation:
    name: ClassVar[str] = "parse_subject_name"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["subject_name"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value or not value.strip():
            return value, []
        name_format = ctx.params.get("name_format")
        try:
            pn = parse_name(value, name_format=name_format)
        except ValueError:
            return value, []
        ctx.record_context.subject = pn
        # Field value untouched; a follow-on constant_replace or similar can
        # blank it. This op only publishes the parsed name.
        return value, []


class SubjectNameScanOperation:
    name: ClassVar[str] = "subject_name_scan"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["string"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    # Optional dependency: which field carries the subject name (if not
    # already published to record_context by an upstream parse_subject_name).
    optional_roles: ClassVar[frozenset[str]] = frozenset(["subject_name_field"])

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        redactor = ctx.text_redactor
        if redactor is None:
            return value, []

        subject = ctx.record_context.subject
        # Fallback: if no subject in context but the schema declares which
        # field holds one, try parsing it lazily.
        if subject is None:
            name_field = ctx.depends_on.get("subject_name_field")
            if name_field is not None:
                raw = ctx.record.get(name_field, "")
                if raw.strip():
                    try:
                        subject = parse_name(raw)
                    except ValueError:
                        subject = None

        if subject is None:
            return value, []

        result: RedactionResult = redactor.redact_with_subject(value, subject)
        return result.text, list(result.spans)


class GenericPhiScanOperation:
    name: ClassVar[str] = "generic_phi_scan"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["string"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        redactor = ctx.text_redactor
        if redactor is None or redactor._generic is None:
            return value, []
        # Piggyback on TextRedactor's existing generic pass. We call the
        # underlying analyzer + whitelist filter directly to avoid mixing in
        # subject-name spans (that layer is `subject_name_scan`'s job).
        from ..detectors import filter_person_hits_by_whitelist
        gen_results = redactor._generic.analyze(
            text=value, entities=redactor.entities, language="en"
        )
        gen_results = filter_person_hits_by_whitelist(value, gen_results)
        from ..redactor import _apply_replacements, _merge_spans, _result_to_span
        spans = [_result_to_span(r, value) for r in gen_results]
        merged = _merge_spans(spans)
        redacted = _apply_replacements(
            value, merged,
            redactor.replacement_style,
            redactor.literal_replacement,
        )
        return redacted, merged
