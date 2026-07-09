"""Whole-field PHI redaction for typed fields.

Different from `generic_phi_scan`: these operations assume the entire field
value is the identifier (e.g. a `dtype: email` field's value IS an email
address), so they redact the whole field rather than scanning for embedded
matches.
"""

from __future__ import annotations

from typing import ClassVar

from ..records import OperationContext
from ..redactor import RedactionSpan


def _replace_whole(
    value: str,
    entity: str,
    recognizer: str,
    ctx: OperationContext,
    default_replacement: str = "",
) -> tuple[str, list[RedactionSpan]]:
    if not value:
        return value, []
    replacement = ctx.params.get("value", default_replacement)
    span = RedactionSpan(
        start=0, end=len(value),
        entity_type=entity, score=1.0,
        recognizer=recognizer, matched_text=value,
    )
    return replacement, [span]


class ZipRedactOperation:
    name: ClassVar[str] = "zip_redact"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["zip_code"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        return _replace_whole(value, "US_ZIP_CODE", self.name, ctx, "00000")


class PhoneRedactOperation:
    name: ClassVar[str] = "phone_redact"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["phone"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        return _replace_whole(value, "PHONE_NUMBER", self.name, ctx, "")


class EmailRedactOperation:
    name: ClassVar[str] = "email_redact"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["email"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        return _replace_whole(value, "EMAIL_ADDRESS", self.name, ctx, "")


class SsnRedactOperation:
    name: ClassVar[str] = "ssn_redact"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["ssn"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        return _replace_whole(value, "US_SSN", self.name, ctx, "")


class MrnRedactOperation:
    name: ClassVar[str] = "mrn_redact"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["mrn"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        return _replace_whole(value, "MRN", self.name, ctx, "")


class UrlRedactOperation:
    name: ClassVar[str] = "url_redact"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["url"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        return _replace_whole(value, "URL", self.name, ctx, "")


class IpRedactOperation:
    name: ClassVar[str] = "ip_redact"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["ip"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        return _replace_whole(value, "IP_ADDRESS", self.name, ctx, "")
