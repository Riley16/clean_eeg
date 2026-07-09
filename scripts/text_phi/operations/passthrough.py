"""Trivial operations: passthrough, constant_replace, hash_field."""

from __future__ import annotations

import hashlib
from typing import ClassVar

from ..records import OperationContext
from ..redactor import RedactionSpan


class PassthroughOperation:
    name: ClassVar[str] = "passthrough"
    allowed_dtypes: ClassVar[frozenset[str] | None] = None
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        return value, []


class ConstantReplaceOperation:
    """Replace the field value with a fixed string. Default replacement is
    `"X"`; override via `params={"value": "[REDACTED]"}` in the schema."""
    name: ClassVar[str] = "constant_replace"
    allowed_dtypes: ClassVar[frozenset[str] | None] = None
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        replacement = ctx.params.get("value", "X")
        span = RedactionSpan(
            start=0, end=len(value),
            entity_type="CONSTANT_REPLACE",
            score=1.0,
            recognizer=self.name,
            matched_text=value,
        )
        return replacement, [span]


class HashFieldOperation:
    """Replace the value with its SHA-256 hex digest. Useful for identifiers
    that should be pseudonymized rather than dropped."""
    name: ClassVar[str] = "hash_field"
    allowed_dtypes: ClassVar[frozenset[str] | None] = None
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset()

    def apply(self, value: str, ctx: OperationContext) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        span = RedactionSpan(
            start=0, end=len(value),
            entity_type="HASHED",
            score=1.0,
            recognizer=self.name,
            matched_text=value,
        )
        return digest, [span]
