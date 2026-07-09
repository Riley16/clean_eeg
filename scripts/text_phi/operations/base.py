"""Operation protocol — the interface every redaction op implements.

An Operation is a *pure function* (value, ctx) → (new_value, spans). It
declares which dtypes it can apply to (`allowed_dtypes`, None = any) and
which `depends_on` role names it requires and optionally consumes.

The processing loop in records.py resolves each field's operations in
order, threading the current value through and collecting spans.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

from ..records import OperationContext
from ..redactor import RedactionSpan


class Operation(Protocol):
    name: ClassVar[str]
    allowed_dtypes: ClassVar[frozenset[str] | None]
    required_roles: ClassVar[frozenset[str]]
    optional_roles: ClassVar[frozenset[str]]

    def apply(
        self, value: str, ctx: OperationContext
    ) -> tuple[str, list[RedactionSpan]]:
        ...
