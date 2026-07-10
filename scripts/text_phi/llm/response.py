"""PHI response schema + offset resolution.

The LLM returns `{spans: [{type, matched_text, reason}, ...]}` (no offsets,
because LLMs are unreliable at character offsets). This module resolves
each `matched_text` back to a (start, end) window inside the source value
and flags ambiguous matches (multiple occurrences).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


PHI_SPAN_TYPES: tuple[str, ...] = (
    "DATE", "PERSON", "PHONE", "EMAIL", "ADDRESS", "MRN", "LOCATION",
    "AGE", "OTHER_PHI",
)


PHI_SPAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(PHI_SPAN_TYPES)},
                    "matched_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["type", "matched_text", "reason"],
            },
        },
    },
    "required": ["spans"],
}


@dataclass(frozen=True)
class LlmSpan:
    """A raw span as returned by the LLM (before offset resolution)."""
    type: str
    matched_text: str
    reason: str


@dataclass(frozen=True)
class ResolvedSpan:
    """An LLM span with (start, end) resolved against the source value."""
    start: int
    end: int
    type: str
    matched_text: str
    reason: str
    ambiguous: bool  # True if matched_text appears >1 time in the value


class LlmResponseError(ValueError):
    """The LLM response body was not the shape we expected."""


def parse_response(content: str) -> list[LlmSpan]:
    """Parse the model's raw response `content` string into `LlmSpan`s.

    Tolerates responses wrapped in extra whitespace / trailing text but
    requires the leading JSON object to match `PHI_SPAN_SCHEMA` shape.
    """
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        raise LlmResponseError(f"response is not JSON: {e}") from e
    if not isinstance(obj, dict) or "spans" not in obj:
        raise LlmResponseError("response missing 'spans' array")
    raw_spans = obj["spans"]
    if not isinstance(raw_spans, list):
        raise LlmResponseError("'spans' must be a list")
    out: list[LlmSpan] = []
    for i, s in enumerate(raw_spans):
        if not isinstance(s, dict):
            raise LlmResponseError(f"spans[{i}] must be an object")
        for k in ("type", "matched_text", "reason"):
            if k not in s or not isinstance(s[k], str):
                raise LlmResponseError(
                    f"spans[{i}] missing string field {k!r}"
                )
        out.append(LlmSpan(
            type=s["type"], matched_text=s["matched_text"], reason=s["reason"],
        ))
    return out


def resolve_spans(value: str, spans: list[LlmSpan]) -> list[ResolvedSpan]:
    """Locate each span's `matched_text` inside `value` and attach offsets.

    Rules:
      * Empty `matched_text` is dropped (LLM leftover).
      * First occurrence wins.
      * Multiple occurrences → `ambiguous=True` flag.
      * `matched_text` not in `value` → span dropped (LLM hallucination).
    """
    resolved: list[ResolvedSpan] = []
    for s in spans:
        m = s.matched_text
        if not m:
            continue
        start = value.find(m)
        if start < 0:
            continue
        # Check for additional occurrences.
        ambiguous = value.find(m, start + 1) >= 0
        resolved.append(ResolvedSpan(
            start=start,
            end=start + len(m),
            type=s.type,
            matched_text=m,
            reason=s.reason,
            ambiguous=ambiguous,
        ))
    return resolved


def apply_spans_labeled(value: str, spans: list[ResolvedSpan]) -> str:
    """Return `value` with each span replaced by `[<TYPE>]`."""
    # Sort so we can replace right-to-left without invalidating earlier offsets.
    for s in sorted(spans, key=lambda x: x.start, reverse=True):
        value = value[:s.start] + f"[{s.type}]" + value[s.end:]
    return value


def apply_spans_literal(value: str, spans: list[ResolvedSpan], token: str = "X") -> str:
    """Return `value` with each span replaced by `token`."""
    for s in sorted(spans, key=lambda x: x.start, reverse=True):
        value = value[:s.start] + token + value[s.end:]
    return value
