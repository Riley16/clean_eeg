"""Per-record LLM review pass — runs *after* the per-field redaction loop.

For each record, we concatenate every already-redacted string field with
`---BEGIN <field>---` / `---END <field>---` delimiters, hand the blob to
the LLM, and ask "does anything still look like PHI?". Findings go to the
`record_flags` bucket of the review report. This pass never modifies the
redacted CSV — it is 100% audit / human-review.

Field-level attribution: after resolving span offsets in the concatenated
blob, we identify which field's `[BEGIN, END]` window contains each span
and record that field name on the flag.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from ..records import Record
from ..schema import Schema
from .cache import LLMCache, build_cache_key, hash_str
from .client import LLMClient
from .response import (
    LlmResponseError,
    PHI_SPAN_SCHEMA,
    ResolvedSpan,
    apply_spans_labeled,
    parse_response,
    resolve_spans,
)
from .review_report import ReviewReport
from .template import PromptRegistry, render_string


DELIMITER_START_TEMPLATE = "\n---BEGIN {field}---\n"
DELIMITER_END_TEMPLATE = "\n---END {field}---\n"


class RecordReviewer:
    def __init__(
        self,
        client: LLMClient,
        prompt_registry: PromptRegistry,
        review_report: ReviewReport,
        cache: LLMCache | None = None,
        prompt_name: str = "record_review",
        model_hint: str = "record_reviewer",
    ):
        self.client = client
        self.prompt_registry = prompt_registry
        self.review_report = review_report
        self.cache = cache
        self.prompt_name = prompt_name
        self.model_hint = model_hint

    # ---------- top-level ----------

    def review_records(self, records: list[Record], schema: Schema) -> None:
        for r in records:
            self.review_record(r, schema)

    def review_record(self, record: Record, schema: Schema) -> None:
        concatenated, field_ranges = self._concatenate(record, schema)
        if not concatenated:
            return

        content = self._llm_scan(concatenated)
        try:
            raw_spans = parse_response(content)
        except LlmResponseError:
            return
        resolved = resolve_spans(concatenated, raw_spans)
        if not resolved:
            return

        # Tag each span with the field it lives in (via ---BEGIN/---END windows).
        tagged = [
            replace(s, reason=self._enrich_reason(s, field_ranges))
            for s in resolved
        ]

        recommended = apply_spans_labeled(concatenated, tagged)
        self.review_report.add_record_flag(
            record_location=dict(record.location),
            concatenated_value=concatenated,
            recommended_redacted=recommended,
            spans=tagged,
            model=self.client.config.resolve_model(self.model_hint),
        )

    # ---------- concatenation ----------

    def _concatenate(
        self, record: Record, schema: Schema
    ) -> tuple[str, dict[str, tuple[int, int]]]:
        """Build the concatenated blob and record the (start, end) window of
        each field's *value body* (excluding delimiters)."""
        parts: list[str] = []
        field_ranges: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name, value in record.fields.items():
            if name not in schema.fields:
                continue
            spec = schema.fields[name]
            if spec.dtype != "string":
                continue
            if not value:
                continue
            begin = DELIMITER_START_TEMPLATE.format(field=name)
            end = DELIMITER_END_TEMPLATE.format(field=name)
            parts.append(begin + value + end)
            body_start = cursor + len(begin)
            body_end = body_start + len(value)
            field_ranges[name] = (body_start, body_end)
            cursor += len(begin) + len(value) + len(end)
        return "".join(parts), field_ranges

    # ---------- LLM call w/ cache ----------

    def _llm_scan(self, concatenated: str) -> str:
        prompt_template = self.prompt_registry.get(self.prompt_name)
        prompt_text = render_string(prompt_template, {"value": concatenated})

        cfg = self.client.config
        model = self.client.config.resolve_model(self.model_hint)
        cache_key = build_cache_key(
            server_type=cfg.server_type.value,
            model=model,
            prompt_hash=hash_str(prompt_template),
            input_hash=hash_str(concatenated),
            context_hash=hash_str("{}"),
            seed=cfg.seed,
            temperature=cfg.temperature,
        )
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        resp = self.client.chat(
            messages=[{"role": "user", "content": prompt_text}],
            model_hint=self.model_hint,
            response_schema=PHI_SPAN_SCHEMA,
        )
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LlmResponseError(f"malformed LLM envelope: {e}") from e

        if self.cache is not None:
            self.cache.put(cache_key, content)
        return content

    # ---------- reason enrichment ----------

    @staticmethod
    def _enrich_reason(
        span: ResolvedSpan, field_ranges: dict[str, tuple[int, int]]
    ) -> str:
        containing = None
        for field_name, (start, end) in field_ranges.items():
            if start <= span.start and span.end <= end:
                containing = field_name
                break
        if containing is None:
            return span.reason
        return f"[field={containing}] {span.reason}"
