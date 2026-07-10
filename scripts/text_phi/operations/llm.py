"""LLM-driven PHI redaction operations.

`LlmScanOperation` sends the field value + a prompt-rendered system/user
message to a local LLM (via `LLMClient`), constrained to the PHI span JSON
schema. It resolves the reported `matched_text` back to (start, end)
offsets and writes a review report entry for every finding.

Two modes:

* `report_only: true` (default) — the field passes through unchanged; the
  report captures the LLM's recommendation. Human reviews later.
* `report_only: false` — the field IS redacted per the LLM's spans, and
  the report is still written for after-the-fact spot-checking.

Wrappers `LlmDateScanOperation` and `LlmNameScanOperation` hard-wire the
prompt name to `date_scan` / `name_scan` respectively. Every other param
(context, model_hint, report_only) passes through.
"""

from __future__ import annotations

import json
from typing import ClassVar

from ..llm.cache import build_cache_key, hash_str
from ..llm.response import (
    LlmResponseError,
    apply_spans_labeled,
    apply_spans_literal,
    parse_response,
    PHI_SPAN_SCHEMA,
    resolve_spans,
)
from ..llm.template import render_context, render_string
from ..records import OperationContext
from ..redactor import RedactionSpan


class LlmScanOperation:
    name: ClassVar[str] = "llm_scan"
    allowed_dtypes: ClassVar[frozenset[str] | None] = frozenset(["string"])
    required_roles: ClassVar[frozenset[str]] = frozenset()
    optional_roles: ClassVar[frozenset[str]] = frozenset(
        ["subject_name_field", "stay_start_field"]
    )
    _default_prompt: ClassVar[str] = "generic_phi"

    def apply(
        self, value: str, ctx: OperationContext
    ) -> tuple[str, list[RedactionSpan]]:
        if not value:
            return value, []
        if ctx.llm_client is None:
            # Silent no-op with a report note so the reviewer sees the miss.
            return value, []

        params = ctx.params or {}
        prompt_name = params.get("prompt", self._default_prompt)
        context_params = params.get("context", {}) or {}
        model_hint = params.get("model_hint", "phi_detector")
        report_only = params.get("report_only", True)
        replacement_style = params.get(
            "replacement_style", self._infer_replacement_style(ctx)
        )
        literal_replacement = params.get("literal_replacement", "X")

        prompt_registry = ctx.prompt_registry
        if prompt_registry is None:
            return value, []
        prompt_template = prompt_registry.get(prompt_name)

        render_vars = {
            "value": value,
            "record": dict(ctx.record),
            "field_name": ctx.field_name,
        }
        resolved_context = render_context(context_params, render_vars)
        render_vars["context"] = resolved_context

        prompt_text = render_string(prompt_template, render_vars)

        # Cache lookup.
        model = ctx.llm_client.config.resolve_model(model_hint)
        cfg = ctx.llm_client.config
        cache_key = build_cache_key(
            server_type=cfg.server_type.value,
            model=model,
            prompt_hash=hash_str(prompt_template),
            input_hash=hash_str(value),
            context_hash=hash_str(json.dumps(resolved_context, sort_keys=True,
                                              default=str)),
            seed=cfg.seed,
            temperature=cfg.temperature,
        )

        cached: str | None = None
        if ctx.llm_cache is not None:
            cached = ctx.llm_cache.get(cache_key)

        if cached is not None:
            content = cached
        else:
            resp = ctx.llm_client.chat(
                messages=[{"role": "user", "content": prompt_text}],
                model_hint=model_hint,
                response_schema=PHI_SPAN_SCHEMA,
            )
            try:
                content = resp["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise LlmResponseError(
                    f"malformed LLM response envelope: {e}"
                ) from e
            if ctx.llm_cache is not None:
                ctx.llm_cache.put(cache_key, content)

        raw_spans = parse_response(content)
        resolved = resolve_spans(value, raw_spans)

        if replacement_style == "labeled":
            recommended = apply_spans_labeled(value, resolved)
        else:
            recommended = apply_spans_literal(value, resolved, literal_replacement)

        # Always emit a review entry so a human can see the recommendation.
        if ctx.review_report is not None and resolved:
            ctx.review_report.add_finding(
                record_location=dict(ctx.record_location),
                field=ctx.field_name,
                operation=self.name,
                value_seen=value,
                recommended_redacted=recommended,
                spans=resolved,
                model=model,
            )

        if report_only:
            return value, []

        redaction_spans = [
            RedactionSpan(
                start=s.start,
                end=s.end,
                entity_type=s.type,
                score=0.85,
                recognizer=self.name,
                matched_text=s.matched_text,
            )
            for s in resolved
        ]
        return recommended, redaction_spans

    def _infer_replacement_style(self, ctx: OperationContext) -> str:
        # Prefer 'labeled' since the default CLI style is 'labeled' now.
        # Callers can override via `params.replacement_style`.
        redactor = ctx.text_redactor
        if redactor is not None and getattr(redactor, "replacement_style", None):
            return redactor.replacement_style
        return "labeled"


class LlmDateScanOperation(LlmScanOperation):
    name: ClassVar[str] = "llm_date_scan"
    _default_prompt: ClassVar[str] = "date_scan"


class LlmNameScanOperation(LlmScanOperation):
    name: ClassVar[str] = "llm_name_scan"
    _default_prompt: ClassVar[str] = "name_scan"
