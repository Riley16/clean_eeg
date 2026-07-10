"""Tests for scripts/text_phi/operations/llm.py — LLM scan operations.

Uses a mocked `httpx.MockTransport` — no live LLM server required.
"""

from __future__ import annotations

import json

import httpx
import pytest

from scripts.text_phi.llm.cache import LLMCache
from scripts.text_phi.llm.client import LLMClient
from scripts.text_phi.llm.config import LLMConfig
from scripts.text_phi.llm.review_report import ReviewReport
from scripts.text_phi.llm.template import PromptRegistry
from scripts.text_phi.operations.llm import (
    LlmDateScanOperation,
    LlmNameScanOperation,
    LlmScanOperation,
)
from scripts.text_phi.records import OperationContext, RecordContext
from scripts.text_phi.schema import FieldSpec, OperationCall, Schema


def _cfg() -> LLMConfig:
    return LLMConfig.from_dict({
        "server_type": "ollama",
        "server_url": "http://localhost:11434/v1",
        "models": {"phi_detector": "qwen2.5:7b-instruct"},
        "max_retries": 0,
    })


def _client_returning(content: str) -> tuple[LLMClient, list[dict]]:
    """Return an LLMClient that always answers with the given content, plus
    a list that will capture requests as they come in."""
    calls: list[dict] = []

    def handler(req):
        calls.append({"body": json.loads(req.content)})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
        })

    return LLMClient(_cfg(), transport=httpx.MockTransport(handler)), calls


def _prompt_registry(tmp_path, prompt_name: str, body: str) -> PromptRegistry:
    (tmp_path / f"{prompt_name}.jinja").write_text(body)
    return PromptRegistry(tmp_path)


def _schema_with(field: str, ops: list) -> Schema:
    raw = {
        "schema_version": "1", "format": "csv",
        "fields": {
            field: {
                "dtype": "string",
                "description": "",
                "operations": ops,
            },
        },
    }
    return Schema.from_dict(raw)


def _make_ctx(
    field: str,
    value: str,
    params: dict,
    client: LLMClient | None,
    registry: PromptRegistry | None,
    cache: LLMCache | None = None,
    report: ReviewReport | None = None,
    record_location: dict | None = None,
) -> OperationContext:
    schema = _schema_with(field, ["passthrough"])
    return OperationContext(
        field_name=field,
        field_spec=schema.fields[field],
        params=params,
        depends_on={},
        record={field: value},
        record_context=RecordContext(),
        schema=schema,
        text_redactor=None,
        llm_client=client,
        llm_cache=cache,
        prompt_registry=registry,
        review_report=report,
        record_location=record_location or {"row": 0},
    )


# ---------- registration ----------

def test_all_three_ops_share_apply_from_base():
    assert LlmDateScanOperation._default_prompt == "date_scan"
    assert LlmNameScanOperation._default_prompt == "name_scan"
    assert LlmScanOperation._default_prompt == "generic_phi"


# ---------- empty value / missing resources ----------

def test_empty_value_passthrough(tmp_path):
    client, _ = _client_returning('{"spans": []}')
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI in: {{ value }}")
    ctx = _make_ctx("note", "", {}, client, reg)
    op = LlmScanOperation()
    assert op.apply("", ctx) == ("", [])


def test_no_client_silent_noop(tmp_path):
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI in: {{ value }}")
    ctx = _make_ctx("note", "some value", {}, None, reg)
    op = LlmScanOperation()
    assert op.apply("some value", ctx) == ("some value", [])


def test_no_prompt_registry_silent_noop():
    client, _ = _client_returning('{"spans": []}')
    ctx = _make_ctx("note", "some value", {}, client, None)
    op = LlmScanOperation()
    assert op.apply("some value", ctx) == ("some value", [])


# ---------- report_only=true (default) ----------

def test_report_only_true_leaves_value_unchanged(tmp_path):
    client, _ = _client_returning(json.dumps({
        "spans": [{"type": "PERSON", "matched_text": "John", "reason": "name"}],
    }))
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI in: {{ value }}")
    report = ReviewReport(tmp_path / "r.json", source="in.csv")
    ctx = _make_ctx("note", "Dr. John was here.", {}, client, reg, report=report)
    op = LlmScanOperation()

    new_val, spans = op.apply("Dr. John was here.", ctx)
    assert new_val == "Dr. John was here."
    assert spans == []


def test_report_only_true_still_writes_report(tmp_path):
    client, _ = _client_returning(json.dumps({
        "spans": [{"type": "PERSON", "matched_text": "John", "reason": "n"}],
    }))
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI in: {{ value }}")
    report = ReviewReport(tmp_path / "r.json", source="in.csv")
    ctx = _make_ctx("note", "Dr. John was here.", {}, client, reg,
                    report=report, record_location={"row": 5})
    op = LlmScanOperation()
    op.apply("Dr. John was here.", ctx)
    report.close()

    data = json.loads((tmp_path / "r.json").read_text())
    assert data["n_findings"] == 1
    e = data["findings"][0]
    assert e["field"] == "note"
    assert e["record_location"] == {"row": 5}
    assert e["value_seen"] == "Dr. John was here."
    assert e["recommended_redacted"] == "Dr. [PERSON] was here."


# ---------- report_only=false ----------

def test_report_only_false_redacts_value(tmp_path):
    client, _ = _client_returning(json.dumps({
        "spans": [
            {"type": "PERSON", "matched_text": "John", "reason": "n"},
            {"type": "DATE", "matched_text": "3/15", "reason": "d"},
        ],
    }))
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI in: {{ value }}")
    report = ReviewReport(tmp_path / "r.json", source="in.csv")
    ctx = _make_ctx(
        "note", "Dr. John saw pt on 3/15.",
        {"report_only": False}, client, reg, report=report,
    )
    op = LlmScanOperation()

    new_val, spans = op.apply("Dr. John saw pt on 3/15.", ctx)
    assert new_val == "Dr. [PERSON] saw pt on [DATE]."
    assert len(spans) == 2
    assert {s.entity_type for s in spans} == {"PERSON", "DATE"}


def test_report_only_false_writes_report_too(tmp_path):
    client, _ = _client_returning(json.dumps({
        "spans": [{"type": "PERSON", "matched_text": "John", "reason": "n"}],
    }))
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI in: {{ value }}")
    report = ReviewReport(tmp_path / "r.json", source="in.csv")
    ctx = _make_ctx(
        "note", "Dr. John was here.",
        {"report_only": False}, client, reg, report=report,
    )
    op = LlmScanOperation()
    op.apply("Dr. John was here.", ctx)
    report.close()

    data = json.loads((tmp_path / "r.json").read_text())
    assert data["n_findings"] == 1


# ---------- context template resolution ----------

def test_context_template_variables_resolved(tmp_path):
    client, calls = _client_returning('{"spans": []}')
    reg = _prompt_registry(
        tmp_path, "generic_phi",
        "You are checking for PHI. Patient={{ context.patient_name }}. "
        "Value={{ value }}",
    )
    ctx = _make_ctx(
        "note", "some text",
        {"context": {"patient_name": "{{ record.subject_name }}"}},
        client, reg,
    )
    ctx.record["subject_name"] = "John Smith"
    op = LlmScanOperation()
    op.apply("some text", ctx)

    sent_msg = calls[0]["body"]["messages"][0]["content"]
    assert "John Smith" in sent_msg
    assert "Value=some text" in sent_msg


# ---------- caching ----------

def test_cache_hit_avoids_llm_call(tmp_path):
    client, calls = _client_returning(json.dumps({
        "spans": [{"type": "PERSON", "matched_text": "John", "reason": "n"}],
    }))
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI: {{ value }}")
    cache = LLMCache(tmp_path / "c.sqlite")
    ctx1 = _make_ctx("note", "John was here", {}, client, reg, cache=cache)
    ctx2 = _make_ctx("note", "John was here", {}, client, reg, cache=cache)

    op = LlmScanOperation()
    op.apply("John was here", ctx1)
    assert len(calls) == 1
    op.apply("John was here", ctx2)
    # Second invocation should hit the cache.
    assert len(calls) == 1
    cache.close()


def test_different_seed_yields_different_cache_key(tmp_path, monkeypatch):
    # Confirm cache key differs when the seed changes by inspecting the cache
    # size after two runs with different seeds.
    client, calls = _client_returning(json.dumps({"spans": []}))
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI: {{ value }}")
    cache = LLMCache(tmp_path / "c.sqlite")

    ctx1 = _make_ctx("note", "value", {}, client, reg, cache=cache)
    LlmScanOperation().apply("value", ctx1)
    assert cache.size() == 1

    # Now use a different seed by mutating the client's config.
    client2, _ = _client_returning('{"spans": []}')
    client2.config.__dict__["seed"] = 999  # frozen dataclass — bypass
    ctx2 = _make_ctx("note", "value", {}, client2, reg, cache=cache)
    LlmScanOperation().apply("value", ctx2)
    assert cache.size() == 2
    cache.close()


# ---------- wrapper ops ----------

def test_wrappers_use_their_own_prompt(tmp_path):
    client, calls = _client_returning('{"spans": []}')
    reg = _prompt_registry(tmp_path, "date_scan", "DATE PROMPT for {{ value }}")
    _prompt_registry(tmp_path, "name_scan", "NAME PROMPT for {{ value }}")

    ctx = _make_ctx("note", "text", {}, client, reg)
    LlmDateScanOperation().apply("text", ctx)
    assert "DATE PROMPT" in calls[-1]["body"]["messages"][0]["content"]

    LlmNameScanOperation().apply("text", ctx)
    assert "NAME PROMPT" in calls[-1]["body"]["messages"][0]["content"]


def test_wrapper_prompt_can_be_overridden_via_params(tmp_path):
    client, calls = _client_returning('{"spans": []}')
    reg = _prompt_registry(tmp_path, "custom", "CUSTOM PROMPT: {{ value }}")
    ctx = _make_ctx("note", "text", {"prompt": "custom"}, client, reg)
    LlmDateScanOperation().apply("text", ctx)
    assert "CUSTOM PROMPT" in calls[0]["body"]["messages"][0]["content"]


# ---------- malformed responses ----------

def test_malformed_llm_content_raises(tmp_path):
    from scripts.text_phi.llm.response import LlmResponseError
    client, _ = _client_returning("this is not valid json")
    reg = _prompt_registry(tmp_path, "generic_phi", "detect PHI: {{ value }}")
    ctx = _make_ctx("note", "value", {}, client, reg)
    with pytest.raises(LlmResponseError):
        LlmScanOperation().apply("value", ctx)
