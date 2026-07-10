"""Tests for scripts/text_phi/llm/record_reviewer.py."""

from __future__ import annotations

import json

import httpx

from scripts.text_phi.llm.cache import LLMCache
from scripts.text_phi.llm.client import LLMClient
from scripts.text_phi.llm.config import LLMConfig
from scripts.text_phi.llm.record_reviewer import RecordReviewer
from scripts.text_phi.llm.review_report import ReviewReport
from scripts.text_phi.llm.template import PromptRegistry
from scripts.text_phi.records import Record
from scripts.text_phi.schema import Schema


def _cfg() -> LLMConfig:
    return LLMConfig.from_dict({
        "server_type": "ollama",
        "server_url": "http://localhost:11434/v1",
        "models": {"record_reviewer": "qwen2.5:7b-instruct"},
        "max_retries": 0,
    })


def _client_returning(content: str) -> tuple[LLMClient, list[dict]]:
    calls: list[dict] = []

    def handler(req):
        calls.append({"body": json.loads(req.content)})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
        })

    return LLMClient(_cfg(), transport=httpx.MockTransport(handler)), calls


def _registry(tmp_path, body: str = "review this: {{ value }}") -> PromptRegistry:
    (tmp_path / "record_review.jinja").write_text(body)
    return PromptRegistry(tmp_path)


def _schema(fields: list[str]) -> Schema:
    return Schema.from_dict({
        "schema_version": "1", "format": "csv",
        "fields": {
            f: {"dtype": "string", "operations": ["passthrough"]}
            for f in fields
        },
    })


# ---------- concatenation ----------

def test_concatenate_wraps_each_field(tmp_path):
    client, _ = _client_returning('{"spans": []}')
    reg = _registry(tmp_path)
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report)
        record = Record(
            location={"row": 0},
            fields={"note": "hi", "channel": "LFPx1"},
        )
        blob, ranges = reviewer._concatenate(record, _schema(["note", "channel"]))
        assert "---BEGIN note---" in blob
        assert "---END note---" in blob
        assert "hi" in blob
        assert "LFPx1" in blob
        # Ranges point to the body of each field.
        s, e = ranges["note"]
        assert blob[s:e] == "hi"
        s, e = ranges["channel"]
        assert blob[s:e] == "LFPx1"


def test_concatenate_skips_empty_fields(tmp_path):
    client, _ = _client_returning('{"spans": []}')
    reg = _registry(tmp_path)
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report)
        record = Record(location={"row": 0}, fields={"a": "", "b": "text"})
        blob, ranges = reviewer._concatenate(record, _schema(["a", "b"]))
        assert "---BEGIN a---" not in blob
        assert "text" in blob
        assert "a" not in ranges
        assert "b" in ranges


def test_concatenate_skips_non_string_dtypes(tmp_path):
    client, _ = _client_returning('{"spans": []}')
    reg = _registry(tmp_path)
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report)
        schema = Schema.from_dict({
            "schema_version": "1", "format": "csv",
            "fields": {
                "note": {"dtype": "string", "operations": ["passthrough"]},
                "age": {"dtype": "integer", "operations": ["passthrough"]},
            },
        })
        record = Record(location={"row": 0}, fields={"note": "hi", "age": "42"})
        blob, ranges = reviewer._concatenate(record, schema)
        assert "42" not in blob
        assert "age" not in ranges


# ---------- review_record ----------

def test_review_record_writes_flag_when_spans_found(tmp_path):
    client, _ = _client_returning(json.dumps({
        "spans": [{"type": "PERSON", "matched_text": "John", "reason": "n"}],
    }))
    reg = _registry(tmp_path)
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report)
        record = Record(location={"row": 3}, fields={"note": "Dr. John was here"})
        reviewer.review_record(record, _schema(["note"]))
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["n_record_flags"] == 1
    flag = data["record_flags"][0]
    assert flag["record_location"] == {"row": 3}
    # Field attribution appears in the reason.
    assert "[field=note]" in flag["spans"][0]["reason"]


def test_review_record_no_findings_no_flag(tmp_path):
    client, _ = _client_returning('{"spans": []}')
    reg = _registry(tmp_path)
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report)
        record = Record(location={"row": 3}, fields={"note": "safe text"})
        reviewer.review_record(record, _schema(["note"]))
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["n_record_flags"] == 0


def test_review_record_empty_record_skipped(tmp_path):
    client, calls = _client_returning('{"spans": []}')
    reg = _registry(tmp_path)
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report)
        record = Record(location={"row": 3}, fields={"note": ""})
        reviewer.review_record(record, _schema(["note"]))
    # Never called the LLM.
    assert calls == []


def test_review_record_field_attribution_for_multiple_fields(tmp_path):
    """Spans in different fields get correctly attributed."""
    client, _ = _client_returning(json.dumps({
        "spans": [
            {"type": "PERSON", "matched_text": "Alice", "reason": "n"},
            {"type": "DATE", "matched_text": "3/15", "reason": "d"},
        ],
    }))
    reg = _registry(tmp_path)
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report)
        record = Record(
            location={"row": 3},
            fields={"note1": "Dr. Alice", "note2": "seen on 3/15"},
        )
        reviewer.review_record(record, _schema(["note1", "note2"]))
    flag = json.loads((tmp_path / "r.json").read_text())["record_flags"][0]
    reasons = [s["reason"] for s in flag["spans"]]
    assert any("[field=note1]" in r for r in reasons)
    assert any("[field=note2]" in r for r in reasons)


# ---------- caching ----------

def test_cache_hit_avoids_llm_call(tmp_path):
    client, calls = _client_returning(json.dumps({"spans": []}))
    reg = _registry(tmp_path)
    cache = LLMCache(tmp_path / "c.sqlite")
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report, cache=cache)
        record = Record(location={"row": 0}, fields={"note": "text"})
        reviewer.review_record(record, _schema(["note"]))
        assert len(calls) == 1
        reviewer.review_record(record, _schema(["note"]))
        # Cached — no additional call.
        assert len(calls) == 1
    cache.close()


# ---------- multi-record ----------

def test_review_records_iterates(tmp_path):
    client, calls = _client_returning('{"spans": []}')
    reg = _registry(tmp_path)
    with ReviewReport(tmp_path / "r.json", source="in.csv") as report:
        reviewer = RecordReviewer(client, reg, report)
        records = [
            Record(location={"row": i}, fields={"note": f"note {i}"})
            for i in range(3)
        ]
        reviewer.review_records(records, _schema(["note"]))
    assert len(calls) == 3
