"""Tests for scripts/text_phi/llm/response.py."""

from __future__ import annotations

import json

import pytest

from scripts.text_phi.llm.response import (
    LlmResponseError,
    LlmSpan,
    PHI_SPAN_SCHEMA,
    PHI_SPAN_TYPES,
    apply_spans_labeled,
    apply_spans_literal,
    parse_response,
    resolve_spans,
)


# ---------- schema shape ----------

def test_phi_span_schema_declares_types():
    types = PHI_SPAN_SCHEMA["properties"]["spans"]["items"]["properties"]["type"]["enum"]
    assert set(types) == set(PHI_SPAN_TYPES)


# ---------- parse_response ----------

def test_parse_response_ok():
    body = json.dumps({"spans": [
        {"type": "PERSON", "matched_text": "John", "reason": "first name"},
        {"type": "DATE", "matched_text": "3/15", "reason": "specific date"},
    ]})
    spans = parse_response(body)
    assert len(spans) == 2
    assert spans[0].type == "PERSON"
    assert spans[0].matched_text == "John"
    assert spans[1].type == "DATE"


def test_parse_response_empty_spans_ok():
    body = json.dumps({"spans": []})
    assert parse_response(body) == []


def test_parse_response_not_json_raises():
    with pytest.raises(LlmResponseError, match="not JSON"):
        parse_response("not json at all")


def test_parse_response_missing_spans_raises():
    with pytest.raises(LlmResponseError, match="'spans'"):
        parse_response('{"other": "thing"}')


def test_parse_response_spans_not_list_raises():
    with pytest.raises(LlmResponseError, match="must be a list"):
        parse_response('{"spans": "oops"}')


def test_parse_response_span_missing_type_raises():
    body = json.dumps({"spans": [{"matched_text": "x", "reason": "y"}]})
    with pytest.raises(LlmResponseError, match="type"):
        parse_response(body)


def test_parse_response_span_type_wrong_kind_raises():
    body = json.dumps({"spans": [{"type": 42, "matched_text": "x", "reason": "y"}]})
    with pytest.raises(LlmResponseError, match="type"):
        parse_response(body)


# ---------- resolve_spans ----------

def test_resolve_span_found():
    v = "The patient John was here."
    r = resolve_spans(v, [LlmSpan("PERSON", "John", "name")])
    assert len(r) == 1
    assert r[0].start == 12
    assert r[0].end == 16
    assert r[0].ambiguous is False


def test_resolve_span_flags_ambiguous_when_multiple_hits():
    v = "John saw John yesterday."
    r = resolve_spans(v, [LlmSpan("PERSON", "John", "name")])
    assert len(r) == 1
    assert r[0].ambiguous is True
    # First occurrence wins.
    assert r[0].start == 0


def test_resolve_span_missing_matched_text_dropped():
    v = "Some text without the phantom."
    r = resolve_spans(v, [LlmSpan("PERSON", "Alice", "hallucinated")])
    assert r == []


def test_resolve_span_empty_matched_text_dropped():
    r = resolve_spans("anything", [LlmSpan("DATE", "", "empty")])
    assert r == []


def test_resolve_span_multiple_findings_all_resolved():
    v = "Dr. John saw the patient on 3/15."
    spans = [
        LlmSpan("PERSON", "John", "n"),
        LlmSpan("DATE", "3/15", "d"),
    ]
    r = resolve_spans(v, spans)
    assert len(r) == 2
    assert v[r[0].start:r[0].end] == "John"
    assert v[r[1].start:r[1].end] == "3/15"


# ---------- application ----------

def test_apply_spans_labeled():
    v = "Dr. John saw pt on 3/15."
    r = resolve_spans(v, [
        LlmSpan("PERSON", "John", "n"),
        LlmSpan("DATE", "3/15", "d"),
    ])
    out = apply_spans_labeled(v, r)
    assert out == "Dr. [PERSON] saw pt on [DATE]."


def test_apply_spans_literal_default_X():
    v = "Contact John."
    r = resolve_spans(v, [LlmSpan("PERSON", "John", "n")])
    assert apply_spans_literal(v, r) == "Contact X."


def test_apply_spans_literal_custom_token():
    v = "Contact John."
    r = resolve_spans(v, [LlmSpan("PERSON", "John", "n")])
    assert apply_spans_literal(v, r, token="[REDACTED]") == "Contact [REDACTED]."


def test_apply_spans_right_to_left_preserves_offsets():
    """Two spans in one value, both applied — offsets stay correct."""
    v = "a b c d"  # 7 chars
    r = resolve_spans(v, [
        LlmSpan("DATE", "a", "first"),
        LlmSpan("PERSON", "d", "last"),
    ])
    assert apply_spans_labeled(v, r) == "[DATE] b c [PERSON]"


def test_apply_spans_empty_list_returns_original():
    v = "unchanged"
    assert apply_spans_labeled(v, []) == v
    assert apply_spans_literal(v, []) == v
