"""Tests for the trivial operations: passthrough, constant_replace, hash_field."""

from __future__ import annotations

import hashlib

from scripts.text_phi.operations.passthrough import (
    ConstantReplaceOperation,
    HashFieldOperation,
    PassthroughOperation,
)


def test_passthrough_returns_value_unchanged(make_ctx):
    op = PassthroughOperation()
    ctx = make_ctx("f", "string")
    new_val, spans = op.apply("anything", ctx)
    assert new_val == "anything"
    assert spans == []


def test_passthrough_empty_value(make_ctx):
    op = PassthroughOperation()
    ctx = make_ctx("f", "string")
    assert op.apply("", ctx) == ("", [])


def test_constant_replace_default_is_X(make_ctx):
    op = ConstantReplaceOperation()
    ctx = make_ctx("f", "string")
    new_val, spans = op.apply("John Smith", ctx)
    assert new_val == "X"
    assert len(spans) == 1
    assert spans[0].entity_type == "CONSTANT_REPLACE"
    assert spans[0].matched_text == "John Smith"


def test_constant_replace_uses_params_value(make_ctx):
    op = ConstantReplaceOperation()
    ctx = make_ctx("f", "string", params={"value": "[NAME]"})
    new_val, spans = op.apply("John Smith", ctx)
    assert new_val == "[NAME]"


def test_constant_replace_empty_returns_empty(make_ctx):
    op = ConstantReplaceOperation()
    ctx = make_ctx("f", "string")
    assert op.apply("", ctx) == ("", [])


def test_hash_field_deterministic(make_ctx):
    op = HashFieldOperation()
    ctx = make_ctx("f", "string")
    v = "12345"
    expected = hashlib.sha256(v.encode()).hexdigest()
    new_val, spans = op.apply(v, ctx)
    assert new_val == expected
    assert len(spans) == 1
    assert spans[0].entity_type == "HASHED"


def test_hash_field_empty_returns_empty(make_ctx):
    op = HashFieldOperation()
    ctx = make_ctx("f", "string")
    assert op.apply("", ctx) == ("", [])
