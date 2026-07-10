"""Tests for scripts/text_phi/llm/client.py.

Every test uses `httpx.MockTransport` — no live LLM server required.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from scripts.text_phi.llm.client import LLMClient, LLMHTTPError
from scripts.text_phi.llm.config import LLMConfig, ServerType


def _cfg(server_type: str = "ollama", **overrides) -> LLMConfig:
    raw = {
        "server_type": server_type,
        "server_url": "http://localhost:11434/v1",
        "models": {"phi_detector": "qwen2.5:7b-instruct"},
        "max_retries": 2,
        "timeout_seconds": 5,
        **overrides,
    }
    return LLMConfig.from_dict(raw)


def _ok_response(content: str = '{"spans": []}') -> dict[str, Any]:
    return {
        "id": "chatcmpl-abc",
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


class _Recorder:
    """Captures every request the client makes."""
    def __init__(self):
        self.requests: list[dict[str, Any]] = []

    def transport(self, response_factory):
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append({
                "method": request.method,
                "url": str(request.url),
                "body": json.loads(request.content),
                "headers": dict(request.headers),
            })
            return response_factory()
        return httpx.MockTransport(handler)


# ---------- happy path ----------

def test_chat_ok_returns_json():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(), transport=t) as c:
        got = c.chat([{"role": "user", "content": "hi"}])
    assert got["choices"][0]["message"]["content"] == '{"spans": []}'


def test_chat_sends_seed_and_temperature_from_config():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(seed=7, temperature=0.25), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}])
    body = rec.requests[0]["body"]
    assert body["seed"] == 7
    assert body["temperature"] == 0.25


def test_chat_per_call_overrides_config_defaults():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(seed=42, temperature=0.0), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}], seed=99, temperature=1.0)
    body = rec.requests[0]["body"]
    assert body["seed"] == 99
    assert body["temperature"] == 1.0


def test_chat_resolves_model_from_hint():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}])
    assert rec.requests[0]["body"]["model"] == "qwen2.5:7b-instruct"


def test_chat_hits_chat_completions_path():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}])
    assert rec.requests[0]["url"].endswith("/chat/completions")


# ---------- structured output per server type ----------

_SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


def test_structured_output_ollama_uses_format():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg("ollama"), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}], response_schema=_SCHEMA)
    body = rec.requests[0]["body"]
    assert body["format"] == _SCHEMA
    assert "extra_body" not in body
    assert "response_format" not in body


def test_structured_output_vllm_uses_guided_json():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg("vllm"), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}], response_schema=_SCHEMA)
    body = rec.requests[0]["body"]
    assert body["extra_body"] == {"guided_json": _SCHEMA}
    assert "format" not in body


@pytest.mark.parametrize("server_type", ["openai", "lm_studio"])
def test_structured_output_openai_style_uses_response_format(server_type):
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(server_type), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}], response_schema=_SCHEMA)
    body = rec.requests[0]["body"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == _SCHEMA


def test_no_schema_means_no_structured_output_field():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg("vllm"), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}], response_schema=None)
    body = rec.requests[0]["body"]
    assert "format" not in body and "extra_body" not in body \
        and "response_format" not in body


# ---------- auth ----------

def test_api_key_sent_as_bearer():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(api_key="secret"), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}])
    assert rec.requests[0]["headers"]["authorization"] == "Bearer secret"


def test_no_api_key_no_auth_header():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}])
    assert "authorization" not in rec.requests[0]["headers"]


# ---------- retries ----------

def test_500_retries_then_succeeds():
    calls = {"n": 0}

    def response_factory():
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="oops")
        return httpx.Response(200, json=_ok_response())

    def handler(req):
        return response_factory()

    t = httpx.MockTransport(handler)
    # max_retries=2 → allows 3 total attempts
    with LLMClient(_cfg(max_retries=2), transport=t) as c:
        got = c.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 3
    assert got["choices"][0]["message"]["content"] == '{"spans": []}'


def test_5xx_after_max_retries_raises():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    t = httpx.MockTransport(handler)
    with LLMClient(_cfg(max_retries=1), transport=t) as c:
        with pytest.raises(LLMHTTPError, match="503"):
            c.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2  # max_retries=1 → 2 attempts total


def test_400_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    t = httpx.MockTransport(handler)
    with LLMClient(_cfg(max_retries=5), transport=t) as c:
        with pytest.raises(LLMHTTPError, match="400"):
            c.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


def test_429_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=_ok_response())

    t = httpx.MockTransport(handler)
    with LLMClient(_cfg(max_retries=2), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2


def test_timeout_retried_then_raises(monkeypatch):
    # Force exponential-backoff sleep to be a no-op so tests are fast.
    from scripts.text_phi.llm import client as client_mod
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)

    def handler(req):
        raise httpx.TimeoutException("slow")

    t = httpx.MockTransport(handler)
    with LLMClient(_cfg(max_retries=1), transport=t) as c:
        with pytest.raises(httpx.TimeoutException):
            c.chat([{"role": "user", "content": "hi"}])


# ---------- max_tokens ----------

def test_max_tokens_omitted_by_default():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}])
    assert "max_tokens" not in rec.requests[0]["body"]


def test_max_tokens_included_when_supplied():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    with LLMClient(_cfg(), transport=t) as c:
        c.chat([{"role": "user", "content": "hi"}], max_tokens=256)
    assert rec.requests[0]["body"]["max_tokens"] == 256


# ---------- messages passthrough ----------

def test_messages_passed_verbatim():
    rec = _Recorder()
    t = rec.transport(lambda: httpx.Response(200, json=_ok_response()))
    msgs = [
        {"role": "system", "content": "you detect PHI"},
        {"role": "user", "content": "text here"},
    ]
    with LLMClient(_cfg(), transport=t) as c:
        c.chat(msgs)
    assert rec.requests[0]["body"]["messages"] == msgs
