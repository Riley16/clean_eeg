"""Tests for scripts/text_phi/llm/config.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.text_phi.llm.config import ConfigError, LLMConfig, ServerType


def _minimal_raw() -> dict:
    return {
        "server_type": "ollama",
        "server_url": "http://localhost:11434/v1",
        "models": {"phi_detector": "qwen2.5:7b-instruct"},
    }


def test_loads_minimal_config():
    c = LLMConfig.from_dict(_minimal_raw())
    assert c.server_type == ServerType.OLLAMA
    assert c.server_url == "http://localhost:11434/v1"
    assert c.models == {"phi_detector": "qwen2.5:7b-instruct"}
    assert c.seed == 42
    assert c.temperature == 0.0
    assert c.max_retries == 3
    assert c.api_key is None


def test_loads_full_config(tmp_path):
    raw = {
        **_minimal_raw(),
        "cache_path": str(tmp_path / "cache.sqlite"),
        "seed": 7,
        "temperature": 0.3,
        "timeout_seconds": 30,
        "max_retries": 5,
        "api_key": "abc123",
    }
    c = LLMConfig.from_dict(raw)
    assert c.cache_path == tmp_path / "cache.sqlite"
    assert c.seed == 7
    assert c.temperature == 0.3
    assert c.timeout_seconds == 30.0
    assert c.max_retries == 5
    assert c.api_key == "abc123"


def test_load_from_file(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(_minimal_raw()))
    c = LLMConfig.load(p)
    assert c.server_type == ServerType.OLLAMA


def test_server_url_trailing_slash_stripped():
    raw = _minimal_raw()
    raw["server_url"] = "http://localhost:11434/v1/"
    c = LLMConfig.from_dict(raw)
    assert c.server_url == "http://localhost:11434/v1"


@pytest.mark.parametrize("missing", ["server_type", "server_url", "models"])
def test_missing_required_key_raises(missing):
    raw = _minimal_raw()
    del raw[missing]
    with pytest.raises(ConfigError, match=missing):
        LLMConfig.from_dict(raw)


def test_bad_server_type_raises():
    raw = _minimal_raw()
    raw["server_type"] = "custom_thing"
    with pytest.raises(ConfigError, match="server_type"):
        LLMConfig.from_dict(raw)


def test_empty_server_url_raises():
    raw = _minimal_raw()
    raw["server_url"] = ""
    with pytest.raises(ConfigError, match="server_url"):
        LLMConfig.from_dict(raw)


def test_empty_models_raises():
    raw = _minimal_raw()
    raw["models"] = {}
    with pytest.raises(ConfigError, match="models"):
        LLMConfig.from_dict(raw)


def test_non_string_model_value_raises():
    raw = _minimal_raw()
    raw["models"] = {"hint": 42}
    with pytest.raises(ConfigError, match="models entries"):
        LLMConfig.from_dict(raw)


def test_resolve_model_returns_concrete_name():
    c = LLMConfig.from_dict(_minimal_raw())
    assert c.resolve_model("phi_detector") == "qwen2.5:7b-instruct"


def test_resolve_model_unknown_hint_raises():
    c = LLMConfig.from_dict(_minimal_raw())
    with pytest.raises(ConfigError, match="not registered"):
        c.resolve_model("nonexistent")


@pytest.mark.parametrize("server_type", ["ollama", "vllm", "openai", "lm_studio"])
def test_all_server_types_supported(server_type):
    raw = _minimal_raw()
    raw["server_type"] = server_type
    c = LLMConfig.from_dict(raw)
    assert c.server_type.value == server_type
