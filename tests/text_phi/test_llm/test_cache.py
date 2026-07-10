"""Tests for scripts/text_phi/llm/cache.py."""

from __future__ import annotations

import pytest

from scripts.text_phi.llm.cache import LLMCache, build_cache_key, hash_str


def _key() -> str:
    return build_cache_key(
        server_type="ollama",
        model="qwen2.5:7b-instruct",
        prompt_hash=hash_str("prompt"),
        input_hash=hash_str("input"),
        context_hash=hash_str("{}"),
        seed=42,
        temperature=0.0,
    )


# ---------- hashing / key stability ----------

def test_hash_str_deterministic():
    assert hash_str("hello") == hash_str("hello")


def test_build_key_deterministic():
    k1 = _key()
    k2 = _key()
    assert k1 == k2


def test_build_key_changes_with_any_component():
    k = _key()

    k_diff_seed = build_cache_key(
        server_type="ollama", model="qwen2.5:7b-instruct",
        prompt_hash=hash_str("prompt"), input_hash=hash_str("input"),
        context_hash=hash_str("{}"), seed=43, temperature=0.0,
    )
    assert k != k_diff_seed

    k_diff_temp = build_cache_key(
        server_type="ollama", model="qwen2.5:7b-instruct",
        prompt_hash=hash_str("prompt"), input_hash=hash_str("input"),
        context_hash=hash_str("{}"), seed=42, temperature=0.5,
    )
    assert k != k_diff_temp

    k_diff_model = build_cache_key(
        server_type="ollama", model="llama3.2:3b",
        prompt_hash=hash_str("prompt"), input_hash=hash_str("input"),
        context_hash=hash_str("{}"), seed=42, temperature=0.0,
    )
    assert k != k_diff_model

    k_diff_input = build_cache_key(
        server_type="ollama", model="qwen2.5:7b-instruct",
        prompt_hash=hash_str("prompt"), input_hash=hash_str("input2"),
        context_hash=hash_str("{}"), seed=42, temperature=0.0,
    )
    assert k != k_diff_input


# ---------- cache CRUD ----------

def test_cache_get_miss_returns_none(tmp_path):
    with LLMCache(tmp_path / "c.sqlite") as c:
        assert c.get("missing_key") is None


def test_cache_put_then_get(tmp_path):
    with LLMCache(tmp_path / "c.sqlite") as c:
        c.put("k1", '{"a": 1}')
        assert c.get("k1") == '{"a": 1}'


def test_cache_put_overwrites(tmp_path):
    with LLMCache(tmp_path / "c.sqlite") as c:
        c.put("k1", "first")
        c.put("k1", "second")
        assert c.get("k1") == "second"


def test_cache_clear(tmp_path):
    with LLMCache(tmp_path / "c.sqlite") as c:
        c.put("k1", "a")
        c.put("k2", "b")
        assert c.size() == 2
        n = c.clear()
        assert n == 2
        assert c.size() == 0


def test_cache_size(tmp_path):
    with LLMCache(tmp_path / "c.sqlite") as c:
        assert c.size() == 0
        c.put("k1", "a")
        assert c.size() == 1
        c.put("k2", "b")
        assert c.size() == 2


def test_cache_survives_process_restart(tmp_path):
    path = tmp_path / "c.sqlite"
    with LLMCache(path) as c:
        c.put("k1", "persistent")
    with LLMCache(path) as c2:
        assert c2.get("k1") == "persistent"


def test_cache_creates_parent_directory(tmp_path):
    nested = tmp_path / "sub" / "dir" / "c.sqlite"
    with LLMCache(nested) as c:
        c.put("k", "v")
    assert nested.exists()


def test_context_manager_closes(tmp_path):
    path = tmp_path / "c.sqlite"
    with LLMCache(path) as c:
        c.put("k1", "v")
    # Should be able to reopen after context manager exits.
    with LLMCache(path) as c2:
        assert c2.get("k1") == "v"
