"""Tests for the PromptRegistry helper in scripts/text_phi/llm/template.py."""

from __future__ import annotations

import pytest

from scripts.text_phi.llm.template import PromptRegistry


def test_loads_by_short_name(tmp_path):
    (tmp_path / "generic_phi.jinja").write_text("system: detect PHI in {{ value }}")
    reg = PromptRegistry(tmp_path)
    assert "detect PHI" in reg.get("generic_phi")


def test_caches_after_first_read(tmp_path):
    path = tmp_path / "cached.jinja"
    path.write_text("original")
    reg = PromptRegistry(tmp_path)
    assert reg.get("cached") == "original"
    # Overwrite the file on disk; cached version is still returned.
    path.write_text("modified")
    assert reg.get("cached") == "original"


def test_accepts_full_path(tmp_path):
    p = tmp_path / "sub" / "custom.jinja"
    p.parent.mkdir()
    p.write_text("custom body")
    reg = PromptRegistry(tmp_path)
    assert reg.get(str(p)) == "custom body"


def test_has_true_when_present(tmp_path):
    (tmp_path / "x.jinja").write_text("y")
    reg = PromptRegistry(tmp_path)
    assert reg.has("x") is True


def test_has_false_when_missing(tmp_path):
    reg = PromptRegistry(tmp_path)
    assert reg.has("missing") is False


def test_get_missing_raises(tmp_path):
    reg = PromptRegistry(tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.get("missing")
