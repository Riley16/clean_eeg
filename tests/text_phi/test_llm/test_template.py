"""Tests for scripts/text_phi/llm/template.py."""

from __future__ import annotations

import jinja2
import pytest

from scripts.text_phi.llm.template import (
    load_prompt,
    render_context,
    render_string,
)


# ---------- render_string ----------

def test_render_string_basic():
    assert render_string("hello {{ name }}", {"name": "world"}) == "hello world"


def test_render_string_nested_dict():
    assert render_string(
        "patient: {{ record.subject_name }}",
        {"record": {"subject_name": "John"}},
    ) == "patient: John"


def test_render_string_missing_variable_raises():
    with pytest.raises(jinja2.UndefinedError):
        render_string("{{ missing }}", {})


def test_render_string_missing_nested_attr_raises():
    with pytest.raises(jinja2.UndefinedError):
        render_string("{{ record.missing }}", {"record": {"present": "x"}})


def test_render_string_literal_passes_through():
    assert render_string("no templating here", {}) == "no templating here"


# ---------- render_context ----------

def test_render_context_all_strings():
    ctx = {"a": "{{ x }}", "b": "static", "c": "{{ y }}"}
    got = render_context(ctx, {"x": "1", "y": "2"})
    assert got == {"a": "1", "b": "static", "c": "2"}


def test_render_context_preserves_iteration_order():
    ctx = {"z": "{{ v }}", "a": "{{ v }}", "m": "{{ v }}"}
    got = render_context(ctx, {"v": "1"})
    assert list(got.keys()) == ["z", "a", "m"]


def test_render_context_non_string_passes_through():
    ctx = {"n": 42, "flag": True, "list": [1, 2, 3], "s": "{{ x }}"}
    got = render_context(ctx, {"x": "resolved"})
    assert got == {"n": 42, "flag": True, "list": [1, 2, 3], "s": "resolved"}


def test_render_context_missing_var_raises():
    with pytest.raises(jinja2.UndefinedError):
        render_context({"a": "{{ missing }}"}, {})


def test_render_context_empty():
    assert render_context({}, {"x": "y"}) == {}


# ---------- load_prompt ----------

def test_load_prompt_returns_verbatim(tmp_path):
    p = tmp_path / "prompt.jinja"
    p.write_text("System: {{ role }}\nUser: {{ input }}")
    assert load_prompt(p) == "System: {{ role }}\nUser: {{ input }}"


def test_load_prompt_preserves_newlines(tmp_path):
    p = tmp_path / "prompt.jinja"
    p.write_text("line 1\nline 2\r\nline 3\n")
    text = load_prompt(p)
    # Newlines survive Python's default newline translation on read.
    assert "line 1" in text
    assert "line 2" in text
    assert "line 3" in text
