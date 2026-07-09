"""Tests for scripts/text_phi/name_parse.py."""

from __future__ import annotations

import pytest

from scripts.text_phi.name_parse import parse_name


# ---------- permissive default ----------

def test_permissive_first_last():
    pn = parse_name("John Smith")
    assert pn.first_name == "John"
    assert pn.middle_names == []
    assert pn.last_name == "Smith"


def test_permissive_first_middle_last():
    pn = parse_name("John Paul Smith")
    assert pn.first_name == "John"
    assert pn.middle_names == ["Paul"]
    assert pn.last_name == "Smith"


def test_permissive_last_comma_first():
    pn = parse_name("Smith, John")
    assert pn.first_name == "John"
    assert pn.middle_names == []
    assert pn.last_name == "Smith"


def test_permissive_last_comma_first_middle():
    pn = parse_name("O'Connor, John P")
    assert pn.first_name == "John"
    assert pn.middle_names == ["P"]
    assert pn.last_name == "O'Connor"


def test_permissive_single_token_is_last_name():
    pn = parse_name("Smith")
    assert pn.first_name == ""
    assert pn.middle_names == []
    assert pn.last_name == "Smith"


def test_permissive_hyphenated_last_name():
    pn = parse_name("John Smith-Jones")
    assert pn.last_name == "Smith-Jones"


def test_permissive_initial_with_dot():
    pn = parse_name("John P. Smith")
    assert pn.first_name == "John"
    assert pn.middle_names == ["P"]
    assert pn.last_name == "Smith"


def test_permissive_multiple_middles():
    pn = parse_name("Mary Ann Catherine Smith")
    assert pn.first_name == "Mary"
    assert pn.middle_names == ["Ann", "Catherine"]
    assert pn.last_name == "Smith"


# ---------- explicit formats ----------

def test_format_first_middle_last():
    pn = parse_name("John Paul Smith", name_format="first middle last")
    assert (pn.first_name, pn.middle_names, pn.last_name) == ("John", ["Paul"], "Smith")


def test_format_last_comma_first_middle():
    pn = parse_name("Smith, John Paul", name_format="last, first middle")
    assert (pn.first_name, pn.middle_names, pn.last_name) == ("John", ["Paul"], "Smith")


def test_format_last_comma_first_no_middles():
    pn = parse_name("Smith, John Paul", name_format="last, first")
    assert pn.middle_names == []
    assert pn.last_name == "Smith"
    assert pn.first_name == "John"


def test_format_first_last_ignores_middles():
    pn = parse_name("John Paul Smith", name_format="first last")
    assert pn.first_name == "John"
    assert pn.last_name == "Smith"
    assert pn.middle_names == []


def test_format_case_insensitive():
    pn = parse_name("John Smith", name_format="FIRST LAST")
    assert pn.first_name == "John"


# ---------- errors ----------

def test_empty_cell_raises():
    with pytest.raises(ValueError):
        parse_name("")


def test_whitespace_only_raises():
    with pytest.raises(ValueError):
        parse_name("   \t  ")


def test_none_raises():
    with pytest.raises(ValueError):
        parse_name(None)  # type: ignore[arg-type]


def test_punctuation_only_raises():
    with pytest.raises(ValueError):
        parse_name(",.")


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        parse_name("John Smith", name_format="weird order")


# ---------- fallback behavior ----------

def test_last_comma_first_format_without_comma_falls_back():
    pn = parse_name("John Smith", name_format="last, first middle")
    assert pn.first_name == "John"
    assert pn.last_name == "Smith"


def test_last_comma_first_single_token_no_comma():
    pn = parse_name("Smith", name_format="last, first middle")
    assert pn.last_name == "Smith"
    assert pn.first_name == ""


def test_first_last_format_single_token():
    pn = parse_name("Smith", name_format="first last")
    assert pn.last_name == "Smith"
    assert pn.first_name == ""


def test_permissive_last_comma_first_empty_first_side_falls_back():
    # Trailing comma → first side is empty; should fall back to whitespace order.
    pn = parse_name("John Smith,")
    assert pn.first_name == "John"
    assert pn.last_name == "Smith"
