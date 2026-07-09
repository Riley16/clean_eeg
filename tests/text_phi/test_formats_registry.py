"""Tests for scripts/text_phi/formats/__init__.py."""

from __future__ import annotations

from scripts.text_phi.formats import (
    CsvFormat,
    REGISTRY,
    TxtFormat,
    get_format,
    infer_columns,
    supported_extensions,
)


def test_registry_has_txt_and_csv():
    assert ".txt" in REGISTRY
    assert ".csv" in REGISTRY


def test_get_format_case_insensitive():
    assert isinstance(get_format(".TXT"), TxtFormat)
    assert isinstance(get_format(".CSV"), CsvFormat)


def test_get_format_unknown_returns_none():
    assert get_format(".docx") is None


def test_supported_extensions_sorted():
    exts = supported_extensions()
    assert exts == sorted(exts)
    assert ".txt" in exts and ".csv" in exts


def test_infer_columns_reads_header(tmp_path):
    import pandas as pd
    p = tmp_path / "in.csv"
    pd.DataFrame([{"a": "1", "b": "2"}]).to_csv(p, index=False)
    assert infer_columns(p) == ["a", "b"]
