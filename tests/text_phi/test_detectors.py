"""Tests for scripts/text_phi/detectors.py."""

from __future__ import annotations

import pytest

from presidio_analyzer import RecognizerResult

from scripts.text_phi.detectors import (
    build_generic_analyzer,
    filter_person_hits_by_whitelist,
    supported_generic_entities,
)


# ---------- entity discovery ----------

def test_supported_entities_include_customs(generic_analyzer):
    ents = supported_generic_entities()
    assert "US_ZIP_CODE" in ents
    assert "AGE_OVER_89" in ents
    assert "PERSON" in ents
    assert "EMAIL_ADDRESS" in ents


def test_supported_entities_include_mrn_when_enabled():
    ents = supported_generic_entities(mrn_regex=r"\bMRN\d+\b")
    assert "MRN" in ents


# ---------- ZIP recognizer ----------

@pytest.mark.parametrize("s", ["19104", "19104-1234", "The clinic ZIP is 02138."])
def test_zip_positive(generic_analyzer, s):
    hits = generic_analyzer.analyze(s, entities=["US_ZIP_CODE"], language="en")
    assert any(h.entity_type == "US_ZIP_CODE" for h in hits), f"missed ZIP in {s!r}"


@pytest.mark.parametrize("s", ["Channel 42", "Age 18"])
def test_zip_negative_short_numbers(generic_analyzer, s):
    hits = generic_analyzer.analyze(s, entities=["US_ZIP_CODE"], language="en")
    assert not [h for h in hits if h.entity_type == "US_ZIP_CODE"], (
        f"unexpected ZIP hit in {s!r}"
    )


# ---------- AGE_OVER_89 recognizer ----------

@pytest.mark.parametrize(
    "s",
    [
        "90 years old",
        "aged 92",
        "aged of 91",
        "92-year-old patient",
        "92 y/o male",
        "93yo",
        "Patient is 105 years old",
        "age 99",
    ],
)
def test_age_positive_over_89(generic_analyzer, s):
    hits = generic_analyzer.analyze(s, entities=["AGE_OVER_89"], language="en")
    assert any(h.entity_type == "AGE_OVER_89" for h in hits), (
        f"missed AGE_OVER_89 in {s!r}"
    )


@pytest.mark.parametrize(
    "s",
    [
        "89 years old",
        "18 y/o",
        "80yo",
        "aged 42",
        "90 channels",           # 90 without age context
        "collected 92 samples",  # 92 without age context
    ],
)
def test_age_negative_below_90_or_no_context(generic_analyzer, s):
    hits = generic_analyzer.analyze(s, entities=["AGE_OVER_89"], language="en")
    assert not [h for h in hits if h.entity_type == "AGE_OVER_89"], (
        f"unexpected AGE_OVER_89 hit in {s!r}"
    )


# ---------- MRN custom recognizer ----------

def test_mrn_positive_when_supplied():
    an = build_generic_analyzer(mrn_regex=r"\bMRN\d{7,10}\b")
    hits = an.analyze("Chart: MRN1234567 follow-up.", entities=["MRN"], language="en")
    assert any(h.entity_type == "MRN" for h in hits)


def test_mrn_negative_off_by_default(generic_analyzer):
    ents = {e for e in supported_generic_entities()}
    assert "MRN" not in ents


# ---------- whitelist filter ----------

# Synthetic whitelist for tests. Real one is built by scripts/build_whitelist.py
# from SUBTLEX-US and gated by presence of data/*.json; injecting an explicit
# set here tests the filter logic independent of the disk layout.
_TEST_WHITELIST = frozenset({"rose", "the", "smelled", "sweet"})


def _fake_result(entity: str, start: int, end: int, score: float = 0.85) -> RecognizerResult:
    return RecognizerResult(entity_type=entity, start=start, end=end, score=score)


def test_whitelist_filter_drops_common_word_person():
    # "Rose" is in the synthetic common-word whitelist.
    text = "Rose smelled sweet."
    hits = [_fake_result("PERSON", 0, 4)]
    kept = filter_person_hits_by_whitelist(text, hits, whitelist=_TEST_WHITELIST)
    assert kept == []


def test_whitelist_filter_keeps_rare_person():
    text = "Zbigniew Herbert wrote poetry."
    hits = [_fake_result("PERSON", 0, 16)]
    kept = filter_person_hits_by_whitelist(text, hits, whitelist=_TEST_WHITELIST)
    assert len(kept) == 1
    assert kept[0].entity_type == "PERSON"


def test_whitelist_filter_ignores_non_person():
    text = "Contact us at 123-45-6789."
    hits = [_fake_result("US_SSN", 14, 25)]
    kept = filter_person_hits_by_whitelist(text, hits, whitelist=_TEST_WHITELIST)
    assert kept == hits


def test_whitelist_filter_drops_multi_word_all_common():
    # Both tokens whitelisted → drop.
    text = "the rose"
    hits = [_fake_result("PERSON", 0, 8)]
    kept = filter_person_hits_by_whitelist(text, hits, whitelist=_TEST_WHITELIST)
    assert kept == []


def test_whitelist_filter_keeps_multi_word_with_rare_token():
    text = "Rose Zbigniew"
    hits = [_fake_result("PERSON", 0, 13)]
    kept = filter_person_hits_by_whitelist(text, hits, whitelist=_TEST_WHITELIST)
    assert len(kept) == 1


def test_whitelist_filter_missing_files_returns_empty(monkeypatch, caplog):
    """When data/ files are absent, the safe loader logs a warning and keeps
    all PERSON hits — the conservative (over-redact) failure mode."""
    from scripts.text_phi import detectors

    monkeypatch.setattr(detectors, "_safe_load_whitelist",
                        lambda: frozenset())
    detectors._safe_load_whitelist.cache_clear = lambda: None  # no-op
    text = "Rose smelled sweet."
    hits = [_fake_result("PERSON", 0, 4)]
    kept = filter_person_hits_by_whitelist(text, hits)
    assert len(kept) == 1


# ---------- analyzer caching ----------

def test_analyzer_cached_by_flags():
    a1 = build_generic_analyzer(enable_zip=True, enable_age=True)
    a2 = build_generic_analyzer(enable_zip=True, enable_age=True)
    assert a1 is a2
    a3 = build_generic_analyzer(enable_zip=False, enable_age=True)
    assert a3 is not a1
