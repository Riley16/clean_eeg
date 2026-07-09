"""Generic PHI detectors: Presidio built-ins + custom regex + PERSON whitelist filter.

The generic layer complements the subject-name-only redaction in
`clean_eeg.anonymize`. It registers every predefined Presidio recognizer for
English, adds three small custom regex recognizers to close HIPAA gaps
(ZIP, ages > 89, optional site-configurable MRN), and provides a post-analyze
filter that drops PERSON hits whose matched text is entirely composed of
common English words (via `clean_eeg.whitelist`).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider

from clean_eeg.whitelist import NAME_WORD_RE, load_whitelist, token_in_whitelist

_log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _safe_load_whitelist() -> frozenset[str]:
    """Load the common-word whitelist, or return an empty set if the data
    files haven't been built.

    The whitelist is a false-positive filter for PERSON hits. When it's
    missing, the safer failure mode is to keep all PERSON hits (over-redact)
    rather than drop them (under-redact PHI)."""
    try:
        return frozenset(load_whitelist())
    except FileNotFoundError:
        _log.warning(
            "Common-word whitelist not found (data/*.json). PERSON hits will "
            "not be filtered — expect more false positives. Build the "
            "whitelist via scripts/build_whitelist.py to enable the filter."
        )
        return frozenset()


# Same spaCy config as clean_eeg.anonymize.build_presidio — reuses the model
# already required by the main pipeline, avoiding a second download.
_SPACY_CONF = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}


# ---------- custom recognizer builders ----------

def _us_zip_recognizer(score: float = 0.5) -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity="US_ZIP_CODE",
        name="us_zip_code",
        patterns=[Pattern(name="us_zip_5_or_9", regex=r"\b\d{5}(?:-\d{4})?\b", score=score)],
    )


# HIPAA Safe Harbor requires collapsing all ages > 89 into a single "90+"
# category. Match "90+" only when age context is present, to avoid firing on
# channel counts, sample indices, etc.
_AGE_SUFFIX_RE = (
    r"\b(?:9\d|[1-9]\d{2,})"
    r"(?:\s*-?\s*year[\-\s]?old|\s*-?\s*yr[\-\s]?old"
    r"|\s+years?\s+old|\s*y[./]?o\.?|\s*yo)\b"
)
_AGE_PREFIX_RE = r"\b(?:age[d]?(?:\s+of)?)\s+(?:9\d|[1-9]\d{2,})\b"


def _age_over_89_recognizer(score: float = 0.85) -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity="AGE_OVER_89",
        name="age_over_89",
        patterns=[
            Pattern(name="age_over_89_suffix", regex=_AGE_SUFFIX_RE, score=score),
            Pattern(name="age_over_89_prefix", regex=_AGE_PREFIX_RE, score=score),
        ],
    )


def _mrn_recognizer(user_regex: str, score: float = 0.85) -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity="MRN",
        name="mrn_site_regex",
        patterns=[Pattern(name="mrn", regex=user_regex, score=score)],
    )


# ---------- analyzer builder ----------

@lru_cache(maxsize=4)
def _cached_analyzer(
    enable_zip: bool,
    enable_age: bool,
    mrn_regex: str | None,
) -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(nlp_configuration=_SPACY_CONF).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"])
    if enable_zip:
        registry.add_recognizer(_us_zip_recognizer())
    if enable_age:
        registry.add_recognizer(_age_over_89_recognizer())
    if mrn_regex:
        registry.add_recognizer(_mrn_recognizer(mrn_regex))
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)


def build_generic_analyzer(
    enable_zip: bool = True,
    enable_age: bool = True,
    mrn_regex: str | None = None,
) -> AnalyzerEngine:
    """Return an AnalyzerEngine with Presidio's predefined English recognizers
    plus the requested custom ones. Cached: same flags → same instance."""
    return _cached_analyzer(enable_zip, enable_age, mrn_regex)


# ---------- entity discovery ----------

def supported_generic_entities(
    enable_zip: bool = True,
    enable_age: bool = True,
    mrn_regex: str | None = None,
) -> list[str]:
    """Sorted list of entity names the generic analyzer can emit."""
    ents: set[str] = set()
    analyzer = build_generic_analyzer(enable_zip, enable_age, mrn_regex)
    for rec in analyzer.registry.recognizers:
        ents.update(rec.supported_entities)
    return sorted(ents)


# ---------- whitelist post-filter for PERSON ----------

def _all_tokens_whitelisted(matched: str, whitelist: set[str]) -> bool:
    tokens = NAME_WORD_RE.findall(matched)
    if not tokens:
        return False
    return all(token_in_whitelist(tok, whitelist) for tok in tokens)


def filter_person_hits_by_whitelist(
    text: str,
    results: list[RecognizerResult],
    whitelist: set[str] | None = None,
) -> list[RecognizerResult]:
    """Drop PERSON hits whose entire matched span is common English words.

    Non-PERSON entity types are returned untouched.
    """
    if whitelist is None:
        whitelist = _safe_load_whitelist()
    kept: list[RecognizerResult] = []
    for r in results:
        if r.entity_type != "PERSON":
            kept.append(r)
            continue
        matched = text[r.start:r.end]
        if _all_tokens_whitelisted(matched, whitelist):
            continue
        kept.append(r)
    return kept
