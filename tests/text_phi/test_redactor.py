"""Tests for scripts/text_phi/redactor.py."""

from __future__ import annotations

import pytest

from clean_eeg.anonymize import PersonalName
from scripts.text_phi.redactor import (
    RedactionResult,
    RedactionSpan,
    TextRedactor,
    _apply_replacements,
    _merge_spans,
)


# ---------- span merging ----------

def test_merge_disjoint_spans_preserved():
    spans = [
        RedactionSpan(0, 4, "PERSON", 0.9, "spacy", "John"),
        RedactionSpan(10, 15, "US_SSN", 0.9, "ssn", "12345"),
    ]
    merged = _merge_spans(spans)
    assert len(merged) == 2


def test_merge_overlap_higher_score_wins():
    a = RedactionSpan(0, 10, "PERSON", 0.5, "spacy", "abcdefghij")
    b = RedactionSpan(0, 10, "DATE_TIME", 0.9, "date", "abcdefghij")
    merged = _merge_spans([a, b])
    assert len(merged) == 1
    assert merged[0].entity_type == "DATE_TIME"


def test_merge_tie_subject_name_wins():
    a = RedactionSpan(0, 4, "SUBJECT_NAME", 0.9, "denylist", "John")
    b = RedactionSpan(0, 4, "PERSON", 0.9, "spacy", "John")
    merged = _merge_spans([a, b])
    assert len(merged) == 1
    assert merged[0].entity_type == "SUBJECT_NAME"


def test_merge_tie_wider_span_wins():
    a = RedactionSpan(0, 4, "PERSON", 0.9, "x", "John")
    b = RedactionSpan(0, 8, "PERSON", 0.9, "y", "John Doe")
    merged = _merge_spans([a, b])
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0, 8)


def test_merge_empty_list():
    assert _merge_spans([]) == []


# ---------- replacement application ----------

def test_apply_replacements_literal_right_to_left():
    text = "hello world"
    spans = [
        RedactionSpan(0, 5, "PERSON", 0.9, "r", "hello"),
        RedactionSpan(6, 11, "PERSON", 0.9, "r", "world"),
    ]
    out = _apply_replacements(text, spans, "literal", "X")
    assert out == "X X"


def test_apply_replacements_labeled():
    text = "hello 123-45-6789"
    spans = [
        RedactionSpan(6, 17, "US_SSN", 0.9, "r", "123-45-6789"),
    ]
    out = _apply_replacements(text, spans, "labeled", "X")
    assert out == "hello [US_SSN]"


def test_apply_replacements_no_spans():
    assert _apply_replacements("hello", [], "literal", "X") == "hello"


# ---------- subject mode (integration with SubjectNameRedactor) ----------

def test_subject_mode_denylist_exact_match(subject_redactor):
    result = subject_redactor.redact("Patient John was seen.")
    assert "John" not in result.text
    assert any(s.entity_type == "SUBJECT_NAME" for s in result.spans)


def test_subject_mode_fuzzy_typo(subject_redactor):
    # Levenshtein distance 1 from "O'Connor" (after strip) → OConor.
    result = subject_redactor.redact("chart labeled Oconor.")
    assert "Oconor" not in result.text


def test_subject_mode_title_and_initials(subject_redactor):
    result = subject_redactor.redact("Note by Dr. John P. O'Connor today.")
    assert "John" not in result.text
    assert "O'Connor" not in result.text


def test_subject_mode_last_only_with_title(subject_redactor):
    result = subject_redactor.redact("Prof O'Connor's lab.")
    assert "O'Connor" not in result.text


def test_subject_mode_ignores_unrelated_name(subject_redactor):
    # "Alice" is a real name unrelated to the subject; subject mode should
    # NOT redact it (it doesn't run the generic PERSON recognizer).
    result = subject_redactor.redact("Alice performed the analysis.")
    assert "Alice" in result.text


# ---------- generic mode ----------

def test_generic_mode_email(generic_redactor):
    result = generic_redactor.redact("contact: user@example.org here.")
    assert "user@example.org" not in result.text
    assert any(s.entity_type == "EMAIL_ADDRESS" for s in result.spans)


def test_generic_mode_ssn(generic_redactor):
    # Presidio's UsSsnRecognizer invalidates known-sample SSNs like
    # 123-45-6789 and 987-65-4321. Use a non-excluded value.
    result = generic_redactor.redact("SSN 555-11-2222 on file.")
    assert "555-11-2222" not in result.text
    assert any(s.entity_type == "US_SSN" for s in result.spans)


def test_generic_mode_zip(generic_redactor):
    result = generic_redactor.redact("ZIP 19104-1234 delivery.")
    assert "19104-1234" not in result.text


def test_generic_mode_age_over_89(generic_redactor):
    result = generic_redactor.redact("Patient is 95 years old.")
    assert "95" not in result.text


def test_generic_mode_url(generic_redactor):
    result = generic_redactor.redact("See https://example.com/x for details.")
    assert "https://example.com" not in result.text


def test_generic_mode_ip(generic_redactor):
    result = generic_redactor.redact("Server at 192.168.1.42 responded.")
    assert "192.168.1.42" not in result.text


def test_generic_mode_no_subject_when_no_name(generic_redactor):
    # Subject-mode-only mechanisms (fuzzy, deny-list) should NOT redact
    # rare surnames that spaCy also can't identify.
    result = generic_redactor.redact("channel notation ok")
    # This text shouldn't emit any span — sanity check the pipeline doesn't
    # over-fire.
    assert "channel" in result.text


# ---------- both mode ----------

def test_both_mode_unions_subject_and_generic(both_redactor):
    result = both_redactor.redact("Dr. O'Connor emailed user@example.org.")
    assert "O'Connor" not in result.text
    assert "user@example.org" not in result.text


def test_both_mode_labeled_style(subject_pn):
    r = TextRedactor(mode="both", subject_names=[subject_pn], replacement_style="labeled")
    result = r.redact("Contact John at user@ex.com.")
    assert "[SUBJECT_NAME]" in result.text
    assert "[EMAIL_ADDRESS]" in result.text


# ---------- per-row subject ----------

def test_redact_with_subject_per_call():
    r = TextRedactor(mode="subject")  # no pre-configured names
    other = PersonalName(first_name="Alice", middle_names=[], last_name="Smith")
    result = r.redact_with_subject("Note by Alice Smith today.", other)
    assert "Alice" not in result.text


def test_redact_with_subject_generic_mode_ignores_subject():
    # Generic mode should not attach subject even if a name is supplied.
    r = TextRedactor(mode="generic")
    pn = PersonalName(first_name="Zbigniew", middle_names=[], last_name="Herbert")
    result = r.redact_with_subject("Random unlikely word", pn)
    # No SUBJECT_NAME span should appear in generic mode.
    assert not any(s.entity_type == "SUBJECT_NAME" for s in result.spans)


# ---------- empty input ----------

def test_empty_text_returns_empty(both_redactor):
    result = both_redactor.redact("")
    assert isinstance(result, RedactionResult)
    assert result.text == ""
    assert result.spans == []


def test_empty_text_with_subject_returns_empty(both_redactor, subject_pn):
    result = both_redactor.redact_with_subject("", subject_pn)
    assert result.text == ""


# ---------- validation ----------

def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        TextRedactor(mode="bogus")  # type: ignore[arg-type]


# ---------- subject cache reuse ----------

def test_subject_cache_reused(subject_pn):
    r = TextRedactor(mode="both", subject_names=[subject_pn])
    same = r._get_subject_redactor(subject_pn)
    again = r._get_subject_redactor(subject_pn)
    assert same is again
