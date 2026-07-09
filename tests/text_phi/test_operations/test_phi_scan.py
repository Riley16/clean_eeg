"""Tests for PHI-scan operations that bridge to the existing detector layer."""

from __future__ import annotations

from clean_eeg.anonymize import PersonalName
from scripts.text_phi.operations.phi_scan import (
    GenericPhiScanOperation,
    ParseSubjectNameOperation,
    SubjectNameScanOperation,
)
from scripts.text_phi.records import RecordContext
from scripts.text_phi.redactor import TextRedactor


# ---------- parse_subject_name ----------

def test_parse_subject_publishes_to_context(make_ctx):
    op = ParseSubjectNameOperation()
    rc = RecordContext()
    ctx = make_ctx("patient_name", "subject_name", record_context=rc)
    new_val, spans = op.apply("John Smith", ctx)
    # Value unchanged; downstream op handles blanking.
    assert new_val == "John Smith"
    assert spans == []
    assert isinstance(rc.subject, PersonalName)
    assert rc.subject.first_name == "John"
    assert rc.subject.last_name == "Smith"


def test_parse_subject_last_comma_first(make_ctx):
    op = ParseSubjectNameOperation()
    rc = RecordContext()
    ctx = make_ctx("patient_name", "subject_name", record_context=rc)
    op.apply("Smith, John P", ctx)
    assert rc.subject.first_name == "John"
    assert rc.subject.last_name == "Smith"


def test_parse_subject_empty_no_op(make_ctx):
    op = ParseSubjectNameOperation()
    rc = RecordContext()
    ctx = make_ctx("patient_name", "subject_name", record_context=rc)
    op.apply("", ctx)
    assert rc.subject is None


def test_parse_subject_name_format_param(make_ctx):
    op = ParseSubjectNameOperation()
    rc = RecordContext()
    ctx = make_ctx("patient_name", "subject_name",
                   record_context=rc, params={"name_format": "last, first middle"})
    op.apply("Smith, John Paul", ctx)
    assert rc.subject.first_name == "John"
    assert rc.subject.middle_names == ["Paul"]
    assert rc.subject.last_name == "Smith"


# ---------- subject_name_scan ----------

def test_subject_scan_uses_context_subject(make_ctx):
    tr = TextRedactor(mode="subject")
    rc = RecordContext(subject=PersonalName("John", [], "O'Connor"))
    ctx = make_ctx("note", "string", record_context=rc, text_redactor=tr)
    new_val, spans = SubjectNameScanOperation().apply(
        "Dr. John O'Connor saw the patient.", ctx,
    )
    assert "John" not in new_val
    assert any(s.entity_type == "SUBJECT_NAME" for s in spans)


def test_subject_scan_no_subject_passthrough(make_ctx):
    tr = TextRedactor(mode="subject")
    ctx = make_ctx("note", "string", record_context=RecordContext(), text_redactor=tr)
    new_val, spans = SubjectNameScanOperation().apply(
        "Dr. John Smith saw the patient.", ctx,
    )
    assert new_val == "Dr. John Smith saw the patient."
    assert spans == []


def test_subject_scan_falls_back_to_depends_on(make_ctx):
    tr = TextRedactor(mode="subject")
    ctx = make_ctx(
        "note", "string",
        depends_on={"subject_name_field": "patient_name"},
        record={"note": "Dr. Alice Chen saw pt.", "patient_name": "Alice Chen"},
        text_redactor=tr,
    )
    new_val, spans = SubjectNameScanOperation().apply("Dr. Alice Chen saw pt.", ctx)
    assert "Alice" not in new_val


def test_subject_scan_no_redactor_passthrough(make_ctx):
    ctx = make_ctx("note", "string", record_context=RecordContext(subject=PersonalName("John", [], "Smith")))
    new_val, spans = SubjectNameScanOperation().apply("John Smith", ctx)
    assert new_val == "John Smith"
    assert spans == []


def test_subject_scan_empty(make_ctx):
    tr = TextRedactor(mode="subject")
    ctx = make_ctx("f", "string", text_redactor=tr)
    assert SubjectNameScanOperation().apply("", ctx) == ("", [])


# ---------- generic_phi_scan ----------

def test_generic_scan_finds_email(make_ctx):
    tr = TextRedactor(mode="generic")
    ctx = make_ctx("note", "string", text_redactor=tr)
    new_val, spans = GenericPhiScanOperation().apply(
        "contact user@example.org today", ctx,
    )
    assert "user@example.org" not in new_val
    assert any(s.entity_type == "EMAIL_ADDRESS" for s in spans)


def test_generic_scan_no_redactor_passthrough(make_ctx):
    ctx = make_ctx("f", "string")
    new_val, spans = GenericPhiScanOperation().apply("contact user@ex.com", ctx)
    assert new_val == "contact user@ex.com"
    assert spans == []


def test_generic_scan_empty(make_ctx):
    tr = TextRedactor(mode="generic")
    ctx = make_ctx("f", "string", text_redactor=tr)
    assert GenericPhiScanOperation().apply("", ctx) == ("", [])
