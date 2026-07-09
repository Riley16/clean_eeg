"""Tests for whole-field typed PHI redaction operations."""

from __future__ import annotations

import pytest

from scripts.text_phi.operations.typed_phi import (
    EmailRedactOperation,
    IpRedactOperation,
    MrnRedactOperation,
    PhoneRedactOperation,
    SsnRedactOperation,
    UrlRedactOperation,
    ZipRedactOperation,
)


@pytest.mark.parametrize("op_cls,dtype,value,expected_entity", [
    (ZipRedactOperation, "zip_code", "19104-1234", "US_ZIP_CODE"),
    (PhoneRedactOperation, "phone", "(215) 555-1212", "PHONE_NUMBER"),
    (EmailRedactOperation, "email", "js@ex.com", "EMAIL_ADDRESS"),
    (SsnRedactOperation, "ssn", "555-11-2222", "US_SSN"),
    (MrnRedactOperation, "mrn", "MRN12345", "MRN"),
    (UrlRedactOperation, "url", "https://ex.com", "URL"),
    (IpRedactOperation, "ip", "192.168.1.1", "IP_ADDRESS"),
])
def test_typed_redact_emits_span(make_ctx, op_cls, dtype, value, expected_entity):
    op = op_cls()
    ctx = make_ctx("f", dtype)
    new_val, spans = op.apply(value, ctx)
    assert new_val != value  # replaced
    assert len(spans) == 1
    assert spans[0].entity_type == expected_entity


@pytest.mark.parametrize("op_cls,dtype", [
    (ZipRedactOperation, "zip_code"),
    (PhoneRedactOperation, "phone"),
    (EmailRedactOperation, "email"),
    (SsnRedactOperation, "ssn"),
    (MrnRedactOperation, "mrn"),
    (UrlRedactOperation, "url"),
    (IpRedactOperation, "ip"),
])
def test_typed_redact_empty_passthrough(make_ctx, op_cls, dtype):
    op = op_cls()
    ctx = make_ctx("f", dtype)
    assert op.apply("", ctx) == ("", [])


def test_typed_redact_custom_replacement(make_ctx):
    op = EmailRedactOperation()
    ctx = make_ctx("f", "email", params={"value": "[EMAIL]"})
    new_val, _ = op.apply("js@ex.com", ctx)
    assert new_val == "[EMAIL]"


def test_zip_default_replacement_is_five_zeros(make_ctx):
    op = ZipRedactOperation()
    ctx = make_ctx("f", "zip_code")
    new_val, _ = op.apply("19104", ctx)
    assert new_val == "00000"
