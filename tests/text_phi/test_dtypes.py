"""Tests for scripts/text_phi/dtypes.py."""

from __future__ import annotations

import datetime as _dt

import pytest

from scripts.text_phi.dtypes import (
    DTYPES,
    DtypeError,
    get_dtype,
    known_dtypes,
)


# ---------- registry ----------

def test_known_dtypes_sorted_and_complete():
    names = known_dtypes()
    assert names == sorted(names)
    expected = {
        "string", "integer", "float", "boolean", "enum",
        "date", "datetime", "bytes",
        "subject_name", "zip_code", "phone", "email", "ssn", "mrn",
        "url", "ip",
    }
    assert set(names) == expected


def test_get_dtype_unknown_raises():
    with pytest.raises(DtypeError, match="unknown dtype"):
        get_dtype("no_such_dtype")


# ---------- opaque dtypes (roundtrip identity) ----------

@pytest.mark.parametrize(
    "name,value",
    [
        ("string", "hello"), ("string", ""), ("string", "\n\t"),
        ("enum", "level_1"),
        ("subject_name", "John Smith"),
        ("zip_code", "19104"), ("phone", "(215) 555-1212"),
        ("email", "a@b.c"), ("ssn", "555-11-2222"),
        ("mrn", "MRN12345"), ("url", "https://ex.com"),
        ("ip", "192.168.1.1"),
    ],
)
def test_opaque_roundtrip_is_identity(name, value):
    d = get_dtype(name)
    d.validate(value)  # never raises
    assert d.parse(value) == value
    assert d.format(value) == value


# ---------- integer ----------

@pytest.mark.parametrize("s", ["0", "-1", "42", "9223372036854775807", "-9223372036854775808"])
def test_integer_valid(s):
    d = get_dtype("integer")
    d.validate(s)
    v = d.parse(s)
    assert isinstance(v, int)
    assert d.format(v) == s


def test_integer_empty_is_none():
    d = get_dtype("integer")
    d.validate("")
    assert d.parse("") is None
    assert d.format(None) == ""


@pytest.mark.parametrize("s", ["3.14", "1e5", "not_an_int", "0x10"])
def test_integer_invalid_raises(s):
    with pytest.raises(DtypeError, match="not an integer"):
        get_dtype("integer").validate(s)


# ---------- float ----------

@pytest.mark.parametrize("s", ["0.0", "-0.0", "3.14", "1e5", "nan", "inf", "-inf"])
def test_float_valid(s):
    d = get_dtype("float")
    d.validate(s)
    v = d.parse(s)
    assert isinstance(v, float)


def test_float_roundtrip_preserves_precision():
    d = get_dtype("float")
    s = "0.1"
    v = d.parse(s)
    assert d.parse(d.format(v)) == v


def test_float_nan_inf_roundtrip():
    d = get_dtype("float")
    for s in ("nan", "inf", "-inf"):
        v = d.parse(s)
        assert d.parse(d.format(v)) == v or (v != v and d.parse(d.format(v)) != d.parse(d.format(v)))  # NaN != NaN


@pytest.mark.parametrize("s", ["not_a_number", "1..2", "1,2"])
def test_float_invalid_raises(s):
    with pytest.raises(DtypeError, match="not a float"):
        get_dtype("float").validate(s)


# ---------- boolean ----------

@pytest.mark.parametrize("s,expected", [
    ("true", True), ("True", True), ("TRUE", True), ("1", True), ("yes", True),
    ("false", False), ("False", False), ("0", False), ("no", False),
])
def test_boolean_valid(s, expected):
    d = get_dtype("boolean")
    d.validate(s)
    assert d.parse(s) is expected


def test_boolean_empty_is_none():
    d = get_dtype("boolean")
    assert d.parse("") is None


@pytest.mark.parametrize("s", ["maybe", "2", "yeah nah"])
def test_boolean_invalid_raises(s):
    with pytest.raises(DtypeError, match="not a boolean"):
        get_dtype("boolean").validate(s)


def test_boolean_format():
    d = get_dtype("boolean")
    assert d.format(True) == "true"
    assert d.format(False) == "false"
    assert d.format(None) == ""


# ---------- date ----------

@pytest.mark.parametrize("s", ["2024-01-15", "2000-02-29", "1985-01-01"])
def test_date_valid(s):
    d = get_dtype("date")
    d.validate(s)
    v = d.parse(s)
    assert isinstance(v, _dt.date)
    assert d.format(v) == s


def test_date_empty_is_none():
    d = get_dtype("date")
    assert d.parse("") is None


@pytest.mark.parametrize("s", ["01/15/2024", "not_a_date", "2024-13-01", "2024-01-32"])
def test_date_invalid_raises(s):
    with pytest.raises(DtypeError, match="not an ISO date"):
        get_dtype("date").validate(s)


# ---------- datetime ----------

@pytest.mark.parametrize("s", [
    "2024-01-15T10:30:00", "2024-01-15T10:30:00+00:00", "1985-01-01T00:00:00",
])
def test_datetime_valid(s):
    d = get_dtype("datetime")
    d.validate(s)
    v = d.parse(s)
    assert isinstance(v, _dt.datetime)
    assert d.format(v) == s


@pytest.mark.parametrize("s", ["not_a_datetime", "2024-13-01T00:00:00"])
def test_datetime_invalid_raises(s):
    # Python 3.11+ `datetime.fromisoformat` accepts date-only strings ISO
    # dates (e.g. "2024-01-15") and treats them as midnight, so those don't
    # error at parse time — only genuinely malformed values do.
    with pytest.raises(DtypeError, match="not an ISO datetime"):
        get_dtype("datetime").validate(s)


# ---------- bytes ----------

def test_bytes_roundtrip_base64():
    d = get_dtype("bytes")
    original = b"\x00\x01\x02\xff\xfe"
    encoded = d.format(original)
    d.validate(encoded)
    assert d.parse(encoded) == original


def test_bytes_empty_is_none():
    d = get_dtype("bytes")
    assert d.parse("") is None
    assert d.format(None) == ""


@pytest.mark.parametrize("s", ["not_base64!", "===="])
def test_bytes_invalid_raises(s):
    with pytest.raises(DtypeError, match="not valid base64"):
        get_dtype("bytes").validate(s)
