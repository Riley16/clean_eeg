"""Dtype vocabulary + parse/format helpers.

A dtype is the semantic type of a field's value. In the records model, values
are stored as raw source strings so roundtrip is lossless when no redaction
happens. A dtype's job is to (1) validate that a source string is parseable
as the declared type, and (2) parse/format when an operation needs semantics.

Simple dtypes (`string`, `phone`, `email`, ...) are opaque — validate is a
no-op, parse/format is identity. Structured dtypes (`date`, `datetime`,
`integer`, `float`, `boolean`, `bytes`) actually parse.
"""

from __future__ import annotations

import base64
import datetime as _dt
from typing import Any, ClassVar, Protocol


class DtypeError(ValueError):
    """A source string does not parse under a declared dtype."""


class Dtype(Protocol):
    name: ClassVar[str]

    def validate(self, source_str: str) -> None:
        """Raise DtypeError if source_str is not parseable as this dtype."""

    def parse(self, source_str: str) -> Any:
        """Return the semantic value corresponding to source_str."""

    def format(self, value: Any) -> str:
        """Serialize a semantic value back to the canonical source string."""


# ---------- opaque dtypes (raw string carriers) ----------

class _OpaqueDtype:
    """Base for dtypes that carry the raw source string unchanged."""
    name: ClassVar[str] = ""

    def validate(self, source_str: str) -> None:
        return None

    def parse(self, source_str: str) -> str:
        return source_str

    def format(self, value: Any) -> str:
        return str(value) if value is not None else ""


class StringDtype(_OpaqueDtype):
    name: ClassVar[str] = "string"


class EnumDtype(_OpaqueDtype):
    """Categorical field. Level set is declared in the field spec, not the
    dtype instance — validation is delegated to the schema layer to keep
    dtypes stateless."""
    name: ClassVar[str] = "enum"


class SubjectNameDtype(_OpaqueDtype):
    """Marker dtype for fields that hold a person name to be used as the
    per-record subject."""
    name: ClassVar[str] = "subject_name"


class ZipCodeDtype(_OpaqueDtype):
    name: ClassVar[str] = "zip_code"


class PhoneDtype(_OpaqueDtype):
    name: ClassVar[str] = "phone"


class EmailDtype(_OpaqueDtype):
    name: ClassVar[str] = "email"


class SsnDtype(_OpaqueDtype):
    name: ClassVar[str] = "ssn"


class MrnDtype(_OpaqueDtype):
    name: ClassVar[str] = "mrn"


class UrlDtype(_OpaqueDtype):
    name: ClassVar[str] = "url"


class IpDtype(_OpaqueDtype):
    name: ClassVar[str] = "ip"


# ---------- structured dtypes (real parsing) ----------

class IntegerDtype:
    name: ClassVar[str] = "integer"

    def validate(self, source_str: str) -> None:
        if source_str == "":
            return None
        try:
            int(source_str)
        except ValueError as e:
            raise DtypeError(f"not an integer: {source_str!r}") from e

    def parse(self, source_str: str) -> int | None:
        if source_str == "":
            return None
        return int(source_str)

    def format(self, value: Any) -> str:
        if value is None:
            return ""
        return str(int(value))


class FloatDtype:
    name: ClassVar[str] = "float"

    def validate(self, source_str: str) -> None:
        if source_str == "":
            return None
        try:
            float(source_str)
        except ValueError as e:
            raise DtypeError(f"not a float: {source_str!r}") from e

    def parse(self, source_str: str) -> float | None:
        if source_str == "":
            return None
        return float(source_str)

    def format(self, value: Any) -> str:
        if value is None:
            return ""
        # `repr` gives round-trippable representation for finite floats and
        # preserves NaN/Inf semantics; str() would also work in 3.11+.
        return repr(float(value))


_TRUE = frozenset({"true", "1", "yes", "y", "t"})
_FALSE = frozenset({"false", "0", "no", "n", "f"})


class BooleanDtype:
    name: ClassVar[str] = "boolean"

    def validate(self, source_str: str) -> None:
        if source_str == "":
            return None
        if source_str.strip().lower() not in (_TRUE | _FALSE):
            raise DtypeError(f"not a boolean: {source_str!r}")

    def parse(self, source_str: str) -> bool | None:
        if source_str == "":
            return None
        s = source_str.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise DtypeError(f"not a boolean: {source_str!r}")

    def format(self, value: Any) -> str:
        if value is None:
            return ""
        return "true" if bool(value) else "false"


class DateDtype:
    """ISO-8601 dates: `YYYY-MM-DD`."""
    name: ClassVar[str] = "date"

    def validate(self, source_str: str) -> None:
        if source_str == "":
            return None
        try:
            _dt.date.fromisoformat(source_str)
        except ValueError as e:
            raise DtypeError(f"not an ISO date: {source_str!r}") from e

    def parse(self, source_str: str) -> _dt.date | None:
        if source_str == "":
            return None
        return _dt.date.fromisoformat(source_str)

    def format(self, value: Any) -> str:
        if value is None:
            return ""
        return _dt.date.isoformat(value)


class DatetimeDtype:
    """ISO-8601 datetimes; timezone optional."""
    name: ClassVar[str] = "datetime"

    def validate(self, source_str: str) -> None:
        if source_str == "":
            return None
        try:
            _dt.datetime.fromisoformat(source_str)
        except ValueError as e:
            raise DtypeError(f"not an ISO datetime: {source_str!r}") from e

    def parse(self, source_str: str) -> _dt.datetime | None:
        if source_str == "":
            return None
        return _dt.datetime.fromisoformat(source_str)

    def format(self, value: Any) -> str:
        if value is None:
            return ""
        return _dt.datetime.isoformat(value)


class BytesDtype:
    """Base64-encoded binary payload."""
    name: ClassVar[str] = "bytes"

    def validate(self, source_str: str) -> None:
        if source_str == "":
            return None
        try:
            base64.b64decode(source_str, validate=True)
        except Exception as e:
            raise DtypeError(f"not valid base64: {source_str!r}") from e

    def parse(self, source_str: str) -> bytes | None:
        if source_str == "":
            return None
        return base64.b64decode(source_str, validate=True)

    def format(self, value: Any) -> str:
        if value is None:
            return ""
        return base64.b64encode(value).decode("ascii")


# ---------- registry ----------

DTYPES: dict[str, Dtype] = {
    d.name: d() for d in [
        StringDtype, EnumDtype, SubjectNameDtype,
        ZipCodeDtype, PhoneDtype, EmailDtype, SsnDtype, MrnDtype, UrlDtype, IpDtype,
        IntegerDtype, FloatDtype, BooleanDtype,
        DateDtype, DatetimeDtype, BytesDtype,
    ]
}


def get_dtype(name: str) -> Dtype:
    if name not in DTYPES:
        raise DtypeError(
            f"unknown dtype: {name!r}. Known: {sorted(DTYPES.keys())}"
        )
    return DTYPES[name]


def known_dtypes() -> list[str]:
    return sorted(DTYPES.keys())
