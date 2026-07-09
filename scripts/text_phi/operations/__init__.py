"""Operation registry + per-dtype default operation lists.

Adding a new operation:
  1. Implement it in one of the submodules.
  2. Import + add to `_ALL_OPERATIONS` below.
  3. If it belongs in a dtype's default pipeline, extend that dtype's
     `DEFAULT_<DTYPE>_REDACT_OPERATIONS` list.
"""

from __future__ import annotations

from .base import Operation
from .dates import (
    DEFAULT_BASE_DATE,
    DateRedactFullOperation,
    DateShiftRelativeToStayStartOperation,
    DateShiftToBaseOperation,
    DateYearOnlyOperation,
)
from .passthrough import (
    ConstantReplaceOperation,
    HashFieldOperation,
    PassthroughOperation,
)
from .phi_scan import (
    GenericPhiScanOperation,
    ParseSubjectNameOperation,
    SubjectNameScanOperation,
)
from .typed_phi import (
    EmailRedactOperation,
    IpRedactOperation,
    MrnRedactOperation,
    PhoneRedactOperation,
    SsnRedactOperation,
    UrlRedactOperation,
    ZipRedactOperation,
)


_ALL_OPERATIONS: list[type[Operation]] = [
    PassthroughOperation, ConstantReplaceOperation, HashFieldOperation,
    ParseSubjectNameOperation, SubjectNameScanOperation, GenericPhiScanOperation,
    DateShiftToBaseOperation, DateShiftRelativeToStayStartOperation,
    DateYearOnlyOperation, DateRedactFullOperation,
    ZipRedactOperation, PhoneRedactOperation, EmailRedactOperation,
    SsnRedactOperation, MrnRedactOperation, UrlRedactOperation, IpRedactOperation,
]

OPERATIONS: dict[str, Operation] = {cls.name: cls() for cls in _ALL_OPERATIONS}


def get_operation(name: str) -> Operation:
    if name not in OPERATIONS:
        raise ValueError(
            f"unknown operation: {name!r}. Known: {sorted(OPERATIONS.keys())}"
        )
    return OPERATIONS[name]


# ---------- default per-dtype pipelines ----------

DEFAULT_STRING_REDACT_OPERATIONS: list[str] = [
    "subject_name_scan", "generic_phi_scan",
]
DEFAULT_SUBJECT_NAME_REDACT_OPERATIONS: list[str] = [
    "parse_subject_name", "constant_replace",
]
DEFAULT_DATE_REDACT_OPERATIONS: list[str] = ["date_shift_to_base"]
DEFAULT_DATETIME_REDACT_OPERATIONS: list[str] = ["date_shift_to_base"]
DEFAULT_INTEGER_REDACT_OPERATIONS: list[str] = ["passthrough"]
DEFAULT_FLOAT_REDACT_OPERATIONS: list[str] = ["passthrough"]
DEFAULT_BOOLEAN_REDACT_OPERATIONS: list[str] = ["passthrough"]
DEFAULT_ENUM_REDACT_OPERATIONS: list[str] = ["passthrough"]
DEFAULT_BYTES_REDACT_OPERATIONS: list[str] = ["passthrough"]
DEFAULT_ZIP_CODE_REDACT_OPERATIONS: list[str] = ["zip_redact"]
DEFAULT_PHONE_REDACT_OPERATIONS: list[str] = ["phone_redact"]
DEFAULT_EMAIL_REDACT_OPERATIONS: list[str] = ["email_redact"]
DEFAULT_SSN_REDACT_OPERATIONS: list[str] = ["ssn_redact"]
DEFAULT_MRN_REDACT_OPERATIONS: list[str] = ["mrn_redact"]
DEFAULT_URL_REDACT_OPERATIONS: list[str] = ["url_redact"]
DEFAULT_IP_REDACT_OPERATIONS: list[str] = ["ip_redact"]

DEFAULT_OPERATIONS_BY_DTYPE: dict[str, list[str]] = {
    "string": DEFAULT_STRING_REDACT_OPERATIONS,
    "subject_name": DEFAULT_SUBJECT_NAME_REDACT_OPERATIONS,
    "date": DEFAULT_DATE_REDACT_OPERATIONS,
    "datetime": DEFAULT_DATETIME_REDACT_OPERATIONS,
    "integer": DEFAULT_INTEGER_REDACT_OPERATIONS,
    "float": DEFAULT_FLOAT_REDACT_OPERATIONS,
    "boolean": DEFAULT_BOOLEAN_REDACT_OPERATIONS,
    "enum": DEFAULT_ENUM_REDACT_OPERATIONS,
    "bytes": DEFAULT_BYTES_REDACT_OPERATIONS,
    "zip_code": DEFAULT_ZIP_CODE_REDACT_OPERATIONS,
    "phone": DEFAULT_PHONE_REDACT_OPERATIONS,
    "email": DEFAULT_EMAIL_REDACT_OPERATIONS,
    "ssn": DEFAULT_SSN_REDACT_OPERATIONS,
    "mrn": DEFAULT_MRN_REDACT_OPERATIONS,
    "url": DEFAULT_URL_REDACT_OPERATIONS,
    "ip": DEFAULT_IP_REDACT_OPERATIONS,
}


def default_operations_for_dtype(dtype_name: str) -> list[str]:
    if dtype_name not in DEFAULT_OPERATIONS_BY_DTYPE:
        raise ValueError(
            f"no default operations for dtype {dtype_name!r}. "
            f"Known: {sorted(DEFAULT_OPERATIONS_BY_DTYPE.keys())}"
        )
    return list(DEFAULT_OPERATIONS_BY_DTYPE[dtype_name])


__all__ = [
    "Operation",
    "OPERATIONS",
    "get_operation",
    "DEFAULT_BASE_DATE",
    "DEFAULT_OPERATIONS_BY_DTYPE",
    "default_operations_for_dtype",
]
