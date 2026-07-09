"""Schema: versioned JSON description of a file's fields, their dtypes, and
the redaction operations to apply.

A Schema is a first-class object. Load it, validate it, then hand it to the
records-processing loop. Validation runs *before* any file I/O so config
errors (unknown operation, unregistered dtype, missing dep role, cycle in
depends_on) fail fast.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dtypes import DtypeError, known_dtypes
from .operations import (
    OPERATIONS,
    default_operations_for_dtype,
)


SCHEMA_VERSION = "1"
DEFAULT_OPERATIONS_SENTINEL = "default"

UNKNOWN_FIELD_POLICIES = frozenset(["error", "passthrough", "redact_as_string"])
MISSING_FIELD_POLICIES = frozenset(["error", "ignore"])


class SchemaError(ValueError):
    """Schema is structurally invalid or references unknown symbols."""


@dataclass(frozen=True)
class OperationCall:
    """One entry in a field's operations list, resolved to (name, params)."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    description: str
    operations: tuple[OperationCall, ...]
    depends_on: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Schema:
    schema_version: str
    format: str
    unknown_field_policy: str
    missing_field_policy: str
    fields: dict[str, FieldSpec]
    raw: dict[str, Any]  # the exact source dict (for canonical hashing)

    # ---------- loading ----------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Schema":
        _require_key(raw, "schema_version", str)
        _require_key(raw, "format", str)
        _require_key(raw, "fields", dict)
        version = raw["schema_version"]
        if version != SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema_version {version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

        unknown = raw.get("unknown_field_policy", "error")
        missing = raw.get("missing_field_policy", "error")
        if unknown not in UNKNOWN_FIELD_POLICIES:
            raise SchemaError(
                f"unknown_field_policy must be one of "
                f"{sorted(UNKNOWN_FIELD_POLICIES)}; got {unknown!r}"
            )
        if missing not in MISSING_FIELD_POLICIES:
            raise SchemaError(
                f"missing_field_policy must be one of "
                f"{sorted(MISSING_FIELD_POLICIES)}; got {missing!r}"
            )

        fields_map: dict[str, FieldSpec] = {}
        for name, spec in raw["fields"].items():
            fields_map[name] = _parse_field_spec(name, spec)

        schema = cls(
            schema_version=version,
            format=raw["format"],
            unknown_field_policy=unknown,
            missing_field_policy=missing,
            fields=fields_map,
            raw=raw,
        )
        schema.validate()
        return schema

    @classmethod
    def load(cls, path: str | Path) -> "Schema":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    # ---------- validation ----------

    def validate(self) -> None:
        """Comprehensive checks. Raises SchemaError on any failure.

        Called once at the end of from_dict, but public so callers can
        re-run after in-memory mutations (there shouldn't be any — Schema
        is frozen — but defensive).
        """
        # Every dtype is known.
        for fs in self.fields.values():
            if fs.dtype not in known_dtypes():
                raise SchemaError(
                    f"field {fs.name!r} has unknown dtype {fs.dtype!r}. "
                    f"Known: {known_dtypes()}"
                )

        # Every operation is registered and dtype-compatible.
        for fs in self.fields.values():
            for op_call in fs.operations:
                if op_call.name not in OPERATIONS:
                    raise SchemaError(
                        f"field {fs.name!r} references unknown operation "
                        f"{op_call.name!r}. Known: {sorted(OPERATIONS.keys())}"
                    )
                op = OPERATIONS[op_call.name]
                if op.allowed_dtypes is not None and fs.dtype not in op.allowed_dtypes:
                    raise SchemaError(
                        f"operation {op_call.name!r} is not compatible with "
                        f"dtype {fs.dtype!r} on field {fs.name!r}. "
                        f"Allowed: {sorted(op.allowed_dtypes)}"
                    )
                # All required roles must appear in depends_on.
                missing_roles = op.required_roles - fs.depends_on.keys()
                if missing_roles:
                    raise SchemaError(
                        f"operation {op_call.name!r} on field {fs.name!r} "
                        f"requires depends_on role(s) {sorted(missing_roles)}"
                    )

        # depends_on values reference real fields.
        for fs in self.fields.values():
            for role, target in fs.depends_on.items():
                if target not in self.fields:
                    raise SchemaError(
                        f"field {fs.name!r} depends_on[{role!r}]={target!r}, "
                        f"but no such field exists in the schema"
                    )

        # No cycles in the depends_on graph.
        try:
            self.processing_order()
        except SchemaError:
            raise

    # ---------- scheduling ----------

    def processing_order(self) -> list[str]:
        """Topological sort of fields by depends_on. Cycle → SchemaError."""
        graph: dict[str, set[str]] = {n: set() for n in self.fields}
        for name, fs in self.fields.items():
            for target in fs.depends_on.values():
                graph[name].add(target)

        result: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                raise SchemaError(
                    f"cycle detected in field dependency graph at {node!r}"
                )
            visiting.add(node)
            for dep in sorted(graph[node]):
                visit(dep)
            visiting.remove(node)
            visited.add(node)
            result.append(node)

        for node in sorted(self.fields):
            visit(node)
        return result

    # ---------- canonical hash for audit ----------

    def sha256(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------- parsing helpers ----------

def _require_key(d: dict[str, Any], key: str, expected_type: type) -> None:
    if key not in d:
        raise SchemaError(f"schema missing required key {key!r}")
    if not isinstance(d[key], expected_type):
        raise SchemaError(
            f"schema key {key!r} must be {expected_type.__name__}, "
            f"got {type(d[key]).__name__}"
        )


def _parse_field_spec(name: str, raw: Any) -> FieldSpec:
    if not isinstance(raw, dict):
        raise SchemaError(f"field {name!r} spec must be a dict")
    if "dtype" not in raw:
        raise SchemaError(f"field {name!r} missing required key 'dtype'")

    dtype = raw["dtype"]
    if not isinstance(dtype, str):
        raise SchemaError(f"field {name!r} dtype must be a string")
    if dtype not in known_dtypes():
        raise SchemaError(
            f"field {name!r} has unknown dtype {dtype!r}. "
            f"Known: {known_dtypes()}"
        )

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise SchemaError(f"field {name!r} description must be a string")

    raw_ops = raw.get("operations", DEFAULT_OPERATIONS_SENTINEL)
    operations = _resolve_operations(name, dtype, raw_ops)

    depends_on = raw.get("depends_on", {})
    if not isinstance(depends_on, dict):
        raise SchemaError(f"field {name!r} depends_on must be a dict")
    depends_on = {str(k): str(v) for k, v in depends_on.items()}

    return FieldSpec(
        name=name,
        dtype=dtype,
        description=description,
        operations=operations,
        depends_on=depends_on,
    )


def _resolve_operations(
    field_name: str, dtype: str, raw_ops: Any
) -> tuple[OperationCall, ...]:
    if raw_ops is None:
        raise SchemaError(
            f"field {field_name!r} operations is null; use "
            f"{DEFAULT_OPERATIONS_SENTINEL!r} for the dtype default, or an "
            f"explicit list."
        )
    if raw_ops == DEFAULT_OPERATIONS_SENTINEL:
        try:
            names = default_operations_for_dtype(dtype)
        except ValueError as e:
            raise SchemaError(str(e)) from e
        return tuple(OperationCall(name=n) for n in names)
    if not isinstance(raw_ops, list):
        raise SchemaError(
            f"field {field_name!r} operations must be a list or the string "
            f"{DEFAULT_OPERATIONS_SENTINEL!r}; got {type(raw_ops).__name__}"
        )
    calls: list[OperationCall] = []
    for entry in raw_ops:
        if isinstance(entry, str):
            calls.append(OperationCall(name=entry))
        elif isinstance(entry, dict):
            if "name" not in entry:
                raise SchemaError(
                    f"field {field_name!r} operation entry missing 'name'"
                )
            name = entry["name"]
            if not isinstance(name, str):
                raise SchemaError(
                    f"field {field_name!r} operation 'name' must be a string"
                )
            params = entry.get("params", {})
            if not isinstance(params, dict):
                raise SchemaError(
                    f"field {field_name!r} operation {name!r} params must "
                    f"be a dict"
                )
            calls.append(OperationCall(name=name, params=dict(params)))
        else:
            raise SchemaError(
                f"field {field_name!r} operations entries must be str or "
                f"dict, got {type(entry).__name__}"
            )
    return tuple(calls)


# ---------- derivation (bootstrap a starting schema) ----------

# Column names commonly containing a subject's name, matched
# case-insensitively during derivation.
_NAME_HINTS = frozenset({
    "name", "full_name", "patient_name", "subject_name", "patient", "subject",
})

# Column names commonly containing dates.
_DATE_HINTS = frozenset({
    "date", "admission_date", "discharge_date", "birth_date", "dob",
    "note_date", "start_date", "end_date", "recorded_date",
})


def derive_schema_from_columns(columns: list[str], format_name: str) -> Schema:
    """Emit a Schema with permissive defaults: every field is `dtype: string`
    with `operations: "default"`, except known-role hints (name, date). All
    descriptions carry a `TODO: verify` marker so the user knows to review.
    """
    fields_raw: dict[str, Any] = {}
    for col in columns:
        low = col.lower()
        if low in _NAME_HINTS:
            dtype = "subject_name"
        elif low in _DATE_HINTS:
            dtype = "date"
        else:
            dtype = "string"
        fields_raw[col] = {
            "dtype": dtype,
            "description": f"TODO: verify — auto-derived as dtype {dtype}",
            "operations": DEFAULT_OPERATIONS_SENTINEL,
        }
    raw = {
        "schema_version": SCHEMA_VERSION,
        "format": format_name,
        "unknown_field_policy": "error",
        "missing_field_policy": "error",
        "fields": fields_raw,
    }
    return Schema.from_dict(raw)


def _has_dtype_error(dtype_name: str, value: str) -> bool:
    try:
        from .dtypes import get_dtype
        get_dtype(dtype_name).validate(value)
        return False
    except DtypeError:
        return True
