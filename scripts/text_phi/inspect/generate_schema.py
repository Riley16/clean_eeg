"""Generate a text_phi schema JSON from a REDCap inspection.json.

Applies design decisions on top of `suggested_our_dtype` / `suggested_operations`:

  * `subject_number` → passthrough integer (retains cross-row linkage).
  * `redcap_event_name`, `redcap_data_access_group` → passthrough (structural,
    not PHI).
  * `subject_name` → `[parse_subject_name, constant_replace]`.
  * `implant_date` → `date_shift_to_base` (the anchor).
  * All other date fields → `date_shift_relative_to_stay_start` with
    `depends_on: {stay_start_field: implant_date}`, preserving intervals
    within each subject.
  * `file`-type identifier fields → `constant_replace` (attachment IDs).
  * `textarea` / `notes` and other likely-free-text `text` fields →
    `[subject_name_scan, generic_phi_scan]` with
    `depends_on: {subject_name_field: subject_name}`.
  * Structural fields (integer, float, enum, boolean, non-PHI text) →
    passthrough.

Usage:
    python -m scripts.text_phi.inspect.generate_schema \\
        --inspection temp/inspection.json \\
        --output temp/notes.schema.json \\
        [--anchor-date implant_date] [--name-field subject_name]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Structural / passthrough columns we shouldn't touch.
STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "subject_number",
    "redcap_event_name",
    "redcap_data_access_group",
    "redcap_repeat_instrument",
    "redcap_repeat_instance",
)

# REDCap FieldTypes that are not free text and should stay passthrough
# unless flagged as identifier.
_STRUCTURAL_FIELD_TYPES = frozenset({
    "radio", "dropdown", "yesno", "truefalse", "select", "checkbox",
    "slider", "calc",
})


# Pseudo-dtypes emitted by inspection that must collapse to a real schema
# dtype. Value → real dtype the schema should use.
_PSEUDO_DTYPE_ALIASES: dict[str, str] = {
    "time_string": "string",
}


def _pick_dtype(inspection_col: dict[str, Any]) -> str:
    """Prefer REDCap-declared dtype from inspection, fall back to 'string'."""
    d = inspection_col.get("suggested_our_dtype") or "string"
    return _PSEUDO_DTYPE_ALIASES.get(d, d)


def _pick_operations(
    col_name: str,
    inspection_col: dict[str, Any],
    anchor_date: str,
    name_field: str,
) -> tuple[list[Any], dict[str, str]]:
    """Return (operations_list, depends_on_dict) for one column."""
    xml = inspection_col.get("xml", {})
    dtype = _pick_dtype(inspection_col)
    # Pseudo-dtypes flagged by inspection (e.g. time_string) → passthrough.
    raw_inspection_dtype = inspection_col.get("suggested_our_dtype") or "string"
    if raw_inspection_dtype in _PSEUDO_DTYPE_ALIASES:
        return ["passthrough"], {}
    identifier = bool(xml.get("identifier"))
    field_type = (xml.get("field_type") or "").lower()
    likely_free_text = bool(inspection_col.get("likely_free_text"))

    # Structural / system columns → always passthrough.
    if col_name in STRUCTURAL_COLUMNS:
        return ["passthrough"], {}

    # Name field itself.
    if col_name == name_field or dtype == "subject_name":
        return ["parse_subject_name", "constant_replace"], {}

    # Anchor date: shift-to-base (no dep).
    if col_name == anchor_date:
        return ["date_shift_to_base"], {}

    # Other date/datetime fields: shift relative to the anchor.
    if dtype in ("date", "datetime"):
        return (
            ["date_shift_relative_to_stay_start"],
            {"stay_start_field": anchor_date},
        )

    # Typed PHI dtypes.
    if dtype == "email":
        return ["email_redact"], {}
    if dtype == "phone":
        return ["phone_redact"], {}
    if dtype == "zip_code":
        return ["zip_redact"], {}
    if dtype == "mrn":
        return ["mrn_redact"], {}

    # REDCap identifier flag: overrides other logic. File attachments and
    # explicit identifier strings both go to constant_replace.
    if identifier:
        return ["constant_replace"], {}

    # File-type without identifier flag (rare) — still constant_replace.
    if field_type == "file":
        return ["constant_replace"], {}

    # Structured field types (checkbox, radio, ...) → passthrough.
    if field_type in _STRUCTURAL_FIELD_TYPES:
        return ["passthrough"], {}

    # Numeric / enum / boolean dtypes → passthrough. These are confirmed by
    # the REDCap dtype declaration to be a bounded, non-PHI value space.
    if dtype in ("integer", "float", "boolean", "enum"):
        return ["passthrough"], {}

    # Every remaining string column runs the full scan. Even short strings
    # can carry a name (e.g., a one-word "Provider" field, a "Study site"
    # free-text override, or a comment box with a single name in it).
    # Passthrough is only chosen when REDCap has declared the value space
    # bounded (radio/dropdown/checkbox/yesno/truefalse).
    if dtype == "string":
        return (
            ["subject_name_scan", "generic_phi_scan"],
            {"subject_name_field": name_field},
        )

    # Any dtype we didn't handle above — conservative default: full scan.
    return (
        ["subject_name_scan", "generic_phi_scan"],
        {"subject_name_field": name_field},
    )


def build_schema(
    inspection: dict[str, Any],
    anchor_date: str,
    name_field: str,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for col_name, col in inspection["columns"].items():
        dtype = _pick_dtype(col)
        # If the field is the name field itself, our internal dtype is
        # `subject_name` regardless of what REDCap declared.
        if col_name == name_field:
            dtype = "subject_name"
        # Ensure anchor date is a date dtype.
        if col_name == anchor_date and dtype not in ("date", "datetime"):
            dtype = "date"

        ops, deps = _pick_operations(col_name, col, anchor_date, name_field)
        spec: dict[str, Any] = {
            "dtype": dtype,
            "description": (col.get("xml") or {}).get("question") or col_name,
            "operations": ops,
        }
        if deps:
            spec["depends_on"] = deps
        fields[col_name] = spec

    return {
        "schema_version": "1",
        "format": "csv",
        "unknown_field_policy": "error",
        "missing_field_policy": "error",
        "fields": fields,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspection", required=True,
                   help="Path to inspection.json from inspect_csv.py")
    p.add_argument("--output", required=True,
                   help="Path to write the schema JSON")
    p.add_argument("--anchor-date", default="implant_date",
                   help="Date field that all other dates shift relative to. "
                        "(Default: implant_date)")
    p.add_argument("--name-field", default="subject_name",
                   help="Column carrying the subject name (Default: subject_name)")
    args = p.parse_args(argv)

    inspection = json.loads(Path(args.inspection).read_text(encoding="utf-8"))
    schema = build_schema(
        inspection,
        anchor_date=args.anchor_date,
        name_field=args.name_field,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    # Summary stats.
    from collections import Counter
    ops_counts = Counter(tuple(f["operations"]) for f in schema["fields"].values())
    print(f"Wrote {out} — {len(schema['fields'])} fields.", file=sys.stderr)
    print(f"Operation pipelines used:", file=sys.stderr)
    for ops, n in ops_counts.most_common():
        print(f"  {n:4d}× {list(ops)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
