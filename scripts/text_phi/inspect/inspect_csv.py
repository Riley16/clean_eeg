"""PHI-safe per-column inspection of a REDCap CSV.

Combines the XML metadata (authoritative dtype/identifier/choices) with
empirical statistics on the actual data. **No raw values are ever emitted
to the output.** Enum levels are surfaced via `xml.enum_choices` (from the
XML CodeList, not the data). Everything else is aggregate — shape
signatures, length distributions, dtype parse rates, cardinality classes.

A `likely_free_text` flag is set when the shape distribution suggests the
column carries free-form prose; in that case suggested operations default
to the full string cleaning pipeline regardless of what REDCap declared.

Usage:
    python -m scripts.text_phi.inspect.inspect_csv \\
        --input DATA.csv \\
        --xml-meta redcap_meta.json \\
        --output inspection.json \\
        [--shape-min-freq 2]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..dtypes import DtypeError, get_dtype, known_dtypes
from .column_mapping import (
    REDCAP_SYSTEM_COLUMNS,
    duplicate_base_name_groups,
    enumerate_expected_labels,
    map_expected_to_csv,
)


# ---------- value shape signatures (no PHI) ----------

_CHAR_CLASS_MAP: dict[str, str] = {}  # lazy


def _char_class(c: str) -> str:
    if c.isalpha():
        return "A"
    if c.isdigit():
        return "9"
    if c.isspace():
        return " "
    return c  # keep punctuation literally


def value_shape(s: str) -> str:
    """Produce a shape signature for a single value.

    * `A` = run of letters (any case, any language) with length as `A{n}`
    * `9` = run of digits with length as `9{n}`
    * literal punctuation preserved verbatim
    * whitespace preserved as a single space per run

    Braces disambiguate lengths so adjacent runs never collide:
      - "John Smith" → "A{4} A{5}"
      - "2024-01-15" → "9{4}-9{2}-9{2}"
      - "js@ex.com"  → "A{2}@A{2}.A{3}"
      - "Test1"      → "A{4}9{1}"
    """
    if not s:
        return ""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        cls = _char_class(s[i])
        j = i + 1
        while j < n and _char_class(s[j]) == cls:
            j += 1
        run_len = j - i
        if cls in ("A", "9"):
            out.append(f"{cls}{{{run_len}}}")
        elif cls == " ":
            out.append(" ")
        else:
            out.append(cls * run_len)
        i = j
    return "".join(out)


# ---------- REDCap → our-dtype mapping ----------

# Simple time patterns: HH:MM (24-hour) and HH:MM:SS. Whitespace tolerated.
_TIME_PATTERN = re.compile(
    r"^\s*([01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\s*$"
)


def _all_match_time(values: list[str]) -> bool:
    if not values:
        return False
    return all(_TIME_PATTERN.match(v) for v in values)


def _parses_as_int(s: str) -> bool:
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def _parses_as_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _empirical_numeric_override(
    xml_item: dict[str, Any] | None,
    parse_rates: dict[str, dict[str, float]],
    filled_values: list[str],
    current_dtype: str,
) -> tuple[str, bool]:
    """Recover a bounded / non-string dtype for generic-text REDCap fields
    whose data is confirmably structured.

    Handles:
      * All-integer / all-float (REDCap `text` scored as numbers) — allows
        surrounding whitespace on values (`" 42 "` still counts as integer)
      * All-HH:MM (task start/end times)

    Override only fires when:
      * XML declared `field_type=text` (or `calc`) AND no `validation_type`
      * XML `data_type=text` (not already numeric)
      * Current inferred dtype is `string`
      * There are ≥ 2 filled values
      * 100% of stripped non-empty values parse under the target dtype
    """
    n_filled = len(filled_values)
    if xml_item is None or n_filled < 2 or current_dtype != "string":
        return current_dtype, False
    ft = (xml_item.get("field_type") or "").lower()
    vt = (xml_item.get("validation_type") or "").lower()
    dt = (xml_item.get("data_type") or "").lower()
    if ft not in ("text", "calc") or vt or dt != "text":
        return current_dtype, False

    # Whitespace-tolerant numeric check: strip each value before parsing.
    # Whitespace-only values collapse to "" and won't parse; those are
    # already excluded from `filled_values` so this is safe.
    stripped = [v.strip() for v in filled_values]
    n_int = sum(1 for v in stripped if _parses_as_int(v))
    if n_int == n_filled:
        return "integer", True
    n_float = sum(1 for v in stripped if _parses_as_float(v))
    if n_float == n_filled:
        return "float", True
    if _all_match_time(filled_values):
        return "time_string", True
    return current_dtype, False


def _redcap_to_our_dtype(item: dict[str, Any] | None) -> str:
    """Pick our internal dtype vocabulary label for a REDCap ItemDef.

    Returns `"string"` when the item is missing or when no more specific
    mapping applies.
    """
    if not item:
        return "string"

    dt = (item.get("data_type") or "").lower()
    ft = (item.get("field_type") or "").lower()
    vt = (item.get("validation_type") or "").lower()
    is_identifier = bool(item.get("identifier"))
    question = (item.get("question") or "").lower()

    # Highest-signal: REDCap identifier flag + name-like question.
    if is_identifier and ("name" in question or "patient" in question):
        return "subject_name"

    # Enum-ish REDCap field types.
    if ft in ("radio", "dropdown", "yesno", "truefalse", "select"):
        return "enum"
    if ft == "checkbox":
        return "boolean"

    # Text-with-validation.
    if vt == "email":
        return "email"
    if vt == "phone":
        return "phone"
    if vt in ("zipcode", "zip", "zip_code"):
        return "zip_code"
    if vt == "mrn":
        return "mrn"
    if vt in ("int", "integer"):
        return "integer"
    if vt in ("float", "number"):
        return "float"
    if vt.startswith("date_"):
        return "date"
    if vt.startswith("datetime_"):
        return "datetime"

    # Data type at the XML level.
    if dt == "integer":
        return "integer"
    if dt == "float":
        return "float"
    if dt == "boolean":
        return "boolean"
    if dt == "date":
        return "date"

    # Free-text / attachment / calc / textarea / notes → string.
    return "string"


# ---------- suggested redaction operations ----------

_PSEUDO_DTYPES: frozenset[str] = frozenset({"time_string"})


def _suggest_operations(
    our_dtype: str, item: dict[str, Any] | None
) -> list[str]:
    """Suggest a default operations list for the derived dtype + REDCap
    context. Uses the same operation names as `scripts.text_phi.operations`."""
    is_identifier = bool(item and item.get("identifier"))
    ft = (item.get("field_type") if item else "" or "").lower()

    if our_dtype == "subject_name":
        return ["parse_subject_name", "constant_replace"]
    # Pseudo-dtypes (time_string, ...): structural, no PHI risk → passthrough.
    if our_dtype in _PSEUDO_DTYPES:
        return ["passthrough"]
    if our_dtype == "date":
        return ["date_shift_to_base"]
    if our_dtype == "datetime":
        return ["date_shift_to_base"]
    if our_dtype == "email":
        return ["email_redact"]
    if our_dtype == "phone":
        return ["phone_redact"]
    if our_dtype == "zip_code":
        return ["zip_redact"]
    if our_dtype == "mrn":
        return ["mrn_redact"]

    # Identifier-flagged but not a name/date/typed-PHI: replace with a fixed
    # placeholder rather than passthrough.
    if is_identifier:
        return ["constant_replace"]

    # File-attachment fields carry filenames/IDs — treat conservatively.
    if ft == "file":
        return ["constant_replace"]

    # Free-text (notes/textarea) needs the generic PHI scan.
    if ft in ("notes", "textarea") and our_dtype == "string":
        return ["subject_name_scan", "generic_phi_scan"]

    # Enum / boolean / numeric / etc. — pass through by default.
    if our_dtype in ("enum", "boolean", "integer", "float"):
        return ["passthrough"]

    return "default"  # let the schema layer pick per-dtype default


# ---------- per-column analysis ----------

def _dtype_parse_rates(values: list[str]) -> dict[str, dict[str, float]]:
    """For each of our internal dtypes, compute what fraction of non-empty
    values parse without error."""
    if not values:
        return {dt: {"n": 0, "rate": 0.0} for dt in known_dtypes()}
    out: dict[str, dict[str, float]] = {}
    for dt in known_dtypes():
        d = get_dtype(dt)
        ok = 0
        for v in values:
            try:
                d.validate(v)
                ok += 1
            except DtypeError:
                pass
        out[dt] = {"n": ok, "rate": ok / len(values)}
    return out


def _length_stats(values: list[str]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    lengths = [len(v) for v in values]
    return {
        "n": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "p95": statistics.quantiles(lengths, n=20)[-1] if len(lengths) >= 20 else max(lengths),
    }


def _cardinality_class(n_unique: int, n_filled: int) -> str:
    if n_filled == 0:
        return "empty"
    if n_unique == 1:
        return "constant"
    if n_unique <= 10:
        return "very_low"
    if n_unique <= 40:
        return "low"
    if n_unique / n_filled >= 0.95:
        return "unique"
    if n_unique <= 200:
        return "medium"
    return "high"


def _phi_hint_from_name(csv_col: str) -> str | None:
    """Return a short hint if the column name matches common PHI patterns.
    Used as a secondary signal alongside REDCap's identifier flag."""
    lc = csv_col.lower()
    checks: list[tuple[str, str]] = [
        (r"\bname\b", "column name mentions 'name'"),
        (r"\b(dob|birth)", "column name mentions birth"),
        (r"\bmrn\b", "column name mentions MRN"),
        (r"\bssn\b", "column name mentions SSN"),
        (r"\b(phone|tel|mobile|cell)\b", "column name mentions phone"),
        (r"\bemail\b", "column name mentions email"),
        (r"\baddress\b", "column name mentions address"),
        (r"\b(zip|zipcode|postal)\b", "column name mentions ZIP"),
        (r"\b(comment|note|description|impression|history)s?\b", "free-text label"),
    ]
    for pattern, msg in checks:
        if re.search(pattern, lc):
            return msg
    return None


def _classify_free_text(
    n_unique_shapes: int,
    max_shape_length: int,
    mean_value_length: float,
    n_filled: int,
) -> bool:
    """A column is "likely free-form text" (and therefore needs full PHI
    scanning, never raw-value exposure) if its content is highly variable
    and long. Heuristic thresholds are conservative in the PHI-safe
    direction: false positives (over-treating structured fields as free
    text) just get more scanning; false negatives leak PHI."""
    if n_filled < 2:
        return False
    # A value length above ~50 chars in aggregate almost never comes from
    # an enum, ID, or timestamp field — it's prose.
    if mean_value_length >= 50:
        return True
    if n_unique_shapes >= 10 and max_shape_length >= 30:
        return True
    if n_unique_shapes >= 5 and mean_value_length >= 25:
        return True
    return False


def analyze_column(
    df: pd.DataFrame,
    col: str,
    xml_item: dict[str, Any] | None,
    shape_min_freq: int,
) -> dict[str, Any]:
    """Compute PHI-safe metadata for one column.

    Values themselves are NEVER emitted. Only:
      * aggregate stats (fill rate, cardinality, length distribution)
      * shape signatures (`A{4} A{5}` — letter/digit run structure, no
        characters leaked)
      * REDCap-declared enum choices (from the XML CodeList, not from data)
      * empirical dtype parse rates
    Enum levels visible via `xml.enum_choices` (safe — from metadata).
    """
    series = df[col].astype(str).where(df[col].notna(), "")
    n_total = len(series)
    filled = [v for v in series if v != ""]
    n_filled = len(filled)
    unique = list(dict.fromkeys(filled))
    n_unique = len(unique)

    parse_rates = _dtype_parse_rates(filled)
    length_stats = _length_stats(filled)

    # Shape signatures. Stats over ALL shapes computed first so tiny
    # unique-per-row shapes (which get filtered out by min_freq) still
    # contribute to the free-text heuristic.
    all_shape_counts = Counter(value_shape(v) for v in filled)
    n_unique_shapes = len(all_shape_counts)
    max_shape_length = max((len(s) for s in all_shape_counts), default=0)
    top_shapes = [
        {"shape": s, "count": c}
        for s, c in all_shape_counts.most_common(15)
        if c >= shape_min_freq
    ]

    mean_value_length = float(length_stats.get("mean", 0) or 0)
    likely_free_text = _classify_free_text(
        n_unique_shapes=n_unique_shapes,
        max_shape_length=max_shape_length,
        mean_value_length=mean_value_length,
        n_filled=n_filled,
    )

    our_dtype = _redcap_to_our_dtype(xml_item)
    our_dtype, dtype_empirical_override = _empirical_numeric_override(
        xml_item, parse_rates, filled, our_dtype,
    )
    suggested_ops = _suggest_operations(our_dtype, xml_item)
    # If the empirical shape distribution says free-form text, override
    # suggested ops with full string cleaning — UNLESS the REDCap FieldType
    # is `file` (which stores filenames / attachment IDs, not prose) or
    # the field is REDCap-flagged as an identifier (which we already
    # forced to `constant_replace` and shouldn't downgrade to a scan).
    xml_ft = ((xml_item or {}).get("field_type") or "").lower()
    xml_identifier = bool(xml_item and xml_item.get("identifier"))
    if (
        likely_free_text
        and our_dtype == "string"
        and xml_ft != "file"
        and not xml_identifier
    ):
        suggested_ops = ["subject_name_scan", "generic_phi_scan"]

    is_identifier = bool(xml_item and xml_item.get("identifier"))
    phi_hint_name = _phi_hint_from_name(col)
    phi_hint_parts: list[str] = []
    if is_identifier:
        phi_hint_parts.append("REDCap Identifier=y")
    if phi_hint_name:
        phi_hint_parts.append(phi_hint_name)
    if likely_free_text:
        phi_hint_parts.append(
            f"likely free-form text ({n_unique_shapes} distinct shapes, "
            f"max shape length {max_shape_length}, mean value length "
            f"{mean_value_length:.0f})"
        )
    phi_hint = "; ".join(phi_hint_parts) or None

    # REDCap-declared enum choices (safe — from XML, not data).
    enum_choices = None
    if xml_item and xml_item.get("code_list"):
        cl = xml_item["code_list"]
        if cl.get("items"):
            enum_choices = cl["items"]

    return {
        "csv_column": col,
        "xml": {
            "item_oid": xml_item.get("oid") if xml_item else None,
            "variable": xml_item.get("variable") if xml_item else None,
            "form": xml_item.get("form") if xml_item else None,
            "data_type": xml_item.get("data_type") if xml_item else None,
            "field_type": xml_item.get("field_type") if xml_item else None,
            "validation_type": xml_item.get("validation_type") if xml_item else None,
            "identifier": is_identifier,
            "required": bool(xml_item and xml_item.get("required")),
            "question": xml_item.get("question") if xml_item else None,
            "checkbox_label": xml_item.get("checkbox_label") if xml_item else None,
            "enum_choices": enum_choices,
        },
        "stats": {
            "n_total": n_total,
            "n_filled": n_filled,
            "n_empty": n_total - n_filled,
            "fill_rate": n_filled / n_total if n_total else 0.0,
            "n_unique": n_unique,
            "cardinality_class": _cardinality_class(n_unique, n_filled),
            "length_stats": length_stats,
            "n_unique_shapes": n_unique_shapes,
            "max_shape_length": max_shape_length,
        },
        "empirical_dtype_parse_rates": parse_rates,
        "top_value_shapes": top_shapes,
        "likely_free_text": likely_free_text,
        "suggested_our_dtype": our_dtype,
        "dtype_empirical_override": dtype_empirical_override,
        "suggested_operations": suggested_ops,
        "phi_hint": phi_hint,
    }


# ---------- cross-column analyses ----------

def _cofill_clusters(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    """Group columns by their exact fill fingerprint across rows.

    Two columns are in the same cluster if they're populated on *exactly*
    the same rows. This reveals REDCap record-type structure: patient-info
    columns share one fingerprint, testing-day columns share another.
    Returns clusters with ≥2 columns.
    """
    # Build a bit-vector-ish tuple for each column (using row indices where filled).
    fp: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for c in cols:
        col_series = df[c].notna() & (df[c].astype(str) != "")
        idx = tuple(int(i) for i, v in enumerate(col_series) if v)
        fp[idx].append(c)

    clusters = []
    for cid, (idx, members) in enumerate(sorted(
        fp.items(), key=lambda kv: -len(kv[1])
    )):
        if len(members) < 2:
            continue
        clusters.append({
            "cluster_id": cid,
            "n_rows_filled": len(idx),
            "n_columns": len(members),
            "columns": members,
        })
    return clusters


def _event_name_breakdown(df: pd.DataFrame, event_col: str | None) -> dict[str, Any] | None:
    """If the CSV has an Event Name column, report unique event values
    (safe — event names are structural) and per-event column fill rates."""
    if event_col is None or event_col not in df.columns:
        return None
    events = df[event_col].astype(str).where(df[event_col].notna(), "")
    unique_events = sorted(events.unique())
    per_event: dict[str, dict[str, int]] = {}
    for ev in unique_events:
        mask = events == ev
        sub = df[mask]
        cols_filled = {
            c: int((sub[c].notna() & (sub[c].astype(str) != "")).sum())
            for c in df.columns
            if (sub[c].notna() & (sub[c].astype(str) != "")).any()
        }
        per_event[ev] = {
            "n_rows": int(mask.sum()),
            "n_columns_with_any_data": len(cols_filled),
            "columns_populated": sorted(cols_filled.keys()),
        }
    return {
        "event_column": event_col,
        "unique_events": unique_events,
        "per_event": per_event,
    }


def _checkbox_groups(meta: dict[str, Any]) -> dict[str, list[str]]:
    """From the XML, group all checkbox items sharing a REDCap variable
    (the common `race___N` pattern)."""
    by_var: dict[str, list[str]] = defaultdict(list)
    for oid, item in meta["items"].items():
        if item.get("field_type") == "checkbox" and item.get("variable"):
            by_var[item["variable"]].append(oid)
    return {v: sorted(items) for v, items in by_var.items() if len(items) > 1}


# ---------- main ----------

def inspect(
    csv_path: str | Path,
    xml_meta: dict[str, Any],
    shape_min_freq: int = 2,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    csv_cols = list(df.columns)

    expected = enumerate_expected_labels(xml_meta)
    mapping = map_expected_to_csv(expected, csv_cols)
    col_to_item: dict[str, dict[str, Any]] = {}
    for pair in mapping["pairs"]:
        col_to_item[pair["csv_column"]] = xml_meta["items"][pair["item_oid"]]

    columns_out: dict[str, dict[str, Any]] = {}
    for c in csv_cols:
        columns_out[c] = analyze_column(
            df, c, col_to_item.get(c),
            shape_min_freq=shape_min_freq,
        )

    # Cross-column analyses.
    event_col = "Event Name" if "Event Name" in csv_cols else None
    structural = {
        "duplicate_base_name_groups": duplicate_base_name_groups(csv_cols),
        "cofill_clusters": _cofill_clusters(df, csv_cols),
        "event_name_breakdown": _event_name_breakdown(df, event_col),
        "checkbox_groups": _checkbox_groups(xml_meta),
    }

    return {
        "meta": {
            "source_csv": str(csv_path),
            "n_rows": len(df),
            "n_columns": len(csv_cols),
            "system_columns": mapping["system_columns"],
            "n_unmapped_xml": mapping["n_unmapped_xml"],
            "n_unmapped_csv": mapping["n_unmapped_csv"],
            "shape_min_freq": shape_min_freq,
            "raw_values_emitted": False,
        },
        "unmapped_xml": mapping["unmapped_xml"],
        "unmapped_csv": mapping["unmapped_csv"],
        "columns": columns_out,
        "structural_analysis": structural,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to REDCap CSV")
    p.add_argument("--xml-meta", required=True,
                   help="Path to redcap_meta.json (from redcap_meta.py)")
    p.add_argument("--output", required=True,
                   help="Path to write inspection.json")
    p.add_argument("--shape-min-freq", type=int, default=2,
                   help="Only report value-shape signatures appearing at "
                        "least this many times (default 2).")
    args = p.parse_args(argv)

    xml_meta = json.loads(Path(args.xml_meta).read_text(encoding="utf-8"))
    result = inspect(
        args.input, xml_meta,
        shape_min_freq=args.shape_min_freq,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(
        f"Wrote {out} — {result['meta']['n_columns']} columns, "
        f"{result['meta']['n_rows']} rows.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
