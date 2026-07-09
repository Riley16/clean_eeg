"""Map REDCap XML ItemDefs → the CSV column labels a label-exported CSV
would carry, and report mismatches.

Key concept: in a label-exported REDCap CSV, the column header for most
fields is the XML `Question` text. Two exceptions:

  * Checkbox fields: `race___8` (Question "Race") becomes the column
    `"Race (choice=Black)"` (choice label pulled from CheckboxChoices).
  * When two ItemDefs produce the same header label, pandas adds a
    `.1`, `.2`, ... suffix in order of appearance. So "Start Time" appearing
    3 times in the XML becomes CSV columns `["Start Time", "Start Time.1",
    "Start Time.2"]`. Any inconsistency (e.g. someone typed `"Start time"`
    lowercase in REDCap) will create a NEW deduplication group.

This module compares the expected labels (from the XML) against actual CSV
column names (from a list file). Mismatches in either direction are
reported so the data owner can investigate before schema design.

Usage:
    python -m scripts.text_phi.inspect.column_mapping \\
        --xml-meta redcap_meta.json --columns columns.txt \\
        --output mapping.json
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_DEDUP_SUFFIX = re.compile(r"\.(\d+)$")

# REDCap adds these to every export; they aren't in the XML ItemDefs.
# Both the label-mode form ("Event Name") and the raw-variable-mode form
# ("redcap_event_name") appear depending on the export flavor.
REDCAP_SYSTEM_COLUMNS: tuple[str, ...] = (
    # Label-mode headers
    "Event Name",
    "Data Access Group",
    "Record ID",
    "Study ID",
    "Repeat Instrument",
    "Repeat Instance",
    "Survey Identifier",
    # Raw-variable-mode headers
    "redcap_event_name",
    "redcap_data_access_group",
    "redcap_repeat_instrument",
    "redcap_repeat_instance",
    "redcap_survey_identifier",
)


def _normalize_label(s: str | None) -> str:
    """Canonical form for label matching.

    * HTML-unescape (CSV export uses `&#039;` for `'`, etc.)
    * Collapse any run of whitespace (spaces, tabs, newlines) to one space
    * Strip outer whitespace

    The collapse is aggressive but consistent across both sides — the CSV
    export replaces newlines with double spaces, and some XML questions have
    trailing whitespace before the checkbox choice suffix gets appended.
    Collapsing to single space normalizes both.
    """
    if s is None:
        return ""
    s = html.unescape(s)
    # REDCap's CSV export converts straight `"` in field labels to `'`.
    # Normalize both sides to `'` so quoted phrases match.
    s = s.replace('"', "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def expected_label(item: dict[str, Any]) -> str:
    """CSV column label a label-exported REDCap CSV would carry for this
    ItemDef (before any deduplication suffix). Normalized to the same form
    as CSV columns so matches succeed. Handles the empty-checkbox-label
    case (`CheckboxChoices="1, "` → `(choice=)`)."""
    question = _normalize_label(item.get("question"))
    field_type = item.get("field_type")
    if field_type == "checkbox":
        cb_label = item.get("checkbox_label")
        # Empty label is legitimate — REDCap emits `(choice=)` for it.
        if cb_label is not None:
            return f"{question} (choice={_normalize_label(cb_label)})"
    return question


def enumerate_expected_labels(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """List every expected column label in XML declaration order, tagged
    with the ItemDef that produced it."""
    out: list[dict[str, Any]] = []
    # Preserve XML order: iterate items dict (Python dicts preserve
    # insertion order and parse_metadata inserts in XML order).
    for oid, item in meta["items"].items():
        label = expected_label(item)
        out.append({
            "item_oid": oid,
            "expected_label": label,
            "form": item.get("form"),
            "field_type": item.get("field_type"),
            "identifier": item.get("identifier", False),
        })
    return out


def load_column_list(path: str | Path) -> list[str]:
    """Load column names from either a JSON array or a Python-list-literal
    (single-quoted, like the current temp/redcap_data_fields.txt)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    # Try JSON first, then Python-literal.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as e:
            # Fall back: one column per non-blank line, ignore commas/quotes.
            parsed = [
                _strip_quotes(line.rstrip(","))
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith(("[", "]"))
            ]
    if not isinstance(parsed, list):
        raise ValueError(f"{p}: expected a list of column names")
    return [str(x) for x in parsed]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s


def base_label(csv_col: str) -> str:
    """Strip a trailing `.N` deduplication suffix and normalize the CSV
    column name into the same canonical form as `expected_label`."""
    m = _DEDUP_SUFFIX.search(csv_col)
    if m is None:
        stripped = csv_col
    else:
        stripped = csv_col[: m.start()]
    return _normalize_label(stripped)


def _detect_raw_mode(csv_columns: list[str], meta_oids: set[str]) -> bool:
    """A REDCap CSV exported in *raw-variable* mode uses ItemDef OIDs as
    column headers (e.g. `subject_number`, `race___8`). Detect by checking
    what fraction of columns match an XML OID exactly."""
    non_system = [c for c in csv_columns if c not in REDCAP_SYSTEM_COLUMNS]
    if not non_system:
        return False
    matches = sum(1 for c in non_system if c in meta_oids)
    return matches / len(non_system) >= 0.5


def map_expected_to_csv(
    expected: list[dict[str, Any]],
    csv_columns: list[str],
) -> dict[str, Any]:
    """Match expected labels to CSV columns.

    Handles both REDCap CSV export modes:
      * **Label mode** — CSV headers are the ItemDef Question texts (with
        checkbox suffixes and `.N` dedup). Base-label + zip matching.
      * **Raw-variable mode** — CSV headers ARE the ItemDef OIDs verbatim
        (e.g. `subject_number`, `race___8`). Direct OID match.
    """
    meta_oids = {e["item_oid"] for e in expected}
    raw_mode = _detect_raw_mode(csv_columns, meta_oids)

    # Pull out REDCap system columns first so they don't appear as unmapped.
    system_cols: list[str] = []
    body_cols: list[str] = []
    for c in csv_columns:
        if c in REDCAP_SYSTEM_COLUMNS:
            system_cols.append(c)
        else:
            body_cols.append(c)

    pairs: list[dict[str, Any]] = []
    unmapped_expected: list[dict[str, Any]] = []
    unmapped_csv: list[str] = []

    if raw_mode:
        # Direct OID-name match; no base-label / dedup handling needed.
        oid_to_expected = {e["item_oid"]: e for e in expected}
        seen_oids: set[str] = set()
        for c in body_cols:
            e = oid_to_expected.get(c)
            if e is None:
                unmapped_csv.append(c)
                continue
            pairs.append({
                "csv_column": c,
                "item_oid": e["item_oid"],
                "form": e["form"],
                "field_type": e["field_type"],
                "identifier": e["identifier"],
            })
            seen_oids.add(e["item_oid"])
        for e in expected:
            if e["item_oid"] not in seen_oids:
                unmapped_expected.append(e)
    else:
        # Label-mode: group CSV columns by base label and zip.
        csv_by_base: dict[str, list[str]] = defaultdict(list)
        for c in body_cols:
            csv_by_base[base_label(c)].append(c)

        exp_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in expected:
            exp_by_base[e["expected_label"]].append(e)

        all_bases = set(csv_by_base) | set(exp_by_base)
        for base in sorted(all_bases):
            exp_list = exp_by_base.get(base, [])
            csv_list = csv_by_base.get(base, [])
            for exp, csv_name in zip(exp_list, csv_list):
                pairs.append({
                    "csv_column": csv_name,
                    "item_oid": exp["item_oid"],
                    "form": exp["form"],
                    "field_type": exp["field_type"],
                    "identifier": exp["identifier"],
                })
            if len(exp_list) > len(csv_list):
                unmapped_expected.extend(exp_list[len(csv_list):])
            elif len(csv_list) > len(exp_list):
                unmapped_csv.extend(csv_list[len(exp_list):])

    return {
        "export_mode": "raw" if raw_mode else "label",
        "n_expected": len(expected),
        "n_csv_columns": len(csv_columns),
        "n_system_columns": len(system_cols),
        "n_mapped": len(pairs),
        "n_unmapped_xml": len(unmapped_expected),
        "n_unmapped_csv": len(unmapped_csv),
        "system_columns": system_cols,
        "pairs": pairs,
        "unmapped_xml": unmapped_expected,
        "unmapped_csv": unmapped_csv,
    }


def duplicate_base_name_groups(csv_columns: list[str]) -> dict[str, list[str]]:
    """Group CSV columns whose base label (after stripping `.N`) collides.

    Only groups with >1 member are returned. Case is preserved (so
    "Start Time" and "Start time" appear as separate groups, exposing the
    kind of REDCap-label inconsistency the user asked about)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for c in csv_columns:
        groups[base_label(c)].append(c)
    return {k: v for k, v in groups.items() if len(v) > 1}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml-meta", required=True,
                   help="Path to redcap_meta.json (from redcap_meta.py)")
    p.add_argument("--columns", required=True,
                   help="Path to CSV column names (JSON array or Python-list "
                        "literal, one column per line also accepted)")
    p.add_argument("--output", required=True, help="Path to write mapping.json")
    args = p.parse_args(argv)

    meta = json.loads(Path(args.xml_meta).read_text(encoding="utf-8"))
    csv_cols = load_column_list(args.columns)
    expected = enumerate_expected_labels(meta)
    mapping = map_expected_to_csv(expected, csv_cols)
    mapping["duplicate_base_name_groups"] = duplicate_base_name_groups(csv_cols)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(
        f"Wrote {out} — mapped {mapping['n_mapped']} pairs, "
        f"{mapping['n_unmapped_xml']} XML items unmapped, "
        f"{mapping['n_unmapped_csv']} CSV columns unmapped, "
        f"{len(mapping['duplicate_base_name_groups'])} duplicate-name groups.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
