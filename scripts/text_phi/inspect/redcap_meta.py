"""Parse a REDCap CDISC-ODM XML metadata export into a canonical JSON.

Extracts per-field: OID / variable name, question label, DataType, REDCap
field type, REDCap identifier flag, choices (for radio/dropdown/checkbox),
form membership, validation type, length.

Handles the checkbox-expansion issue: a single REDCap checkbox field like
`race` becomes N columns in a label-exported CSV, one per choice, labeled
like "Race (choice=Black)". This module resolves each `race___N` item's
choice label so downstream code can generate the expected column labels.

Usage as a CLI:
    python -m scripts.text_phi.inspect.redcap_meta \\
        --input Penn_REDCap_meta_data.xml --output redcap_meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


_ODM = "{http://www.cdisc.org/ns/odm/v1.3}"
_REDCAP = "{https://projectredcap.org}"


def _translated_text(elem: ET.Element, tag: str) -> str | None:
    child = elem.find(tag)
    if child is None:
        return None
    tt = child.find(f"{_ODM}TranslatedText")
    if tt is None or tt.text is None:
        return None
    return tt.text


def _parse_checkbox_choices(attr_value: str | None) -> dict[str, str]:
    """Parse the CheckboxChoices attribute value.

    Format: `"8, Black | 9, Asian | 10, Caribbean"`. Returns `{"8": "Black",
    "9": "Asian", ...}`.
    """
    if not attr_value:
        return {}
    out: dict[str, str] = {}
    for chunk in attr_value.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        code, _, label = chunk.partition(",")
        code = code.strip()
        label = label.strip()
        # Empty label is legitimate (`"1, "` appears in the DCC bad_session
        # fields) — REDCap emits `(choice=)` in the CSV. Preserve the entry.
        if code:
            out[code] = label
    return out


def _parse_code_list(cl: ET.Element) -> dict[str, Any]:
    items = []
    for it in cl.findall(f"{_ODM}CodeListItem"):
        code = it.get("CodedValue")
        label = _translated_text(it, f"{_ODM}Decode")
        items.append({"code": code, "label": label})
    checkbox_attr = cl.get(f"{_REDCAP}CheckboxChoices")
    return {
        "oid": cl.get("OID"),
        "items": items,
        "checkbox_choices": _parse_checkbox_choices(checkbox_attr) if checkbox_attr else None,
    }


_CHECKBOX_SUFFIX = re.compile(r"___(?P<code>[A-Za-z0-9_]+)$")


def _checkbox_code_from_oid(oid: str) -> str | None:
    """`race___8` → `"8"`; anything else → None."""
    m = _CHECKBOX_SUFFIX.search(oid)
    return m.group("code") if m else None


def _parse_item(item: ET.Element, code_lists: dict[str, dict]) -> dict[str, Any]:
    oid = item.get("OID")
    ref = item.find(f"{_ODM}CodeListRef")
    code_list_oid = ref.get("CodeListOID") if ref is not None else None
    code_list = code_lists.get(code_list_oid) if code_list_oid else None

    field_type = item.get(f"{_REDCAP}FieldType")
    checkbox_code = _checkbox_code_from_oid(oid) if field_type == "checkbox" else None
    checkbox_label = None
    if checkbox_code and code_list and code_list.get("checkbox_choices"):
        checkbox_label = code_list["checkbox_choices"].get(checkbox_code)

    return {
        "oid": oid,
        "name": item.get("Name"),
        "variable": item.get(f"{_REDCAP}Variable"),
        "data_type": item.get("DataType"),
        "field_type": field_type,
        "validation_type": item.get(f"{_REDCAP}TextValidationType"),
        "identifier": item.get(f"{_REDCAP}Identifier") == "y",
        "required": item.get(f"{_REDCAP}RequiredField") == "y",
        "length": item.get("Length"),
        "section_header": item.get(f"{_REDCAP}SectionHeader"),
        "calculation": item.get(f"{_REDCAP}Calculation"),
        "question": _translated_text(item, f"{_ODM}Question"),
        "code_list_oid": code_list_oid,
        "code_list": code_list,
        "checkbox_code": checkbox_code,
        "checkbox_label": checkbox_label,
        "form": None,  # filled in second pass
    }


def parse_metadata(xml_path: str | Path) -> dict[str, Any]:
    """Parse the whole REDCap XML into a canonical dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mv = root.find(f"{_ODM}Study/{_ODM}MetaDataVersion")
    if mv is None:
        raise ValueError("XML has no Study/MetaDataVersion element")

    # Code lists first (items reference them).
    code_lists: dict[str, dict] = {}
    for cl in mv.findall(f"{_ODM}CodeList"):
        parsed = _parse_code_list(cl)
        code_lists[parsed["oid"]] = parsed

    # Items.
    items: dict[str, dict] = {}
    for item in mv.findall(f"{_ODM}ItemDef"):
        parsed = _parse_item(item, code_lists)
        items[parsed["oid"]] = parsed

    # ItemGroup → ItemRef list.
    ig_items: dict[str, list[str]] = {}
    for ig in mv.findall(f"{_ODM}ItemGroupDef"):
        ig_oid = ig.get("OID")
        ig_items[ig_oid] = [
            ir.get("ItemOID") for ir in ig.findall(f"{_ODM}ItemRef")
        ]

    # Form → items via ItemGroupRef.
    forms: dict[str, dict] = {}
    for form in mv.findall(f"{_ODM}FormDef"):
        form_oid = form.get("OID")
        form_name = form.get(f"{_REDCAP}FormName") or form.get("Name")
        item_list: list[str] = []
        for igr in form.findall(f"{_ODM}ItemGroupRef"):
            igr_oid = igr.get("ItemGroupOID")
            item_list.extend(ig_items.get(igr_oid, []))
        forms[form_oid] = {
            "oid": form_oid,
            "name": form_name,
            "display_name": form.get("Name"),
            "repeating": form.get("Repeating") == "Yes",
            "items": item_list,
        }
        for item_oid in item_list:
            if item_oid in items and not items[item_oid]["form"]:
                items[item_oid]["form"] = form_name

    return {
        "items": items,
        "code_lists": code_lists,
        "forms": forms,
        "counts": {
            "items": len(items),
            "code_lists": len(code_lists),
            "forms": len(forms),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to REDCap XML")
    p.add_argument("--output", required=True, help="Path to write JSON")
    args = p.parse_args(argv)
    meta = parse_metadata(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} — {meta['counts']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
