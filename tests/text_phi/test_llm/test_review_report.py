"""Tests for scripts/text_phi/llm/review_report.py."""

from __future__ import annotations

import json

from scripts.text_phi.llm.response import ResolvedSpan
from scripts.text_phi.llm.review_report import ReviewReport


def _span(t="PERSON", matched="John", start=12, end=16) -> ResolvedSpan:
    return ResolvedSpan(
        start=start, end=end, type=t, matched_text=matched,
        reason="test", ambiguous=False,
    )


def test_writes_expected_top_level(tmp_path):
    out = tmp_path / "r.json"
    with ReviewReport(out, source="in.csv", schema_sha256="deadbeef") as r:
        r.add_finding(
            record_location={"row": 3},
            field="note",
            operation="llm_scan",
            value_seen="Dr. John was here",
            recommended_redacted="Dr. [PERSON] was here",
            spans=[_span()],
            model="qwen2.5:7b-instruct",
        )
    data = json.loads(out.read_text())
    assert data["source"] == "in.csv"
    assert data["schema_sha256"] == "deadbeef"
    assert data["n_findings"] == 1
    assert data["n_record_flags"] == 0


def test_finding_carries_full_context(tmp_path):
    out = tmp_path / "r.json"
    with ReviewReport(out, source="in.csv") as r:
        r.add_finding(
            record_location={"row": 3},
            field="note",
            operation="llm_scan",
            value_seen="Dr. John was here on 3/15.",
            recommended_redacted="Dr. [PERSON] was here on [DATE].",
            spans=[
                _span("PERSON", "John", 4, 8),
                _span("DATE", "3/15", 21, 25),
            ],
        )
    entry = json.loads(out.read_text())["findings"][0]
    assert entry["record_location"] == {"row": 3}
    assert entry["field"] == "note"
    assert entry["operation"] == "llm_scan"
    assert entry["value_seen"] == "Dr. John was here on 3/15."
    assert entry["recommended_redacted"] == "Dr. [PERSON] was here on [DATE]."
    assert len(entry["spans"]) == 2


def test_empty_spans_does_not_add_entry(tmp_path):
    out = tmp_path / "r.json"
    with ReviewReport(out, source="in.csv") as r:
        r.add_finding(
            record_location={"row": 3},
            field="note",
            operation="llm_scan",
            value_seen="x",
            recommended_redacted="x",
            spans=[],
        )
    data = json.loads(out.read_text())
    assert data["n_findings"] == 0
    assert data["findings"] == []


def test_record_flag_writes_to_record_flags_bucket(tmp_path):
    out = tmp_path / "r.json"
    with ReviewReport(out, source="in.csv") as r:
        r.add_record_flag(
            record_location={"row": 3},
            concatenated_value="---BEGIN note---\nDr. John\n---END note---",
            recommended_redacted="---BEGIN note---\nDr. [PERSON]\n---END note---",
            spans=[_span()],
        )
    data = json.loads(out.read_text())
    assert data["n_record_flags"] == 1
    assert data["n_findings"] == 0
    assert data["record_flags"][0]["record_location"] == {"row": 3}


def test_findings_and_flags_both(tmp_path):
    out = tmp_path / "r.json"
    with ReviewReport(out, source="in.csv") as r:
        r.add_finding(
            record_location={"row": 0}, field="note", operation="llm_scan",
            value_seen="a", recommended_redacted="[PERSON]",
            spans=[_span()],
        )
        r.add_record_flag(
            record_location={"row": 1},
            concatenated_value="b", recommended_redacted="[PERSON]",
            spans=[_span()],
        )
    data = json.loads(out.read_text())
    assert data["n_findings"] == 1
    assert data["n_record_flags"] == 1


def test_multiple_findings_accumulate(tmp_path):
    out = tmp_path / "r.json"
    with ReviewReport(out, source="in.csv") as r:
        for i in range(3):
            r.add_finding(
                record_location={"row": i}, field="note", operation="llm_scan",
                value_seen="v", recommended_redacted="[PERSON]",
                spans=[_span()],
            )
    data = json.loads(out.read_text())
    assert data["n_findings"] == 3
    assert [e["record_location"]["row"] for e in data["findings"]] == [0, 1, 2]
