"""Human-reviewable JSON report of LLM findings.

Every entry captures enough for a reviewer to accept or reject the LLM's
recommendation without going back to the raw CSV. The report file is
written at close time (JSON, indented).

There are two kinds of entries:

* Per-field findings — a `LlmScanOperation` produced one or more spans on
  a specific field in a specific record.
* Record-level flags — the `RecordReviewer` post-pass flagged one or more
  spans somewhere in a concatenated record.

Both share the same shape; they live under `findings` and `record_flags`
respectively.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .response import ResolvedSpan


class ReviewReport:
    def __init__(
        self,
        path: str | Path,
        source: str | Path,
        schema_sha256: str | None = None,
    ):
        self.path = Path(path)
        self._payload: dict[str, Any] = {
            "source": str(source),
            "schema_sha256": schema_sha256,
            "n_findings": 0,
            "n_record_flags": 0,
            "findings": [],
            "record_flags": [],
        }

    def add_finding(
        self,
        record_location: dict[str, Any],
        field: str,
        operation: str,
        value_seen: str,
        recommended_redacted: str,
        spans: list[ResolvedSpan],
        model: str | None = None,
    ) -> None:
        if not spans:
            return
        self._payload["findings"].append({
            "record_location": dict(record_location),
            "field": field,
            "operation": operation,
            "model": model,
            "value_seen": value_seen,
            "recommended_redacted": recommended_redacted,
            "spans": [asdict(s) for s in spans],
        })
        self._payload["n_findings"] = len(self._payload["findings"])

    def add_record_flag(
        self,
        record_location: dict[str, Any],
        concatenated_value: str,
        recommended_redacted: str,
        spans: list[ResolvedSpan],
        model: str | None = None,
    ) -> None:
        if not spans:
            return
        self._payload["record_flags"].append({
            "record_location": dict(record_location),
            "model": model,
            "value_seen": concatenated_value,
            "recommended_redacted": recommended_redacted,
            "spans": [asdict(s) for s in spans],
        })
        self._payload["n_record_flags"] = len(self._payload["record_flags"])

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._payload, f, indent=2, ensure_ascii=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
