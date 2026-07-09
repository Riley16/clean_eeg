"""Audit sidecar JSON writer.

Consumes `FieldRedactionEvent`s produced by `RecordRedactor.process_records`.
Each event becomes one entry that carries the record location, the field
name, the operations that ran, the resulting spans, and a SHA-256 of the
original value (or the raw value, if `include_original=True`).

The audit's top-level object also embeds `schema_sha256` so a reviewer can
reproduce the exact rule set that produced this redaction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .records import FieldRedactionEvent
from .schema import Schema


class AuditWriter:
    def __init__(
        self,
        path: str | Path,
        source_path: str | Path,
        mode: str,
        schema: Schema | None,
        replacement_style: str,
        include_original: bool = False,
        subject_names: list[str] | None = None,
    ):
        self.path = Path(path)
        self.include_original = include_original
        self._payload: dict[str, Any] = {
            "source": str(source_path),
            "mode": mode,
            "schema_version": schema.schema_version if schema else None,
            "schema_sha256": schema.sha256() if schema else None,
            "replacement_style": replacement_style,
            "include_original": include_original,
            "subject_names": subject_names or [],
            "n_entries": 0,
            "entries": [],
        }

    def add_event(self, event: FieldRedactionEvent) -> None:
        entry: dict[str, Any] = {
            "location": dict(event.location),
            "field": event.field_name,
            "operations_applied": list(event.operations_applied),
            "original_sha256": hashlib.sha256(
                event.original.encode("utf-8")
            ).hexdigest(),
            "redacted_value": event.redacted,
            "spans": [asdict(s) for s in event.spans],
        }
        if self.include_original:
            entry["original_value"] = event.original
        self._payload["entries"].append(entry)
        self._payload["n_entries"] = len(self._payload["entries"])

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._payload, f, indent=2, ensure_ascii=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
