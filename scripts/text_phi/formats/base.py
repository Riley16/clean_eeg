"""Pluggable file-format protocol (records edition).

Each supported file type implements `.load()` → records and `.save()` ← records
against the given Schema. Format-specific validation (shape, column list) lives
in `.validate_output`. Preserved-field byte-identity is enforced by the shared
`RecordRedactor` in records.py — not per format.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from ..records import Record
from ..schema import Schema


@runtime_checkable
class FileFormat(Protocol):
    extensions: ClassVar[tuple[str, ...]]
    name: ClassVar[str]

    def load(
        self,
        path: str | Path,
        schema: Schema | None = None,
        allow_unknown: bool = False,
        allow_parse_errors: bool = False,
    ) -> list[Record]:
        ...

    def save(
        self,
        path: str | Path,
        records: list[Record],
        schema: Schema,
    ) -> None:
        ...

    def validate_output(
        self,
        input_path: str | Path,
        output_path: str | Path,
        schema: Schema,
        n_records: int,
    ) -> None:
        ...

    def default_schema(self) -> Schema | None:
        """Optional. Formats without a natural default (e.g. CSV) return None."""


def infer_columns(path: str | Path) -> list[str]:
    """Read the header of a delimited file to inform schema derivation."""
    import pandas as pd
    return list(pd.read_csv(path, nrows=0).columns)
