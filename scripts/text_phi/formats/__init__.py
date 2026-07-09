"""Registry of file-format handlers.

Adding a new format: one file under `formats/`, one entry in `_ALL_FORMATS`.
"""

from __future__ import annotations

from .base import FileFormat, infer_columns
from .csv import CsvFormat
from .txt import TxtFormat


_ALL_FORMATS: list[type[FileFormat]] = [TxtFormat, CsvFormat]

REGISTRY: dict[str, FileFormat] = {}
for _cls in _ALL_FORMATS:
    _inst = _cls()
    for _ext in _cls.extensions:
        REGISTRY[_ext.lower()] = _inst


def get_format(extension: str) -> FileFormat | None:
    return REGISTRY.get(extension.lower())


def supported_extensions() -> list[str]:
    return sorted(REGISTRY.keys())


__all__ = [
    "FileFormat",
    "TxtFormat",
    "CsvFormat",
    "REGISTRY",
    "get_format",
    "supported_extensions",
    "infer_columns",
]
