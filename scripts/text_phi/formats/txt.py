"""Plain-text format: one Record per line, synthetic single field `text`.

The trailing newline for each line is preserved on `Record.location["newline"]`
so `.save()` reproduces the source byte-for-byte when nothing is redacted.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..records import Record
from ..schema import Schema


TXT_TEXT_FIELD = "text"


class TxtFormat:
    extensions: ClassVar[tuple[str, ...]] = (".txt",)
    name: ClassVar[str] = "txt"

    def default_schema(self) -> Schema:
        return Schema.from_dict({
            "schema_version": "1",
            "format": "txt",
            "fields": {
                TXT_TEXT_FIELD: {
                    "dtype": "string",
                    "description": "One line of the source text.",
                    "operations": "default",
                },
            },
        })

    def load(
        self,
        path: str | Path,
        schema: Schema | None = None,
        allow_unknown: bool = False,
        allow_parse_errors: bool = False,
    ) -> list[Record]:
        if schema is None:
            schema = self.default_schema()
        # TXT has a fixed field structure; we don't apply unknown/missing/
        # parse-error policies since there's only one synthetic field.
        # newline="" disables Python's universal-newline translation so
        # CRLF/LF/CR are preserved verbatim into `location.newline`.
        with open(path, "r", encoding="utf-8", newline="") as f:
            text = f.read()
        records: list[Record] = []
        for i, line in enumerate(text.splitlines(keepends=True)):
            stripped = line.rstrip("\r\n")
            newline = line[len(stripped):]
            records.append(Record(
                location={"line": i, "newline": newline},
                fields={TXT_TEXT_FIELD: stripped},
            ))
        return records

    def save(
        self,
        path: str | Path,
        records: list[Record],
        schema: Schema,
    ) -> None:
        parts = [
            r.fields.get(TXT_TEXT_FIELD, "") + r.location.get("newline", "\n")
            for r in records
        ]
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write("".join(parts))

    def validate_output(
        self,
        input_path: str | Path,
        output_path: str | Path,
        schema: Schema,
        n_records: int,
    ) -> None:
        out_path = Path(output_path)
        if not out_path.exists():
            raise IOError(f"{self.name}: output {out_path} was not written")
        try:
            with open(out_path, "r", encoding="utf-8", newline="") as f:
                reread = f.read()
        except UnicodeDecodeError as e:
            raise IOError(
                f"{self.name}: output {out_path} is not valid UTF-8: {e}"
            ) from e
        n_out_lines = len(reread.splitlines(keepends=True))
        if n_out_lines != n_records:
            raise IOError(
                f"{self.name}: line count mismatch after write "
                f"(records={n_records}, output={n_out_lines})"
            )
