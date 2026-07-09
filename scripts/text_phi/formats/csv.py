"""CSV format: records ↔ pandas DataFrame via `to_dict(orient="records")` /
`pd.DataFrame(records).to_csv()`.

Load-time behaviors:
  * Unknown columns (present in CSV, absent from schema): controlled by
    `schema.unknown_field_policy` and the `allow_unknown` override.
  * Missing columns (present in schema, absent from CSV): controlled by
    `schema.missing_field_policy`.
  * Eager dtype coercion via each field's `dtype.validate`; failures raise
    unless `allow_parse_errors=True`, in which case the row is dropped with
    a warning.

Post-write validation: shape (row count, column list) + preserved-field
byte-identity (that check lives in `RecordRedactor`, not here).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ..dtypes import DtypeError, get_dtype
from ..records import Record
from ..schema import Schema


_log = logging.getLogger(__name__)


class CsvFormat:
    extensions: ClassVar[tuple[str, ...]] = (".csv",)
    name: ClassVar[str] = "csv"

    def default_schema(self) -> Schema | None:
        # No universal CSV schema — each dataset has its own columns. The
        # CLI's `schema derive` subcommand bootstraps one from the file header.
        return None

    def load(
        self,
        path: str | Path,
        schema: Schema | None = None,
        allow_unknown: bool = False,
        allow_parse_errors: bool = False,
    ) -> list[Record]:
        if schema is None:
            raise ValueError(
                f"{self.name}: a schema is required. Use "
                f"`text_phi schema derive` to bootstrap one."
            )
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df_cols = list(df.columns)

        self._check_column_policies(
            df_cols, schema, allow_unknown=allow_unknown,
        )

        records: list[Record] = []
        for i in range(len(df)):
            fields = {c: str(df.at[i, c]) for c in df_cols}
            if not self._validate_row_dtypes(
                i, fields, schema, allow_parse_errors=allow_parse_errors,
            ):
                # Row failed parse and errors are being demoted — skip it.
                continue
            records.append(Record(location={"row": i}, fields=fields))
        return records

    def save(
        self,
        path: str | Path,
        records: list[Record],
        schema: Schema,
    ) -> None:
        # Preserve the column order from the first record; if empty, fall
        # back to the schema's declaration order (dict insertion order).
        if records:
            columns = list(records[0].fields.keys())
        else:
            columns = list(schema.fields.keys())
        df = pd.DataFrame([r.fields for r in records], columns=columns)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

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
            reread = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        except Exception as e:
            raise IOError(
                f"{self.name}: output {out_path} not a readable CSV: {e}"
            ) from e
        if len(reread) != n_records:
            raise IOError(
                f"{self.name}: row count changed after write "
                f"(records={n_records}, output={len(reread)})"
            )

    # ---------- helpers ----------

    def _check_column_policies(
        self,
        df_cols: list[str],
        schema: Schema,
        allow_unknown: bool,
    ) -> None:
        df_set = set(df_cols)
        schema_set = set(schema.fields)

        unknown = df_set - schema_set
        if unknown:
            if schema.unknown_field_policy == "error" and not allow_unknown:
                raise ValueError(
                    f"{self.name}: CSV has columns not in schema: "
                    f"{sorted(unknown)}. Use --allow-unknown to pass them "
                    f"through, or add them to the schema."
                )
            _log.warning(
                "%s: unknown columns present, passing through: %s",
                self.name, sorted(unknown),
            )

        missing = schema_set - df_set
        if missing:
            if schema.missing_field_policy == "error":
                raise ValueError(
                    f"{self.name}: schema fields missing from CSV: "
                    f"{sorted(missing)}"
                )
            _log.warning(
                "%s: schema fields absent from CSV: %s",
                self.name, sorted(missing),
            )

    def _validate_row_dtypes(
        self,
        row_index: int,
        fields: dict[str, str],
        schema: Schema,
        allow_parse_errors: bool,
    ) -> bool:
        """Return True if row is OK, False if it should be skipped."""
        for col, val in fields.items():
            if col not in schema.fields:
                continue
            spec = schema.fields[col]
            dt = get_dtype(spec.dtype)
            try:
                dt.validate(val)
            except DtypeError as e:
                if allow_parse_errors:
                    _log.warning(
                        "%s: row %d col %r: %s — skipping row",
                        self.name, row_index, col, e,
                    )
                    return False
                raise ValueError(
                    f"{self.name}: row {row_index} col {col!r}: {e}"
                ) from e
        return True
