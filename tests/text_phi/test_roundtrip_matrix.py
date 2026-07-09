"""Lossless-roundtrip matrix.

For every (format × dtype × value_class), when the schema declares only
`passthrough` operations, the pipeline (load → process → save) must produce
byte-identical output. Any drift here is a data-integrity bug.

Also verifies that pure I/O without the redactor loop is lossless (a lower
bar; catches format-level bugs independent of the redactor).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.text_phi.formats import CsvFormat, TxtFormat
from scripts.text_phi.records import RecordRedactor
from scripts.text_phi.schema import Schema


# ---------- CSV value classes per dtype ----------

CSV_VALUES: dict[str, list[str]] = {
    "string": [
        "",                                # empty
        "hello",                           # ascii
        "  leading and trailing  ",        # whitespace
        "unicode: héllo Ω π",              # BMP unicode
        "high-plane: 🧠 🦕",               # emoji / astral
        "embedded, comma",                 # CSV special
        'embedded "quote"',                # CSV quote escaping
        "embedded\nnewline",               # CSV newline handling
        "tab\there",                       # whitespace preserved
    ],
    "integer": [
        "", "0", "-1", "42",
        "9223372036854775807",             # int64 max
        "-9223372036854775808",            # int64 min
        "12345678901234567890",            # beyond int64 (arbitrary precision)
    ],
    "float": [
        "", "0.0", "-0.0", "3.14", "1e-10",
        "3.141592653589793",               # ~17 sig figs
        "inf", "-inf", "nan",
    ],
    "boolean": ["", "true", "false", "1", "0", "yes", "no"],
    "date": [
        "", "2024-01-15", "1985-01-01", "2000-02-29",  # leap year
        "9999-12-31", "0001-01-01",
    ],
    "datetime": [
        "", "2024-01-15T10:30:00",
        "2024-01-15T10:30:00+00:00",
        "1985-01-01T00:00:00",
    ],
    "enum": ["", "level_1", "control", "case"],
    "bytes": [
        "", "aGVsbG8=",                    # b"hello"
        "AAECAwQF",                        # b"\x00\x01\x02\x03\x04\x05"
    ],
}


# ---------- TXT value classes ----------

TXT_LINES = [
    "",                                    # empty line
    "simple ascii line",
    "  whitespace  padded  ",
    "unicode: héllo Ω π",
    "emoji: 🧠 🦕",
    "tab\there",                           # embedded tab
]


# ---------- CSV roundtrip ----------

def _csv_schema(dtype: str) -> Schema:
    return Schema.from_dict({
        "schema_version": "1", "format": "csv",
        "fields": {
            "v": {
                "dtype": dtype,
                "description": "",
                "operations": ["passthrough"],
            },
        },
    })


def _csv_write_source(tmp_path, dtype: str, values: list[str]):
    """Emit a canonical CSV with one column 'v'. We write via pandas so the
    baseline matches how our loader reads."""
    in_path = tmp_path / "in.csv"
    pd.DataFrame({"v": values}).to_csv(in_path, index=False)
    return in_path


@pytest.mark.parametrize(
    "dtype,value",
    [(dt, v) for dt, vals in CSV_VALUES.items() for v in vals],
)
def test_csv_lossless_via_format_only(tmp_path, dtype, value):
    """Pure I/O — load then save, no processing. Bytes must match."""
    fmt = CsvFormat()
    schema = _csv_schema(dtype)
    in_path = _csv_write_source(tmp_path, dtype, [value])
    out_path = tmp_path / "out.csv"

    records = fmt.load(in_path, schema)
    fmt.save(out_path, records, schema)

    assert in_path.read_bytes() == out_path.read_bytes(), (
        f"format-only roundtrip lost data: dtype={dtype!r} value={value!r}"
    )


@pytest.mark.parametrize(
    "dtype,value",
    [(dt, v) for dt, vals in CSV_VALUES.items() for v in vals],
)
def test_csv_lossless_via_redactor_passthrough(tmp_path, dtype, value):
    """load → RecordRedactor(passthrough) → save must be byte-identical."""
    fmt = CsvFormat()
    schema = _csv_schema(dtype)
    in_path = _csv_write_source(tmp_path, dtype, [value])
    out_path = tmp_path / "out.csv"

    records = fmt.load(in_path, schema)
    processed, events = RecordRedactor(schema).process_records(records)
    assert events == []  # passthrough-only → no events
    fmt.save(out_path, processed, schema)

    assert in_path.read_bytes() == out_path.read_bytes(), (
        f"passthrough roundtrip lost data: dtype={dtype!r} value={value!r}"
    )


def test_csv_multi_column_multi_row_lossless(tmp_path):
    """Roundtrip a small multi-column, multi-row CSV under all passthrough."""
    fmt = CsvFormat()
    schema = Schema.from_dict({
        "schema_version": "1", "format": "csv",
        "fields": {
            "a": {"dtype": "string", "operations": ["passthrough"]},
            "n": {"dtype": "integer", "operations": ["passthrough"]},
            "b": {"dtype": "boolean", "operations": ["passthrough"]},
            "d": {"dtype": "date", "operations": ["passthrough"]},
        },
    })
    in_path = tmp_path / "in.csv"
    pd.DataFrame([
        {"a": "hello", "n": "42", "b": "true", "d": "2024-01-15"},
        {"a": "", "n": "", "b": "", "d": ""},
        {"a": "unicode π", "n": "-1", "b": "false", "d": "2000-02-29"},
    ]).to_csv(in_path, index=False)
    out_path = tmp_path / "out.csv"

    records = fmt.load(in_path, schema)
    processed, _ = RecordRedactor(schema).process_records(records)
    fmt.save(out_path, processed, schema)

    assert in_path.read_bytes() == out_path.read_bytes()


# ---------- TXT roundtrip ----------

def _txt_source(tmp_path, lines: list[str], line_endings: str) -> str:
    """Emit a raw TXT file with the given line endings ('\\n', '\\r\\n', or
    mixed) so we can test each preservation case."""
    content = line_endings.join(lines) + line_endings
    p = tmp_path / "in.txt"
    p.write_bytes(content.encode("utf-8"))
    return p


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_txt_lossless_via_format_only(tmp_path, line_ending):
    fmt = TxtFormat()
    schema = fmt.default_schema()
    in_path = _txt_source(tmp_path, TXT_LINES, line_ending)
    out_path = tmp_path / "out.txt"

    records = fmt.load(in_path, schema)
    fmt.save(out_path, records, schema)

    assert in_path.read_bytes() == out_path.read_bytes()


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_txt_lossless_via_redactor_passthrough(tmp_path, line_ending):
    fmt = TxtFormat()
    schema = Schema.from_dict({
        "schema_version": "1", "format": "txt",
        "fields": {
            "text": {"dtype": "string", "operations": ["passthrough"]},
        },
    })
    in_path = _txt_source(tmp_path, TXT_LINES, line_ending)
    out_path = tmp_path / "out.txt"

    records = fmt.load(in_path, schema)
    processed, events = RecordRedactor(schema).process_records(records)
    assert events == []
    fmt.save(out_path, processed, schema)

    assert in_path.read_bytes() == out_path.read_bytes()


def test_txt_no_trailing_newline_preserved(tmp_path):
    fmt = TxtFormat()
    schema = fmt.default_schema()
    in_path = tmp_path / "in.txt"
    in_path.write_bytes(b"first line\nsecond line")  # no trailing NL
    out_path = tmp_path / "out.txt"

    records = fmt.load(in_path, schema)
    fmt.save(out_path, records, schema)

    assert in_path.read_bytes() == out_path.read_bytes()


def test_txt_mixed_line_endings_preserved(tmp_path):
    fmt = TxtFormat()
    schema = fmt.default_schema()
    in_path = tmp_path / "in.txt"
    in_path.write_bytes(b"lf\nrn\r\nmixed\nend\r\n")
    out_path = tmp_path / "out.txt"

    records = fmt.load(in_path, schema)
    fmt.save(out_path, records, schema)

    assert in_path.read_bytes() == out_path.read_bytes()


def test_txt_empty_file(tmp_path):
    fmt = TxtFormat()
    schema = fmt.default_schema()
    in_path = tmp_path / "in.txt"
    in_path.write_bytes(b"")
    out_path = tmp_path / "out.txt"

    records = fmt.load(in_path, schema)
    fmt.save(out_path, records, schema)

    assert out_path.read_bytes() == b""
