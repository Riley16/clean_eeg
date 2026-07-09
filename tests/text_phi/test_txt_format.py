"""Tests for scripts/text_phi/formats/txt.py (records API)."""

from __future__ import annotations

import pytest

from scripts.text_phi.formats import TxtFormat


def _fmt():
    return TxtFormat()


def test_default_schema_single_text_field():
    schema = _fmt().default_schema()
    assert list(schema.fields.keys()) == ["text"]
    assert schema.fields["text"].dtype == "string"


def test_load_returns_one_record_per_line(tmp_path):
    p = tmp_path / "in.txt"
    p.write_text("first line\nsecond line\nthird\n")
    recs = _fmt().load(p)
    assert len(recs) == 3
    assert [r.fields["text"] for r in recs] == ["first line", "second line", "third"]
    assert [r.location["line"] for r in recs] == [0, 1, 2]
    # Every line ended in "\n" → each record's location.newline preserves it.
    assert all(r.location["newline"] == "\n" for r in recs)


def test_load_preserves_no_trailing_newline_on_last_line(tmp_path):
    p = tmp_path / "in.txt"
    p.write_text("first\nsecond")  # no trailing newline
    recs = _fmt().load(p)
    assert recs[-1].location["newline"] == ""


def test_save_roundtrip_is_byte_identical(tmp_path):
    original = "alpha\nbeta\r\ngamma\n"
    in_path = tmp_path / "in.txt"
    out_path = tmp_path / "out.txt"
    in_path.write_text(original)

    fmt = _fmt()
    schema = fmt.default_schema()
    recs = fmt.load(in_path, schema)
    fmt.save(out_path, recs, schema)

    # Compare raw bytes — read_text() would apply universal-newline
    # translation and turn CRLF into LF, hiding a real round-trip loss.
    assert out_path.read_bytes() == original.encode("utf-8")


def test_validate_output_line_count_mismatch(tmp_path):
    fmt = _fmt()
    schema = fmt.default_schema()
    in_path = tmp_path / "in.txt"
    out_path = tmp_path / "out.txt"
    in_path.write_text("a\nb\n")
    out_path.write_text("a\n")
    with pytest.raises(IOError, match="line count mismatch"):
        fmt.validate_output(in_path, out_path, schema, n_records=2)


def test_validate_output_missing_file(tmp_path):
    fmt = _fmt()
    schema = fmt.default_schema()
    with pytest.raises(IOError, match="was not written"):
        fmt.validate_output(tmp_path / "in.txt", tmp_path / "nope.txt",
                            schema, n_records=0)


def test_validate_output_bad_utf8(tmp_path):
    fmt = _fmt()
    schema = fmt.default_schema()
    out_path = tmp_path / "out.txt"
    out_path.write_bytes(b"\xff\xfe\xc3\x28")
    with pytest.raises(IOError, match="not valid UTF-8"):
        fmt.validate_output(tmp_path / "in.txt", out_path, schema, n_records=1)


def test_load_empty_file(tmp_path):
    in_path = tmp_path / "empty.txt"
    in_path.write_text("")
    recs = _fmt().load(in_path)
    assert recs == []
