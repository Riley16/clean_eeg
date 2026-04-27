"""Tests for the EDF header debug utility.

The standalone CLI behavior (file vs directory traversal, signal
selection, etc.) is exercised end-to-end via the pipeline tests. This
file focuses on the ``redact_phi`` flag — the contract that the four
PHI-bearing main-header fields are masked when the dump is destined for
a shared log.
"""

import io

from clean_eeg.print_edf_header import (
    print_header,
    PHI_MAIN_HEADER_FIELDS,
    MAIN_HEADER_FIELDS,
)
from tests.generate_edf import generate_partial_record_edf


def _make_edf(tmp_path):
    full, _ = generate_partial_record_edf(tmp_path / "src.edf",
                                            n_channels=3,
                                            sample_rate=100,
                                            duration_sec=5)
    return full


def test_phi_main_header_fields_match_spec():
    """Lock in the set of fields treated as PHI: patient_id,
    recording_id, startdate, starttime. If a future change adds another
    PHI-bearing field, this test should fail until the policy is updated
    deliberately."""
    assert PHI_MAIN_HEADER_FIELDS == frozenset(
        {"patient_id", "recording_id", "startdate", "starttime"}
    )


def test_full_dump_shows_phi_fields(tmp_path):
    """Default mode (redact_phi=False) prints raw bytes for every field
    — that's what the standalone debugging command should do."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, out=buf)
    output = buf.getvalue()
    # Toy file's patient_id / recording_id / dates contain readable
    # text — confirm at least one piece is present unredacted.
    assert "patient_id" in output
    assert "recording_id" in output
    assert "startdate" in output
    assert "starttime" in output
    assert "[PHI_REDACTED]" not in output


def _field_row(output: str, field_name: str) -> str:
    """Return the dump line for ``field_name``. Each main-header field
    row has format ``  bytes  X-Y    (NB)  field_name           raw=...``.
    Match on the unique fixed-width field-name slot to avoid false hits
    on the redaction-announcement banner."""
    needle = f"  {field_name:<19}  "
    for line in output.splitlines():
        if needle in line:
            return line
    raise AssertionError(f"no row found for field {field_name!r} in: {output!r}")


def test_redact_phi_masks_only_phi_fields(tmp_path):
    """``redact_phi=True`` must mask each of the four PHI fields and
    leave every other field untouched — the data team still needs to
    see n_signals, n_records, record_duration, bytes_in_header to
    triage parse failures."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, redact_phi=True, out=buf)
    output = buf.getvalue()

    # Each PHI field row must show [PHI_REDACTED].
    for phi_field in PHI_MAIN_HEADER_FIELDS:
        row = _field_row(output, phi_field)
        assert "[PHI_REDACTED]" in row, \
            f"PHI field {phi_field!r} not masked: {row!r}"

    # Non-PHI main-header fields must NOT be masked.
    non_phi_fields = {n for _, _, n, _ in MAIN_HEADER_FIELDS} - PHI_MAIN_HEADER_FIELDS
    for field in non_phi_fields:
        row = _field_row(output, field)
        assert "[PHI_REDACTED]" not in row, \
            f"Non-PHI field {field!r} should NOT be masked: {row!r}"


def test_redact_phi_keeps_signal_headers_visible(tmp_path):
    """Per-signal headers carry no spec-defined PHI (transducer and
    prefilter are non-numeric descriptive strings; label is a channel
    name like 'C3' / 'Fz'). They must remain visible even with
    ``redact_phi=True``."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, redact_phi=True, out=buf)
    output = buf.getvalue()
    # Signal-table marker is present.
    assert "Signal headers" in output
    # Channel labels from the toy file (CH0/CH1/CH2) appear unredacted.
    for label in ("CH0", "CH1", "CH2"):
        assert label in output


def test_partial_header_dump_shows_available_fields(tmp_path):
    """For a file shorter than 256 bytes, print_header should dump
    every field whose byte range fits within the available bytes and
    mark the rest as missing — instead of bailing with just a warning."""
    edf = tmp_path / "partial.edf"
    # 100 bytes: covers version (0-7), patient_id (8-87), and 12 bytes
    # of recording_id (88-99). Everything past byte 99 is missing.
    edf.write_bytes(b"0       " + b"PATIENT-ID-CONTENT" + b" " * (88 - 8 - 18) + b"RECORDING-ID")
    buf = io.StringIO()
    print_header(str(edf), out=buf)
    output = buf.getvalue()
    # Truncation warning
    assert "shorter than 256" in output
    assert "100 bytes available" in output
    # Fields that fit are shown.
    assert "version" in output
    assert "patient_id" in output  # bytes 8-87 — full
    # Fields whose byte range crosses the cut are marked missing.
    assert "<missing — file ends at byte 100>" in output
    # n_signals lives at bytes 252-255, well past EOF — must be missing.
    n_signals_line = next(l for l in output.splitlines() if " n_signals" in l)
    assert "missing" in n_signals_line


def test_partial_header_with_redact_phi_still_masks_phi_fields(tmp_path):
    """Truncated file with redact_phi=True: PHI fields must still be
    masked even though only some of their bytes are present on disk."""
    edf = tmp_path / "partial.edf"
    edf.write_bytes(b"0       " + b"PATIENT_NAME_HERE" + b" " * 75)  # 100 bytes
    buf = io.StringIO()
    print_header(str(edf), redact_phi=True, out=buf)
    output = buf.getvalue()
    # The actual bytes of patient_id were on disk, but redact_phi must
    # still mask the row.
    patient_row = _field_row(output, "patient_id")
    assert "[PHI_REDACTED]" in patient_row
    assert "PATIENT_NAME_HERE" not in output
    # And the truncation warning was still printed.
    assert "shorter than 256" in output


def test_partial_header_skips_signal_table(tmp_path):
    """If main header is incomplete, signal headers cannot be parsed
    (they live past the main header). Skip the per-signal section."""
    edf = tmp_path / "partial.edf"
    edf.write_bytes(b"\x00" * 50)
    buf = io.StringIO()
    print_header(str(edf), out=buf)
    output = buf.getvalue()
    assert "Signal headers" not in output


def test_redact_phi_announces_redaction_in_header(tmp_path):
    """The dump should include a one-line note explaining that PHI
    fields have been masked, so a reader who doesn't already know the
    redaction policy understands what they're looking at."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, redact_phi=True, out=buf)
    output = buf.getvalue()
    assert "PHI fields" in output and "masked" in output
