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


def test_redact_phi_announces_redaction_in_header(tmp_path):
    """The dump should include a one-line note explaining that PHI
    fields have been masked, so a reader who doesn't already know the
    redaction policy understands what they're looking at."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, redact_phi=True, out=buf)
    output = buf.getvalue()
    assert "PHI fields" in output and "masked" in output
