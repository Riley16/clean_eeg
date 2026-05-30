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
    DEFAULT_SIGNAL_PREVIEW_COUNT,
    _gather_paths,
    _is_annotation_stub,
    _parse_record_tals,
    _has_annotation_stub_sibling,
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
    ``redact_phi=True``. ``full_signal=True`` so the assertion sees
    every channel — the default cap is exercised separately."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, redact_phi=True, full_signal=True, out=buf)
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


# ---- _gather_paths: recursive + annotation-stub filtering ----

def _touch(path):
    """Create an empty placeholder file. _gather_paths walks the
    filesystem only — it does not parse the EDFs — so an empty file is
    sufficient for the path-collection tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_is_annotation_stub_recognizes_inplace_sibling_suffix():
    """``*_annotations.edf`` is the exact suffix used by
    create_annotations_only_edf for the inplace-mode sibling stub.
    Match it case-insensitively (Windows volumes uppercase extensions)."""
    assert _is_annotation_stub("EEG2412_R1764A_03.04__12.13.14_annotations.edf")
    assert _is_annotation_stub("foo_ANNOTATIONS.EDF")
    # Not a stub: the suffix has to be the literal '_annotations.edf' tail.
    assert not _is_annotation_stub("subject_data.edf")
    assert not _is_annotation_stub("annotations.edf")  # no leading underscore
    assert not _is_annotation_stub("notes_annotations.txt")


def test_gather_paths_directory_skips_annotation_stubs_by_default(tmp_path):
    """Default directory scan filters out the inplace-mode stubs so the
    operator sees one row per recording, not two."""
    main = _touch(tmp_path / "EEG2412_R1764A_03.04__12.13.14.edf")
    _touch(tmp_path / "EEG2412_R1764A_03.04__12.13.14_annotations.edf")
    paths = _gather_paths(str(tmp_path))
    assert paths == [str(main)]


def test_gather_paths_include_annotation_stubs_opt_in(tmp_path):
    """``--include-annotation-stubs`` brings the stubs back."""
    main = _touch(tmp_path / "main.edf")
    stub = _touch(tmp_path / "main_annotations.edf")
    paths = _gather_paths(str(tmp_path), include_annotation_stubs=True)
    assert paths == [str(main), str(stub)]


def test_gather_paths_recursive_walks_subdirectories(tmp_path):
    """``-r`` should pick up EDFs nested under subject/session folders.
    Without the flag the same call must NOT descend — that's the
    pre-existing shallow contract."""
    top = _touch(tmp_path / "top.edf")
    nested = _touch(tmp_path / "R1764A" / "session1" / "deep.edf")
    _touch(tmp_path / "R1764A" / "session1" / "deep_annotations.edf")
    shallow = _gather_paths(str(tmp_path), recursive=False)
    assert shallow == [str(top)]
    deep = _gather_paths(str(tmp_path), recursive=True)
    assert deep == [str(nested), str(top)]  # sorted


def test_gather_paths_recursive_with_include_stubs(tmp_path):
    """Recursive + include-annotation-stubs should yield every .edf."""
    a = _touch(tmp_path / "a.edf")
    a_stub = _touch(tmp_path / "a_annotations.edf")
    nested = _touch(tmp_path / "sub" / "b.edf")
    nested_stub = _touch(tmp_path / "sub" / "b_annotations.edf")
    paths = _gather_paths(str(tmp_path), recursive=True,
                          include_annotation_stubs=True)
    assert paths == sorted([str(a), str(a_stub),
                            str(nested), str(nested_stub)])


# ---- presentation: cap, GB filesize, no byte-range prefix ----

def test_default_shows_label_summary_plus_one_example(tmp_path):
    """Default behavior: print all channel labels in a one-line
    comma-separated summary so the operator sees what's in the file,
    then dump exactly DEFAULT_SIGNAL_PREVIEW_COUNT full signal block
    as a formatting example. Multi-hundred-channel NK exports would
    otherwise flood the terminal."""
    assert DEFAULT_SIGNAL_PREVIEW_COUNT == 1  # if this moves, update --full-signal help text
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, out=buf)
    output = buf.getvalue()
    # Every label appears (in the summary line) so the user can audit
    # what channels are present without flag-toggling.
    for label in ("CH0", "CH1", "CH2"):
        assert label in output
    # ...but only ONE channel gets the full per-field block. Count by
    # looking for the per-signal-block "transducer" row, which is
    # unique to the per-signal expansion (it never appears in a
    # main-header row or the summary line).
    transducer_rows = [line for line in output.splitlines()
                       if line.lstrip().startswith("transducer")]
    assert len(transducer_rows) == DEFAULT_SIGNAL_PREVIEW_COUNT
    # Summary line is labeled, and the user-facing note points at the
    # escape hatch.
    assert "Labels:" in output
    assert "--full-signal" in output


def test_full_signal_flag_shows_every_channel(tmp_path):
    """``full_signal=True`` overrides the default and prints every
    signal's full block. The label-summary line is suppressed in this
    mode — every label is already visible inside its own block, so
    the summary would just duplicate."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, full_signal=True, out=buf)
    output = buf.getvalue()
    for label in ("CH0", "CH1", "CH2"):
        assert label in output
    # Every signal gets a full block — count by per-signal header
    # lines ("\n  Signal i  label=..."). Generated EDFs include an
    # EDF Annotations channel alongside the data channels, so the
    # exact total exceeds the requested n_channels by one. The
    # contract being tested is "no truncation" rather than the
    # exact count, so check it matches the n_signals advertised in
    # the dump itself.
    import re
    n_signals_match = re.search(r"# Signal headers \((\d+) signals\)", output)
    assert n_signals_match, "expected 'Signal headers (N signals)' line"
    n_signals_advertised = int(n_signals_match.group(1))
    signal_blocks = re.findall(r"^  Signal \d+  label=", output, re.MULTILINE)
    assert len(signal_blocks) == n_signals_advertised
    assert n_signals_advertised >= 3  # 3 data channels are present
    # No label-summary line when full_signal=True.
    assert "Labels:" not in output


def test_explicit_signal_indices_override_cap(tmp_path):
    """``signal_indices=[2]`` is a deliberate selection — show that
    signal even though it's past the default cap, and skip the cap
    note (the user already knows what they asked for)."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, signal_indices=[2], out=buf)
    output = buf.getvalue()
    assert "CH2" in output
    assert "CH0" not in output and "CH1" not in output
    assert "--full-signal" not in output


def test_filesize_reported_in_gb(tmp_path):
    """Filesize line uses GiB (matching scan_cluster_edfs.py), 3
    decimal places. The toy file is tiny so the value is 0.000 GB —
    that's fine; what we're locking in is the unit string."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, out=buf)
    output = buf.getvalue()
    assert "filesize:" in output and " GB" in output
    # Old "filesize: <bytes> bytes" format must be gone.
    assert "bytes\n" not in output.split("filesize:", 1)[1].splitlines()[0]


def test_field_lines_omit_byte_range_prefix(tmp_path):
    """Removed for readability: rows used to lead with
    ``  bytes 168-175 ( 8B)  startdate  ...``. After the cleanup the
    line starts with ``  startdate  ...`` and there is no row anywhere
    in the output that starts with the literal ``  bytes ``."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, full_signal=True, out=buf)
    output = buf.getvalue()
    for line in output.splitlines():
        assert not line.startswith("  bytes "), \
            f"unexpected byte-range prefix: {line!r}"


def test_field_lines_show_parsed_before_raw(tmp_path):
    """Operators care about the parsed value first; raw bytes are a
    fallback for when parsing fails. The row order should reflect
    that — ``parsed=...`` must appear before ``raw=...`` on every
    rendered field row."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, full_signal=True, out=buf)
    output = buf.getvalue()
    rows_with_both = [
        line for line in output.splitlines()
        if "parsed=" in line and "raw=" in line
    ]
    assert rows_with_both, "expected at least one field row to be rendered"
    for line in rows_with_both:
        assert line.index("parsed=") < line.index("raw="), \
            f"parsed= should precede raw= in: {line!r}"


def test_main_header_raw_column_is_aligned(tmp_path):
    """All main-header rows should land their ``raw=`` token at the
    same column. Without padding, the column staircases by parsed-value
    width — which is exactly the visual noise the alignment is meant
    to remove."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, out=buf)
    output = buf.getvalue()
    main_rows = [
        line for line in output.splitlines()
        # Two-space indent + "parsed=" + "raw=" identifies the main-
        # header rows (signal-block rows lead with four spaces).
        if line.startswith("  ") and not line.startswith("   ")
        and "  parsed=" in line and "  raw=" in line
    ]
    assert len(main_rows) >= 5  # main header has 10 fields
    raw_columns = {line.index("  raw=") for line in main_rows}
    assert len(raw_columns) == 1, \
        f"main-header raw= columns misaligned: {sorted(raw_columns)} in:\n" + "\n".join(main_rows)


def test_signal_block_raw_column_is_aligned(tmp_path):
    """Within a single signal block, every ``raw=`` should land at the
    same column. We test the block by itself rather than across blocks
    because the helper aligns per-block — different signals can carry
    different parsed-value widths and that's fine."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, out=buf)
    output = buf.getvalue()
    # The default mode prints exactly one signal block. Pick the rows
    # that lead with four spaces and contain both tokens.
    signal_rows = [
        line for line in output.splitlines()
        if line.startswith("    ") and "  parsed=" in line and "  raw=" in line
    ]
    assert len(signal_rows) >= 5  # signal block has 9 fields minus 'label'
    raw_columns = {line.index("  raw=") for line in signal_rows}
    assert len(raw_columns) == 1, \
        f"signal-block raw= columns misaligned: {sorted(raw_columns)} in:\n" + "\n".join(signal_rows)


def test_redact_phi_row_also_shows_parsed_before_raw(tmp_path):
    """The redaction path uses a separate format string — confirm it
    keeps the parsed-before-raw ordering too."""
    edf = _make_edf(tmp_path)
    buf = io.StringIO()
    print_header(edf, redact_phi=True, out=buf)
    output = buf.getvalue()
    redacted_rows = [line for line in output.splitlines()
                     if "[PHI_REDACTED]" in line and "parsed=" in line]
    assert redacted_rows, "expected at least one PHI-redacted row"
    for line in redacted_rows:
        assert line.index("parsed=") < line.index("raw="), \
            f"parsed= should precede raw= in: {line!r}"


def test_gather_paths_file_argument_unaffected_by_recursive(tmp_path):
    """When PATH points at a file, recursive/include-stubs are no-ops:
    the file is returned even if it's an annotation stub (the user
    explicitly named it)."""
    stub = _touch(tmp_path / "x_annotations.edf")
    assert _gather_paths(str(stub)) == [str(stub)]
    assert _gather_paths(str(stub), recursive=True) == [str(stub)]


# ---- annotation TAL parser (unit) ----

def test_parse_record_tals_timekeeping_only():
    """A record containing only the timekeeping TAL should yield a
    single TAL with empty texts."""
    record = b"+0\x14\x14\x00" + b"\x00" * 50  # padded to fixed record size
    tals = _parse_record_tals(record)
    assert len(tals) == 1
    onset, duration, texts = tals[0]
    assert onset == 0.0
    assert duration is None
    assert texts == []


def test_parse_record_tals_extracts_real_annotation_text():
    """A timekeeping TAL followed by one real annotation should yield
    two TALs — the first with empty texts, the second with the actual
    text."""
    record = b"+0\x14\x14\x00+1.5\x14seizure\x14\x00" + b"\x00" * 30
    tals = _parse_record_tals(record)
    assert len(tals) == 2
    assert tals[0][2] == []  # timekeeping
    onset, duration, texts = tals[1]
    assert onset == 1.5
    assert duration is None
    assert texts == ["seizure"]


def test_parse_record_tals_extracts_duration():
    """When ``\\x15<duration>`` follows the onset, the duration should
    be parsed too."""
    record = b"+0\x14\x14\x00+2.5\x152.0\x14event\x14\x00"
    tals = _parse_record_tals(record)
    assert len(tals) == 2
    assert tals[1] == (2.5, 2.0, ["event"])


def test_parse_record_tals_skips_malformed_chunks():
    """Bytes that can't be decoded as a TAL (e.g. random data, no
    field separator) should be silently skipped — never raise."""
    record = b"+0\x14\x14\x00garbage_with_no_separator\x00+1\x14ok\x14\x00"
    tals = _parse_record_tals(record)
    # Should successfully extract the timekeeping + real TAL, ignoring
    # the malformed middle chunk.
    onsets = [tal[0] for tal in tals]
    assert 0.0 in onsets
    assert 1.0 in onsets
    assert all(isinstance(tal[2], list) for tal in tals)


# ---- annotations section (end-to-end) ----

def _make_annotated_edf(tmp_path, filename, annotations):
    """Generate a small EDF with the given annotations using the
    project's existing fixture helper. Returns the path."""
    import datetime
    import pyedflib
    from tests.generate_edf import generate_test_edf
    header = {
        "patientname": "X", "patientcode": "X", "sex": "X",
        "birthdate": "", "patient_additional": "", "admincode": "X",
        "technician": "X", "equipment": "X", "recording_additional": "",
        "startdate": datetime.datetime(2025, 1, 1),
    }
    sigs = [{"label": f"C{i}", "dimension": "uV",
             "sample_frequency": 100, "physical_min": -3200,
             "physical_max": 3200, "digital_min": -32768,
             "digital_max": 32767, "transducer": "", "prefilter": ""}
            for i in range(2)]
    # generate_test_edf hard-codes a 5-annotation list; for tests
    # that need a custom annotation set, write directly.
    import numpy as np
    path = tmp_path / filename
    with pyedflib.EdfWriter(str(path), 2,
                            file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader(header)
        f.setSignalHeaders(sigs)
        f.writeSamples([np.zeros(500, dtype=np.float64),
                        np.zeros(500, dtype=np.float64)])
        for onset, duration, text in annotations:
            f.writeAnnotation(onset, duration, text)
    return path


def test_annotations_section_lists_real_events(tmp_path):
    """The annotations section should print onset/duration/text for
    each non-timekeeping TAL — that's the inspection-stub use case."""
    edf = _make_annotated_edf(tmp_path, "src.edf", [
        (0.5, -1, "Seizure onset"),
        (2.5, 1.5, "Patient awake"),
    ])
    buf = io.StringIO()
    print_header(edf, out=buf)
    output = buf.getvalue()
    assert "# Annotations" in output
    assert "label='EDF Annotations'" in output
    assert "non-timekeeping events: 2" in output
    assert "Seizure onset" in output
    assert "Patient awake" in output


def test_annotations_section_handles_no_annotation_channel(tmp_path):
    """Pyedflib always adds an EDF Annotations channel for EDF+ files,
    so to test the 'no channel' path we synthesize a minimal EDF
    without one — a plain EDF (not EDF+)."""
    import pyedflib
    import numpy as np
    path = tmp_path / "plain.edf"
    sigs = [{"label": "C0", "dimension": "uV", "sample_frequency": 100,
             "physical_min": -3200, "physical_max": 3200,
             "digital_min": -32768, "digital_max": 32767,
             "transducer": "", "prefilter": ""}]
    with pyedflib.EdfWriter(str(path), 1,
                            file_type=pyedflib.FILETYPE_EDF) as f:
        f.setSignalHeaders(sigs)
        f.writeSamples([np.zeros(500, dtype=np.float64)])
    buf = io.StringIO()
    print_header(path, out=buf)
    output = buf.getvalue()
    assert "# Annotations" in output
    assert "no annotation channel" in output


def test_annotations_section_masks_text_with_redact_phi(tmp_path):
    """Annotation text can contain PHI (e.g. patient names attached
    to events). Under ``--redact-phi`` it must be masked. Onset and
    duration are not PHI per spec — they remain visible."""
    edf = _make_annotated_edf(tmp_path, "src.edf", [
        (0.5, -1, "Patient John Smith awake"),
    ])
    buf = io.StringIO()
    print_header(edf, redact_phi=True, out=buf)
    output = buf.getvalue()
    assert "John Smith" not in output
    assert "[PHI_REDACTED]" in output
    # Onset survived — that's the operator's anchor for the event.
    assert "0.5000" in output


# ---- strip-check warning ----

def test_strip_warning_fires_when_main_has_sibling_and_non_tk_annotations(tmp_path):
    """Inplace-mode invariant: the main EDF must have its annotations
    stripped (only timekeeping TALs remain). If we see a sibling
    ``*_annotations.edf`` AND the main still carries non-timekeeping
    annotations, the de-id pipeline failed — print a loud banner."""
    main = _make_annotated_edf(tmp_path, "EEG_R1764A_03.04__12.13.14.edf", [
        (0.5, -1, "Seizure onset"),  # not stripped — should trip warning
    ])
    # Create the sibling stub (contents don't matter for the warning,
    # only its existence). Re-using the same fixture is fine.
    _make_annotated_edf(
        tmp_path, "EEG_R1764A_03.04__12.13.14_annotations.edf",
        [(0.5, -1, "Seizure onset")])
    buf = io.StringIO()
    print_header(main, out=buf)
    output = buf.getvalue()
    assert "WARNING" in output
    assert "DO NOT SHARE" in output
    assert "non-timekeeping annotation" in output


def test_strip_warning_silent_when_no_sibling_stub_present(tmp_path):
    """Without a sibling stub, the file is presumed to be a copy-mode
    output (annotations live in the main EDF, that's the normal
    state). The warning should NOT fire."""
    main = _make_annotated_edf(tmp_path, "copy_mode_output.edf", [
        (0.5, -1, "Seizure onset"),
    ])
    buf = io.StringIO()
    print_header(main, out=buf)
    output = buf.getvalue()
    assert "WARNING" not in output
    assert "non-timekeeping events: 1" in output


def test_strip_warning_silent_on_annotation_stub_itself(tmp_path):
    """The stub IS the file holding the annotations — printing them
    is the whole point of `--include-annotation-stubs`. The warning
    is for main EDFs only (it would be triggered in the wrong
    direction for a stub)."""
    stub = _make_annotated_edf(tmp_path,
                               "EEG_R1764A_03.04__12.13.14_annotations.edf",
                               [(0.5, -1, "Seizure onset")])
    # Even with a 'main' file present, the file we're inspecting is
    # the stub, so the warning shouldn't be about this one.
    _make_annotated_edf(tmp_path, "EEG_R1764A_03.04__12.13.14.edf", [])
    buf = io.StringIO()
    print_header(stub, out=buf)
    output = buf.getvalue()
    assert "WARNING" not in output
    assert "Seizure onset" in output  # but the contents ARE shown


def test_has_annotation_stub_sibling_detects_pair(tmp_path):
    """Helper unit test — sibling detection is purely filesystem
    based, no I/O on the EDF itself."""
    main = _touch(tmp_path / "main.edf")
    assert not _has_annotation_stub_sibling(str(main))
    _touch(tmp_path / "main_annotations.edf")
    assert _has_annotation_stub_sibling(str(main))
    # The stub itself reports False — it's not a "main with sibling".
    stub = tmp_path / "other_annotations.edf"
    stub.write_bytes(b"")
    assert not _has_annotation_stub_sibling(str(stub))
