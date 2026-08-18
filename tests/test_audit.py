"""Tests for the per-subject post-transfer audit checks."""

from __future__ import annotations

from pathlib import Path

from clean_eeg.audit.checks import (
    check_annotation_pairing,
    check_byte_geometry,
    check_filename_convention,
    check_header_phi_residue,
    check_recording_gaps,
    check_signal_header_uniformity,
    check_subject_code_consistency,
)
from clean_eeg.audit.annotations import (
    check_annotation_phi_scan,
    extract_annotations,
    scan_annotation_texts,
)
from clean_eeg.audit.hashes import (
    check_transfer_integrity,
    sha256_fast_of_file,
    sha256_of_file,
)
from clean_eeg.audit.logs import check_log_file
from clean_eeg.audit.select import select_files
from clean_eeg.audit.signals import read_signal_window
from clean_eeg.audit.notebook import build_audit_notebook
from clean_eeg.audit.subject import (
    AUDIT_JSON_FILENAME,
    IN_PROGRESS_FILENAME,
    AuditInterruptedError,
    audit_subject,
)


_SENTINEL_PID = "R1755J X 01-JAN-1900 unknown unknown"


def _write_edf_stub(path: Path,
                    patient_id: str = _SENTINEL_PID,
                    startdate: str = "01.01.85",
                    starttime: str = "00.00.00",
                    recording_id: str = "",
                    n_records: int = -1,
                    record_duration: float = 1.0) -> None:
    """Write a minimal 256-byte EDF main header. Only the fields the
    audit reads (patient_id, startdate, starttime, recording_id,
    n_records, record_duration) are meaningful; the rest are
    ASCII-padded with spaces.
    """
    if len(patient_id) > 80:
        raise ValueError("patient_id must fit in 80 bytes")
    if len(startdate) != 8:
        raise ValueError("startdate must be exactly 8 bytes (DD.MM.YY)")
    if len(starttime) != 8:
        raise ValueError("starttime must be exactly 8 bytes (HH.MM.SS)")
    header = bytearray(b" " * 256)
    header[0:8] = b"0".ljust(8, b" ")
    header[8:88] = patient_id.encode("ascii").ljust(80, b" ")
    header[88:168] = recording_id.encode("ascii").ljust(80, b" ")
    header[168:176] = startdate.encode("ascii")
    header[176:184] = starttime.encode("ascii")
    header[184:192] = b"     256"
    header[192:236] = b"EDF+C".ljust(44, b" ")
    header[236:244] = f"{n_records:>8d}".encode("ascii")
    header[244:252] = f"{record_duration:>8g}".encode("ascii")
    header[252:256] = b"   0"
    path.write_bytes(bytes(header))


def _write_edf_with_signals(path: Path,
                            n_records: int,
                            samples_per_record: int,
                            n_signals: int = 1,
                            data_bytes_override: int | None = None,
                            starttime: str = "00.00.00",
                            label_prefix: str = "EEG",
                            phys_min: float = -3200.0,
                            phys_max: float = 3200.0,
                            dig_min: int = -32768,
                            dig_max: int = 32767,
                            phys_dim: str = "uV",
                            channel_samples: list | None = None) -> None:
    """Write a valid-geometry EDF stub with real signal headers + data.

    ``data_bytes_override`` lets tests deliberately produce TRUNCATED
    (less than expected) or OVER-SIZED (more than expected) files. The
    label/phys/dig kwargs let tests vary signal-header signatures.
    """
    main = bytearray(b" " * 256)
    main[0:8] = b"0".ljust(8, b" ")
    main[8:88] = _SENTINEL_PID.encode("ascii").ljust(80, b" ")
    main[88:168] = b" " * 80
    main[168:176] = b"01.01.85"
    main[176:184] = starttime.encode("ascii")
    main[184:192] = f"{256 * (1 + n_signals):>8d}".encode("ascii")
    main[192:236] = b"EDF+C".ljust(44, b" ")
    main[236:244] = f"{n_records:>8d}".encode("ascii")
    main[244:252] = b"       1"
    main[252:256] = f"{n_signals:>4d}".encode("ascii")

    # Signal-header block: fields laid out as [all labels][all transducers]...
    sig_block = bytearray(b" " * (256 * n_signals))
    def _write(off_per_sig: int, width: int, value: bytes) -> None:
        base = off_per_sig * n_signals
        for i in range(n_signals):
            sig_block[base + i * width:base + (i + 1) * width] = value.ljust(width, b" ")[:width]

    for i in range(n_signals):
        sig_block[i * 16:(i + 1) * 16] = f"{label_prefix}{i}".encode("ascii").ljust(16, b" ")
    _write(96,  8, phys_dim.encode("ascii"))
    _write(104, 8, f"{phys_min:>8g}".encode("ascii"))
    _write(112, 8, f"{phys_max:>8g}".encode("ascii"))
    _write(120, 8, f"{dig_min:>8d}".encode("ascii"))
    _write(128, 8, f"{dig_max:>8d}".encode("ascii"))
    _write(216, 8, f"{samples_per_record:>8d}".encode("ascii"))

    record_bytes = samples_per_record * n_signals * 2
    if data_bytes_override is not None:
        data = b"\x00" * data_bytes_override
    elif channel_samples is not None:
        # Interleave per-record: [ch0_record0][ch1_record0]...[ch0_record1]...
        import numpy as np
        if len(channel_samples) != n_signals:
            raise ValueError(f"channel_samples must have {n_signals} entries")
        arr = np.zeros((n_records, n_signals * samples_per_record), dtype="<i2")
        for i, samples in enumerate(channel_samples):
            samples = np.asarray(samples, dtype="<i2")
            if samples.size != n_records * samples_per_record:
                raise ValueError(
                    f"channel {i} has {samples.size} samples, "
                    f"expected {n_records * samples_per_record}")
            arr[:, i * samples_per_record:(i + 1) * samples_per_record] = \
                samples.reshape(n_records, samples_per_record)
        data = arr.tobytes()
    else:
        data = b"\x00" * (n_records * record_bytes)

    path.write_bytes(bytes(main) + bytes(sig_block) + data)


def _encode_tal_record(record_bytes_size: int,
                       record_start: float,
                       events: list[tuple[float, float | None, str]]) -> bytes:
    """Build one EDF+ annotation-channel record: timekeeping TAL first,
    then event TALs, then null-padding to ``record_bytes_size``.
    """
    def _tal(onset: float, duration: float | None, text: str) -> bytes:
        onset_s = f"{'+' if onset >= 0 else ''}{onset:g}"
        if duration is None:
            head = onset_s.encode("ascii")
        else:
            head = f"{onset_s}\x15{duration:g}".encode("ascii")
        return head + b"\x14" + text.encode("utf-8") + b"\x14\x00"

    body = _tal(record_start, None, "")
    for onset, dur, text in events:
        body += _tal(onset, dur, text)
    if len(body) > record_bytes_size:
        raise ValueError("annotation TALs exceed record size")
    return body + b"\x00" * (record_bytes_size - len(body))


def _write_edf_with_annotations(path: Path,
                                annotations: list[tuple[float, float | None, str]],
                                *,
                                ann_bytes_per_record: int = 128,
                                record_duration: float = 1.0) -> None:
    """Write a minimal EDF+ file whose single signal is the annotation
    channel, exercising ``extract_annotations`` end-to-end without pyedflib.
    """
    n_signals = 1
    samples_per_record = ann_bytes_per_record // 2
    n_records = 1

    main = bytearray(b" " * 256)
    main[0:8] = b"0".ljust(8, b" ")
    main[8:88] = _SENTINEL_PID.encode("ascii").ljust(80, b" ")
    main[88:168] = b" " * 80
    main[168:176] = b"01.01.85"
    main[176:184] = b"00.00.00"
    main[184:192] = f"{256 * (1 + n_signals):>8d}".encode("ascii")
    main[192:236] = b"EDF+C".ljust(44, b" ")
    main[236:244] = f"{n_records:>8d}".encode("ascii")
    main[244:252] = f"{record_duration:>8g}".encode("ascii")
    main[252:256] = f"{n_signals:>4d}".encode("ascii")

    sig_block = bytearray(b" " * 256)
    sig_block[0:16] = b"EDF Annotations".ljust(16, b" ")
    sig_block[104:112] = b"      -1"
    sig_block[112:120] = b"       1"
    sig_block[120:128] = b"  -32768"
    sig_block[128:136] = b"   32767"
    sig_block[216:224] = f"{samples_per_record:>8d}".encode("ascii")

    data = _encode_tal_record(ann_bytes_per_record, 0.0, annotations)
    path.write_bytes(bytes(main) + bytes(sig_block) + data)


def test_pass_valid_subject_code_across_files(tmp_path):
    for name in ("a.edf", "b.edf", "c.edf"):
        _write_edf_stub(tmp_path / name, "R1755J")

    result = check_subject_code_consistency(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert result["subject_code"] == "R1755J"
    assert result["n_files"] == 3
    assert result["unique_subject_codes"] == ["R1755J"]
    assert result["issues"] == []


def test_pass_realistic_edfplus_patient_id(tmp_path):
    """A properly-cleaned pyedflib EDF+ patient_id has 5 space-separated
    subfields (code, sex, birthdate, first, last). The audit must extract
    just the subject-code token before matching against the pattern.
    """
    pid = "R1770J X 01-JAN-1900 unknown unknown"
    _write_edf_stub(tmp_path / "a.edf", pid)
    _write_edf_stub(tmp_path / "b.edf", pid)

    result = check_subject_code_consistency(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert result["subject_code"] == "R1770J"
    assert result["subject_codes_by_file"] == {"a.edf": "R1770J", "b.edf": "R1770J"}
    assert result["patient_ids_by_file"] == {"a.edf": pid, "b.edf": pid}


def test_warn_patientcode_not_matching_pattern(tmp_path):
    _write_edf_stub(tmp_path / "a.edf", "1234567")

    result = check_subject_code_consistency([tmp_path / "a.edf"])

    assert result["status"] == "warn"
    assert result["subject_code"] is None
    assert result["non_matching_subject_codes"] == ["1234567"]
    assert any("does not match" in msg for msg in result["issues"])


def test_fail_mixed_subject_codes(tmp_path):
    _write_edf_stub(tmp_path / "a.edf", "R1755J")
    _write_edf_stub(tmp_path / "b.edf", "R1756J")

    result = check_subject_code_consistency(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert result["subject_code"] is None
    assert set(result["unique_subject_codes"]) == {"R1755J", "R1756J"}
    assert any("Multiple distinct" in msg for msg in result["issues"])


def test_fail_mixed_when_first_token_differs(tmp_path):
    """Subfields after the code may differ across cleaned files (birthdate
    is a constant '01-JAN-1900', but rare cases in older pipelines might
    diverge) — the check should still pass if the *code* matches. Here we
    verify the inverse: same trailing subfields but different codes still fails.
    """
    _write_edf_stub(tmp_path / "a.edf", "R1755J X 01-JAN-1900 unknown unknown")
    _write_edf_stub(tmp_path / "b.edf", "R1756J X 01-JAN-1900 unknown unknown")

    result = check_subject_code_consistency(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert set(result["unique_subject_codes"]) == {"R1755J", "R1756J"}


def test_pass_patient_id_with_trailing_whitespace(tmp_path):
    # pyedflib right-pads to 80 bytes; the raw string retains trailing
    # spaces that .strip().split() must handle.
    _write_edf_stub(tmp_path / "a.edf", patient_id="R1755J")  # no subfields
    result = check_subject_code_consistency([tmp_path / "a.edf"])
    assert result["status"] == "pass"
    assert result["subject_code"] == "R1755J"


def test_fail_empty_input():
    result = check_subject_code_consistency([])

    assert result["status"] == "fail"
    assert result["n_files"] == 0
    assert result["subject_code"] is None
    assert any("No EDF files" in msg for msg in result["issues"])


# --- filename convention ----------------------------------------------------


def test_filename_convention_pass_pipeline_output(tmp_path):
    # The pipeline renames to {stem}_{subject_code}_{MM.DD__HH.MM.SS}.edf
    for name in (
        "NA3621LS_R1747A_01.14__14.32.22.edf",
        "NA3621LT_R1747A_01.14__15.22.10.edf",
        "NA3621LS_R1747A_01.14__14.32.22_annotations.edf",  # stub sidecar
    ):
        _write_edf_stub(tmp_path / name)
    result = check_filename_convention(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "pass"
    assert result["unrenamed_files"] == []


def test_filename_convention_fail_missing_timestamp_suffix(tmp_path):
    # The problematic file from the user report: original name kept,
    # no _R1XXXY_MM.DD__HH.MM.SS suffix. Sibling that WAS renamed
    # should still be recognized as OK.
    _write_edf_stub(tmp_path / "NA3621LS_R1747A_01.14__14.32.22.edf")
    _write_edf_stub(tmp_path / "NA3621K6.edf")  # bypassed the pipeline
    result = check_filename_convention(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["unrenamed_files"] == ["NA3621K6.edf"]
    assert any("bypassed" in msg and "NA3621K6.edf" in msg for msg in result["issues"])


def test_filename_convention_fail_empty_input():
    result = check_filename_convention([])
    assert result["status"] == "fail"


# --- header PHI-residue -----------------------------------------------------


def test_residue_pass_cleaned_subject(tmp_path):
    _write_edf_stub(tmp_path / "a.edf", startdate="01.01.85")
    _write_edf_stub(tmp_path / "b.edf", startdate="15.01.85")
    _write_edf_stub(tmp_path / "c.edf", startdate="03.02.85")

    result = check_header_phi_residue(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert result["n_files"] == 3
    assert result["earliest_startdate"] == "1985-01-01"
    assert result["unexpected_patient_id_tokens_by_file"] == {}
    assert result["issues"] == []


def test_residue_fail_leaked_name_in_patient_id(tmp_path):
    _write_edf_stub(tmp_path / "clean.edf")
    _write_edf_stub(tmp_path / "leaked.edf",
                    patient_id="R1755J X 01-JAN-1900 John Smith")

    result = check_header_phi_residue(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert result["unexpected_patient_id_tokens_by_file"] == {
        "leaked.edf": ["John", "Smith"],
    }
    assert any("non-sentinel tokens" in msg and "leaked.edf" in msg
               for msg in result["issues"])


def test_residue_fail_real_year_in_startdate(tmp_path):
    _write_edf_stub(tmp_path / "clean.edf", startdate="01.01.85")
    _write_edf_stub(tmp_path / "leaked.edf", startdate="15.07.24")  # 2024

    result = check_header_phi_residue(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert any("year 2024" in msg for msg in result["issues"])


def test_residue_warn_earliest_not_base(tmp_path):
    # All sentinel tokens fine, dates parse and are in year range, but
    # earliest is 1985-01-15 instead of 1985-01-01 — pipeline invariant
    # violation without any actual PHI leak, so warn.
    _write_edf_stub(tmp_path / "a.edf", startdate="15.01.85")
    _write_edf_stub(tmp_path / "b.edf", startdate="16.01.85")

    result = check_header_phi_residue(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "warn"
    assert result["earliest_startdate"] == "1985-01-15"
    assert any("Earliest startdate" in msg for msg in result["issues"])


def test_residue_fail_unparseable_startdate(tmp_path):
    _write_edf_stub(tmp_path / "bad.edf", startdate="ZZ.ZZ.ZZ")

    result = check_header_phi_residue([tmp_path / "bad.edf"])

    assert result["status"] == "fail"
    assert any("unparseable startdate" in msg for msg in result["issues"])


def test_residue_pass_subject_code_only_patient_id(tmp_path):
    # Degenerate cleaned file where patient_id contains only the
    # subject code (no subfields). No tokens[1:] to leak PHI into.
    _write_edf_stub(tmp_path / "a.edf", patient_id="R1755J")
    result = check_header_phi_residue([tmp_path / "a.edf"])
    assert result["status"] == "pass"
    assert result["unexpected_patient_id_tokens_by_file"] == {}


def test_residue_permits_recording_span_up_to_max_years(tmp_path):
    # Long recording that lands in 1987 — inside the default 2-year cap.
    _write_edf_stub(tmp_path / "a.edf", startdate="01.01.85")
    _write_edf_stub(tmp_path / "b.edf", startdate="31.12.87")

    result = check_header_phi_residue(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert result["expected_year_range"] == [1985, 1987]


def test_residue_pass_recording_id_in_expected_year_range(tmp_path):
    _write_edf_stub(tmp_path / "a.edf", startdate="01.01.85",
                    recording_id="Startdate 01-JAN-1985 X X NKC-EEG-1200A_V01.00")
    result = check_header_phi_residue([tmp_path / "a.edf"])
    assert result["status"] == "pass"
    assert result["recording_id_years_by_file"]["a.edf"] == 1985


def test_residue_fail_real_year_in_recording_id_bypasses_header_shift(tmp_path):
    """The collaborator-cleaned file the user reported: patient_id was
    scrubbed to sentinels but the recording_id still embeds the real
    recording year. Distinct from the startdate check because a file
    could have startdate shifted but recording_id still carry the
    original year if the write path was patchwork.
    """
    _write_edf_stub(
        tmp_path / "clean.edf",
        startdate="01.01.85",
        recording_id="Startdate 01-JAN-1985 X X NKC-EEG-1200A_V01.00",
    )
    _write_edf_stub(
        tmp_path / "leaked.edf",
        startdate="01.01.85",
        recording_id="Startdate 13-JUN-2024 X X NKC-EEG-1200A_V01.00",
    )
    result = check_header_phi_residue(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["recording_id_years_by_file"]["leaked.edf"] == 2024
    assert any("recording_id embedded year 2024" in msg for msg in result["issues"])
    assert any("bypassed" in msg for msg in result["issues"])


def test_residue_recording_id_missing_startdate_prefix_is_None(tmp_path):
    # Non-EDF+-standard recording_id without the "Startdate DD-MMM-YYYY"
    # prefix should NOT be reported as year-off — we can't derive a year.
    _write_edf_stub(tmp_path / "a.edf",
                    recording_id="Free-form recording metadata")
    result = check_header_phi_residue([tmp_path / "a.edf"])
    assert result["recording_id_years_by_file"]["a.edf"] is None
    assert not any("recording_id embedded year" in msg for msg in result["issues"])


# --- recording gaps ---------------------------------------------------------


def test_gaps_pass_single_file(tmp_path):
    _write_edf_stub(tmp_path / "only.edf",
                    starttime="00.00.00", n_records=3600, record_duration=1.0)

    result = check_recording_gaps([tmp_path / "only.edf"])

    assert result["status"] == "pass"
    assert result["gaps"] == []
    assert result["large_gaps"] == []
    assert result["overlaps"] == []
    assert result["files_by_start"][0]["duration_s"] == 3600.0


def test_gaps_pass_contiguous_files(tmp_path):
    # Three 1-hour files back-to-back: 00:00, 01:00, 02:00. Zero gap.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=3600, record_duration=1.0)
    _write_edf_stub(tmp_path / "b.edf",
                    starttime="01.00.00", n_records=3600, record_duration=1.0)
    _write_edf_stub(tmp_path / "c.edf",
                    starttime="02.00.00", n_records=3600, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert len(result["gaps"]) == 2
    assert all(g["gap_seconds"] == 0.0 for g in result["gaps"])


def test_gaps_pass_within_threshold(tmp_path):
    # Two 1-hour files with a 5-minute gap. Per-pair threshold =
    # min(600, 3600-300) = 600 s (10 min). 300 s gap < 600 s → pass.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=3600, record_duration=1.0)
    _write_edf_stub(tmp_path / "b.edf",
                    starttime="01.05.00", n_records=3600, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert result["gaps"][0]["gap_seconds"] == 300.0
    assert result["gaps"][0]["threshold_seconds"] == 600.0


def test_gaps_pass_at_absolute_cap(tmp_path):
    # Two 1-hour files with a 9-min gap — under the 10-min absolute cap.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=3600, record_duration=1.0)
    _write_edf_stub(tmp_path / "b.edf",
                    starttime="01.09.00", n_records=3600, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert result["gaps"][0]["gap_seconds"] == 540.0


def test_gaps_threshold_scales_with_short_recordings(tmp_path):
    # Two 10-min files: threshold = min(600, 600-300) = 300 s.
    # A 4-min (240 s) gap passes; a 6-min (360 s) gap fails.
    _write_edf_stub(tmp_path / "short_a.edf",
                    starttime="00.00.00", n_records=600, record_duration=1.0)
    _write_edf_stub(tmp_path / "short_b.edf",
                    starttime="00.14.00", n_records=600, record_duration=1.0)  # 4 min gap
    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "pass"
    assert result["gaps"][0]["threshold_seconds"] == 300.0

    for f in tmp_path.glob("*.edf"):
        f.unlink()
    _write_edf_stub(tmp_path / "short_a.edf",
                    starttime="00.00.00", n_records=600, record_duration=1.0)
    _write_edf_stub(tmp_path / "short_b.edf",
                    starttime="00.16.00", n_records=600, record_duration=1.0)  # 6 min gap
    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert len(result["large_gaps"]) == 1
    assert result["large_gaps"][0]["gap_seconds"] == 360.0


def test_gaps_fail_large_gap_missing_file(tmp_path):
    # 1-hour gap between two 1-hour files — well past the 10-min cap.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=3600, record_duration=1.0)
    _write_edf_stub(tmp_path / "c.edf",
                    starttime="02.00.00", n_records=3600, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert len(result["large_gaps"]) == 1
    assert result["large_gaps"][0]["prev_file"] == "a.edf"
    assert result["large_gaps"][0]["next_file"] == "c.edf"
    assert result["large_gaps"][0]["gap_seconds"] == 3600.0
    assert result["large_gaps"][0]["threshold_seconds"] == 600.0
    assert any("possibly missing" in msg for msg in result["issues"])
    # 3600s hits the >= 1 h threshold → formatted as hours, not seconds.
    assert any("1.00h" in msg for msg in result["issues"]), result["issues"]


def test_gaps_multi_hour_gap_formats_as_hours(tmp_path):
    # 12-hour gap between two 1-hour files — the raw seconds number
    # (43200s) is hard to eyeball, so the summary switches to hours.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=3600, record_duration=1.0)
    _write_edf_stub(tmp_path / "c.edf",
                    starttime="13.00.00", n_records=3600, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert result["large_gaps"][0]["gap_seconds"] == 12 * 3600.0
    issue = next(m for m in result["issues"] if "Large gap" in m)
    assert "12.00h" in issue
    # Threshold (10 min = 600s) still renders in seconds.
    assert "600.0s" in issue


def test_format_duration_switches_units_at_one_hour():
    from clean_eeg.audit.checks import _format_duration
    assert _format_duration(0.0) == "0.0s"
    assert _format_duration(30.0) == "30.0s"
    assert _format_duration(3599.9) == "3599.9s"
    # Exactly one hour — threshold is inclusive.
    assert _format_duration(3600.0) == "1.00h"
    assert _format_duration(7200.0) == "2.00h"
    assert _format_duration(90000.0) == "25.00h"


def test_gaps_fail_overlap(tmp_path):
    # Second file starts 10s before the first one ends.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=60, record_duration=1.0)
    _write_edf_stub(tmp_path / "b.edf",
                    starttime="00.00.50", n_records=60, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert len(result["overlaps"]) == 1
    assert result["overlaps"][0]["gap_seconds"] == -10.0
    assert any("duplicate/reorder" in msg for msg in result["issues"])


def test_gaps_overlaps_compress_after_first_two(tmp_path):
    """Multi-day recordings often carry the same sub-second boundary
    overlap on every consecutive pair. The check surfaces the first 2
    individually, then collapses the remaining similar-magnitude
    overlaps into a single count-summary line rather than one line per
    consecutive pair (which drowned the summary for full-length
    admissions)."""
    # 12 files, each 60s long, each starting 1s before the previous
    # ended → 11 identical 1s overlaps.
    for i in range(12):
        total_s = i * 59
        hh, rem = divmod(total_s, 3600)
        mm, ss = divmod(rem, 60)
        _write_edf_stub(tmp_path / f"g{i:02d}.edf",
                        starttime=f"{hh:02d}.{mm:02d}.{ss:02d}",
                        n_records=60, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert len(result["overlaps"]) == 11
    overlap_issue_lines = [m for m in result["issues"]
                           if "Overlap of" in m or "consecutive-file overlap" in m]
    individual = [m for m in overlap_issue_lines if "Overlap of" in m]
    compressed = [m for m in overlap_issue_lines if "consecutive-file overlap" in m]
    assert len(individual) == 2, (
        f"expected first 2 overlaps shown individually, got {len(individual)}: "
        f"{individual}"
    )
    assert len(compressed) == 1, (
        f"expected 1 compressed summary line, got {compressed}"
    )
    # 11 total - 2 shown = 9 compressed.
    assert "9 more" in compressed[0], compressed[0]


def test_gaps_overlaps_new_max_breaks_compression(tmp_path):
    """An overlap that exceeds the running max by more than 2s should
    be surfaced individually even after the first-2 budget is spent —
    it signals a genuinely worse anomaly the operator should see."""
    # 8 files, 7 overlap pairs: 6 at 1s + 1 at 10s.
    starts = []
    accum = 0
    for stride in [0, 59, 59, 59, 59, 59, 59, 59, 50]:
        accum += stride
        starts.append(accum)
    for i, total_s in enumerate(starts):
        hh, rem = divmod(total_s, 3600)
        mm, ss = divmod(rem, 60)
        _write_edf_stub(tmp_path / f"g{i:02d}.edf",
                        starttime=f"{hh:02d}.{mm:02d}.{ss:02d}",
                        n_records=60, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    individual = [m for m in result["issues"] if "Overlap of" in m]
    # 2 first-shown (1.0s each) + 1 new-max (10.0s > 1.0s + 2s tolerance)
    # = 3 individually shown.
    assert len(individual) == 3, individual
    # The 10s overlap must be one of the individually-shown lines.
    assert any("10.0s" in m for m in individual), individual


def test_gaps_fail_unparseable_header(tmp_path):
    _write_edf_stub(tmp_path / "bad.edf", startdate="ZZ.ZZ.ZZ")

    result = check_recording_gaps([tmp_path / "bad.edf"])

    assert result["status"] == "fail"
    assert "bad.edf" in result["unparseable_files"]
    assert any("could not parse" in msg for msg in result["issues"])


def test_gaps_fail_no_files():
    result = check_recording_gaps([])

    assert result["status"] == "fail"
    assert result["n_files"] == 0
    assert any("No EDF files" in msg for msg in result["issues"])


def test_gaps_fail_unrepaired_n_records_sentinel(tmp_path):
    # n_records=-1 is the EDF "unknown/streaming" sentinel; the pipeline
    # normalizes it before de-id, so seeing it in a transferred file is
    # itself an integrity red flag. Audit should treat as unparseable.
    _write_edf_stub(tmp_path / "streaming.edf",
                    starttime="00.00.00", n_records=-1, record_duration=1.0)
    result = check_recording_gaps([tmp_path / "streaming.edf"])
    assert result["status"] == "fail"
    assert "streaming.edf" in result["unparseable_files"]


def test_gaps_pass_ignores_sort_order(tmp_path):
    # Files handed to the check in reverse chronological order still
    # produce correct gap analysis because we sort by parsed start time.
    _write_edf_stub(tmp_path / "later.edf",
                    starttime="01.00.00", n_records=3600, record_duration=1.0)
    _write_edf_stub(tmp_path / "earlier.edf",
                    starttime="00.00.00", n_records=3600, record_duration=1.0)
    # Pass in reverse alphabetical order (later first)
    result = check_recording_gaps([tmp_path / "later.edf", tmp_path / "earlier.edf"])
    assert result["status"] == "pass"
    assert result["files_by_start"][0]["file"] == "earlier.edf"
    assert result["files_by_start"][1]["file"] == "later.edf"


# --- windowed signal reader (for notebook EEG plots) -----------------------


def test_read_signal_window_returns_channel_arrays(tmp_path):
    import numpy as np
    n_records, spr, n_signals = 3, 100, 2
    ch0 = np.arange(n_records * spr, dtype="<i2")
    ch1 = np.full(n_records * spr, 42, dtype="<i2")
    _write_edf_with_signals(tmp_path / "a.edf",
                            n_records=n_records, samples_per_record=spr,
                            n_signals=n_signals,
                            channel_samples=[ch0, ch1])
    window = read_signal_window(tmp_path / "a.edf", window_seconds=10.0)
    assert list(window.keys()) == ["EEG0", "EEG1"]
    assert window["EEG0"].tolist() == ch0.tolist()
    assert window["EEG1"].tolist() == ch1.tolist()


def test_read_signal_window_skips_annotation_channel(tmp_path):
    _write_edf_with_annotations(tmp_path / "ann.edf", [(1.0, None, "test")])
    assert read_signal_window(tmp_path / "ann.edf") == {}


def test_read_signal_window_returns_empty_on_broken_header(tmp_path):
    _write_edf_stub(tmp_path / "no_data.edf")  # header stub, no data records
    assert read_signal_window(tmp_path / "no_data.edf") == {}


# --- file-subset selection -------------------------------------------------


def test_select_all_when_n_is_none():
    xs = ["a", "b", "c", "d"]
    assert select_files(xs, n_files=None) == xs


def test_select_all_when_n_exceeds_len():
    xs = ["a", "b", "c"]
    assert select_files(xs, n_files=10) == xs


def test_select_empty_input_returns_empty():
    assert select_files([], n_files=3) == []


def test_select_single_returns_first():
    xs = ["a", "b", "c", "d"]
    assert select_files(xs, n_files=1) == ["a"]


def test_select_two_returns_first_and_last():
    xs = ["a", "b", "c", "d", "e"]
    assert select_files(xs, n_files=2) == ["a", "e"]


def test_select_three_always_includes_endpoints():
    xs = list(range(10))
    picked = select_files(xs, n_files=3, seed=42)
    assert picked[0] == 0
    assert picked[-1] == 9
    assert len(picked) == 3
    assert 0 < picked[1] < 9  # middle drawn from inner indices


def test_select_preserves_input_order():
    xs = list(range(20))
    picked = select_files(xs, n_files=5, seed=42)
    assert picked == sorted(picked)
    assert picked[0] == 0 and picked[-1] == 19


def test_select_seed_is_deterministic():
    xs = list(range(20))
    a = select_files(xs, n_files=6, seed=123)
    b = select_files(xs, n_files=6, seed=123)
    assert a == b


def test_select_different_seeds_differ():
    xs = list(range(20))
    a = select_files(xs, n_files=6, seed=1)
    b = select_files(xs, n_files=6, seed=2)
    # Extremely unlikely to collide with n=20, k=4 middle draws
    assert a != b


def test_select_negative_or_zero_returns_empty():
    xs = ["a", "b", "c"]
    assert select_files(xs, n_files=0) == []
    assert select_files(xs, n_files=-1) == []


# --- log-file surfacing ----------------------------------------------------


def test_log_pass_clean(tmp_path):
    log = tmp_path / "log.out"
    log.write_text("=== clean_eeg log started 2026-07-22 ===\n"
                   "Loading files ...\n"
                   "Done.\n")
    result = check_log_file(log)
    assert result["status"] == "pass"
    assert result["log_present"] is True
    assert result["n_warnings"] == 0
    assert result["n_errors"] == 0
    assert result["n_redactions"] == 0


def test_log_warn_missing(tmp_path):
    # Passing a non-existent path is warn (missing log = missing provenance,
    # not a fatal transfer error).
    result = check_log_file(tmp_path / "log.out")
    assert result["status"] == "warn"
    assert result["log_present"] is False
    assert any("No pipeline" in msg for msg in result["issues"])


def test_log_warn_none_path():
    result = check_log_file(None)
    assert result["status"] == "warn"
    assert result["log_present"] is False


def test_log_warn_on_warnings(tmp_path):
    log = tmp_path / "log.out"
    log.write_text(
        "Loading files ...\n"
        "WARNING: Gap of 3600s between neighboring recordings a.edf, c.edf\n"
        "WARNING: Multiple unique subject names found across EDF files:\n"
    )
    result = check_log_file(log)
    assert result["status"] == "warn"
    assert result["n_warnings"] == 2
    assert result["warnings"][0]["line_number"] == 2
    assert "Gap of 3600s" in result["warnings"][0]["text"]
    assert any("2 WARNING" in msg for msg in result["issues"])


def test_log_fail_on_errors(tmp_path):
    log = tmp_path / "log.out"
    log.write_text(
        "WARNING: some warning\n"
        "ERROR: signal integrity audit FAILED on file X\n"
    )
    result = check_log_file(log)
    assert result["status"] == "fail"
    assert result["n_errors"] == 1
    assert any("additional ERROR" in msg for msg in result["issues"])


def test_log_fail_when_pipeline_skipped_file_via_load_error(tmp_path):
    """Positive: 'ERROR: Failed to load EDF file NA3621K6.edf: ...'
    gets parsed into ``failed_deid_files`` with the bare filename.
    """
    log = tmp_path / "log.out"
    log.write_text(
        "Loading files...\n"
        "ERROR: Failed to load EDF file NA3621K6.edf: OSError: ...\n"
        "ERROR: Failed to de-identify EDF file NB0102XX.edf: KeyError('startdate')\n"
        "Done.\n"
    )
    result = check_log_file(log)
    assert result["status"] == "fail"
    assert result["n_failed_deid_files"] == 2
    names = [f["filename"] for f in result["failed_deid_files"]]
    assert names == ["NA3621K6.edf", "NB0102XX.edf"]
    assert any("failed pipeline de-identification" in msg
               and "NA3621K6.edf" in msg
               for msg in result["issues"])


def test_log_pass_when_no_failed_deid_lines(tmp_path):
    """Negative: a log with no 'Failed to load/de-identify EDF file'
    lines should NOT populate failed_deid_files.
    """
    log = tmp_path / "log.out"
    log.write_text("Loading files ...\nDone.\n")
    result = check_log_file(log)
    assert result["n_failed_deid_files"] == 0
    assert result["failed_deid_files"] == []


def test_log_failed_deid_captures_error_message(tmp_path):
    """The exception the pipeline prints right after the ERROR line
    should be captured into failed_deid_files[i]['error_message'] so
    the audit's header-dump section can show it inline (spares the
    operator a trip back to log.out for the actual failure reason).
    """
    log = tmp_path / "log.out"
    log.write_text(
        "Loading files ...\n"
        "ERROR: Failed to load EDF file NA3621K6.edf:\n"
        "\n"
        "OSError: the file is not EDF(+) or BDF(+) compliant (Filesize)\n"
        "\n"
        "Stack trace (for the data team):\n"
        "Traceback (most recent call last):\n"
        "  File 'x.py', line 1, in <module>\n"
        "    raise OSError(...)\n"
        "\n"
        "Check if the file is corrupted. Skipping this file...\n"
    )
    result = check_log_file(log)
    assert result["n_failed_deid_files"] == 1
    entry = result["failed_deid_files"][0]
    assert entry["filename"] == "NA3621K6.edf"
    assert "OSError" in entry["error_message"]
    assert "not EDF" in entry["error_message"]
    # Negative: the stack trace itself should NOT bleed into the message.
    assert "Traceback" not in entry["error_message"]
    assert "raise OSError" not in entry["error_message"]


def test_log_failed_deid_matches_when_concatenated_to_tqdm_progress(tmp_path):
    """Regression: on some terminals / tee configurations, the pipeline's
    tqdm progress bar writes its update without a trailing newline
    (\\r-based overwrite), and the very next print() from the pipeline
    lands ON THE SAME LINE. Real captured example from a user report:

        Loading EDF meta-data...:  13%|xxxx| 6/47 [00:06<...]ERROR: Failed to load EDF file DA1564PX.edf:

    A ^ERROR: anchor would miss this. The parser must search
    ANYWHERE in the line.
    """
    log = tmp_path / "log.out"
    log.write_text(
        "Loading EDF meta-data...:   0%|                     | 0/47 [00:00<?, ?it/s]\n"
        "Loading EDF meta-data...:  13%|xxxxx        | 6/47 [00:06<00:40,  1.02it/s]"
        "ERROR: Failed to load EDF file DA1564PX.edf:\n"
        "\n"
        "OSError: the file is not EDF(+) or BDF(+) compliant "
        "(it contains format errors)\n"
        "\n"
        "Stack trace (for the data team):\n"
        "Traceback ...\n"
    )
    result = check_log_file(log)
    assert result["n_failed_deid_files"] == 1, (
        f"expected 1 failed_deid_file, got errors={result['errors']!r}, "
        f"failed_deid_files={result['failed_deid_files']!r}"
    )
    entry = result["failed_deid_files"][0]
    assert entry["filename"] == "DA1564PX.edf"
    assert "not EDF" in entry["error_message"]


def test_log_failed_deid_survives_tqdm_carriage_returns(tmp_path):
    """Regression: tqdm writes progress-bar updates with \r-terminated
    lines when stderr isn't a TTY (headless / SSH), and the pipeline's
    tee captures them all into log.out. A naive readline() would see
    'Loading files...\\rERROR: Failed to load X:' as ONE line and the
    ^ERROR: anchor would fail. The parser must split on \\r too.
    """
    log = tmp_path / "log.out"
    log.write_bytes(
        b"Loading files...\r50%|xxxxx     |\r"
        b"ERROR: Failed to load EDF file NA3621K6.edf: OSError: cannot open\n"
        b"Done.\n"
    )
    result = check_log_file(log)
    assert result["n_failed_deid_files"] == 1
    assert result["failed_deid_files"][0]["filename"] == "NA3621K6.edf"


def test_log_warn_on_redactions(tmp_path):
    log = tmp_path / "log.out"
    log.write_text(
        "Loading ...\n"
        'Subject protected health information detected in EDF annotation; '
        'redacted value: "seizure noted at bedside by <REDACTED>". '
        'Alert the data analysis team.\n'
        'Subject protected health information detected in EDF patientname; '
        'redacted value: "X X X". Alert the data analysis team.\n'
    )
    result = check_log_file(log)
    assert result["status"] == "warn"
    assert result["n_redactions"] == 2
    assert result["redactions"][0]["field"] == "annotation"
    assert "seizure noted" in result["redactions"][0]["redacted_value"]
    assert result["redactions"][1]["field"] == "patientname"
    assert any("annotation redaction" in msg for msg in result["issues"])


def test_log_fail_beats_warn(tmp_path):
    # Log has both WARNING and ERROR — ERROR wins the overall status.
    log = tmp_path / "log.out"
    log.write_text(
        "WARNING: something\n"
        "ERROR: something worse\n"
        'Subject protected health information detected in EDF annotation; '
        'redacted value: "X". Alert the data analysis team.\n'
    )
    result = check_log_file(log)
    assert result["status"] == "fail"
    assert result["n_warnings"] == 1
    assert result["n_errors"] == 1
    assert result["n_redactions"] == 1


# --- annotation scan (pure-Python matching logic) --------------------------


_TEST_NAMES = {"john", "smith", "sarah", "o'connor", "jean-luc"}


def test_scan_pass_no_annotations():
    per_ann, inv, _stats = scan_annotation_texts([], _TEST_NAMES)
    assert per_ann == []
    assert inv == {}


def test_scan_pass_no_name_tokens():
    anns = [{"onset": 1.0, "text": "seizure onset"}]
    per_ann, inv, _stats = scan_annotation_texts(anns, _TEST_NAMES)
    assert per_ann == []
    assert inv == {}


def test_scan_fail_dictionary_hit():
    anns = [{"onset": 1.0, "text": "patient John reports headache"}]
    per_ann, inv, _stats = scan_annotation_texts(anns, _TEST_NAMES)
    assert len(per_ann) == 1
    assert per_ann[0]["matched_tokens"] == ["john"]
    assert list(inv) == ["john"]


def test_scan_case_insensitive_match():
    anns = [{"onset": 1.0, "text": "SMITH visit"},
            {"onset": 2.0, "text": "Smith visit"},
            {"onset": 3.0, "text": "smith visit"}]
    per_ann, inv, _stats = scan_annotation_texts(anns, _TEST_NAMES)
    assert len(per_ann) == 3
    assert inv["smith"] and len(inv["smith"]) == 3


def test_scan_multiple_hits_in_one_annotation():
    anns = [{"onset": 1.0, "text": "John Smith saw Sarah"}]
    per_ann, inv, _stats = scan_annotation_texts(anns, _TEST_NAMES)
    assert len(per_ann) == 1
    assert set(per_ann[0]["matched_tokens"]) == {"john", "smith", "sarah"}
    assert set(inv) == {"john", "smith", "sarah"}


def test_scan_pass_whitelisted_token_ignored():
    anns = [{"onset": 1.0, "text": "seizure noted by John"}]
    per_ann, inv, _stats = scan_annotation_texts(anns, _TEST_NAMES,
                                          vocab_whitelist={"John"})
    assert per_ann == []
    assert inv == {}


def test_scan_handles_hyphenated_and_apostrophe_names():
    anns = [{"onset": 1.0, "text": "seen by Jean-Luc"},
            {"onset": 2.0, "text": "notes from O'Connor"}]
    per_ann, inv, _stats = scan_annotation_texts(anns, _TEST_NAMES)
    assert len(per_ann) == 2
    assert set(inv) == {"jean-luc", "o'connor"}


def test_scan_ignores_numeric_and_punctuation():
    # +1.5s and (12:03) contain no letter tokens, so no false matches.
    anns = [{"onset": 1.0, "text": "+1.5s (12:03)"}]
    per_ann, _, _stats = scan_annotation_texts(anns, _TEST_NAMES)
    assert per_ann == []


def test_scan_skips_short_annotations():
    """Annotations under MIN_ANNOTATION_LENGTH_TO_SCAN (6) are not
    scanned — short status codes like 'PT', 'OFF', 'EEG', 'RN' are
    almost never PHI-carrying. Would-be name hits inside short
    annotations are silenced."""
    anns = [
        {"onset": 1.0, "text": "PT"},           # 2 chars → skip
        {"onset": 2.0, "text": "John"},          # 4 chars → skip, even though
                                                  #    'john' IS in the name dict
        {"onset": 3.0, "text": "AWAKE"},         # 5 chars → skip
        {"onset": 4.0, "text": "JOHN IN"},       # 7 chars → scan (matches 'john')
    ]
    per_ann, inv, stats = scan_annotation_texts(anns, _TEST_NAMES)
    assert stats["n_skipped_short"] == 3
    assert stats["n_scanned"] == 1
    # Only the JOHN IN annotation was scanned + flagged.
    assert len(per_ann) == 1
    assert per_ann[0]["text"] == "JOHN IN"
    assert list(inv) == ["john"]


def test_scan_boilerplate_whitelist_suppresses_matching_annotation():
    """A per-site boilerplate regex that fullmatches an annotation
    prevents that annotation from being scanned at all — so any name
    tokens inside are silenced along with the boilerplate."""
    from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
    import re as _re
    wl = BoilerplateWhitelist(
        shared=[],
        per_site={"A": [_re.compile(r"PAT REF EEG")]},
    )
    anns = [
        {"onset": 1.0, "text": "PAT REF EEG"},       # matches → skip
        {"onset": 2.0, "text": "seizure by Sarah"},  # no match → scan
    ]
    per_ann, inv, stats = scan_annotation_texts(
        anns, _TEST_NAMES, boilerplate_whitelist=wl, site_code="A")
    assert stats["n_skipped_boilerplate"] == 1
    assert stats["n_scanned"] == 1
    assert len(per_ann) == 1
    assert per_ann[0]["text"] == "seizure by Sarah"
    assert "sarah" in inv


def test_scan_boilerplate_whitelist_fullmatch_semantics():
    """Boilerplate matching uses fullmatch — a permissive pattern like
    'CAL IN' only silences the exact annotation, NOT longer annotations
    that happen to contain it. Otherwise 'CAL IN CAROL AT 3PM' would
    silence the Carol PHI."""
    from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
    import re as _re
    wl = BoilerplateWhitelist(
        shared=[],
        per_site={"A": [_re.compile(r"CAL IN")]},
    )
    _names = _TEST_NAMES | {"carol"}
    anns = [
        {"onset": 1.0, "text": "CAL IN"},                # fullmatch → skip
        {"onset": 2.0, "text": "CAL IN CAROL AT 3PM"},   # substr only → SCAN
    ]
    per_ann, inv, stats = scan_annotation_texts(
        anns, _names, boilerplate_whitelist=wl, site_code="A")
    assert stats["n_skipped_boilerplate"] == 1
    assert stats["n_scanned"] == 1
    # The longer annotation was scanned and Carol was caught.
    assert len(per_ann) == 1
    assert "carol" in per_ann[0]["matched_tokens"]


def test_scan_boilerplate_whitelist_wrong_site_does_not_apply():
    """A per-site pattern only fires for its own site. Otherwise a
    CUDA-specific phrase could silence PHI at UTHSCSA."""
    from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
    import re as _re
    wl = BoilerplateWhitelist(
        shared=[],
        per_site={"A": [_re.compile(r"CAL IN")]},
    )
    anns = [{"onset": 1.0, "text": "CAL IN"}]
    per_ann, _, stats = scan_annotation_texts(
        anns, _TEST_NAMES, boilerplate_whitelist=wl, site_code="S")
    assert stats["n_skipped_boilerplate"] == 0
    assert stats["n_scanned"] == 1  # not silenced at site S


# --- annotation extraction + end-to-end check ------------------------------


def test_extract_annotations_reads_events(tmp_path):
    _write_edf_with_annotations(tmp_path / "ann.edf", [
        (1.5, None, "seizure onset"),
        (10.0, 2.0, "John visited"),
    ])
    anns = extract_annotations(tmp_path / "ann.edf")
    assert len(anns) == 2
    assert anns[0] == {"onset": 1.5, "duration": None, "text": "seizure onset"}
    assert anns[1] == {"onset": 10.0, "duration": 2.0, "text": "John visited"}


def test_extract_annotations_returns_empty_when_no_annotation_channel(tmp_path):
    # A file with only an EEG channel (no "EDF Annotations" signal).
    _write_edf_with_signals(tmp_path / "no_ann.edf",
                            n_records=5, samples_per_record=100)
    assert extract_annotations(tmp_path / "no_ann.edf") == []


def test_phi_scan_pass_no_matches(tmp_path):
    _write_edf_with_annotations(tmp_path / "clean.edf", [
        (1.0, None, "seizure onset"),
        (2.0, None, "focal activity"),
    ])
    result = check_annotation_phi_scan([tmp_path / "clean.edf"],
                                       name_dictionary=_TEST_NAMES)
    assert result["status"] == "pass"
    assert result["n_annotations_scanned"] == 2
    assert result["n_matches"] == 0


def test_phi_scan_fail_name_hit(tmp_path):
    _write_edf_with_annotations(tmp_path / "leaked.edf", [
        (5.0, None, "seizure — nurse Sarah at bedside"),
    ])
    result = check_annotation_phi_scan([tmp_path / "leaked.edf"],
                                       name_dictionary=_TEST_NAMES)
    assert result["status"] == "fail"
    assert result["n_matches"] == 1
    assert "sarah" in result["matched_tokens"]
    assert result["matches_by_file"]["leaked.edf"][0]["matched_tokens"] == ["sarah"]
    assert any("sarah" in msg for msg in result["issues"])


def test_phi_scan_whitelist_suppresses_hit(tmp_path):
    # "John" is in the dictionary but has been added to the operator's
    # annotation-vocab whitelist (perhaps as a spurious frequent term).
    _write_edf_with_annotations(tmp_path / "clean.edf", [
        (1.0, None, "reviewed by John"),
    ])
    result = check_annotation_phi_scan([tmp_path / "clean.edf"],
                                       name_dictionary=_TEST_NAMES,
                                       vocab_whitelist={"john"})
    assert result["status"] == "pass"
    assert result["n_vocab_whitelist_tokens"] == 1


def test_phi_scan_fail_empty_input():
    result = check_annotation_phi_scan([], name_dictionary=_TEST_NAMES)
    assert result["status"] == "fail"
    assert result["n_files"] == 0


def test_phi_scan_across_multiple_files(tmp_path):
    _write_edf_with_annotations(tmp_path / "a.edf",
                                [(1.0, None, "seizure")])
    _write_edf_with_annotations(tmp_path / "b.edf",
                                [(2.0, None, "greeted Smith")])
    result = check_annotation_phi_scan(sorted(tmp_path.glob("*.edf")),
                                       name_dictionary=_TEST_NAMES)
    assert result["status"] == "fail"
    assert set(result["matches_by_file"]) == {"b.edf"}
    assert result["matched_tokens"]["smith"][0]["file"] == "b.edf"


# --- end-to-end audit orchestrator -----------------------------------------


def _build_clean_subject(tmp_path: Path,
                         *,
                         subject_code: str = "R1755J",
                         annotations: list | None = None) -> Path:
    """Build a synthetic cleaned-subject directory with:
      - 2 recording EDFs with real signal data (100 samples/sec, 60s)
      - 2 matching annotation stubs (in-place mode)
      - a log.out with a WARNING (exercises the log check)
    """
    import numpy as np
    subject_dir = tmp_path / subject_code
    subject_dir.mkdir()
    pid = f"{subject_code} X 01-JAN-1900 unknown unknown"

    # Filenames must match the pipeline's rename convention
    # ({stem}_{subject_code}_{MM.DD__HH.MM.SS}.edf) so the new
    # check_filename_convention pass on this "clean" fixture.
    for i, (name_stem, starttime, stamp) in enumerate([
        ("a", "00.00.00", "01.01__00.00.00"),
        ("b", "00.01.00", "01.01__00.01.00"),
    ]):
        clean_name = f"{name_stem}_{subject_code}_{stamp}.edf"
        sig = (np.sin(np.linspace(0, 20, 6000)) * 1000).astype("<i2")
        _write_edf_with_signals(subject_dir / clean_name,
                                n_records=60, samples_per_record=100,
                                starttime=starttime,
                                channel_samples=[sig])
        # Rewrite patient_id to include the subject code (helper defaults
        # to R1755J so this is just belt-and-braces for varied fixtures).
        _patch_patient_id(subject_dir / clean_name, pid)

        stub_name = f"{name_stem}_{subject_code}_{stamp}_annotations.edf"
        _write_edf_with_annotations(subject_dir / stub_name,
                                    annotations or [(0.5, None, "seizure")])
        _patch_patient_id(subject_dir / stub_name, pid)

    (subject_dir / "log.out").write_text(
        "=== clean_eeg log started 2026-07-22 ===\n"
        "Loading files ...\n"
    )
    return subject_dir


def _patch_patient_id(edf_path: Path, patient_id: str) -> None:
    data = bytearray(edf_path.read_bytes())
    data[8:88] = patient_id.encode("ascii").ljust(80, b" ")
    edf_path.write_bytes(bytes(data))


def test_e2e_audit_pass_on_clean_subject(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    audit = audit_subject(subject_dir, name_dictionary={"nonexistent"})

    assert (subject_dir / AUDIT_JSON_FILENAME).exists()
    assert audit["subject_code"] == "R1755J"
    assert audit["mode"] == "full"
    expected_checks = {
        "subject_code_consistency", "filename_convention", "header_phi_residue",
        "recording_gaps", "byte_geometry", "annotation_pairing",
        "signal_header_uniformity", "annotation_phi_scan",
        "transfer_integrity", "log_file",
    }
    assert set(audit["checks"]) == expected_checks
    # log has no WARNING/ERROR/redactions → pass; everything else pass.
    non_passing = {n: r["status"] for n, r in audit["checks"].items()
                   if r["status"] != "pass"}
    assert non_passing == {}, non_passing
    assert audit["overall_status"] == "pass"


def test_e2e_audit_fail_when_annotation_contains_name(tmp_path):
    subject_dir = _build_clean_subject(
        tmp_path, annotations=[(1.0, None, "seen by Sarah")])
    audit = audit_subject(subject_dir, name_dictionary={"sarah"})
    assert audit["overall_status"] == "fail"
    assert audit["checks"]["annotation_phi_scan"]["status"] == "fail"
    assert "sarah" in audit["checks"]["annotation_phi_scan"]["matched_tokens"]


def test_e2e_audit_skips_second_run_but_rechecks_hashes(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    first = audit_subject(subject_dir, name_dictionary={"nonexistent"})
    assert first.get("skipped") is not True

    second = audit_subject(subject_dir, name_dictionary={"nonexistent"})
    assert second["skipped"] is True
    assert "rechecked_at" in second
    # Hash check still ran and passed (no bit rot on disk).
    assert second["checks"]["transfer_integrity"]["status"] == "pass"


def test_e2e_audit_force_reruns_all_checks(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    audit_subject(subject_dir, name_dictionary={"nonexistent"})

    forced = audit_subject(subject_dir, name_dictionary={"nonexistent"}, force=True)
    assert forced.get("skipped") is not True
    assert "log_file" in forced["checks"]  # full re-run, not just hashes


def test_e2e_audit_annotation_only_skips_other_checks(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    audit = audit_subject(subject_dir, name_dictionary={"nonexistent"},
                          annotation_only=True)
    assert audit["mode"] == "annotation_only"
    assert set(audit["checks"]) == {"transfer_integrity", "annotation_phi_scan"}


def test_e2e_audit_detects_bit_rot_on_second_run(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    audit_subject(subject_dir, name_dictionary={"nonexistent"})

    # Modify a file post-audit — the always-on hash check must catch it.
    target = next(subject_dir.glob("a_*.edf"))
    with open(target, "r+b") as f:
        f.seek(255)
        f.write(b"\xff")

    second = audit_subject(subject_dir, name_dictionary={"nonexistent"})
    assert second["checks"]["transfer_integrity"]["status"] == "fail"


def test_e2e_audit_skip_hashes_omits_transfer_integrity(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    audit = audit_subject(subject_dir, name_dictionary={"nonexistent"},
                          skip_hashes=True)
    assert "transfer_integrity" not in audit["checks"]


# --- progress callback + per-check timings --------------------------------


def test_progress_callback_fires_start_and_end_for_every_check(tmp_path):
    """Every check that runs must emit exactly one 'start' event followed
    by one 'end' event with a status + elapsed time. This is what
    downstream CLIs (audit-subject-eeg's streaming printer) rely on."""
    subject_dir = _build_clean_subject(tmp_path)
    events: list[dict] = []

    def _cb(*, name, phase, elapsed_s=None, status=None):
        events.append({"name": name, "phase": phase,
                       "elapsed_s": elapsed_s, "status": status})

    audit = audit_subject(subject_dir, name_dictionary={"nonexistent"},
                          progress=_cb)

    import pytest as _pytest
    starts = [e for e in events if e["phase"] == "start"]
    ends = [e for e in events if e["phase"] == "end"]
    ran_checks = list(audit["checks"].keys())

    # Same set of checks, same order, one start per check followed by
    # one end per check.
    assert [e["name"] for e in starts] == ran_checks
    assert [e["name"] for e in ends] == ran_checks

    # Each end carries the check's own status + a non-negative elapsed.
    for end in ends:
        assert end["status"] in ("pass", "warn", "fail")
        assert end["status"] == audit["checks"][end["name"]]["status"]
        assert end["elapsed_s"] is not None and end["elapsed_s"] >= 0

    # Timing map matches the checks + roughly aligns with the callback
    # (allowing loose bounds for measurement jitter).
    assert set(audit["_timings_by_check_s"]) == set(ran_checks)
    for end in ends:
        recorded = audit["_timings_by_check_s"][end["name"]]
        assert recorded == _pytest.approx(end["elapsed_s"], abs=0.05)


def test_progress_callback_optional_when_omitted(tmp_path):
    """audit_subject must work when no progress callback is passed —
    the callback is a pure UX addition and cannot be a required arg."""
    subject_dir = _build_clean_subject(tmp_path)
    # Should not raise
    audit = audit_subject(subject_dir, name_dictionary={"nonexistent"})
    assert "_timings_by_check_s" in audit


# --- name-dictionary in-process memoization ------------------------------


# --- interruption sentinel ------------------------------------------------


def test_sentinel_removed_after_successful_audit(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    audit_subject(subject_dir, name_dictionary={"nonexistent"})
    assert not (subject_dir / IN_PROGRESS_FILENAME).exists()


def test_sentinel_persists_when_a_check_raises(tmp_path, monkeypatch):
    """Simulate a Ctrl-C mid-audit: the sentinel must remain so the
    next invocation can detect the interruption."""
    subject_dir = _build_clean_subject(tmp_path)

    # Any check will do — patch one that runs late so the sentinel is
    # definitely already written.
    import clean_eeg.audit.subject as subject_mod

    def _boom(*_args, **_kwargs):
        raise KeyboardInterrupt("simulated Ctrl-C")

    monkeypatch.setattr(subject_mod, "check_annotation_phi_scan", _boom)
    import pytest as _pytest
    with _pytest.raises(KeyboardInterrupt):
        audit_subject(subject_dir, name_dictionary={"nonexistent"})

    sentinel = subject_dir / IN_PROGRESS_FILENAME
    assert sentinel.exists(), "sentinel must survive an interrupted run"
    # And no completed audit JSON exists — the audit did not finish.
    assert not (subject_dir / AUDIT_JSON_FILENAME).exists()


def test_re_running_after_interruption_raises(tmp_path, monkeypatch):
    subject_dir = _build_clean_subject(tmp_path)
    import clean_eeg.audit.subject as subject_mod
    monkeypatch.setattr(subject_mod, "check_annotation_phi_scan",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        audit_subject(subject_dir, name_dictionary={"nonexistent"})

    # Second invocation must refuse loudly rather than silently restart.
    with _pytest.raises(AuditInterruptedError) as exc_info:
        audit_subject(subject_dir, name_dictionary={"nonexistent"})

    assert exc_info.value.sentinel_path == subject_dir / IN_PROGRESS_FILENAME
    # Sentinel content should give the operator timestamp + host + pid
    # for cluster-log correlation.
    assert exc_info.value.started_at is not None
    assert exc_info.value.hostname is not None
    assert exc_info.value.pid is not None


def test_force_clears_sentinel_and_completes_audit(tmp_path, monkeypatch):
    subject_dir = _build_clean_subject(tmp_path)
    import clean_eeg.audit.subject as subject_mod

    # First run: interrupted mid-audit
    monkeypatch.setattr(subject_mod, "check_annotation_phi_scan",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        audit_subject(subject_dir, name_dictionary={"nonexistent"})
    assert (subject_dir / IN_PROGRESS_FILENAME).exists()

    # Unpatch: subsequent calls run normally
    monkeypatch.undo()

    # force=True clears the stale sentinel and completes the audit
    audit = audit_subject(subject_dir, name_dictionary={"nonexistent"},
                          force=True)
    assert audit["overall_status"] in ("pass", "warn")
    assert (subject_dir / AUDIT_JSON_FILENAME).exists()
    assert not (subject_dir / IN_PROGRESS_FILENAME).exists()


def test_sentinel_records_start_metadata(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    import clean_eeg.audit.subject as subject_mod
    import pytest as _pytest

    # Interrupt before completion so we can read the sentinel
    original = subject_mod.check_annotation_phi_scan
    def _boom(*a, **kw):
        raise RuntimeError("boom")
    subject_mod.check_annotation_phi_scan = _boom
    try:
        with _pytest.raises(RuntimeError):
            audit_subject(subject_dir, name_dictionary={"nonexistent"})
    finally:
        subject_mod.check_annotation_phi_scan = original

    import json as _json
    meta = _json.loads((subject_dir / IN_PROGRESS_FILENAME).read_text())
    assert "started_at" in meta
    assert "hostname" in meta
    assert "pid" in meta
    assert isinstance(meta["pid"], int)


def test_corrupted_sentinel_still_signals_interruption(tmp_path):
    """A malformed sentinel (e.g., disk corruption, partial write) must
    still block silent restart. The error just carries less metadata."""
    subject_dir = _build_clean_subject(tmp_path)
    # Complete one successful audit so edf_audit.json exists (baseline)
    audit_subject(subject_dir, name_dictionary={"nonexistent"})
    # Now plant a garbage sentinel by hand — simulates a crashed run
    # that couldn't finish writing.
    (subject_dir / IN_PROGRESS_FILENAME).write_text("not json {")

    import pytest as _pytest
    with _pytest.raises(AuditInterruptedError) as exc_info:
        audit_subject(subject_dir, name_dictionary={"nonexistent"})
    # Metadata gracefully missing rather than crashing on parse
    assert exc_info.value.started_at is None
    assert exc_info.value.hostname is None


def test_load_us_name_dictionary_is_memoized_in_process():
    """Repeated calls in the same process must return the identical
    frozenset object, not a fresh copy loaded from disk each time.
    This is what makes --parent mode fast across many subjects."""
    import pytest as _pytest
    from clean_eeg.audit.name_dictionary import load_us_name_dictionary
    from clean_eeg.paths import DATA_DIR

    cache_pkl = DATA_DIR / "name_dictionary_cache" / "US.pkl"
    if not cache_pkl.exists():
        _pytest.skip(f"US name-dictionary cache not present at {cache_pkl}")

    a = load_us_name_dictionary(("US",))
    b = load_us_name_dictionary(("US",))
    # `is` — same object → memoization is in effect (not just same content).
    assert a is b


def test_e2e_output_dir_isolates_audit_outputs(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    out_dir = tmp_path / "elsewhere"
    audit = audit_subject(subject_dir, output_dir=out_dir,
                          name_dictionary={"nonexistent"})
    # JSON lands in output_dir, NOT in subject_dir — avoids polluting fixtures.
    assert (out_dir / AUDIT_JSON_FILENAME).exists()
    assert not (subject_dir / AUDIT_JSON_FILENAME).exists()
    assert audit["output_dir"] == str(out_dir)
    assert audit["subject_dir"] == str(subject_dir)


def test_e2e_output_dir_skip_reads_prior_manifest_from_output_dir(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    out_dir = tmp_path / "elsewhere"
    audit_subject(subject_dir, output_dir=out_dir,
                  name_dictionary={"nonexistent"})

    # Second run should skip based on the prior JSON in the ISOLATED output_dir,
    # not look for one in subject_dir.
    second = audit_subject(subject_dir, output_dir=out_dir,
                           name_dictionary={"nonexistent"})
    assert second.get("skipped") is True
    assert second["checks"]["transfer_integrity"]["status"] == "pass"


def test_render_audit_notebook_html_excludes_code_cells(tmp_path):
    # End-to-end: run the audit, render the notebook + HTML, and
    # verify the HTML has NO code-cell content (only outputs).
    # Requires jupyter kernel; skip if the socket bind is sandbox-blocked.
    import socket
    try:
        s = socket.socket(); s.bind(("127.0.0.1", 0)); s.close()
    except PermissionError:
        import pytest
        pytest.skip("socket bind blocked by sandbox")

    from clean_eeg.audit.notebook import render_audit_notebook
    subject_dir = _build_clean_subject(tmp_path)
    audit_subject(subject_dir, name_dictionary={"nonexistent"})
    ipynb_path, html_path = render_audit_notebook(subject_dir)
    html = html_path.read_text()
    # Uniquely identifying strings from generated code cells:
    for marker in ("AUDIT_JSON_PATH", "import matplotlib", "read_signal_window"):
        assert marker not in html, f"code content leaked into HTML: {marker!r}"


def test_build_audit_notebook_bakes_plot_params(tmp_path):
    nb = build_audit_notebook(tmp_path, tmp_path / "edf_audit.json",
                              n_channel_plot=7, n_files_plot=2,
                              plot_seconds=3.5)
    joined = "\n".join(c["source"] for c in nb["cells"])
    assert "N_CHANNEL_PLOT = 7" in joined
    assert "N_FILES_PLOT = 2" in joined
    assert "PLOT_SECONDS = 3.5" in joined


def test_cli_looks_like_boilerplate():
    from clean_eeg.audit.cli import _looks_like_boilerplate
    # Positive: real annotation content is kept.
    assert not _looks_like_boilerplate("seizure onset")
    assert not _looks_like_boilerplate("SEGMENT 1")
    assert not _looks_like_boilerplate("XY")
    # Negative: timekeeping-shaped / trivial strings are filtered.
    assert _looks_like_boilerplate("")
    assert _looks_like_boilerplate("   ")
    assert _looks_like_boilerplate("+1.5")
    assert _looks_like_boilerplate("-12.5")
    assert _looks_like_boilerplate("1234")
    assert _looks_like_boilerplate("X")  # single char


def test_build_audit_notebook_has_expected_cells(tmp_path):
    nb = build_audit_notebook(tmp_path, tmp_path / "edf_audit.json")
    cell_sources = [c["source"] for c in nb["cells"]]
    kinds = [c["cell_type"] for c in nb["cells"]]
    # Alternating markdown headers + code cells; check structure.
    assert kinds == [
        "markdown", "code",  # title + load_audit
        "markdown", "code",  # summary heading + counts
        "markdown", "code",  # per-check issues heading + code
        "markdown", "code",  # name-dictionary matches
        "markdown", "code",  # pipeline annotation redactions
        "markdown", "code",  # eeg snippets
    ]
    joined = "\n".join(cell_sources)
    assert "SUBJECT_DIR" in joined
    assert "annotation_phi_scan" in joined
    assert "log_file" in joined and "redactions" in joined  # ann-redaction cell
    assert "read_signal_window" in joined
    assert "matplotlib" in joined
    assert nb["metadata"]["kernelspec"]["name"] == "python3"


def test_cli_banner_prints_absolute_paths_for_flagged_files(tmp_path, capsys):
    """Regression: the critical-findings banner (top+bottom) should print
    absolute paths to the flagged files so operators can copy-paste the
    output into a shell without reconstructing the parent directory.
    Applies to all three critical categories: failed-deid, unrenamed,
    off-year recording_id.
    """
    from clean_eeg.audit.cli import _critical_finding_lines
    subject_dir = tmp_path / "R1755J"
    subject_dir.mkdir()
    audit = {
        "subject_dir": str(subject_dir),
        "checks": {
            "log_file": {"failed_deid_files": [
                {"filename": "SKIPPED.edf", "line_number": 10, "text": "..."},
            ]},
            "filename_convention": {"unrenamed_files": ["UNRENAMED.edf"]},
            "header_phi_residue": {
                "expected_year_range": [1985, 1987],
                "recording_id_years_by_file": {
                    "OFFYEAR.edf": 2024,
                    "ok.edf": 1985,
                },
            },
        },
    }
    lines = _critical_finding_lines(audit)
    joined = "\n".join(lines)
    # Positive: every flagged filename is prefixed with the resolved
    # subject_dir path.
    assert str(subject_dir.resolve() / "SKIPPED.edf") in joined
    assert str(subject_dir.resolve() / "UNRENAMED.edf") in joined
    assert str(subject_dir.resolve() / "OFFYEAR.edf") in joined
    # Negative: an in-range file should not appear.
    assert "ok.edf" not in joined


def test_cli_always_prints_annotation_redactions(capsys):
    from clean_eeg.audit.cli import _always_print_warnings
    audit = {"checks": {
        "annotation_phi_scan": {"matched_tokens": {}},
        "log_file": {"redactions": [
            {"line_number": 42, "field": "annotation",
             "redacted_value": "seen by <REDACTED>"},
            {"line_number": 43, "field": "patientname",  # NOT annotation
             "redacted_value": "X X X"},
            {"line_number": 44, "field": "annotation",
             "redacted_value": "noted by <REDACTED>"},
        ]},
    }}
    _always_print_warnings(audit)
    out = capsys.readouterr().out
    # Positive: both annotation redactions flagged.
    assert "Pipeline redacted 2 annotation" in out
    assert "'seen by <REDACTED>'" in out
    assert "'noted by <REDACTED>'" in out
    assert "log line 42" in out and "log line 44" in out
    # Negative: the patientname redaction (field != 'annotation') is NOT
    # in the annotation-redactions block (patientname is a header field,
    # not annotation content — different auditor concern).
    assert "log line 43" not in out
    assert "'X X X'" not in out


def test_cli_annotation_redaction_block_absent_when_none(capsys):
    from clean_eeg.audit.cli import _always_print_warnings
    audit = {"checks": {
        "annotation_phi_scan": {"matched_tokens": {}},
        "log_file": {"redactions": []},
    }}
    _always_print_warnings(audit)
    out = capsys.readouterr().out
    # No annotation redactions → no block at all (keeps output tight).
    assert "Pipeline redacted" not in out


# --- annotation stub pairing -----------------------------------------------


def test_pairing_pass_inline_mode(tmp_path):
    # Rewrite mode: annotations embedded in main EDF, no stubs at all.
    _write_edf_stub(tmp_path / "a.edf")
    _write_edf_stub(tmp_path / "b.edf")
    result = check_annotation_pairing(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "pass"
    assert result["mode"] == "inline"
    assert result["n_recordings"] == 2
    assert result["n_stubs"] == 0
    assert result["paired"] == []


def test_pairing_pass_all_paired(tmp_path):
    for base in ("a", "b", "c"):
        _write_edf_stub(tmp_path / f"{base}.edf")
        _write_edf_stub(tmp_path / f"{base}_annotations.edf")
    result = check_annotation_pairing(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "pass"
    assert result["mode"] == "stub_pair"
    assert result["n_recordings"] == 3
    assert result["n_stubs"] == 3
    assert {tuple(sorted(p.values())) for p in result["paired"]} == {
        ("a.edf", "a_annotations.edf"),
        ("b.edf", "b_annotations.edf"),
        ("c.edf", "c_annotations.edf"),
    }


def test_pairing_fail_orphan_recording(tmp_path):
    _write_edf_stub(tmp_path / "a.edf")
    _write_edf_stub(tmp_path / "a_annotations.edf")
    _write_edf_stub(tmp_path / "b.edf")  # no stub sibling
    result = check_annotation_pairing(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["orphan_recordings"] == ["b.edf"]
    assert result["orphan_stubs"] == []
    assert any("b.edf" in msg and "no paired" in msg for msg in result["issues"])


def test_pairing_fail_orphan_stub(tmp_path):
    _write_edf_stub(tmp_path / "a.edf")
    _write_edf_stub(tmp_path / "a_annotations.edf")
    _write_edf_stub(tmp_path / "b_annotations.edf")  # no recording sibling
    result = check_annotation_pairing(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["orphan_stubs"] == ["b_annotations.edf"]
    assert result["orphan_recordings"] == []


def test_pairing_fail_empty_input():
    result = check_annotation_pairing([])
    assert result["status"] == "fail"
    assert result["n_recordings"] == 0 and result["n_stubs"] == 0


def test_pairing_pass_single_pair(tmp_path):
    # Smallest valid stub_pair case: 1 recording + 1 stub.
    _write_edf_stub(tmp_path / "only.edf")
    _write_edf_stub(tmp_path / "only_annotations.edf")
    result = check_annotation_pairing(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "pass"
    assert result["mode"] == "stub_pair"
    assert result["paired"][0] == {"recording": "only.edf", "stub": "only_annotations.edf"}


# --- signal-header uniformity ----------------------------------------------


def test_uniformity_pass_identical_headers(tmp_path):
    for name in ("a.edf", "b.edf", "c.edf"):
        _write_edf_with_signals(tmp_path / name,
                                n_records=5, samples_per_record=100, n_signals=2)
    result = check_signal_header_uniformity(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "pass"
    assert result["n_unique_signatures"] == 1
    assert result["n_files"] == 3


def test_uniformity_pass_single_file(tmp_path):
    _write_edf_with_signals(tmp_path / "only.edf",
                            n_records=5, samples_per_record=100)
    result = check_signal_header_uniformity([tmp_path / "only.edf"])
    assert result["status"] == "pass"
    assert result["n_unique_signatures"] == 1


def test_uniformity_fail_different_sample_rates(tmp_path):
    _write_edf_with_signals(tmp_path / "fast.edf",
                            n_records=5, samples_per_record=500)
    _write_edf_with_signals(tmp_path / "slow.edf",
                            n_records=5, samples_per_record=100)
    result = check_signal_header_uniformity(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["n_unique_signatures"] == 2
    assert any("distinct signal-header signatures" in msg for msg in result["issues"])


def test_uniformity_fail_different_labels(tmp_path):
    _write_edf_with_signals(tmp_path / "a.edf",
                            n_records=5, samples_per_record=100, label_prefix="EEG")
    _write_edf_with_signals(tmp_path / "b.edf",
                            n_records=5, samples_per_record=100, label_prefix="ECG")
    result = check_signal_header_uniformity(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["n_unique_signatures"] == 2


def test_uniformity_ignores_phys_range_drift(tmp_path):
    """phys_min / phys_max / dig_min / dig_max are calibration values
    the recorder derives from the actual signal extremes within each
    file, so they legitimately vary from recording to recording even
    when the montage (labels, sample rate, units) is unchanged.
    Including them in the signature spuriously fragmented every real
    subject's summary; the montage is what should be compared."""
    _write_edf_with_signals(tmp_path / "a.edf",
                            n_records=5, samples_per_record=100,
                            phys_min=-3200.0, phys_max=3200.0)
    _write_edf_with_signals(tmp_path / "b.edf",
                            n_records=5, samples_per_record=100,
                            phys_min=-1600.0, phys_max=1600.0)
    result = check_signal_header_uniformity(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "pass"
    assert result["n_unique_signatures"] == 1


def test_uniformity_fail_different_channel_counts(tmp_path):
    _write_edf_with_signals(tmp_path / "a.edf",
                            n_records=5, samples_per_record=100, n_signals=2)
    _write_edf_with_signals(tmp_path / "b.edf",
                            n_records=5, samples_per_record=100, n_signals=3)
    result = check_signal_header_uniformity(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["n_unique_signatures"] == 2


def test_uniformity_fail_empty_input():
    result = check_signal_header_uniformity([])
    assert result["status"] == "fail"


def test_uniformity_canonicalizes_string_padding():
    """Trailing ASCII padding on labels/dims (from fixed-width slots)
    must not fragment signatures — otherwise files that are
    functionally identical get counted as distinct montages."""
    from clean_eeg.audit.checks import _signal_header_signature

    file_a = [{"label": "EEG Fp1", "samples_per_record": 250,
               "phys_dim": "uV"}]
    file_b = [{"label": "EEG Fp1     ",  # trailing padding
               "samples_per_record": 250,
               "phys_dim": "uV  "}]

    sig_a = _signal_header_signature(file_a, ignore_annotation_channel=True)
    sig_b = _signal_header_signature(file_b, ignore_annotation_channel=True)
    assert sig_a == sig_b, (
        f"Functionally identical headers must produce identical "
        f"signatures. Got:\n  a={sig_a}\n  b={sig_b}"
    )


def test_uniformity_reports_per_field_variability(tmp_path):
    """When signatures fragment, the check must name the specific
    montage fields with per-file variation so the operator can tell
    which axis of the montage changed."""
    # Two files where sample rate differs — labels and units match.
    # Signature fragments, and varying_fields should name only
    # samples_per_record.
    _write_edf_with_signals(tmp_path / "a.edf",
                            n_records=5, samples_per_record=100)
    _write_edf_with_signals(tmp_path / "b.edf",
                            n_records=5, samples_per_record=250)

    result = check_signal_header_uniformity(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert set(result["varying_fields"]) == {"samples_per_record"}, (
        result["varying_fields"]
    )
    # And the human-readable issues list surfaces that field name.
    assert any("samples_per_record" in m for m in result["issues"]), result["issues"]


def test_uniformity_caps_per_signature_enumeration(tmp_path):
    """When many signatures fragment, the summary must not enumerate
    every one — that's exactly the noise the compression is meant to
    avoid. Full mapping still lives in edf_audit.json."""
    # 8 files with 8 distinct sample rates → 8 signatures.
    for i in range(8):
        _write_edf_with_signals(tmp_path / f"f{i}.edf",
                                n_records=5,
                                samples_per_record=100 + i * 10)

    result = check_signal_header_uniformity(sorted(tmp_path.glob("*.edf")))
    assert result["n_unique_signatures"] == 8

    sig_lines = [m for m in result["issues"] if "signature #" in m]
    # Cap of 5 individual signature lines.
    assert len(sig_lines) == 5, sig_lines
    # Plus a "…and N more" trailer.
    assert any("more signature" in m for m in result["issues"]), result["issues"]


def test_uniformity_records_representative_channels(tmp_path):
    # Result should carry a representative channel list per signature
    # so --print-edf-signal-header can display it.
    _write_edf_with_signals(tmp_path / "a.edf",
                            n_records=5, samples_per_record=250, n_signals=2)
    result = check_signal_header_uniformity([tmp_path / "a.edf"])
    channels = result["signatures"]["signature_1"]["channels"]
    assert len(channels) == 2
    assert channels[0]["label"] == "EEG0"
    assert channels[0]["samples_per_record"] == 250


# --- byte geometry ---------------------------------------------------------


def test_geometry_pass_matching_filesize(tmp_path):
    _write_edf_with_signals(tmp_path / "a.edf", n_records=10, samples_per_record=100)
    result = check_byte_geometry([tmp_path / "a.edf"])
    assert result["status"] == "pass"
    assert result["ok_files"] == ["a.edf"]
    assert result["verdicts_by_file"]["a.edf"] == "OK"
    d = result["details_by_file"]["a.edf"]
    assert d["n_records_claimed"] == 10
    assert d["n_records_actual"] == 10
    assert d["record_bytes"] == 200  # 100 spr * 1 signal * 2 bytes


def test_geometry_pass_multichannel(tmp_path):
    _write_edf_with_signals(tmp_path / "a.edf",
                            n_records=5, samples_per_record=50, n_signals=3)
    result = check_byte_geometry([tmp_path / "a.edf"])
    assert result["status"] == "pass"
    assert result["details_by_file"]["a.edf"]["record_bytes"] == 300  # 50*3*2


def test_geometry_fail_truncated(tmp_path):
    # Header claims 10 records but only 5 records' worth of data on disk.
    _write_edf_with_signals(tmp_path / "trunc.edf",
                            n_records=10, samples_per_record=100,
                            data_bytes_override=5 * 200)
    result = check_byte_geometry([tmp_path / "trunc.edf"])
    assert result["status"] == "fail"
    assert result["truncated_files"] == ["trunc.edf"]
    assert "TRUNCATED" in result["verdicts_by_file"]["trunc.edf"]
    assert any("TRUNCATED" in msg for msg in result["issues"])


def test_geometry_warn_oversized(tmp_path):
    # Header claims 3 records but disk holds 5 records' worth.
    _write_edf_with_signals(tmp_path / "extra.edf",
                            n_records=3, samples_per_record=100,
                            data_bytes_override=5 * 200)
    result = check_byte_geometry([tmp_path / "extra.edf"])
    assert result["status"] == "warn"
    assert result["oversized_files"] == ["extra.edf"]
    assert "OVER-SIZED" in result["verdicts_by_file"]["extra.edf"]


def test_geometry_warn_uncomputable_spr(tmp_path):
    # samples_per_record = 0 — pyedflib rejects, we mark UNCOMPUTABLE.
    _write_edf_with_signals(tmp_path / "bad.edf",
                            n_records=10, samples_per_record=0)
    result = check_byte_geometry([tmp_path / "bad.edf"])
    assert result["status"] == "warn"
    assert result["uncomputable_files"] == ["bad.edf"]


def test_geometry_fail_no_files():
    result = check_byte_geometry([])
    assert result["status"] == "fail"


def test_geometry_fail_beats_warn_when_mixed(tmp_path):
    # One truncated + one oversized in the same audit → status is fail
    # (truncated is the more serious signal).
    _write_edf_with_signals(tmp_path / "trunc.edf",
                            n_records=10, samples_per_record=100,
                            data_bytes_override=5 * 200)
    _write_edf_with_signals(tmp_path / "extra.edf",
                            n_records=3, samples_per_record=100,
                            data_bytes_override=5 * 200)
    result = check_byte_geometry(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["truncated_files"] == ["trunc.edf"]
    assert result["oversized_files"] == ["extra.edf"]


# --- transfer integrity (SHA-256 manifest) ---------------------------------


def test_hash_pass_first_run_records_manifest(tmp_path):
    _write_edf_stub(tmp_path / "a.edf")
    _write_edf_stub(tmp_path / "b.edf", starttime="01.00.00")

    result = check_transfer_integrity(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert result["first_run"] is True
    assert result["n_files"] == 2
    assert set(result["file_hashes"].keys()) == {"a.edf", "b.edf"}
    assert all(len(h) == 64 for h in result["file_hashes"].values())
    assert result["mismatches"] == {}
    assert result["new_files"] == []
    assert result["missing_files"] == []


def test_hash_pass_second_run_unchanged(tmp_path):
    _write_edf_stub(tmp_path / "a.edf")
    first = check_transfer_integrity([tmp_path / "a.edf"])

    second = check_transfer_integrity([tmp_path / "a.edf"],
                                      previous_hashes=first["file_hashes"])

    assert second["status"] == "pass"
    assert second["first_run"] is False
    assert second["mismatches"] == {}


def test_hash_fail_content_changed(tmp_path):
    _write_edf_stub(tmp_path / "a.edf")
    first = check_transfer_integrity([tmp_path / "a.edf"])

    # Modify the file — flip one byte after the header.
    with open(tmp_path / "a.edf", "r+b") as f:
        f.seek(255)
        f.write(b"\x01")

    second = check_transfer_integrity([tmp_path / "a.edf"],
                                      previous_hashes=first["file_hashes"])

    assert second["status"] == "fail"
    assert "a.edf" in second["mismatches"]
    assert (second["mismatches"]["a.edf"]["stored"]
            != second["mismatches"]["a.edf"]["current"])
    assert any("hash changed" in msg for msg in second["issues"])


def test_hash_fail_file_missing_from_transfer(tmp_path):
    _write_edf_stub(tmp_path / "a.edf")
    _write_edf_stub(tmp_path / "b.edf", starttime="01.00.00")
    first = check_transfer_integrity(sorted(tmp_path.glob("*.edf")))

    (tmp_path / "b.edf").unlink()

    second = check_transfer_integrity([tmp_path / "a.edf"],
                                      previous_hashes=first["file_hashes"])

    assert second["status"] == "fail"
    assert second["missing_files"] == ["b.edf"]
    assert any("not present now" in msg for msg in second["issues"])


def test_hash_pass_new_file_is_additive(tmp_path):
    _write_edf_stub(tmp_path / "a.edf")
    first = check_transfer_integrity([tmp_path / "a.edf"])

    _write_edf_stub(tmp_path / "b.edf", starttime="01.00.00")

    second = check_transfer_integrity(sorted(tmp_path.glob("*.edf")),
                                      previous_hashes=first["file_hashes"])

    assert second["status"] == "pass"
    assert second["new_files"] == ["b.edf"]
    assert second["mismatches"] == {}
    assert second["missing_files"] == []


def test_hash_fail_empty_input():
    result = check_transfer_integrity([])
    assert result["status"] == "fail"
    assert result["n_files"] == 0


def test_sha256_of_file_matches_known_digest(tmp_path):
    # Positive sanity check: known content → known digest.
    (tmp_path / "hi.txt").write_bytes(b"hello world")
    # `printf 'hello world' | shasum -a 256`
    assert sha256_of_file(tmp_path / "hi.txt") == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


# --- fast-hash (head/middle/tail sampling) --------------------------------


def _write_edf_for_fast_hash(path, *, n_records=10, samples_per_record=8,
                             n_signals=1):
    """Wrapper: a valid EDF with enough records that fast-hash's three
    2 s windows (2 records each at 1 s/record) don't overlap.
    """
    _write_edf_with_signals(path, n_records=n_records,
                            samples_per_record=samples_per_record,
                            n_signals=n_signals)


def test_fast_hash_returns_fast_mode_on_a_long_file(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")

    digest, mode, details = sha256_fast_of_file(tmp_path / "a.edf")

    assert mode == "fast"
    assert len(digest) == 64
    assert details["records_per_window"] == 2  # 2 s @ 1 s/record
    assert details["window_bytes"] > 0
    # Windows must be in strict order and non-overlapping
    offsets = details["window_offsets"]
    assert offsets["start"] < offsets["middle"] < offsets["end"]


def test_fast_hash_deterministic_on_identical_file(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")
    d1, _, _ = sha256_fast_of_file(tmp_path / "a.edf")
    d2, _, _ = sha256_fast_of_file(tmp_path / "a.edf")
    assert d1 == d2


def test_fast_hash_falls_back_to_full_on_short_file(tmp_path):
    # Only 4 records @ 1 s each — three 2 s windows would cover the whole
    # data body, so the implementation should fall back to full hashing.
    _write_edf_for_fast_hash(tmp_path / "a.edf", n_records=4)

    digest, mode, details = sha256_fast_of_file(tmp_path / "a.edf")

    assert mode == "full"
    assert digest == sha256_of_file(tmp_path / "a.edf")
    assert "too short" in details.get("reason", "")


def test_fast_hash_falls_back_to_full_on_unparseable_header(tmp_path):
    # A 256-byte main-header-only stub has no signal headers and no data —
    # sha256_fast_of_file cannot compute record geometry and must fall back.
    _write_edf_stub(tmp_path / "a.edf", n_records=0)

    digest, mode, details = sha256_fast_of_file(tmp_path / "a.edf")

    assert mode == "full"
    assert digest == sha256_of_file(tmp_path / "a.edf")
    assert "unparseable" in details.get("reason", "") or "no data" in details.get("reason", "")


def test_fast_hash_detects_header_tampering(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")
    original, _, _ = sha256_fast_of_file(tmp_path / "a.edf")

    with open(tmp_path / "a.edf", "r+b") as f:
        f.seek(8)  # inside patient_id
        f.write(b"X")

    tampered, _, _ = sha256_fast_of_file(tmp_path / "a.edf")
    assert original != tampered


def test_fast_hash_detects_start_window_bit_rot(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")
    original, _, details = sha256_fast_of_file(tmp_path / "a.edf")

    # Flip a byte inside the start window (immediately after the header)
    with open(tmp_path / "a.edf", "r+b") as f:
        f.seek(details["window_offsets"]["start"] + 4)
        f.write(b"\xff")

    corrupted, _, _ = sha256_fast_of_file(tmp_path / "a.edf")
    assert original != corrupted


def test_fast_hash_detects_middle_window_bit_rot(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")
    original, _, details = sha256_fast_of_file(tmp_path / "a.edf")

    with open(tmp_path / "a.edf", "r+b") as f:
        f.seek(details["window_offsets"]["middle"] + 4)
        f.write(b"\xff")

    corrupted, _, _ = sha256_fast_of_file(tmp_path / "a.edf")
    assert original != corrupted


def test_fast_hash_detects_end_window_bit_rot(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")
    original, _, details = sha256_fast_of_file(tmp_path / "a.edf")

    # Last byte of the file — inside the end window's tail.
    file_size = (tmp_path / "a.edf").stat().st_size
    with open(tmp_path / "a.edf", "r+b") as f:
        f.seek(file_size - 1)
        f.write(b"\xff")

    corrupted, _, _ = sha256_fast_of_file(tmp_path / "a.edf")
    assert original != corrupted


def test_fast_hash_documented_blind_spot_between_windows(tmp_path):
    """Regression / feature-lock: bit-rot in the gap *between* windows
    is NOT detected by fast-hash. This is by design — the operator opts
    into head/middle/tail sampling and accepts the coverage trade-off.

    If this test starts failing (i.e., the write is being detected), the
    windowing math changed and someone should audit whether the new
    coverage still matches the documentation.
    """
    # Long enough to have a real gap: 20 records → windows at
    # start (records 0-1), middle (records 9-10), end (records 18-19).
    # Byte between the start and middle windows should be untouched.
    _write_edf_for_fast_hash(tmp_path / "a.edf", n_records=20)
    original, _, details = sha256_fast_of_file(tmp_path / "a.edf")

    gap_offset = (details["window_offsets"]["start"]
                  + details["window_bytes"] + 8)
    assert gap_offset < details["window_offsets"]["middle"]
    with open(tmp_path / "a.edf", "r+b") as f:
        f.seek(gap_offset)
        f.write(b"\xff")

    same, _, _ = sha256_fast_of_file(tmp_path / "a.edf")
    assert same == original, (
        "fast-hash should not see bit-rot in the untouched gap between "
        "windows — if this assertion flips, the windowing changed and "
        "the doc/coverage claim needs updating"
    )


# --- check_transfer_integrity(hash_mode=...) -----------------------------


def test_check_transfer_integrity_records_fast_mode_metadata(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")
    _write_edf_for_fast_hash(tmp_path / "b.edf")

    result = check_transfer_integrity(sorted(tmp_path.glob("*.edf")),
                                      hash_mode="fast")

    assert result["hash_mode"] == "fast"
    assert result["hash_mode_by_file"] == {"a.edf": "fast", "b.edf": "fast"}
    assert set(result["hash_details_by_file"].keys()) == {"a.edf", "b.edf"}


def test_check_transfer_integrity_none_mode_skips_hashing(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")

    result = check_transfer_integrity([tmp_path / "a.edf"], hash_mode="none")

    assert result["hash_mode"] == "none"
    assert result["status"] == "warn"
    assert result["file_hashes"] == {}
    assert any("was not run" in msg for msg in result["issues"])


def test_check_transfer_integrity_mode_mismatch_warns_not_fails(tmp_path):
    """Switching hash_mode invalidates the digest comparison — the
    check should warn and re-record, not spuriously fail every file."""
    _write_edf_for_fast_hash(tmp_path / "a.edf")

    prior = check_transfer_integrity([tmp_path / "a.edf"], hash_mode="full")
    switched = check_transfer_integrity(
        [tmp_path / "a.edf"],
        previous_hashes=prior["file_hashes"],
        previous_hash_mode="full",
        hash_mode="fast",
    )

    assert switched["status"] == "warn"
    assert switched["hash_mode"] == "fast"
    assert switched["mismatches"] == {}
    assert any("hash_mode changed" in msg for msg in switched["issues"])
    # And the newly recorded hashes are the fast digests, not the old full ones
    assert switched["file_hashes"]["a.edf"] != prior["file_hashes"]["a.edf"]


def test_check_transfer_integrity_same_mode_still_detects_mismatch(tmp_path):
    """Regression guard: the mode-mismatch escape hatch must NOT hide a
    real content change when the mode is unchanged."""
    _write_edf_for_fast_hash(tmp_path / "a.edf")
    prior = check_transfer_integrity([tmp_path / "a.edf"], hash_mode="fast")

    # Real bit-rot inside the start window
    with open(tmp_path / "a.edf", "r+b") as f:
        f.seek(prior["hash_details_by_file"]["a.edf"]["window_offsets"]["start"])
        f.write(b"\xff")

    second = check_transfer_integrity(
        [tmp_path / "a.edf"],
        previous_hashes=prior["file_hashes"],
        previous_hash_mode="fast",
        hash_mode="fast",
    )
    assert second["status"] == "fail"
    assert "a.edf" in second["mismatches"]


def test_check_transfer_integrity_invalid_hash_mode_raises(tmp_path):
    _write_edf_for_fast_hash(tmp_path / "a.edf")
    import pytest as _pytest
    with _pytest.raises(ValueError, match="hash_mode"):
        check_transfer_integrity([tmp_path / "a.edf"], hash_mode="quantum")


# Removed test_gaps_custom_threshold_recovers_pass — the max_gap_seconds
# override parameter was dropped in favor of a per-pair adaptive threshold
# (see GAP_THRESHOLD_ABSOLUTE_MAX_S in audit/checks.py). Custom overrides
# would fight the adaptive logic; if a knob is needed later, it should
# adjust the absolute cap rather than replace the per-pair scaling.


# --- summary printer: [OK] hiding by default ------------------------------

def _fake_audit_result(check_statuses: dict) -> dict:
    """Assemble a minimal audit dict for _print_summary tests."""
    return {
        "subject_dir": "/some/path",
        "subject_code": "R1755A",
        "n_files": 3,
        "mode": "full",
        "overall_status": ("fail" if "fail" in check_statuses.values()
                           else "warn" if "warn" in check_statuses.values()
                           else "pass"),
        "checks": {name: {"status": s, "issues": []}
                    for name, s in check_statuses.items()},
    }


def test_summary_hides_ok_checks_by_default(capsys):
    from clean_eeg.audit.cli import _print_summary
    audit = _fake_audit_result({
        "subject_code_consistency": "pass",
        "header_phi_residue": "pass",
        "recording_gaps": "fail",
    })
    _print_summary(audit)
    out = capsys.readouterr().out
    # FAIL check surfaced; PASS checks hidden.
    assert "[FAIL] recording_gaps" in out
    assert "[OK  ] subject_code_consistency" not in out
    assert "[OK  ] header_phi_residue" not in out
    # But operator is told that hidden passes exist.
    assert "2 passing check(s) hidden" in out


def test_summary_shows_ok_checks_when_verbose(capsys):
    from clean_eeg.audit.cli import _print_summary
    audit = _fake_audit_result({
        "subject_code_consistency": "pass",
        "recording_gaps": "fail",
    })
    _print_summary(audit, show_passes=True)
    out = capsys.readouterr().out
    assert "[OK  ] subject_code_consistency" in out
    assert "[FAIL] recording_gaps" in out
    # The "hidden" hint should not appear when we're showing everything.
    assert "hidden" not in out


def test_summary_skips_subject_header_when_banner_printed(capsys):
    from clean_eeg.audit.cli import _print_summary
    audit = _fake_audit_result({"recording_gaps": "pass"})
    _print_summary(audit, print_subject_header=False)
    out = capsys.readouterr().out
    # Path banner (=== Audit: ... ===) should NOT appear.
    assert "=== Audit:" not in out
    # But per-check counts still render.
    assert "Subject code:" in out
