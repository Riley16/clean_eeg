"""Tests for the per-subject post-transfer audit checks."""

from __future__ import annotations

from pathlib import Path

from clean_eeg.audit.checks import (
    check_annotation_pairing,
    check_byte_geometry,
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
from clean_eeg.audit.hashes import check_transfer_integrity, sha256_of_file
from clean_eeg.audit.logs import check_log_file
from clean_eeg.audit.select import select_files
from clean_eeg.audit.signals import read_signal_window
from clean_eeg.audit.notebook import build_audit_notebook
from clean_eeg.audit.subject import AUDIT_JSON_FILENAME, audit_subject


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
    # 30-second gap between two files — under the 60s pipeline threshold.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=60, record_duration=1.0)
    _write_edf_stub(tmp_path / "b.edf",
                    starttime="00.01.30", n_records=60, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "pass"
    assert result["gaps"][0]["gap_seconds"] == 30.0


def test_gaps_fail_large_gap_missing_file(tmp_path):
    # 1-hour gap between two 60s files — well past the 60s threshold.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=60, record_duration=1.0)
    _write_edf_stub(tmp_path / "c.edf",
                    starttime="01.01.00", n_records=60, record_duration=1.0)

    result = check_recording_gaps(sorted(tmp_path.glob("*.edf")))

    assert result["status"] == "fail"
    assert len(result["large_gaps"]) == 1
    assert result["large_gaps"][0]["prev_file"] == "a.edf"
    assert result["large_gaps"][0]["next_file"] == "c.edf"
    assert result["large_gaps"][0]["gap_seconds"] == 3600.0
    assert any("possibly missing" in msg for msg in result["issues"])


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
    assert any("1 ERROR" in msg for msg in result["issues"])


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
    per_ann, inv = scan_annotation_texts([], _TEST_NAMES)
    assert per_ann == []
    assert inv == {}


def test_scan_pass_no_name_tokens():
    anns = [{"onset": 1.0, "text": "seizure onset"}]
    per_ann, inv = scan_annotation_texts(anns, _TEST_NAMES)
    assert per_ann == []
    assert inv == {}


def test_scan_fail_dictionary_hit():
    anns = [{"onset": 1.0, "text": "patient John reports headache"}]
    per_ann, inv = scan_annotation_texts(anns, _TEST_NAMES)
    assert len(per_ann) == 1
    assert per_ann[0]["matched_tokens"] == ["john"]
    assert list(inv) == ["john"]


def test_scan_case_insensitive_match():
    anns = [{"onset": 1.0, "text": "SMITH visit"},
            {"onset": 2.0, "text": "Smith visit"},
            {"onset": 3.0, "text": "smith visit"}]
    per_ann, inv = scan_annotation_texts(anns, _TEST_NAMES)
    assert len(per_ann) == 3
    assert inv["smith"] and len(inv["smith"]) == 3


def test_scan_multiple_hits_in_one_annotation():
    anns = [{"onset": 1.0, "text": "John Smith saw Sarah"}]
    per_ann, inv = scan_annotation_texts(anns, _TEST_NAMES)
    assert len(per_ann) == 1
    assert set(per_ann[0]["matched_tokens"]) == {"john", "smith", "sarah"}
    assert set(inv) == {"john", "smith", "sarah"}


def test_scan_pass_whitelisted_token_ignored():
    anns = [{"onset": 1.0, "text": "seizure noted by John"}]
    per_ann, inv = scan_annotation_texts(anns, _TEST_NAMES,
                                          vocab_whitelist={"John"})
    assert per_ann == []
    assert inv == {}


def test_scan_handles_hyphenated_and_apostrophe_names():
    anns = [{"onset": 1.0, "text": "seen by Jean-Luc"},
            {"onset": 2.0, "text": "notes from O'Connor"}]
    per_ann, inv = scan_annotation_texts(anns, _TEST_NAMES)
    assert len(per_ann) == 2
    assert set(inv) == {"jean-luc", "o'connor"}


def test_scan_ignores_numeric_and_punctuation():
    # +1.5s and (12:03) contain no letter tokens, so no false matches.
    anns = [{"onset": 1.0, "text": "+1.5s (12:03)"}]
    per_ann, _ = scan_annotation_texts(anns, _TEST_NAMES)
    assert per_ann == []


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

    for i, (name, starttime) in enumerate([("a", "00.00.00"), ("b", "00.01.00")]):
        sig = (np.sin(np.linspace(0, 20, 6000)) * 1000).astype("<i2")
        _write_edf_with_signals(subject_dir / f"{name}.edf",
                                n_records=60, samples_per_record=100,
                                starttime=starttime,
                                channel_samples=[sig])
        # Rewrite patient_id to include the subject code (helper defaults
        # to R1755J so this is just belt-and-braces for varied fixtures).
        _patch_patient_id(subject_dir / f"{name}.edf", pid)

        _write_edf_with_annotations(subject_dir / f"{name}_annotations.edf",
                                    annotations or [(0.5, None, "seizure")])
        _patch_patient_id(subject_dir / f"{name}_annotations.edf", pid)

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
        "subject_code_consistency", "header_phi_residue", "recording_gaps",
        "byte_geometry", "annotation_pairing", "signal_header_uniformity",
        "annotation_phi_scan", "transfer_integrity", "log_file",
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
    with open(subject_dir / "a.edf", "r+b") as f:
        f.seek(255)
        f.write(b"\xff")

    second = audit_subject(subject_dir, name_dictionary={"nonexistent"})
    assert second["checks"]["transfer_integrity"]["status"] == "fail"


def test_e2e_audit_skip_hashes_omits_transfer_integrity(tmp_path):
    subject_dir = _build_clean_subject(tmp_path)
    audit = audit_subject(subject_dir, name_dictionary={"nonexistent"},
                          skip_hashes=True)
    assert "transfer_integrity" not in audit["checks"]


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


def test_uniformity_fail_different_phys_range(tmp_path):
    _write_edf_with_signals(tmp_path / "a.edf",
                            n_records=5, samples_per_record=100,
                            phys_min=-3200.0, phys_max=3200.0)
    _write_edf_with_signals(tmp_path / "b.edf",
                            n_records=5, samples_per_record=100,
                            phys_min=-1600.0, phys_max=1600.0)
    result = check_signal_header_uniformity(sorted(tmp_path.glob("*.edf")))
    assert result["status"] == "fail"
    assert result["n_unique_signatures"] == 2


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


def test_gaps_custom_threshold_recovers_pass(tmp_path):
    # A 30-min gap fails at the default 60s threshold but should pass
    # when the operator overrides to 3600s.
    _write_edf_stub(tmp_path / "a.edf",
                    starttime="00.00.00", n_records=60, record_duration=1.0)
    _write_edf_stub(tmp_path / "b.edf",
                    starttime="00.31.00", n_records=60, record_duration=1.0)

    default = check_recording_gaps(sorted(tmp_path.glob("*.edf")))
    lenient = check_recording_gaps(sorted(tmp_path.glob("*.edf")),
                                   max_gap_seconds=3600)

    assert default["status"] == "fail"
    assert lenient["status"] == "pass"
    assert lenient["max_gap_seconds_threshold"] == 3600
