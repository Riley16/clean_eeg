"""Tests for src/clean_eeg/scrub_header_dates.py.

Coverage priorities:
  1. Signal bytes byte-identical before/after (SHA256).
  2. Annotation channel bytes byte-identical (annotation-review edits
     stay intact even though the pipeline's startdate scrub ran on top).
  3. startdate + recording_id updated as expected on disk (via pyedflib
     re-open AND via a raw byte read of the "Startdate DD-MMM-YYYY"
     text prefix).
  4. Relative offsets preserved across multi-file subjects.
  5. Sidecar main-header stays in sync with its main file.
  6. Rejection of the corruption-safety guard: an invalid base date
     doesn't get written silently.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from clean_eeg.scrub_header_dates import (
    BASE_START_DATE,
    audit_headers,
    compute_shifted_startdates,
    main as cli_main,
    scrub_subject_startdates,
)


def _write_edf(path: Path, startdate: datetime,
                annotations: list[tuple[float, str]] | None = None) -> None:
    """Small 2-channel EDF fixture with a chosen startdate. Uses the
    same primitives as the annotation-review test suite so behavior is
    consistent across tests."""
    n_ch = 2
    sr = 100
    duration_s = 5
    signal_headers = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sr,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_ch)
    ]
    t = np.arange(0, duration_s, 1.0 / sr, dtype=np.float32)
    # Deterministic synthesis so signal-hash comparisons are stable
    # across runs.
    signals = [(1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
               for i in range(n_ch)]
    with pyedflib.EdfWriter(str(path), n_ch,
                             file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            "technician": "T", "recording_additional": "",
            "patientname": "X", "patient_additional": "",
            "patientcode": "R1TEST", "equipment": "X", "admincode": "",
            "sex": "X",
            "startdate": startdate,
            "birthdate": "01 jan 1970", "gender": "X",
        })
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)
        for onset, text in annotations or []:
            f.writeAnnotation(onset, -1, text)


def _signal_hash(path: Path) -> str:
    with pyedflib.EdfReader(str(path)) as f:
        sigs = [f.readSignal(i, digital=True)
                for i in range(f.signals_in_file)]
    h = hashlib.sha256()
    for s in sigs:
        h.update(np.ascontiguousarray(s).tobytes())
    return h.hexdigest()


def _annotation_texts(path: Path) -> list[str]:
    with pyedflib.EdfReader(str(path)) as f:
        _, _, texts = f.readAnnotations()
    return [str(t) for t in texts if str(t).strip()]


def _recording_id_prefix(path: Path) -> str:
    """Bytes 88-167 of the main header, stripped."""
    with open(path, "rb") as f:
        f.seek(88)
        return f.read(80).decode("ascii", errors="replace").rstrip()


# ---------------------------------------------------------------------------
# Pure functions (no I/O side effects worth guarding)
# ---------------------------------------------------------------------------

def test_compute_shifted_startdates_preserves_relative_offsets(tmp_path):
    a = tmp_path / "a.edf"
    b = tmp_path / "b.edf"
    c = tmp_path / "c.edf"
    # a is earliest; b is 2 hours later; c is 3 days + 4h later.
    _write_edf(a, datetime(2023, 7, 4, 10, 0, 0))
    _write_edf(b, datetime(2023, 7, 4, 12, 0, 0))
    _write_edf(c, datetime(2023, 7, 7, 14, 0, 0))

    proposed = compute_shifted_startdates([a, b, c])
    assert proposed[a] == BASE_START_DATE                     # 1985-01-01 00:00:00
    assert proposed[b] == BASE_START_DATE + timedelta(hours=2)
    assert proposed[c] == BASE_START_DATE + timedelta(days=3, hours=4)


def test_compute_shifted_startdates_flatten_mode(tmp_path):
    a = tmp_path / "a.edf"
    b = tmp_path / "b.edf"
    _write_edf(a, datetime(2023, 7, 4, 10, 0, 0))
    _write_edf(b, datetime(2023, 7, 7, 14, 0, 0))

    proposed = compute_shifted_startdates([a, b], preserve_offsets=False)
    assert proposed[a] == BASE_START_DATE
    assert proposed[b] == BASE_START_DATE   # both flat to base


def test_compute_shifted_startdates_empty_input():
    assert compute_shifted_startdates([]) == {}


# ---------------------------------------------------------------------------
# I/O + corruption-safety
# ---------------------------------------------------------------------------

def test_scrub_updates_startdate_on_disk(tmp_path):
    p = tmp_path / "R1665J.edf"
    _write_edf(p, datetime(2023, 7, 4, 10, 30, 15))

    scrub_subject_startdates(tmp_path)

    with pyedflib.EdfReader(str(p)) as f:
        after = f.getHeader()["startdate"]
    assert after == BASE_START_DATE


def test_scrub_updates_recording_id_startdate_prefix(tmp_path):
    """The 'Startdate DD-MMM-YYYY' text at bytes 88-167 (the
    recording_id field) is what shows up in header viewers as
    'Startdate 04-JUL-2023'. Pyedflib re-derives it from the datetime
    on write. Guard: raw byte read must show the sentinel year."""
    p = tmp_path / "R1665J.edf"
    _write_edf(p, datetime(2023, 7, 4, 10, 30, 15))
    assert "2023" in _recording_id_prefix(p), (
        "test fixture: raw recording_id must contain the identifiable year")

    scrub_subject_startdates(tmp_path)

    rid_after = _recording_id_prefix(p)
    assert "2023" not in rid_after, (
        f"recording_id still leaks original year: {rid_after!r}")
    assert "1985" in rid_after, (
        f"recording_id must show BASE_START_DATE year 1985; got {rid_after!r}")


def test_scrub_preserves_signal_bytes_exactly(tmp_path):
    """HARD REQUIREMENT: the header rewrite must not touch any signal
    sample. Verified via SHA256 of the raw signal bytes."""
    p = tmp_path / "R1665J.edf"
    _write_edf(p, datetime(2023, 7, 4, 10, 30, 15))
    before = _signal_hash(p)

    scrub_subject_startdates(tmp_path)

    after = _signal_hash(p)
    assert before == after, (
        "signal bytes changed after header-only scrub -- byte-scope of "
        "update_edf_header_inplace regressed")


def test_scrub_preserves_annotations_verbatim(tmp_path):
    """Annotation-channel bytes must survive verbatim. Any existing
    annotation-review edits stay applied."""
    p = tmp_path / "R1665J.edf"
    _write_edf(p, datetime(2023, 7, 4, 10, 30, 15),
                annotations=[(0.5, "eyes closed"),
                             (2.0, "Segment: REC START X E")])
    before = _annotation_texts(p)

    scrub_subject_startdates(tmp_path)

    after = _annotation_texts(p)
    assert before == after, (
        f"annotations mutated by header scrub. before={before} after={after}")


def test_scrub_handles_multi_file_shift(tmp_path):
    a = tmp_path / "a.edf"
    b = tmp_path / "b.edf"
    _write_edf(a, datetime(2023, 7, 4, 10, 0, 0))
    _write_edf(b, datetime(2023, 7, 4, 12, 30, 0))

    scrub_subject_startdates(tmp_path)

    with pyedflib.EdfReader(str(a)) as f:
        sa = f.getHeader()["startdate"]
    with pyedflib.EdfReader(str(b)) as f:
        sb = f.getHeader()["startdate"]
    assert sa == BASE_START_DATE
    assert sb == BASE_START_DATE + timedelta(hours=2, minutes=30)


def test_scrub_updates_sidecar_alongside_main(tmp_path):
    """In-place cleaning writes a `<base>_annotations.edf` next to
    every main .edf. The sidecar's startdate must land at the SAME
    shifted value as its main, or downstream tools that read the
    sidecar header see a stale year."""
    main = tmp_path / "R1665J.edf"
    side = tmp_path / "R1665J_annotations.edf"
    _write_edf(main, datetime(2023, 7, 4, 10, 30, 15),
                annotations=[(0.0, "keep me")])
    _write_edf(side, datetime(2023, 7, 4, 10, 30, 15),
                annotations=[(0.0, "sidecar")])

    proposed = scrub_subject_startdates(tmp_path)
    # Sidecar isn't in `proposed` (only mains are keyed) but should be
    # updated to the same shift as its main.
    assert main in proposed

    with pyedflib.EdfReader(str(side)) as f:
        after = f.getHeader()["startdate"]
    assert after == BASE_START_DATE


def test_scrub_skip_sidecars_leaves_them_stale(tmp_path):
    """Negative regression: with include_sidecars=False the sidecar
    keeps its original startdate. Confirms the flag is respected --
    the CLI's --no-sidecars is a load-bearing knob."""
    main = tmp_path / "R1665J.edf"
    side = tmp_path / "R1665J_annotations.edf"
    _write_edf(main, datetime(2023, 7, 4, 10, 30, 15))
    _write_edf(side, datetime(2023, 7, 4, 10, 30, 15))

    scrub_subject_startdates(tmp_path, include_sidecars=False)

    with pyedflib.EdfReader(str(side)) as f:
        after = f.getHeader()["startdate"]
    assert after == datetime(2023, 7, 4, 10, 30, 15)


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------

def test_audit_headers_returns_expected_fields(tmp_path):
    p = tmp_path / "R1665J.edf"
    _write_edf(p, datetime(2023, 7, 4, 10, 30, 15))
    rows = audit_headers([p])
    assert len(rows) == 1
    r = rows[0]
    assert r["path"] == p
    assert r["patientcode"] == "R1TEST"
    assert r["startdate"] == datetime(2023, 7, 4, 10, 30, 15)
    # recording_id is a stripped bytes-88-167 read; must show the year.
    assert "2023" in r["recording_id"]


# ---------------------------------------------------------------------------
# CLI happy path (--audit + --yes apply)
# ---------------------------------------------------------------------------

def test_cli_audit_prints_current_state_no_writes(tmp_path, capsys):
    subj = tmp_path / "R1665J"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    p = inner / "R1665J_1.edf"
    _write_edf(p, datetime(2023, 7, 4, 10, 30, 15))
    before = _signal_hash(p)

    rc = cli_main(["--subject-dir", str(subj), "--audit"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Current header state" in out
    assert "2023" in out
    # No writes: signal hash unchanged (audit is read-only).
    assert _signal_hash(p) == before


def test_cli_apply_with_yes_shifts_all_files(tmp_path, capsys):
    subj = tmp_path / "R1665J"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    a = inner / "R1665J_1.edf"
    b = inner / "R1665J_2.edf"
    _write_edf(a, datetime(2023, 7, 4, 10, 0, 0))
    _write_edf(b, datetime(2023, 7, 4, 15, 0, 0))

    rc = cli_main(["--subject-dir", str(subj), "--yes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Proposed changes" in out
    assert "Post-update audit" in out

    with pyedflib.EdfReader(str(a)) as f:
        sa = f.getHeader()["startdate"]
    with pyedflib.EdfReader(str(b)) as f:
        sb = f.getHeader()["startdate"]
    assert sa == BASE_START_DATE
    assert sb == BASE_START_DATE + timedelta(hours=5)


def test_cli_invalid_base_date_rejects(tmp_path):
    subj = tmp_path / "R1665J"
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    _write_edf(inner / "a.edf", datetime(2023, 7, 4, 10, 0, 0))

    rc = cli_main(["--subject-dir", str(subj), "--yes",
                    "--base-date", "not-a-date"])
    assert rc == 2   # arg parse / validation error


def test_cli_no_subfolder_errors_cleanly(tmp_path, capsys):
    subj = tmp_path / "R1665J"
    subj.mkdir()  # subfolder missing on purpose
    rc = cli_main(["--subject-dir", str(subj), "--audit"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "does not exist" in err
