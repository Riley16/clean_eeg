"""Self-contained tutorial for the clean -> audit -> manual-review -> transfer pipeline.

Generates its own synthetic subject data in a temp dir, runs every
non-interactive stage of the pipeline, then hands the operator a
copy-pasteable command to launch the TUI. Cleans up on completion
(or on abort).

Two modes:

    python scripts/tutorial_manual_review.py
        Runs stages 1-4, then pauses and prints the exact
        annotation-review-eeg command with the temp path substituted.
        Waits for the operator to hit Enter after quitting the TUI,
        then re-audits (should show the manual-review-complete
        banner if the operator marked files reviewed) and cleans up.

    python scripts/tutorial_manual_review.py --headless-smoke
        No TUI launch. Runs stages 1-4, then simulates a completed
        manual review by writing the tracker + applied-session
        artifacts directly, then runs stage 5 (re-audit) to confirm
        the ✓ banner fires. Fully non-interactive -- for CI / regression
        catching without a human at the keyboard.

Both modes clean up the scratch dir at exit (including Ctrl-C).
"""

from __future__ import annotations

import argparse
import atexit
import io
import shutil
import signal
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib

from clean_eeg.annotation_review.controller import preflight_subject_for_review
from clean_eeg.annotation_review.journal import ReviewedTracker
from clean_eeg.annotation_review.models import ReviewedFile
from clean_eeg.anonymize import PersonalName
from clean_eeg.audit.cli import _always_print_warnings
from clean_eeg.audit.subject import audit_subject
from clean_eeg.clean_subject_eeg import LOG_FILENAME, clean_subject_edf_files
from clean_eeg.deidentify_manifest import MANIFEST_FILENAME
from clean_eeg.log import close_logger, setup_logger


SUBJECT_CODE = "R1755J"
PATIENT_NAME = PersonalName(first_name="L.", middle_names=[], last_name="Smith")


# ---------------------------------------------------------------------------
# Synthetic fixture generator (self-contained -- no pytest, no session fixture)
# ---------------------------------------------------------------------------

def _write_minimal_edf(path: Path, *,
                       n_channels: int = 3,
                       sample_rate: int = 100,
                       duration_s: int = 4,
                       startdate: datetime = datetime(2023, 1, 1, 10, 0, 0),
                       annotation_texts: tuple[str, ...] = (
                           "REC START",
                           "seizure onset",
                           "dr. smith noted",
                       )) -> None:
    """Write one small EDF+C with a few realistic annotations. Copies
    the pattern from test_integration_transfer.py so the pipeline
    accepts it as a valid Nihon-Kohden-like recording."""
    signal_headers = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sample_rate,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_channels)
    ]
    t = np.arange(0, duration_s, 1.0 / sample_rate, dtype=np.float32)
    signals = [(1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
               for i in range(n_channels)]
    with pyedflib.EdfWriter(str(path), n_channels,
                             file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            "technician": "T", "recording_additional": "",
            "patientname": f"{PATIENT_NAME.first_name} {PATIENT_NAME.last_name}",
            "patient_additional": "",
            "patientcode": "PRE_CLEAN", "equipment": "test",
            "admincode": "", "sex": "Male",
            "startdate": startdate,
            "birthdate": "01 feb 1970", "gender": "Male",
        })
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)
        for i, text in enumerate(annotation_texts):
            f.writeAnnotation(0.5 + i * 0.5, -1, text)


def _generate_synthetic_subject(subject_inner: Path) -> None:
    """Two EDFs in the TUI-required <subject>/clinical_eeg/ layout.
    Second file starts right after the first ends (no gap, no overlap)
    so the pipeline doesn't stop to prompt about recording continuity."""
    subject_inner.mkdir(parents=True, exist_ok=True)
    duration_s = 4
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    t1 = datetime(2023, 1, 1, 10, 0, duration_s)
    _write_minimal_edf(subject_inner / "f01.edf",
                       duration_s=duration_s, startdate=t0,
                       annotation_texts=("REC START",
                                         "spike wave",
                                         "asleep"))
    _write_minimal_edf(subject_inner / "f02.edf",
                       duration_s=duration_s, startdate=t1,
                       annotation_texts=("REC START",
                                         "seizure onset (dr. smith)",
                                         "END"))


# ---------------------------------------------------------------------------
# Scratch dir lifecycle (auto-cleanup on any exit path)
# ---------------------------------------------------------------------------

def _make_scratch() -> Path:
    """Fresh temp dir. Registers cleanup on interpreter exit AND on
    SIGINT/SIGTERM so Ctrl-C doesn't leak."""
    root = Path(tempfile.mkdtemp(prefix="clean_eeg_tut_"))

    def _cleanup():
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)

    atexit.register(_cleanup)
    # SIGINT -> KeyboardInterrupt -> normal exit -> atexit fires.
    # But SIGTERM (kill) bypasses atexit, so install a handler.
    def _on_sigterm(*_args):
        _cleanup()
        sys.exit(130)
    signal.signal(signal.SIGTERM, _on_sigterm)
    return root


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _stage_generate(scratch: Path) -> Path:
    subject_dir = scratch / SUBJECT_CODE
    inner = subject_dir / "clinical_eeg"
    _generate_synthetic_subject(inner)
    n_edfs = len(list(inner.glob("*.edf")))
    print(f"[1/5] scratch → {scratch}")
    print(f"      generated {n_edfs} synthetic EDF(s) under {inner}")
    return subject_dir


def _stage_clean(subject_dir: Path, *, launch_review: bool) -> None:
    """Run the cleaner. When ``launch_review`` is True and stdin/stdout
    are TTYs, ``clean_subject_edf_files`` itself auto-runs the audit
    and launches the annotation-review TUI at end-of-clean -- the
    tutorial's post-review audit then just verifies the operator
    actually marked files reviewed."""
    inner = subject_dir / "clinical_eeg"
    log_path = inner / LOG_FILENAME
    setup_logger(str(log_path))
    try:
        clean_subject_edf_files(
            subject_name=PATIENT_NAME,
            subject_code=SUBJECT_CODE,
            input_path=str(inner),
            output_path=str(inner),
            inplace=True,
            auto_transfer_response="n",
            raise_errors=True,
            launch_review=launch_review,
        )
    finally:
        close_logger()
    print(f"[2/5] cleaned in-place"
          f"{' (audit + TUI auto-launched)' if launch_review else ''}")


def _stage_verify_artifacts(subject_dir: Path) -> tuple[list[Path], list[Path]]:
    inner = subject_dir / "clinical_eeg"
    main_edfs = sorted(p for p in inner.glob("*.edf")
                       if not p.name.endswith("_annotations.edf"))
    sidecars = sorted(p for p in inner.glob("*_annotations.edf"))
    manifest_path = inner / MANIFEST_FILENAME
    log_path = inner / LOG_FILENAME
    for p in [manifest_path, log_path]:
        assert p.exists(), f"expected {p.name} in {inner}"
    assert len(main_edfs) == 2, f"expected 2 cleaned EDFs, got {main_edfs}"
    assert len(sidecars) == 2, f"expected 2 sidecars, got {sidecars}"
    print(f"[3/5] verified: {len(main_edfs)} main EDF(s), "
          f"{len(sidecars)} sidecar(s), manifest, log")

    # Confirm the reviewer's preflight picks up this dir + prefers the
    # sidecars (this is what the TUI is about to do).
    carriers = preflight_subject_for_review(subject_dir)
    assert len(carriers) == 2
    assert all(c.name.endswith("_annotations.edf") for c in carriers), (
        f"preflight did not prefer sidecars: {[c.name for c in carriers]}"
    )
    print(f"      annotation-review preflight OK — will read from sidecars")
    return main_edfs, sidecars


def _stage_audit_pre_review(subject_dir: Path) -> dict:
    inner = subject_dir / "clinical_eeg"
    audit = audit_subject(inner, name_dictionary={"nonexistent"},
                          hash_mode="none")
    review = audit["checks"]["annotation_review_state"]
    assert review["state"] == "none", review
    assert review["n_annotation_carriers"] == 2
    print(f"[4/5] audit (pre-review): review_state=none, "
          f"n_carriers={review['n_annotation_carriers']}")
    return audit


def _stage_simulate_completed_review(subject_dir: Path,
                                      sidecars: list[Path]) -> None:
    """Write the exact on-disk artifacts a successful TUI pass would
    produce. Used by --headless-smoke."""
    tracker = ReviewedTracker(subject_dir / "clinical_eeg")
    for p in sidecars:
        tracker.mark_reviewed(ReviewedFile.new(
            file_path=str(p), n_annotations=3, n_edited=1))
    applied_dir = (subject_dir / "clinical_eeg"
                   / ".annotation_review" / "applied")
    applied_dir.mkdir(parents=True, exist_ok=True)
    (applied_dir / "session_smoke.jsonl").write_text(
        '{"edit":1}\n{"edit":2}\n')


def _stage_audit_post_review(subject_dir: Path) -> None:
    inner = subject_dir / "clinical_eeg"
    audit = audit_subject(inner, name_dictionary={"nonexistent"},
                          hash_mode="none", force=True)
    review = audit["checks"]["annotation_review_state"]
    print(f"[5/5] audit (post-review): review_state={review['state']}, "
          f"n_reviewed={review['n_reviewed']}, "
          f"n_edits_applied={review['n_edits_applied']}")
    buf = io.StringIO()
    _always_print_warnings(audit, out=buf)
    banner_line = next((L for L in buf.getvalue().splitlines()
                        if "Manual annotation review" in L), None)
    assert review["state"] == "complete", (
        f"expected review state 'complete' but got {review['state']}. "
        f"Renderer output was: {buf.getvalue()!r}"
    )
    assert banner_line is not None, (
        f"expected suppression banner; got: {buf.getvalue()!r}"
    )
    print(f"      renderer output: {banner_line.strip()}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tutorial_manual_review",
        description="Self-contained smoke of the clean + audit + review "
                    "pipeline. See module docstring for the two modes.",
    )
    p.add_argument("--headless-smoke", action="store_true",
                   help="No TUI launch. Simulates a completed review "
                        "via on-disk artifacts and verifies the audit "
                        "renderer flips to the ✓ banner. For CI / "
                        "regression catching.")
    args = p.parse_args(argv)

    scratch = _make_scratch()
    subject_dir = _stage_generate(scratch)

    if args.headless_smoke:
        # No TTY -> cleaner's auto-launch would no-op even if enabled,
        # but skip it explicitly for clarity. Then drive the review
        # via on-disk artifacts to prove the audit picks up completion.
        _stage_clean(subject_dir, launch_review=False)
        _main_edfs, sidecars = _stage_verify_artifacts(subject_dir)
        _stage_audit_pre_review(subject_dir)
        _stage_simulate_completed_review(subject_dir, sidecars)
        _stage_audit_post_review(subject_dir)
        print()
        print("SMOKE OK — full pipeline works end-to-end (scratch cleaned).")
        return 0

    # Interactive: cleaner auto-runs audit + launches TUI + prompts
    # transfer at end-of-clean. When the operator quits the TUI and
    # answers the transfer prompt, control returns here.
    _stage_clean(subject_dir, launch_review=True)
    _stage_verify_artifacts(subject_dir)
    # Confirm the operator's TUI session left the expected artifacts.
    try:
        _stage_audit_post_review(subject_dir)
    except AssertionError as e:
        print()
        print(f"[!] post-review audit did not detect a completed review: {e}")
        print("    (this is expected if you quit the TUI without marking "
              "files reviewed / applying edits)")
    print()
    print("TUTORIAL DONE — scratch cleaned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
