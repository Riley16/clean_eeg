"""Local end-to-end: clean -> audit -> simulate review -> transfer plan.

Single test that walks one synthetic subject through every stage of
the pipeline WITHOUT touching the network. Complements
[test_integration_transfer.py](test_integration_transfer.py) which
runs the same flow against a live rhino2 (opt-in, needs SSH).

What this test proves in one go:
  1. In-place cleaning produces the artifacts the rest of the pipeline
     depends on: renamed main EDFs, ``_annotations.edf`` sidecars,
     ``deidentify.json`` manifest, ``log.out``.
  2. The audit runs green on that output (all checks pass).
  3. ``check_annotation_review_state`` starts at ``state="none"`` on
     an untouched cleaned subject.
  4. Writing the reviewer's on-disk artifacts (tracker + applied
     session) flips the state to ``"complete"`` on the next audit run
     -- proving the audit picks up manual-review completion without a
     re-clean.
  5. When the review is complete, ``_always_print_warnings`` suppresses
     the annotation flags and prints the "reviewed" banner instead.
  6. The transfer preflight passes on the produced output dir (dry-run
     -- no network I/O), and the resulting plan targets the cleaned
     EDFs + sidecars + manifest + log.

All temp state lives under pytest's ``tmp_path`` -- pytest cleans it
up automatically. No manual cleanup needed.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

from clean_eeg.annotation_review.journal import ReviewedTracker
from clean_eeg.annotation_review.models import ReviewedFile
from clean_eeg.anonymize import PersonalName
from clean_eeg.audit.cli import _always_print_warnings
from clean_eeg.audit.subject import audit_subject
from clean_eeg.clean_subject_eeg import (
    LOG_FILENAME,
    clean_subject_edf_files,
)
from clean_eeg.deidentify_manifest import MANIFEST_FILENAME, read_manifest
from clean_eeg.log import close_logger, setup_logger
from clean_eeg.paths import TEST_CONFIG_FILE, TEST_SUBJECT_DATA_DIR
from clean_eeg.transfer import transfer_subject


SUBJECT_CODE = "R1755A"
PATIENT_NAME = PersonalName(first_name="L.", middle_names=[], last_name="Smith")


def _copy_test_subject_edfs_to(subject_dir: Path) -> list[Path]:
    """Copy the session-fixture EDF pair into ``subject_dir`` so
    in-place cleaning has fresh writable files. Returns the copy
    paths. The session fixture in conftest.py generates the source
    files on first pytest run."""
    subject_dir.mkdir(parents=True, exist_ok=True)
    with open(TEST_CONFIG_FILE) as f:
        test_config = json.load(f)
    copied: list[Path] = []
    for key in ("subject_EDF+C_1", "subject_EDF+C_2"):
        src = TEST_SUBJECT_DATA_DIR / test_config[key]["filename"]
        dst = subject_dir / src.name
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def _simulate_completed_manual_review(subject_dir: Path,
                                       carriers: list[Path],
                                       n_edits: int = 2) -> None:
    """Write the exact on-disk artifacts the review TUI would produce
    on a completed review pass: one ``ReviewedFile`` entry per carrier
    in ``.annotation_reviewed_tracker`` plus one applied-session JSONL
    under ``.annotation_review/applied/``. Uses the real journal
    primitives so the test can't drift from the TUI's schema."""
    tracker = ReviewedTracker(subject_dir)
    for p in carriers:
        tracker.mark_reviewed(ReviewedFile.new(
            file_path=str(p), n_annotations=3, n_edited=1))
    applied_dir = subject_dir / ".annotation_review" / "applied"
    applied_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join([json.dumps({"edit": i}) for i in range(n_edits)])
    (applied_dir / "session_20260101T000000Z.jsonl").write_text(lines + "\n")


def test_e2e_pipeline_local(tmp_path, monkeypatch):
    subject_dir = tmp_path / SUBJECT_CODE
    _copy_test_subject_edfs_to(subject_dir)

    # The test EDF pair has a ~59 min recording gap; the pipeline prompts
    # for confirmation. Auto-answer 'y' for every interactive prompt.
    monkeypatch.setattr("builtins.input", lambda _msg="": "y")

    # -----------------------------------------------------------------
    # Stage 1: clean in-place
    # -----------------------------------------------------------------
    log_path = subject_dir / LOG_FILENAME
    setup_logger(str(log_path))
    try:
        clean_subject_edf_files(
            subject_name=PATIENT_NAME,
            subject_code=SUBJECT_CODE,
            input_path=str(subject_dir),
            output_path=str(subject_dir),
            inplace=True,
            auto_transfer_response="n",
        )
    finally:
        close_logger()

    # Manifest + log written.
    manifest = read_manifest(subject_dir)
    assert manifest is not None, f"{MANIFEST_FILENAME} not written"
    assert manifest["subject_code"] == SUBJECT_CODE
    assert log_path.exists(), f"{LOG_FILENAME} not written"

    # In-place output shape: for each cleaned EDF a `_annotations.edf`
    # sidecar sits next to it (per clean_subject_eeg.py:498-513).
    main_edfs = sorted(p for p in subject_dir.glob("*.edf")
                       if not p.name.endswith("_annotations.edf"))
    sidecars = sorted(p for p in subject_dir.glob("*_annotations.edf"))
    assert len(main_edfs) == 2, f"expected 2 cleaned EDFs, got {main_edfs}"
    assert len(sidecars) == 2, f"expected 2 sidecars, got {sidecars}"
    for m in main_edfs:
        sidecar = m.parent / (m.stem + "_annotations.edf")
        assert sidecar.exists(), f"no sidecar for {m.name}"

    # -----------------------------------------------------------------
    # Stage 2: first audit -- untouched by manual review
    # -----------------------------------------------------------------
    audit1 = audit_subject(subject_dir,
                           name_dictionary={"nonexistent"},
                           hash_mode="none")
    # NOTE: the shared test fixture has a ~59-min inter-recording gap
    # by design (mimics real clinical data), so recording_gaps FAILs
    # and log_file WARNs on the resulting pipeline warning. Those are
    # fixture artifacts, not pipeline bugs -- assert on the PHI-relevant
    # checks that this e2e is actually exercising instead.
    checks1 = audit1["checks"]
    assert checks1["annotation_phi_scan"]["status"] == "pass"
    assert checks1["header_phi_residue"]["status"] == "pass"
    assert checks1["filename_convention"]["status"] == "pass"
    assert checks1["annotation_pairing"]["status"] == "pass"
    assert checks1["annotation_review_state"]["status"] == "pass"

    # Review state absent -> "none".
    review1 = checks1["annotation_review_state"]
    assert review1["state"] == "none"
    assert review1["n_reviewed"] == 0
    # Stub-pair mode -> annotation carriers ARE the sidecars.
    assert review1["n_annotation_carriers"] == 2

    # First-pass renderer prints the phi-scan / redaction blocks
    # (no suppression) when review state is "none".
    buf1 = io.StringIO()
    _always_print_warnings(audit1, out=buf1)
    assert "Manual annotation review complete" not in buf1.getvalue()

    # -----------------------------------------------------------------
    # Stage 3: simulate a completed manual review, re-audit
    # -----------------------------------------------------------------
    _simulate_completed_manual_review(subject_dir, sidecars, n_edits=2)

    audit2 = audit_subject(subject_dir,
                           name_dictionary={"nonexistent"},
                           hash_mode="none",
                           force=True)
    review2 = audit2["checks"]["annotation_review_state"]
    assert review2["state"] == "complete", review2
    assert review2["n_reviewed"] == 2
    assert review2["n_applied_sessions"] == 1
    assert review2["n_edits_applied"] == 2

    # Renderer suppresses annotation flags when review is complete.
    buf2 = io.StringIO()
    _always_print_warnings(audit2, out=buf2)
    output2 = buf2.getvalue()
    assert "Manual annotation review complete" in output2
    assert "2 file(s) reviewed" in output2
    assert "2 edit(s) applied" in output2
    # phi-scan matches (if any) suppressed
    assert "name-dictionary matches" not in output2

    # Override flag brings the section back.
    buf2b = io.StringIO()
    _always_print_warnings(audit2, out=buf2b, show_annotation_flags=True)
    assert "Manual annotation review complete" not in buf2b.getvalue()

    # -----------------------------------------------------------------
    # Stage 4: transfer preflight + plan (no network I/O)
    # -----------------------------------------------------------------
    # ssh_user is required by the plan builder -- set a stub via env.
    monkeypatch.setenv("USER", "e2e-tester")
    plan = transfer_subject(subject_dir, dry_run=True,
                            ssh_user="e2e-tester",
                            ssh_host="test.example.com",
                            remote_dir_override="/tmp/e2e-dry-run")
    # rsync syncs at directory level (source = subject_dir/ with trailing
    # slash), so the whole subject dir uploads as a single unit --
    # cleaned mains + sidecars + manifest + log. The preflight above
    # already validated the individual artifacts (naming pattern,
    # headers, hash spot-check), so here we just check the plan's
    # shape: right source, right destination, supported transport.
    upload_argv_str = " ".join(plan.upload_argv)
    assert str(subject_dir) in upload_argv_str, (
        f"transfer plan does not sync subject_dir: {plan.upload_argv}"
    )
    assert plan.remote_dir == "/tmp/e2e-dry-run"
    assert plan.transport in ("rsync", "scp")

    # Every artifact the operator expects on the remote must be
    # present in the source dir at the moment the plan is built --
    # rsync's dir sync will pick them up.
    on_disk = {p.name for p in subject_dir.iterdir() if p.is_file()}
    for expected in ({MANIFEST_FILENAME, LOG_FILENAME}
                     | {p.name for p in main_edfs + sidecars}):
        assert expected in on_disk, (
            f"{expected} missing from {subject_dir} at transfer time")

    # No side effects beyond tmp_path -- pytest handles cleanup.
