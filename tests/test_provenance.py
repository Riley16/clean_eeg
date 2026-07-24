"""Tests for the log.out provenance block."""

from __future__ import annotations

from clean_eeg.log import PipelineLogger
from clean_eeg.provenance import (
    KEY_DEPENDENCIES,
    _package_version,
    build_provenance_block,
    log_environment_provenance,
)


def test_block_contains_every_expected_section():
    block = build_provenance_block(
        argv=["clean-subject-eeg", "--subject_code", "R1000A"],
        git_provenance=lambda: ("deadbeef1234", False),
    )
    for expected in (
        "=== Provenance ===",
        "clean_eeg version:",
        "clean_eeg commit:",
        "command:",
        "python:",
        "platform:",
        "dependencies:",
        "==================",
    ):
        assert expected in block, f"missing section: {expected!r}"


def test_dirty_marker_present_when_tree_is_dirty():
    block = build_provenance_block(
        argv=["clean-subject-eeg"],
        git_provenance=lambda: ("abc123", True),
    )
    assert "DIRTY" in block
    assert "uncommitted edits at run time" in block


def test_dirty_marker_absent_when_tree_is_clean():
    block = build_provenance_block(
        argv=["clean-subject-eeg"],
        git_provenance=lambda: ("abc123", False),
    )
    assert "DIRTY" not in block


def test_reports_unknown_commit_when_not_a_git_checkout():
    """A wheel install has no git dir; the block must degrade gracefully
    rather than mask the fact that the SHA is unknowable."""
    block = build_provenance_block(
        argv=["clean-subject-eeg"],
        git_provenance=lambda: (None, False),
    )
    assert "not a git checkout" in block
    # And no dirty marker on the "unknown" line — dirty is only
    # meaningful when we have a SHA to attach it to.
    assert "DIRTY" not in block


def test_command_line_is_recorded_verbatim():
    argv = ["clean-subject-eeg", "--first-name", "John", "--subject_code", "R1000A"]
    block = build_provenance_block(argv=argv,
                                   git_provenance=lambda: ("abc123", False))
    # Full command reconstructable from the block; the tee is what
    # handles PHI scrubbing (tested in test_log_environment_provenance_...
    # below), so the raw builder shouldn't modify the argv.
    assert "clean-subject-eeg --first-name John --subject_code R1000A" in block


def test_every_key_dependency_is_reported():
    block = build_provenance_block(
        argv=["x"], git_provenance=lambda: (None, False),
    )
    for dep in KEY_DEPENDENCIES:
        assert f"    {dep}:" in block, f"missing dep: {dep}"


def test_missing_dep_reports_not_installed():
    """A dep that isn't installed must not raise — it should surface
    'not installed' so the operator sees exactly what was missing."""
    block = build_provenance_block(
        argv=["x"],
        git_provenance=lambda: (None, False),
        dependencies=("this-package-definitely-does-not-exist-12345",),
    )
    assert "this-package-definitely-does-not-exist-12345: not installed" in block


def test_package_version_survives_unknown_package():
    assert _package_version("nonexistent-xxxxxxxx") == "not installed"


# --- integration with PipelineLogger --------------------------------------


def test_log_environment_provenance_writes_via_logger(tmp_path):
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    try:
        log_environment_provenance(logger)
    finally:
        logger.close()

    content = open(log_path).read()
    assert "=== Provenance ===" in content
    assert "clean_eeg version:" in content


def test_command_line_phi_is_scrubbed_via_tee(tmp_path, monkeypatch):
    """The whole point of logging sys.argv after registering PHI is
    that the tee scrubs on write. Verify a name arg in argv never
    appears verbatim in the log file."""
    import sys as _sys
    monkeypatch.setattr(
        _sys, "argv",
        ["clean-subject-eeg", "--first-name", "Aloysius",
         "--subject_code", "R1000A"],
    )

    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    try:
        # Register PHI first — matches the call order in
        # clean_subject_eeg.py's __main__ block.
        logger.add_phi("Aloysius")
        logger.rescrub()
        log_environment_provenance(logger)
    finally:
        logger.close()

    content = open(log_path).read()
    assert "Aloysius" not in content, (
        "PHI name in sys.argv leaked to the log file — the tee's "
        "on-write scrubbing should have masked it"
    )
    assert "[PHI_REDACTED]" in content
    # The rest of the command should still be there
    assert "--subject_code R1000A" in content
