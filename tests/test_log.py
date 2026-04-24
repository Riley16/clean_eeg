import argparse
import os
import sys
from unittest.mock import patch

from clean_eeg.log import (
    PipelineLogger,
    setup_logger,
    close_logger,
    logged_input,
)


def test_tee_captures_print(tmp_path):
    """print() output should appear in the log file."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    try:
        print("hello from test")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "hello from test" in content


def test_tee_captures_stderr(tmp_path):
    """stderr output should also appear in the log file."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    try:
        print("error message", file=sys.stderr)
    finally:
        logger.close()

    content = open(log_path).read()
    assert "error message" in content


def test_phi_scrubbed_from_log(tmp_path):
    """PHI patterns should be replaced in the log but NOT on the console."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("John")
    logger.add_phi("Smith")
    try:
        print("Patient name is John Smith")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "John" not in content
    assert "Smith" not in content
    assert "[PHI_REDACTED]" in content
    assert "Patient name is" in content


def test_phi_scrub_case_insensitive(tmp_path):
    """PHI scrubbing should be case-insensitive."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("Connor")
    try:
        print("CONNOR connor Connor")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "Connor" not in content
    assert "CONNOR" not in content
    assert "connor" not in content
    assert content.count("[PHI_REDACTED]") == 3


def test_rescrub_retroactive(tmp_path):
    """rescrub() should scrub PHI from log entries written before the pattern was registered."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    try:
        print("The patient is Jane Doe")
        # PHI registered AFTER the print
        logger.add_phi("Jane")
        logger.add_phi("Doe")
        logger.rescrub()
    finally:
        logger.close()

    content = open(log_path).read()
    assert "Jane" not in content
    assert "Doe" not in content
    assert "[PHI_REDACTED]" in content


def test_logged_input_captures_response(tmp_path):
    """logged_input() should log the user's typed response."""
    log_path = str(tmp_path / "log.out")
    logger = setup_logger(log_path)
    try:
        with patch("builtins.input", return_value="yes"):
            result = logged_input("Continue? ")
        assert result == "yes"
    finally:
        close_logger()

    content = open(log_path).read()
    assert "yes" in content


def test_logged_input_scrubs_phi(tmp_path):
    """logged_input() should scrub PHI from the logged response."""
    log_path = str(tmp_path / "log.out")
    logger = setup_logger(log_path)
    logger.add_phi("Riley")
    try:
        with patch("builtins.input", return_value="Riley"):
            result = logged_input("Enter name: ")
        assert result == "Riley"  # console gets the real value
    finally:
        close_logger()

    content = open(log_path).read()
    assert "Riley" not in content
    assert "[PHI_REDACTED]" in content


def test_log_args_scrubs_phi(tmp_path):
    """log_args() should scrub PHI from CLI argument values."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("Alice")
    logger.add_phi("Wonder")
    try:
        args = argparse.Namespace(
            first_name="Alice",
            last_name="Wonder",
            subject_code="R1234A",
            input_path="/data/edf",
        )
        logger.log_args(args)
    finally:
        logger.close()

    content = open(log_path).read()
    assert "Alice" not in content
    assert "Wonder" not in content
    assert "R1234A" in content
    assert "/data/edf" in content
    assert "[PHI_REDACTED]" in content


def test_close_restores_streams(tmp_path):
    """After close(), sys.stdout and sys.stderr should be restored."""
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    assert sys.stdout is not orig_stdout
    logger.close()
    assert sys.stdout is orig_stdout
    assert sys.stderr is orig_stderr


def test_log_contains_timestamps(tmp_path):
    """Log file should contain start and end timestamps."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.close()

    content = open(log_path).read()
    assert "clean_eeg log started" in content
    assert "clean_eeg log ended" in content


def test_empty_phi_ignored(tmp_path):
    """Empty or whitespace-only PHI patterns should not cause issues."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("")
    logger.add_phi("   ")
    try:
        print("normal text here")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "normal text here" in content
    assert "[PHI_REDACTED]" not in content


def test_redact_log_file_catches_name_variants(tmp_path):
    """redact_log_file() should catch fuzzy name matches and nicknames that
    pattern-based scrubbing would miss."""
    from clean_eeg.anonymize import PersonalName
    from clean_eeg.clean_subject_eeg import redact_log_file

    log_path = str(tmp_path / "log.out")

    # Write log content that includes the exact name, a nickname, and a fuzzy typo
    with open(log_path, "w") as f:
        f.write("CLI arg: first_name=John\n")
        f.write("CLI arg: last_name=O'Connor\n")
        f.write("User typed: John O'Connor\n")
        f.write("EDF header patientname: OConnor, John\n")
        f.write("subject_code: R1234A\n")

    subject_name = PersonalName(
        first_name="John",
        middle_names=[],
        last_name="O'Connor",
    )
    redact_log_file(log_path, subject_name)

    content = open(log_path).read()
    assert "John" not in content
    assert "O'Connor" not in content
    assert "OConnor" not in content
    assert "R1234A" in content
    from clean_eeg.anonymize import REDACT_NAME_REPLACEMENT
    assert REDACT_NAME_REPLACEMENT in content


def test_redact_log_file_with_middle_name(tmp_path):
    """redact_log_file() should redact middle names from the log."""
    from clean_eeg.anonymize import PersonalName
    from clean_eeg.clean_subject_eeg import redact_log_file

    log_path = str(tmp_path / "log.out")
    with open(log_path, "w") as f:
        f.write("first_name: Alice\n")
        f.write("middle_name: Marie\n")
        f.write("last_name: Smith\n")
        f.write("Patient: Alice Marie Smith\n")

    subject_name = PersonalName(
        first_name="Alice",
        middle_names=["Marie"],
        last_name="Smith",
    )
    redact_log_file(log_path, subject_name)

    content = open(log_path).read()
    assert "Alice" not in content
    assert "Marie" not in content
    assert "Smith" not in content
