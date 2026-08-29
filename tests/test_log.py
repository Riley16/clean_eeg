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


def test_middle_initial_single_letter_scrubbed_case_sensitive(tmp_path):
    """Regression: a single-letter middle initial should scrub standalone
    uppercase occurrences (as in a name context) without exploding into
    every 'L' inside 'Loading', 'False', etc.
    """
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("L")
    try:
        print("Dr. L. Smith arrived")
        print("Loading files and Volumes into cache; False alarms")
    finally:
        logger.close()

    content = open(log_path).read()
    # Positive: standalone 'L' in name context is redacted.
    assert "Dr. L. Smith" not in content
    assert "[PHI_REDACTED]" in content
    # Negative: interior 'L' inside common English words is preserved
    # because \b keeps it out AND we're case-sensitive so lowercase
    # 'l' inside 'Loading' etc. is safe too.
    assert "Loading" in content
    assert "Volumes" in content
    assert "False" in content


def test_middle_initial_lowercase_l_preserved(tmp_path):
    """Case-sensitive matching: a lowercase standalone 'l' is NOT
    scrubbed when the PHI pattern is 'L' (initials are conventionally
    written uppercase; a lowercase 'l' in prose is almost never PHI).
    """
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("L")
    try:
        print("some rare lowercase l here")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "some rare lowercase l here" in content
    assert "[PHI_REDACTED]" not in content


def test_middle_initial_with_trailing_period_normalized(tmp_path):
    """``add_phi('L.')`` should be equivalent to ``add_phi('L')`` — the
    trailing period gets stripped so the pattern still matches ``L``
    at word boundaries.
    """
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("L.")
    try:
        print("Dr. L. Smith and L there")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "L. Smith" not in content or "[PHI_REDACTED]" in content
    # Both 'L.' and standalone 'L' should be scrubbed.
    # 'Dr.' should survive.
    assert "Dr." in content


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


def test_logged_input_never_blocks_when_stdin_not_tty(tmp_path, capsys):
    """Headless-safety regression: when stdin isn't a TTY,
    logged_input must NEVER call input() (which would block forever
    under nohup/cron/piped-stdin). It should print a diagnostic to
    stderr and return the empty string so downstream code aborts the
    subject without stalling the batch.

    A prior version called input() unconditionally and turned overnight
    27-subject cleaning batches into 3-subject-completed / 24-stalled
    disasters."""
    log_path = str(tmp_path / "log.out")
    setup_logger(log_path)
    try:
        # No monkeypatch of builtins.input -- production shape, where
        # nothing intercepts and input() would block on stdin. Only the
        # isatty guard should short-circuit.
        with patch("sys.stdin.isatty", return_value=False):
            result = logged_input("Continue? yes/no: ")
        assert result == ""
        err = capsys.readouterr().err
        assert "[non-interactive]" in err
        assert "Continue?" in err
    finally:
        close_logger()


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


def test_short_phi_pattern_ignored(tmp_path):
    """Patterns with fewer than 3 alphabetic characters must NOT be
    registered — otherwise a single-letter middle initial like 'L' would
    replace every L in 'Loading', 'Volumes', 'False', etc., mangling
    the entire log file. Regression test for that exact incident."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("L")
    logger.add_phi("P.")  # 1 alpha char even though len > 1
    logger.add_phi("Jo")  # 2 alpha chars — still below threshold
    try:
        print("Loading EDF files from /Volumes/KahaDrive — Failed: False")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "Loading EDF files from /Volumes/KahaDrive" in content
    assert "Failed: False" in content
    assert "[PHI_REDACTED]" not in content


def test_realistic_log_with_lane_middle_name(tmp_path):
    """End-to-end scenario that previously mutilated the field log.

    Patient is 'John Lane Smith'. The pipeline registers each name part as
    PHI. Standalone 'L' characters in realistic stdout — file paths
    ('/Volumes/...'), status text ('Loading', 'Failed: False',
    'log.out') — must all survive intact. Only the actual word 'Lane'
    should be redacted, not every L."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    for part in ["John", "Lane", "Smith"]:
        logger.add_phi(part)
    try:
        print("Loading EDF files from /Volumes/KahaDrive/R1760A/")
        print("Cleaned EDF file at /Volumes/log.out — Failed: False")
        print("All Lane signed.")
    finally:
        logger.close()

    content = open(log_path).read()
    # Every readable bit of the log must survive intact — the bug
    # replaced every 'l'/'L' with [PHI_REDACTED].
    assert "Loading EDF files from /Volumes/KahaDrive/R1760A/" in content
    assert "Cleaned EDF file at /Volumes/log.out" in content
    assert "Failed: False" in content
    # The actual middle-name word IS redacted.
    assert "Lane" not in content
    assert "[PHI_REDACTED] signed." in content


def test_phi_pattern_uses_word_boundaries(tmp_path):
    """A PHI pattern for 'Mark' must NOT match inside 'Marks', 'Markup',
    or 'remark' — only at word boundaries."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    logger.add_phi("Mark")
    try:
        print("Marks the spot. Markup language. A remark.")
        print("Mark arrived.")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "Marks the spot." in content
    assert "Markup language." in content
    assert "A remark." in content
    # The standalone 'Mark' must still be redacted.
    assert "Mark arrived." not in content
    assert "[PHI_REDACTED] arrived." in content


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


def test_relocate_moves_log_and_continues_writing(tmp_path):
    """relocate() should move the existing log to the new path and continue
    writing there. Old path should no longer exist."""
    initial_path = str(tmp_path / "initial.log")
    new_path = str(tmp_path / "subject_dir" / "log.out")

    logger = PipelineLogger(initial_path)
    try:
        print("line before relocate")
        logger.relocate(new_path)
        print("line after relocate")
    finally:
        logger.close()

    assert not os.path.exists(initial_path), "original log file should have been moved"
    assert os.path.exists(new_path)
    assert logger.log_path == os.path.abspath(new_path)

    content = open(new_path).read()
    assert "line before relocate" in content
    assert "line after relocate" in content


def test_relocate_same_path_is_noop(tmp_path):
    """relocate() to the current path should be a no-op and not lose content."""
    log_path = str(tmp_path / "log.out")
    logger = PipelineLogger(log_path)
    try:
        print("content")
        logger.relocate(log_path)
        print("more content")
    finally:
        logger.close()

    content = open(log_path).read()
    assert "content" in content
    assert "more content" in content
