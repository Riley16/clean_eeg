"""Pipeline logging with PHI scrubbing.

Sets up a tee so all stdout/stderr output is duplicated to a log file.
PHI patterns (patient name parts) are scrubbed from the log but shown
on the console so the operator can verify correctness.
"""

import os
import re
import shutil
import sys
from datetime import datetime


class PipelineLogger:
    """Duplicate stdout/stderr to a log file, scrubbing PHI from the log only."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._phi_patterns = []
        self.log_file = open(log_path, "w")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self)
        sys.stderr = _TeeStream(self._orig_stderr, self)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write_to_log(f"=== clean_eeg log started {ts} ===\n\n")

    def add_phi(self, text: str):
        """Register a string as PHI to be scrubbed from all log output."""
        text = text.strip()
        if text:
            self._phi_patterns.append(
                re.compile(re.escape(text), re.IGNORECASE)
            )

    def scrub(self, text: str) -> str:
        """Replace all registered PHI patterns in text."""
        for pat in self._phi_patterns:
            text = pat.sub("[PHI_REDACTED]", text)
        return text

    def write_to_log(self, text: str):
        self.log_file.write(self.scrub(text))
        self.log_file.flush()

    def relocate(self, new_path: str):
        """Move the active log file to new_path and continue writing there.

        Preserves content already written. Safe to call mid-run — any further
        writes (including future prompts and traceback output) go to the new
        location.
        """
        new_path = os.path.abspath(new_path)
        if os.path.abspath(self.log_path) == new_path:
            return
        self.log_file.flush()
        self.log_file.close()
        os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
        shutil.move(self.log_path, new_path)
        self.log_path = new_path
        self.log_file = open(self.log_path, "a")

    def rescrub(self):
        """Re-scrub the entire log file with all currently registered PHI patterns.

        Call after registering new PHI patterns to ensure earlier log entries
        (written before the patterns were known) are also scrubbed.
        """
        self.log_file.flush()
        self.log_file.close()
        with open(self.log_path, "r") as f:
            content = f.read()
        self.log_file = open(self.log_path, "w")
        self.log_file.write(self.scrub(content))
        self.log_file.flush()

    def log_args(self, args):
        """Log CLI arguments (PHI is auto-scrubbed)."""
        self.write_to_log("=== CLI Arguments ===\n")
        for key, value in sorted(vars(args).items()):
            self.write_to_log(f"  {key}: {value}\n")
        self.write_to_log("=====================\n\n")

    def close(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write_to_log(f"\n=== clean_eeg log ended {ts} ===\n")
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self.log_file.close()


class _TeeStream:
    """Stream wrapper that writes to both the original stream and the log file."""

    def __init__(self, original, logger: PipelineLogger):
        self._original = original
        self._logger = logger

    def write(self, text):
        self._original.write(text)
        self._logger.write_to_log(text)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


# ---- Module-level singleton ----

_logger: PipelineLogger | None = None


def setup_logger(log_path: str) -> PipelineLogger:
    """Initialize the pipeline logger. Call once at program start."""
    global _logger
    _logger = PipelineLogger(log_path)
    return _logger


def get_logger() -> PipelineLogger | None:
    """Return the active PipelineLogger, or None."""
    return _logger


def close_logger():
    """Close the logger and restore original stdout/stderr."""
    global _logger
    if _logger is not None:
        _logger.close()
        _logger = None


def logged_input(prompt: str = "") -> str:
    """Drop-in replacement for input() that logs the user's response.

    The prompt itself is already captured by the TeeStream when input()
    writes it to stdout. This function additionally logs the user's
    typed response, which is read from stdin and not echoed through stdout.
    """
    response = input(prompt)
    if _logger is not None:
        _logger.write_to_log(f"{response}\n")
    return response
