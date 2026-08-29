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
        """Register a string as PHI to be scrubbed from all log output.

        The pattern is anchored with ``\\b`` boundaries so ``"Mark"``
        does not match inside ``"Marks"`` or ``"Markup"``. A trailing
        period is stripped so ``"L."`` and ``"L"`` produce the same
        pattern (the operator often passes middle initials either way).

        Case-sensitivity depends on the number of alphabetic characters:
          - 3+ chars → case-INSENSITIVE (typical name matching).
          - 1-2 chars (single/double-letter initials) → case-SENSITIVE.
            A middle initial ``"L"`` scrubs ``"L Smith"`` and
            ``"Dr. L."`` but not every ``L`` in ``"Loading"``,
            ``"Volumes"``, ``"False"`` (Loading's L is followed by a
            word char, so ``\\b`` already excludes it) — and case
            sensitivity ensures we also don't scrub every lowercase
            ``l`` in ordinary English text.
        """
        text = text.strip().rstrip(".")
        n_alpha = sum(c.isalpha() for c in text)
        if n_alpha == 0:
            return
        flags = 0 if n_alpha < 3 else re.IGNORECASE
        self._phi_patterns.append(
            re.compile(r"\b" + re.escape(text) + r"\b", flags)
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


import builtins as _builtins
_ORIG_INPUT = _builtins.input


def logged_input(prompt: str = "") -> str:
    """Drop-in replacement for input() that logs the user's response.

    The prompt itself is already captured by the TeeStream when input()
    writes it to stdout. This function additionally logs the user's
    typed response, which is read from stdin and not echoed through stdout.

    Headless-safety: when stdin isn't a TTY (nohup, cron, subprocess
    pipe, SSH-without-PTY), NEVER block on input(). Emit the prompt
    text + a clear "[non-interactive]" banner to stderr and return the
    empty string. Downstream callers that treat any non-'y'/'yes'
    answer as "abort" then raise a RuntimeError, which the batch runner
    catches -- that subject fails, the batch keeps moving. This makes
    overnight batches bulletproof against a prompt we forgot to gate:
    the worst case is one subject fails, not the whole batch stalling
    silently until morning.

    Test-env carve-out: when unit tests monkeypatch ``builtins.input``,
    they're supplying the prompt response deterministically; honour the
    mock and skip the isatty guard. Only pure ``builtins.input`` under
    a non-TTY stdin triggers the fail-safe.
    """
    input_is_mocked = _builtins.input is not _ORIG_INPUT
    if not input_is_mocked and not sys.stdin.isatty():
        # Surface enough to diagnose after the fact WITHOUT ever
        # blocking. Prompt goes to stderr so a stdout tee doesn't
        # swallow it, and the [non-interactive] tag makes it grep-able
        # in the log.
        sys.stderr.write(
            f"[non-interactive] refusing to prompt: {prompt}\n"
            f"[non-interactive] auto-answered empty (subject will be "
            f"aborted). Fix the underlying issue, add a bypass flag "
            f"(e.g. --approve-confirmations, --fail-on-name-mismatch), "
            f"or re-run this subject interactively.\n")
        sys.stderr.flush()
        if _logger is not None:
            _logger.write_to_log(
                f"[non-interactive] auto-answered empty for prompt: "
                f"{prompt}\n")
        return ""
    response = input(prompt)
    if _logger is not None:
        _logger.write_to_log(f"{response}\n")
    return response
