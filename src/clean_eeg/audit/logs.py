"""Pipeline log-file anomaly surfacing.

The pipeline's ``PipelineLogger`` writes a PHI-scrubbed ``log.out`` to
the output directory. Transferred subject folders should include this
file for provenance. The audit greps it for:

  - ``WARNING:`` lines (gaps, overlaps, header inconsistencies, ...)
  - ``ERROR:`` lines
  - Annotation-redaction events ("Subject protected health information
    detected in EDF <field>; redacted value: ...") — the pipeline
    doing its job, but a human should eyeball the redacted values to
    confirm the redaction was correct.

Because the log is PHI-scrubbed, the extracted lines are safe to
embed in ``edf_audit.json`` verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path


LOG_FILENAME = "log.out"

_WARNING_RE = re.compile(r"^WARNING:", re.IGNORECASE)
_ERROR_RE = re.compile(r"^ERROR:", re.IGNORECASE)
# The pipeline emits (from [clean_subject_eeg.py:181]):
#   Subject protected health information detected in EDF <field>;
#   redacted value: "<value>". Alert the data analysis team.
_REDACTION_RE = re.compile(
    r'Subject protected health information detected in EDF (\S+); '
    r'redacted value: "(.+)"\.'
)
# The pipeline emits either of these when a file fails de-id and is
# skipped from the run (from [clean_subject_eeg.py] _load_edf_metadata
# and the per-file cleaning try/except):
#   ERROR: Failed to load EDF file <name>: <exception>
#   ERROR: Failed to de-identify EDF file <name>: <exception>
# The <name> is what the file was called BEFORE any pipeline rename,
# so it's the original untimestamped filename (which is also what
# ends up sitting in the transferred output dir as a not-cleaned file).
_FAILED_DEID_RE = re.compile(
    r"^ERROR:\s+Failed to (?:load|de-identify) EDF file\s+([^\s:]+)",
    re.IGNORECASE,
)


# When we look ahead past a matched ERROR line to grab the exception
# message, cap the search at this many lines. The pipeline prints:
#     ERROR: Failed to load EDF file X:
#     <blank>
#     <exception message — usually 1 line, occasionally multi-line>
#     <blank>
#     Stack trace (for the data team):
# We stop at "Stack trace" / "Skipping this file" or on the next
# ERROR:/WARNING: line, whichever comes first — but the cap defends
# against a runaway log where the delimiter is missing.
_ERR_LOOKAHEAD_LINES = 30


def _extract_error_message_after(lines: list[str], start_idx: int) -> str:
    """After a matched ERROR line at ``lines[start_idx]``, return the
    exception summary that the pipeline emits (blank, message, blank,
    Stack trace ...). Empty string if the shape doesn't match.
    """
    msg_parts: list[str] = []
    for line in lines[start_idx + 1 : start_idx + 1 + _ERR_LOOKAHEAD_LINES]:
        s = line.rstrip()
        if s.startswith("Stack trace") or s.startswith("Skipping this file") \
                or s.startswith("Partially-processed"):
            break
        if _ERROR_RE.match(s) or _WARNING_RE.match(s):
            break
        if s.strip():
            msg_parts.append(s.strip())
    return " | ".join(msg_parts)


def check_log_file(log_path: str | Path | None) -> dict:
    """Scan a pipeline ``log.out`` for warnings, errors, and
    annotation-redaction events.

    ``log_path`` may be ``None`` or a non-existent path — the audit
    returns ``warn`` (missing log means missing provenance, but the
    transfer isn't necessarily broken).

    Status:
      - ``pass`` — log present, no warnings/errors/redactions
      - ``warn`` — log missing, OR warnings-only, OR redactions-only
      - ``fail`` — any ``ERROR:`` line present
    """
    if log_path is None or not Path(log_path).exists():
        return {
            "check": "log_file",
            "status": "warn",
            "log_path": str(log_path) if log_path is not None else None,
            "log_present": False,
            "n_warnings": 0, "n_errors": 0, "n_redactions": 0,
            "n_failed_deid_files": 0,
            "warnings": [], "errors": [], "redactions": [],
            "failed_deid_files": [],
            "issues": [f"No pipeline '{LOG_FILENAME}' present — "
                       "provenance/warnings from cleaning are unavailable"],
        }

    warnings: list[dict] = []
    errors: list[dict] = []
    redactions: list[dict] = []
    failed_deid: list[dict] = []
    # Read the whole file and normalize CR into LF before line-splitting.
    # tqdm writes progress bars with \r-terminated updates when its
    # stream isn't a TTY, so a naive readline() would see something like
    # "Loading files...\rERROR: Failed to load X:" as ONE line and the
    # ^ERROR: anchor would fail to match. Splitting on \r|\n\r|\n keeps
    # the ERROR line separate regardless of how the pipeline's tee
    # captured stderr.
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    all_lines = text.split("\n")
    for i, line in enumerate(all_lines, start=1):
        stripped = line.rstrip("\n")
        if _WARNING_RE.match(stripped):
            warnings.append({"line_number": i, "text": stripped})
        if _ERROR_RE.match(stripped):
            errors.append({"line_number": i, "text": stripped})
        m_fail = _FAILED_DEID_RE.match(stripped)
        if m_fail:
            failed_deid.append({
                "line_number": i,
                "filename": m_fail.group(1),
                "text": stripped,
                "error_message": _extract_error_message_after(all_lines, i - 1),
            })
        m = _REDACTION_RE.search(stripped)
        if m:
            redactions.append({
                "line_number": i,
                "field": m.group(1),
                "redacted_value": m.group(2),
            })

    issues: list[str] = []
    if errors or failed_deid:
        status = "fail"
        if failed_deid:
            names = sorted({f["filename"] for f in failed_deid})
            issues.append(
                f"{len(failed_deid)} file(s) failed pipeline de-identification "
                f"and were SKIPPED from the run: {names}"
            )
        # If there are additional ERROR lines beyond the failed-deid
        # summary, surface a count too (some ERRORs are unrelated to
        # per-file failure and worth flagging separately).
        other_errors = len(errors) - len(failed_deid)
        if other_errors > 0:
            issues.append(f"{other_errors} additional ERROR line(s) in pipeline log")
    elif warnings or redactions:
        status = "warn"
        if warnings:
            issues.append(f"{len(warnings)} WARNING line(s) in pipeline log")
        if redactions:
            issues.append(
                f"{len(redactions)} annotation redaction(s) by pipeline — "
                "human should verify each redacted_value is correct"
            )
    else:
        status = "pass"

    return {
        "check": "log_file",
        "status": status,
        "log_path": str(log_path),
        "log_present": True,
        "n_warnings": len(warnings),
        "n_errors": len(errors),
        "n_redactions": len(redactions),
        "n_failed_deid_files": len(failed_deid),
        "warnings": warnings,
        "errors": errors,
        "redactions": redactions,
        "failed_deid_files": failed_deid,
        "issues": issues,
    }
