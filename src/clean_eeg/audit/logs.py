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
            "warnings": [], "errors": [], "redactions": [],
            "issues": [f"No pipeline '{LOG_FILENAME}' present — "
                       "provenance/warnings from cleaning are unavailable"],
        }

    warnings: list[dict] = []
    errors: list[dict] = []
    redactions: list[dict] = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.rstrip("\n")
            if _WARNING_RE.match(stripped):
                warnings.append({"line_number": i, "text": stripped})
            if _ERROR_RE.match(stripped):
                errors.append({"line_number": i, "text": stripped})
            m = _REDACTION_RE.search(stripped)
            if m:
                redactions.append({
                    "line_number": i,
                    "field": m.group(1),
                    "redacted_value": m.group(2),
                })

    issues: list[str] = []
    if errors:
        status = "fail"
        issues.append(f"{len(errors)} ERROR line(s) in pipeline log")
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
        "warnings": warnings,
        "errors": errors,
        "redactions": redactions,
        "issues": issues,
    }
