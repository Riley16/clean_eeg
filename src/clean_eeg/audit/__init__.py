"""Per-subject post-transfer audit of de-identified EDF files.

See TODO.md ("Per-Subject Post-Transfer Audit") for the full check
inventory. Each check function in `checks.py` returns a plain dict so
the results serialize directly into `edf_audit.json`.
"""

from clean_eeg.audit.subject import (
    AUDIT_JSON_FILENAME,
    AuditInterruptedError,
    IN_PROGRESS_FILENAME,
    audit_subject,
)

__all__ = [
    "AUDIT_JSON_FILENAME",
    "AuditInterruptedError",
    "IN_PROGRESS_FILENAME",
    "audit_subject",
]
