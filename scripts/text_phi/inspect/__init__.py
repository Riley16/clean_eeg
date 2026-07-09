"""CSV inspection tools: REDCap metadata parsing, column-name mapping, and
PHI-safe per-column statistics for schema design.

Nothing here needs to be run by the redaction pipeline — these are one-shot
utilities the data owner runs locally on their PHI-bearing CSV to produce
metadata safe to share with a collaborator (or an assistant) for schema
design. No raw values are ever emitted to the output JSON.
"""
