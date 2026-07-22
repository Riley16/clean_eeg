"""Detect the subject's name from EDF headers.

The name the operator would otherwise type on the command line is already
in the file: EDF+ packs bytes 8-87 of the main header (``patient_id``) as
``code sex birthdate name``, with spaces inside the name replaced by
underscores. Nihon Kohden exports follow this loosely, and technicians use
several orderings (``SMITH^JOHN^P``, ``Smith, John P``, ``John Smith``),
so parsing is heuristic and the result is always shown to the operator
rather than silently trusted.

Header bytes are read via :func:`print_edf_header.read_main_header` rather
than pyedflib so detection also works on the broken NK exports that
pyedflib refuses to open, and so no signal data is touched.

This module also writes the ID-to-name CSV used for a manual review pass
(``detect-edf-names``). That CSV holds PHI in the clear — see
:func:`write_name_csv_row`.
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from typing import List, Optional, Tuple

from clean_eeg.names import PersonalName
from clean_eeg.print_edf_header import (
    _is_annotation_stub,
    read_main_header,
)

# Default location for the ID->name review CSV. Deliberately outside the
# repo and outside any pipeline output directory (see write_name_csv_row).
DEFAULT_NAME_CSV = os.path.join(os.path.expanduser("~"), "sens_data",
                                "detected_names.csv")

NAME_CSV_COLUMNS = [
    "subject_code", "patient_code", "raw_patient_id",
    "detected_first", "detected_middle", "detected_last",
    "parse_status", "n_edf_files", "example_file", "input_path",
    "scanned_at", "clean_eeg_version",
]

# Parse outcomes, also used as the CSV parse_status value.
STATUS_OK = "ok"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_PLACEHOLDER = "placeholder"
STATUS_INITIAL_ONLY = "initial_only"
STATUS_DISAGREE = "disagree"
STATUS_NO_FILES = "no_edf_files"

# Values technicians / prior de-identification runs leave behind that carry
# no name. 'X' is the EDF+ "unknown" placeholder.
_PLACEHOLDER_RE = re.compile(r"^[\sX]*$", re.IGNORECASE)
_SUBJECT_CODE_RE = re.compile(r"^R1\d{3}[A-Z]$", re.IGNORECASE)

_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]")

# Titles and generational suffixes technicians append (HL7 PID fields 5/6,
# e.g. "DOE^JANE^MARIE^^MS"). Dropped so they aren't mistaken for middle
# names — they are not PHI and would only add noise to the deny-list.
_TITLES_AND_SUFFIXES = {
    "mr", "mrs", "ms", "miss", "dr", "prof",
    "jr", "sr", "ii", "iii", "iv", "v",
    "md", "do", "phd", "rn",
}


def _alpha_len(token: str) -> int:
    """Number of alphabetic characters, ignoring dots/hyphens/apostrophes."""
    return sum(c.isalpha() for c in token)


def is_placeholder_name(raw: str) -> bool:
    """True when the header name field carries no actual name.

    Covers the EDF+ 'X' placeholder, empty fields, values already replaced
    by a subject code from a previous de-identification run, and the
    generic strings pyedflib writes for anonymized files.
    """
    if not raw:
        return True
    s = raw.strip()
    if not s:
        return True
    if _PLACEHOLDER_RE.match(s):
        return True
    if _SUBJECT_CODE_RE.match(s):
        return True
    if s.lower().replace("_", " ") in {"anonymized", "patient x", "unknown",
                                       "redacted", "redacted-name"}:
        return True
    return False


def _titlecase(token: str) -> str:
    """Normalize ALL-CAPS header tokens to title case, leaving mixed-case
    tokens (``McDonald``, ``O'Brien``) alone."""
    if token.isupper():
        # str.title() mangles apostrophes ("O'BRIEN" -> "O'Brien" is fine,
        # but "O'BRIEN".title() gives "O'Brien" — capitalize each
        # apostrophe/hyphen-delimited part explicitly to be safe.
        return re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), token.lower())
    return token


def _split_tokens(text: str) -> List[str]:
    """Split a name string into tokens, dropping empties and non-alpha junk."""
    parts = re.split(r"[\s_]+", text.strip())
    tokens = [p.strip(" ,") for p in parts
              if p.strip(" ,") and _ALPHA_TOKEN_RE.search(p)]
    return [t for t in tokens
            if t.lower().strip(".") not in _TITLES_AND_SUFFIXES]


def parse_patient_name(raw: str) -> Tuple[Optional[PersonalName], str]:
    """Parse an EDF header name string into first/middle/last.

    Ordering is not standardized across recording systems, so dispatch on
    the delimiter:

    - ``^``-delimited (HL7 PID style, ``SMITH^JOHN^P``)  -> Last^First^Middle
    - comma-delimited (``Smith, John P``)                -> Last, First Middle
    - otherwise (underscores treated as spaces)          -> First [Middle] Last

    Returns ``(name, status)``. ``name`` is None unless status is
    :data:`STATUS_OK`.
    """
    if raw is None or is_placeholder_name(raw):
        return None, STATUS_PLACEHOLDER

    text = raw.strip()

    if "^" in text:
        fields = [f for f in (p.strip() for p in text.split("^")) if f]
        last_part = fields[0] if fields else ""
        rest = " ".join(fields[1:])
        last_tokens = _split_tokens(last_part)
        rest_tokens = _split_tokens(rest)
    elif "," in text:
        last_part, _, rest = text.partition(",")
        last_tokens = _split_tokens(last_part)
        rest_tokens = _split_tokens(rest)
    else:
        tokens = _split_tokens(text)
        if len(tokens) < 2:
            return None, STATUS_AMBIGUOUS
        # First [Middle...] Last
        last_tokens = tokens[-1:]
        rest_tokens = tokens[:-1]

    if not last_tokens or not rest_tokens:
        return None, STATUS_AMBIGUOUS

    last = _titlecase(last_tokens[0])
    first = _titlecase(rest_tokens[0])
    middles = [_titlecase(t) for t in rest_tokens[1:]]

    # A bare initial as the first name gives the redactor nothing to work
    # with — add_subject_name_detectors drops single-letter tokens from the
    # deny-list and fuzzy targets, so auto-filling "L. Smith" would leave
    # the real first name unredacted everywhere it appears.
    if _alpha_len(first) < 2 or _alpha_len(last) < 2:
        return None, STATUS_INITIAL_ONLY

    return PersonalName(first_name=first, middle_names=middles,
                        last_name=last), STATUS_OK


def extract_name_field(patient_id: str) -> str:
    """Pull the name portion out of a raw EDF ``patient_id`` field.

    EDF+ layout is ``code sex birthdate name`` (4+ space-separated fields,
    name may itself contain underscores). Files that don't follow the
    layout get the whole field treated as the name.
    """
    if not patient_id:
        return ""
    fields = patient_id.strip().split()
    if len(fields) >= 4:
        return " ".join(fields[3:])
    return patient_id.strip()


def extract_patient_code(patient_id: str) -> str:
    """First field of ``patient_id`` — the hospital MRN / patient code."""
    if not patient_id:
        return ""
    fields = patient_id.strip().split()
    return fields[0] if fields else ""


def list_edf_files(input_path: str, recursive: bool = True) -> List[str]:
    """Full paths of the subject's EDF files, excluding annotation stubs
    written by a previous in-place de-identification run.

    Recurses into subdirectories by default, so a subject folder is found
    regardless of how the export nested the files (e.g.
    ``R1001P/R1001P/day1/*.edf``). Pass ``recursive=False`` to read only
    the files directly in ``input_path``.
    """
    if not os.path.isdir(input_path):
        return [input_path] if input_path.lower().endswith(".edf") else []
    out = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(input_path):
            for fname in filenames:
                if not fname.lower().endswith(".edf"):
                    continue
                if _is_annotation_stub(fname):
                    continue
                out.append(os.path.join(dirpath, fname))
        return sorted(out)
    for fname in sorted(os.listdir(input_path)):
        if not fname.lower().endswith(".edf"):
            continue
        if _is_annotation_stub(fname):
            continue
        out.append(os.path.join(input_path, fname))
    return out


class DetectionResult:
    """Outcome of scanning one subject directory."""

    def __init__(self, name, status, per_file, input_path):
        self.name: Optional[PersonalName] = name
        self.status: str = status
        # {filepath: raw patient_id string}
        self.per_file: dict = per_file
        self.input_path: str = input_path

    @property
    def ok(self) -> bool:
        return self.name is not None and self.status == STATUS_OK

    @property
    def n_files(self) -> int:
        return len(self.per_file)

    @property
    def raw_patient_id(self) -> str:
        """Representative raw patient_id (first file scanned)."""
        for raw in self.per_file.values():
            return raw
        return ""

    @property
    def patient_code(self) -> str:
        return extract_patient_code(self.raw_patient_id)

    @property
    def example_file(self) -> str:
        for path in self.per_file:
            return os.path.basename(path)
        return ""

    def reason(self) -> str:
        """Operator-facing explanation. Contains no name text."""
        return {
            STATUS_OK: "name detected",
            STATUS_AMBIGUOUS: ("header name could not be split into first and "
                               "last name (too few name tokens)"),
            STATUS_PLACEHOLDER: ("header name field is empty or already "
                                 "de-identified"),
            STATUS_INITIAL_ONLY: ("header name gives only an initial for the "
                                  "first or last name, which is not enough "
                                  "for reliable redaction"),
            STATUS_DISAGREE: ("EDF files in this directory carry different "
                              "patient names — they may not all be from the "
                              "same subject"),
            STATUS_NO_FILES: "no EDF files found to read a name from",
        }.get(self.status, self.status)

    def name_tokens(self) -> List[str]:
        """All detected name tokens, for PHI registration with the logger."""
        if self.name is None:
            tokens = []
        else:
            tokens = ([self.name.first_name] + list(self.name.middle_names)
                      + [self.name.last_name])
        # Include the raw header text so the raw form is scrubbed too.
        for raw in self.per_file.values():
            tokens.extend(_split_tokens(extract_name_field(raw)))
        return [t for t in tokens if t]


def detect_subject_name(input_path: str,
                        recursive: bool = True) -> DetectionResult:
    """Read the patient name from every EDF in ``input_path`` and parse it.

    By default recurses into subdirectories, treating every EDF found
    beneath ``input_path`` as belonging to the same subject. All
    non-placeholder files must agree on the name; a disagreement means the
    tree may hold more than one subject, so nothing is detected (the
    pipeline's own ``_check_subject_name_consistency`` reports the details
    later).
    """
    per_file = {}
    for path in list_edf_files(input_path, recursive=recursive):
        header = read_main_header(path)
        patient_id = header.get("patient_id", "")
        if not isinstance(patient_id, str):
            patient_id = ""
        per_file[path] = patient_id

    if not per_file:
        return DetectionResult(None, STATUS_NO_FILES, per_file, input_path)

    name_fields = {extract_name_field(raw) for raw in per_file.values()}
    real_names = {n for n in name_fields if not is_placeholder_name(n)}

    if not real_names:
        return DetectionResult(None, STATUS_PLACEHOLDER, per_file, input_path)
    if len(real_names) > 1:
        return DetectionResult(None, STATUS_DISAGREE, per_file, input_path)

    name, status = parse_patient_name(real_names.pop())
    return DetectionResult(name, status, per_file, input_path)


# ---------------------------------------------------------------------------
# ID -> name CSV (review pass)
# ---------------------------------------------------------------------------

def _clean_eeg_version() -> str:
    try:
        from importlib.metadata import version
        return version("clean_eeg")
    except Exception:
        return "unknown"


def write_name_csv_row(result: DetectionResult,
                       subject_code: str = "",
                       csv_path: str = DEFAULT_NAME_CSV,
                       scanned_at: Optional[str] = None) -> str:
    """Append (or update) one ID-to-name row in the review CSV.

    WARNING: this file holds PHI in the clear. It is written outside the
    repository and outside any pipeline output directory, is never copied
    alongside de-identified EDFs, and its contents must never be echoed
    into ``log.out``. The directory is created 0700 and the file 0600.

    Rows are keyed on ``input_path`` — re-scanning the same directory
    updates its row in place instead of appending a duplicate.
    """
    csv_path = os.path.abspath(os.path.expanduser(csv_path))
    parent = os.path.dirname(csv_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, mode=0o700, exist_ok=True)

    name = result.name
    row = {
        "subject_code": subject_code,
        "patient_code": result.patient_code,
        "raw_patient_id": result.raw_patient_id,
        "detected_first": name.first_name if name else "",
        "detected_middle": " ".join(name.middle_names) if name else "",
        "detected_last": name.last_name if name else "",
        "parse_status": result.status,
        "n_edf_files": result.n_files,
        "example_file": result.example_file,
        "input_path": os.path.abspath(result.input_path),
        "scanned_at": scanned_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clean_eeg_version": _clean_eeg_version(),
    }

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            rows = [r for r in csv.DictReader(f)]

    replaced = False
    for i, existing in enumerate(rows):
        if existing.get("input_path") == row["input_path"]:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NAME_CSV_COLUMNS,
                               extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in NAME_CSV_COLUMNS})
    os.replace(tmp_path, csv_path)
    os.chmod(csv_path, 0o600)
    return csv_path


# ---------------------------------------------------------------------------
# CLI:  detect-edf-names
# ---------------------------------------------------------------------------

def _subject_dirs(path: str, per_subject_dir: bool) -> List[str]:
    """Subject directories to scan.

    Default: ``[path]`` — every EDF found anywhere beneath ``path`` is
    treated as one subject (``detect_subject_name`` recurses).

    ``per_subject_dir``: batch mode for a parent folder — each immediate
    subdirectory that has any EDF beneath it is scanned as its own
    subject. Use this when ``path`` holds many subjects side by side."""
    if not per_subject_dir:
        return [path]
    if not os.path.isdir(path):
        return [path]
    subs = []
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        if os.path.isdir(full) and list_edf_files(full, recursive=True):
            subs.append(full)
    return subs


def main():
    parser = argparse.ArgumentParser(
        prog="detect-edf-names",
        description="Read subject names from EDF headers and record them in a "
                    "CSV (ID -> name) for a manual review pass. Does not "
                    "modify any EDF file. The CSV contains PHI in the clear — "
                    "keep it out of shared/backed-up locations.")
    parser.add_argument("paths", nargs="+",
                        help="Subject EDF directory (or a single .edf file). "
                             "EDF files are found recursively beneath it and "
                             "treated as one subject.")
    parser.add_argument("--per_subject_dir", "--recursive", action="store_true",
                        dest="per_subject_dir",
                        help="Batch mode: treat each immediate subdirectory "
                             "of the given path as a separate subject (each "
                             "still searched recursively for its EDFs). Use "
                             "when the path holds many subjects side by side. "
                             "(--recursive is a deprecated alias.)")
    parser.add_argument("--subject_code", type=str, default="",
                        help="Subject code to record alongside the detected "
                             "name (only meaningful for a single directory)")
    parser.add_argument("--csv", type=str, default=DEFAULT_NAME_CSV,
                        help=f"CSV to write/update (default: {DEFAULT_NAME_CSV})")
    parser.add_argument("--no_csv", action="store_true",
                        help="Print results only; do not write the CSV")
    args = parser.parse_args()

    targets = []
    for path in args.paths:
        if not os.path.exists(path):
            print(f"ERROR: path does not exist: {path}", file=sys.stderr)
            return 1
        targets.extend(_subject_dirs(path, args.per_subject_dir))

    if args.subject_code and len(targets) > 1:
        print("ERROR: --subject_code is only valid when scanning a single "
              "subject directory.", file=sys.stderr)
        return 1

    n_ok = 0
    csv_path = None
    for target in targets:
        result = detect_subject_name(target)
        if result.ok:
            n_ok += 1
        # Console output is for the operator sitting at the terminal; it
        # intentionally shows the parsed split so they can eyeball it.
        if result.ok:
            middle = " ".join(result.name.middle_names)
            detail = (f"first={result.name.first_name!r} "
                      f"middle={middle!r} last={result.name.last_name!r}")
        else:
            detail = f"{result.status} — {result.reason()}"
        print(f"{target}  [{result.n_files} EDF files]  {detail}")
        if not args.no_csv:
            csv_path = write_name_csv_row(result,
                                          subject_code=args.subject_code,
                                          csv_path=args.csv)

    print(f"\nScanned {len(targets)} director{'y' if len(targets) == 1 else 'ies'}; "
          f"{n_ok} with a usable name.")
    if csv_path:
        print(f"Wrote ID->name CSV (contains PHI): {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
