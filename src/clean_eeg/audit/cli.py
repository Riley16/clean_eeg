"""CLI: audit a single subject dir or every subject subfolder of a parent.

  audit-subject-eeg SUBJECT_DIR [OPTIONS]
  audit-subject-eeg --parent PARENT_DIR [OPTIONS]

Options:
  --output-dir PATH        Write edf_audit.{json,ipynb,html} here instead
                           of the subject dir (avoids polluting fixtures
                           or read-only archives). In --parent mode,
                           each subject's outputs land in
                           OUTPUT_DIR/<subject_name>/.
  --force                  Re-run all checks (else: skip if audit exists;
                           hash-consistency step always runs)
  --annotation-only        Only run the annotation-dictionary scan
                           (for fast whitelist-seeding iteration)
  --skip-hashes            Skip the SHA-256 manifest (fast for slow FS)
  --quiet                  Suppress terminal output (JSON + notebook still written)
  --no-notebook            Skip notebook + HTML rendering
  --print-annot            Print every annotation (subject to future
                           boilerplate-filtering; today: prints all)
  --print-edf-header       Print unique main-header values across subject files
  --print-edf-signal-header  Print unique signal-header values across subject files
  --vocab-whitelist PATH   JSON list of tokens to exempt from the name scan
                           (default: data/annotation_vocab_whitelist.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import re

from clean_eeg.audit.annotations import extract_annotations
from clean_eeg.audit.select import select_files
from clean_eeg.audit.subject import (
    AUDIT_JSON_FILENAME,
    _discover_edf_files,
    audit_subject,
)
from clean_eeg.paths import DATA_DIR


# Matches EDF+ timekeeping-shaped strings the pipeline treats as
# non-PHI ([clean_subject_eeg.py:167]): empty, all-whitespace, pure
# numeric (with optional sign / decimal), or single-char lines.
_BOILERPLATE_RE = re.compile(r"^\s*[+-]?\d*\.?\d*\s*$")


def _looks_like_boilerplate(text: str) -> bool:
    return not text or len(text.strip()) < 2 or bool(_BOILERPLATE_RE.match(text))


# Absolute path (via clean_eeg.paths.DATA_DIR) so the default whitelist
# is found regardless of the user's cwd. The file is shipped in-repo
# and grows over time as operators run more audits.
DEFAULT_VOCAB_WHITELIST = DATA_DIR / "annotation_vocab_whitelist.json"


def _load_vocab_whitelist(path: Path | None) -> tuple[set[str], str]:
    """Return ``(tokens, status_message)``. The status message tells
    the operator which whitelist was used and how many tokens loaded —
    surfaces silent whiffs (e.g., wrong path, malformed JSON).
    """
    if path is None:
        return set(), "vocab whitelist: none provided"
    if not path.exists():
        return set(), f"vocab whitelist: {path} does not exist (using empty set)"
    tokens = set(json.loads(path.read_text()))
    return tokens, f"vocab whitelist: {len(tokens)} token(s) from {path}"


def _print_summary(audit: dict, out=None) -> None:
    out = out or sys.stdout
    print(f"\n=== Audit: {audit['subject_dir']} ===", file=out)
    print(f"Subject code: {audit.get('subject_code')}", file=out)
    print(f"Files: {audit['n_files']}   Mode: {audit['mode']}   "
          f"Overall: {audit['overall_status'].upper()}", file=out)
    for name, r in audit["checks"].items():
        marker = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[r["status"]]
        print(f"  [{marker}] {name}", file=out)
        for issue in r.get("issues", []):
            print(f"          - {issue}", file=out)


def _print_annotations(subject_dir: Path,
                       *,
                       sample_n: int | None = None,
                       verbosity: int = 0,
                       out=None) -> None:
    """Print annotations across the subject's EDFs.

    - ``sample_n=None`` prints from every file; otherwise picks that
      many via ``select_files`` (always includes first + last).
    - ``verbosity < 3``: skip timekeeping-shaped boilerplate.
    - ``verbosity >= 3``: full verbatim, no filter.
    """
    out = out or sys.stdout
    files = _discover_edf_files(subject_dir)
    picks = files if sample_n is None else select_files(files, n_files=sample_n)
    filter_boilerplate = verbosity < 3

    hdr = ("all annotations" if sample_n is None
           else f"{len(picks)}-file random sample of annotations")
    filt = "" if not filter_boilerplate else "  (boilerplate filtered; -vvv for full)"
    print(f"\n--- {hdr} in {subject_dir.name}{filt} ---", file=out)
    for p in picks:
        anns = extract_annotations(p)
        if filter_boilerplate:
            anns = [a for a in anns if not _looks_like_boilerplate(a["text"])]
        if not anns:
            continue
        print(f"  {p.name}:", file=out)
        for a in anns:
            print(f"    {a['onset']:>10.3f}s "
                  f"(dur={a['duration']!r})  {a['text']!r}", file=out)


def _print_unique_header_values(audit: dict, out=None) -> None:
    out = out or sys.stdout
    residue = audit["checks"].get("header_phi_residue", {})
    pids = set(residue.get("patient_ids_by_file", {}).values())
    startdates = set(residue.get("startdates_by_file", {}).values())
    print("\n--- Unique main-header values ---", file=out)
    print(f"  patient_id ({len(pids)} unique):", file=out)
    for v in sorted(pids):
        print(f"    {v!r}", file=out)
    print(f"  startdate ({len(startdates)} unique):", file=out)
    for v in sorted(startdates):
        print(f"    {v!r}", file=out)


def _print_unique_signal_headers(audit: dict, out=None) -> None:
    out = out or sys.stdout
    uni = audit["checks"].get("signal_header_uniformity", {})
    sigs = uni.get("signatures", {})
    print(f"\n--- Signal-header signatures ({len(sigs)} unique) ---", file=out)
    for sig_id, info in sigs.items():
        print(f"  {sig_id}: {info['n_files']} file(s), "
              f"e.g. {info['files'][:3]}", file=out)
        for ch in info.get("channels", []):
            print(f"      {ch}", file=out)


def _always_print_warnings(audit: dict, out=None) -> None:
    """Always echo name-dictionary matches and any pipeline redactions
    into annotations, even under --quiet — these are the load-bearing
    PHI signals the auditor cares about most."""
    out = out or sys.stdout
    scan = audit["checks"].get("annotation_phi_scan", {})
    matches = scan.get("matched_tokens", {})
    if matches:
        print(f"\n[!] Annotation name-dictionary matches — {len(matches)} token(s):",
              file=out)
        for token, hits in matches.items():
            print(f"    '{token}' × {len(hits)}", file=out)
            for h in hits[:3]:
                print(f"        {h['file']} @ {h['onset']}s: {h['text']!r}", file=out)

    log = audit["checks"].get("log_file", {})
    ann_redactions = [r for r in log.get("redactions", [])
                      if r.get("field") == "annotation"]
    if ann_redactions:
        print(f"\n[!] Pipeline redacted {len(ann_redactions)} annotation(s) during "
              "de-identification — human should verify each redacted_value:",
              file=out)
        for r in ann_redactions:
            print(f"    log line {r['line_number']}: {r['redacted_value']!r}", file=out)


def _run_one_subject(subject_dir: Path, args) -> dict:
    # Per-subject output dir: if --output-dir was given, nest under it
    # by subject-folder name (so --parent mode doesn't collide multiple
    # subjects into a single dir). Otherwise write alongside the EDFs.
    if args.output_dir is not None:
        out_dir = args.output_dir / subject_dir.name
    else:
        out_dir = subject_dir

    audit_exists = (out_dir / AUDIT_JSON_FILENAME).exists()
    if audit_exists and not args.force:
        print(f"[skip] {subject_dir.name}: {out_dir / AUDIT_JSON_FILENAME} exists "
              f"(pass --force to re-run all checks; hash-consistency check still runs)")

    vocab, vocab_status = _load_vocab_whitelist(args.vocab_whitelist)
    print(f"[audit] {vocab_status}")
    audit = audit_subject(
        subject_dir,
        output_dir=out_dir,
        force=args.force,
        annotation_only=args.annotation_only,
        skip_hashes=args.skip_hashes,
        vocab_whitelist=vocab,
    )

    if not args.quiet:
        _print_summary(audit)
    _always_print_warnings(audit)  # never suppressed
    if args.print_annot:
        _print_annotations(subject_dir,
                           sample_n=args.print_annot_sample_n,
                           verbosity=args.verbose)
    if args.print_edf_header:
        _print_unique_header_values(audit)
    if args.print_edf_signal_header:
        _print_unique_signal_headers(audit)

    if not args.no_notebook:
        from clean_eeg.audit.notebook import render_audit_notebook
        try:
            render_audit_notebook(subject_dir, output_dir=out_dir,
                                  n_channel_plot=args.n_channel_plot,
                                  n_files_plot=args.n_files_plot)
        except Exception as e:
            print(f"[!] Notebook rendering failed for {subject_dir.name}: {e}",
                  file=sys.stderr)

    return audit


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audit-subject-eeg",
        description="Per-subject audit of de-identified EDFs (PHI-focused).",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("subject_dir", nargs="?", type=Path,
                   help="Single subject directory to audit.")
    g.add_argument("--parent", type=Path,
                   help="Parent directory — audit every subject subfolder.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Write edf_audit.{json,ipynb,html} here instead "
                        "of alongside the EDFs. In --parent mode, per-subject "
                        "outputs land in OUTPUT_DIR/<subject_name>/.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--annotation-only", action="store_true")
    p.add_argument("--skip-hashes", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-notebook", action="store_true")
    p.add_argument("--print-annot", action="store_true")
    p.add_argument("--print-annot-sample-n", type=int, default=None,
                   help="Print annotations from a randomized N-file sample "
                        "(always includes the first and last files). "
                        "Default: all files.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase --print-annot detail. Default filters "
                        "timekeeping-shaped boilerplate; -vvv prints "
                        "every annotation verbatim.")
    p.add_argument("--print-edf-header", action="store_true")
    p.add_argument("--print-edf-signal-header", action="store_true")
    p.add_argument("--n-channel-plot", type=int, default=5,
                   help="Channels per EEG snippet plot in the notebook.")
    p.add_argument("--n-files-plot", type=int, default=4,
                   help="Files to plot in the notebook EEG snippet section.")
    p.add_argument("--vocab-whitelist", type=Path,
                   default=DEFAULT_VOCAB_WHITELIST,
                   help="JSON list of tokens to exempt from the name scan.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.parent:
        subjects = sorted(p for p in args.parent.iterdir() if p.is_dir())
        if not subjects:
            print(f"No subdirectories found in {args.parent}", file=sys.stderr)
            return 1
        overall_fail = False
        for s in subjects:
            audit = _run_one_subject(s, args)
            if audit.get("overall_status") == "fail":
                overall_fail = True
        return 1 if overall_fail else 0

    audit = _run_one_subject(args.subject_dir, args)
    return 1 if audit.get("overall_status") == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
