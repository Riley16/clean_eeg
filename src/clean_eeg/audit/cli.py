"""CLI: audit a single subject dir or every subject subfolder of a parent.

  audit-subject-eeg SUBJECT_DIR [OPTIONS]
  audit-subject-eeg --parent PARENT_DIR [OPTIONS]

Options:
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

from clean_eeg.audit.annotations import extract_annotations
from clean_eeg.audit.subject import (
    AUDIT_JSON_FILENAME,
    _discover_edf_files,
    audit_subject,
)


DEFAULT_VOCAB_WHITELIST = Path("data/annotation_vocab_whitelist.json")


def _load_vocab_whitelist(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def _print_summary(audit: dict, out=sys.stdout) -> None:
    print(f"\n=== Audit: {audit['subject_dir']} ===", file=out)
    print(f"Subject code: {audit.get('subject_code')}", file=out)
    print(f"Files: {audit['n_files']}   Mode: {audit['mode']}   "
          f"Overall: {audit['overall_status'].upper()}", file=out)
    for name, r in audit["checks"].items():
        marker = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[r["status"]]
        print(f"  [{marker}] {name}", file=out)
        for issue in r.get("issues", []):
            print(f"          - {issue}", file=out)


def _print_annotations(subject_dir: Path, out=sys.stdout) -> None:
    print(f"\n--- All annotations in {subject_dir.name} ---", file=out)
    for p in _discover_edf_files(subject_dir):
        anns = extract_annotations(p)
        if not anns:
            continue
        print(f"  {p.name}:", file=out)
        for a in anns:
            print(f"    {a['onset']:>10.3f}s "
                  f"(dur={a['duration']!r})  {a['text']!r}", file=out)


def _print_unique_header_values(audit: dict, out=sys.stdout) -> None:
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


def _print_unique_signal_headers(audit: dict, out=sys.stdout) -> None:
    uni = audit["checks"].get("signal_header_uniformity", {})
    sigs = uni.get("signatures", {})
    print(f"\n--- Signal-header signatures ({len(sigs)} unique) ---", file=out)
    for sig_id, info in sigs.items():
        print(f"  {sig_id}: {info['n_files']} file(s), "
              f"e.g. {info['files'][:3]}", file=out)
        for ch in info.get("channels", []):
            print(f"      {ch}", file=out)


def _always_print_warnings(audit: dict, out=sys.stdout) -> None:
    """Always echo name-dictionary matches and any fail-status checks
    even under --quiet — these are the load-bearing PHI signals."""
    scan = audit["checks"].get("annotation_phi_scan", {})
    matches = scan.get("matched_tokens", {})
    if matches:
        print(f"\n[!] Annotation name-dictionary matches — {len(matches)} token(s):",
              file=out)
        for token, hits in matches.items():
            print(f"    '{token}' × {len(hits)}", file=out)
            for h in hits[:3]:
                print(f"        {h['file']} @ {h['onset']}s: {h['text']!r}", file=out)


def _run_one_subject(subject_dir: Path, args) -> dict:
    audit_exists = (subject_dir / AUDIT_JSON_FILENAME).exists()
    if audit_exists and not args.force:
        print(f"[skip] {subject_dir.name}: {AUDIT_JSON_FILENAME} exists "
              f"(pass --force to re-run all checks; hash-consistency check still runs)")

    vocab = _load_vocab_whitelist(args.vocab_whitelist)
    audit = audit_subject(
        subject_dir,
        force=args.force,
        annotation_only=args.annotation_only,
        skip_hashes=args.skip_hashes,
        vocab_whitelist=vocab,
    )

    if not args.quiet:
        _print_summary(audit)
    _always_print_warnings(audit)  # never suppressed
    if args.print_annot:
        _print_annotations(subject_dir)
    if args.print_edf_header:
        _print_unique_header_values(audit)
    if args.print_edf_signal_header:
        _print_unique_signal_headers(audit)

    if not args.no_notebook:
        from clean_eeg.audit.notebook import render_audit_notebook
        try:
            render_audit_notebook(subject_dir)
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
    p.add_argument("--force", action="store_true")
    p.add_argument("--annotation-only", action="store_true")
    p.add_argument("--skip-hashes", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-notebook", action="store_true")
    p.add_argument("--print-annot", action="store_true")
    p.add_argument("--print-edf-header", action="store_true")
    p.add_argument("--print-edf-signal-header", action="store_true")
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
