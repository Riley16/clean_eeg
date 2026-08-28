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
                           hash-consistency step always runs). Also
                           clears a stale ``edf_audit.in_progress``
                           sentinel left by a previous interrupted run.
  --annotation-only        Only run the annotation-dictionary scan
                           (for fast whitelist-seeding iteration)
  --hash-mode {fast,full,none}
                           fast (default): hash header + 2 s at start,
                           middle, and end of each file (catches
                           tampering, truncation, endpoint bit-rot at
                           O(MB) per file). full: SHA-256 the whole
                           file (O(GB) per file). none: skip hashing.
  --skip-hashes            DEPRECATED alias for --hash-mode none.
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

from clean_eeg.annotation_boilerplate import (
    BoilerplateWhitelistError,
    load_whitelist,
)
from clean_eeg.audit.annotations import extract_annotations
from clean_eeg.audit.hashes import VALID_HASH_MODES
from clean_eeg.audit.select import select_files
from clean_eeg.audit.subject import (
    AUDIT_JSON_FILENAME,
    AuditInterruptedError,
    _discover_edf_files,
    audit_subject,
)
from clean_eeg.paths import ANNOTATION_BOILERPLATE_WHITELIST_PATH, DATA_DIR


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


def _load_boilerplate_whitelist(path: Path | None):
    """Return ``(BoilerplateWhitelist, status_message)`` — the audit's
    per-site + shared regex list for annotation-level pre-filtering
    (annotations that fullmatch a listed pattern are skipped before
    the name-dict scan runs). Malformed JSON returns an empty
    whitelist rather than crashing the audit, but surfaces the error
    loudly in the status message so operators notice."""
    if path is None or not path.exists():
        wl = load_whitelist(None)
        return wl, "boilerplate whitelist: none loaded"
    try:
        wl = load_whitelist(path)
    except BoilerplateWhitelistError as e:
        wl = load_whitelist(None)
        return wl, f"boilerplate whitelist: MALFORMED ({e}) — using empty"
    n_shared = len(wl.shared)
    n_per_site = sum(len(v) for v in wl.per_site.values())
    return wl, (f"boilerplate whitelist: {n_shared} shared + {n_per_site} "
                f"per-site pattern(s) from {path}")


SUMMARY_SKIP_CHECKS = frozenset({
    # annotation_phi_scan issues are the same tokens the
    # 'Annotation name-dictionary matches' block in
    # _always_print_warnings prints in more detail — the summary line
    # would duplicate that info without adding anything. The check
    # still runs, still lands in edf_audit.json, and still contributes
    # to overall_status (so a FAIL is still visible at the top).
    "annotation_phi_scan",
    # annotation_review_state is a state summary (always status="pass");
    # its content is reported by _always_print_warnings as a ✓ / ~ line.
    "annotation_review_state",
})


def _print_summary(audit: dict, out=None,
                   print_subject_header: bool = True,
                   show_passes: bool = False) -> None:
    out = out or sys.stdout
    if print_subject_header:
        # Skipped in --parent mode where a bigger visual banner has
        # already announced the subject; showing the single-line
        # header here as well would duplicate the path immediately
        # below the banner.
        print(f"\n=== Audit: {audit['subject_dir']} ===", file=out)
    print(f"Subject code: {audit.get('subject_code')}", file=out)
    print(f"Files: {audit['n_files']}   Mode: {audit['mode']}   "
          f"Overall: {audit['overall_status'].upper()}", file=out)
    n_hidden_passes = 0
    for name, r in audit["checks"].items():
        if name in SUMMARY_SKIP_CHECKS:
            continue
        # [OK] lines are noise for the common "everything passed" case
        # — hide them by default so WARN/FAIL stands out. -v surfaces
        # them again when the operator wants to confirm nothing got
        # silently skipped.
        if r["status"] == "pass" and not show_passes:
            n_hidden_passes += 1
            continue
        marker = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[r["status"]]
        print(f"  [{marker}] {name}", file=out)
        for issue in r.get("issues", []):
            print(f"          - {issue}", file=out)
    if n_hidden_passes and not show_passes:
        print(f"  ({n_hidden_passes} passing check(s) hidden — pass -v to show)",
              file=out)


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
    """Dump every free-text field in the EDF main header, grouped by
    unique value. The EDF+ spec has exactly four such fields
    (see [print_edf_header.py:42-45](../print_edf_header.py#L42-L45)):

        patient_id     80-byte subject/sex/birthdate/name subfield string
        recording_id   80-byte startdate/admin/technician/equipment string
        startdate      DD.MM.YY
        starttime      HH.MM.SS

    All four can carry residual PHI if the pipeline missed a field, so
    the operator needs eyes on the unique values across every file.
    Grouping by unique value keeps the dump compact for uniform
    subject dirs (usually 1 patient_id + 1 recording_id + N distinct
    startdates + N distinct starttimes).
    """
    out = out or sys.stdout
    residue = audit["checks"].get("header_phi_residue", {})
    pids = set(residue.get("patient_ids_by_file", {}).values())
    startdates = set(residue.get("startdates_by_file", {}).values())
    starttimes = set(residue.get("starttimes_by_file", {}).values())
    recording_ids = set(residue.get("recording_ids_by_file", {}).values())
    print("\n--- Unique main-header values ---", file=out)
    print(f"  patient_id ({len(pids)} unique):", file=out)
    for v in sorted(pids):
        print(f"    {v!r}", file=out)
    print(f"  recording_id ({len(recording_ids)} unique):", file=out)
    for v in sorted(recording_ids):
        print(f"    {v!r}", file=out)
    print(f"  startdate ({len(startdates)} unique):", file=out)
    for v in sorted(startdates):
        print(f"    {v!r}", file=out)
    print(f"  starttime ({len(starttimes)} unique):", file=out)
    for v in sorted(starttimes):
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


def _critical_finding_lines(audit: dict) -> list[str]:
    """Collect the load-bearing "you must look at this" findings that
    demand the banner at the very top and bottom of the audit output.
    Each returned string is one line inside the banner box.

    File paths are printed as ABSOLUTE paths (resolved against the
    audit's ``subject_dir``) so an operator scrolling scrollback or
    copy-pasting into a shell can act on the output without having to
    reconstruct the parent directory.

    Critical categories:
      - Files the pipeline explicitly failed to de-identify and skipped
        (from ``log_file.failed_deid_files``).
      - Files sitting in the transferred subject dir that don't match
        the pipeline's timestamped rename pattern (from
        ``filename_convention.unrenamed_files``).
      - Files with a recording_id year outside the expected
        de-identified range (from
        ``header_phi_residue.recording_id_years_by_file``) — signals
        the file bypassed the header-shift step.
    """
    checks = audit.get("checks", {})
    subject_dir_str = audit.get("subject_dir") or "."
    subject_dir = Path(subject_dir_str).resolve()

    def _abs(name: str) -> str:
        return str((subject_dir / name).resolve())

    lines: list[str] = []
    log = checks.get("log_file", {})
    failed = log.get("failed_deid_files") or []
    if failed:
        names = sorted({f["filename"] for f in failed})
        lines.append(f"{len(failed)} FILE(S) PIPELINE FAILED TO DE-IDENTIFY (SKIPPED):")
        for name in names:
            lines.append(f"  - {_abs(name)}")

    fname_check = checks.get("filename_convention", {})
    unrenamed = fname_check.get("unrenamed_files") or []
    if unrenamed:
        lines.append(
            f"{len(unrenamed)} FILE(S) DO NOT MATCH PIPELINE RENAME PATTERN "
            "(BYPASSED CLEANING):")
        for name in unrenamed:
            lines.append(f"  - {_abs(name)}")

    residue = checks.get("header_phi_residue", {})
    year_range = residue.get("expected_year_range") or []
    if year_range:
        rid_years = residue.get("recording_id_years_by_file", {}) or {}
        off_recid = {name: yr for name, yr in rid_years.items()
                     if yr is not None and not (year_range[0] <= yr <= year_range[1])}
        if off_recid:
            lines.append(
                f"{len(off_recid)} FILE(S) HAVE REAL RECORDING YEAR IN "
                "recording_id (HEADER-SHIFT BYPASSED):")
            for name, yr in sorted(off_recid.items()):
                lines.append(f"  - {_abs(name)}: year {yr}")
    return lines


def _print_critical_banner(audit: dict, *, label: str, out=None) -> None:
    """Emit a big UPPERCASE banner listing critical findings, or nothing
    if there are none. ``label`` distinguishes TOP vs BOTTOM placement."""
    out = out or sys.stdout
    lines = _critical_finding_lines(audit)
    if not lines:
        return
    width = max(70, max(len(ln) for ln in lines) + 4)
    bar = "!" * width
    print(f"\n{bar}", file=out)
    header = f"!! CRITICAL AUDIT FINDINGS ({label}) — MANUAL REVIEW REQUIRED !!"
    print(header.center(width, "!"), file=out)
    print(bar, file=out)
    for ln in lines:
        print(f"!! {ln}", file=out)
    print(bar + "\n", file=out)


def _collect_flagged_filenames(audit: dict) -> dict[str, list[str]]:
    """Return ``{filename: [reason, ...]}`` for every file the audit
    flagged as critical (pipeline-failed, unrenamed, or off-year
    recording_id). The same file can be flagged by multiple categories
    — reasons are accumulated so the header dump can show all of them.

    Log-derived reasons include the exception message the pipeline
    emitted right after the ERROR: line — that's usually the exact
    pyedflib / repair-pass error that made the file unloadable, and
    seeing it inline saves the operator a trip back to log.out.
    """
    checks = audit.get("checks", {})
    flagged: dict[str, list[str]] = {}

    for f in checks.get("log_file", {}).get("failed_deid_files") or []:
        msg = f.get("error_message") or ""
        reason = "log.out reports pipeline failed to load/de-identify"
        if msg:
            reason += f": {msg}"
        flagged.setdefault(f["filename"], []).append(reason)

    for name in checks.get("filename_convention", {}).get("unrenamed_files") or []:
        flagged.setdefault(name, []).append(
            "filename lacks pipeline's rename suffix")

    residue = checks.get("header_phi_residue", {})
    year_range = residue.get("expected_year_range") or []
    if year_range:
        for name, yr in (residue.get("recording_id_years_by_file") or {}).items():
            if yr is not None and not (year_range[0] <= yr <= year_range[1]):
                flagged.setdefault(name, []).append(
                    f"recording_id year {yr} outside expected "
                    f"[{year_range[0]}, {year_range[1]}]")
    return flagged


def _delete_unclean_files(audit: dict, subject_dir: Path,
                          *, auto_confirm: bool, out=None
                          ) -> tuple[list[Path], list[Path]]:
    """Delete every file the audit flagged as critical (pipeline-
    failed / unrenamed / off-year recording_id). READ-ONLY-BY-DEFAULT
    audit primitive; only invoked when the operator passes
    ``--delete-unclean``.

    Returns ``(deleted, skipped)`` — resolved absolute paths. ``skipped``
    covers "file doesn't exist" cases (the flagged filename didn't map
    to a real path). Every deletion is announced on ``out`` so the
    scrollback carries the receipt.

    Interactive confirmation: unless ``auto_confirm`` is True, prompts
    with the exact list about to be deleted and requires an explicit
    ``DELETE N FILES`` string match (N = the exact count). Cheap
    ceremony that makes accidental invocation impossible under
    autocomplete or typo.
    """
    out = out or sys.stdout
    flagged = _collect_flagged_filenames(audit)
    if not flagged:
        return [], []

    paths = sorted((subject_dir / n).resolve() for n in flagged)
    n = len(paths)
    print(f"\n[!] --delete-unclean requested. {n} file(s) queued for "
          "PERMANENT deletion (also removes their _annotations.edf "
          "sidecars if present):", file=out)
    for p, name in zip(paths, sorted(flagged)):
        reasons = flagged[name]
        print(f"    - {p}", file=out)
        for r in reasons:
            print(f"        reason: {r}", file=out)

    if not auto_confirm:
        need = f"DELETE {n} FILES"
        got = input(
            f"\nType exactly {need!r} to confirm, anything else aborts: "
        ).strip()
        if got != need:
            print("Aborted; no files deleted.", file=out)
            return [], []

    deleted: list[Path] = []
    skipped: list[Path] = []
    for p in paths:
        # Delete both the main EDF and its _annotations.edf sidecar if
        # one exists. Silent no-op on missing sidecar -- inline-mode
        # cleaned files don't have sidecars.
        for target in (p, Path(str(p).replace(".edf", "_annotations.edf"))):
            if not target.exists():
                if target == p:
                    skipped.append(target)
                continue
            target.unlink()
            deleted.append(target)
            print(f"    deleted: {target}", file=out)
    return deleted, skipped


def _print_failed_deid_headers(audit: dict, subject_dir: Path,
                               *, redact_phi: bool = False,
                               out=None) -> None:
    """Dump the EDF header for every file the audit's critical-findings
    banner flagged — pipeline-failed (from ``log_file``), unrenamed
    (from ``filename_convention``), and off-year recording_id (from
    ``header_phi_residue``). Read via the byte-level parser so files
    pyedflib refuses to open still yield a header. Read-only — NEVER
    attempts to re-clean.

    ``redact_phi=False`` (default) shows the raw header AS IT EXISTS
    ON DISK POST-CLEANING. The whole point of dumping a failed-file
    header is diagnostic — if we mask the PHI-carrying fields, we
    hide the exact evidence needed to decide what went wrong. Opt
    into masking (``--redact-header-dump``) only when the audit
    output is going to be shared with someone who shouldn't see raw
    PHI in the case that cleaning failed.

    Reasons accumulate per file so an operator seeing e.g. the same
    file flagged as BOTH unrenamed AND off-year gets the full context
    in one dump section instead of two disjoint sections.
    """
    from clean_eeg.print_edf_header import print_header

    out = out or sys.stdout
    flagged = _collect_flagged_filenames(audit)
    if not flagged:
        return
    print(f"\n[!] Dumping headers for {len(flagged)} flagged file(s) "
          "(no re-clean attempted):", file=out)
    if not redact_phi:
        print("    NOTE: PHI-carrying header fields shown UNREDACTED so a "
              "failed cleaning is diagnosable. Pass --redact-header-dump "
              "before sharing this output.", file=out)
    for name in sorted(flagged):
        # Absolute path so an operator running the audit from an arbitrary
        # cwd (or copy-pasting from a scrollback) can act on the output
        # without having to reconstruct the parent directory.
        path = (subject_dir / name).resolve()
        reasons = flagged[name]
        print(f"\n--- header dump: {path} ---", file=out)
        for r in reasons:
            print(f"    reason: {r}", file=out)
        if not path.exists():
            print(f"    (file not present at {path}; skipping)", file=out)
            continue
        try:
            print_header(str(path), redact_phi=redact_phi, out=out)
        except Exception as e:
            print(f"    ERROR while reading header: {e}", file=out)


def _always_print_warnings(audit: dict, out=None, *,
                           show_annotation_flags: bool = False,
                           hide_annotation_flags: bool = False) -> None:
    """Always echo name-dictionary matches and any pipeline redactions
    into annotations, even under --quiet — these are the load-bearing
    PHI signals the auditor cares about most.

    When manual annotation review is complete (``annotation_review_state
    == 'complete'``) both sections are suppressed by default and replaced
    with a single ✓ line, because the operator has already inspected
    every annotation and any remaining matches are known-safe. Pass
    ``show_annotation_flags=True`` to render them anyway (useful when
    re-auditing a subject whose review may have been done against an
    older whitelist).

    ``hide_annotation_flags`` unconditionally suppresses the phi-scan
    matches block AND the pipeline annotation-redactions block, even
    when review state is "none" or "partial". Used by the cleaner's
    end-of-run auto-audit when it's about to launch the TUI: listing
    the flagged annotations in the audit output is redundant with the
    TUI, which the operator is about to open on the exact same file.
    Wins over ``show_annotation_flags`` if both are True (the caller
    asked twice for suppression, so honor it).
    """
    out = out or sys.stdout
    review = audit["checks"].get("annotation_review_state", {})
    review_state = review.get("state", "none")
    suppress = (hide_annotation_flags
                or (review_state == "complete" and not show_annotation_flags))

    if suppress:
        if review_state == "complete":
            # Standalone re-audit after review: celebrate the completed
            # review + tell operators how to see the flags anyway.
            print(f"\n[✓] Manual annotation review complete: "
                  f"{review.get('n_reviewed', 0)} file(s) reviewed, "
                  f"{review.get('n_edits_applied', 0)} edit(s) applied across "
                  f"{review.get('n_applied_sessions', 0)} session(s). "
                  f"Flagged-annotation section suppressed — pass "
                  f"--show-annotation-flags to render it.", file=out)
        else:
            # hide_annotation_flags forced (pre-TUI auto-audit from the
            # cleaner). Different message — a review has NOT happened
            # yet; we're just not repeating what the TUI is about to
            # show verbatim.
            print(f"\n[i] Annotation flags suppressed for this render "
                  f"(--hide-annotation-flags). The TUI you're about to "
                  f"open shows every annotation directly; re-run "
                  f"audit-subject-eeg after review to see the flag "
                  f"summary with the review-complete banner.", file=out)
        return

    if review_state == "partial":
        n_r = review.get("n_reviewed", 0)
        n_c = review.get("n_annotation_carriers", 0)
        print(f"\n[~] Manual annotation review in progress: "
              f"{n_r}/{n_c} file(s) reviewed. Flagged annotations below "
              f"may already be resolved for the reviewed files.", file=out)

    scan = audit["checks"].get("annotation_phi_scan", {})
    matches = scan.get("matched_tokens", {})
    if matches:
        print(f"\n[!] Annotation name-dictionary matches — {len(matches)} token(s):",
              file=out)
        for token, hits in matches.items():
            print(f"    '{token}' × {len(hits)}", file=out)
            # Every hit is shown (not just the first 3) — the operator
            # audits each context to decide whether it's real PHI or a
            # false positive. Annotation text goes at the START of the
            # line, then a tab, then file/onset — so file names align
            # in a column and scanning across many hits is fast.
            for h in hits:
                print(f"        {h['text']!r}\t{h['file']} @ {h['onset']}s",
                      file=out)

    log = audit["checks"].get("log_file", {})
    ann_redactions = [r for r in log.get("redactions", [])
                      if r.get("field") == "annotation"]
    if ann_redactions:
        print(f"\n[!] Pipeline redacted {len(ann_redactions)} annotation(s) during "
              "de-identification — human should verify each redacted_value:",
              file=out)
        for r in ann_redactions:
            print(f"    log line {r['line_number']}: {r['redacted_value']!r}", file=out)


def _resolve_interrupted_prior(err: AuditInterruptedError,
                               subject_dir: Path) -> str:
    """Decide how to handle a prior interrupted audit for ``subject_dir``.

    Returns one of:
      - ``'wipe'``  — clear the sentinel and re-run this subject.
      - ``'skip'``  — leave the sentinel in place; skip this subject.
      - ``'quit'``  — abort the whole batch (only meaningful under
                      ``--parent``; single-subject mode treats it as
                      the same exit as ``skip`` after printing).

    Interactive (stdin is a TTY): prompt the operator once per subject.
    Non-interactive (batch scheduler, redirect): refuse the subject
    with instructions and continue — silently proceeding could destroy
    provenance the operator wants to inspect.
    """
    print(f"\n[!] {subject_dir.name}: previous audit was interrupted "
          f"(started {err.started_at or '?'}, host {err.hostname or '?'}, "
          f"pid {err.pid or '?'})", file=sys.stderr, flush=True)
    if not sys.stdin.isatty():
        print(f"    non-interactive session — skipping {subject_dir.name}. "
              f"Re-run with --force to wipe the sentinel and audit from "
              f"scratch, or delete {err.sentinel_path} manually.",
              file=sys.stderr, flush=True)
        return "skip"

    while True:
        resp = input(f"    [w]ipe and re-run, [s]kip, [q]uit? ").strip().lower()
        if resp in ("w", "wipe"):
            return "wipe"
        if resp in ("s", "skip", ""):
            return "skip"
        if resp in ("q", "quit"):
            return "quit"
        print("    please answer w, s, or q.", file=sys.stderr, flush=True)


def _print_subject_banner(subject_dir: Path, out=None) -> None:
    """Loud visual separator printed at the top of each subject in
    ``--parent`` mode — makes it unambiguous where one subject's
    output ends and the next begins. Format:

        ============================================================
        ============================================================
        === Audit: <subject_dir> ===
        ============================================================
        ============================================================
    """
    out = out or sys.stdout
    bar = "=" * 60
    print(bar, file=out)
    print(bar, file=out)
    print(f"=== Audit: {subject_dir} ===", file=out)
    print(bar, file=out)
    print(bar, file=out)


def _run_one_subject(subject_dir: Path, args,
                     *, printed_banner: bool = False) -> dict | None:
    """Audit one subject. Returns the audit dict, or ``None`` if the
    operator (or batch mode) chose to skip this subject because a prior
    run was interrupted.

    ``printed_banner`` = True when the caller (parent-mode loop)
    already printed a subject-header banner; ``_print_summary`` then
    skips its own single-line ``=== Audit ===`` header to avoid
    duplicating the subject path immediately below the banner.
    """
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
              f"(pass --force to re-run all checks; hash-consistency check still runs)",
              flush=True)

    vocab, vocab_status = _load_vocab_whitelist(args.vocab_whitelist)
    boilerplate, bp_status = _load_boilerplate_whitelist(
        ANNOTATION_BOILERPLATE_WHITELIST_PATH)
    print(f"[audit] {vocab_status}", flush=True)
    print(f"[audit] {bp_status}", flush=True)
    print(f"[audit] auditing {subject_dir}", flush=True)

    def _do_audit(force: bool) -> dict:
        # No streaming progress callback — the audit is fast enough on
        # real subjects (~10 s) that per-check readout added noise
        # without helping. The end-of-run summary + warnings block
        # still show status per check and detailed content for the
        # checks that produce actionable output.
        return audit_subject(
            subject_dir,
            output_dir=out_dir,
            force=force,
            annotation_only=args.annotation_only,
            skip_hashes=args.skip_hashes,
            hash_mode=args.hash_mode,
            vocab_whitelist=vocab,
            boilerplate_whitelist=boilerplate,
            progress=None,
        )

    try:
        audit = _do_audit(args.force)
    except AuditInterruptedError as e:
        decision = _resolve_interrupted_prior(e, subject_dir)
        if decision == "skip":
            return None
        if decision == "quit":
            print("Operator quit at interrupted subject; aborting batch.",
                  file=sys.stderr, flush=True)
            raise SystemExit(1)
        # 'wipe' — clear sentinel by re-running with force=True.
        audit = _do_audit(True)

    # Critical-findings banner at the TOP of the audit output — before
    # the summary — so an operator scrolling through many subjects can
    # immediately spot the ones that need manual review. Same banner
    # repeats at the BOTTOM (see below) so critical findings survive
    # the tail-of-terminal view when the middle scrolls off.
    _print_critical_banner(audit, label="TOP")
    if not args.quiet:
        _print_summary(audit,
                       print_subject_header=not printed_banner,
                       show_passes=args.verbose >= 1)
    _always_print_warnings(audit,  # never suppressed by --quiet
                           show_annotation_flags=args.show_annotation_flags,
                           hide_annotation_flags=args.hide_annotation_flags)
    # Read-only header dump for every file the pipeline failed on.
    # NEVER attempts to re-run the cleaner — the operator inspects
    # the header and decides what to do (repair, exclude, escalate).
    _print_failed_deid_headers(audit, subject_dir,
                               redact_phi=args.redact_header_dump)
    if args.print_annot:
        _print_annotations(subject_dir,
                           sample_n=args.print_annot_sample_n,
                           verbosity=args.verbose)
    if args.print_edf_header:
        _print_unique_header_values(audit)
    if args.print_edf_signal_header:
        _print_unique_signal_headers(audit)

    notebook_rendered = False
    if not args.no_notebook:
        from clean_eeg.audit.notebook import render_audit_notebook
        try:
            render_audit_notebook(subject_dir, output_dir=out_dir,
                                  n_channel_plot=args.n_channel_plot,
                                  n_files_plot=args.n_files_plot)
            notebook_rendered = True
        except Exception as e:
            print(f"[!] Notebook rendering failed for {subject_dir.name}: {e}",
                  file=sys.stderr)

    # Point the operator at the full results — the terminal output is
    # a scannable summary, but the JSON has every per-file detail and
    # the HTML render carries plots + inline docstrings.
    _print_full_results_footer(out_dir, notebook_rendered=notebook_rendered)

    # Optional destructive action: PERMANENTLY delete the flagged
    # files. Runs BEFORE the bottom banner so the banner still lists
    # every critical finding for provenance -- the deleted-files log
    # then documents which of those findings actually got wiped.
    if args.delete_unclean:
        _delete_unclean_files(audit, subject_dir,
                              auto_confirm=args.yes_delete_unclean)

    # Same banner as at the top, repeated at the BOTTOM so critical
    # findings can't be missed after the middle of the audit scrolls
    # off in a long tail-of-terminal view. Reflects the pre-delete
    # audit state on purpose -- documents what was flagged, even for
    # entries that just got deleted above.
    _print_critical_banner(audit, label="BOTTOM")

    return audit


def _print_full_results_footer(out_dir: Path, *, notebook_rendered: bool) -> None:
    """Emit the trailing 'full results at' pointer block. Uses absolute
    paths so the operator can copy-paste them regardless of cwd."""
    from clean_eeg.audit.notebook import HTML_FILENAME, NOTEBOOK_FILENAME
    out_dir = out_dir.resolve()
    print("\nFull results:")
    print(f"  JSON:     {out_dir / AUDIT_JSON_FILENAME}")
    if notebook_rendered:
        print(f"  Notebook: {out_dir / NOTEBOOK_FILENAME}")
        print(f"  HTML:     {out_dir / HTML_FILENAME}")


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
    p.add_argument("--force", action="store_true",
                   help="Re-run all checks (overrides idempotent skip) "
                        "and clear any stale 'edf_audit.in_progress' "
                        "sentinel left by a previous interrupted run.")
    p.add_argument("--annotation-only", action="store_true")
    p.add_argument("--hash-mode", choices=VALID_HASH_MODES, default="fast",
                   help="fast (default): hash header + 2 s at start, middle, "
                        "and end of each file. full: SHA-256 the whole file. "
                        "none: skip hashing entirely.")
    p.add_argument("--skip-hashes", action="store_true",
                   help="DEPRECATED alias for --hash-mode none.")
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
    p.add_argument("--delete-unclean", "--delete_unclean",
                   dest="delete_unclean", action="store_true",
                   help="PERMANENTLY DELETE every file the audit flags in "
                        "the critical banner (pipeline-failed, unrenamed, or "
                        "off-year recording_id). Also removes matching "
                        "_annotations.edf sidecars. Requires typing the "
                        "exact confirmation string 'DELETE N FILES' unless "
                        "--yes-delete-unclean is also passed. Use for "
                        "cleaning up subjects where partially-cleaned files "
                        "were transferred and now need to be removed from "
                        "the server-side archive. NOT reversible — verify "
                        "the banner list is what you want deleted BEFORE "
                        "passing this flag.")
    p.add_argument("--yes-delete-unclean", "--yes_delete_unclean",
                   dest="yes_delete_unclean", action="store_true",
                   help="Skip the interactive 'DELETE N FILES' confirmation "
                        "for --delete-unclean. Intended for headless / SSH "
                        "batch runs after the operator has verified the "
                        "flagged list in an earlier interactive audit.")
    p.add_argument("--redact-header-dump", "--redact_header_dump",
                   dest="redact_header_dump", action="store_true",
                   help="Mask patient_id / recording_id / startdate / "
                        "starttime in the failed-file header dump. Default "
                        "OFF because the dump is diagnostic — masking hides "
                        "the exact evidence of what went wrong. Enable this "
                        "flag ONLY when the audit output will be shared "
                        "with someone who shouldn't see raw PHI in the case "
                        "that cleaning failed.")
    p.add_argument("--n-channel-plot", type=int, default=5,
                   help="Channels per EEG snippet plot in the notebook.")
    p.add_argument("--n-files-plot", type=int, default=4,
                   help="Files to plot in the notebook EEG snippet section.")
    p.add_argument("--vocab-whitelist", type=Path,
                   default=DEFAULT_VOCAB_WHITELIST,
                   help="JSON list of tokens to exempt from the name scan.")
    p.add_argument("--show-annotation-flags", "--show_annotation_flags",
                   dest="show_annotation_flags", action="store_true",
                   help="Render the annotation name-dictionary matches and "
                        "pipeline-redaction blocks even when the manual "
                        "annotation review is complete. Default hides them "
                        "in that case (a ✓ line is printed instead) since "
                        "the operator has already inspected every "
                        "annotation. Use when re-auditing a subject whose "
                        "review may have been done against an older "
                        "vocab / boilerplate whitelist.")
    p.add_argument("--hide-annotation-flags", "--hide_annotation_flags",
                   dest="hide_annotation_flags", action="store_true",
                   help="Unconditionally suppress the annotation flag "
                        "blocks (phi-scan matches + pipeline redactions), "
                        "even when review state is 'none' or 'partial'. "
                        "Used by the cleaner's end-of-run auto-audit when "
                        "it's about to launch the TUI: the TUI shows every "
                        "annotation directly, so listing flagged ones in "
                        "the audit output is redundant. Wins over "
                        "--show-annotation-flags if both are passed.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.parent:
        subjects = sorted(p for p in args.parent.iterdir() if p.is_dir())
        if not subjects:
            print(f"No subdirectories found in {args.parent}", file=sys.stderr)
            return 1
        overall_fail = False
        skipped: list[str] = []
        for i, s in enumerate(subjects):
            # 5 blank lines between subjects — visually separates the
            # end of one subject's audit (which can be dozens of lines
            # for a fragmented one) from the start of the next.
            if i > 0:
                print("\n" * 5, end="")
            _print_subject_banner(s)
            audit = _run_one_subject(s, args, printed_banner=True)
            if audit is None:
                skipped.append(s.name)
                continue
            if audit.get("overall_status") == "fail":
                overall_fail = True
        if skipped:
            print(f"\n[!] Skipped {len(skipped)} subject(s) with interrupted "
                  f"prior audits: {skipped}. Re-run with --force to wipe and "
                  f"audit from scratch.", file=sys.stderr, flush=True)
        # Any skip or fail is a non-zero exit — batch schedulers rely on
        # this to notice partial completions.
        return 1 if (overall_fail or skipped) else 0

    audit = _run_one_subject(args.subject_dir, args)
    if audit is None:
        return 1
    return 1 if audit.get("overall_status") == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
