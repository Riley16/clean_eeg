"""CLI entrypoint for the manual annotation review TUI.

Flow:
    1. Parse args, preflight the subject dir.
    2. Launch the prompt_toolkit TUI (see annotation_review.tui).
    3. On quit: show a plain-text approval gate listing every
       pending edit as ``<orig>`` -> ``<new>``. Operator confirms
       with 'y' to apply, anything else to discard.
    4. Apply pending edits (corruption-safe pass via
       clean_eeg.annotation_review.apply_edits).

Exit codes:
    0  reviewed cleanly, edits applied (or none)
    1  reviewed cleanly, operator discarded pending edits
    2  preflight failure (subject not ready for review)
    3  apply pass hit an error mid-file; original files preserved
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clean_eeg.annotation_review.apply_edits import (
    ApplyResult,
    apply_pending_edits,
)
from clean_eeg.annotation_review.controller import (
    AnnotationReviewController,
    PreflightFailure,
)
from clean_eeg.annotation_review.models import EditRecord


def _print_approval_gate(pending: list[EditRecord]) -> None:
    """Show every pending edit as ``<orig>`` -> ``<new>``. Nothing
    fancy -- plain stdout so the operator sees the full diff even
    on a terminal that doesn't support ANSI colors.
    """
    print(f"\n=== Pending edits for approval "
          f"({len(pending)} edit(s)) ===\n")
    for i, e in enumerate(pending, start=1):
        loc = f"{Path(e.file_path).name}  onset={e.onset_s:.2f}s"
        print(f"  [{i:3d}] {loc}")
        print(f"        <orig: {e.orig_text!r}>")
        print(f"        <new:  {e.new_text!r}>")


def _print_apply_summary(results: list[ApplyResult]) -> int:
    """Print outcome of the apply pass. Returns process exit code."""
    n_ok = sum(1 for r in results if r.succeeded)
    n_fail = len(results) - n_ok
    print(f"\n=== Apply summary ===")
    print(f"  files applied: {n_ok}")
    print(f"  files failed:  {n_fail}")
    for r in results:
        if not r.succeeded:
            print(f"    FAIL {r.file_path.name}: {r.error_message}")
    if n_fail:
        return 3
    return 0


def _prompt_apply(pending: list[EditRecord]) -> bool:
    """Interactive y/n. Rejects anything other than exact 'y' /
    'yes' to be explicit -- avoids the trap of a fat-finger 'n'
    accidentally applying edits the operator wanted to discard."""
    if not pending:
        return False
    _print_approval_gate(pending)
    resp = input("\nApply these edits to the EDFs? [y/N]: ").strip().lower()
    return resp in ("y", "yes")


def _prompt_mark_all_reviewed(unreviewed: list[Path]) -> bool:
    """Interactive Y/n with default YES. The operator who quits the
    TUI usually did so because they finished looking at everything --
    "no edits" and "unreviewed" should not be conflated. Default YES
    honors that intent; a fat-finger empty return marks reviewed
    (which is easily reversible: delete the tracker entries or use
    --include-reviewed on a re-run)."""
    n = len(unreviewed)
    filenames = ", ".join(p.name for p in unreviewed[:5])
    if n > 5:
        filenames += f", ... (+{n - 5} more)"
    print(f"\n{n} file(s) were reviewable but not explicitly marked "
          f"reviewed via 'n' during the session:")
    print(f"  {filenames}")
    resp = input(
        f"Mark all {n} as reviewed (they'll be skipped on future "
        f"annotation-review-eeg runs of this subject)? [Y/n]: "
    ).strip().lower()
    # Default YES: empty response is affirmative. Only an explicit 'n'
    # or 'no' rejects. This matches the operator's typical mental
    # model of "I quit because I'm done."
    return resp not in ("n", "no")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="annotation-review-eeg",
        description=(
            "Interactive TUI for manual annotation review after the "
            "auto-cleaning pass. Only runs on cleaned data (checks "
            "for deidentify.json). Journal + per-file reviewed "
            "tracker live under <subject>/.annotation_review/ so "
            "sessions survive crashes and skip already-reviewed "
            "files on restart."))
    p.add_argument("--subject-dir", type=Path, required=True,
                   help="Per-subject dir containing "
                        "<subject>/<subfolder>/*.edf and deidentify.json.")
    p.add_argument("--subfolder", type=str, default="clinical_eeg",
                   help="Per-subject sub-folder for EDFs "
                        "(default: clinical_eeg).")
    p.add_argument("--whitelist-path", type=Path, default=None,
                   help="Boilerplate whitelist JSON (per-site regex "
                        "fullmatch). Whitelisted annotations are "
                        "greyed in the scroll view. Press 'w' during "
                        "review to append the current annotation. "
                        "Default: auto-locate the standard whitelist "
                        "at data/annotation_boilerplate_whitelist.json. "
                        "Pass --no-whitelist to disable auto-load "
                        "entirely (empty whitelist).")
    p.add_argument("--no-whitelist", action="store_true",
                   help="Disable auto-loading the standard whitelist. "
                        "Use when you want to see every annotation "
                        "including known-boilerplate.")
    p.add_argument("--include-reviewed", action="store_true",
                   help="Include files listed in "
                        "<subject>/.annotation_reviewed_tracker. "
                        "Default: skip them (fresh sessions resume "
                        "where the previous one left off).")
    p.add_argument("--auto-apply", action="store_true",
                   help="Skip the y/N approval prompt at end of "
                        "review and apply pending edits automatically. "
                        "For scripted / unattended runs -- interactive "
                        "operators should leave this off.")
    p.add_argument("--preload-all", action="store_true",
                   help="Eagerly load every reviewable file's "
                        "annotations at startup with a tqdm progress "
                        "bar (one-time cost), then auto-skip files "
                        "whose annotations are entirely matched by "
                        "the whitelist or delete bucket. Best when "
                        "the operator reads faster than files load "
                        "lazily -- turns per-file wait into one "
                        "up-front pause.")
    p.add_argument("--show-whitelisted", action="store_true",
                   help="Keep whitelisted annotations visible in the "
                        "scroll view (greyed out, marked with '~'). "
                        "Default hides them entirely -- the operator "
                        "only sees + can navigate to annotations that "
                        "need review. Use this flag when you want to "
                        "audit what the whitelist is silencing.")
    p.add_argument("--rerun-annot-review", action="store_true",
                   help="Reset this subject's review state before "
                        "launching: deletes the reviewed-files tracker "
                        "and any pending-edit journal from a prior "
                        "aborted session, so the TUI treats every file "
                        "as fresh. PRESERVES the applied/ and "
                        "discarded/ audit trails inside .annotation_review/ "
                        "-- those record edits that already landed on disk "
                        "(applied) or were explicitly rejected "
                        "(discarded) in prior sessions. Use when you "
                        "aborted a review partway through and want to "
                        "restart cleanly from the first file.")
    args = p.parse_args(argv)

    # --rerun-annot-review: reset per-subject review state so the TUI
    # treats every file as fresh. Deletes tracker + pending-edit
    # journal; preserves applied/ + discarded/ audit trails inside
    # .annotation_review/. Runs BEFORE controller construction so the
    # controller sees the clean state.
    if args.rerun_annot_review:
        from clean_eeg.annotation_review.journal import reset_review_state
        inner = args.subject_dir / args.subfolder
        deleted = reset_review_state(inner)
        if not deleted:
            print(f"[rerun] {inner}: nothing to reset (no prior tracker "
                  f"or pending-edit journal). Proceeding with normal launch.",
                  file=sys.stderr)
        else:
            summary = ", ".join(f"{k} ({v})" for k, v in deleted.items())
            print(f"[rerun] {inner}: reset {summary}. Applied + discarded "
                  f"audit trails preserved.", file=sys.stderr)

    # Resolve whitelist path: explicit --whitelist-path wins; else
    # auto-locate the standard tracked whitelist unless --no-whitelist
    # disables. Was previously None-by-default -- meaning the TUI ran
    # with an EMPTY whitelist unless the operator remembered to pass
    # --whitelist-path, silently showing every '*Mark' / boilerplate
    # annotation the whitelist was supposed to hide.
    if args.whitelist_path is not None:
        resolved_wl_path = args.whitelist_path
    elif args.no_whitelist:
        resolved_wl_path = None
    else:
        from clean_eeg.paths import ANNOTATION_BOILERPLATE_WHITELIST_PATH
        resolved_wl_path = ANNOTATION_BOILERPLATE_WHITELIST_PATH
        # Loud visual confirmation so operators can see the auto-locate
        # is doing what they expect. Matches the pattern established by
        # count_annotations / sample_annotations.
        print(f"[TUI] applying whitelist: {resolved_wl_path}",
              file=sys.stderr)

    try:
        controller = AnnotationReviewController(
            args.subject_dir,
            subfolder=args.subfolder,
            whitelist_path=resolved_wl_path,
            respect_reviewed_tracker=not args.include_reviewed,
            preload_all=args.preload_all,
            hide_whitelisted=not args.show_whitelisted)
    except PreflightFailure as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    # Import the TUI lazily so unit tests can exercise the CLI's
    # error paths without prompt_toolkit's terminal detection getting
    # in the way.
    from clean_eeg.annotation_review.tui import build_review_app

    if controller.num_files_to_review == 0:
        # Distinguish auto-skipped-because-fully-whitelisted from
        # already-reviewed-by-human. When --preload-all just drained the
        # queue, the tracker entries are the ones IT wrote a moment ago,
        # so blaming the tracker would misdirect the operator (there's
        # nothing they missed reviewing -- there's nothing to review).
        n_wl = controller.num_files_auto_skipped_whitelist
        if n_wl and n_wl == controller.num_files:
            print(f"[info] all {controller.num_files} EDF file(s) had "
                  f"every annotation whitelisted -- nothing to review. "
                  f"Pass --show-whitelisted to inspect the whitelisted "
                  f"annotations, or edit the whitelist JSON if any of "
                  f"them shouldn't be silenced.")
        elif n_wl:
            print(f"[info] all {controller.num_files} EDF file(s) either "
                  f"had every annotation whitelisted ({n_wl} file(s)) or "
                  f"were already reviewed in a prior session. Nothing "
                  f"to review -- pass --include-reviewed to revisit "
                  f"previously-reviewed files, or --show-whitelisted to "
                  f"inspect the whitelisted ones.")
        else:
            print(f"[info] all {controller.num_files} EDF file(s) already "
                  f"reviewed (per {args.subject_dir.name}/"
                  ".annotation_reviewed_tracker). Nothing to do -- pass "
                  "--include-reviewed to re-review.")
        return 0

    app = build_review_app(controller)
    try:
        app.run()
    finally:
        controller.close()

    # Bulk-mark reviewed prompt: the operator quit -- typically that
    # means "I'm done looking at this subject", not "I only care about
    # files I explicitly pressed 'n' on". Prompting here closes the
    # gap where a review with zero edits (nothing needed changing)
    # would otherwise leave state=none in the audit. Default YES.
    # --auto-apply also implies auto-mark-reviewed for scripted runs.
    unreviewed = controller.unreviewed_reviewable_files()
    if unreviewed:
        should_mark = args.auto_apply or _prompt_mark_all_reviewed(unreviewed)
        if should_mark:
            marked = controller.mark_all_reviewable_files_reviewed()
            print(f"[ok] marked {len(marked)} file(s) as reviewed in the "
                  f"tracker.")
        else:
            print(f"[info] {len(unreviewed)} file(s) left unmarked in the "
                  f"tracker. Re-run annotation-review-eeg to revisit "
                  f"(or delete .annotation_reviewed_tracker to force "
                  f"a fresh pass).")

    pending = controller.pending_edits()
    if not pending:
        print("[info] no pending edits at quit -- nothing to apply.")
        return 0

    should_apply = args.auto_apply or _prompt_apply(pending)
    if not should_apply:
        rotated = controller.rotate_discarded()
        print(f"[info] {len(pending)} pending edit(s) discarded. "
              f"Audit trail at {rotated}." if rotated
              else "[info] no journal to rotate.")
        return 1

    results = apply_pending_edits(pending)
    exit_code = _print_apply_summary(results)
    if exit_code == 0:
        rotated = controller.rotate_applied()
        if rotated:
            print(f"[ok] session journal archived at {rotated}")
        # Refresh manifest hashes for the annotation sidecars we
        # modified so the transfer preflight's file-integrity check
        # stops flagging them. Signal-EDF hashes are left untouched --
        # apply never mutates signal bytes, so a mismatch there still
        # means legitimate corruption. Silent no-op if there's no
        # manifest (e.g. running annotation-review on a directory that
        # wasn't produced by the clean_subject_eeg pipeline).
        from clean_eeg.deidentify_manifest import (
            manifest_exists, refresh_annotation_sidecar_hashes)
        manifest_dir = args.subject_dir / args.subfolder
        if manifest_exists(manifest_dir):
            modified = [r.file_path for r in results if r.succeeded]
            try:
                changed = refresh_annotation_sidecar_hashes(
                    manifest_dir, modified)
                if changed:
                    print(f"[ok] refreshed {len(changed)} sidecar hash(es) "
                          f"in {manifest_dir}/deidentify.json so transfer "
                          f"preflight sees a consistent manifest.")
            except Exception as e:
                # Non-fatal: the edits already landed on disk. Warn the
                # operator so they can re-run the manifest step manually
                # rather than silently ship a stale hash to the transfer
                # step.
                print(f"[warn] manifest refresh failed ({type(e).__name__}: "
                      f"{e}). Edits applied; re-generate deidentify.json "
                      f"before transferring.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
