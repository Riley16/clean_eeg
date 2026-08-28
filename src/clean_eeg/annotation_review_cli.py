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
    on a terminal that doesn't support ANSI colors."""
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
                        "review to append the current annotation.")
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
    args = p.parse_args(argv)

    try:
        controller = AnnotationReviewController(
            args.subject_dir,
            subfolder=args.subfolder,
            whitelist_path=args.whitelist_path,
            respect_reviewed_tracker=not args.include_reviewed,
            preload_all=args.preload_all)
    except PreflightFailure as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    # Import the TUI lazily so unit tests can exercise the CLI's
    # error paths without prompt_toolkit's terminal detection getting
    # in the way.
    from clean_eeg.annotation_review.tui import build_review_app

    if controller.num_files_to_review == 0:
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
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
