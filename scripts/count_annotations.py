"""Count annotations + words across every subject dir under a parent.

Reports total annotations, per-subject stats, and estimated manual-
review reading time at a given WPM. Purely informational -- no
mutation. Intended for scoping a manual-annotation-review pass
before committing operator hours to it.

Usage:
    python scripts/count_annotations.py \\
        --parent-dir /oceanus/collab/herz-lab/raw_data/kahana/subjects \\
        --subfolder clinical_eeg \\
        --wpm 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_WPM = 150


def _derive_site_code(subject_dir_name: str) -> str | None:
    """R1XXXY[_M] -> Y (single site letter). Returns None if the
    folder name doesn't match the R-code shape -- callers then fall
    back to shared-whitelist-only matching.
    """
    import re
    m = re.match(r"^R1\d{3}([ACDEFHJMNPST])(?:_\d+)?$", subject_dir_name)
    return m.group(1) if m else None


def count_edf_annotations(edf_path: Path,
                           whitelist=None,
                           site_code: str | None = None,
                           ) -> tuple[int, int, int]:
    """Return ``(n_annotations, n_words, n_whitelisted)`` for the non-
    timekeeping annotations in ``edf_path``. Whitespace-only entries
    excluded.

    ``whitelist`` (an optional :class:`BoilerplateWhitelist`): if
    given, annotations that ``fullmatch`` a whitelist pattern for the
    subject's ``site_code`` are counted separately as
    ``n_whitelisted`` and EXCLUDED from ``n_annotations`` /
    ``n_words``. Lets the review-time estimate shrink as the operator
    grows the whitelist during review.
    """
    from clean_eeg.annotation_reader import (
        count_words_in_annotations,
        iter_annotations,
    )
    anns = iter_annotations(edf_path)
    kept = [a for a in anns if a.text.strip()]
    if whitelist is None:
        return len(kept), count_words_in_annotations(kept), 0
    to_review, whitelisted = [], 0
    for a in kept:
        if whitelist.matches(a.text, site_code=site_code):
            whitelisted += 1
        else:
            to_review.append(a)
    return len(to_review), count_words_in_annotations(to_review), whitelisted


def _reviewed_paths_for(subj_dir: Path) -> set[str]:
    """Load ``<subj_dir>/.annotation_reviewed_tracker`` if present.
    Returns the deduped set of absolute paths already reviewed."""
    from clean_eeg.annotation_review.journal import ReviewedTracker
    return ReviewedTracker(subj_dir).reviewed_paths()


def scan_parent(parent_dir: Path,
                subfolder: str = "clinical_eeg",
                whitelist=None,
                respect_reviewed_tracker: bool = True,
                show_progress: bool = True,
                ) -> tuple[dict[str, tuple[int, int, int, int, int, int]],
                            list[tuple[str, str]]]:
    """Walk ``parent_dir`` looking for per-subject folders. Only
    subjects with a ``<subject>/<subfolder>/`` subdir are counted;
    subjects missing that subdir are skipped (reported in the second
    return value). Skips ``*_annotations.edf`` sidecars so we don't
    double-count post-clean inplace-mode annotation stubs.

    ``whitelist``: if given, annotations matched by the per-site
    whitelist are excluded from the review-time count and reported
    as a separate ``n_whitelisted`` bucket. Site code is derived
    from the folder name via the R1XXXY[_M] pattern.

    ``respect_reviewed_tracker``: if True (default), files whose paths
    appear in ``<subject_dir>/.annotation_reviewed_tracker`` are
    skipped and reported in ``n_files_reviewed`` -- lets the estimate
    shrink as the operator makes progress.

    Returns ``(per_subject, skipped_subjects)`` where:
        per_subject = {subject_dir_name: (n_ann, n_words, n_files_ok,
                       n_files_skipped, n_files_reviewed, n_whitelisted)}
        skipped_subjects = [(subject_dir_name, reason), ...]  ordered

    Reasons include ``"no <subfolder>/ subdir"`` and
    ``"permission denied: <path>"``. Permission errors on any read
    inside a subject are captured and the subject is skipped rather
    than halting the whole scan -- an operator running against a
    shared collab drive routinely doesn't own everything.
    """
    per_subject: dict[str, tuple[int, int, int, int, int, int]] = {}
    skipped_subjects: list[tuple[str, str]] = []
    if not parent_dir.exists():
        raise FileNotFoundError(f"{parent_dir} does not exist")

    try:
        subject_dirs = sorted(parent_dir.iterdir())
    except PermissionError as e:
        raise PermissionError(
            f"{parent_dir}: cannot list children: {e}") from e

    # Progress bar: writes to stderr so the final report on stdout
    # stays a clean, greppable block. disable=True gives silent
    # scan for programmatic callers / tests.
    from tqdm import tqdm
    iterator = tqdm(subject_dirs, desc="scanning subjects",
                    unit="subj", disable=not show_progress,
                    dynamic_ncols=True)

    for subj_dir in iterator:
        if hasattr(iterator, "set_postfix_str"):
            iterator.set_postfix_str(subj_dir.name, refresh=False)
        try:
            if not subj_dir.is_dir():
                continue
        except PermissionError as e:
            skipped_subjects.append(
                (subj_dir.name, f"permission denied: {e}"))
            continue

        inner = subj_dir / subfolder
        # Explicit: no fallback to the subject dir itself. If the
        # expected subfolder is missing, the subject is not laid out
        # the way this tool expects and gets skipped with a clear
        # reason.
        try:
            inner_exists = inner.exists()
        except PermissionError as e:
            skipped_subjects.append(
                (subj_dir.name, f"permission denied: {e}"))
            continue
        if not inner_exists:
            skipped_subjects.append(
                (subj_dir.name, f"no {subfolder}/ subdir"))
            continue

        site_code = _derive_site_code(subj_dir.name)
        try:
            reviewed_paths = (_reviewed_paths_for(subj_dir)
                              if respect_reviewed_tracker else set())
        except PermissionError as e:
            skipped_subjects.append(
                (subj_dir.name, f"permission denied on tracker: {e}"))
            continue

        n_ann = n_words = n_ok = n_skipped = n_reviewed = n_whitelisted = 0
        # Force an explicit permission check -- pathlib.Path.rglob
        # silently returns [] on a chmod-000 directory instead of
        # raising, which would leave the subject in the results with
        # 0 counts and mask the coverage gap. os.listdir DOES raise.
        try:
            import os as _os
            _os.listdir(inner)
        except PermissionError as e:
            skipped_subjects.append(
                (subj_dir.name,
                 f"permission denied while listing EDFs: {e}"))
            continue
        edfs = sorted(inner.rglob("*.edf"))

        for edf in edfs:
            if edf.name.endswith("_annotations.edf"):
                continue
            if str(edf) in reviewed_paths:
                n_reviewed += 1
                continue
            try:
                a, w, wl = count_edf_annotations(
                    edf, whitelist=whitelist, site_code=site_code)
            except PermissionError:
                # Per-file permission problem -- report as skipped so
                # the operator knows coverage is incomplete for this
                # subject.
                n_skipped += 1
                continue
            except Exception:
                n_skipped += 1
                continue
            n_ann += a
            n_words += w
            n_whitelisted += wl
            n_ok += 1
        per_subject[subj_dir.name] = (n_ann, n_words, n_ok, n_skipped,
                                       n_reviewed, n_whitelisted)
    return per_subject, skipped_subjects


def print_report(per_subject: dict[str, tuple[int, int, int, int, int, int]],
                 wpm: int,
                 skipped_subjects: list[tuple[str, str]] | None = None,
                 ) -> None:
    total_ann = sum(a for a, _, _, _, _, _ in per_subject.values())
    total_words = sum(w for _, w, _, _, _, _ in per_subject.values())
    total_skipped = sum(s for _, _, _, s, _, _ in per_subject.values())
    total_reviewed = sum(r for _, _, _, _, r, _ in per_subject.values())
    total_whitelisted = sum(wl for _, _, _, _, _, wl in per_subject.values())
    with_data = [k for k, v in per_subject.items() if v[2] > 0]

    print(f"\n=== Annotation review estimate ===")
    print(f"Subjects scanned:       {len(per_subject)}  "
          f"({len(with_data)} with unreviewed readable EDFs)")
    print(f"Remaining annotations:  {total_ann:,}")
    print(f"Remaining words:        {total_words:,}")
    if total_whitelisted:
        print(f"Whitelisted (excluded): {total_whitelisted:,} annotations")
    if total_reviewed:
        print(f"Files already reviewed: {total_reviewed}  (skipped)")
    if with_data:
        print(f"Mean / subject:         "
              f"{total_ann / len(with_data):,.0f} annotations, "
              f"{total_words / len(with_data):,.0f} words")
        est_total_min = total_words / wpm
        est_per_min = total_words / len(with_data) / wpm
        print(f"Estimated review @ {wpm} wpm:")
        print(f"  total remaining:     "
              f"{est_total_min:,.0f} min  ({est_total_min / 60:.1f} h)")
        print(f"  mean per subject:    "
              f"{est_per_min:,.0f} min")
    if total_skipped:
        print(f"\n[warn] {total_skipped} EDF file(s) could not be read "
              f"(raw NK / EDF+D unsplit / corrupt). These are excluded "
              f"from the totals -- clean the affected subjects to include "
              f"them.")

    print(f"\n=== Per-subject ({len(per_subject)} rows) ===")
    header = (f"{'subject':<20s}  {'files':>5s}  {'skip':>4s}  "
              f"{'done':>4s}  {'wlist':>5s}  {'ann':>7s}  "
              f"{'words':>8s}  {'min@' + str(wpm):>7s}")
    print(header)
    print("-" * len(header))
    for code in sorted(per_subject):
        a, w, f_ok, s, r, wl = per_subject[code]
        mins = w / wpm if w else 0
        print(f"{code:<20s}  {f_ok:>5d}  {s:>4d}  {r:>4d}  "
              f"{wl:>5d}  {a:>7,}  {w:>8,}  {mins:>7.0f}")

    if skipped_subjects:
        print(f"\n=== Skipped subjects ({len(skipped_subjects)}) ===")
        for name, reason in skipped_subjects:
            print(f"  {name:<20s}  {reason}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Count annotations + words across every subject "
                    "under a parent dir; estimate manual review time. "
                    "Whitelist-matched annotations and files listed "
                    "in .annotation_reviewed_tracker are excluded by "
                    "default so the estimate shrinks as you make "
                    "progress.")
    p.add_argument("--parent-dir", type=Path, required=True,
                   help="Parent dir with per-subject subfolders")
    p.add_argument("--subfolder", type=str, default="clinical_eeg",
                   help="Per-subject sub-folder for EDFs "
                        "(default: clinical_eeg)")
    p.add_argument("--wpm", type=int, default=DEFAULT_WPM,
                   help=f"Words-per-minute reading rate for the review "
                        f"estimate (default: {DEFAULT_WPM})")
    p.add_argument("--whitelist-path", type=Path, default=None,
                   metavar="FILE",
                   help="Path to a boilerplate whitelist JSON (per-site "
                        "regex fullmatch). Matched annotations are "
                        "excluded from the review-time count and "
                        "reported separately. Site code is derived "
                        "from the R1XXXY[_M] subject folder name.")
    p.add_argument("--include-reviewed", action="store_true",
                   help="Include EDF files listed in "
                        "<subject>/.annotation_reviewed_tracker. Default "
                        "is to skip them (so re-runs during a long "
                        "review show only the remaining work).")
    args = p.parse_args(argv)

    whitelist = None
    if args.whitelist_path is not None:
        from clean_eeg.annotation_boilerplate import (
            BoilerplateWhitelistError,
            load_whitelist,
        )
        try:
            whitelist = load_whitelist(args.whitelist_path)
        except BoilerplateWhitelistError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2

    try:
        per_subject, skipped_subjects = scan_parent(
            args.parent_dir, args.subfolder,
            whitelist=whitelist,
            respect_reviewed_tracker=not args.include_reviewed)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    print_report(per_subject, args.wpm,
                 skipped_subjects=skipped_subjects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
