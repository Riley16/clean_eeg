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
from dataclasses import dataclass
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
        raw_subject_dirs = sorted(parent_dir.iterdir())
    except PermissionError as e:
        raise PermissionError(
            f"{parent_dir}: cannot list children: {e}") from e

    # PRE-FILTER: drop subjects that lack the expected subfolder AND
    # subjects whose subfolder contains zero EDFs. Both are 'just
    # empty' from this tool's perspective; including them in the
    # tqdm total makes the bar 'jump' as we skip past them and
    # obscures the true remaining work.
    subject_dirs = []
    for subj_dir in raw_subject_dirs:
        try:
            if not subj_dir.is_dir():
                continue
        except PermissionError:
            # Preserve permission-denied dirs so they surface in the
            # skipped list below rather than being silently dropped.
            subject_dirs.append(subj_dir)
            continue
        inner = subj_dir / subfolder
        try:
            if not inner.exists():
                continue                       # no subfolder -> silent skip
        except PermissionError:
            subject_dirs.append(subj_dir)      # surface as permission error
            continue
        # Existence probe. os.listdir raises PermissionError on
        # chmod-000 dirs (unlike rglob, which silently returns []).
        # We only need to see ONE .edf so a top-level listing beats
        # rglob when EDFs are direct children. If EDFs live in
        # further subdirs, fall through to a rglob probe.
        import os as _os
        try:
            entries = _os.listdir(inner)
        except PermissionError:
            subject_dirs.append(subj_dir)      # surface as permission err
            continue
        has_edf = any(e.endswith(".edf")
                       and not e.endswith("_annotations.edf")
                       for e in entries)
        if not has_edf:
            # Deeper: nested layouts. rglob-any on a readable dir is
            # safe (permission errors won't be swallowed since we
            # already confirmed inner is readable).
            has_edf = any(
                p.name.endswith(".edf")
                and not p.name.endswith("_annotations.edf")
                for p in inner.rglob("*.edf"))
        if not has_edf:
            continue                           # empty subfolder -> silent skip
        subject_dirs.append(subj_dir)

    # ---- Second pass: enumerate all EDFs across all subjects so
    # the tqdm bar can track FILES (uniform load time) rather than
    # SUBJECTS (highly variable file counts). Also computes per-
    # subject scan metadata once so the per-file inner loop is
    # cheap. ----
    from tqdm import tqdm

    @dataclass
    class _SubjectScanMeta:
        subj_dir: Path
        site_code: str | None
        reviewed_paths: set[str]
        edfs: list[Path]

    subject_metas: list[_SubjectScanMeta] = []
    import os as _os
    for subj_dir in subject_dirs:
        inner = subj_dir / subfolder
        # Explicit permission check: rglob silently swallows
        # PermissionError, so a chmod-000 subject would appear to have
        # 0 EDFs and pollute per_subject with (0,0,...). listdir DOES
        # raise, which is what we want to surface in skipped_subjects.
        try:
            _os.listdir(inner)
        except PermissionError as e:
            skipped_subjects.append(
                (subj_dir.name,
                 f"permission denied while listing EDFs: {e}"))
            continue
        try:
            reviewed_paths = (_reviewed_paths_for(subj_dir)
                              if respect_reviewed_tracker else set())
        except PermissionError as e:
            skipped_subjects.append(
                (subj_dir.name, f"permission denied on tracker: {e}"))
            continue
        edfs = [e for e in sorted(inner.rglob("*.edf"))
                if not e.name.endswith("_annotations.edf")]
        if not edfs:
            # Slipped past the pre-filter (which checks 'any' -- can
            # race with concurrent deletes), or the subject genuinely
            # has zero EDFs by the time we look. Silent skip.
            continue
        subject_metas.append(_SubjectScanMeta(
            subj_dir=subj_dir,
            site_code=_derive_site_code(subj_dir.name),
            reviewed_paths=reviewed_paths,
            edfs=edfs))

    total_files = sum(len(m.edfs) for m in subject_metas)
    file_iter = tqdm(total=total_files, desc="scanning EDFs",
                     unit="file", disable=not show_progress,
                     dynamic_ncols=True)

    # Running totals across the whole scan.
    running_ann = running_words = running_files_ok = running_wl = 0

    def _update_bar_postfix(current_subject: str) -> None:
        """Refresh the tqdm postfix with the always-visible running
        mean + extrapolation-to-total. Called after every file so the
        operator sees the number growing / stabilizing in real time,
        without needing per-file tqdm.write spam.
        """
        if not hasattr(file_iter, "set_postfix_str"):
            return
        if running_files_ok > 0:
            mean_ann = running_ann / running_files_ok
            extrap_total_ann = mean_ann * total_files
            file_iter.set_postfix_str(
                f"{current_subject}  "
                f"μ={mean_ann:.0f} ann/file  "
                f"extrap≈{extrap_total_ann:,.0f} ann across "
                f"{total_files:,} files",
                refresh=False)
        else:
            file_iter.set_postfix_str(current_subject, refresh=False)

    for meta in subject_metas:
        # Per-subject accumulators; flushed to per_subject +
        # incremental summary print at end of each subject.
        n_ann = n_words = n_ok = n_skipped = n_reviewed = n_whitelisted = 0
        _update_bar_postfix(meta.subj_dir.name)

        for edf in meta.edfs:
            if str(edf) in meta.reviewed_paths:
                n_reviewed += 1
                file_iter.update(1)
                _update_bar_postfix(meta.subj_dir.name)
                continue
            try:
                a, w, wl = count_edf_annotations(
                    edf, whitelist=whitelist, site_code=meta.site_code)
            except PermissionError:
                n_skipped += 1
                file_iter.update(1)
                _update_bar_postfix(meta.subj_dir.name)
                continue
            except Exception:
                n_skipped += 1
                file_iter.update(1)
                _update_bar_postfix(meta.subj_dir.name)
                continue
            n_ann += a
            n_words += w
            n_whitelisted += wl
            n_ok += 1
            # Bump running totals BEFORE the postfix update so the
            # mean reflects the file that just completed.
            running_ann += a
            running_words += w
            running_files_ok += 1
            running_wl += wl
            file_iter.update(1)
            _update_bar_postfix(meta.subj_dir.name)

        per_subject[meta.subj_dir.name] = (n_ann, n_words, n_ok, n_skipped,
                                            n_reviewed, n_whitelisted)
        # Running totals were already bumped inside the per-file
        # loop above (so postfix updates every file reflect the
        # latest data). Nothing to add here -- just print the
        # per-subject summary line.
        if show_progress and running_files_ok > 0:
            mean_ann_per_file = running_ann / running_files_ok
            mean_wd_per_file = running_words / running_files_ok
            n_subj_done = len(per_subject)
            mean_ann_per_subj = running_ann / n_subj_done
            mean_wd_per_subj = running_words / n_subj_done
            tqdm.write(
                f"  [{meta.subj_dir.name}] this subj: "
                f"{n_ann:,} ann, {n_words:,} wd, {n_whitelisted:,} wl, "
                f"{n_ok} file(s)  ||  "
                f"running: {running_ann:,} ann, {running_words:,} wd "
                f"across {n_subj_done} subj / {running_files_ok} file(s) "
                f"(mean {mean_ann_per_file:.0f} ann/file, "
                f"{mean_ann_per_subj:.0f} ann/subj, "
                f"{mean_wd_per_file:.0f} wd/file, "
                f"{mean_wd_per_subj:.0f} wd/subj)")

    file_iter.close()
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
    total_files_ok = sum(f_ok for _, _, f_ok, _, _, _
                          in per_subject.values())
    if with_data:
        print(f"Mean / subject:         "
              f"{total_ann / len(with_data):,.0f} annotations, "
              f"{total_words / len(with_data):,.0f} words")
        if total_files_ok:
            print(f"Mean / file:            "
                  f"{total_ann / total_files_ok:,.0f} annotations, "
                  f"{total_words / total_files_ok:,.0f} words  "
                  f"(across {total_files_ok:,} readable file(s))")
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
