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


def count_edf_annotations(edf_path: Path) -> tuple[int, int]:
    """Return ``(n_annotations, n_words)`` for the non-timekeeping
    annotations in ``edf_path``. Whitespace-only entries excluded
    (they'd inflate the annotation count without carrying anything
    to read). Word count is whitespace-tokenized on each annotation
    text -- close enough for a review-time estimate.

    Uses the mmap-based reader in ``clean_eeg.annotation_reader``,
    which skips signal-data reads entirely -- multi-GB files count
    in seconds instead of minutes. Also works on files pyedflib
    refuses (raw EDF+D not yet split), so PRE-clean and POST-clean
    counts use the same code path.
    """
    from clean_eeg.annotation_reader import (
        count_words_in_annotations,
        iter_annotations,
    )
    anns = iter_annotations(edf_path)
    kept = [a for a in anns if a.text.strip()]
    return len(kept), count_words_in_annotations(kept)


def scan_parent(parent_dir: Path,
                subfolder: str = "clinical_eeg",
                ) -> dict[str, tuple[int, int, int, int]]:
    """Walk ``parent_dir`` looking for per-subject folders. For each
    subject, look for ``<subject>/<subfolder>/*.edf`` first; fall back
    to ``<subject>/*.edf`` if the subfolder doesn't exist. Skips
    ``*_annotations.edf`` sidecars so we don't double-count post-clean
    inplace-mode annotation stubs.

    Returns ``{subject_dir_name: (n_ann, n_words, n_files_ok,
    n_files_skipped)}``. Skipped-file count includes any EDF that
    pyedflib refuses (raw NK exports, EDF+D not yet split, corrupt
    files).
    """
    per_subject: dict[str, tuple[int, int, int, int]] = {}
    if not parent_dir.exists():
        raise FileNotFoundError(f"{parent_dir} does not exist")
    for subj_dir in sorted(parent_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        inner = subj_dir / subfolder
        target = inner if inner.exists() else subj_dir
        n_ann = n_words = n_ok = n_skipped = 0
        for edf in sorted(target.rglob("*.edf")):
            if edf.name.endswith("_annotations.edf"):
                # Inplace-mode sidecar: annotations here are a copy of
                # what's already inline in the main EDF. Counting both
                # would double the estimate.
                continue
            try:
                a, w = count_edf_annotations(edf)
            except Exception:
                n_skipped += 1
                continue
            n_ann += a
            n_words += w
            n_ok += 1
        per_subject[subj_dir.name] = (n_ann, n_words, n_ok, n_skipped)
    return per_subject


def print_report(per_subject: dict[str, tuple[int, int, int, int]],
                 wpm: int) -> None:
    total_ann = sum(a for a, _, _, _ in per_subject.values())
    total_words = sum(w for _, w, _, _ in per_subject.values())
    total_skipped = sum(s for _, _, _, s in per_subject.values())
    with_data = [k for k, v in per_subject.items() if v[2] > 0]

    print(f"\n=== Annotation review estimate ===")
    print(f"Subjects scanned:       {len(per_subject)}  "
          f"({len(with_data)} with readable EDFs)")
    print(f"Total annotations:      {total_ann:,}")
    print(f"Total words:            {total_words:,}")
    if with_data:
        print(f"Mean / subject:         "
              f"{total_ann / len(with_data):,.0f} annotations, "
              f"{total_words / len(with_data):,.0f} words")
        est_total_min = total_words / wpm
        est_per_min = total_words / len(with_data) / wpm
        print(f"Estimated review @ {wpm} wpm:")
        print(f"  total:               "
              f"{est_total_min:,.0f} min  ({est_total_min / 60:.1f} h)")
        print(f"  mean per subject:    "
              f"{est_per_min:,.0f} min")
    if total_skipped:
        print(f"\n[warn] {total_skipped} EDF file(s) could not be read "
              f"(raw NK / EDF+D unsplit / corrupt). These are excluded "
              f"from the totals -- clean the affected subjects to include "
              f"them.")

    print(f"\n=== Per-subject ({len(per_subject)} rows) ===")
    print(f"{'subject':<20s}  {'files':>5s}  {'skip':>4s}  "
          f"{'ann':>7s}  {'words':>8s}  {'min@' + str(wpm):>7s}")
    print("-" * 66)
    for code in sorted(per_subject):
        a, w, f_ok, s = per_subject[code]
        mins = w / wpm if w else 0
        print(f"{code:<20s}  {f_ok:>5d}  {s:>4d}  "
              f"{a:>7,}  {w:>8,}  {mins:>7.0f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Count annotations + words across every subject "
                    "under a parent dir; estimate manual review time.")
    p.add_argument("--parent-dir", type=Path, required=True,
                   help="Parent dir with per-subject subfolders")
    p.add_argument("--subfolder", type=str, default="clinical_eeg",
                   help="Per-subject sub-folder for EDFs "
                        "(default: clinical_eeg)")
    p.add_argument("--wpm", type=int, default=DEFAULT_WPM,
                   help=f"Words-per-minute reading rate for the review "
                        f"estimate (default: {DEFAULT_WPM})")
    args = p.parse_args(argv)
    try:
        per_subject = scan_parent(args.parent_dir, args.subfolder)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    print_report(per_subject, args.wpm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
