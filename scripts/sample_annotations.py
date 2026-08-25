"""Dump annotations from a few EDFs, sorted by frequency.

Motivated by manual whitelist iteration: pick the most-common
patterns, add them to the whitelist, re-run count_annotations, see
the review-time estimate shrink.

Two modes:
    --top-n N        (default 30) print the N most-common annotation
                     texts with per-text counts across all inspected
                     files. Best for building whitelist entries --
                     one regex silencing a top-count text kills all
                     of them from future counts.
    --sample-n N     (default 0) also print N randomly-sampled
                     annotation texts. Best for spot-checking that
                     lower-frequency annotations are also boilerplate
                     (vs. real clinical content).

Uses the fast mmap-based reader so multi-GB EDFs load in seconds.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from clean_eeg.annotation_reader import iter_annotations


def collect_annotation_texts(edf_paths: list[Path]) -> list[tuple[Path, str]]:
    """Return every (file, text) pair across the given EDFs. Empty
    or whitespace-only texts are dropped. Order preserved so
    --sample-n's ``random.sample`` gets a uniform draw."""
    from tqdm import tqdm
    out: list[tuple[Path, str]] = []
    for p in tqdm(edf_paths, desc="loading annotations", unit="file"):
        try:
            for a in iter_annotations(p):
                if a.text.strip():
                    out.append((p, a.text))
        except Exception as e:
            print(f"[skip-read] {p.name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
    return out


def _resolve_edfs(args: argparse.Namespace) -> list[Path]:
    """Resolve the EDF source. Exactly one of --edf-file /
    --subject-dir / --parent-dir. Sidecar '*_annotations.edf' files
    are excluded so we don't double-count."""
    edfs: list[Path] = []
    if args.edf_file:
        edfs.extend(args.edf_file)
    elif args.subject_dir:
        inner = args.subject_dir / args.subfolder
        target = inner if inner.exists() else args.subject_dir
        edfs.extend(
            p for p in sorted(target.rglob("*.edf"))
            if not p.name.endswith("_annotations.edf"))
    elif args.parent_dir:
        # Multi-subject: <parent>/<subject>/<subfolder>/*.edf across
        # every subject folder. Silent-drop subjects with no
        # subfolder or no EDFs (same rule as count_annotations).
        for subj_dir in sorted(args.parent_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            inner = subj_dir / args.subfolder
            if not inner.exists():
                continue
            edfs.extend(
                p for p in sorted(inner.rglob("*.edf"))
                if not p.name.endswith("_annotations.edf"))

    # --random-sample N: pick N at random from the resolved set.
    # Applied BEFORE --max-files so the operator can combine them
    # (e.g. random-sample 100 then max-files 10 for a fast peek).
    if args.random_sample and len(edfs) > args.random_sample:
        edfs = random.sample(edfs, args.random_sample)
        edfs.sort()   # readable output order

    if args.max_files and len(edfs) > args.max_files:
        edfs = edfs[:args.max_files]
    return edfs


def _print_top_n(pairs: list[tuple[Path, str]], top_n: int) -> None:
    counter = Counter(text for _, text in pairs)
    if not counter:
        print("\n(no annotations found)")
        return
    print(f"\n=== Top {min(top_n, len(counter))} most-frequent "
          f"annotation texts ({len(pairs):,} total annotations, "
          f"{len(counter):,} unique) ===\n")
    total = len(pairs)
    for text, count in counter.most_common(top_n):
        pct = 100.0 * count / total
        print(f"  {count:>6d}  ({pct:>5.1f}%)  {text!r}")


def _print_sample(pairs: list[tuple[Path, str]], sample_n: int) -> None:
    if sample_n <= 0 or not pairs:
        return
    n = min(sample_n, len(pairs))
    sampled = random.sample(pairs, n)
    print(f"\n=== {n} random samples ===\n")
    for path, text in sampled:
        print(f"  [{path.name}]  {text!r}")


def _print_all_annotations(pairs: list[tuple[Path, str]]) -> None:
    """Full dump grouped by file, chronological within each file.
    For eyeballing near-duplicates and one-off annotations that
    frequency counting doesn't catch. Loud on large scans -- pair
    with --random-sample N to keep it tractable."""
    if not pairs:
        return
    by_file: dict[Path, list[str]] = {}
    for path, text in pairs:
        by_file.setdefault(path, []).append(text)
    total = len(pairs)
    print(f"\n=== All annotations ({total:,} across "
          f"{len(by_file)} file(s)) ===")
    for path in sorted(by_file):
        anns = by_file[path]
        print(f"\n--- {path}  ({len(anns)} annotation(s)) ---")
        for text in anns:
            print(f"  {text!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Print annotation texts sorted by frequency + "
                    "random samples. For iterating on the "
                    "boilerplate whitelist.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--subject-dir", type=Path,
                     help="Per-subject dir. Scans all EDFs under "
                          "<subject>/<subfolder>/ (default subfolder: "
                          "clinical_eeg).")
    src.add_argument("--parent-dir", type=Path,
                     help="Parent dir containing many subjects. Scans "
                          "every <subject>/<subfolder>/*.edf across "
                          "all subjects. Combine with --random-sample "
                          "N to peek at N random files across the "
                          "whole cohort.")
    src.add_argument("--edf-file", type=Path, nargs="+",
                     help="One or more explicit .edf files. Skips "
                          "the --subfolder lookup.")
    p.add_argument("--subfolder", type=str, default="clinical_eeg")
    p.add_argument("--random-sample", type=int, default=0,
                   metavar="N",
                   help="Pick N random EDFs from the resolved set "
                        "(useful with --parent-dir for a cross-cohort "
                        "peek). 0 = no sampling.")
    p.add_argument("--max-files", type=int, default=0,
                   help="Cap the number of files inspected (applied "
                        "AFTER --random-sample). 0 = no cap.")
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--sample-n", type=int, default=10,
                   help="Random samples in addition to top-N. Set 0 "
                        "to skip the sample block.")
    p.add_argument("--all-annotations", action="store_true",
                   help="ALSO print every annotation grouped by file "
                        "(chronological within each file). Best paired "
                        "with --random-sample N to keep the output "
                        "tractable. Catches near-duplicate patterns "
                        "that frequency counting misses.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for --sample-n / --random-sample "
                        "(default 0 -> reproducible).")
    args = p.parse_args(argv)

    random.seed(args.seed)
    edfs = _resolve_edfs(args)
    if not edfs:
        print("[error] no EDF files found", file=sys.stderr)
        return 2
    print(f"inspecting {len(edfs)} file(s)", file=sys.stderr)
    pairs = collect_annotation_texts(edfs)
    _print_top_n(pairs, args.top_n)
    _print_sample(pairs, args.sample_n)
    if args.all_annotations:
        _print_all_annotations(pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
