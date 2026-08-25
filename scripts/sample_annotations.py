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
    """Union of --edf-file (single) and --subject-dir (all EDFs
    under it). Sidecar '*_annotations.edf' files are excluded so we
    don't double-count."""
    edfs: list[Path] = []
    if args.edf_file:
        edfs.extend(args.edf_file)
    if args.subject_dir:
        inner = args.subject_dir / args.subfolder
        target = inner if inner.exists() else args.subject_dir
        edfs.extend(
            p for p in sorted(target.rglob("*.edf"))
            if not p.name.endswith("_annotations.edf"))
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
    src.add_argument("--edf-file", type=Path, nargs="+",
                     help="One or more explicit .edf files. Skips "
                          "the --subfolder lookup.")
    p.add_argument("--subfolder", type=str, default="clinical_eeg")
    p.add_argument("--max-files", type=int, default=0,
                   help="Cap the number of files inspected (in sorted "
                        "order). Useful for a fast first pass on a "
                        "many-file subject. 0 = no cap.")
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--sample-n", type=int, default=10,
                   help="Random samples in addition to top-N. Set 0 "
                        "to skip the sample block.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for --sample-n (default 0 -> "
                        "reproducible).")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
