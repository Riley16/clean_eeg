"""Generate a review file of common English words for bulk-adding to
the audit's vocab whitelist.

Source: ``data/SUBTLEX-US_frequency_list_PoS_Zipf.xlsx`` (movie/TV
subtitle frequency corpus, ~74K words). We take the top N by
``FREQcount`` and casefold. Then apply three filters:

  1. Drop words already in ``data/annotation_vocab_whitelist.json``
     (no point reviewing them again).
  2. **Drop words whose dominant part-of-speech is 'Name'** — the
     SUBTLEX PoS tagger flags proper nouns like ``michael``,
     ``john``, ``mary``, ``sam``, ``mark``. These are exactly the
     words we want the audit to KEEP flagging as potential PHI, so
     they shouldn't appear in a whitelist-candidate list. Removing
     them here means the operator only reviews genuinely ambiguous
     cases instead of scrolling past hundreds of obvious names.
  3. Cross-reference with the US name dictionary
     (``data/name_dictionary/us_names.txt.gz``) — words that STILL
     match after dropping Name-dominant PoS ARE what the audit
     currently flags falsely (common English words that also appear
     as surnames somewhere in the multi-language name dataset). These
     are the load-bearing subset of the review. Non-name-matches
     would be no-ops if whitelisted (the audit never flagged them),
     included in a separate section for reference.

Writes ``temp/top_10k_english_review.txt`` with two sections,
frequency-ordered within each so highest-impact silences surface
first:

  --- name-dictionary matches (REVIEW: delete lines you want to KEEP
      as PHI detection; everything else gets whitelisted) ---
  you        (freq: 2,134,713)
  it         (freq: 963,712)
  ...

  --- non-matches (safe to bulk-whitelist; no audit effect) ---
  the        (freq: 1,041,179)
  of         (freq: ...)
  ...
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

from clean_eeg.paths import DATA_DIR


# Recognizes review-file word lines like ``you        (freq: 2,134,713)``.
_WORD_LINE_RE = re.compile(r"^([a-z][a-z'\-]*)\s+\(freq:")


DEFAULT_TOP_N = 10_000
SUBTLEX_PATH = DATA_DIR / "SUBTLEX-US_frequency_list_PoS_Zipf.xlsx"
NAME_DICT_PATH = DATA_DIR / "name_dictionary" / "us_names.txt.gz"
VOCAB_WHITELIST_PATH = DATA_DIR / "annotation_vocab_whitelist.json"
OUTPUT_PATH = Path("temp") / "top_10k_english_review.txt"


def _load_name_dict() -> set[str]:
    with gzip.open(NAME_DICT_PATH, "rt", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.strip()}


def _load_existing_whitelist() -> set[str]:
    if not VOCAB_WHITELIST_PATH.exists():
        return set()
    return {str(w).lower() for w in json.loads(VOCAB_WHITELIST_PATH.read_text())}


def _load_top_words(top_n: int) -> tuple[list[tuple[str, int]], set[str]]:
    """Return ``([(word, freq), ...], name_dominant_words)`` for the
    top ``top_n`` casefolded words by FREQcount. The second element
    is the subset whose SUBTLEX dominant PoS is 'Name' (proper nouns
    the audit should keep flagging)."""
    df = pd.read_excel(SUBTLEX_PATH,
                       usecols=["Word", "FREQcount", "Dom_PoS_SUBTLEX"])
    df = df.dropna(subset=["Word", "FREQcount"])
    df["Word"] = df["Word"].astype(str).str.strip().str.casefold()
    df = df[df["Word"].str.len() > 0]
    # Dedupe by taking max FREQcount per casefolded form; keep the PoS
    # tag from the highest-freq row (via first-in-group after sort).
    df = df.sort_values("FREQcount", ascending=False)
    df = df.drop_duplicates(subset=["Word"], keep="first")
    df = df.head(top_n)
    name_dominant = set(df.loc[df["Dom_PoS_SUBTLEX"] == "Name", "Word"])
    tuples = list(df[["Word", "FREQcount"]].itertuples(index=False, name=None))
    return tuples, name_dominant


def _read_existing_review(path: Path) -> set[str]:
    """Return the set of words in an existing review file, ignoring
    comments and blanks. Returns empty set if the file doesn't exist."""
    if not path.exists():
        return set()
    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _WORD_LINE_RE.match(s)
        if m:
            words.add(m.group(1))
    return words


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                   help=f"Take top-N most-frequent words (default {DEFAULT_TOP_N}).")
    p.add_argument("--from-scratch", action="store_true",
                   help="Ignore any existing review file — start with the "
                        "full fresh top-N minus already-whitelisted and "
                        "Name-dominant. Default is INCREMENTAL: preserve "
                        "deletions the operator has already made.")
    args = p.parse_args(argv)
    top_n = args.top_n

    if not SUBTLEX_PATH.exists():
        print(f"ERROR: {SUBTLEX_PATH} not found", file=sys.stderr)
        return 1
    if not NAME_DICT_PATH.exists():
        print(f"ERROR: {NAME_DICT_PATH} not found (audit shipped names file)",
              file=sys.stderr)
        return 1

    print(f"Loading top {top_n} words from {SUBTLEX_PATH.name}...")
    words_freq, name_dominant = _load_top_words(top_n)
    print(f"  loaded {len(words_freq)} unique casefolded words")
    print(f"  {len(name_dominant)} SUBTLEX Name-dominant "
          f"(will be pre-filtered from whitelist candidates)")

    print(f"Loading name dictionary from {NAME_DICT_PATH.name}...")
    name_dict = _load_name_dict()
    print(f"  {len(name_dict):,} entries")

    print(f"Loading existing vocab whitelist from {VOCAB_WHITELIST_PATH.name}...")
    existing = _load_existing_whitelist()
    print(f"  {len(existing)} already-whitelisted tokens (will be excluded)")

    # Two-stage drop: already-whitelisted + Name-dominant proper nouns.
    # The Name-dominant filter is the meaningful pre-review pass —
    # 'michael', 'sam', 'mary' etc. are exactly what the audit should
    # keep flagging as potential PHI, so they don't belong in a
    # whitelist-candidate list at all.
    dropped_names = sorted(w for w, _ in words_freq if w in name_dominant)

    # Incremental mode (default): preserve the operator's existing
    # deletions. Anything in top-N that ISN'T in the current review
    # file was deleted deliberately — do not resurrect it.
    if args.from_scratch:
        prior = None
    else:
        prior = _read_existing_review(OUTPUT_PATH)
        if prior:
            print(f"Existing review file has {len(prior)} words; preserving "
                  "manual deletions (pass --from-scratch to regenerate).")

    def _keep(word: str) -> bool:
        if word in existing:
            return False
        if word in name_dominant:
            return False
        if prior is not None and word not in prior:
            return False
        return True

    to_review = [(w, f) for (w, f) in words_freq if _keep(w)]

    name_matches = [(w, f) for (w, f) in to_review if w in name_dict]
    non_matches = [(w, f) for (w, f) in to_review if w not in name_dict]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    max_word_len = max((len(w) for w, _ in to_review), default=8)
    col_w = max(max_word_len, 8) + 2

    lines: list[str] = []
    lines.append(f"# Top {top_n} SUBTLEX-US English words for vocab-whitelist review.")
    lines.append(f"# Pre-filtered out:")
    lines.append(f"#   - {len(existing)} tokens already in {VOCAB_WHITELIST_PATH.name}")
    lines.append(f"#   - {len(name_dominant)} SUBTLEX Name-dominant proper nouns "
                  f"(michael, john, mary, sam, ...)")
    lines.append(f"# Remaining to review: {len(name_matches):,} name-dict matches "
                  f"+ {len(non_matches):,} non-matches = {len(to_review):,} words.")
    lines.append("#")
    lines.append("# INSTRUCTIONS: for each section, DELETE any line whose word")
    lines.append("# you want to KEEP as name-match (i.e., you want future audits")
    lines.append("# to keep flagging it as potential PHI). Everything remaining")
    lines.append("# in the file gets added to the vocab whitelist.")
    lines.append("#")
    lines.append("# Ordered by SUBTLEX frequency count (highest first) so the")
    lines.append("# most-common English words are at the top of each section.")
    lines.append("#")
    lines.append("")
    lines.append("# --- Section 1: matches the US name dictionary — REVIEW carefully ---")
    lines.append("# (These are the entries that actually affect audit behavior. Non-")
    lines.append("#  matches in section 2 would be no-ops if whitelisted.)")
    lines.append("")
    for w, freq in name_matches:
        lines.append(f"{w:<{col_w}}(freq: {int(freq):,})")

    lines.append("")
    lines.append("# --- Section 2: NOT in the name dictionary — safe to bulk-whitelist ---")
    lines.append("# (Included for completeness. The audit already ignores these — but")
    lines.append("#  whitelisting them defensively means the FIRST time one accidentally")
    lines.append("#  matches the name dict in a future dict update, it stays silenced.)")
    lines.append("")
    for w, freq in non_matches:
        lines.append(f"{w:<{col_w}}(freq: {int(freq):,})")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  review size: {len(name_matches):,} name-match words + "
          f"{len(non_matches):,} non-match words")
    if dropped_names:
        print(f"  dropped {len(dropped_names)} SUBTLEX Name-dominant words, "
              f"e.g. {dropped_names[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
