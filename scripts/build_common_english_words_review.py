"""Generate a review file of common English words for bulk-adding to
the audit's vocab whitelist.

Source: ``data/SUBTLEX-US_frequency_list_PoS_Zipf.xlsx`` (movie/TV
subtitle frequency corpus, ~74K words). We take the top N by
``FREQcount`` and casefold. Then:

  1. Drop words already in ``data/annotation_vocab_whitelist.json``
     (no point reviewing them again).
  2. Cross-reference with the US name dictionary
     (``data/name_dictionary/us_names.txt.gz``) — words that match ARE
     what the audit currently flags, and are the load-bearing subset
     of the review. Non-name-matches would be no-ops if whitelisted
     (the audit never flagged them in the first place), so we present
     them in a separate section for reference only.

Writes ``temp/top_10k_english_review.txt`` with two sections:

  --- name-dictionary matches (REVIEW: delete lines you want to KEEP
      as PHI detection; everything else gets whitelisted) ---
  will       (freq: 4998.02)
  may        (freq: 4321.09)
  ...

  --- non-matches (safe to bulk-whitelist; no audit effect) ---
  the        (freq: 1041179)
  of         (freq: ...)
  ...

Ordered by frequency descending within each section so operators
scanning top-to-bottom see the highest-impact words first.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

from clean_eeg.paths import DATA_DIR


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


def _load_top_words(top_n: int) -> list[tuple[str, int]]:
    """Return ``[(word, freq), ...]`` sorted by frequency descending,
    casefolded and deduped."""
    df = pd.read_excel(SUBTLEX_PATH, usecols=["Word", "FREQcount"])
    df = df.dropna(subset=["Word", "FREQcount"])
    df["Word"] = df["Word"].astype(str).str.strip().str.casefold()
    df = df[df["Word"].str.len() > 0]
    # Dedupe by taking max FREQcount per casefolded form.
    df = df.groupby("Word", as_index=False)["FREQcount"].max()
    df = df.sort_values("FREQcount", ascending=False)
    return list(df.head(top_n).itertuples(index=False, name=None))


def main(argv: list[str] | None = None) -> int:
    top_n = int(argv[0]) if argv else DEFAULT_TOP_N

    if not SUBTLEX_PATH.exists():
        print(f"ERROR: {SUBTLEX_PATH} not found", file=sys.stderr)
        return 1
    if not NAME_DICT_PATH.exists():
        print(f"ERROR: {NAME_DICT_PATH} not found (audit shipped names file)",
              file=sys.stderr)
        return 1

    print(f"Loading top {top_n} words from {SUBTLEX_PATH.name}...")
    words_freq = _load_top_words(top_n)
    print(f"  loaded {len(words_freq)} unique casefolded words")

    print(f"Loading name dictionary from {NAME_DICT_PATH.name}...")
    name_dict = _load_name_dict()
    print(f"  {len(name_dict):,} entries")

    print(f"Loading existing vocab whitelist from {VOCAB_WHITELIST_PATH.name}...")
    existing = _load_existing_whitelist()
    print(f"  {len(existing)} already-whitelisted tokens (will be excluded)")

    # Drop already-whitelisted words per user instruction.
    to_review = [(w, f) for (w, f) in words_freq if w not in existing]

    name_matches = [(w, f) for (w, f) in to_review if w in name_dict]
    non_matches = [(w, f) for (w, f) in to_review if w not in name_dict]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    max_word_len = max((len(w) for w, _ in to_review), default=8)
    col_w = max(max_word_len, 8) + 2

    lines: list[str] = []
    lines.append(f"# Top {top_n} SUBTLEX-US English words for vocab-whitelist review.")
    lines.append(f"# Excluded: {len(existing)} tokens already in "
                  f"{VOCAB_WHITELIST_PATH.name}.")
    lines.append(f"# {len(name_matches):,} name-dict matches + "
                  f"{len(non_matches):,} non-matches = "
                  f"{len(to_review):,} words to review.")
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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
