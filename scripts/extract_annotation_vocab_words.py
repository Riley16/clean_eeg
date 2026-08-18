"""Extract vocab-whitelist tokens from ``audit-subject-eeg`` printouts.

The audit's ``_always_print_warnings`` block emits lines like::

    [!] Annotation name-dictionary matches — 123 token(s):
        'rec' × 80
            FA3712ZT_R1652A_..._annotations.edf @ 0.0s: 'Segment: REC ...'
            ...
        'start' × 83
            ...

Operators reviewing an audit paste these blocks into a text file (any
tokens they've decided are safe / non-PHI can be whitelisted so future
audits don't re-surface them). This script parses those files, unions
the tokens with the existing
:data:`clean_eeg.paths.ANNOTATION_VOCAB_WHITELIST_PATH`, deduplicates,
sorts, and writes the result back as pretty-printed JSON so future
diffs are line-per-token.

Idempotent — running twice adds nothing the second time.

Usage::

    python scripts/extract_annotation_vocab_words.py FILE [FILE ...]
    python scripts/extract_annotation_vocab_words.py FILE --dry-run
    python scripts/extract_annotation_vocab_words.py FILE --whitelist PATH

The default whitelist path is ``data/annotation_vocab_whitelist.json``,
which is tracked in the repo.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from clean_eeg.paths import DATA_DIR


DEFAULT_WHITELIST = DATA_DIR / "annotation_vocab_whitelist.json"

# Matches lines the audit prints for each match: indent + single-quoted
# token + " × " (U+00D7 multiplication sign) + integer count. Loose
# whitespace handling in case future audit versions tweak the format.
_TOKEN_LINE_RE = re.compile(r"^\s+'([^']+)'\s*×\s*\d+\s*$")

# Matches lines the review file writes: "word            (freq: 960)".
# Word is the first token; the (freq: N) suffix identifies review lines.
_REVIEW_WORD_RE = re.compile(r"^(\S+)\s+\(freq:\s*\d+\)\s*$")

# Matches plain lemma lines: a single lowercase word (with optional
# internal apostrophe / hyphen), flush-left, no other content. Used
# for LemmInflect-expanded lemma lines in the review file. Comment
# lines (leading '#') and freq-suffixed lines fail the anchor.
_PLAIN_TOKEN_RE = re.compile(r"^([a-z][a-z'\-]*)$")


def extract_tokens(paths: list[Path]) -> set[str]:
    """Return the set of unique lowercased tokens parsed from the given
    files. Non-matching lines are ignored (annotation-context lines,
    section headers, comments, blank lines).

    Recognizes three formats:
      1. Audit printout — ``    'token' × N`` (indented, quoted)
      2. Review file — ``word            (freq: 960)`` (word + freq)
      3. Review lemma — ``broken`` (bare lowercase token, flush-left)
    """
    tokens: set[str] = set()
    for p in paths:
        if not p.exists():
            print(f"WARNING: {p} does not exist — skipping", file=sys.stderr)
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            for pattern in (_TOKEN_LINE_RE, _REVIEW_WORD_RE, _PLAIN_TOKEN_RE):
                m = pattern.match(line)
                if m:
                    tokens.add(m.group(1).lower())
                    break
    return tokens


def load_whitelist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected top-level JSON array, got {type(data).__name__}"
        )
    return {str(x).lower() for x in data}


def write_whitelist(path: Path, tokens: set[str]) -> None:
    """Pretty-print sorted tokens, one per line, so future diffs are
    readable per-token instead of one-massive-line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sorted(tokens), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge audit vocab-match tokens into the whitelist JSON.",
    )
    parser.add_argument("input_files", nargs="+", type=Path,
                        help="Text file(s) containing audit printouts to parse.")
    parser.add_argument("--whitelist", type=Path, default=DEFAULT_WHITELIST,
                        help=f"Whitelist JSON to update (default: {DEFAULT_WHITELIST}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be added; do not write.")
    args = parser.parse_args(argv)

    extracted = extract_tokens(args.input_files)
    existing = load_whitelist(args.whitelist)
    new_only = extracted - existing
    already = extracted & existing
    merged = existing | extracted

    print(f"Parsed {len(extracted)} unique tokens from "
          f"{len(args.input_files)} file(s)")
    print(f"  {len(new_only)} new")
    print(f"  {len(already)} already in whitelist")
    print(f"Whitelist size: {len(existing)} → {len(merged)}")

    if new_only:
        # Show a preview so the operator can catch obvious PHI-adjacent
        # entries (staff initials, physician surnames) before the diff
        # hits git.
        sample_n = min(30, len(new_only))
        print(f"\nNew tokens (first {sample_n} alphabetically):")
        for tok in sorted(new_only)[:sample_n]:
            print(f"  {tok}")
        if len(new_only) > sample_n:
            print(f"  ... and {len(new_only) - sample_n} more")

    if args.dry_run:
        print("\n(dry run — no changes written)")
        return 0

    if new_only:
        write_whitelist(args.whitelist, merged)
        print(f"\nWrote {args.whitelist}")
    else:
        print("\nNothing to add — whitelist unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
