"""Annotation extraction + hard-dictionary PHI scan for cleaned EDFs.

Extraction reuses the byte-level TAL parser from ``print_edf_header``
so the audit works even on files pyedflib refuses to open. The scan
does a case-insensitive hard match of every alphabetic token against a
US-name dictionary, minus a persistent operator-curated
annotation-vocab whitelist that grows over successive audits.

The check is intentionally noisy at first — the operator seeds the
whitelist with legitimate annotation vocabulary (``seizure``,
``focal``, ``clinical``, ...) across a handful of subjects, after
which only real name hits remain.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from clean_eeg.print_edf_header import (
    _find_annotation_signal_index,
    _parse_record_tals,
    _read_annotation_blocks,
    MAIN_HEADER_BYTES,
    SIGNAL_HEADER_BYTES_PER_SIGNAL,
    read_main_header,
    read_signal_headers,
)


# Alphabetic tokens with optional internal apostrophes / hyphens
# (e.g. ``O'Connor``, ``Jean-Luc``). Numbers and punctuation are
# stripped; name-dictionary entries are pure letters.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")


def _tokenize(text: str) -> list[str]:
    """Return lowercase alphabetic tokens from ``text``."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def extract_annotations(edf_path: str | Path) -> list[dict]:
    """Return every non-empty annotation from an EDF as
    ``{'onset', 'duration', 'text'}`` dicts. Empty results (no
    annotation channel, unparseable header, broken file) all yield
    ``[]`` rather than raising — the caller aggregates.
    """
    p = Path(edf_path)
    header = read_main_header(str(p))
    n_signals = header.get("n_signals")
    n_records = header.get("n_records")
    if (not isinstance(n_signals, int) or not isinstance(n_records, int)
            or n_signals <= 0 or n_records <= 0):
        return []
    sigs = read_signal_headers(str(p), n_signals)
    ann_idx = _find_annotation_signal_index(sigs)
    if ann_idx is None:
        return []
    header_bytes_total = MAIN_HEADER_BYTES + n_signals * SIGNAL_HEADER_BYTES_PER_SIGNAL
    file_size = p.stat().st_size
    blocks = _read_annotation_blocks(str(p), sigs, n_records, ann_idx,
                                     header_bytes_total, file_size)
    out: list[dict] = []
    for block in blocks:
        for onset, duration, texts in _parse_record_tals(block):
            for text in texts:
                if text:  # skip the empty text of the timekeeping TAL
                    out.append({"onset": onset, "duration": duration, "text": text})
    return out


def scan_annotation_texts(annotations: Iterable[dict],
                          name_set,  # set[str] | frozenset[str]
                          vocab_whitelist: set[str] | None = None
                          ) -> tuple[list[dict], dict[str, list[dict]]]:
    """Hard-match every token in each annotation text against
    ``name_set`` (lowercased), skipping tokens in ``vocab_whitelist``.

    Returns ``(per_annotation_matches, matched_tokens_inverted)`` where:
      - each per-annotation entry carries ``onset``, ``text``, and the
        list of ``matched_tokens`` from that annotation
      - the inverted index maps each matched token to the list of
        annotations it fired on.
    """
    vocab = {v.lower() for v in (vocab_whitelist or set())}
    per_ann_matches: list[dict] = []
    inverted: dict[str, list[dict]] = {}
    for ann in annotations:
        tokens = _tokenize(ann.get("text", ""))
        hits = [t for t in tokens if t in name_set and t not in vocab]
        if hits:
            entry = {
                "onset": ann.get("onset"),
                "text": ann.get("text"),
                "matched_tokens": hits,
            }
            per_ann_matches.append(entry)
            for t in hits:
                inverted.setdefault(t, []).append(entry)
    return per_ann_matches, inverted


def check_annotation_phi_scan(edf_paths: Iterable[str | Path],
                              *,
                              name_dictionary: Iterable[str] | None = None,
                              vocab_whitelist: Iterable[str] | None = None
                              ) -> dict:
    """Scan every annotation across ``edf_paths`` for tokens that match
    a US-name dictionary. Any match fails the audit.

    ``name_dictionary``: iterable of names (usually millions of entries
    from ``scripts.build_whitelist.load_names_dataset_names(['US'])``).
    Loaded lazily if omitted; tests should pass a small set to avoid
    the ~32M-row CSV load.
    ``vocab_whitelist``: operator-curated tokens to exempt (e.g.
    ``seizure``, ``focal``). Grows over successive audits.
    """
    paths = [Path(p) for p in edf_paths]

    if name_dictionary is None:
        # Disk-cached loader: cold ~23s (full CSV rebuild), warm <1s.
        from clean_eeg.audit.name_dictionary import load_us_name_dictionary
        name_set: frozenset[str] | set[str] = load_us_name_dictionary(
            countries=('US',))
    else:
        name_set = {str(n).lower() for n in name_dictionary if isinstance(n, str)}
    vocab = {v.lower() for v in (vocab_whitelist or set())}

    matches_by_file: dict[str, list[dict]] = {}
    inverted: dict[str, list[dict]] = {}
    n_annotations_scanned = 0
    for p in paths:
        anns = extract_annotations(p)
        n_annotations_scanned += len(anns)
        per_ann, inv = scan_annotation_texts(anns, name_set, vocab)
        if per_ann:
            matches_by_file[p.name] = per_ann
        for token, entries in inv.items():
            for entry in entries:
                inverted.setdefault(token, []).append({
                    "file": p.name,
                    "onset": entry["onset"],
                    "text": entry["text"],
                })

    issues: list[str] = []
    if not paths:
        status = "fail"
        issues.append("No EDF files were provided")
    elif inverted:
        status = "fail"
        for token in sorted(inverted, key=lambda t: -len(inverted[t])):
            issues.append(
                f"'{token}': matched US-name dictionary in "
                f"{len(inverted[token])} annotation(s)"
            )
    else:
        status = "pass"

    return {
        "check": "annotation_phi_scan",
        "status": status,
        "n_files": len(paths),
        "n_annotations_scanned": n_annotations_scanned,
        "n_matches": sum(len(v) for v in inverted.values()),
        "matches_by_file": matches_by_file,
        "matched_tokens": inverted,
        "n_vocab_whitelist_tokens": len(vocab),
        "dictionary_size": len(name_set),
        "issues": issues,
    }
