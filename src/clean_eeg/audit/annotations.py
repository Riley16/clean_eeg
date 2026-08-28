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

from clean_eeg.annotation_boilerplate import BoilerplateWhitelist
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

# Annotations shorter than this (after strip) are never scanned. Short
# clinical tags like "OFF", "PT", "EEG", "RN", "AWAKE" are almost never
# PHI-carrying — and the handful of short US names this rule silences
# ("Al", "Bo", "Ed", "Ann", "Amy") are an acceptable false-negative
# trade for the huge false-positive reduction on real clinical
# annotations. Bump this if you start seeing short-name PHI leaks.
MIN_ANNOTATION_LENGTH_TO_SCAN = 6


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
                          vocab_whitelist: set[str] | None = None,
                          *,
                          boilerplate_whitelist: BoilerplateWhitelist | None = None,
                          site_code: str | None = None,
                          ) -> tuple[list[dict], dict[str, list[dict]], dict]:
    """Hard-match every token in each annotation text against
    ``name_set`` (lowercased), skipping tokens in ``vocab_whitelist``.

    Two pre-filters run before the token scan:
      1. Length filter — annotations with < :data:`MIN_ANNOTATION_LENGTH_TO_SCAN`
         non-whitespace characters are skipped outright.
      2. Boilerplate filter — if ``boilerplate_whitelist`` is provided
         and the annotation fully matches one of its (per-site or
         shared) regex patterns, the annotation is skipped. Full-match
         semantics — see :meth:`BoilerplateWhitelist.matches`.

    Returns ``(per_annotation_matches, matched_tokens_inverted, stats)``:
      - each per-annotation entry carries ``onset``, ``text``, and the
        list of ``matched_tokens`` from that annotation
      - the inverted index maps each matched token to the list of
        annotations it fired on
      - ``stats`` counts what got filtered by each pre-filter (so the
        operator can see the whitelist working / catch it silencing
        too much).
    """
    vocab = {v.lower() for v in (vocab_whitelist or set())}
    per_ann_matches: list[dict] = []
    inverted: dict[str, list[dict]] = {}
    n_skipped_short = 0
    n_skipped_boilerplate = 0
    n_scanned = 0
    for ann in annotations:
        text = ann.get("text", "") or ""
        if len(text.strip()) < MIN_ANNOTATION_LENGTH_TO_SCAN:
            n_skipped_short += 1
            continue
        if (boilerplate_whitelist is not None
                and boilerplate_whitelist.matches(text, site_code=site_code)):
            n_skipped_boilerplate += 1
            continue
        n_scanned += 1
        tokens = _tokenize(text)
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
    stats = {
        "n_scanned": n_scanned,
        "n_skipped_short": n_skipped_short,
        "n_skipped_boilerplate": n_skipped_boilerplate,
    }
    return per_ann_matches, inverted, stats


def check_annotation_phi_scan(edf_paths: Iterable[str | Path],
                              *,
                              name_dictionary: Iterable[str] | None = None,
                              vocab_whitelist: Iterable[str] | None = None,
                              boilerplate_whitelist: BoilerplateWhitelist | None = None,
                              site_code: str | None = None,
                              ) -> dict:
    """Scan every annotation across ``edf_paths`` for tokens that match
    a US-name dictionary. Any match fails the audit.

    ``name_dictionary``: iterable of names (usually millions of entries
    from ``scripts.build_whitelist.load_names_dataset_names(['US'])``).
    Loaded lazily if omitted; tests should pass a small set to avoid
    the ~32M-row CSV load.
    ``vocab_whitelist``: operator-curated tokens to exempt (e.g.
    ``seizure``, ``focal``). Grows over successive audits.
    ``boilerplate_whitelist`` + ``site_code``: annotation-level filter
    applied BEFORE the token scan, so known-safe recurring phrases
    (e.g. ``"PAT REF EEG"``) are silenced outright and don't spawn
    per-token whitelist entries. See :func:`scan_annotation_texts`.
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
    n_annotations_extracted = 0
    n_skipped_short = 0
    n_skipped_boilerplate = 0
    for p in paths:
        anns = extract_annotations(p)
        n_annotations_extracted += len(anns)
        per_ann, inv, stats = scan_annotation_texts(
            anns, name_set, vocab,
            boilerplate_whitelist=boilerplate_whitelist,
            site_code=site_code,
        )
        n_annotations_scanned += stats["n_scanned"]
        n_skipped_short += stats["n_skipped_short"]
        n_skipped_boilerplate += stats["n_skipped_boilerplate"]
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
        "n_annotations_extracted": n_annotations_extracted,
        "n_annotations_scanned": n_annotations_scanned,
        "n_annotations_skipped_short": n_skipped_short,
        "n_annotations_skipped_boilerplate": n_skipped_boilerplate,
        "min_annotation_length_to_scan": MIN_ANNOTATION_LENGTH_TO_SCAN,
        "site_code": site_code,
        "n_matches": sum(len(v) for v in inverted.values()),
        "matches_by_file": matches_by_file,
        "matched_tokens": inverted,
        "n_vocab_whitelist_tokens": len(vocab),
        "dictionary_size": len(name_set),
        "issues": issues,
    }


def check_annotation_review_state(subject_dir: str | Path,
                                  annotation_carriers: Iterable[str | Path],
                                  ) -> dict:
    """Summarize the manual-annotation-review state for a subject.

    Reads two on-disk artifacts written by ``annotation_review``:
      - ``.annotation_reviewed_tracker`` — one JSONL line per fully-
        reviewed EDF (see [journal.py](../annotation_review/journal.py)).
      - ``.annotation_review/applied/session_*.jsonl`` — every session
        whose edits landed in the EDFs.

    ``state`` values:
      - ``"none"``: no tracker, no applied sessions. Manual review has
        not started. ``annotation_phi_scan`` output is authoritative.
      - ``"partial"``: tracker exists but does not cover every
        annotation carrier. ``annotation_phi_scan`` still worth
        rendering — some flagged files may not have been reviewed yet.
      - ``"complete"``: every annotation carrier is in the tracker.
        The operator has looked at every non-boilerplate annotation.
        Downstream renderers (``_always_print_warnings``, notebook)
        may hide the phi-scan block by default.

    ``status`` is always ``"pass"`` — this is a state summary, not a
    correctness check. A partial-review isn't a failure; it's just
    where the operator paused.
    """
    from clean_eeg.annotation_review.journal import (
        APPLIED_SUBDIR,
        REVIEWED_TRACKER_NAME,
        SESSION_SUBDIR,
        ReviewedTracker,
    )
    import json as _json

    subject_dir = Path(subject_dir)
    carriers = [Path(p) for p in annotation_carriers]
    carrier_names = {p.name for p in carriers}
    tracker_path = subject_dir / REVIEWED_TRACKER_NAME
    tracker_present = tracker_path.exists()

    reviewed_paths: set[str] = set()
    if tracker_present:
        try:
            reviewed_paths = ReviewedTracker(subject_dir).reviewed_paths()
        except (OSError, _json.JSONDecodeError):
            reviewed_paths = set()

    # Tracker stores absolute paths; compare by basename against the
    # audited annotation carriers so that a subject-dir move / rename
    # doesn't invalidate the state.
    reviewed_names = {Path(p).name for p in reviewed_paths}
    reviewed_carriers = carrier_names & reviewed_names
    n_reviewed = len(reviewed_carriers)
    n_carriers = len(carrier_names)

    applied_dir = subject_dir / SESSION_SUBDIR / APPLIED_SUBDIR
    applied_sessions: list[str] = []
    n_edits_applied = 0
    if applied_dir.is_dir():
        for f in sorted(applied_dir.glob("session_*.jsonl")):
            applied_sessions.append(f.name)
            try:
                for line in f.read_text().splitlines():
                    line = line.strip()
                    if line:
                        n_edits_applied += 1
            except OSError:
                continue

    if not tracker_present and not applied_sessions:
        state = "none"
    elif n_carriers > 0 and n_reviewed >= n_carriers:
        state = "complete"
    else:
        state = "partial"

    return {
        "check": "annotation_review_state",
        "status": "pass",
        "state": state,
        "tracker_present": tracker_present,
        "n_reviewed": n_reviewed,
        "n_annotation_carriers": n_carriers,
        "unreviewed_carriers": sorted(carrier_names - reviewed_carriers),
        "n_applied_sessions": len(applied_sessions),
        "applied_sessions": applied_sessions,
        "n_edits_applied": n_edits_applied,
        "issues": [],
    }
