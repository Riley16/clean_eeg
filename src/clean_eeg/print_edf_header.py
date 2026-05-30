"""Print EDF header bytes + parsed values for manual debugging.

Operates on a single ``.edf`` file or every ``.edf`` (case-insensitive) in
a directory. Reads the raw bytes itself — does not call pyedflib — so it
works even when pyedflib refuses to open the file (the typical reason
you'd reach for this tool).

Output layout per file:
  * Main header — every byte range, raw bytes, parsed value, parse error.
  * Signal headers — per-signal table of all 10 fields.
  * Derived geometry — total header size, record stride, filesize, an
    OK/truncated/oversized verdict.

Unparseable numeric fields are reported as ``<empty>`` or
``<unparseable: ...>`` rather than crashing the script.

CLI usage:
    print-edf-header PATH
    print-edf-header PATH --signals 0,1,2
    print-edf-header PATH --no-signals       # skip per-signal table
    print-edf-header DIR  -r                 # walk subdirectories
    print-edf-header DIR  -r --include-annotation-stubs

PATH may be a file or a directory. Directory scans skip
``*_annotations.edf`` stubs (paired sibling files written only by
inplace-mode de-identification) unless ``--include-annotation-stubs``.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Optional


# ---- EDF main header layout (bytes within the first 256 bytes) ----
MAIN_HEADER_FIELDS = [
    # (start, width, name, kind)  — kind: 'str' | 'int' | 'float'
    (  0,  8, "version",          "str"),
    (  8, 80, "patient_id",       "str"),
    ( 88, 80, "recording_id",     "str"),
    (168,  8, "startdate",        "str"),
    (176,  8, "starttime",        "str"),
    (184,  8, "bytes_in_header",  "int"),
    (192, 44, "reserved",         "str"),
    (236,  8, "n_records",        "int"),  # -1 sentinel allowed
    (244,  8, "record_duration",  "float"),
    (252,  4, "n_signals",        "int"),
]

# ---- Per-signal header field layout (each field is repeated for ALL
# signals before the next field starts). Widths in bytes per signal. ----
SIGNAL_HEADER_FIELDS = [
    # (offset_within_block_per_signal, width, name, kind)
    (  0, 16, "label",              "str"),
    ( 16, 80, "transducer",         "str"),
    ( 96,  8, "phys_dim",           "str"),
    (104,  8, "phys_min",           "float"),
    (112,  8, "phys_max",           "float"),
    (120,  8, "dig_min",            "int"),
    (128,  8, "dig_max",            "int"),
    (136, 80, "prefilter",          "str"),
    (216,  8, "samples_per_record", "int"),
    (224, 32, "reserved",           "str"),
]

MAIN_HEADER_BYTES = 256
SIGNAL_HEADER_BYTES_PER_SIGNAL = 256


def _parse_value(raw: bytes, kind: str):
    """Decode ``raw`` (a fixed-width header field) and try to parse it as
    ``kind``. Returns the parsed value, or a string sentinel on
    empty/unparseable input. Always returns SOMETHING — never raises."""
    text = raw.decode("ascii", errors="replace").strip()
    if kind == "str":
        return text
    if not text:
        return "<empty>"
    try:
        if kind == "int":
            return int(text)
        if kind == "float":
            return float(text)
    except ValueError as e:
        return f"<unparseable as {kind}: {text!r} ({e})>"
    return text


# Main-header fields that contain PHI per EDF+ spec:
#   patient_id      (bytes 8-87)   — MRN, sex, birthdate, name
#   recording_id    (bytes 88-167) — startdate, hospital admin code,
#                                    technician/investigator name,
#                                    equipment code
#   startdate       (bytes 168-175) — original recording date
#   starttime       (bytes 176-183) — original recording time
# A header dump that lands in a shared log file (e.g. via the
# error-handler diagnostic dump) must mask these. The remaining
# numeric/structural fields (bytes_in_header, n_records,
# record_duration, n_signals) carry no PHI and are exactly what a
# data-team triage needs.
PHI_MAIN_HEADER_FIELDS = frozenset(
    {"patient_id", "recording_id", "startdate", "starttime"}
)


def _render_aligned_field_rows(field_names: list, source: dict, *,
                               indent: str = "  ",
                               redact_phi: bool = False,
                               max_raw_len: int = 80) -> list:
    """Render a block of header fields with the ``raw=`` column
    aligned across rows.

    Each row has the shape ``{indent}{name:<19}  parsed=...  raw=...``.
    The ``parsed=...`` slot is padded with trailing spaces so every
    row's ``raw=`` lands at the same column — what the operator
    actually wants is to sweep their eyes down a single column to spot
    a malformed value, and that requires alignment.

    ``source`` is a dict containing both parsed values (``source[name]``)
    and the corresponding raw bytes (``source[f'_raw_{name}']``); both
    ``read_main_header`` and ``read_signal_headers`` populate it that
    way."""
    parts = []
    for name in field_names:
        if redact_phi and name in PHI_MAIN_HEADER_FIELDS:
            parsed_part = "parsed=[PHI_REDACTED]"
            raw_part = "raw=[PHI_REDACTED]"
        else:
            parsed_part = f"parsed={source[name]!r}"
            raw_text = source[f"_raw_{name}"].decode("ascii", errors="replace")
            raw_disp = (raw_text if len(raw_text) <= max_raw_len
                        else raw_text[:max_raw_len - 3] + "...")
            raw_part = f"raw={raw_disp!r}"
        parts.append((name, parsed_part, raw_part))
    parsed_col_width = max(len(p) for _, p, _ in parts)
    return [
        f"{indent}{name:<19}  {parsed_part:<{parsed_col_width}}  {raw_part}"
        for name, parsed_part, raw_part in parts
    ]


def read_main_header(edf_path: str) -> dict:
    """Read and parse the EDF main header. Always returns a dict — never
    raises. Fields whose byte range falls past end-of-file are marked
    ``<missing — file ends at byte N>`` instead of dropped, so a partial
    header still tells the data team what bytes were present."""
    with open(edf_path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
    n_read = len(main)
    out = {"_main_bytes_read": n_read, "_raw_main": main,
           "_truncated_main_header": n_read < MAIN_HEADER_BYTES}
    for start, width, name, kind in MAIN_HEADER_FIELDS:
        raw = main[start:start + width]
        if len(raw) < width:
            out[name] = f"<missing — file ends at byte {n_read}>"
        else:
            out[name] = _parse_value(raw, kind)
        out[f"_raw_{name}"] = raw
    return out


def read_signal_headers(edf_path: str, n_signals: int) -> list:
    """Read and parse all per-signal headers. Returns a list of dicts,
    one per signal. Always returns SOMETHING — falls back to a list with
    fewer entries if the file is shorter than the header claims."""
    expected = SIGNAL_HEADER_BYTES_PER_SIGNAL * n_signals
    with open(edf_path, "rb") as f:
        f.seek(MAIN_HEADER_BYTES)
        block = f.read(expected)
    actual_signals = min(n_signals, len(block) // (expected // n_signals or 1))
    actual_signals = min(actual_signals, n_signals)

    sigs = []
    for i in range(n_signals):
        sig = {"_signal_idx": i}
        for offset_per_sig, width, name, kind in SIGNAL_HEADER_FIELDS:
            # Per-signal layout: all N labels, then all N transducers, ...
            slot_start = offset_per_sig * n_signals + i * width
            slot_end = slot_start + width
            raw = block[slot_start:slot_end] if slot_end <= len(block) else b""
            sig[name] = _parse_value(raw, kind) if raw else "<missing>"
            sig[f"_raw_{name}"] = raw
        sigs.append(sig)
    return sigs


DEFAULT_SIGNAL_PREVIEW_COUNT = 1


def print_header(edf_path: str, *,
                 signal_indices: Optional[list] = None,
                 print_signals: bool = True,
                 full_signal: bool = False,
                 redact_phi: bool = False,
                 out=None) -> None:
    """Print everything we can determine about ``edf_path``'s header.

    When ``redact_phi=True``, the four main-header fields that carry
    identifying information per EDF+ spec (``patient_id``,
    ``recording_id``, ``startdate``, ``starttime``) are masked as
    ``[PHI_REDACTED]``. Use this when the dump is going to a shared log
    file. Default is ``False`` (full content) so the standalone
    ``print-edf-header`` command shows the operator everything when
    debugging their own files.

    ``out`` defaults to whatever ``sys.stdout`` points to at call time
    (late-binding) so the dump cooperates with pytest's ``capsys``,
    Python's ``contextlib.redirect_stdout``, and the pipeline's
    ``_TeeStream`` log capture. Pass an explicit file-like object when
    you want a different destination."""
    if out is None:
        out = sys.stdout
    print(f"\n=== {edf_path} ===", file=out)
    if redact_phi:
        print(
            "# (PHI fields — patient_id, recording_id, startdate, "
            "starttime — are masked in this dump)",
            file=out,
        )

    file_size = os.path.getsize(edf_path)
    print(f"# filesize: {file_size / (1024 ** 3):.3f} GB", file=out)

    # --- main header ---
    print(f"\n# Main header", file=out)
    main = read_main_header(edf_path)
    if main.get("_truncated_main_header"):
        print(f"  ! WARNING: file is shorter than {MAIN_HEADER_BYTES} bytes — "
              f"only {main['_main_bytes_read']} bytes available. "
              f"Showing fields that fit; the rest are marked missing.",
              file=out)

    main_field_names = [name for _, _, name, _ in MAIN_HEADER_FIELDS]
    for line in _render_aligned_field_rows(main_field_names, main,
                                           indent="  ",
                                           redact_phi=redact_phi,
                                           max_raw_len=80):
        print(line, file=out)

    # Signal headers live past the main header — skip if main is incomplete.
    if main.get("_truncated_main_header"):
        return

    # --- signal headers ---
    n_signals = main.get("n_signals")
    if not isinstance(n_signals, int) or n_signals <= 0:
        print(f"\n# Cannot dump signal headers: n_signals={n_signals!r} "
              f"is not a usable positive integer.", file=out)
        return
    if not print_signals:
        print(f"\n# Skipping per-signal table ({n_signals} signals; "
              f"--no-signals was set)", file=out)
        return

    sigs = read_signal_headers(edf_path, n_signals)
    # Selection priority: explicit --signals > --full-signal > default
    # one-example preview. The default keeps multi-hundred-channel NK
    # exports readable: print every channel's label so the operator
    # can see what's in the file, then dump one full signal block as a
    # representative example of the per-field formatting.
    if signal_indices is not None:
        indices = [i for i in signal_indices if 0 <= i < n_signals]
        selection_note = f"# (showing only signals {indices})"
        print_label_summary = False
    elif full_signal:
        indices = list(range(n_signals))
        selection_note = None
        print_label_summary = False
    else:
        indices = list(range(min(DEFAULT_SIGNAL_PREVIEW_COUNT, n_signals)))
        selection_note = None
        # Skip the summary if there's only one signal anyway — the
        # full block right below already shows its label.
        print_label_summary = n_signals > len(indices)

    print(f"\n# Signal headers ({n_signals} signals)", file=out)
    if print_label_summary:
        labels_text = ", ".join(repr(sigs[i]["label"]) for i in range(n_signals))
        print(textwrap.fill(labels_text, width=76,
                            initial_indent="# Labels: ",
                            subsequent_indent="#         "),
              file=out)
        print(f"# (showing {len(indices)} example signal header below; "
              f"pass --full-signal to dump every signal, or --signals "
              f"i,j,k to pick specific ones)", file=out)
    elif selection_note:
        print(selection_note, file=out)

    signal_field_names = [name for _, _, name, _ in SIGNAL_HEADER_FIELDS
                          if name != "label"]
    for i in indices:
        sig = sigs[i]
        print(f"\n  Signal {i}  label={sig['label']!r}", file=out)
        for line in _render_aligned_field_rows(signal_field_names, sig,
                                               indent="    ",
                                               redact_phi=False,
                                               max_raw_len=60):
            print(line, file=out)

    # --- derived geometry / status ---
    print(f"\n# Derived geometry", file=out)
    expected_header_bytes = MAIN_HEADER_BYTES + n_signals * SIGNAL_HEADER_BYTES_PER_SIGNAL
    print(f"  expected header bytes  = 256 + {n_signals} * 256 "
          f"= {expected_header_bytes}", file=out)

    spr_values = [s["samples_per_record"] for s in sigs]
    if all(isinstance(v, int) and v > 0 for v in spr_values):
        record_bytes = sum(spr_values) * 2  # int16 samples
        data_bytes = file_size - expected_header_bytes
        n_records_claimed = main.get("n_records")
        n_records_actual = data_bytes // record_bytes if record_bytes else "?"
        print(f"  record stride          = sum(spr) * 2 = {record_bytes} bytes",
              file=out)
        print(f"  data bytes on disk     = {data_bytes}", file=out)
        print(f"  n_records claimed      = {n_records_claimed!r}", file=out)
        print(f"  n_records computed     = {n_records_actual}", file=out)
        if isinstance(n_records_claimed, int) and isinstance(n_records_actual, int):
            if n_records_actual == n_records_claimed:
                verdict = "OK (filesize matches header)"
            elif n_records_actual < n_records_claimed:
                verdict = (f"TRUNCATED — header claims more records than disk "
                           f"holds ({n_records_claimed - n_records_actual} missing)")
            else:
                verdict = (f"OVER-SIZED — disk has more data than header claims "
                           f"({n_records_actual - n_records_claimed} extra records' "
                           f"worth, possibly trailing junk)")
            print(f"  verdict                = {verdict}", file=out)
    else:
        print(f"  (cannot compute record stride — samples_per_record contains "
              f"non-integer/non-positive values: {spr_values})", file=out)

    # --- annotations ---
    # Always run after geometry: the operator typically wants to
    # inspect annotation text in stubs, AND wants a strip-check
    # warning when the file is the main EDF of an inplace-mode pair.
    _print_annotations_section(
        edf_path, sigs, n_signals,
        n_records=main.get("n_records"),
        header_bytes_total=expected_header_bytes,
        file_size=file_size,
        redact_phi=redact_phi,
        out=out,
    )


ANNOTATION_STUB_SUFFIX = "_annotations.edf"

# Per EDF+ spec, the annotation channel's label is the literal
# "EDF Annotations" string padded to 16 bytes with spaces. We strip
# whitespace before comparing.
EDF_ANNOTATION_LABEL = "EDF Annotations"

# How many non-timekeeping events to enumerate in the dump before
# truncating with a "...and N more" line. Tuned to fit typical NK
# stubs (a few dozen events) without flooding when a file has
# thousands of annotations.
ANNOTATION_PREVIEW_LIMIT = 50


def _find_annotation_signal_index(sigs: list) -> Optional[int]:
    """Return the index of the first ``EDF Annotations`` signal, or
    ``None`` if the file has no annotation channel."""
    for i, sig in enumerate(sigs):
        label = str(sig.get("label", "")).strip()
        if label == EDF_ANNOTATION_LABEL:
            return i
    return None


def _parse_record_tals(record_bytes: bytes) -> list:
    """Parse one annotation record into a list of TALs.

    Each TAL is ``(onset, duration_or_None, [texts])``. Per EDF+ spec
    item 2.2.4 the format is::

        +onset[\\x15duration]\\x14text[\\x14text2[\\x14...]]\\x00

    TALs are separated by null-byte padding inside the record. The
    first TAL of each record is the "timekeeping" TAL — it carries the
    record's start time and an empty text list (rendered as a trailing
    ``\\x14\\x14`` before the ``\\x00``).

    Malformed TALs (unparseable onset, missing field separator) are
    skipped silently — the function never raises. Returns an empty
    list for a record that is entirely null-padding."""
    tals = []
    for chunk in record_bytes.split(b"\x00"):
        if not chunk:
            continue
        parts = chunk.split(b"\x14")
        if len(parts) < 2:
            continue  # missing the onset/text separator
        onset_part = parts[0]
        if b"\x15" in onset_part:
            o, d = onset_part.split(b"\x15", 1)
            try:
                onset = float(o.decode("ascii"))
                duration = float(d.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            try:
                onset = float(onset_part.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                continue
            duration = None
        # parts[1:] are text annotations, with a trailing empty entry
        # from the closing \x14 before \x00. Skip the empties so a
        # timekeeping TAL renders as an empty text list.
        texts = [tb.decode("utf-8", errors="replace")
                 for tb in parts[1:] if tb]
        tals.append((onset, duration, texts))
    return tals


def _read_annotation_blocks(edf_path: str, sigs: list, n_records: int,
                            ann_idx: int, header_bytes_total: int,
                            file_size: int) -> list:
    """Read the annotation-channel byte block for each data record.

    Uses plain seek+read rather than mmap so the helper stays usable
    on Windows (where mmap of small slices of a 3+ GB file can
    behave oddly without memoryview gymnastics). Annotation channels
    are tiny — typically a few hundred bytes per record — so the
    seek/read overhead is negligible relative to actual debugging
    workflows.

    Returns a list of byte blocks (length ≤ n_records). Stops early if
    the file is truncated past the requested record."""
    samples_per_record = [s.get("samples_per_record") for s in sigs]
    if not all(isinstance(spr, int) and spr >= 0 for spr in samples_per_record):
        return []
    record_bytes = sum(samples_per_record) * 2  # int16 samples
    ann_offset_in_record = sum(samples_per_record[:ann_idx]) * 2
    ann_bytes_per_record = samples_per_record[ann_idx] * 2
    if record_bytes <= 0 or ann_bytes_per_record <= 0:
        return []

    blocks = []
    with open(edf_path, "rb") as f:
        for r in range(n_records):
            start = header_bytes_total + r * record_bytes + ann_offset_in_record
            end = start + ann_bytes_per_record
            if end > file_size:
                break
            f.seek(start)
            data = f.read(ann_bytes_per_record)
            if len(data) < ann_bytes_per_record:
                break
            blocks.append(data)
    return blocks


def _has_annotation_stub_sibling(edf_path: str) -> bool:
    """True iff a sibling ``<basename>_annotations.edf`` file exists in
    the same directory. Used to decide whether the file we're
    inspecting is supposed to have had its annotations stripped (i.e.
    the file is the main EDF of an inplace-mode de-id pair)."""
    if _is_annotation_stub(os.path.basename(edf_path)):
        return False
    base, ext = os.path.splitext(edf_path)
    if ext.lower() != ".edf":
        return False
    return os.path.exists(base + ANNOTATION_STUB_SUFFIX)


def _print_annotations_section(edf_path: str, sigs: list, n_signals: int,
                               n_records: int, header_bytes_total: int,
                               file_size: int, *,
                               redact_phi: bool, out) -> None:
    """Append an ``# Annotations`` section to the dump.

    Behavior:
    - No annotation channel → one-line note, return.
    - Annotation channel present → counts of timekeeping vs.
      non-timekeeping TALs, plus a preview of up to
      ``ANNOTATION_PREVIEW_LIMIT`` non-timekeeping events.
    - File looks like the main EDF of an inplace-mode de-id pair (a
      sibling ``*_annotations.edf`` exists) but still contains
      non-timekeeping TALs → big banner warning so the operator does
      not ship a half-de-identified file.
    - ``redact_phi=True`` masks annotation text content (onset and
      duration are not PHI per spec)."""
    print(f"\n# Annotations", file=out)
    ann_idx = _find_annotation_signal_index(sigs)
    if ann_idx is None:
        print("  (no annotation channel — main EDF carries data only, "
              "annotations may live in a sibling stub)", file=out)
        return

    label = sigs[ann_idx].get("label")
    print(f"  channel index: {ann_idx}  label={label!r}", file=out)

    if not isinstance(n_records, int) or n_records <= 0:
        print(f"  (cannot read TALs: n_records={n_records!r})", file=out)
        return

    blocks = _read_annotation_blocks(edf_path, sigs, n_records, ann_idx,
                                     header_bytes_total, file_size)
    if not blocks:
        print("  (annotation channel layout unparseable or file "
              "truncated before any record)", file=out)
        return
    print(f"  records read: {len(blocks)} of {n_records}", file=out)

    timekeeping_count = 0
    non_tk_events = []  # (record_idx, onset, duration, texts)
    for record_idx, block in enumerate(blocks):
        tals = _parse_record_tals(block)
        for tal_idx, (onset, duration, texts) in enumerate(tals):
            # Position-based classification per EDF+ spec: first TAL
            # in each record IS the timekeeping TAL by definition. A
            # malformed file with extra leading TALs would mis-tag
            # the first as timekeeping, but that's still useful — the
            # extra TALs would then count as non-timekeeping and trip
            # the warning.
            if tal_idx == 0:
                timekeeping_count += 1
                # Some files attach text to the timekeeping TAL too;
                # report those as non-tk so they don't slip past the
                # strip check.
                if texts:
                    non_tk_events.append((record_idx, onset, duration, texts))
            else:
                non_tk_events.append((record_idx, onset, duration, texts))

    print(f"  timekeeping TALs: {timekeeping_count}", file=out)
    print(f"  non-timekeeping events: {len(non_tk_events)}", file=out)

    if non_tk_events:
        preview = non_tk_events[:ANNOTATION_PREVIEW_LIMIT]
        for record_idx, onset, duration, texts in preview:
            text_disp = "[PHI_REDACTED]" if redact_phi else " | ".join(
                repr(t) for t in texts)
            dur_disp = f"{duration:.6g}" if duration is not None else "-"
            print(f"    record={record_idx:<5} onset={onset:>10.4f}  "
                  f"dur={dur_disp:>8}  text={text_disp}", file=out)
        if len(non_tk_events) > ANNOTATION_PREVIEW_LIMIT:
            print(f"    ...and {len(non_tk_events) - ANNOTATION_PREVIEW_LIMIT} "
                  f"more (raise ANNOTATION_PREVIEW_LIMIT to see all)",
                  file=out)

    # --- strip-check warning ---
    if non_tk_events and _has_annotation_stub_sibling(edf_path):
        banner = "#" * 72
        print(f"\n  {banner}", file=out)
        print(f"  # WARNING: this main EDF still contains "
              f"{len(non_tk_events)} non-timekeeping annotation(s),",
              file=out)
        print(f"  # but a sibling '*_annotations.edf' stub exists. "
              f"Inplace-mode de-identification", file=out)
        print(f"  # was supposed to strip these. The pipeline likely "
              f"failed mid-run and the", file=out)
        print(f"  # remaining annotations may carry PHI. DO NOT SHARE "
              f"this file before re-running", file=out)
        print(f"  # the strip step (clear_edf_annotations_inplace).",
              file=out)
        print(f"  {banner}", file=out)


def _is_annotation_stub(filename: str) -> bool:
    """Match the sibling stubs written by inplace-mode de-identification
    (see clean_subject_eeg.deidentify_edf). Only those files have the
    exact ``_annotations.edf`` suffix; main EDFs go through
    ``_R1XXXY_MM.DD__HH.MM.SS.edf``."""
    return filename.lower().endswith(ANNOTATION_STUB_SUFFIX.lower())


def _gather_paths(path: str, *, recursive: bool = False,
                  include_annotation_stubs: bool = False) -> list:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        if recursive:
            collected = []
            for root, _dirs, files in os.walk(path):
                for f in files:
                    if not f.lower().endswith(".edf"):
                        continue
                    if not include_annotation_stubs and _is_annotation_stub(f):
                        continue
                    collected.append(os.path.join(root, f))
            return sorted(collected)
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".edf")
            and (include_annotation_stubs or not _is_annotation_stub(f))
        )
    raise FileNotFoundError(f"No such file or directory: {path}")


def main():
    p = argparse.ArgumentParser(
        description="Print EDF header bytes and parsed values for debugging.",
        epilog="Reads bytes directly — does not depend on pyedflib being "
               "able to open the file."
    )
    p.add_argument("path", help="Path to a .edf file OR a directory of .edf files")
    p.add_argument("--signals", type=str, default=None,
                   help="Comma-separated signal indices to print "
                        "(default: all). Example: --signals 0,1,5")
    p.add_argument("--no-signals", action="store_true",
                   help="Skip the per-signal table; only print main header.")
    p.add_argument("--full-signal", action="store_true",
                   help="Print every signal header in full. Default is "
                        "a comma-separated list of all signal labels "
                        f"plus {DEFAULT_SIGNAL_PREVIEW_COUNT} example "
                        "signal block — enough to spot a malformed "
                        "transducer/dim/range pattern without flooding "
                        "the terminal on multi-hundred-channel NK "
                        "exports. Ignored when --signals is set "
                        "explicitly.")
    p.add_argument("--redact-phi", action="store_true",
                   help="Mask the four main-header fields that carry PHI "
                        "(patient_id, recording_id, startdate, starttime). "
                        "Use when sharing the output with a third party.")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="When PATH is a directory, walk subdirectories "
                        "for .edf files. Sibling '*_annotations.edf' "
                        "stubs (from inplace-mode de-identification) are "
                        "skipped unless --include-annotation-stubs is set.")
    p.add_argument("--include-annotation-stubs", action="store_true",
                   help="Include '*_annotations.edf' stubs when scanning a "
                        "directory. Off by default — those files only carry "
                        "the annotation channel and are paired with a "
                        "main EDF that holds the signal data.")
    args = p.parse_args()

    indices = None
    if args.signals:
        indices = [int(s) for s in args.signals.split(",") if s.strip()]

    paths = _gather_paths(
        args.path,
        recursive=args.recursive,
        include_annotation_stubs=args.include_annotation_stubs,
    )
    if not paths:
        print(f"No .edf files found at {args.path!r}", file=sys.stderr)
        sys.exit(1)

    for edf_path in paths:
        try:
            print_header(edf_path,
                         signal_indices=indices,
                         print_signals=not args.no_signals,
                         full_signal=args.full_signal,
                         redact_phi=args.redact_phi)
        except Exception as e:
            print(f"\n=== {edf_path} ===", file=sys.stderr)
            print(f"  FAILED to inspect: {type(e).__name__}: {e}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
