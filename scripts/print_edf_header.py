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
    python scripts/print_edf_header.py PATH
    python scripts/print_edf_header.py PATH --signals 0,1,2
    python scripts/print_edf_header.py PATH --no-signals      # skip per-signal table

PATH may be a file or a directory.
"""

from __future__ import annotations

import argparse
import os
import sys
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


def _format_field_line(start: int, width: int, name: str, raw: bytes,
                       parsed) -> str:
    """One row of the human-readable header dump."""
    end = start + width - 1
    raw_text = raw.decode("ascii", errors="replace")
    # Truncate long string fields for readability — keep raw bytes
    # available via the byte range note.
    raw_disp = raw_text if len(raw_text) <= 80 else raw_text[:77] + "..."
    return (f"  bytes {start:>4}-{end:<4} ({width:>2}B)  "
            f"{name:<19}  raw={raw_disp!r}  "
            f"parsed={parsed!r}")


def read_main_header(edf_path: str) -> dict:
    """Read and parse the EDF main header. Always returns a dict — never
    raises. Each field is in the dict with its parsed value or a sentinel
    string ('<empty>' / '<unparseable: ...>')."""
    with open(edf_path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
    if len(main) < MAIN_HEADER_BYTES:
        return {
            "_truncated_main_header": True,
            "_main_bytes_read": len(main),
        }
    out = {"_main_bytes_read": MAIN_HEADER_BYTES, "_raw_main": main}
    for start, width, name, kind in MAIN_HEADER_FIELDS:
        raw = main[start:start + width]
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


def print_header(edf_path: str, *,
                 signal_indices: Optional[list] = None,
                 print_signals: bool = True,
                 out=sys.stdout) -> None:
    """Print everything we can determine about ``edf_path``'s header."""
    print(f"\n=== {edf_path} ===", file=out)

    file_size = os.path.getsize(edf_path)
    print(f"# filesize: {file_size} bytes", file=out)

    # --- main header ---
    print(f"\n# Main header (bytes 0..{MAIN_HEADER_BYTES - 1})", file=out)
    main = read_main_header(edf_path)
    if main.get("_truncated_main_header"):
        print(f"  ! WARNING: file is shorter than 256 bytes — only "
              f"{main['_main_bytes_read']} bytes available. Cannot parse.",
              file=out)
        return

    for start, width, name, kind in MAIN_HEADER_FIELDS:
        line = _format_field_line(start, width, name,
                                  main[f"_raw_{name}"], main[name])
        print(line, file=out)

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
    indices = (list(range(n_signals)) if signal_indices is None
               else [i for i in signal_indices if 0 <= i < n_signals])
    print(f"\n# Signal headers ({n_signals} signals × "
          f"{SIGNAL_HEADER_BYTES_PER_SIGNAL}B each)", file=out)
    if signal_indices is not None:
        print(f"# (showing only signals {indices})", file=out)

    for i in indices:
        sig = sigs[i]
        print(f"\n  Signal {i}  label={sig['label']!r}", file=out)
        for _, _, name, _ in SIGNAL_HEADER_FIELDS:
            if name == "label":
                continue
            raw = sig[f"_raw_{name}"]
            raw_disp = raw.decode("ascii", errors="replace")
            if len(raw_disp) > 60:
                raw_disp = raw_disp[:57] + "..."
            print(f"    {name:<19}  raw={raw_disp!r}  "
                  f"parsed={sig[name]!r}", file=out)

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


def _gather_paths(path: str) -> list:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".edf")
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
    args = p.parse_args()

    indices = None
    if args.signals:
        indices = [int(s) for s in args.signals.split(",") if s.strip()]

    paths = _gather_paths(args.path)
    if not paths:
        print(f"No .edf files found at {args.path!r}", file=sys.stderr)
        sys.exit(1)

    for edf_path in paths:
        try:
            print_header(edf_path,
                         signal_indices=indices,
                         print_signals=not args.no_signals)
        except Exception as e:
            print(f"\n=== {edf_path} ===", file=sys.stderr)
            print(f"  FAILED to inspect: {type(e).__name__}: {e}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
