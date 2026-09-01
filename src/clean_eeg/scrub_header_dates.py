"""De-identify EDF header dates for a subject whose pipeline run left
startdate / recording_id fields unscrubbed (e.g. R1665J after partial
prior-clean recovery).

Rewrites the 256-byte main header of each recording so the following
byte ranges hold de-identified values:

  * bytes  88-167  ``recording_id`` (contains "Startdate DD-MMM-YYYY ...")
  * bytes 168-175  ``startdate`` ("DD.MM.YY")
  * bytes 176-183  ``starttime`` ("HH.MM.SS")

Signal channel bytes and annotation channel bytes are guaranteed
UNTOUCHED by construction: :func:`update_edf_header_inplace` only ever
writes the header prefix (the first ``256 + 256 * n_signals`` bytes).

Preserves relative offsets across a subject's recordings: the earliest
recording lands at ``BASE_START_DATE`` and every other is shifted by
``(its_startdate - earliest_startdate)``. Callers who prefer a flat
'all files at 1985-01-01 00:00:00' can pass ``preserve_offsets=False``.

CLI:

    python -m clean_eeg.scrub_header_dates \\
        --subject-dir /path/to/R1665J \\
        --audit                # print current + proposed; no writes
    python -m clean_eeg.scrub_header_dates \\
        --subject-dir /path/to/R1665J   # interactive apply
    python -m clean_eeg.scrub_header_dates \\
        --subject-dir /path/to/R1665J \\
        --yes                  # skip per-batch confirm prompt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pyedflib

from clean_eeg.modify_edf_inplace import update_edf_header_inplace


BASE_START_DATE = datetime(1985, 1, 1)
SIDECAR_SUFFIX = "_annotations.edf"


def _list_main_edfs(inner: Path) -> list[Path]:
    return sorted(p for p in inner.iterdir()
                  if p.is_file() and p.suffix.lower() == ".edf"
                  and not p.name.endswith(SIDECAR_SUFFIX))


def _sidecar_for(main: Path) -> Path:
    return main.with_name(main.stem + SIDECAR_SUFFIX)


def _read_startdate(path: Path) -> datetime:
    with pyedflib.EdfReader(str(path)) as f:
        return f.getHeader()["startdate"]


def _read_recording_id(path: Path) -> str:
    """Bytes 88-167 of the main header, stripped."""
    with open(path, "rb") as f:
        f.seek(88)
        return f.read(80).decode("ascii", errors="replace").rstrip()


def compute_shifted_startdates(edf_paths: Iterable[Path],
                                base: datetime = BASE_START_DATE,
                                preserve_offsets: bool = True
                                ) -> dict[Path, datetime]:
    """Return the proposed new startdate per input path.

    ``preserve_offsets=True`` (default): earliest file -> ``base``,
    others shifted by their delta from the earliest. Multi-day
    recordings keep their day-of-recording relative structure.

    ``preserve_offsets=False``: every file -> ``base`` verbatim. Flatter
    but loses relative timing.
    """
    starts = {p: _read_startdate(p) for p in edf_paths}
    if not starts:
        return {}
    if not preserve_offsets:
        return {p: base for p in starts}
    earliest = min(starts.values())
    return {p: base + (s - earliest) for p, s in starts.items()}


def _write_startdate(path: Path, new_start: datetime) -> None:
    """Apply the date change via the corruption-safe primitive.

    The underlying primitive snapshots the header bytes before writing
    and rolls back if post-write validation fails, so a partial write
    can never leave the file in an unloadable state.
    """
    update_edf_header_inplace(str(path), {"startdate": new_start})


def audit_headers(edf_paths: Iterable[Path]) -> list[dict]:
    """Return one dict per file with the header fields relevant to
    date-based re-identification. Includes both main and sidecar files
    if passed."""
    rows = []
    for p in edf_paths:
        with pyedflib.EdfReader(str(p)) as f:
            h = f.getHeader()
        rows.append({
            "path": p,
            "patientname": h.get("patientname"),
            "patientcode": h.get("patientcode"),
            "birthdate": h.get("birthdate"),
            "sex": h.get("sex"),
            "startdate": h.get("startdate"),
            "recording_id": _read_recording_id(p),
        })
    return rows


def _print_audit(rows: list[dict], stream=None) -> None:
    stream = stream or sys.stdout
    for r in rows:
        print(f"  {r['path'].name}", file=stream)
        print(f"    patientname   : {r['patientname']!r}", file=stream)
        print(f"    patientcode   : {r['patientcode']!r}", file=stream)
        print(f"    birthdate     : {r['birthdate']!r}", file=stream)
        print(f"    sex           : {r['sex']!r}", file=stream)
        print(f"    startdate     : {r['startdate']}", file=stream)
        print(f"    recording_id  : {r['recording_id']!r}", file=stream)


def _print_proposal(mains: list[Path],
                     proposed: dict[Path, datetime],
                     stream=None) -> None:
    stream = stream or sys.stdout
    for p in mains:
        orig = _read_startdate(p)
        rid = _read_recording_id(p)
        print(f"  {p.name}", file=stream)
        print(f"    startdate    {orig}  ->  {proposed[p]}", file=stream)
        print(f"    recording_id {rid!r}", file=stream)
        print(f"    (recording_id will be re-derived by pyedflib from the "
              f"new startdate; expect 'Startdate {proposed[p]:%d-%b-%Y}' prefix)",
              file=stream)


def scrub_subject_startdates(inner: Path, *,
                              preserve_offsets: bool = True,
                              include_sidecars: bool = True,
                              base: datetime = BASE_START_DATE
                              ) -> dict[Path, datetime]:
    """Non-interactive helper: compute + apply the shift. Returns the
    (path, new_startdate) map. Callers writing scripts around this
    should print :func:`audit_headers` before and after."""
    mains = _list_main_edfs(inner)
    if not mains:
        return {}
    proposed = compute_shifted_startdates(
        mains, base=base, preserve_offsets=preserve_offsets)
    for main in mains:
        _write_startdate(main, proposed[main])
        if include_sidecars:
            side = _sidecar_for(main)
            if side.exists():
                _write_startdate(side, proposed[main])
    return proposed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="De-identify EDF header startdate/recording_id for a subject.")
    p.add_argument("--subject-dir", type=Path, required=True,
                   help="Path to the subject dir (must contain --subfolder).")
    p.add_argument("--subfolder", type=str, default="clinical_eeg",
                   help="Subfolder under --subject-dir holding the .edf files.")
    p.add_argument("--audit", action="store_true",
                   help="Print current header state for every file (main + "
                        "sidecar) and exit. No writes.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the confirmation prompt after the proposal is "
                        "printed. Still shows the proposal.")
    p.add_argument("--no-sidecars", action="store_true",
                   help="Skip _annotations.edf sidecars. Default updates them "
                        "with the same shift as their main file so sidecar + "
                        "main headers stay consistent.")
    p.add_argument("--no-preserve-offsets", action="store_true",
                   help="Set every file's startdate to BASE_START_DATE verbatim "
                        "(loses relative day-of-recording info). Default "
                        "shifts each file by its delta from the earliest.")
    p.add_argument("--base-date", type=str, default="1985-01-01",
                   help="Anchor date the earliest recording maps to. "
                        "ISO format (YYYY-MM-DD). Default: 1985-01-01.")
    args = p.parse_args(argv)

    inner = args.subject_dir / args.subfolder
    if not inner.exists():
        print(f"[error] {inner} does not exist", file=sys.stderr)
        return 2

    mains = _list_main_edfs(inner)
    if not mains:
        print(f"[error] no main .edf files under {inner}", file=sys.stderr)
        return 2

    # Audit path: dump state and exit.
    if args.audit:
        all_files = mains + [_sidecar_for(m) for m in mains
                              if _sidecar_for(m).exists()]
        print(f"=== Current header state ({len(all_files)} file(s)) ===")
        _print_audit(audit_headers(all_files))
        return 0

    try:
        base = datetime.strptime(args.base_date, "%Y-%m-%d")
    except ValueError:
        print(f"[error] --base-date must be YYYY-MM-DD; got {args.base_date!r}",
              file=sys.stderr)
        return 2

    proposed = compute_shifted_startdates(
        mains, base=base,
        preserve_offsets=not args.no_preserve_offsets)

    print(f"=== Proposed changes ({len(mains)} main file(s)"
          f"{', sidecars included' if not args.no_sidecars else ''}) ===")
    _print_proposal(mains, proposed)

    if not args.yes:
        resp = input(f"\nApply to {len(mains)} main file(s)"
                     f"{' + sidecars' if not args.no_sidecars else ''}? "
                     f"[y/N]: ")
        if resp.strip().lower() != "y":
            print("[abort] no changes applied")
            return 1

    n_ok = 0
    for main in mains:
        try:
            _write_startdate(main, proposed[main])
        except Exception as e:
            print(f"[FAIL] {main.name}: {type(e).__name__}: {e}")
            print(f"[abort] stopping to avoid partial-shift state across the "
                  f"subject. {n_ok} file(s) already updated.")
            return 3
        n_ok += 1
        if not args.no_sidecars:
            side = _sidecar_for(main)
            if side.exists():
                try:
                    _write_startdate(side, proposed[main])
                except Exception as e:
                    print(f"[FAIL] {side.name}: {type(e).__name__}: {e}")
                    return 3
        print(f"[ok] {main.name}")

    print(f"\n=== Post-update audit ===")
    _print_audit(audit_headers(mains + [_sidecar_for(m) for m in mains
                                          if _sidecar_for(m).exists()]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
