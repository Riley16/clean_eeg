"""Verify that files in a subject's quarantine/ subdir have safely-
equivalent counterparts already in place in the parent subject dir.

Motivation: a partial prior-clean state (see
``_prior_clean_artifacts_present`` in clean_subject_eeg.py) can leave
a quarantine/ dir behind full of files that failed a defensive re-write
check on a redundant second-clean pass. If the parent dir still holds
the ORIGINAL first-clean output for each quarantined file (byte-for-
byte equivalent header + signal-header block + sampled data windows),
the quarantine dir is redundant and safe to delete.

This module provides that check as a proper function + CLI so
operators don't have to eyeball a directory listing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from clean_eeg.audit.hashes import sha256_fast_of_file
from clean_eeg.clean_subject_eeg import QUARANTINE_SUFFIX


# The re-clean attempt renames files by appending an EXTRA
# _R1XXXY_MM.DD__HH.MM.SS stamp before the .edf / _annotations.edf
# suffix (doubly-stamped). Recovering the original parent-dir name
# means stripping the LAST such stamp from the pre-quarantine name.
_OUTER_DEID_RE = re.compile(
    r"_R1\d{3}[ACDEFHJMNPST]_\d{2}\.\d{2}__\d{2}\.\d{2}\.\d{2}"
    r"(?=(_annotations)?\.edf$)"
)

MAIN_HEADER_BYTES = 256
SIGNAL_HEADER_BYTES_PER_SIGNAL = 256


def recover_original_name(quarantine_name: str) -> str:
    """Strip the ``.QUARANTINED-DO-NOT-USE`` suffix and the OUTER
    (last-occurring) de-identified filename stamp from a quarantine
    filename, returning the name the file would have had in the parent
    dir before the failed re-clean attempt renamed it.

    Examples:
      >>> recover_original_name(
      ...   "GA_R1665J_01.01__18.08.10_R1665J_01.01__18.08.10.edf.QUARANTINED-DO-NOT-USE")
      'GA_R1665J_01.01__18.08.10.edf'
      >>> recover_original_name("raw_input.edf.QUARANTINED-DO-NOT-USE")
      'raw_input.edf'
    """
    inner = quarantine_name.removesuffix(QUARANTINE_SUFFIX)
    m = _OUTER_DEID_RE.search(inner)
    if m is None:
        # No stamp -> raw un-renamed name; pre-quarantine name == parent.
        return inner
    stripped = inner[:m.start()] + inner[m.end():]
    # Detection: if stripping the trailing stamp leaves ANOTHER deid
    # stamp still anchored to the .edf$ tail, we removed the OUTER
    # (double-stamp) and the parent-dir name is the single-stamped
    # result. If it doesn't, we would over-strip a single-stamp name
    # (removing legit clean output stamp) -> return unchanged.
    if _OUTER_DEID_RE.search(stripped):
        return stripped
    return inner


def _read_headers(path: Path) -> tuple[bytes, bytes, int]:
    """Return ``(main_header_bytes, signal_headers_bytes, n_signals)``.
    Never raises: on any failure returns ``(b"", b"", -1)`` so the
    caller reports it as a mismatch rather than propagating an
    exception."""
    try:
        with open(path, "rb") as f:
            main = f.read(MAIN_HEADER_BYTES)
            try:
                n_signals = int(main[252:256].decode("ascii").strip())
            except (ValueError, UnicodeDecodeError):
                return main, b"", -1
            sig_hdrs = f.read(SIGNAL_HEADER_BYTES_PER_SIGNAL * n_signals)
        return main, sig_hdrs, n_signals
    except OSError:
        return b"", b"", -1


def _pyedflib_loadable(path: Path) -> tuple[bool, str | None]:
    """Return ``(loadable, error_message)``. Wraps pyedflib.EdfReader
    open/close; any exception counts as not-loadable and its str is
    surfaced in the returned tuple for the report."""
    try:
        import pyedflib
    except ImportError as e:
        return False, f"pyedflib import: {e}"
    try:
        with pyedflib.EdfReader(str(path)):
            pass
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@dataclass
class QuarantineFileResult:
    """Verification outcome for one file under quarantine/. All flag
    fields default False; only fields the check actually set to True
    represent verified equality."""
    quarantine_path: Path
    inferred_orig_name: str
    orig_path: Path
    orig_exists: bool = False
    quarantine_loads: bool = False
    orig_loads: bool = False
    header_match: bool = False
    signal_header_match: bool = False
    fast_hash_match: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def fully_equivalent(self) -> bool:
        """True iff the parent counterpart exists, both files load
        cleanly under pyedflib, and every byte-level comparison
        (header, signal headers, fast hash of header + 3 data
        windows) matched. This is the bar for "quarantine copy is
        redundant and safe to delete."""
        return (self.orig_exists and self.quarantine_loads
                and self.orig_loads and self.header_match
                and self.signal_header_match and self.fast_hash_match)


@dataclass
class QuarantineReport:
    """Verification summary for a subject's quarantine/ subdir."""
    subject_dir: Path
    quarantine_dir: Path
    files: list[QuarantineFileResult]

    @property
    def n_total(self) -> int:
        return len(self.files)

    @property
    def n_fully_equivalent(self) -> int:
        return sum(1 for f in self.files if f.fully_equivalent)

    @property
    def all_safe_to_delete(self) -> bool:
        """True iff every quarantine file has a fully-equivalent parent
        counterpart. Callers checking this before rm-rf'ing the
        quarantine dir will not lose data."""
        return self.n_total > 0 and self.n_fully_equivalent == self.n_total


def verify_quarantine_matches_originals(
        subject_inner: Path) -> QuarantineReport:
    """Walk ``<subject_inner>/quarantine/`` and, for each file:
    (1) recover the parent-dir filename the failed re-clean was
    trying to write; (2) confirm that name exists in
    ``subject_inner``; (3) confirm both files open with pyedflib;
    (4) byte-compare the main header + signal-header block;
    (5) fast-hash both (header + 3 sampled data-record windows) and
    compare digests.

    Returns a :class:`QuarantineReport` with per-file outcomes and
    aggregate flags. Never raises -- every per-file failure surfaces
    as a False flag + a note on the result.
    """
    subject_inner = Path(subject_inner)
    q_dir = subject_inner / "quarantine"
    files: list[QuarantineFileResult] = []
    if not q_dir.is_dir():
        return QuarantineReport(subject_dir=subject_inner,
                                 quarantine_dir=q_dir, files=files)

    for qf in sorted(q_dir.iterdir()):
        if not qf.is_file() or not qf.name.endswith(QUARANTINE_SUFFIX):
            continue
        orig_name = recover_original_name(qf.name)
        orig_path = subject_inner / orig_name
        result = QuarantineFileResult(
            quarantine_path=qf,
            inferred_orig_name=orig_name,
            orig_path=orig_path,
        )
        result.orig_exists = orig_path.exists()
        if not result.orig_exists:
            result.notes.append(
                f"no counterpart at {orig_path}")
            files.append(result)
            continue

        q_loads, q_err = _pyedflib_loadable(qf)
        result.quarantine_loads = q_loads
        if q_err:
            result.notes.append(f"quarantine load: {q_err}")
        o_loads, o_err = _pyedflib_loadable(orig_path)
        result.orig_loads = o_loads
        if o_err:
            result.notes.append(f"original load: {o_err}")

        # Cheap fast-fail: size mismatch → definitely not duplicates,
        # skip the header/hash reads. Instant vs multiple network
        # round-trips.
        q_size = qf.stat().st_size
        o_size = orig_path.stat().st_size
        if q_size != o_size:
            result.notes.append(
                f"file size differs (q={q_size} o={o_size}); skipping "
                f"header + hash comparisons")
            files.append(result)
            continue

        q_main, q_sig, q_n = _read_headers(qf)
        o_main, o_sig, o_n = _read_headers(orig_path)
        result.header_match = bool(q_main) and q_main == o_main
        result.signal_header_match = bool(q_sig) and q_sig == o_sig
        if q_n != o_n:
            result.notes.append(f"n_signals differ (q={q_n} o={o_n})")

        # sha256_fast_of_file is already the optimized sampled hash:
        # 256-byte main header + 3 x 2s data-record windows (start /
        # middle / end). ~500 KB read per file even on multi-GB EDFs
        # -- no need for mmap here since we're already reading only
        # the sampled offsets, not scanning the payload.
        try:
            q_h, _mode, _det = sha256_fast_of_file(qf)
            o_h, _mode, _det = sha256_fast_of_file(orig_path)
            result.fast_hash_match = q_h == o_h
        except Exception as e:
            result.notes.append(f"fast_hash: {type(e).__name__}: {e}")

        files.append(result)

    return QuarantineReport(subject_dir=subject_inner,
                             quarantine_dir=q_dir, files=files)


def format_report(report: QuarantineReport) -> str:
    """Multi-line human-readable summary. Fully-equivalent files
    collapse to the counter; only files with any mismatch print
    per-flag detail so a large clean subject dir stays readable."""
    lines = [
        f"QUARANTINE CHECK for {report.subject_dir}",
        f"  quarantine dir: {report.quarantine_dir}",
        f"  total quarantined:                    {report.n_total}",
        f"  fully-equivalent parent counterpart:  "
        f"{report.n_fully_equivalent} / {report.n_total}",
    ]
    for r in report.files:
        if r.fully_equivalent:
            continue
        lines.append(f"\n  {r.quarantine_path.name}")
        lines.append(f"    -> orig_name: {r.inferred_orig_name}")
        lines.append(
            f"    exists={r.orig_exists!s:5}  "
            f"q_loads={r.quarantine_loads!s:5}  "
            f"o_loads={r.orig_loads!s:5}  "
            f"hdr={r.header_match!s:5}  "
            f"sig_hdr={r.signal_header_match!s:5}  "
            f"hash={r.fast_hash_match!s:5}")
        for note in r.notes:
            lines.append(f"    NOTE: {note}")
    if report.n_total == 0:
        lines.append("\n  [!] No quarantine dir (or empty). Nothing to check.")
    elif report.all_safe_to_delete:
        lines.append(
            f"\n  [OK] All {report.n_total} quarantined files have "
            f"byte-equivalent counterparts. Safe to delete:")
        lines.append(f"      rm -rf {report.quarantine_dir}")
    else:
        n_flagged = report.n_total - report.n_fully_equivalent
        lines.append(
            f"\n  [!] {n_flagged} quarantined file(s) do NOT have a "
            f"byte-equivalent parent counterpart. Manually inspect "
            f"before deleting the quarantine dir.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse, sys
    p = argparse.ArgumentParser(
        prog="verify-quarantine-eeg",
        description=(
            "For each file in <subject_inner>/quarantine/, verify "
            "there's a byte-equivalent counterpart already in place "
            "in <subject_inner>. Exit 0 = all quarantined files are "
            "safe duplicates (quarantine dir is redundant). Exit 1 = "
            "at least one quarantined file lacks a safe counterpart "
            "(don't delete quarantine yet)."))
    p.add_argument("subject_inner", type=Path,
                   help="Subject's cleaned dir (contains the quarantine/ "
                        "subdir), typically <subject>/clinical_eeg.")
    args = p.parse_args(argv)
    report = verify_quarantine_matches_originals(args.subject_inner)
    print(format_report(report))
    if report.n_total == 0:
        return 0
    return 0 if report.all_safe_to_delete else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
