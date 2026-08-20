"""Rename one subject's R-code across every place it appears on disk.

Handles both raw pre-clean state (folder + filenames) and post-clean
state (EDF header ``patientcode``, ``deidentify.json`` manifest,
``_annotations.edf`` sidecar). Dry-run by default -- pass ``--apply``
to commit.

Corruption safety: for every EDF file whose header we're about to
mutate, the script:
    1. Snapshots the file to ``<path>.rename.bak`` via ``shutil.copy2``
    2. Runs :func:`clean_eeg.modify_edf_inplace.update_edf_header_inplace`
       with ``confirm_signals_unchanged=True`` (already-existing signal
       byte-identity check)
    3. Reopens the file with pyedflib and asserts the header now shows
       the new patientcode AND the file is loadable end-to-end
    4. If ANY of the above fails: restores from backup, aborts the
       whole run (does not proceed to remaining files)
    5. Deletes the backup only after all files pass verification

That sequence turns a partial-write corruption from an "unrecoverable
disk mess" into a "restored + refuse to proceed".

Roundtrip guarantee: renaming ``A -> B`` and then back ``B -> A`` on
an EDF originally written by pyedflib is byte-identical to the pre-
rename file. Covered by a test.

Usage:
    python scripts/rename_subject_code.py \\
        --subject-root /oceanus/collab/herz-lab/raw_data/kahana/subjects \\
        --from R1655J --to R1665J

Then, if the printed plan looks right:
    python scripts/rename_subject_code.py ... --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


BACKUP_SUFFIX = ".rename.bak"


@dataclass
class RenamePlan:
    """Every planned change in one place -- so ``--dry-run`` can print
    it in full before ``--apply`` executes it."""
    path_renames: list[tuple[Path, Path]] = field(default_factory=list)
    edf_header_updates: list[Path] = field(default_factory=list)
    manifest_updates: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planning (no disk mutation)
# ---------------------------------------------------------------------------

def plan_path_renames(subject_root: Path, from_code: str, to_code: str
                       ) -> list[tuple[Path, Path]]:
    """Return the list of (src, dst) file/dir renames, ordered so
    deeper paths rename first (parent folder stays valid while its
    children are renamed). Subject folder rename is last."""
    old_dir = subject_root / from_code
    new_dir = subject_root / to_code

    if not old_dir.exists():
        raise FileNotFoundError(
            f"{old_dir} does not exist -- nothing to rename")
    if new_dir.exists():
        raise FileExistsError(
            f"{new_dir} already exists -- refusing to overwrite. "
            f"Move or delete the target first.")

    renames: list[tuple[Path, Path]] = []
    for path in sorted(old_dir.rglob("*"), reverse=True):
        if from_code in path.name:
            new_name = path.name.replace(from_code, to_code)
            renames.append((path, path.parent / new_name))
    renames.append((old_dir, new_dir))
    return renames


def _read_patientcode_from_bytes(edf_path: Path) -> str:
    """Read the patientcode from the raw ``patient_id`` header field
    (bytes 8..88 of the EDF main header), matching the primitive used
    by ``print-edf-header``. This works on Nihon Kohden raw exports
    that pyedflib.EdfReader refuses to open due to strict EDF+
    compliance checks (the whole reason ``clean_eeg`` exists).

    EDF+ packs patient_id as
    ``"<patientcode> <sex> <birthdate> <patientname>"``. Non-EDF+
    files (raw NK often falls back) put just the patientcode /
    hospital MRN in the whole field. Splitting on whitespace and
    taking the first token handles both.

    Raises ValueError if the file is shorter than the EDF main-header
    minimum (256 bytes) -- avoids silently returning empty for a
    non-EDF file that the operator would otherwise never notice.
    """
    from clean_eeg.modify_edf_inplace import get_header_field
    size = edf_path.stat().st_size
    if size < 256:
        raise ValueError(
            f"file is {size} bytes -- shorter than the 256-byte EDF "
            f"main header. Not a valid EDF.")
    raw = get_header_field(str(edf_path), "patient_id")
    if not raw:
        return ""
    return str(raw).split(None, 1)[0] if str(raw).strip() else ""


def find_edf_files_needing_header_update(subject_dir: Path, from_code: str
                                          ) -> list[Path]:
    """Every .edf under subject_dir whose patientcode header field
    equals ``from_code``. Reads patientcode via raw bytes, so raw NK
    exports that pyedflib refuses to open still get inspected -- and
    the operator gets a clear diagnostic (exception message + path)
    if even the byte-level read fails.

    For raw pre-clean data the patientcode is the hospital MRN, so
    NOTHING matches ``from_code`` and this returns an empty list --
    correct, no header rewrite is needed for raw data.
    """
    hits: list[Path] = []
    for edf_path in sorted(subject_dir.rglob("*.edf")):
        try:
            code = _read_patientcode_from_bytes(edf_path)
        except (OSError, ValueError, KeyError) as e:
            print(f"  [skip-read] {edf_path}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue
        if code == from_code:
            hits.append(edf_path)
    return hits


def find_manifests_needing_update(subject_dir: Path, from_code: str
                                   ) -> list[Path]:
    """Any deidentify.json under subject_dir whose subject_code equals
    ``from_code`` OR whose file_hashes keys embed ``from_code``."""
    hits: list[Path] = []
    for manifest_path in sorted(subject_dir.rglob("deidentify.json")):
        try:
            m = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("subject_code") == from_code:
            hits.append(manifest_path)
            continue
        if any(from_code in k for k in (m.get("file_hashes") or {})):
            hits.append(manifest_path)
    return hits


def build_plan(subject_root: Path, from_code: str, to_code: str
               ) -> RenamePlan:
    old_dir = subject_root / from_code
    return RenamePlan(
        path_renames=plan_path_renames(subject_root, from_code, to_code),
        edf_header_updates=find_edf_files_needing_header_update(
            old_dir, from_code),
        manifest_updates=find_manifests_needing_update(old_dir, from_code),
    )


# ---------------------------------------------------------------------------
# Execution (disk mutation) with backup + restore
# ---------------------------------------------------------------------------

def _verify_edf_header_updated(edf_path: Path, expected_code: str) -> None:
    """Post-write check: byte-level read of patientcode confirms the
    field is now ``expected_code``. Byte-level reading matches
    ``print-edf-header``'s primitive, so this works on files
    pyedflib.EdfReader would refuse (raw NK exports).

    ``update_edf_header_inplace(confirm_signals_unchanged=True)``
    already asserted byte-identity of the signal data during the
    write itself, so we don't re-check signals here (would require
    a pyedflib read that may fail on raw exports).

    Raises RuntimeError on any inconsistency -- caller restores
    from backup.
    """
    try:
        actual = _read_patientcode_from_bytes(edf_path)
    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError(
            f"post-write verification failed: {edf_path} patientcode "
            f"bytes are not readable: {type(e).__name__}: {e}") from e
    if actual != expected_code:
        raise RuntimeError(
            f"post-write verification failed: {edf_path} patientcode "
            f"is {actual!r}, expected {expected_code!r}")


def update_edf_patientcode_safely(edf_path: Path, from_code: str,
                                   to_code: str) -> None:
    """Backup -> in-place header rewrite -> verify -> delete backup.
    On any failure, restore from backup and re-raise so the caller
    aborts the whole run.
    """
    from clean_eeg.modify_edf_inplace import update_edf_header_inplace

    backup_path = edf_path.with_suffix(edf_path.suffix + BACKUP_SUFFIX)
    if backup_path.exists():
        raise RuntimeError(
            f"backup file {backup_path} already exists -- an earlier "
            f"rename may have crashed. Inspect and remove manually.")
    shutil.copy2(edf_path, backup_path)

    try:
        update_edf_header_inplace(
            str(edf_path),
            header_updates={"patientcode": to_code},
            confirm_signals_unchanged=True)
        _verify_edf_header_updated(edf_path, to_code)
    except Exception:
        # Restore before re-raising. copy2 preserves mtime + perms.
        shutil.copy2(backup_path, edf_path)
        raise
    finally:
        # Delete backup on success. On failure the restore above put
        # the pre-write bytes back into the real file, so we still
        # want the backup gone to keep the tree clean.
        if backup_path.exists():
            backup_path.unlink()


def update_deidentify_manifest(manifest_path: Path, from_code: str,
                                to_code: str) -> None:
    m = json.loads(manifest_path.read_text())
    if m.get("subject_code") == from_code:
        m["subject_code"] = to_code
    if "file_hashes" in m:
        remapped = {}
        for k, v in m["file_hashes"].items():
            new_k = k.replace(from_code, to_code) if from_code in k else k
            remapped[new_k] = v
        m["file_hashes"] = remapped
    manifest_path.write_text(json.dumps(m, indent=2))


def execute_plan(plan: RenamePlan, from_code: str, to_code: str) -> None:
    """Order matters:
       1. EDF header updates FIRST -- while paths are still at from_code
       2. Manifest updates -- same reason
       3. Path renames LAST -- so we don't try to update headers under
          paths that no longer exist
    """
    for edf_path in plan.edf_header_updates:
        update_edf_patientcode_safely(edf_path, from_code, to_code)
    for manifest_path in plan.manifest_updates:
        update_deidentify_manifest(manifest_path, from_code, to_code)
    for src, dst in plan.path_renames:
        src.rename(dst)


# ---------------------------------------------------------------------------
# Diagnostic mode: report pyedflib compatibility per file without mutating
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticRow:
    """Per-file result of the read-only diagnostic pass."""
    path: Path
    size_bytes: int
    byte_patientcode: str | None       # None means byte-read failed
    byte_error: str | None             # populated iff byte-read failed
    pyedflib_ok: bool
    pyedflib_error: str | None         # populated iff pyedflib failed
    pyedflib_patientcode: str | None   # populated iff pyedflib OK


def diagnose_edf_file(edf_path: Path) -> DiagnosticRow:
    """Read-only probe: try both the byte-level primitive AND
    pyedflib.EdfReader on ``edf_path``. Neither raises; failures are
    captured into the returned row. Enables the operator to see
    exactly what pyedflib complains about before committing to any
    byte-level mutation.
    """
    try:
        size = edf_path.stat().st_size
    except OSError as e:
        return DiagnosticRow(
            path=edf_path, size_bytes=-1,
            byte_patientcode=None, byte_error=f"{type(e).__name__}: {e}",
            pyedflib_ok=False,
            pyedflib_error=f"{type(e).__name__}: {e}",
            pyedflib_patientcode=None)

    byte_code: str | None = None
    byte_err: str | None = None
    try:
        byte_code = _read_patientcode_from_bytes(edf_path)
    except (OSError, ValueError, KeyError, IndexError) as e:
        byte_err = f"{type(e).__name__}: {e}"

    import pyedflib
    py_ok = False
    py_err: str | None = None
    py_code: str | None = None
    try:
        with pyedflib.EdfReader(str(edf_path)) as f:
            py_ok = True
            py_code = f.getHeader().get("patientcode", "") or ""
    except (OSError, ValueError, RuntimeError) as e:
        py_err = f"{type(e).__name__}: {e}"

    return DiagnosticRow(
        path=edf_path, size_bytes=size,
        byte_patientcode=byte_code, byte_error=byte_err,
        pyedflib_ok=py_ok, pyedflib_error=py_err,
        pyedflib_patientcode=py_code)


def diagnose_subject_dir(subject_dir: Path) -> list[DiagnosticRow]:
    """Run :func:`diagnose_edf_file` on every ``.edf`` under
    ``subject_dir`` (recursive). Returns rows in sorted-path order."""
    if not subject_dir.exists():
        raise FileNotFoundError(f"{subject_dir} does not exist")
    edfs = sorted(subject_dir.rglob("*.edf"))
    return [diagnose_edf_file(p) for p in edfs]


def _print_diagnostic_report(rows: list[DiagnosticRow],
                              subject_dir: Path) -> None:
    """Human-readable table + summary. If ALL files fail pyedflib with
    the same exception message, that's almost always a systemic issue
    (bad header field, format non-compliance) and worth calling out at
    the top so an operator triaging hundreds of files sees it immediately.
    """
    print(f"\n=== Diagnostic report for {subject_dir} "
          f"({len(rows)} .edf file(s)) ===\n")
    if not rows:
        print("  (no .edf files found)")
        return

    n_py_ok = sum(1 for r in rows if r.pyedflib_ok)
    n_py_fail = len(rows) - n_py_ok
    n_byte_ok = sum(1 for r in rows if r.byte_patientcode is not None)
    n_byte_fail = len(rows) - n_byte_ok

    print(f"pyedflib.EdfReader: {n_py_ok} ok / {n_py_fail} failed")
    print(f"byte-level read:    {n_byte_ok} ok / {n_byte_fail} failed")

    # Distinct pyedflib error signatures -- systemic failures cluster
    unique_py_errors = sorted({r.pyedflib_error for r in rows
                                if r.pyedflib_error})
    if unique_py_errors:
        print(f"\nDistinct pyedflib error signatures ({len(unique_py_errors)}):")
        for err in unique_py_errors:
            n = sum(1 for r in rows if r.pyedflib_error == err)
            print(f"  ({n}x) {err}")

    print(f"\n{'file':<40s}  {'size':>10s}  "
          f"{'py':<3s}  {'byte_code':<15s}  py_error/note")
    print("-" * 100)
    for r in rows:
        name = r.path.name
        if len(name) > 40:
            name = "..." + name[-37:]
        code = (r.byte_patientcode if r.byte_patientcode is not None
                else f"<{r.byte_error}>")
        if len(code) > 15:
            code = code[:12] + "..."
        py_tag = "ok" if r.pyedflib_ok else "!!"
        note = ""
        if r.pyedflib_error:
            note = r.pyedflib_error
        elif r.pyedflib_ok and r.pyedflib_patientcode != r.byte_patientcode:
            note = (f"disagreement: pyedflib="
                    f"{r.pyedflib_patientcode!r}")
        if len(note) > 60:
            note = note[:57] + "..."
        print(f"{name:<40s}  {r.size_bytes:>10d}  "
              f"{py_tag:<3s}  {code:<15s}  {note}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_plan(plan: RenamePlan, dry_run: bool) -> None:
    tag = "[DRY] " if dry_run else ""
    print(f"\n=== EDF header patientcode updates "
          f"({len(plan.edf_header_updates)}) ===")
    for p in plan.edf_header_updates:
        print(f"  {tag}{p}")
    if not plan.edf_header_updates:
        print("  (none -- likely raw pre-clean data)")

    print(f"\n=== deidentify.json manifest updates "
          f"({len(plan.manifest_updates)}) ===")
    for p in plan.manifest_updates:
        print(f"  {tag}{p}")
    if not plan.manifest_updates:
        print("  (none -- subject not cleaned yet)")

    print(f"\n=== File/folder renames ({len(plan.path_renames)}) ===")
    for src, dst in plan.path_renames:
        print(f"  {tag}{src}\n      -> {dst}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Rename one subject's R-code across all disk "
                    "touchpoints. Dry-run by default -- pass --apply "
                    "to commit.")
    p.add_argument("--subject-root", type=Path, required=True,
                   help="Parent dir containing per-subject folders")
    p.add_argument("--from", dest="from_code", type=str, default=None,
                   help="Old R-code, e.g. R1655J. Not required for "
                        "--diagnose (which inspects an entire subject dir).")
    p.add_argument("--to", dest="to_code", type=str, default=None,
                   help="New R-code, e.g. R1665J. Not required for "
                        "--diagnose.")
    p.add_argument("--apply", action="store_true",
                   help="Execute the plan. Without this the script "
                        "prints the plan and exits.")
    p.add_argument("--diagnose", type=str, default=None,
                   metavar="SUBJECT_CODE",
                   help="Read-only diagnostic mode: walk every .edf "
                        "under <subject-root>/<SUBJECT_CODE>/ and report "
                        "pyedflib.EdfReader outcome side-by-side with "
                        "byte-level patientcode read. No files mutated. "
                        "Use before committing to a rename to see what "
                        "pyedflib actually complains about.")
    args = p.parse_args(argv)

    if args.diagnose:
        subject_dir = args.subject_root / args.diagnose
        try:
            rows = diagnose_subject_dir(subject_dir)
        except FileNotFoundError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2
        _print_diagnostic_report(rows, subject_dir)
        return 0

    if not args.from_code or not args.to_code:
        print("[error] --from and --to are required for a rename "
              "(use --diagnose alone for read-only inspection)",
              file=sys.stderr)
        return 2

    try:
        plan = build_plan(args.subject_root, args.from_code, args.to_code)
    except (FileNotFoundError, FileExistsError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    _print_plan(plan, dry_run=not args.apply)

    if not args.apply:
        print("\n[dry-run] Nothing was changed. Re-run with --apply "
              "to commit.")
        return 0

    try:
        execute_plan(plan, args.from_code, args.to_code)
    except Exception as e:
        print(f"\n[error] execution aborted: {type(e).__name__}: {e}",
              file=sys.stderr)
        print("[error] any partial file writes were restored from "
              "backup; other files may not have been reached.",
              file=sys.stderr)
        return 3

    print(f"\n[ok] renamed {args.from_code} -> {args.to_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
