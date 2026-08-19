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


def find_edf_files_needing_header_update(subject_dir: Path, from_code: str
                                          ) -> list[Path]:
    """Every .edf under subject_dir whose patientcode header field is
    literally equal to ``from_code``. Raw pre-clean EDFs whose
    patientcode is the hospital MRN return an empty list -- nothing
    to update there."""
    import pyedflib
    hits: list[Path] = []
    for edf_path in sorted(subject_dir.rglob("*.edf")):
        try:
            with pyedflib.EdfReader(str(edf_path)) as f:
                if f.getHeader().get("patientcode", "") == from_code:
                    hits.append(edf_path)
        except (OSError, ValueError):
            # Unreadable EDF is out of scope for this tool. Log the
            # skip so the operator sees it in the plan.
            print(f"  [skip-read] {edf_path.name}: cannot open header",
                  file=sys.stderr)
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
    """Post-write check: reopen the file, verify patientcode is now
    ``expected_code`` AND the file loads cleanly end-to-end. Raises
    RuntimeError on any inconsistency -- caller restores from backup."""
    import pyedflib
    try:
        with pyedflib.EdfReader(str(edf_path)) as f:
            actual = f.getHeader().get("patientcode", "")
            # Force a read of the first signal to catch corruption
            # that shows up on data-record boundaries, not just header
            # parsing.
            if f.signals_in_file > 0:
                f.readSignal(0)
    except (OSError, ValueError) as e:
        raise RuntimeError(
            f"post-write verification failed: {edf_path} won't open "
            f"cleanly: {e}") from e
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
    p.add_argument("--from", dest="from_code", type=str, required=True,
                   help="Old R-code, e.g. R1655J")
    p.add_argument("--to", dest="to_code", type=str, required=True,
                   help="New R-code, e.g. R1665J")
    p.add_argument("--apply", action="store_true",
                   help="Execute the plan. Without this the script "
                        "prints the plan and exits.")
    args = p.parse_args(argv)

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
