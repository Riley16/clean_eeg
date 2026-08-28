"""``transfer-subject-eeg`` — upload a de-identified subject dir.

  transfer-subject-eeg OUTPUT_DIR [OPTIONS]

Options:
  --dry-run          Preflight and print the composed rsync/scp
                     commands without invoking anything.
  --user USER        SSH username (defaults to $USER).
  --yes, -y          Skip the interactive confirmation prompt.

Exit codes:
  0 — transfer completed successfully (or dry-run preflight passed)
  1 — preflight failed
  2 — transfer subprocess failed
  130 — user answered 'no' at the confirmation prompt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from clean_eeg.transfer import (
    preflight_deidentified_output,
    transfer_subject,
    build_transfer_plan,
)


def _confirm(prompt: str) -> bool:
    try:
        resp = input(prompt).strip().lower()
    except EOFError:
        return False
    return resp in ("y", "yes")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transfer-subject-eeg",
        description=(
            "Upload a de-identified subject directory to the CML rhino "
            "server. Refuses to run against a directory that hasn't "
            "been fully de-identified (missing manifest, quarantine "
            "leftovers, unredacted headers, etc.)."
        ),
    )
    p.add_argument("output_dir", type=Path,
                   help="Output directory produced by clean-subject-eeg "
                        "(contains deidentify.json + the *.edf files).")
    p.add_argument("--dry-run", action="store_true",
                   help="Preflight + print composed commands without uploading.")
    p.add_argument("--user", type=str, default=None,
                   help="SSH username (defaults to $USER).")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the interactive confirmation prompt.")
    p.add_argument("--background", "-b", action="store_true",
                   help="Launch the transfer in a detached background process "
                        "(via nohup) that survives SSH disconnects. Writes "
                        "transfer.sh + transfer.log alongside the output dir; "
                        "tail -f the log to monitor progress.")
    p.add_argument("--ssh-key", type=Path, default=None,
                   help="SSH private key path for auto-loading into "
                        "ssh-agent (default: ~/.ssh/id_ed25519). Only "
                        "used when ssh-agent isn't already running with "
                        "keys; the passphrase is entered ONCE per "
                        "invocation.")
    p.add_argument("--no-auto-ssh-agent", action="store_true",
                   help="Disable the auto-spawn-ssh-agent + auto-add-key "
                        "behaviour. Use when you're managing the agent "
                        "externally (keychain integration, custom "
                        "setup) or already have SSH_AUTH_SOCK exported "
                        "in your shell. Prints the manual-setup hint "
                        "instead if the agent is empty.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Preflight up-front so the operator sees the exact reasons the
    # tool would refuse to run BEFORE we ask them to confirm.
    result = preflight_deidentified_output(args.output_dir)
    print(result.summary())
    if not result.passed:
        return 1

    # Compose and print the plan so the operator can eyeball what will
    # execute. Uses the same rsync/scp resolution the transfer_subject
    # call will use downstream.
    assert result.manifest is not None
    # Threading the manifest's failed_files roster into the plan so the
    # rsync --exclude=<name> / scp glob filter drops them from the
    # upload argv. preflight already surfaces the skip as a warning;
    # this line is what actually enforces it in the composed command.
    from clean_eeg.transfer import _failed_names_from_manifest
    excluded_names = _failed_names_from_manifest(result.manifest)
    plan = build_transfer_plan(
        args.output_dir,
        subject_code=result.manifest["subject_code"],
        site_incoming_folder=result.manifest["site_incoming_folder"],
        ssh_user=args.user or _default_user(),
        use_rsync=shutil.which("rsync") is not None,
        excluded_names=excluded_names,
    )
    if excluded_names:
        print(f"\n[!] {len(excluded_names)} file(s) excluded from transfer "
              "(from manifest.failed_files):")
        for n in sorted(excluded_names):
            print(f"    - {n}")
    print()
    print(f"Transport: {plan.transport}")
    print(f"Remote:    {plan.remote_dir}")
    print("Will run:")
    for step, argv_ in (("1. mkdir", plan.mkdir_argv),
                        ("2. upload", plan.upload_argv),
                        ("3. perms", plan.perms_argv)):
        print(f"  {step}: {' '.join(argv_)}")

    if args.dry_run:
        return 0

    if not args.yes:
        if not _confirm("\nProceed with transfer? [y/N]: "):
            print("Aborted at confirmation prompt.")
            return 130

    try:
        result = transfer_subject(
            args.output_dir,
            ssh_user=args.user,
            dry_run=False,
            background=args.background,
            ssh_key=args.ssh_key,
            auto_ssh_agent=not args.no_auto_ssh_agent,
        )
    except RuntimeError as e:
        print(f"Transfer failed: {e}", file=sys.stderr)
        return 2
    if args.background:
        print(f"\nTransfer launched in background (pid {result.background_pid}).")
        print(f"  script: {result.background_script}")
        print(f"  log:    {result.background_log}")
        print(f"  monitor: tail -f {result.background_log}")
        print("The transfer will continue if this shell exits or the SSH "
              "connection drops.")
    else:
        print("\nTransfer complete.")
    return 0


def _default_user() -> str:
    import os
    return os.environ.get("USER", "USER")


if __name__ == "__main__":
    sys.exit(main())
