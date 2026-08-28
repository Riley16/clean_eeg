"""Transfer a de-identified subject directory to the CML rhino server.

Called two ways:

  1. From the ``clean-subject-eeg`` pipeline's end-of-run prompt
     ("Ready to transfer? [y/N]").
  2. Directly via the ``transfer-subject-eeg`` console script — useful
     when a previous transfer was interrupted mid-way, or when
     de-identification and upload happen on different machines.

Both entry points share :func:`transfer_subject`; the standalone CLI
lives in :mod:`clean_eeg.transfer_cli`.

Preflight refuses to run against a directory that doesn't look like
a clean de-identified subject output. This is defense-in-depth on top
of the pipeline's in-run PHI redaction — even if an operator points
the transfer tool at the wrong directory (a raw NK export, a
half-completed run), preflight catches the obvious signs before any
bytes leave the machine.

The transfer itself uses ``rsync --partial --append-verify`` so an
interrupted upload can be safely resumed by simply re-running the
command. ``scp`` fallback covers systems without rsync (e.g., Windows
cmd.exe without WSL); it lacks resume-on-failure but is one-shot
runnable.
"""

from __future__ import annotations

import os
import random
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pyedflib

from clean_eeg.audit.hashes import sha256_fast_of_file
from clean_eeg.deidentify_manifest import (
    MANIFEST_FILENAME,
    ManifestSchemaError,
    read_manifest,
)


SSH_HOST = "rhino2.psych.upenn.edu"
REMOTE_BASE = "/data10/RAM/incoming"
SUBJECT_SUBFOLDER = "all_clinical_eeg"

# Matches the de-identified filename pattern produced by
# clean_subject_edf_files: <orig>_R1XXXY_MM.DD__HH.MM.SS.edf
# (optionally followed by _annotations before the extension for the
# inplace-mode annotation stubs).
_DEID_FILENAME_RE = re.compile(
    r"^.+_R1\d{3}[ACDEFHJMNPST]_\d{2}\.\d{2}__\d{2}\.\d{2}\.\d{2}"
    r"(_annotations)?\.edf$"
)

# Header-field expectations for a properly de-identified file. All four
# must hold; any mismatch fails preflight.
_PATIENTNAME_X_RE = re.compile(r"^[Xx\s]+$")
_DEIDENTIFIED_BIRTHDATE = "01 jan 1900"
_DEIDENTIFIED_YEAR = 1900 + 85  # BASE_START_DATE = datetime(1985, 1, 1)


@dataclass
class PreflightResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: dict | None = None

    def summary(self) -> str:
        lines = [
            f"Preflight: {'PASS' if self.passed else 'FAIL'}",
        ]
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        for f in self.failures:
            lines.append(f"  FAIL: {f}")
        return "\n".join(lines)


@dataclass
class TransferPlan:
    """Command lines that ``execute_plan`` will run. Exposed for tests
    (so dry-run mode returns an inspectable plan) and for the
    end-of-run prompt (so the operator can see exactly what will be
    invoked before confirming). When ``transfer_subject`` is called
    with ``background=True``, the returned plan carries the child
    process's pid + the paths to the shell script and log file so the
    CLI can surface them to the operator."""
    mkdir_argv: list[str]
    upload_argv: list[str]
    perms_argv: list[str]
    remote_dir: str
    transport: str  # "rsync" or "scp"
    background_pid: int | None = None
    background_script: Path | None = None
    background_log: Path | None = None


def _iter_edfs(output_path: Path,
               excluded_names: set[str] | None = None) -> list[Path]:
    """All *.edf files directly in ``output_path`` (non-recursive —
    quarantine/ subdir is skipped by construction).

    ``excluded_names`` is a set of basenames that the caller wants to
    skip (typically the ``failed_files`` roster from the manifest — see
    :func:`_failed_names_from_manifest`). Excluding by basename lets the
    caller both list the file and refuse to include it in the transfer
    plan without extra filesystem trickery.
    """
    excluded = excluded_names or set()
    return sorted(p for p in output_path.iterdir()
                  if p.is_file() and p.suffix.lower() == ".edf"
                  and p.name not in excluded)


def _failed_names_from_manifest(manifest: dict) -> set[str]:
    """Return the set of basenames the pipeline reported as failed
    (either at load or de-id). These are the files transfer refuses to
    upload -- they may still carry PHI or be structurally invalid.
    """
    entries = manifest.get("failed_files") or []
    return {entry["filename"] for entry in entries
            if isinstance(entry, dict) and entry.get("filename")}


def _check_edf_headers(edf_paths: Iterable[Path], subject_code: str,
                       failures: list[str]) -> None:
    for p in edf_paths:
        try:
            with pyedflib.EdfReader(str(p)) as f:
                header = f.getHeader()
                startdate = f.getStartdatetime()
        except OSError as e:
            failures.append(f"{p.name}: pyedflib cannot open — {e}")
            continue
        if not _PATIENTNAME_X_RE.match(str(header.get("patientname", ""))):
            failures.append(
                f"{p.name}: patientname is not fully redacted "
                f"(expected X-pattern, got {header.get('patientname')!r})"
            )
        if header.get("patientcode") != subject_code:
            failures.append(
                f"{p.name}: patientcode {header.get('patientcode')!r} "
                f"!= manifest subject_code {subject_code!r}"
            )
        if str(header.get("birthdate", "")).strip().lower() != _DEIDENTIFIED_BIRTHDATE:
            failures.append(
                f"{p.name}: birthdate {header.get('birthdate')!r} "
                f"!= {_DEIDENTIFIED_BIRTHDATE!r}"
            )
        if startdate.year != _DEIDENTIFIED_YEAR:
            failures.append(
                f"{p.name}: startdate.year {startdate.year} "
                f"!= {_DEIDENTIFIED_YEAR} (BASE_START_DATE anchor)"
            )


def _spot_check_hash(edf_paths: list[Path], manifest: dict,
                     failures: list[str]) -> None:
    """Recompute the fast-hash on one randomly-chosen file and compare
    to the manifest. Bounded to one file so preflight stays cheap; the
    audit runs the full comparison post-upload."""
    stored = manifest.get("file_hashes", {})
    candidates = [p for p in edf_paths if p.name in stored]
    if not candidates:
        return
    victim = random.choice(candidates)
    fresh, _mode, _det = sha256_fast_of_file(victim)
    if fresh != stored[victim.name]:
        failures.append(
            f"{victim.name}: hash on disk ({fresh[:12]}…) disagrees "
            f"with manifest ({stored[victim.name][:12]}…) — output "
            "directory has been modified since de-identification"
        )


def preflight_deidentified_output(output_path: str | Path,
                                  site_map: dict[str, str] | None = None,
                                  ) -> PreflightResult:
    """Verify ``output_path`` is a fully de-identified subject dir
    ready for upload.

    ``site_map`` defaults to :data:`clean_eeg.clean_subject_eeg.SITE_CODE_TO_INCOMING_FOLDER`;
    an injection point for tests.
    """
    if site_map is None:
        from clean_eeg.clean_subject_eeg import SITE_CODE_TO_INCOMING_FOLDER
        site_map = SITE_CODE_TO_INCOMING_FOLDER

    output_path = Path(output_path)
    failures: list[str] = []
    warnings: list[str] = []

    if not output_path.is_dir():
        return PreflightResult(passed=False,
                               failures=[f"{output_path} is not a directory"])

    # 1. Manifest presence + schema.
    try:
        manifest = read_manifest(output_path)
    except ManifestSchemaError as e:
        return PreflightResult(passed=False, failures=[str(e)])
    if manifest is None:
        return PreflightResult(
            passed=False,
            failures=[
                f"{output_path / MANIFEST_FILENAME} is missing — this "
                "directory has not been de-identified (or the manifest "
                "was deleted). Run clean-subject-eeg first."
            ],
        )

    subject_code = manifest.get("subject_code", "")
    site_code = manifest.get("site_code", "")

    # 2. Quarantine subdir absent OR empty.
    quarantine = output_path / "quarantine"
    if quarantine.is_dir() and any(quarantine.iterdir()):
        failures.append(
            f"{quarantine} is non-empty — one or more files failed "
            "de-identification and were quarantined. Investigate and "
            "clear the directory before transferring."
        )

    # 6. Site letter must be in the map (before we consult it later).
    if site_code not in site_map:
        failures.append(
            f"site_code {site_code!r} (from subject {subject_code!r}) "
            f"is not in SITE_CODE_TO_INCOMING_FOLDER — refusing to "
            f"upload to an unknown site. Known: {sorted(site_map)}"
        )

    # 3. Naming pattern for every EDF in the output dir. Files listed
    #    in manifest.failed_files are EXCLUDED entirely -- the pipeline
    #    already told us they can't be cleaned, so we treat them as
    #    not-present-for-transfer and skip every downstream check on
    #    them (they'd fail all of these anyway and drag the whole
    #    preflight to fail-status).
    excluded_names = _failed_names_from_manifest(manifest)
    if excluded_names:
        warnings.append(
            f"SKIPPING {len(excluded_names)} file(s) that failed cleaning "
            f"(from manifest.failed_files); they will NOT be transferred: "
            f"{sorted(excluded_names)}"
        )
    edfs = _iter_edfs(output_path, excluded_names=excluded_names)
    if not edfs:
        failures.append(f"{output_path} contains no .edf files to upload")
    for p in edfs:
        if not _DEID_FILENAME_RE.match(p.name):
            failures.append(
                f"{p.name}: filename does not match the de-identified "
                "pattern *_R1XXXY_MM.DD__HH.MM.SS(.edf|_annotations.edf) "
                "— did this file skip the rename step?"
            )

    # 4. Per-file header expectations (only for the transfer-eligible set).
    _check_edf_headers(edfs, subject_code, failures)

    # 5. Spot-check hash on one file.
    _spot_check_hash(edfs, manifest, failures)

    # 6. Defensive: the raw pre-Presidio annotation dump (created during
    # cleaning at <subject>/<subfolder>_original_annotations/, sibling
    # to output_path) contains PHI and MUST NOT be inside the transfer
    # source. Under normal operation this dir is a SIBLING of
    # output_path so it's outside the transfer scope by construction;
    # this check catches any accidental refactor that puts the raw dump
    # underneath the transfer source (rename, symlink, misplaced write,
    # etc.). Fires BEFORE the rsync --exclude belt-and-suspenders
    # kicks in, so the operator sees the failure loudly instead of
    # trusting a silent rsync filter.
    from clean_eeg.original_annotations import sibling_dir_inside
    offender = sibling_dir_inside(output_path)
    if offender is not None:
        failures.append(
            f"raw-annotations dump found INSIDE transfer source: "
            f"{offender}. This directory contains PHI and MUST NOT "
            f"be transferred. Move it out of the transfer source or "
            f"delete it before rerunning transfer-subject-eeg."
        )

    return PreflightResult(
        passed=not failures,
        failures=failures,
        warnings=warnings,
        manifest=manifest,
    )


def _build_transfer_plan(output_path: Path, *, subject_code: str,
                         site_incoming_folder: str, ssh_user: str,
                         use_rsync: bool,
                         remote_dir_override: str | None = None,
                         skip_perms: bool = False,
                         excluded_names: set[str] | None = None,
                         ) -> TransferPlan:
    if remote_dir_override is not None:
        # Test/scratch mode: caller supplies the full remote path
        # directly and (by default) opts out of the site-group perms
        # fixup, which only makes sense against a real incoming dir.
        remote_dir = remote_dir_override
        subject_remote_dir = remote_dir
        site_parent_dir = str(Path(remote_dir).parent)
    else:
        site_parent_dir = f"{REMOTE_BASE}/{site_incoming_folder}"
        subject_remote_dir = f"{site_parent_dir}/{subject_code}"
        remote_dir = f"{subject_remote_dir}/{SUBJECT_SUBFOLDER}"

    # umask 007 → newly-created intermediate dirs are group-rwx. Pre-existing
    # dirs are untouched.
    mkdir_argv = [
        "ssh", f"{ssh_user}@{SSH_HOST}",
        f'umask 007 && mkdir -p {remote_dir}',
    ]

    excluded_names = excluded_names or set()
    if use_rsync:
        # --partial: keep partial destination file on interrupt so a
        # re-run resumes cheaply via rsync's delta algorithm (which
        # already block-checksums the shared prefix on resume — no
        # need for --append-verify, which is only meaningful alongside
        # --append and is rejected by macOS's shipped rsync-2.6.9).
        # --exclude='quarantine/': never ship quarantined files even
        # if a partial run left some behind.
        # --exclude=<name> per failed file: the manifest-recorded
        # failures never make it upstream even if they're sitting in
        # output_path alongside the successfully-cleaned files.
        # Trailing slash on source path → copy directory contents
        # (incl. log.out and deidentify.json), not the directory itself.
        upload_argv = [
            "rsync", "-avzh", "--partial", "--progress",
            "--exclude=quarantine/",
            # Belt-and-suspenders: the raw pre-Presidio annotation dump
            # (clinical_eeg_original_annotations sibling of the transfer
            # source) contains PHI and MUST NOT ship. It's already OUTSIDE
            # the transfer source by design, so this --exclude is a defense
            # against a future refactor that changes the source path to
            # e.g. the subject root. The preflight assertion in
            # preflight_deidentified_output catches the same failure mode
            # at a different layer.
            "--exclude=*_original_annotations/",
            *(f"--exclude={n}" for n in sorted(excluded_names)),
            f"{output_path}/",
            f"{ssh_user}@{SSH_HOST}:{remote_dir}/",
        ]
        transport = "rsync"
    else:
        # scp fallback: no resume, no exclude semantics — but the *.edf
        # glob is non-recursive so quarantine/ is skipped anyway. Must
        # list log.out and deidentify.json explicitly since *.edf misses
        # both. Sidecar files that don't exist on disk are dropped from
        # the argv (unlike rsync, scp fails hard on a missing source
        # file, and log.out is legitimately absent when the caller ran
        # the pipeline via the library API rather than the CLI's
        # __main__ block that sets up the logger).
        sidecars = [output_path / "log.out",
                    output_path / MANIFEST_FILENAME]
        existing_sidecars = [str(p) for p in sidecars if p.exists()]
        # -p preserves mtime + mode, matching what rsync's -a does.
        # Consistent mtime lets the round-trip verification (and any
        # future "did this file change on the server?" audit) compare
        # against the pipeline-write time instead of the upload time.
        upload_argv = [
            "scp", "-p",
            *sorted(str(p) for p in output_path.glob("*.edf")
                    if p.name not in excluded_names),
            *existing_sidecars,
            f"{ssh_user}@{SSH_HOST}:{remote_dir}/",
        ]
        transport = "scp"

    # chgrp -R --reference=<site_parent_dir> ensures the data team's
    # group (as set on the site's incoming dir) owns everything.
    # chmod -R g+rwX,o-rwx: group can read/write/traverse; other is
    # blocked. `;` (not `&&`) so chmod still runs if chgrp can't touch
    # a few entries the operator doesn't own — those failures are
    # printed but non-fatal.
    #
    # Skipped when ``remote_dir_override`` is used (test / scratch
    # destinations have no site group to inherit from) unless the
    # caller explicitly re-enables it.
    if skip_perms or remote_dir_override is not None:
        perms_argv: list[str] = []
    else:
        perms_argv = [
            "ssh", f"{ssh_user}@{SSH_HOST}",
            f"chgrp -R --reference={site_parent_dir} {subject_remote_dir}; "
            f"chmod -R g+rwX,o-rwx {subject_remote_dir}",
        ]

    return TransferPlan(
        mkdir_argv=mkdir_argv,
        upload_argv=upload_argv,
        perms_argv=perms_argv,
        remote_dir=remote_dir,
        transport=transport,
    )


def build_transfer_plan(output_path: str | Path, *, subject_code: str,
                        site_incoming_folder: str, ssh_user: str,
                        use_rsync: bool | None = None,
                        remote_dir_override: str | None = None,
                        skip_perms: bool = False,
                        excluded_names: set[str] | None = None,
                        ) -> TransferPlan:
    """Public helper — resolves ``use_rsync`` from ``shutil.which`` if
    not supplied. Exposed so tests can inspect the composed commands
    without invoking them, and so the CLI can print the plan for the
    operator before executing.

    ``remote_dir_override``: if given, the composed remote destination
    is this path verbatim instead of being derived from the site map.
    Intended for integration tests that transfer to a scratch or
    ``edf_transfer_test/`` subdir — should not be used in production.

    ``excluded_names``: basenames to omit from the upload. Usually the
    ``manifest.failed_files`` set — files the pipeline reported as
    unable to be cleaned. Encoded as ``--exclude=<name>`` in rsync
    mode; filtered out of the glob in scp mode. When ``None``, no
    filtering (identical to the pre-``failed_files`` behavior).
    """
    if use_rsync is None:
        use_rsync = shutil.which("rsync") is not None
    return _build_transfer_plan(
        Path(output_path),
        subject_code=subject_code,
        site_incoming_folder=site_incoming_folder,
        ssh_user=ssh_user,
        use_rsync=use_rsync,
        remote_dir_override=remote_dir_override,
        skip_perms=skip_perms,
        excluded_names=excluded_names,
    )


def _run(argv: list[str]) -> None:
    """Run ``argv`` and raise if it exits non-zero. Streams stdout/stderr
    to the current terminal so rsync's progress bar and ssh's prompts
    surface to the operator in real time."""
    proc = subprocess.run(argv)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {proc.returncode}: {' '.join(argv)}"
        )


def execute_plan(plan: TransferPlan) -> None:
    """Run the three-step transfer: mkdir → upload → chgrp/chmod. The
    perms step is skipped when ``plan.perms_argv`` is empty (test /
    scratch destinations, see :func:`build_transfer_plan`)."""
    _run(plan.mkdir_argv)
    _run(plan.upload_argv)
    if plan.perms_argv:
        _run(plan.perms_argv)


def execute_plan_background(plan: TransferPlan, output_path: Path,
                            ) -> tuple[int, Path, Path]:
    """Run the full transfer in a detached background process. Survives
    the parent shell exiting (nohup + start_new_session) so an SSH
    connection drop mid-upload doesn't kill the transfer.

    Composes all three plan steps into a shell script written next to
    the output ``deidentify.json`` so the operator can inspect exactly
    what will run (and re-run manually if the background process
    fails). Stdout+stderr from the child stream to ``transfer.log``
    in the same directory — ``tail -f transfer.log`` shows live
    progress.

    Returns ``(pid, script_path, log_path)``. The parent exits
    immediately after the child is launched; the caller prints the
    pid + paths so the operator can monitor.
    """
    script_path = output_path / "transfer.sh"
    log_path = output_path / "transfer.log"

    # shlex.quote every arg so paths with spaces, quotes, etc. survive
    # the shell round-trip. Each step goes on its own line so the log
    # is readable; `set -e` aborts on the first failure so we don't
    # chmod a directory whose upload failed halfway.
    lines = ["#!/usr/bin/env bash", "set -e", ""]
    for label, argv in (("mkdir", plan.mkdir_argv),
                        ("upload", plan.upload_argv),
                        ("perms", plan.perms_argv)):
        if not argv:
            continue
        lines.append(f"echo '=== {label} ==='")
        lines.append(" ".join(shlex.quote(a) for a in argv))
        lines.append("")
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)

    # nohup + start_new_session detaches the child from the controlling
    # terminal so SIGHUP on ssh disconnect doesn't kill it. stdin
    # redirected from /dev/null so the child can't try to read from a
    # (soon-vanishing) terminal.
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        ["nohup", "bash", str(script_path)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid, script_path, log_path


def _check_ssh_agent_loaded() -> None:
    """Warn (non-fatal) if ssh-agent isn't running with any keys loaded.

    Bulk transfers open many SSH connections in sequence. Without
    ssh-agent, each one prompts for the key passphrase; on a 27-
    subject batch that's ~54+ prompts. Users have gotten burned by
    this; the check + hint saves them.

    `ssh-add -l` semantics (per man page):
      exit 0 -> keys are loaded
      exit 1 -> agent running but no keys, OR agent not running
                (message differs but exit code doesn't -- both
                warrant the same hint)
      exit 2 -> can't connect to agent

    Non-fatal: some setups use host-based auth, ProxyJump chains, or
    per-connection ControlPersist and don't need the agent. We print
    a hint and continue. The transfer will still function if the
    operator's setup handles auth differently.
    """
    import subprocess as _sp
    try:
        proc = _sp.run(["ssh-add", "-l"], capture_output=True,
                        text=True, timeout=5)
    except (FileNotFoundError, _sp.SubprocessError, OSError):
        # No ssh-add binary or agent misconfigured -- can't hint
        # confidently, don't spam.
        return
    if proc.returncode == 0:
        return    # keys loaded -- silent OK
    # exit 1 or 2 -- agent not usable
    print(
        "\n[transfer] hint: ssh-agent has no keys loaded (`ssh-add -l` "
        f"exit {proc.returncode}). Bulk transfers may prompt for your "
        "SSH passphrase repeatedly. To load your key once for this "
        "shell session:\n"
        "    eval $(ssh-agent)\n"
        "    ssh-add ~/.ssh/id_ed25519      # enter passphrase once\n"
        "Then re-run this transfer. Continuing anyway.",
        flush=True)


def transfer_subject(output_path: str | Path, *,
                     ssh_user: str | None = None,
                     dry_run: bool = False,
                     use_rsync: bool | None = None,
                     site_map: dict[str, str] | None = None,
                     remote_dir_override: str | None = None,
                     skip_perms: bool = False,
                     background: bool = False,
                     ) -> TransferPlan:
    """Preflight, then execute (unless ``dry_run``). Returns the
    composed :class:`TransferPlan` either way.

    ``ssh_user`` defaults to ``$USER``; the CLI's ``--user`` override
    takes precedence. Raises ``RuntimeError`` on preflight failure or
    subprocess error — the CLI catches these and prints a friendly
    error.

    ``remote_dir_override``: send to this remote path verbatim instead
    of the site-derived destination. Test-only; production code should
    not pass this.

    ``background``: launch the transfer in a detached background
    process that survives SSH disconnects (see
    :func:`execute_plan_background`). Returns as soon as the child is
    launched; the caller is responsible for surfacing pid + log path
    to the operator.
    """
    output_path = Path(output_path)
    result = preflight_deidentified_output(output_path, site_map=site_map)
    if not result.passed:
        raise RuntimeError(
            "Preflight failed — refusing to upload:\n" + result.summary()
        )
    assert result.manifest is not None  # preflight guarantees this
    manifest = result.manifest

    if ssh_user is None:
        ssh_user = os.environ.get("USER", "")
    if not ssh_user:
        raise RuntimeError(
            "ssh_user is empty — set $USER or pass --user on the "
            "command line."
        )

    # ssh-agent hint: bulk transfers open many SSH connections in
    # sequence. Without ssh-agent, each one prompts for the key
    # passphrase -- the operator would enter it hundreds of times.
    # `ssh-add -l` lists loaded keys; exit-1 means "no keys" or
    # "agent not running", exit-2 means "can't connect to agent".
    # Non-fatal warning: the transfer still WORKS without agent, just
    # painfully, and the check may spuriously fail (e.g. key-less
    # setups using host-based auth or ProxyJump chains). Print a hint
    # and continue.
    _check_ssh_agent_loaded()

    plan = build_transfer_plan(
        output_path,
        subject_code=manifest["subject_code"],
        site_incoming_folder=manifest["site_incoming_folder"],
        ssh_user=ssh_user,
        use_rsync=use_rsync,
        remote_dir_override=remote_dir_override,
        skip_perms=skip_perms,
    )
    if dry_run:
        return plan
    if background:
        pid, script_path, log_path = execute_plan_background(plan, output_path)
        # Stash on the plan so the CLI can surface these to the
        # operator without needing to duplicate the return signature.
        plan.background_pid = pid
        plan.background_script = script_path
        plan.background_log = log_path
    else:
        execute_plan(plan)
    return plan
