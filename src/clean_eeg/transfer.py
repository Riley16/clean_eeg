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
import sys
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


# Transfer endpoint is deliberately UNCONFIGURED at the code level -- no
# institutional hostnames or paths hardcoded. Callers must supply
# ssh_host + (remote_base or remote_dir_override) explicitly. Keeps the
# public repo free of any specific site's storage layout.
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


# patient_id and recording_id are space-separated per EDF+ spec:
#   patient_id   = "<mrn> <sex> <birthdate> <name...>"
#   recording_id = "Startdate <DD-MMM-YYYY> <admin> <technician> <equipment>"
# The de-identified pipeline writes canonical tokens for each field, so
# preflight verifies against those tokens directly on the raw header
# bytes -- no need to open the file as an EDF.
_DEIDENTIFIED_BIRTHDATE_TOKEN = "01-JAN-1900"
_DEIDENTIFIED_YEAR_TOKEN = "1985"
# Name may consist of multiple space-separated tokens; each must be X.
_NAME_TOKEN_RE = re.compile(r"^[Xx]+$")


def _check_edf_headers(edf_paths: Iterable[Path], subject_code: str,
                       failures: list[str]) -> None:
    """Verify the 4 de-identification invariants (patientname redacted,
    patientcode == subject_code, birthdate anonymised, startdate.year
    anchored) using ONLY the first 256 bytes of each EDF. Prior version
    opened every file via pyedflib.EdfReader, which parsed the full
    signal-header structure and cost network round-trips per file on
    NFS-mounted storage. Raw byte reads finish preflight in seconds
    instead of minutes on 100+ file Jefferson subjects."""
    from clean_eeg.print_edf_header import read_main_header
    for p in edf_paths:
        try:
            hdr = read_main_header(str(p))
        except OSError as e:
            failures.append(f"{p.name}: cannot read main header — {e}")
            continue
        if hdr.get("_truncated_main_header"):
            failures.append(
                f"{p.name}: file shorter than 256-byte EDF main header")
            continue

        patient_id = str(hdr.get("patient_id", "")).strip()
        parts = patient_id.split()
        # EDF+ layout: mrn sex birthdate name[+]. Anything shorter means
        # the pipeline never wrote a proper de-identified header.
        if len(parts) < 4:
            failures.append(
                f"{p.name}: patient_id {patient_id!r} does not have "
                f"the 4-field de-identified layout "
                f"'<mrn> <sex> <birthdate> <name>'")
            continue
        mrn, _sex, birthdate, *name_parts = parts

        if mrn != subject_code:
            failures.append(
                f"{p.name}: patient_id MRN {mrn!r} != "
                f"manifest subject_code {subject_code!r}")
        if not all(_NAME_TOKEN_RE.match(tok) for tok in name_parts):
            failures.append(
                f"{p.name}: patient_id name field {' '.join(name_parts)!r} "
                f"is not fully redacted (expected all-X tokens)")
        if birthdate.upper() != _DEIDENTIFIED_BIRTHDATE_TOKEN:
            failures.append(
                f"{p.name}: patient_id birthdate {birthdate!r} != "
                f"{_DEIDENTIFIED_BIRTHDATE_TOKEN!r}")

        # recording_id: 'Startdate <DD-MMM-YYYY> ...' -- the year is the
        # last 4 chars of the second whitespace-token when present.
        recording_id = str(hdr.get("recording_id", "")).strip()
        rec_parts = recording_id.split()
        year_token = rec_parts[1][-4:] if len(rec_parts) >= 2 else ""
        if year_token != _DEIDENTIFIED_YEAR_TOKEN:
            failures.append(
                f"{p.name}: recording_id startdate year "
                f"{year_token!r} != {_DEIDENTIFIED_YEAR_TOKEN!r} "
                f"(BASE_START_DATE anchor; recording_id={recording_id!r})")


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

    # 4. Manual annotation review must be complete. Cleaned-but-not-
    # reviewed subjects are held back so the operator can't accidentally
    # upload a subject whose annotations they haven't yet audited.
    #
    # ORDERING: this check runs BEFORE the pyedflib-based header checks
    # and the hash spot-check because it's cheap (a small JSONL read)
    # and eliminates ~all unreviewed subjects in a batch. On network
    # storage (Oceanus / NFS), the per-file pyedflib open dominates
    # preflight cost -- fast-failing here takes bulk preflight from
    # seconds-per-unreviewed-subject to milliseconds. When it fails we
    # return immediately: an unreviewed subject can't transfer for any
    # reason, and later checks would just add noise to the log.
    #
    # In stub-pair mode (in-place cleaning), carriers are the sidecars;
    # in inline mode, they're the recordings themselves.
    from clean_eeg.audit.annotations import check_annotation_review_state
    from clean_eeg.print_edf_header import ANNOTATION_STUB_SUFFIX
    stubs = [p for p in edfs if p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    review_carriers = (stubs if stubs
                       else [p for p in edfs
                             if not p.name.endswith(ANNOTATION_STUB_SUFFIX)])
    if review_carriers:
        review = check_annotation_review_state(output_path, review_carriers)
        if review.get("state") != "complete":
            n_r = review.get("n_reviewed", 0)
            n_c = review.get("n_annotation_carriers",
                              len(review_carriers))
            return PreflightResult(
                passed=False,
                failures=[
                    f"annotation review not complete for {subject_code}: "
                    f"{n_r}/{n_c} file(s) marked reviewed. Run "
                    f"annotation-review-eeg on this subject before transfer, "
                    f"or pass a session that ends with the "
                    f"'Mark all as reviewed?' prompt answered Y."
                ],
                warnings=warnings,
                manifest=manifest,
            )

    # 5. Naming pattern for every EDF in the output dir.
    for p in edfs:
        if not _DEID_FILENAME_RE.match(p.name):
            failures.append(
                f"{p.name}: filename does not match the de-identified "
                "pattern *_R1XXXY_MM.DD__HH.MM.SS(.edf|_annotations.edf) "
                "— did this file skip the rename step?"
            )

    # 6. Per-file header expectations. Only checks the main recordings
    # (sidecars carry a stub header with an empty patient_id, since they
    # only exist to hold annotations and share the parent recording's
    # provenance -- validated indirectly via the paired main EDF).
    recordings = [p for p in edfs
                  if not p.name.endswith(ANNOTATION_STUB_SUFFIX)]
    _check_edf_headers(recordings, subject_code, failures)

    # 7. Spot-check hash on one file.
    _spot_check_hash(edfs, manifest, failures)

    # 8. Defensive: the raw pre-Presidio annotation dump (created during
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
                         site_incoming_folder: str, ssh_user: str | None,
                         ssh_host: str,
                         use_rsync: bool,
                         remote_dir_override: str | None = None,
                         remote_base: str | None = None,
                         skip_perms: bool = False,
                         excluded_names: set[str] | None = None,
                         ) -> TransferPlan:
    # ssh_host is required (no default in the code -- keeps the public
    # repo free of institutional hostnames). Callers can supply
    # remote_base (composes <base>/<site>/<subject>/all_clinical_eeg via
    # the site-map layout) OR remote_dir_override (full per-subject path,
    # supports the {subject_code} placeholder). Neither is enforced at
    # this layer -- the CLIs enforce that a real target is supplied;
    # callers who skip it get a placeholder path that would fail the
    # rsync step loudly (safer than silently defaulting to some
    # institution's storage layout).
    if not ssh_host:
        raise ValueError("ssh_host is required (no code-level default)")
    effective_host = ssh_host
    if remote_dir_override is not None:
        # Test/scratch mode: caller supplies the full remote path
        # directly and (by default) opts out of the site-group perms
        # fixup, which only makes sense against a real incoming dir.
        remote_dir = remote_dir_override
        subject_remote_dir = remote_dir
        site_parent_dir = str(Path(remote_dir).parent)
    else:
        # No remote_dir_override -- fall back to site-map layout with
        # the (required) remote_base. When remote_base is also None
        # (only tests reach here; production CLIs enforce a real value)
        # we use a placeholder that would fail loudly if actually rsynced.
        effective_base = remote_base or "/UNCONFIGURED-REMOTE-BASE"
        site_parent_dir = f"{effective_base}/{site_incoming_folder}"
        subject_remote_dir = f"{site_parent_dir}/{subject_code}"
        remote_dir = f"{subject_remote_dir}/{SUBJECT_SUBFOLDER}"

    # ssh_user prepended only when the operator explicitly asked for
    # one. When None, ssh sees just `host` and picks up the User
    # directive from ~/.ssh/config -- required for endpoints whose
    # remote user differs from the local $USER (Windows tunnel, shared
    # accounts, etc.). Prepending would override the config's User.
    ssh_target = (f"{ssh_user}@{effective_host}"
                  if ssh_user else effective_host)
    rsync_target_prefix = f"{ssh_target}:"

    # umask 007 → newly-created intermediate dirs are group-rwx. Pre-existing
    # dirs are untouched.
    mkdir_argv = [
        "ssh", ssh_target,
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
            f"{rsync_target_prefix}{remote_dir}/",
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
            f"{rsync_target_prefix}{remote_dir}/",
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
            "ssh", ssh_target,
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
                        site_incoming_folder: str, ssh_user: str | None,
                        ssh_host: str,
                        use_rsync: bool | None = None,
                        remote_dir_override: str | None = None,
                        remote_base: str | None = None,
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
        ssh_host=ssh_host,
        remote_base=remote_base,
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


DEFAULT_SSH_KEY_PATH = Path.home() / ".ssh" / "id_ed25519"


def _agent_has_keys() -> bool:
    """True iff `ssh-add -l` reports at least one loaded key.
    Exit 0 = keys loaded; anything else (1 = no keys or no agent,
    2 = can't connect) treated as 'no'."""
    try:
        proc = subprocess.run(["ssh-add", "-l"], capture_output=True,
                                text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0


def _key_fingerprint(key_path: Path) -> str | None:
    """Fingerprint of the (private OR public) key at ``key_path`` via
    ``ssh-keygen -lf``. ssh-keygen accepts either half of the pair and
    returns the same fingerprint, so we can call it against the private
    key path the operator supplied. Returns None on any failure."""
    try:
        proc = subprocess.run(["ssh-keygen", "-lf", str(key_path)],
                                capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    # Format: "<bits> <fingerprint> <comment> (<type>)"
    parts = proc.stdout.strip().split()
    return parts[1] if len(parts) >= 2 else None


def _agent_has_key(key_path: Path) -> bool:
    """True iff the agent has THIS SPECIFIC key loaded (by fingerprint).
    Distinct from _agent_has_keys() which only checks "any key loaded" --
    a shell agent holding a github key but NOT the transfer key would
    fool the naive check and let SSH fall back to a passphrase prompt
    that never fires under BatchMode."""
    fp = _key_fingerprint(key_path)
    if fp is None:
        return False
    try:
        proc = subprocess.run(["ssh-add", "-l"], capture_output=True,
                                text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0 and fp in proc.stdout


def _spawn_ssh_agent() -> bool:
    """Spawn a fresh ssh-agent, parse its stdout env exports, set the
    corresponding env vars in this process's os.environ so subprocesses
    (rsync, ssh, ssh-add) inherit the agent socket. Registers atexit
    cleanup so the spawned agent doesn't leak between invocations.
    Returns True on success, False on any failure."""
    try:
        proc = subprocess.run(["ssh-agent", "-s"], capture_output=True,
                                text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    if proc.returncode != 0:
        return False
    # Output shape (POSIX / -s flag):
    #   SSH_AUTH_SOCK=/tmp/ssh-XXXX/agent.NNN; export SSH_AUTH_SOCK;
    #   SSH_AGENT_PID=NNN; export SSH_AGENT_PID;
    #   echo Agent pid NNN;
    parsed: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        for part in line.split(";"):
            part = part.strip()
            if "=" not in part or part.lower().startswith("export"):
                continue
            k, _, v = part.partition("=")
            parsed[k.strip()] = v.strip()
    sock = parsed.get("SSH_AUTH_SOCK")
    pid = parsed.get("SSH_AGENT_PID")
    if not sock or not pid:
        return False
    os.environ["SSH_AUTH_SOCK"] = sock
    os.environ["SSH_AGENT_PID"] = pid
    # Kill the spawned agent when Python exits so successive
    # transfer runs don't pile up orphaned agent processes.
    import atexit as _atexit

    def _kill_agent():
        try:
            subprocess.run(
                ["ssh-agent", "-k"],
                env={**os.environ, "SSH_AGENT_PID": pid},
                capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass
    _atexit.register(_kill_agent)
    return True


def _print_ssh_agent_manual_hint(key_path: Path,
                                  ssh_add_returncode: int | None = None) -> None:
    """Print the manual `eval $(ssh-agent); ssh-add <key>` recipe. Used
    when auto-setup fails or is disabled. Non-fatal -- the transfer
    still WORKS, it'll just prompt for the passphrase repeatedly."""
    rc_note = (f" (`ssh-add -l` exit {ssh_add_returncode})"
                if ssh_add_returncode is not None else "")
    print(
        f"\n[transfer] hint: ssh-agent has no keys loaded{rc_note}. "
        "Bulk transfers may prompt for your SSH passphrase repeatedly. "
        "Set up your agent manually with:\n"
        "    eval $(ssh-agent)\n"
        f"    ssh-add {key_path}      # enter passphrase once\n"
        "Then re-run this transfer. Continuing anyway.",
        flush=True)


def ensure_ssh_agent(key_path: Path | None = None,
                      auto: bool = True) -> None:
    """Ensure ssh-agent is running with a key loaded. Idempotent -- safe
    to call at the top of every transfer invocation (subsequent calls
    see the agent already up and no-op).

    Flow:
      1. If `ssh-add -l` already reports keys -> done, silent.
      2. If auto=False -> print manual-setup hint, return.
      3. If no SSH_AUTH_SOCK -> spawn a fresh ssh-agent + set env vars
         (atexit cleanup registered so we don't leak agent processes).
      4. If key file exists AND stdin is a TTY -> run `ssh-add <key>`.
         This prompts for the passphrase ONCE per invocation.
      5. Any step fails -> print manual-setup hint, continue.

    Non-fatal by design: transfers still work without the agent (just
    with per-connection prompts). Some setups (host-based auth,
    ProxyJump chains) don't need the agent at all -- we silently
    proceed in that case if the operator has authenticated elsewhere.
    """
    key_path = key_path or DEFAULT_SSH_KEY_PATH

    # Step 1: is THIS SPECIFIC key already in the agent? A prior
    # version only checked "any key loaded" which let a shell agent
    # holding e.g. a github key mask that the transfer key wasn't
    # loaded -- rsync would then fall back to a passphrase prompt that
    # BatchMode refused to answer, and every subject failed with
    # "Permission denied (publickey)".
    if key_path.exists() and _agent_has_key(key_path):
        return
    # Fallback: if we can't fingerprint the key file (e.g. permissions,
    # ssh-keygen not installed), fall back to the coarse "any key
    # loaded" check so we don't gratuitously re-prompt.
    if not key_path.exists() and _agent_has_keys():
        return

    if not auto:
        _print_ssh_agent_manual_hint(key_path)
        return

    # Step 3: no agent -> spawn one.
    if not os.environ.get("SSH_AUTH_SOCK"):
        if not _spawn_ssh_agent():
            _print_ssh_agent_manual_hint(key_path)
            return
        print(f"[transfer] started ssh-agent for this session (pid "
              f"{os.environ.get('SSH_AGENT_PID')}); agent dies with this "
              f"process.", flush=True)

    # Step 4: ssh-add the key.
    if not key_path.exists():
        print(f"[transfer] SSH key not found at {key_path} -- skipping "
              f"auto-add. Pass --ssh-key <path> if the key lives "
              f"elsewhere.", flush=True)
        _print_ssh_agent_manual_hint(key_path)
        return
    if not sys.stdin.isatty():
        # ssh-add prompts for the passphrase on a TTY. Under nohup /
        # cron / SSH-without-PTY there's no way to enter it.
        print(f"[transfer] no TTY -- can't prompt for SSH passphrase "
              f"non-interactively. Load the key before invoking:\n"
              f"    eval $(ssh-agent)\n"
              f"    ssh-add {key_path}\n"
              f"then re-run under nohup / batch.", flush=True)
        return
    print(f"[transfer] loading SSH key {key_path} into agent "
          f"(passphrase prompt below; entered once per invocation)...",
          flush=True)
    try:
        proc = subprocess.run(["ssh-add", str(key_path)], timeout=120)
        if proc.returncode != 0:
            _print_ssh_agent_manual_hint(
                key_path, ssh_add_returncode=proc.returncode)
            return
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        print(f"[transfer] ssh-add failed ({type(e).__name__}: {e}). "
              f"Continuing without the agent.", flush=True)
        _print_ssh_agent_manual_hint(key_path)
        return

    # Post-add verification: confirm the fingerprint we intended to load
    # actually shows up in `ssh-add -l`. Catches subtle failures like a
    # mistyped passphrase that ssh-add still exits 0 for on some
    # OpenSSH builds, and gives the operator a confirmation line before
    # the batch dispatches parallel rsyncs.
    if _agent_has_key(key_path):
        print(f"[transfer] key loaded into agent (fingerprint verified).",
              flush=True)
    else:
        print(f"[transfer] WARNING: ssh-add exited 0 but the key's "
              f"fingerprint is not in `ssh-add -l`. Downstream rsyncs "
              f"will likely fail with Permission denied. Investigate:\n"
              f"    ssh-add -l\n"
              f"    ssh-add {key_path}",
              file=sys.stderr, flush=True)


def transfer_subject(output_path: str | Path, *,
                     ssh_host: str,
                     ssh_user: str | None = None,
                     dry_run: bool = False,
                     use_rsync: bool | None = None,
                     site_map: dict[str, str] | None = None,
                     remote_dir_override: str | None = None,
                     remote_base: str | None = None,
                     skip_perms: bool = False,
                     background: bool = False,
                     ssh_key: Path | None = None,
                     auto_ssh_agent: bool = True,
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

    # ssh_user None -> defer to ssh_config's User directive (or SSH's
    # own default). Only set from $USER if the operator explicitly
    # asked for that fallback via passing the empty string as an
    # override, which no CLI does today. Prior behaviour of implicitly
    # using $USER broke endpoints whose remote user differed from the
    # local login.

    # ssh-agent: bulk transfers open many SSH connections in sequence.
    # Without an agent, each prompts for the key passphrase -- on a
    # 27-subject batch that's dozens of prompts. ensure_ssh_agent is
    # idempotent: called at the top of every transfer_subject; the
    # FIRST call spawns the agent + prompts once for the passphrase,
    # subsequent calls see the agent already up and no-op. Setup is
    # non-fatal (transfer still works with per-connection prompts if
    # the agent can't be set up), and auto=False lets callers with
    # their own agent management disable this entirely.
    ensure_ssh_agent(key_path=ssh_key, auto=auto_ssh_agent)

    plan = build_transfer_plan(
        output_path,
        subject_code=manifest["subject_code"],
        site_incoming_folder=manifest["site_incoming_folder"],
        ssh_user=ssh_user,
        ssh_host=ssh_host,
        use_rsync=use_rsync,
        remote_dir_override=remote_dir_override,
        remote_base=remote_base,
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
