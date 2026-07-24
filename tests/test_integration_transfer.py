"""End-to-end integration test: clean a small subject dir, transfer it
to rhino2, pull it back, verify round-trip integrity.

Opt-in only — the ``integration`` marker is deselected by default in
``pyproject.toml``. Run with::

    pytest -m integration -s

The ``-s`` flag is required because the test prompts the operator once
for a rhino2 login ID and an optional hospital site code. All
subsequent pipeline prompts (recording-gap, ready-to-transfer, etc.)
are auto-answered via monkeypatch, so the test runs to completion
without further interaction.

Behavior:
  - No site code  → transfer into
    ``/scratch/<LOGIN>/edf_transfer_test/<FAKE_CODE>/``
  - Site code given → transfer into
    ``/data10/RAM/incoming/<SITE_FOLDER>/edf_transfer_test/<FAKE_CODE>/``

Remote destination is created if missing; on completion the remote
files are LEFT IN PLACE (the operator can delete them by hand, or the
next run of the test will overwrite by re-running rsync). Local
scratch is cleaned up.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from clean_eeg.anonymize import PersonalName
from clean_eeg.audit.hashes import sha256_fast_of_file
from clean_eeg.clean_subject_eeg import (
    LOG_FILENAME,
    SITE_CODE_TO_INCOMING_FOLDER,
    clean_subject_edf_files,
    redact_log_file,
)
from clean_eeg.deidentify_manifest import MANIFEST_FILENAME, read_manifest
from clean_eeg.log import close_logger, setup_logger
from clean_eeg.provenance import log_environment_provenance
from clean_eeg.transfer import SSH_HOST, transfer_subject


# Same PATIENT_NAME shape as the unit tests use, so operators inspecting
# the round-tripped files see familiar de-identified values.
PATIENT_NAME = PersonalName(first_name="Test", middle_names=[], last_name="Person")


def _write_minimal_deidentifiable_edf(path: Path,
                                       n_channels: int = 3,
                                       sample_rate: int = 100,
                                       duration_s: int = 2) -> None:
    """Tiny EDF+C with a couple of annotations. Small enough that the
    end-to-end run finishes in seconds even on slow links."""
    signal_headers = [
        {"label": f"CH{i}", "dimension": "uV",
         "sample_frequency": sample_rate,
         "physical_max": 3200.0, "physical_min": -3200.0,
         "digital_max": 32767, "digital_min": -32768,
         "prefilter": "", "transducer": ""}
        for i in range(n_channels)
    ]
    t = np.arange(0, duration_s, 1.0 / sample_rate, dtype=np.float32)
    signals = [(1000.0 * np.sin(2 * np.pi * (i + 1) * t)).astype(np.float64)
               for i in range(n_channels)]
    from datetime import datetime as _dt
    with pyedflib.EdfWriter(str(path), n_channels,
                             file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            "technician": "T", "recording_additional": "",
            "patientname": f"{PATIENT_NAME.first_name} {PATIENT_NAME.last_name}",
            "patient_additional": "",
            "patientcode": "PRE_CLEAN", "equipment": "test",
            "admincode": "", "sex": "Male",
            "startdate": _dt(2023, 1, 1, 10, 0, 0),
            "birthdate": "01 feb 1970", "gender": "Male",
        })
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)
        f.writeAnnotation(0.5, -1, "START")
        f.writeAnnotation(float(duration_s) - 0.5, -1, "END")


@pytest.fixture(scope="module")
def transfer_credentials() -> tuple[str, str | None]:
    """Module-scoped so parametrized transport variants share one
    prompt instead of asking the operator twice."""
    return _prompt_credentials()


def _prompt_credentials() -> tuple[str, str | None]:
    """One-shot prompt for the rhino2 login ID and optional hospital
    site code. Prints the site-letter → folder map so the operator
    doesn't have to remember it."""
    default_user = getpass.getuser()
    print()
    print("=" * 60)
    print("End-to-end transfer integration test")
    print("=" * 60)
    print(f"Enter your rhino2 login ID (default: {default_user!r}):")
    login = input("  Login: ").strip() or default_user
    if not login:
        pytest.skip("no login provided")

    print()
    print("Optional hospital site code — determines the remote parent dir.")
    print("Site letters (leave blank to use /scratch/<login>/ instead):")
    for letter, folder in sorted(SITE_CODE_TO_INCOMING_FOLDER.items()):
        print(f"  {letter}: /data10/RAM/incoming/{folder}/")
    site_raw = input("  Site letter (Enter to skip): ").strip().upper()
    site = site_raw or None
    if site is not None and site not in SITE_CODE_TO_INCOMING_FOLDER:
        pytest.skip(
            f"site code {site!r} not in SITE_CODE_TO_INCOMING_FOLDER "
            f"({sorted(SITE_CODE_TO_INCOMING_FOLDER)!r})"
        )
    return login, site


def _resolve_destination(login: str, site: str | None) -> tuple[str, str]:
    """Return ``(fake_subject_code, remote_dir)``. Fake subject code
    matches SUBJECT_CODE_PATTERN so the pipeline's internal checks
    accept it; the trailing letter picks a valid site (A when the
    operator didn't specify one — arbitrary because the scratch path
    doesn't route by site)."""
    if site is None:
        fake_code = "R1000A"
        remote_dir = f"/scratch/{login}/edf_transfer_test/{fake_code}"
    else:
        fake_code = f"R1000{site}"
        site_folder = SITE_CODE_TO_INCOMING_FOLDER[site]
        remote_dir = (f"/data10/RAM/incoming/{site_folder}/"
                      f"edf_transfer_test/{fake_code}")
    return fake_code, remote_dir


def _remote_mtime(login: str, remote_path: str) -> float:
    """Get the mtime of a remote file via ``ssh stat``. Fresh
    connection each call — this is diagnostic only, called on a
    handful of files."""
    proc = subprocess.run(
        ["ssh", f"{login}@{SSH_HOST}", f"stat -c %Y {remote_path}"],
        capture_output=True, text=True, check=True,
    )
    return float(proc.stdout.strip())


@pytest.mark.integration
@pytest.mark.parametrize("transport", ["rsync", "scp"])
def test_clean_transfer_and_roundtrip(tmp_path, monkeypatch, capsys,
                                       transfer_credentials, transport):
    """Full pipeline against a real cluster: generate → clean →
    transfer → pull back → compare. Runs once with rsync and once
    with scp (fallback path for systems without rsync). Leaves the
    remote copy in place for manual inspection / re-runs; cleans the
    local scratch."""
    if transport == "scp" and shutil.which("scp") is None:
        pytest.skip("scp not on PATH")
    if transport == "rsync" and shutil.which("rsync") is None:
        pytest.skip("rsync not on PATH")

    login, site = transfer_credentials
    fake_code, remote_dir = _resolve_destination(login, site)
    # Tag the remote path with the transport so parallel/sequential
    # rsync + scp runs don't collide on the same subject dir.
    remote_dir = f"{remote_dir}_{transport}"

    # Auto-answer every pipeline prompt (recording-gap, name-consistency,
    # signal-header, etc.) with 'y'. The transfer prompt is handled
    # separately via auto_transfer_response — we invoke transfer manually
    # below so we can pass remote_dir_override.
    responses = iter(["y"] * 20)
    monkeypatch.setattr("builtins.input", lambda _msg="": next(responses))

    # 1. Generate a small subject: two EDFs so recording-gap and
    #    signal-header consistency checks both exercise (they no-op
    #    since the files are structurally identical).
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(2):
        _write_minimal_deidentifiable_edf(input_dir / f"f{i}.edf")

    # 2. Set up the pipeline logger the same way __main__ does, so the
    #    run produces a real log.out sidecar (with provenance block and
    #    PHI-scrubbed transcript). We want the full artifact set on the
    #    cluster, not just EDFs + manifest.
    log_path = input_dir / LOG_FILENAME
    logger = setup_logger(str(log_path))
    try:
        for part in (PATIENT_NAME.first_name, PATIENT_NAME.last_name):
            logger.add_phi(part)
        log_environment_provenance(logger)

        # 3. Clean in place. auto_transfer_response="n" so the pipeline's
        #    end-of-run prompt bails out and we drive transfer explicitly.
        print(f"\n[integration/{transport}] cleaning in {input_dir}")
        clean_subject_edf_files(
            input_path=str(input_dir),
            output_path=str(input_dir),
            subject_code=fake_code,
            subject_name=PATIENT_NAME,
            inplace=True,
            raise_errors=True,
            auto_transfer_response="n",
        )
    finally:
        # Close + redact + confirm log survived — matches __main__'s
        # finally-block. Must happen BEFORE the transfer step so the
        # log.out on disk is the final, scrubbed version.
        close_logger()
        if log_path.exists():
            redact_log_file(str(log_path), PATIENT_NAME)

    assert log_path.exists(), "logger setup should have produced log.out"

    # 4. Manifest must be present after a successful clean.
    manifest = read_manifest(input_dir)
    assert manifest is not None, "clean_subject_edf_files must write a manifest"
    assert manifest["subject_code"] == fake_code

    # 4. Transfer with remote_dir_override — creates the dest, uploads.
    #    use_rsync=False forces the scp fallback path so the parametrized
    #    'scp' variant actually exercises it.
    print(f"\n[integration/{transport}] transferring to "
          f"{login}@{SSH_HOST}:{remote_dir}")
    plan = transfer_subject(
        input_dir,
        ssh_user=login,
        remote_dir_override=remote_dir,
        use_rsync=(transport == "rsync"),
    )
    assert plan.remote_dir == remote_dir
    assert plan.transport == transport

    # 5. Round-trip: pull the remote files back to an adjacent local
    #    dir. Use the SAME transport as the upload so this exercises
    #    both directions of the chosen protocol.
    roundtrip_dir = tmp_path / "roundtrip"
    roundtrip_dir.mkdir()
    print(f"\n[integration/{transport}] pulling back to {roundtrip_dir}")
    if transport == "rsync":
        subprocess.run(
            ["rsync", "-avzt",
             f"{login}@{SSH_HOST}:{remote_dir}/",
             str(roundtrip_dir) + "/"],
            check=True,
        )
    else:
        # scp -rp: recursive + preserve mtimes/perms.
        subprocess.run(
            ["scp", "-rp",
             f"{login}@{SSH_HOST}:{remote_dir}/",
             str(roundtrip_dir),
             ],
            check=True,
        )
        # scp -r copies the source directory *into* the destination
        # dir when the destination exists, creating an extra level of
        # nesting (roundtrip_dir/R1000A_scp/*) — flatten it so the
        # comparison below is symmetric with the rsync branch.
        nested = list(roundtrip_dir.iterdir())
        if len(nested) == 1 and nested[0].is_dir():
            for child in nested[0].iterdir():
                shutil.move(str(child), str(roundtrip_dir / child.name))
            nested[0].rmdir()

    # 6. Verify every file the pipeline produced round-trips
    #    byte-identically (size + mtime match).
    local_files = {p.name: p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix in (".edf", ".json", ".out")}
    remote_files = {p.name: p for p in roundtrip_dir.iterdir()
                    if p.is_file()}

    missing = set(local_files) - set(remote_files)
    assert not missing, f"files missing after round-trip: {sorted(missing)}"

    # Manifest MUST survive the round-trip (it's what the audit uses
    # to verify byte-identity post-transfer).
    assert MANIFEST_FILENAME in remote_files, (
        f"{MANIFEST_FILENAME} was not round-tripped — the audit tool "
        "won't be able to seed previous_hashes from it"
    )
    # log.out must survive too — it's the operational provenance the
    # data team relies on when triaging problems reported downstream.
    assert LOG_FILENAME in remote_files, (
        f"{LOG_FILENAME} was not round-tripped — the transferred subject "
        "would land on the cluster without its provenance log"
    )
    # log.out must have been scrubbed (name parts replaced with the
    # [PHI_REDACTED] marker) BEFORE upload. Verifying on the
    # round-tripped copy proves both that the scrub ran and that it
    # survived the transfer.
    log_text = (remote_files[LOG_FILENAME]).read_text()
    assert "[PHI_REDACTED]" in log_text or PATIENT_NAME.first_name not in log_text, (
        f"{LOG_FILENAME} still contains the un-scrubbed patient name — "
        "either the scrub did not run or it did not persist through the transfer"
    )
    # And the provenance block wrote something identifiable.
    assert "Provenance" in log_text, (
        f"{LOG_FILENAME} does not contain the '=== Provenance ===' header "
        "— log_environment_provenance did not run"
    )

    # For each EDF: (a) size matches, (b) mtime matches within tolerance,
    # (c) cryptographic content check — the round-tripped file's fresh
    # fast-hash must equal the hash the manifest recorded pre-upload.
    # (c) is the load-bearing correctness check; (a) and (b) are cheap
    # extra signals that catch e.g. truncated transfers before we spend
    # cycles hashing.
    mtime_tolerance_s = 2.0
    stored_hashes = manifest["file_hashes"]
    for name, local in local_files.items():
        if not name.endswith(".edf"):
            continue
        remote = remote_files[name]
        assert local.stat().st_size == remote.stat().st_size, (
            f"{name}: size mismatch — local {local.stat().st_size}, "
            f"remote {remote.stat().st_size}"
        )
        delta = abs(local.stat().st_mtime - remote.stat().st_mtime)
        assert delta < mtime_tolerance_s, (
            f"{name}: mtime differs by {delta:.2f}s (> {mtime_tolerance_s}s "
            f"tolerance) — the file may have been altered mid-transfer"
        )
        # Content check: hash the round-tripped file with the exact
        # same fast-hash the pipeline used pre-upload, compare to the
        # manifest's recorded digest. Cryptographic proof the bytes
        # were preserved across upload + download.
        assert name in stored_hashes, (
            f"{name}: not in manifest.file_hashes — the manifest and "
            "output dir have drifted, cannot verify round-trip content"
        )
        fresh_hash, _mode, _details = sha256_fast_of_file(remote)
        assert fresh_hash == stored_hashes[name], (
            f"{name}: round-tripped bytes DIFFER from manifest hash — "
            f"manifest {stored_hashes[name][:12]}…, "
            f"round-trip {fresh_hash[:12]}…. Transfer corrupted the file."
        )

    # 7. Cleanup local roundtrip. Remote is intentionally left in place.
    shutil.rmtree(roundtrip_dir)
    print(f"\n[integration/{transport}] SUCCESS — remote copy left at "
          f"{login}@{SSH_HOST}:{remote_dir}")
    print(f"[integration/{transport}] to clean up remote: "
          f"ssh {login}@{SSH_HOST} 'rm -rf {remote_dir}'")
