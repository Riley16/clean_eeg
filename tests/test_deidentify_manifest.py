"""Tests for the ``deidentify.json`` sidecar written on successful
completion of ``clean_subject_edf_files``.

Uses the conftest-generated subject fixtures for real fast-hash coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clean_eeg.deidentify_manifest import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    ManifestSchemaError,
    ReviewEvent,
    build_manifest,
    manifest_exists,
    read_manifest,
    refresh_annotation_sidecar_hashes,
    write_manifest,
)
from clean_eeg.paths import TEST_CONFIG_FILE, TEST_SUBJECT_DATA_DIR


with open(TEST_CONFIG_FILE, "r") as f:
    TEST_CONFIG = json.load(f)
SUBJECT_EDF_1 = TEST_SUBJECT_DATA_DIR / TEST_CONFIG["subject_EDF+C_1"]["filename"]
SUBJECT_EDF_2 = TEST_SUBJECT_DATA_DIR / TEST_CONFIG["subject_EDF+C_2"]["filename"]


def _base_kwargs(tmp_path: Path, edf_paths: list[Path]) -> dict:
    return dict(
        subject_code="R1755A",
        site_code="A",
        site_incoming_folder="CUDA",
        input_path=str(tmp_path / "in"),
        output_path=str(tmp_path),
        inplace=True,
        output_edf_paths=edf_paths,
        n_files_deidentified=len(edf_paths),
        n_files_failed=0,
        n_files_quarantined=0,
    )


def test_write_then_read_roundtrip(tmp_path):
    manifest = build_manifest(**_base_kwargs(tmp_path, [SUBJECT_EDF_1]))
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    # Compare structural fields; timestamps and hashes are deterministic
    # within a single test run so equality of the whole dict holds.
    assert loaded == manifest


def test_read_returns_none_when_absent(tmp_path):
    assert read_manifest(tmp_path) is None
    assert not manifest_exists(tmp_path)


def test_manifest_exists_flips_after_write(tmp_path):
    assert not manifest_exists(tmp_path)
    write_manifest(tmp_path, build_manifest(**_base_kwargs(tmp_path, [])))
    assert manifest_exists(tmp_path)


def test_schema_version_mismatch_raises(tmp_path):
    p = tmp_path / MANIFEST_FILENAME
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}))
    with pytest.raises(ManifestSchemaError, match="schema_version"):
        read_manifest(tmp_path)


def test_manifest_hashes_every_output_edf(tmp_path):
    edfs = [SUBJECT_EDF_1, SUBJECT_EDF_2]
    manifest = build_manifest(**_base_kwargs(tmp_path, edfs))
    assert set(manifest["file_hashes"].keys()) == {p.name for p in edfs}
    assert set(manifest["hash_mode_by_file"].keys()) == {p.name for p in edfs}
    for digest in manifest["file_hashes"].values():
        # Fast hash returns a SHA-256 hex digest (64 chars).
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


def test_manifest_records_site_metadata(tmp_path):
    manifest = build_manifest(**_base_kwargs(tmp_path, []))
    assert manifest["site_code"] == "A"
    assert manifest["site_incoming_folder"] == "CUDA"
    assert manifest["subject_code"] == "R1755A"


def test_manifest_serializes_review_events(tmp_path):
    events = [
        ReviewEvent(kind="annotation_redaction",
                    file="R1755A_..._01.01__10.00.00.edf",
                    details={"redacted_value": "XXXX", "boilerplate": False}),
        ReviewEvent(kind="header_truncation",
                    file="R1755A_..._01.01__10.00.00.edf",
                    details={"message": "patient_id truncated"}),
    ]
    kwargs = _base_kwargs(tmp_path, [])
    kwargs["review_events"] = events
    manifest = build_manifest(**kwargs)
    assert len(manifest["review_events"]) == 2
    assert manifest["review_events"][0]["kind"] == "annotation_redaction"
    assert manifest["review_events"][0]["redacted_value"] == "XXXX"
    assert manifest["review_events"][0]["boilerplate"] is False
    assert manifest["review_events"][1]["message"] == "patient_id truncated"


def test_manifest_handles_zero_files(tmp_path):
    """An aborted-early run should still be writable — no files, no
    hashes, but the manifest structure is intact."""
    manifest = build_manifest(**_base_kwargs(tmp_path, []))
    assert manifest["file_hashes"] == {}
    assert manifest["n_files_deidentified"] == 0
    write_manifest(tmp_path, manifest)
    assert read_manifest(tmp_path) == manifest


def test_refresh_annotation_sidecar_hashes_updates_only_annotation_files(tmp_path):
    """Post-annotation-review manifest refresh: sidecar hashes get
    recomputed but signal-EDF hashes stay at their pipeline-write
    values. Guards against the fix accidentally overwriting the
    signal-integrity guarantee that the transfer preflight relies
    on."""
    signal_edf = tmp_path / "clean_R1755A_01.edf"
    sidecar = tmp_path / "clean_R1755A_01_annotations.edf"
    signal_edf.write_bytes(b"\x00" * 512)   # placeholder signal EDF
    sidecar.write_bytes(b"orig-sidecar-bytes")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "subject_code": "R1755A",
        "file_hashes": {
            signal_edf.name: "SIGNAL_HASH_FROM_PIPELINE",
            sidecar.name: "OLD_SIDECAR_HASH",
        },
        "hash_mode_by_file": {
            signal_edf.name: "fast", sidecar.name: "fast"},
        "hash_details_by_file": {
            signal_edf.name: {}, sidecar.name: {}},
    }
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))

    # Annotation-review mutates the sidecar; simulate.
    sidecar.write_bytes(b"NEW-sidecar-bytes-after-apply")

    changed = refresh_annotation_sidecar_hashes(
        tmp_path, [signal_edf, sidecar])

    # The sidecar's hash should have changed; the signal EDF isn't a
    # sidecar so it's skipped even though it's in modified_paths.
    assert sidecar.name in changed
    assert signal_edf.name not in changed

    refreshed = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert refreshed["file_hashes"][sidecar.name] != "OLD_SIDECAR_HASH"
    assert refreshed["file_hashes"][signal_edf.name] == \
        "SIGNAL_HASH_FROM_PIPELINE"


def test_refresh_annotation_sidecar_hashes_noop_when_unchanged(tmp_path):
    """If the sidecar bytes haven't changed since the manifest was
    written (e.g. annotation-review ran but every edit was a no-op),
    the refresh reports no changed files and doesn't rewrite the
    manifest."""
    sidecar = tmp_path / "clean_R1755A_01_annotations.edf"
    sidecar.write_bytes(b"stable")

    # Compute the real hash so the manifest is already consistent.
    from clean_eeg.audit.hashes import sha256_fast_of_file
    digest, mode, det = sha256_fast_of_file(sidecar)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "subject_code": "R1755A",
        "file_hashes": {sidecar.name: digest},
        "hash_mode_by_file": {sidecar.name: mode},
        "hash_details_by_file": {sidecar.name: det},
    }
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest))
    mtime_before = manifest_path.stat().st_mtime_ns

    changed = refresh_annotation_sidecar_hashes(tmp_path, [sidecar])
    assert changed == {}
    # Manifest not rewritten -> mtime unchanged.
    assert manifest_path.stat().st_mtime_ns == mtime_before


def test_manifest_records_provenance_fields(tmp_path):
    manifest = build_manifest(**_base_kwargs(tmp_path, []))
    # clean_eeg_version always resolves to something (either a real
    # version or the "not installed" sentinel from provenance).
    assert isinstance(manifest["clean_eeg_version"], str)
    assert manifest["clean_eeg_version"] != ""
    # git_commit can be None (non-checkout) or a str (checkout).
    assert manifest["git_commit"] is None or isinstance(manifest["git_commit"], str)
    assert isinstance(manifest["git_dirty"], bool)
    assert manifest["schema_version"] == SCHEMA_VERSION
