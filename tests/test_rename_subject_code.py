"""Tests for scripts/rename_subject_code.py.

Coverage priorities (in order):
    1. Corruption safety: pre-write verification + backup + restore
    2. Byte-identical roundtrip: A -> B -> A produces the same bytes
    3. All touchpoints get updated: folder, filenames, headers, manifest
    4. Dry-run touches nothing; refuses on missing source / existing target

Real pyedflib is used (no mocks) because the header-byte manipulation
is the whole point of the tool -- mocking it would test nothing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import pytest


# Import the script directly by path -- it's under scripts/, not on
# sys.path as an installed module.
_SCRIPT_PATH = (Path(__file__).parent.parent
                / "scripts" / "rename_subject_code.py")
_spec = importlib.util.spec_from_file_location(
    "rename_subject_code", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
rename_module = importlib.util.module_from_spec(_spec)
sys.modules["rename_subject_code"] = rename_module
_spec.loader.exec_module(rename_module)

build_plan = rename_module.build_plan
execute_plan = rename_module.execute_plan
main = rename_module.main
update_edf_patientcode_safely = rename_module.update_edf_patientcode_safely
update_deidentify_manifest = rename_module.update_deidentify_manifest
BACKUP_SUFFIX = rename_module.BACKUP_SUFFIX


FROM_CODE = "R1655J"
TO_CODE = "R1665J"


def _write_test_edf(path: Path, patientcode: str,
                     patientname: str = "TEST NAME",
                     n_channels: int = 3, sample_rate: int = 100,
                     duration_s: int = 2) -> None:
    """Minimal EDF+C written by pyedflib with a known patientcode.
    Uses pyedflib for both this initial write AND the in-place header
    rewrite so the roundtrip test can assume consistent byte layout."""
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
    with pyedflib.EdfWriter(str(path), n_channels,
                             file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader({
            "technician": "T", "recording_additional": "",
            "patientname": patientname, "patient_additional": "",
            "patientcode": patientcode, "equipment": "X",
            "admincode": "", "sex": "X",
            "startdate": datetime(2023, 1, 1, 10, 0, 0),
            "birthdate": "01 jan 1970", "gender": "X",
        })
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)
        f.writeAnnotation(0.5, -1, "START")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_subject_tree(root: Path, code: str, n_edfs: int = 2) -> Path:
    """Layout mirroring the real deployment:
        subjects/<code>/clinical_eeg/<code>_file_N.edf
        subjects/<code>/clinical_eeg/deidentify.json
    Returns the subject directory (subjects/<code>)."""
    subj = root / code
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)
    for i in range(n_edfs):
        _write_test_edf(inner / f"{code}_file_{i}.edf", patientcode=code)
    manifest = {
        "subject_code": code,
        "file_hashes": {f"{code}_file_{i}.edf": f"fakehash{i}"
                        for i in range(n_edfs)},
        "generated_at": "2026-08-20T00:00:00",
    }
    (inner / "deidentify.json").write_text(json.dumps(manifest, indent=2))
    return subj


# ---------------------------------------------------------------------------
# Planning: correctness + safety refusals
# ---------------------------------------------------------------------------

def test_plan_lists_all_touchpoints(tmp_path):
    """The plan must enumerate every place the code needs to change:
    per-file renames, EDF header updates, manifest update, folder
    rename. Regression guard against a future change that silently
    stops finding one category."""
    root = tmp_path / "subjects"
    _make_subject_tree(root, FROM_CODE, n_edfs=2)

    plan = build_plan(root, FROM_CODE, TO_CODE)

    # 2 EDF files -> 2 file renames + 2 header updates + 1 manifest +
    # 1 folder rename. clinical_eeg/ is not renamed because its name
    # doesn't contain the code.
    assert len(plan.edf_header_updates) == 2
    assert len(plan.manifest_updates) == 1
    # File renames: 2 EDFs whose names contain the code. Plus the
    # subject folder itself. Depth-first: files first, then folder.
    file_renames = [(s, d) for s, d in plan.path_renames
                    if s.suffix == ".edf"]
    assert len(file_renames) == 2
    folder_rename = plan.path_renames[-1]  # last
    assert folder_rename[0].name == FROM_CODE
    assert folder_rename[1].name == TO_CODE


def test_plan_refuses_when_source_missing(tmp_path):
    """Guards against operator typo in --from. Silently proceeding
    with no work done would be confusing; explicit error is better."""
    with pytest.raises(FileNotFoundError):
        build_plan(tmp_path, "R9999Z", TO_CODE)


def test_plan_refuses_when_target_already_exists(tmp_path):
    """Guards against overwriting an existing subject's dir."""
    root = tmp_path / "subjects"
    _make_subject_tree(root, FROM_CODE)
    (root / TO_CODE).mkdir()

    with pytest.raises(FileExistsError):
        build_plan(root, FROM_CODE, TO_CODE)


# ---------------------------------------------------------------------------
# Header safety: pre-write verification + backup + restore
# ---------------------------------------------------------------------------

def test_update_edf_patientcode_safely_changes_header(tmp_path):
    """Positive path: header updates, file remains loadable, backup
    is cleaned up."""
    edf = tmp_path / f"{FROM_CODE}_file.edf"
    _write_test_edf(edf, patientcode=FROM_CODE)

    update_edf_patientcode_safely(edf, FROM_CODE, TO_CODE)

    with pyedflib.EdfReader(str(edf)) as f:
        assert f.getHeader()["patientcode"] == TO_CODE
    # Backup must be cleaned up on success
    assert not edf.with_suffix(edf.suffix + BACKUP_SUFFIX).exists()


def test_update_edf_patientcode_safely_preserves_signal_bytes(tmp_path):
    """Signal data must be byte-identical after header update.
    ``update_edf_header_inplace`` already asserts this internally with
    ``confirm_signals_unchanged=True``; this test proves the safety
    wrapper doesn't break that guarantee."""
    edf = tmp_path / f"{FROM_CODE}_file.edf"
    _write_test_edf(edf, patientcode=FROM_CODE)

    with pyedflib.EdfReader(str(edf)) as f:
        before = [f.readSignal(i) for i in range(f.signals_in_file)]

    update_edf_patientcode_safely(edf, FROM_CODE, TO_CODE)

    with pyedflib.EdfReader(str(edf)) as f:
        after = [f.readSignal(i) for i in range(f.signals_in_file)]

    for i, (b, a) in enumerate(zip(before, after)):
        assert np.array_equal(b, a), f"signal {i} changed after rename"


def test_update_edf_patientcode_safely_restores_on_failure(tmp_path,
                                                            monkeypatch):
    """CORRUPTION-SAFETY REGRESSION: if the post-write verification
    fails (simulating a bad write that produced wrong patientcode or
    an unreadable file), the file must be restored from backup and
    the exception must propagate. The user's data is protected even
    when the underlying primitive misbehaves.
    """
    edf = tmp_path / f"{FROM_CODE}_file.edf"
    _write_test_edf(edf, patientcode=FROM_CODE)
    original_bytes = edf.read_bytes()

    # Force the verifier to fail no matter what -- simulates a bad
    # write that produced garbage. In the real script this is what
    # would fire if pyedflib silently wrote wrong bytes.
    def fake_verify(path, expected):
        raise RuntimeError("simulated corruption detected")
    monkeypatch.setattr(rename_module, "_verify_edf_header_updated",
                        fake_verify)

    with pytest.raises(RuntimeError, match="simulated corruption"):
        update_edf_patientcode_safely(edf, FROM_CODE, TO_CODE)

    # File must be restored to pre-write bytes
    assert edf.read_bytes() == original_bytes, (
        "file was NOT restored from backup after simulated write failure")
    # Backup file must be cleaned up (no leftover .bak)
    assert not edf.with_suffix(edf.suffix + BACKUP_SUFFIX).exists()


def test_update_edf_patientcode_safely_refuses_leftover_backup(tmp_path):
    """Refuses to overwrite a pre-existing .rename.bak file (would mean
    an earlier run crashed mid-rename and the operator hasn't
    inspected the situation)."""
    edf = tmp_path / f"{FROM_CODE}_file.edf"
    _write_test_edf(edf, patientcode=FROM_CODE)
    (edf.with_suffix(edf.suffix + BACKUP_SUFFIX)).write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="backup file"):
        update_edf_patientcode_safely(edf, FROM_CODE, TO_CODE)


# ---------------------------------------------------------------------------
# Roundtrip byte-identity
# ---------------------------------------------------------------------------

def test_roundtrip_rename_is_byte_identical(tmp_path):
    """HARD REQUIREMENT: renaming A -> B -> A on a pyedflib-written
    EDF must produce a file byte-identical to the original. Guards
    against subtle format drift where the two rewrites produce
    slightly different byte layouts (padding, whitespace, field
    ordering) that would over time accumulate spurious diffs.

    Only holds when the source EDF was already written by pyedflib
    (as it will be in production, since our pipeline writes EDFs via
    pyedflib). An EDF written by another tool may have different
    header padding on its very first pyedflib rewrite -- but from
    THEN on, subsequent roundtrips are byte-identical.
    """
    edf = tmp_path / f"{FROM_CODE}_file.edf"
    _write_test_edf(edf, patientcode=FROM_CODE)

    # First rewrite: A -> B. This is where format drift (if any)
    # would happen. We compute the hash AFTER this initial pyedflib-
    # written-and-rewritten state, not before, so the roundtrip check
    # tests the property that matters in production: two rewrites of
    # the same pyedflib-managed file are consistent.
    update_edf_patientcode_safely(edf, FROM_CODE, TO_CODE)
    hash_at_B = _sha256(edf)

    # Now roundtrip: B -> A -> B
    update_edf_patientcode_safely(edf, TO_CODE, FROM_CODE)
    update_edf_patientcode_safely(edf, FROM_CODE, TO_CODE)
    hash_after_roundtrip = _sha256(edf)

    assert hash_after_roundtrip == hash_at_B, (
        "roundtrip A->B->A->B is NOT byte-identical -- format drift "
        "in update_edf_header_inplace. This means repeated renames "
        "would silently accumulate diffs.")


# ---------------------------------------------------------------------------
# Manifest updates
# ---------------------------------------------------------------------------

def test_manifest_update_rewrites_subject_code_and_hash_keys(tmp_path):
    manifest_path = tmp_path / "deidentify.json"
    manifest_path.write_text(json.dumps({
        "subject_code": FROM_CODE,
        "file_hashes": {
            f"{FROM_CODE}_a.edf": "hash_a",
            f"{FROM_CODE}_b.edf": "hash_b",
            # Non-code-embedded key must NOT be touched
            "log.out": "hash_log",
        },
        "generated_at": "2026-08-20",
    }))

    update_deidentify_manifest(manifest_path, FROM_CODE, TO_CODE)

    m = json.loads(manifest_path.read_text())
    assert m["subject_code"] == TO_CODE
    # Old keys gone, new keys present, values preserved
    assert f"{FROM_CODE}_a.edf" not in m["file_hashes"]
    assert m["file_hashes"][f"{TO_CODE}_a.edf"] == "hash_a"
    assert m["file_hashes"][f"{TO_CODE}_b.edf"] == "hash_b"
    # Unrelated key untouched
    assert m["file_hashes"]["log.out"] == "hash_log"
    # Other fields untouched
    assert m["generated_at"] == "2026-08-20"


# ---------------------------------------------------------------------------
# End-to-end: execute_plan touches every touchpoint
# ---------------------------------------------------------------------------

def test_execute_plan_updates_all_touchpoints(tmp_path):
    """POSITIVE integration test: after execute_plan completes, the
    old code appears NOWHERE under the subject tree, and the new
    code appears in every expected place."""
    root = tmp_path / "subjects"
    subj = _make_subject_tree(root, FROM_CODE, n_edfs=2)
    assert subj.exists()

    plan = build_plan(root, FROM_CODE, TO_CODE)
    execute_plan(plan, FROM_CODE, TO_CODE)

    # Old folder gone, new folder present
    assert not (root / FROM_CODE).exists()
    new_dir = root / TO_CODE
    assert new_dir.exists()

    # EDF files renamed
    edfs = sorted(new_dir.rglob("*.edf"))
    assert len(edfs) == 2
    for edf in edfs:
        assert FROM_CODE not in edf.name
        assert TO_CODE in edf.name
        # Headers updated
        with pyedflib.EdfReader(str(edf)) as f:
            assert f.getHeader()["patientcode"] == TO_CODE

    # Manifest updated
    manifest = json.loads(
        (new_dir / "clinical_eeg" / "deidentify.json").read_text())
    assert manifest["subject_code"] == TO_CODE
    for k in manifest["file_hashes"]:
        assert FROM_CODE not in k
        assert TO_CODE in k


def test_execute_plan_leaves_no_backup_files_on_success(tmp_path):
    """Negative regression: after a successful run, no .rename.bak
    files should be left behind."""
    root = tmp_path / "subjects"
    _make_subject_tree(root, FROM_CODE, n_edfs=2)

    plan = build_plan(root, FROM_CODE, TO_CODE)
    execute_plan(plan, FROM_CODE, TO_CODE)

    leftovers = list((root / TO_CODE).rglob(f"*{BACKUP_SUFFIX}"))
    assert leftovers == [], f"backup files leaked: {leftovers}"


# ---------------------------------------------------------------------------
# CLI: dry-run touches nothing
# ---------------------------------------------------------------------------

def test_cli_dry_run_touches_nothing(tmp_path):
    """The CLI without --apply must be a pure preview: no file
    renamed, no header changed, no manifest modified."""
    root = tmp_path / "subjects"
    subj = _make_subject_tree(root, FROM_CODE, n_edfs=2)

    # Snapshot the whole tree
    before = {p.relative_to(root): (p.stat().st_size if p.is_file() else -1)
              for p in root.rglob("*")}
    before_hashes = {p: _sha256(p) for p in subj.rglob("*.edf")}
    before_manifest = (subj / "clinical_eeg" / "deidentify.json").read_text()

    rc = main([
        "--subject-root", str(root),
        "--from", FROM_CODE, "--to", TO_CODE,
    ])
    assert rc == 0

    after = {p.relative_to(root): (p.stat().st_size if p.is_file() else -1)
             for p in root.rglob("*")}
    assert after == before, "dry-run modified the tree"
    for p, h in before_hashes.items():
        assert _sha256(p) == h, f"dry-run modified {p}"
    after_manifest = (subj / "clinical_eeg" / "deidentify.json").read_text()
    assert after_manifest == before_manifest


def test_cli_apply_actually_renames(tmp_path):
    """Sanity: with --apply the tree IS mutated."""
    root = tmp_path / "subjects"
    _make_subject_tree(root, FROM_CODE, n_edfs=1)

    rc = main([
        "--subject-root", str(root),
        "--from", FROM_CODE, "--to", TO_CODE,
        "--apply",
    ])
    assert rc == 0
    assert not (root / FROM_CODE).exists()
    assert (root / TO_CODE).exists()


# ---------------------------------------------------------------------------
# Raw-NK regression: pyedflib.EdfReader failure must NOT hide files
# whose byte-level patient_id is still readable
# ---------------------------------------------------------------------------

def _write_edf_then_truncate_signal_region(path: Path, patientcode: str
                                            ) -> None:
    """Write a valid EDF via pyedflib, then truncate the file so only
    the main-header 256 bytes survive. The patient_id field lives at
    bytes 8..88 of that header, so byte-level reads still work; but
    pyedflib.EdfReader will refuse the truncated file (declared
    num_data_records don't match the physically-present bytes).

    Simulates the class of files that made
    'find_edf_files_needing_header_update' spuriously skip everything
    in the original bug report.
    """
    _write_test_edf(path, patientcode=patientcode)
    full = path.read_bytes()
    assert len(full) > 256, "test fixture too small to truncate"
    # Keep the main header only. pyedflib.EdfReader will now fail
    # on the signal-region reads.
    path.write_bytes(full[:256])


def test_find_edf_files_needing_header_update_reads_via_bytes(tmp_path,
                                                                capsys):
    """REGRESSION: raw NK exports that pyedflib.EdfReader refuses
    (declared record count doesn't match on-disk bytes, non-EDF+
    reserved field, etc.) MUST still have their patient_id inspected
    via byte-level reading. Missing this class of file was the
    original 'cannot open header' bug -- the operator would see every
    EDF spuriously skipped.
    """
    subj = tmp_path / FROM_CODE
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)

    # File 1: patient_id contains FROM_CODE, pyedflib-unreadable
    match_path = inner / f"{FROM_CODE}_match.edf"
    _write_edf_then_truncate_signal_region(match_path, FROM_CODE)

    # File 2: patient_id is a hospital MRN (simulating raw NK data),
    # also pyedflib-unreadable. MUST NOT be picked up for header
    # update -- raw NK data shouldn't be touched.
    mrn_path = inner / "MRN12345.edf"
    _write_edf_then_truncate_signal_region(mrn_path, "MRN12345")

    # Confirm the test premise: pyedflib.EdfReader really does fail
    # on these. If pyedflib gets more lenient in a future version,
    # this assertion will let us know to rebuild the fixture.
    with pytest.raises((OSError, ValueError)):
        with pyedflib.EdfReader(str(match_path)):
            pass

    hits = rename_module.find_edf_files_needing_header_update(
        subj, FROM_CODE)

    assert hits == [match_path], (
        f"expected only {match_path.name} to match (bytes contain "
        f"{FROM_CODE}), got: {[p.name for p in hits]}")
    # No spurious skip warnings for either file -- both had readable
    # patient_id bytes.
    err = capsys.readouterr().err
    assert "skip-read" not in err, (
        f"byte-level read should succeed on both fixtures; got: {err}")


def test_find_edf_files_surfaces_actual_exception_on_unreadable_file(
        tmp_path, capsys):
    """REGRESSION (UX): when the byte-level read itself fails (file
    too short to contain the patient_id field, permission denied,
    etc.), the operator must see WHY -- not a generic 'cannot open
    header' with no diagnostic. That was the second half of the
    original bug: silent failure with no way to debug.
    """
    subj = tmp_path / FROM_CODE
    inner = subj / "clinical_eeg"
    inner.mkdir(parents=True)

    # File shorter than the patient_id field's end (byte 88)
    tiny = inner / "corrupt.edf"
    tiny.write_bytes(b"garbage")

    hits = rename_module.find_edf_files_needing_header_update(
        subj, FROM_CODE)
    assert hits == []
    err = capsys.readouterr().err
    # Diagnostic must include: the full path, an exception class,
    # and something interpretable about what went wrong. The path
    # matters because in a batch of hundreds of files the operator
    # needs to pinpoint the offender.
    assert str(tiny) in err
    # Exception CLASS surfaced, not just an opaque generic message
    assert any(cls in err for cls in
               ("KeyError", "ValueError", "OSError", "IndexError")), (
        f"expected an exception class name in stderr, got: {err}")


def test_cli_returns_nonzero_when_source_missing(tmp_path, capsys):
    """Operator typo in --from: helpful message + exit code 2."""
    rc = main([
        "--subject-root", str(tmp_path),
        "--from", "R9999Z", "--to", TO_CODE,
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "R9999Z" in err or "does not exist" in err
