"""Individual audit checks. Each returns a JSON-serializable dict with
at minimum ``check``, ``status`` (``pass`` | ``warn`` | ``fail``), and
``issues`` (list of human-readable strings). Callers aggregate the
returned dicts into ``edf_audit.json``.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

from clean_eeg.clean_subject_eeg import (
    BASE_START_DATE,
    MAX_RECORDING_GAP_SECONDS,
    SUBJECT_CODE_PATTERN,
    is_valid_subject_code,
)
from clean_eeg.print_edf_header import (
    ANNOTATION_STUB_SUFFIX,
    EDF_ANNOTATION_LABEL,
    MAIN_HEADER_BYTES,
    SIGNAL_HEADER_BYTES_PER_SIGNAL,
    read_main_header,
    read_signal_headers,
)


PATIENT_ID_SENTINEL_TOKENS: frozenset[str] = frozenset({
    "X",          # sex / gender / patient_additional placeholder
    "unknown",    # pyedflib default when patientname is missing
    "01-JAN-1900",  # birthdate placeholder written by the pipeline
})


def _parse_edf_startdate(startdate_str: str) -> date | None:
    """Parse EDF main-header ``startdate`` (``DD.MM.YY``). Per EDF+ clip
    semantics, ``YY`` in 85..99 → 1985..1999, in 00..84 → 2000..2084.
    Returns ``None`` on any parse error.
    """
    try:
        dd_s, mm_s, yy_s = startdate_str.split(".")
        yy = int(yy_s)
        year = 1900 + yy if yy >= 85 else 2000 + yy
        return date(year, int(mm_s), int(dd_s))
    except (ValueError, AttributeError):
        return None


def check_header_phi_residue(edf_paths: Iterable[str | Path],
                             *,
                             max_span_years: int = 2) -> dict:
    """Verify cleaned EDFs carry only sentinel PHI values in the main header.

    Checks per file:
      - ``patient_id`` subfields after the subject code are all in
        ``PATIENT_ID_SENTINEL_TOKENS``. Any other token is flagged as
        potential residue (e.g., a leaked name or real birthdate).
      - ``startdate`` parses and its year is in
        ``[BASE_START_DATE.year, BASE_START_DATE.year + max_span_years]``.
        A pre-clean year (e.g., 2024) fails loudly.
    Additionally verifies the earliest startdate across all files equals
    ``BASE_START_DATE`` (pipeline invariant).
    """
    per_file: dict[str, dict] = {}
    unexpected_tokens_by_file: dict[str, list[str]] = {}
    startdates_by_file: dict[str, str] = {}
    parsed_startdates: dict[str, date] = {}

    for p in edf_paths:
        p = Path(p)
        header = read_main_header(str(p))
        pid = header.get("patient_id", "")
        pid = pid if isinstance(pid, str) else str(pid)
        tokens = pid.strip().split()
        non_sentinel = [t for t in tokens[1:] if t not in PATIENT_ID_SENTINEL_TOKENS]
        unexpected_tokens_by_file[p.name] = non_sentinel

        sd_raw = header.get("startdate", "")
        sd_raw = sd_raw if isinstance(sd_raw, str) else str(sd_raw)
        startdates_by_file[p.name] = sd_raw
        sd = _parse_edf_startdate(sd_raw)
        if sd is not None:
            parsed_startdates[p.name] = sd

        per_file[p.name] = {"patient_id": pid, "startdate": sd_raw}

    issues: list[str] = []
    year_lo = BASE_START_DATE.year
    year_hi = BASE_START_DATE.year + max_span_years
    out_of_range = {name: sd for name, sd in parsed_startdates.items()
                    if not (year_lo <= sd.year <= year_hi)}
    unparseable = [name for name in startdates_by_file
                   if name not in parsed_startdates]
    residue_files = {name: toks for name, toks in unexpected_tokens_by_file.items() if toks}

    if not per_file:
        status = "fail"
        issues.append("No EDF files were provided")
        earliest = None
    else:
        status = "pass"
        earliest = min(parsed_startdates.values()) if parsed_startdates else None
        if residue_files:
            status = "fail"
            for name, toks in residue_files.items():
                issues.append(f"{name}: non-sentinel tokens in patient_id: {toks}")
        if out_of_range:
            status = "fail"
            for name, sd in out_of_range.items():
                issues.append(
                    f"{name}: startdate year {sd.year} outside expected "
                    f"[{year_lo}, {year_hi}]"
                )
        if unparseable:
            status = "fail"
            for name in unparseable:
                issues.append(f"{name}: unparseable startdate {startdates_by_file[name]!r}")
        if earliest is not None and earliest != BASE_START_DATE.date():
            status = "warn" if status == "pass" else "fail"
            issues.append(
                f"Earliest startdate {earliest.isoformat()} != pipeline "
                f"invariant {BASE_START_DATE.date().isoformat()}"
            )

    return {
        "check": "header_phi_residue",
        "status": status,
        "n_files": len(per_file),
        "expected_year_range": [year_lo, year_hi],
        "earliest_startdate": earliest.isoformat() if earliest else None,
        "startdates_by_file": startdates_by_file,
        "patient_ids_by_file": {name: v["patient_id"] for name, v in per_file.items()},
        "unexpected_patient_id_tokens_by_file": {
            name: toks for name, toks in unexpected_tokens_by_file.items() if toks
        },
        "sentinel_tokens": sorted(PATIENT_ID_SENTINEL_TOKENS),
        "issues": issues,
    }


def _parse_edf_starttime(starttime_str: str) -> time | None:
    """Parse EDF main-header ``starttime`` (``HH.MM.SS``). Returns
    ``None`` on any parse error."""
    try:
        hh_s, mm_s, ss_s = starttime_str.split(".")
        return time(int(hh_s), int(mm_s), int(ss_s))
    except (ValueError, AttributeError):
        return None


def check_recording_gaps(edf_paths: Iterable[str | Path],
                         *,
                         max_gap_seconds: float = MAX_RECORDING_GAP_SECONDS) -> dict:
    """Detect anomalous gaps or overlaps between consecutive recordings
    on the *cleaned* (relative-to-1985) timeline. Large gaps typically
    indicate a file missing from the transfer; overlaps indicate a
    duplicate or reordering.

    Duration per file = ``n_records * record_duration`` from the main
    header. Files are sorted by their start ``datetime`` (startdate +
    starttime). Gaps < 0 are overlaps, gaps > ``max_gap_seconds`` are
    flagged as large. Consistent with the pipeline's threshold.
    """
    per_file: list[dict] = []
    unparseable: list[str] = []
    for p in edf_paths:
        p = Path(p)
        header = read_main_header(str(p))
        sd = _parse_edf_startdate(str(header.get("startdate", "")))
        st = _parse_edf_starttime(str(header.get("starttime", "")))
        n_rec = header.get("n_records")
        rec_dur = header.get("record_duration")
        if (sd is None or st is None
                or not isinstance(n_rec, int) or not isinstance(rec_dur, float)
                or n_rec <= 0 or rec_dur <= 0):
            # n_rec == -1 is the EDF sentinel for "unknown/streaming" —
            # the pipeline repairs it before de-id, so seeing it in a
            # transferred file is itself a red flag.
            unparseable.append(p.name)
            continue
        start = datetime.combine(sd, st)
        duration_s = n_rec * rec_dur
        per_file.append({
            "file": p.name,
            "start": start.isoformat(),
            "duration_s": duration_s,
            "end": (start + timedelta(seconds=duration_s)).isoformat(),
        })

    per_file.sort(key=lambda d: d["start"])

    gaps: list[dict] = []
    large_gaps: list[dict] = []
    overlaps: list[dict] = []
    for prev, nxt in zip(per_file, per_file[1:]):
        gap_s = (datetime.fromisoformat(nxt["start"])
                 - datetime.fromisoformat(prev["end"])).total_seconds()
        entry = {"prev_file": prev["file"], "next_file": nxt["file"],
                 "gap_seconds": gap_s}
        gaps.append(entry)
        if gap_s < 0:
            overlaps.append(entry)
        elif gap_s > max_gap_seconds:
            large_gaps.append(entry)

    issues: list[str] = []
    if not per_file and not unparseable:
        status = "fail"
        issues.append("No EDF files were provided")
    else:
        status = "pass"
        for u in unparseable:
            status = "fail"
            issues.append(f"{u}: could not parse startdate/starttime/n_records/record_duration")
        for g in large_gaps:
            status = "fail"
            issues.append(
                f"Large gap of {g['gap_seconds']:.1f}s between {g['prev_file']!r} "
                f"and {g['next_file']!r} (threshold {max_gap_seconds}s) — file possibly missing"
            )
        # Overlaps compress: multi-day recordings often carry the same
        # sub-second boundary overlap on every consecutive-file pair (a
        # recorder-format quirk, not per-pair drama). Show the first
        # OVERLAP_INDIVIDUAL_MAX individually plus any that exceed the
        # running max by more than OVERLAP_JUMP_S seconds — the rest
        # collapse into one line so the summary stays scannable. Full
        # per-pair detail is preserved in the JSON under 'overlaps'.
        if overlaps:
            status = "fail"
            OVERLAP_INDIVIDUAL_MAX = 5
            OVERLAP_JUMP_S = 2.0
            running_max_abs = 0.0
            n_shown = 0
            n_compressed = 0
            for o in overlaps:
                abs_gap = abs(o["gap_seconds"])
                if (n_shown < OVERLAP_INDIVIDUAL_MAX
                        or abs_gap > running_max_abs + OVERLAP_JUMP_S):
                    issues.append(
                        f"Overlap of {abs_gap:.1f}s between "
                        f"{o['prev_file']!r} and {o['next_file']!r} — "
                        "possible duplicate/reorder"
                    )
                    n_shown += 1
                    running_max_abs = max(running_max_abs, abs_gap)
                else:
                    n_compressed += 1
            if n_compressed > 0:
                issues.append(
                    f"{n_compressed} more consecutive-file overlap(s) "
                    f"within {OVERLAP_JUMP_S:.0f}s of the "
                    f"{running_max_abs:.1f}s max — full list in "
                    f"'overlaps' in edf_audit.json"
                )

    return {
        "check": "recording_gaps",
        "status": status,
        "n_files": len(per_file) + len(unparseable),
        "max_gap_seconds_threshold": max_gap_seconds,
        "files_by_start": per_file,
        "gaps": gaps,
        "large_gaps": large_gaps,
        "overlaps": overlaps,
        "unparseable_files": unparseable,
        "issues": issues,
    }


def check_byte_geometry(edf_paths: Iterable[str | Path]) -> dict:
    """Verify each EDF's on-disk byte count matches what its header claims.

    Per file:
      - expected header bytes  = 256 * (1 + n_signals)
      - record stride          = sum(samples_per_record) * 2  (int16)
      - n_records computed     = (filesize - header_bytes) // record_stride
      - verdict is OK / TRUNCATED / OVER-SIZED / UNCOMPUTABLE.
    Reuses the same logic as ``print_edf_header``'s derived-geometry
    section so audit and manual debugging agree.

    Status:
      - ``pass`` — every file OK
      - ``fail`` — any TRUNCATED file (real data loss)
      - ``warn`` — any OVER-SIZED or UNCOMPUTABLE (unusual but pyedflib
        tolerates over-sized on read, per EDF+ spec)
    """
    per_file: dict[str, dict] = {}
    truncated: list[str] = []
    oversized: list[str] = []
    uncomputable: list[str] = []
    ok: list[str] = []

    for p in edf_paths:
        p = Path(p)
        header = read_main_header(str(p))
        n_signals = header.get("n_signals")
        n_records_claimed = header.get("n_records")
        file_size = p.stat().st_size

        info: dict = {"file_size": file_size, "n_records_claimed": n_records_claimed}

        if not isinstance(n_signals, int) or n_signals <= 0:
            info["verdict"] = "UNCOMPUTABLE (n_signals not a positive int)"
            uncomputable.append(p.name)
            per_file[p.name] = info
            continue

        sigs = read_signal_headers(str(p), n_signals)
        spr_values = [s.get("samples_per_record") for s in sigs]
        expected_header_bytes = MAIN_HEADER_BYTES + n_signals * SIGNAL_HEADER_BYTES_PER_SIGNAL
        info["expected_header_bytes"] = expected_header_bytes
        info["n_signals"] = n_signals

        if not all(isinstance(v, int) and v > 0 for v in spr_values):
            info["verdict"] = f"UNCOMPUTABLE (bad samples_per_record: {spr_values})"
            uncomputable.append(p.name)
            per_file[p.name] = info
            continue

        record_bytes = sum(spr_values) * 2  # int16 samples
        data_bytes = file_size - expected_header_bytes
        n_records_actual = data_bytes // record_bytes if record_bytes else None
        info.update({
            "record_bytes": record_bytes,
            "data_bytes": data_bytes,
            "n_records_actual": n_records_actual,
        })

        if not isinstance(n_records_claimed, int) or n_records_actual is None:
            info["verdict"] = "UNCOMPUTABLE (n_records unparseable)"
            uncomputable.append(p.name)
        elif n_records_actual == n_records_claimed:
            info["verdict"] = "OK"
            ok.append(p.name)
        elif n_records_actual < n_records_claimed:
            missing = n_records_claimed - n_records_actual
            info["verdict"] = f"TRUNCATED (header claims {missing} more records than disk holds)"
            truncated.append(p.name)
        else:
            extra = n_records_actual - n_records_claimed
            info["verdict"] = f"OVER-SIZED (disk has {extra} extra records' worth, possibly trailing junk)"
            oversized.append(p.name)
        per_file[p.name] = info

    issues: list[str] = []
    if not per_file:
        status = "fail"
        issues.append("No EDF files were provided")
    elif truncated:
        status = "fail"
        for name in truncated:
            issues.append(f"{name}: {per_file[name]['verdict']}")
    elif oversized or uncomputable:
        status = "warn"
        for name in oversized + uncomputable:
            issues.append(f"{name}: {per_file[name]['verdict']}")
    else:
        status = "pass"

    return {
        "check": "byte_geometry",
        "status": status,
        "n_files": len(per_file),
        "verdicts_by_file": {name: v["verdict"] for name, v in per_file.items()},
        "details_by_file": per_file,
        "truncated_files": sorted(truncated),
        "oversized_files": sorted(oversized),
        "uncomputable_files": sorted(uncomputable),
        "ok_files": sorted(ok),
        "issues": issues,
    }


_SIGNAL_UNIFORMITY_FIELDS = (
    "label", "samples_per_record", "phys_min", "phys_max",
    "dig_min", "dig_max", "phys_dim",
)

# Float precision for canonicalizing phys_min / phys_max. The EDF ASCII
# fields carry ~8 chars of precision; rounding to 6 decimals absorbs
# any floating-point round-trip noise (e.g. 0.9999999 vs 1.0000001)
# that would otherwise fragment signatures across files that are
# functionally identical.
_FLOAT_FIELDS = frozenset({"phys_min", "phys_max"})
_FLOAT_ROUND_DECIMALS = 6


def _canonical_field_value(field: str, value):
    """Normalize one signal-header field for signature comparison.

    Strings → stripped (guards against trailing padding from fixed-width
    ASCII slots). Floats → rounded to a stable precision (guards against
    parse round-trip noise). Everything else passes through."""
    if value is None:
        return None
    if field in _FLOAT_FIELDS:
        try:
            return round(float(value), _FLOAT_ROUND_DECIMALS)
        except (TypeError, ValueError):
            return value
    if isinstance(value, str):
        return value.strip()
    return value


def _signal_header_signature(sigs: list, *, ignore_annotation_channel: bool) -> tuple:
    """Deterministic hashable signature of a file's signal headers.

    Fields are canonicalized (stripped / rounded) so functionally
    identical headers produce identical signatures — otherwise minor
    representation drift across files (float precision, ASCII padding)
    would make every file appear to have a unique montage.
    """
    channels = []
    for s in sigs:
        label = str(s.get("label", "")).strip()
        if ignore_annotation_channel and label == EDF_ANNOTATION_LABEL:
            continue
        channels.append(tuple(
            _canonical_field_value(f, s.get(f))
            for f in _SIGNAL_UNIFORMITY_FIELDS
        ))
    return tuple(channels)


def _per_field_uniqueness(per_file_channels: dict[str, list[dict]],
                          *, ignore_annotation_channel: bool
                          ) -> dict[str, int]:
    """Diagnostic: for each signal-header field, how many distinct
    per-file sequences appear? A value of 1 means that field is uniform
    across the subject; values > 1 name the fields that fragment the
    signature. Uses the same canonicalization as the signature itself
    so what's reported matches what caused divergence."""
    result: dict[str, int] = {}
    for field in _SIGNAL_UNIFORMITY_FIELDS:
        sequences: set = set()
        for chans in per_file_channels.values():
            seq = tuple(
                _canonical_field_value(field, c.get(field))
                for c in chans
                if not (ignore_annotation_channel
                        and str(c.get("label", "")).strip() == EDF_ANNOTATION_LABEL)
            )
            sequences.add(seq)
        result[field] = len(sequences)
    return result


def check_signal_header_uniformity(edf_paths: Iterable[str | Path],
                                   *,
                                   ignore_annotation_channel: bool = True) -> dict:
    """Verify signal headers (labels, sample rates, calibration) are
    identical across all files of a subject.

    The pipeline warns during cleaning if headers diverge, but transferred
    files should be re-checked (a scrambled transfer could mix files from
    different subjects, or a partial re-run could leave inconsistent
    calibration in place). The ``EDF Annotations`` channel is ignored by
    default because the pipeline can add/remove it independently.
    """
    signature_to_files: dict[tuple, list[str]] = {}
    per_file_channels: dict[str, list[dict]] = {}
    for p in edf_paths:
        p = Path(p)
        header = read_main_header(str(p))
        n_signals = header.get("n_signals")
        if not isinstance(n_signals, int) or n_signals <= 0:
            sig = ("__unparseable__", p.name)
        else:
            sigs = read_signal_headers(str(p), n_signals)
            per_file_channels[p.name] = [
                {f: s.get(f) for f in _SIGNAL_UNIFORMITY_FIELDS} for s in sigs
            ]
            sig = _signal_header_signature(sigs,
                                           ignore_annotation_channel=ignore_annotation_channel)
        signature_to_files.setdefault(sig, []).append(p.name)

    field_variability = _per_field_uniqueness(
        per_file_channels, ignore_annotation_channel=ignore_annotation_channel)
    varying_fields = sorted(f for f, n in field_variability.items() if n > 1)

    issues: list[str] = []
    if not signature_to_files:
        status = "fail"
        issues.append("No EDF files were provided")
    elif len(signature_to_files) == 1:
        status = "pass"
    else:
        status = "fail"
        n_files = sum(len(v) for v in signature_to_files.values())
        issues.append(
            f"{len(signature_to_files)} distinct signal-header signatures "
            f"across {n_files} files — headers should be uniform within "
            f"a subject"
        )
        # Diagnostic: name the specific fields that fragment. If N==files
        # this is usually float-precision or trailing-padding noise;
        # varying_fields makes that obvious at a glance instead of
        # forcing the operator to diff N per-signature channel lists
        # themselves.
        if varying_fields:
            per_field_counts = {f: field_variability[f] for f in varying_fields}
            issues.append(
                f"  fragmentation is driven by fields with per-file "
                f"variation: {per_field_counts}"
            )
        # Cap the per-signature enumeration — showing 80 lines of file
        # lists is worse than useless. Full mapping is in edf_audit.json
        # under 'signatures'.
        max_sig_lines = 5
        for i, (sig, files) in enumerate(
                list(signature_to_files.items())[:max_sig_lines]):
            issues.append(f"  signature #{i + 1}: {len(files)} file(s), "
                          f"e.g. {sorted(files)[:3]}")
        remaining = len(signature_to_files) - max_sig_lines
        if remaining > 0:
            issues.append(
                f"  … and {remaining} more signature(s); see 'signatures' "
                f"in edf_audit.json for the full mapping"
            )

    # Serialize signatures for JSON: each signature becomes an id → files.
    signatures_out = {}
    for i, (sig, files) in enumerate(signature_to_files.items()):
        rep_file = files[0]
        signatures_out[f"signature_{i + 1}"] = {
            "files": sorted(files),
            "n_files": len(files),
            "channels": per_file_channels.get(rep_file, []),
        }

    return {
        "check": "signal_header_uniformity",
        "status": status,
        "n_files": sum(len(v) for v in signature_to_files.values()),
        "n_unique_signatures": len(signature_to_files),
        "signatures": signatures_out,
        "field_variability": field_variability,
        "varying_fields": varying_fields,
        "ignore_annotation_channel": ignore_annotation_channel,
        "issues": issues,
    }


def check_annotation_pairing(edf_paths: Iterable[str | Path]) -> dict:
    """Verify every recording has a paired ``*_annotations.edf`` stub
    (in-place mode) — or that no stubs exist at all (rewrite mode).

    The pipeline runs in one of two modes: rewrite (annotations embedded
    in the main EDF, no stub sibling) or in-place (annotations only in a
    sidecar ``<name>_annotations.edf``). The audit detects the mode
    from the file inventory and only enforces pairing when at least one
    stub is present.

    Status:
      - ``pass`` — inline mode (no stubs) OR every recording paired
      - ``fail`` — any orphan recording (missing its stub) or orphan
        stub (missing its recording)
    """
    recordings: dict[str, Path] = {}
    stubs: dict[str, Path] = {}
    for p in edf_paths:
        p = Path(p)
        if p.name.endswith(ANNOTATION_STUB_SUFFIX):
            base = p.name[:-len(ANNOTATION_STUB_SUFFIX)]  # foo_annotations.edf -> foo
            stubs[base] = p
        elif p.name.endswith(".edf"):
            recordings[p.name[:-len(".edf")]] = p

    paired: list[dict[str, str]] = []
    orphan_recordings: list[str] = []
    orphan_stubs: list[str] = []

    for base, rec_path in recordings.items():
        if base in stubs:
            paired.append({"recording": rec_path.name, "stub": stubs[base].name})
        else:
            orphan_recordings.append(rec_path.name)
    for base, stub_path in stubs.items():
        if base not in recordings:
            orphan_stubs.append(stub_path.name)

    mode = "inline" if not stubs else "stub_pair"
    issues: list[str] = []
    if not recordings and not stubs:
        status = "fail"
        issues.append("No EDF files were provided")
    elif mode == "inline":
        status = "pass"
    elif orphan_recordings or orphan_stubs:
        status = "fail"
        for name in orphan_recordings:
            issues.append(f"{name}: recording has no paired '_annotations.edf' stub")
        for name in orphan_stubs:
            issues.append(f"{name}: annotation stub has no paired recording")
    else:
        status = "pass"

    return {
        "check": "annotation_pairing",
        "status": status,
        "mode": mode,
        "n_recordings": len(recordings),
        "n_stubs": len(stubs),
        "paired": paired,
        "orphan_recordings": sorted(orphan_recordings),
        "orphan_stubs": sorted(orphan_stubs),
        "issues": issues,
    }


def _extract_subject_code(patient_id: str) -> str:
    """Return the first whitespace-separated token of an EDF+ patient_id.

    A cleaned EDF+ patient_id looks like ``"R1770J X 01-JAN-1900 unknown
    unknown"`` (code, sex, birthdate, first, last). Only the first
    subfield holds the subject code.
    """
    return patient_id.strip().split()[0] if patient_id.strip() else ""


def check_subject_code_consistency(edf_paths: Iterable[str | Path]) -> dict:
    """Extract the subject code from each EDF's ``patient_id`` field and
    verify a single valid code appears across every file.

    ``patient_id`` is read from bytes 8-88 of the main header directly
    (no pyedflib), so broken files are still audited. The subject code
    is the first whitespace-separated token (EDF+ subfield layout).
    ``status``:
      - ``pass``  — exactly one subject code, matches ``SUBJECT_CODE_PATTERN``.
      - ``warn``  — one subject code but it doesn't match the pattern.
      - ``fail``  — multiple distinct subject codes across files.
    """
    per_file: dict[str, str] = {}
    codes_by_file: dict[str, str] = {}
    for p in edf_paths:
        p = Path(p)
        header = read_main_header(str(p))
        pid = header.get("patient_id", "")
        pid = pid if isinstance(pid, str) else str(pid)
        per_file[p.name] = pid
        codes_by_file[p.name] = _extract_subject_code(pid)

    unique_codes = sorted(set(codes_by_file.values()))
    matching = [c for c in unique_codes if c and is_valid_subject_code(c, raise_error=False)]
    non_matching = [c for c in unique_codes if c not in matching]

    issues: list[str] = []
    if len(per_file) == 0:
        status = "fail"
        issues.append("No EDF files were provided")
    elif len(unique_codes) > 1:
        status = "fail"
        issues.append(f"Multiple distinct subject codes across files: {unique_codes}")
    elif not matching:
        status = "warn"
        issues.append(
            f"Subject code {unique_codes[0]!r} does not match "
            f"SUBJECT_CODE_PATTERN ({SUBJECT_CODE_PATTERN})"
        )
    else:
        status = "pass"

    return {
        "check": "subject_code_consistency",
        "status": status,
        "subject_code": matching[0] if len(matching) == 1 and len(unique_codes) == 1 else None,
        "n_files": len(per_file),
        "unique_subject_codes": unique_codes,
        "non_matching_subject_codes": non_matching,
        "subject_codes_by_file": codes_by_file,
        "patient_ids_by_file": per_file,
        "pattern": SUBJECT_CODE_PATTERN,
        "issues": issues,
    }
