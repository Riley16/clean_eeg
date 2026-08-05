import random
import re
import os
import shutil
import traceback
import numpy as np
import pyedflib
from copy import deepcopy
from typing import Union
from datetime import datetime, timedelta
from tqdm import tqdm
from clean_eeg.anonymize import redact_subject_name, PersonalName, SubjectNameRedactor
from clean_eeg.annotation_boilerplate import load_whitelist
from clean_eeg.deidentify_manifest import (
    MANIFEST_FILENAME,
    ReviewEvent,
    build_manifest,
    manifest_exists,
    read_manifest,
    write_manifest,
)
from clean_eeg.load_eeg import load_edf, write_edf_pyedflib
from clean_eeg.log import logged_input, setup_logger, get_logger, close_logger
from clean_eeg.modify_edf_inplace import (
    update_edf_header_inplace,
    clear_edf_annotations_inplace,
    create_annotations_only_edf,
    validate_header_roundtrip,
)
from clean_eeg.paths import ANNOTATION_BOILERPLATE_WHITELIST_PATH

BASE_START_DATE = datetime(1985, 1, 1)
DEFAULT_REDACT_HEADER_KEYS = ['patientname', 'sex', 'gender', 'patient_additional']
REDACT_REPLACEMENT = 'X'  # match pyedflib default for missing field
MAX_RECORDING_GAP_SECONDS = 60
MIN_RECORDING_GAP_ERROR_SECONDS = -2  # allow small overlaps in files
MIN_RECORDING_GAP_WARNING_SECONDS = -0.25
SITE_CODE_TO_INCOMING_FOLDER = {'S': 'UTHSCSA',
                                'A': 'CUDA',
                                'H': 'harvard',
                                'J': 'TJ'}

# Consecutive load failures after which _load_edf_metadata aborts the
# subject rather than dutifully churning through more files. When N
# neighboring files in a row fail to load, the input directory is
# almost certainly broken in some systematic way (wrong export format,
# permissions issue, truncated USB dump) and continuing wastes time
# and floods stdout with per-file header dumps. Override with
# --force_load_all on the CLI.
MAX_CONSECUTIVE_LOAD_FAILURES = 5


class ConsecutiveLoadFailureLimit(RuntimeError):
    """Raised by _load_edf_metadata when the consecutive-failure cap
    fires. Caught in clean_subject_edf_files to skip the de-id loop
    AND skip the transfer prompt — an operator whose files won't load
    should never be shown a transfer command."""


def deidentify_edf(edf_data, subject_name, subject_code, earliest_recording_start_time,
                   redactor: Union[SubjectNameRedactor, None] = None,
                   review_events: Union[list, None] = None,
                   source_file: Union[str, None] = None,
                   wipe_annotations: bool = False):
    # remove protected health information (PHI) from EEG
    # accepts EDF data in 'pyedflib' format

    # de-identification operations:
    # 1) rename subject to subject code and remove meta-data fields for gender, birthdate, patient hospital code
    # 2) replace recording start time with time relative to the earliest recording start time
    # 3) remove any recording annotations containing regex patterns indicating PHI (name, gender)
    # 4) save the modified EDF file with a new name in the format SUBJECT_CODE__RELATIVE.START.DATE_RELATIVE:START:TIME.edf
    #        RELATIVE.START.DATE_RELATIVE:START:TIME corresponds to YEAR.MONTH.DAY__HOUR:MINUTE:SECOND relative to the earliest recording start time
    #        relative times are offset by the EDF standard clipping date of 1985-01-01

    # Build a fresh top-level dict. Each helper already constructs new
    # objects for the fields it modifies (deidentify_edf_header deepcopies
    # its input dict; deidentify_edf_annotations builds fresh arrays), so
    # an outer deepcopy would double the memory of the signal arrays for
    # no additional isolation. Signals are not mutated by de-identification,
    # so we share the reference.
    clean_signal_headers = [
        deidentify_edf_header(sh,
                              subject_name=subject_name,
                              subject_code=subject_code,
                              earliest_recording_start_time=None,  # signal headers do not have a start time
                              redact_keys=list(),  # check all
                              redactor=redactor)
        for sh in edf_data['signal_headers']
    ]
    if wipe_annotations:
        # Skip Presidio entirely — the write path will call
        # clear_edf_annotations_inplace on the output, which zeros every
        # non-timekeeping annotation TAL byte-for-byte. Handing pyedflib
        # an empty annotation tuple in the rewrite path keeps its
        # annotation channel wired up correctly (only timekeeping TALs
        # get written, which is what we want).
        clean_annotations = (np.array([]), np.array([]), np.array([]))
    else:
        clean_annotations = deidentify_edf_annotations(
            edf_data['annotations'],
            subject_name=subject_name,
            redactor=redactor,
            review_events=review_events,
            source_file=source_file,
        )
    return {
        'header': deidentify_edf_header(edf_data['header'],
                                        subject_name=subject_name,
                                        subject_code=subject_code,
                                        earliest_recording_start_time=earliest_recording_start_time,
                                        redactor=redactor),
        'signal_headers': clean_signal_headers,
        'annotations': clean_annotations,
        'signals': edf_data['signals'],
    }


def deidentify_edf_header(header: dict,
                          subject_code: str,
                          subject_name: PersonalName,
                          earliest_recording_start_time: Union[datetime,None]=None,
                          redact_keys: list[str]=DEFAULT_REDACT_HEADER_KEYS,
                          redactor: Union[SubjectNameRedactor, None] = None):
    header = deepcopy(header)
    is_signal_header = 'label' in header
    if earliest_recording_start_time is None:
        assert 'startdate' not in header
    else:
        header['startdate'] = deidentify_start_date_time(header['startdate'],
                                                         earliest_recording_start_time)
    if not is_signal_header:
        # Overwrite the entire birthdate field with a standard placeholder.
        # The whole string is replaced, so any PHI that was there is gone —
        # no need to run the redactor on it, and doing so would risk
        # mangling "01 jan 1900" into e.g. "01 X 1900" when the subject's
        # name shares a substring with the month abbreviation (pyedflib
        # writes this field via strptime("%d %b %Y") and would crash).
        header['birthdate'] = '01 jan 1900'
    for key in redact_keys:
        header[key] = REDACT_REPLACEMENT
    header['patientcode'] = subject_code
    # Check for patient name, gendered pronouns in all other string fields.
    # birthdate is skipped — we just overwrote it entirely above.
    for key, val in header.items():
        if key in redact_keys or key == 'birthdate':
            continue
        if isinstance(val, str):
            header[key] = redact_string(val,
                                        field_name=key,
                                        subject_name=subject_name,
                                        redactor=redactor)
        elif isinstance(val, (int, float, datetime)):
            pass
        else:
            raise ValueError(f'Unknown type in header field {key}: type: {type(val)}; value: {val}')
    return header


def deidentify_edf_annotations(annotations: tuple[np.ndarray], subject_name: PersonalName,
                                redactor: Union[SubjectNameRedactor, None] = None,
                                review_events: Union[list, None] = None,
                                source_file: Union[str, None] = None):
    clean_start_times = list()
    clean_durations = list()
    clean_descriptions = list()
    for (start_time, duration, text) in zip(*annotations):
        assert isinstance(text, str)
        redacted_text = redact_string(str(text),
                                      field_name='annotation',
                                      subject_name=subject_name,
                                      alert=True,
                                      redactor=redactor,
                                      review_events=review_events,
                                      source_file=source_file)
        clean_start_times.append(start_time)
        clean_durations.append(duration)
        clean_descriptions.append(redacted_text)
        
    clean_annotations = (np.array(clean_start_times),
                         np.array(clean_durations), 
                         np.array(clean_descriptions))
    return clean_annotations


SUBJECT_CODE_PATTERN = r'^R1\d{3}[ACDEFHJMNPST]$'


def is_valid_subject_code(subject_code,
                          pattern=SUBJECT_CODE_PATTERN,
                          raise_error=True):
    """
    Validate the format of <subject_code> matches regex <pattern>.
    Default pattern matches DARPA RAM subject codes like R1755A, R1234C, etc. in which 
    the last three digits give the subject number and the letter gives the hospital code.
    Note: this default pattern does not cover subject-montage codes (e.g., R1755A_1)
    """
    if len(subject_code.split('_')) > 1:
        raise NotImplementedError("Subject-montage codes (e.g., R1755A_1) not implemented yet.")
    if raise_error and not re.match(pattern, subject_code):
        raise ValueError(f'Invalid subject code: "{subject_code}". '
                         f"Expected regex pattern: {pattern}")
    return re.match(pattern, subject_code) is not None


def confirm_wipe_annotations(subject_code: str,
                             approved: set[str],
                             input_fn=None) -> bool:
    """Interactive per-subject confirmation for --wipe-annotations.

    The operator must type the exact ``subject_code`` (case-sensitive)
    to authorize deletion of all non-timekeeping annotations for that
    subject. Rejects "yes", "y", empty input, or any wrong code.

    ``approved`` is the set from ``--approve-confirmations``: if it
    contains ``'wipe-annotations'``, the prompt is skipped and a loud
    banner is printed instead. Each destructive confirmation type must
    be listed explicitly — this is intentionally NOT a global bypass.

    ``input_fn`` is a hook for testing (default: ``logged_input`` from
    ``clean_eeg.log``, which wraps ``input()`` and also writes the
    typed response to log.out so ``--wipe-annotations`` invocations
    are auditable).
    """
    if "wipe-annotations" in approved:
        print(f"[!] --wipe-annotations auto-approved for {subject_code} "
              "via --approve-confirmations. All non-timekeeping annotations "
              "will be permanently deleted from the output EDFs.")
        return True

    if input_fn is None:
        input_fn = logged_input
    print(f"[!] --wipe-annotations will PERMANENTLY DELETE all non-timekeeping "
          f"annotations from the output EDFs for subject {subject_code}.")
    typed = input_fn(f"Type the subject code ({subject_code}) to CONFIRM: ")
    if typed == subject_code:
        print(f"[wipe] Confirmed. Proceeding with wipe of annotations for {subject_code}.")
        return True
    print(f"[wipe] Confirmation input {typed!r} does not match subject code "
          f"{subject_code!r}. Aborting — no files were modified.")
    return False


def deidentify_start_date_time(recording_start_time, earliest_recording_start_time):
    shifted_time = recording_start_time - earliest_recording_start_time + BASE_START_DATE
    return shifted_time


# Matches empty, all-whitespace, pure-numeric (incl. sign and decimal), or
# EDF+ timekeeping-TAL-shaped strings like "+0.086" / "-12.5" / "+1234".
# These cannot contain PHI — skip the Presidio pass entirely.
_NON_PHI_TEXT_RE = re.compile(r"^\s*[+-]?\d*\.?\d*\s*$")


def redact_string(text: str, field_name: str, subject_name: PersonalName,
                  alert: bool = False,
                  redactor: Union[SubjectNameRedactor, None] = None,
                  review_events: Union[list, None] = None,
                  source_file: Union[str, None] = None) -> str:
    if _NON_PHI_TEXT_RE.match(text):
        # Empty, numeric, or timekeeping-shaped — cannot hold PHI; skip Presidio.
        return text
    redacted = redact_subject_name(text, subject_full_name=subject_name, redactor=redactor)
    redacted = remove_gendered_pronouns(redacted)
    if alert and text != redacted:
        # Collect the *redacted* value (not the raw one) so it stays
        # PHI-free while still showing what was flagged and what
        # survived. Appended to review_events for the end-of-run
        # 'Human review needed' block; boilerplate suppression happens
        # at print time (all events are recorded in the manifest).
        if review_events is not None:
            review_events.append(ReviewEvent(
                kind="annotation_redaction",
                file=source_file or "",
                details={
                    "field": field_name,
                    "redacted_value": redacted,
                },
            ))
    return redacted


_GENDERED_PRONOUNS = [
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
]
REDACT_PRONOUN_REPLACEMENT = "X"

# \b-boundaries ensure we don't hit substrings (e.g., "her" in "other").
PRONOUN_RE = re.compile(r"\b(" + "|".join(map(re.escape, _GENDERED_PRONOUNS)) + r")\b",
                           flags=re.IGNORECASE | re.UNICODE)

def remove_gendered_pronouns(text: str, replacement: str = REDACT_PRONOUN_REPLACEMENT) -> str:
    """
    Remove (or replace) gendered pronouns. Default behavior is deletion.
    Pass replacement='[REDACTED-PRONOUN]' if you prefer explicit redaction.
    """
    if replacement == "":
        return PRONOUN_RE.sub("", text)
    else:
        return PRONOUN_RE.sub(replacement, text)


def clean_subject_edf_files(
    input_path: str,
    output_path: str,
    subject_code: str,
    subject_name: Union[PersonalName, None] = None,
    load_method: str = "pyedflib",
    raise_errors: bool = False,
    inplace: bool = False,
    verbosity: int = 1,
    skip_header_name_check: bool = False,
    benchmark: bool = False,
    read_digital: bool = True,
    skip_audit: bool = False,
    force: bool = False,
    force_load_all: bool = False,
    auto_transfer_response: Union[str, None] = None,
    wipe_annotations: bool = False,
):
    from clean_eeg.benchmark import BenchmarkCollector
    bench = BenchmarkCollector(enabled=benchmark)

    if inplace:
        assert input_path == output_path, "For inplace cleaning, input_path must equal output_path."

    # Completion-marker fast path: a prior successful run wrote
    # deidentify.json to output_path. Presence == de-id done. Offer to
    # skip straight to transfer unless --force says re-run from scratch.
    if not force and manifest_exists(output_path):
        _maybe_skip_to_transfer(output_path,
                                auto_response=auto_transfer_response)
        return

    try:
        EDF_meta_data = _load_edf_metadata(input_path=input_path,
                                           verbosity=verbosity,
                                           load_method=load_method,
                                           raise_errors=raise_errors,
                                           force_load_all=force_load_all,
                                           bench=bench)
    except ConsecutiveLoadFailureLimit as e:
        # Do NOT write a manifest and do NOT offer transfer — an
        # operator whose files won't load must not be handed an upload
        # command. Print the message so the tee captures it in log.out.
        print(f"\n{e}\n")
        return

    if not EDF_meta_data:
        raise RuntimeError(
            f"No EDF files were successfully loaded from {input_path}. "
            "This can happen if the directory contains no .edf files, or if "
            "all .edf files failed to parse (see errors above — e.g. filesize "
            "mismatches from Nihon Kohden exports that don't strictly follow "
            "the EDF standard). Aborting."
        )

    _validate_EDF_meta_data(EDF_meta_data, subject_name=subject_name, verbosity=verbosity,
                            skip_header_name_check=skip_header_name_check)
    min_start_time = _get_start_time_earliest_recording(EDF_meta_data, verbosity=verbosity)

    # Select files for signal integrity audit. When skip_audit is True,
    # the set stays empty so no files are audited; inplace runs then also
    # skip the signal preload for every file (see need_signals below),
    # which is the bulk of the I/O time on multi-GB NK recordings.
    # Otherwise every file in the subject is audited — the mmap-based
    # streamed comparison plus a single-channel pyedflib cross-check are
    # both fast enough that exhaustive auditing is the right default.
    all_filenames = list(EDF_meta_data.keys())
    if skip_audit:
        audit_filenames: set = set()
    else:
        audit_filenames = set(all_filenames)

    # Build Presidio once per subject and reuse across all redact_string calls.
    # This amortizes the spaCy-model + recognizer-registry construction cost.
    with bench.step("build_presidio_redactor"):
        redactor = SubjectNameRedactor(subject_name) if subject_name is not None else None

    # Review events accumulated across the whole subject — printed once
    # in the end-of-run 'Human review needed' block and persisted in
    # deidentify.json for post-hoc audit.
    review_events: list[ReviewEvent] = []
    output_edf_paths: list = []  # accumulated for the manifest's hash step

    # de-identify EDF files and save out
    print("Cleaning EDF files... Saving to output path:", output_path)
    # Quarantine subdir for partial outputs from any file that fails
    # mid-pipeline. Only created on demand (the directory must not be
    # left empty in the output for clean runs). The end-of-run summary
    # tells operators NOT to send anything in this subdir.
    quarantine_dir = os.path.join(output_path, "quarantine")
    failed_files: list[tuple[str, str, list]] = []  # (filename, error, moved_paths)
    progress = tqdm(EDF_meta_data.items())
    n_audited = 0
    for filename, _ in progress:
        progress.set_postfix(current=filename[:24],
                             redactions=len(review_events),
                             quarantined=len(failed_files))
        # Track output artifacts created for this file so we can move
        # them to quarantine if anything fails mid-pipeline.
        output_artifacts: list = []
        try:
            input_file_path = os.path.join(input_path, filename)
            # In inplace mode, signals are never rewritten — the pipeline only
            # moves the file and patches headers/annotations in place. Signals
            # are therefore only needed for the audit files. For non-audit
            # files in inplace mode we skip preload entirely (load_edf returns
            # signals=None in that case). Copy mode always needs signals.
            need_signals = (not inplace) or (filename in audit_filenames)
            step_label = ("load_preload_signals" if need_signals
                          else "load_metadata_only")
            with bench.step(step_label, file=filename):
                # use_mmap=True: on digital preloads, use the mmap-based
                # record-deinterleaver instead of pyedflib's per-channel
                # readSignal loop. Orders of magnitude faster on multi-GB
                # NK files. Falls back to pyedflib automatically on any
                # exception inside load_edf, so correctness is preserved
                # even when the mmap path has a bug.
                edf = load_edf(input_file_path, load_method=load_method,
                               preload=need_signals, read_digital=read_digital,
                               use_mmap=True)
            assert isinstance(edf, dict)

            # Hold on to a reference to the original signals for the audit.
            # deidentify_edf does not mutate signals (and no longer deep-copies
            # them), so the same array objects remain valid across the call.
            orig_signals = edf['signals'] if filename in audit_filenames else None

            with bench.step("deidentify_edf", file=filename):
                edf = deidentify_edf(
                    edf_data=edf,
                    subject_name=subject_name,
                    subject_code=subject_code,
                    earliest_recording_start_time=min_start_time,
                    redactor=redactor,
                    review_events=review_events,
                    source_file=filename,
                    wipe_annotations=wipe_annotations,
                )
            with bench.step("validate_header_roundtrip", file=filename):
                truncation_warnings = validate_header_roundtrip(
                    edf['header'], edf['signal_headers'])
            for warning in truncation_warnings:
                # Header-field truncation is PHI-adjacent: patient_id
                # packs patientname + patientcode + birthdate; a
                # truncation there could leave partial name bytes in
                # the file. Surface in the review block AND log
                # immediately for visibility.
                print(f"WARNING: {warning}")
                review_events.append(ReviewEvent(
                    kind="header_truncation",
                    file=filename,
                    details={"message": str(warning)},
                ))

            clean_start_time = edf['header']['startdate']
            filename_no_ext = os.path.splitext(filename)[0]
            subject_val = subject_code
            # Year deliberately omitted: it would always be 1985 (the
            # BASE_START_DATE used to anchor de-identified relative
            # timestamps) and confuses operators who read the filename.
            # Month/day still encode the relative offset between the
            # subject's recordings within a session.
            clean_filename = f"{filename_no_ext}_{subject_val}_{clean_start_time.strftime('%m.%d__%H.%M.%S')}.edf"
            clean_full_path = os.path.join(output_path, clean_filename)
            clean_annotations_path = str(clean_full_path).replace('.edf', '_annotations.edf')
            if inplace:
                with bench.step("write_inplace", file=filename):
                    shutil.move(input_file_path, clean_full_path)
                    output_artifacts.append(clean_full_path)
                    if not wipe_annotations:
                        # Sidecar stub — skipped under --wipe-annotations because
                        # the annotations we'd write into it are the ones the
                        # operator asked us to delete.
                        create_annotations_only_edf(clean_annotations_path,
                                                    header=edf['header'],
                                                    annotations=edf['annotations'])
                        output_artifacts.append(clean_annotations_path)
                    update_edf_header_inplace(clean_full_path,
                                              header_updates=edf['header'],
                                              signal_header_updates=edf['signal_headers'])
                    clear_edf_annotations_inplace(clean_full_path)
            else:
                with bench.step("write_edf_pyedflib", file=filename):
                    write_edf_pyedflib(edf, clean_full_path, digital=read_digital)
                    output_artifacts.append(clean_full_path)
                    if wipe_annotations:
                        # Same primitive as the in-place branch so both modes
                        # converge on byte-identical annotation-channel state
                        # (timekeeping TALs preserved, all event TAL bytes zeroed).
                        clear_edf_annotations_inplace(clean_full_path)
            if wipe_annotations:
                print(f"[wipe] {filename}: wiped, validated "
                      "(0 non-timekeeping annotations remain)")
            # Per-file success is reflected in the tqdm postfix
            # (redactions/quarantined counters) — dropping the old
            # scrolling 'Cleaned EDF file at:' line keeps the terminal
            # legible on multi-file subjects.

            # Audit signal integrity immediately after write
            if filename in audit_filenames:
                with bench.step("audit_signal_integrity", file=filename):
                    _audit_signal_integrity(orig_signals, clean_full_path, filename,
                                            inplace=inplace, digital=read_digital)
                n_audited += 1

            # Track output artifacts for the manifest's hash step —
            # AFTER the audit so a failed audit's quarantined artifacts
            # never end up in the hash manifest (would crash on
            # FileNotFoundError at hash time).
            for artifact in output_artifacts:
                output_edf_paths.append(artifact)
        except Exception as e:
            if raise_errors:
                raise e
            # Move any partial output artifacts out of the standard output
            # directory so operators using `scp output/*.edf` will not pick
            # them up. The standard `*.edf` glob is non-recursive, so a
            # subdir-quarantine works without further action.
            moved = _quarantine_partial_outputs(output_artifacts, quarantine_dir)
            failed_files.append((filename, f"{type(e).__name__}: {e}", moved))
            err_msg_lines = [
                f"\nERROR: Failed to de-identify EDF file {filename}:",
                "",
                str(e),
                "",
                "Stack trace (for the data team):",
                traceback.format_exc().rstrip(),
                "",
            ]
            if moved:
                err_msg_lines.extend([
                    "Partially-processed output files for this EDF have been "
                    "moved to the 'quarantine/' subdirectory:",
                    *[f"  {p}" for p in moved],
                    "",
                    "DO NOT send these quarantined files to the data "
                    "management team. They may not be fully de-identified.",
                    "",
                ])
            err_msg_lines.append("Skipping this file and continuing...")
            print("\n".join(err_msg_lines))
            # Dump the header for the data team. Try the original input
            # first; if the inplace-write step already moved it, fall
            # back to whichever quarantined path now holds the file.
            _dump_edf_header_for_diagnosis(input_file_path, *moved)

    print("Done cleaning EDF files. Saved to output path:", output_path)
    if benchmark:
        print(bench.report())

    site_code = subject_code[-1]
    site_incoming_folder = SITE_CODE_TO_INCOMING_FOLDER.get(site_code, 'UNKNOWN_SITE')

    n_quarantined_files = sum(len(moved) for _, _, moved in failed_files)
    if failed_files:
        any_quarantined = any(moved for _, _, moved in failed_files)
        print(
            f"\nWARNING: {len(failed_files)} EDF file(s) were not successfully "
            f"de-identified:"
        )
        for fname, err, moved in failed_files:
            print(f"  - {fname}: {err}")
            for p in moved:
                print(f"      → moved to quarantine: {p}")
        print()
        if any_quarantined:
            print(
                "Files in the 'quarantine/' subdirectory have NOT been "
                "fully de-identified and MUST NOT be sent to the data "
                "management team. The transfer step will refuse to run "
                "until quarantine/ is empty — investigate the failures "
                "above, DO NOT include the quarantine/ subdirectory in "
                "any manual copy, and re-run transfer-subject-eeg after "
                "resolving the issues."
            )
            print()
        print(
            "Please send log.out (in the EDF directory) to the data "
            "management team so the failures above can be investigated.\n"
        )

    _print_review_block(review_events)

    manifest = build_manifest(
        subject_code=subject_code,
        site_code=site_code,
        site_incoming_folder=site_incoming_folder,
        input_path=input_path,
        output_path=output_path,
        inplace=inplace,
        output_edf_paths=output_edf_paths,
        n_files_deidentified=len(output_edf_paths),
        n_files_failed=len(failed_files),
        n_files_quarantined=n_quarantined_files,
        review_events=review_events,
    )
    write_manifest(output_path, manifest)

    # Refuse to prompt for transfer if any file failed or was
    # quarantined — the operator needs to resolve those first.
    if failed_files:
        print(
            "Refusing to offer transfer while failures/quarantine remain. "
            "Investigate, then re-run `transfer-subject-eeg <output_dir>` "
            "once resolved.\n"
        )
        return

    _prompt_ready_to_transfer(output_path, auto_response=auto_transfer_response)


def _maybe_skip_to_transfer(output_path: str,
                            auto_response: Union[str, None] = None) -> None:
    """Called when re-invoking the pipeline on an already-completed
    output directory. Confirms with the operator, then hands off to
    the transfer tool. Called with a fresh (non-force) invocation
    only — --force short-circuits this path in ``clean_subject_edf_files``.
    """
    try:
        manifest = read_manifest(output_path)
    except Exception as e:
        print(f"WARNING: could not read existing deidentify.json: {e}")
        print("Falling back to a fresh de-identification run. Pass "
              "--force to skip this check next time.")
        return
    assert manifest is not None
    print(
        f"\nDe-identification manifest already present in {output_path}\n"
        f"  generated_at: {manifest.get('generated_at')}\n"
        f"  clean_eeg version: {manifest.get('clean_eeg_version')}\n"
        f"  n_files: {manifest.get('n_files_deidentified')}\n"
    )
    resp = (auto_response
            if auto_response is not None
            else logged_input(
                "De-identification already completed for this directory. "
                "Skip to transfer? [Y/n]: ")).strip().lower()
    if resp in ("", "y", "yes"):
        _invoke_transfer(output_path)
    else:
        print(
            "Aborting. To re-run de-identification from scratch, pass "
            "--force (this will overwrite deidentify.json). To transfer "
            "without re-running de-id, run `transfer-subject-eeg "
            f"{output_path}` directly.")


def _print_review_block(events: list) -> None:
    """Emit the end-of-run 'Human review needed' block. Boilerplate-matched
    annotations are suppressed here (they still live in the manifest)
    to avoid drowning the review in site-specific noise."""
    whitelist = load_whitelist(ANNOTATION_BOILERPLATE_WHITELIST_PATH)
    # Filter annotation redactions through the boilerplate whitelist so
    # recurring site-specific patterns don't flood the block. Header
    # truncations are always surfaced — they're rare and always relevant.
    to_show: list = []
    for e in events:
        if e.kind == "annotation_redaction":
            text = e.details.get("redacted_value", "")
            site_code = None  # boilerplate is applied broadly for now
            if not whitelist.matches(text, site_code=site_code):
                to_show.append(e)
        else:
            to_show.append(e)

    if not to_show:
        return
    print("\n=== Human review needed ===")
    ann = [e for e in to_show if e.kind == "annotation_redaction"]
    trunc = [e for e in to_show if e.kind == "header_truncation"]
    if ann:
        print(f"  {len(ann)} annotation(s) contained PHI-adjacent text "
              f"(redacted values shown; boilerplate suppressed):")
        for e in ann:
            print(f"    {e.file}: {e.details.get('redacted_value')!r}")
    if trunc:
        print(f"  {len(trunc)} header field(s) were truncated on write "
              f"— verify the affected fields did not contain PHI:")
        for e in trunc:
            print(f"    {e.file}: {e.details.get('message')}")
    print("===========================\n")


def _prompt_ready_to_transfer(output_path: str,
                              auto_response: Union[str, None] = None) -> None:
    """End-of-run prompt for the transfer step. Non-interactive via
    ``auto_response``; empty response defaults to 'no' so a truncated
    stdin never accidentally triggers an upload."""
    resp = (auto_response
            if auto_response is not None
            else logged_input(
                "Ready to transfer de-identified files to the CML server? "
                "[y/N]: ")).strip().lower()
    if resp in ("y", "yes"):
        _invoke_transfer(output_path)
    else:
        print(
            f"Transfer skipped. Run `transfer-subject-eeg {output_path}` "
            "at any time to upload."
        )


def _invoke_transfer(output_path: str) -> None:
    """Thin wrapper so tests can monkeypatch a single seam. Errors are
    caught here so a transfer failure does not tear down the log file
    close-out in ``__main__``."""
    from clean_eeg.transfer import transfer_subject
    try:
        transfer_subject(output_path)
    except RuntimeError as e:
        print(f"\nTransfer failed: {e}")
        print(f"Re-run `transfer-subject-eeg {output_path}` after "
              "resolving the issue above.")


QUARANTINE_SUFFIX = ".QUARANTINED-DO-NOT-USE"


def _dump_edf_header_for_diagnosis(*candidate_paths: str) -> None:
    """Best-effort: dump the EDF header of the first candidate path that
    exists on disk. Output goes to stdout so the live tee captures it
    into log.out. Always passes ``redact_phi=True`` because log.out is
    typically shared with the data team — the four PHI-bearing
    main-header fields (patient_id, recording_id, startdate, starttime)
    are masked. The numeric/structural fields that the data team
    actually needs to triage parse failures are preserved.

    Swallows any exception raised by the dump itself — it must NEVER
    mask the original error that triggered the diagnostic call."""
    from clean_eeg.print_edf_header import print_header
    for p in candidate_paths:
        if not p or not os.path.exists(p):
            continue
        try:
            print(f"\nEDF header dump (for the data team) — {p}:")
            print_header(p, redact_phi=True)
        except Exception as dump_err:
            print(
                f"  (header dump failed: "
                f"{type(dump_err).__name__}: {dump_err})"
            )
        return


def _quarantine_partial_outputs(artifact_paths: list, quarantine_dir: str) -> list:
    """Move any existing output artifacts out of the standard output
    directory and into a ``quarantine/`` subdirectory, with a renamed
    extension that does not match the standard ``*.edf`` glob.

    Defense-in-depth against accidental upload of partially-processed
    files:
    1. Files live in a subdirectory, so non-recursive glob copies
       (``scp output/*.edf`` or ``rsync --exclude='quarantine/'``) skip
       them automatically.
    2. The trailing extension is renamed from ``.edf`` to
       ``.edf.QUARANTINED-DO-NOT-USE``. Even if an operator runs a
       fully recursive transfer (``scp -r``, ``rsync`` without an
       ``--exclude``), any subsequent ``*.edf`` glob — server-side
       or client-side — will not match these files, and the data
       team can identify mis-uploaded files at a glance.

    Returns the list of new paths (in quarantine) the artifacts were
    moved to. Empty if none of the listed paths existed on disk.
    """
    moved: list = []
    if not artifact_paths:
        return moved
    for src in artifact_paths:
        if not src or not os.path.exists(src):
            continue
        os.makedirs(quarantine_dir, exist_ok=True)
        dest_name = os.path.basename(src) + QUARANTINE_SUFFIX
        dest = os.path.join(quarantine_dir, dest_name)
        # If a prior failure already quarantined a file with the same
        # name, append a counter to avoid clobbering its evidence.
        counter = 1
        base_dest = dest
        while os.path.exists(dest):
            dest = f"{base_dest}.{counter}"
            counter += 1
        shutil.move(src, dest)
        moved.append(dest)
    return moved


def _audit_signal_integrity(orig_signals: list, clean_file_path: str, filename: str,
                            inplace: bool = False, digital: bool = False):
    """Spot-check that signal data in the output file matches the original.

    For inplace mode, signals must be bit-identical since only headers are modified.
    For rewrite mode, pyedflib's digital/physical conversion introduces floating-point
    differences, so the audit is skipped (this is a known pyedflib limitation and the
    reason the in-place approach was developed).

    ``digital`` must match the mode used to read ``orig_signals``; when True, the
    clean file is also read in digital mode so the bit-comparison is meaningful.

    Memory-efficient: streams the clean file signal-by-signal via mmap and
    compares each to the corresponding ``orig_signals[i]`` before the next
    signal is read. Peak RAM stays at ``sizeof(orig_signals) + one_channel``
    instead of ``sizeof(orig_signals) + sizeof(clean_signals)``. For a 3.8
    GB file, that's ~3.82 GB peak instead of ~7.6 GB.
    """
    if not inplace:
        return
    import mmap as _mmap

    # --- parse on-disk geometry from the clean file bytes ---
    with open(clean_file_path, "rb") as f:
        main = f.read(256)
        n_signals_on_disk = int(main[252:256].decode().strip())
        n_records = int(main[236:244].decode().strip())
        sig_header = f.read(256 * n_signals_on_disk)

    labels = []
    samples_per_record = []
    for i in range(n_signals_on_disk):
        lab_b = sig_header[0 * n_signals_on_disk + i * 16:
                           0 * n_signals_on_disk + (i + 1) * 16]
        spr_b = sig_header[216 * n_signals_on_disk + i * 8:
                           216 * n_signals_on_disk + (i + 1) * 8]
        labels.append(lab_b.decode("ascii", errors="replace").rstrip())
        samples_per_record.append(int(spr_b.decode("ascii").strip()))

    # Map each public (non-annotation) signal index to its on-disk position
    # so we stream clean signals in the same order pyedflib would have.
    data_signal_disk_indices = [
        i for i, lab in enumerate(labels)
        if lab.strip().lower() != "edf annotations"
    ]
    if len(orig_signals) != len(data_signal_disk_indices):
        raise RuntimeError(
            f"AUDIT FAILURE for {filename}: the audit cross-checks that "
            "the de-identified file still contains exactly the same "
            "signal data as the original. The number of (non-annotation) "
            f"signal channels does not match — the file was loaded with "
            f"{len(orig_signals)} channels, but the de-identified file "
            f"on disk now reports {len(data_signal_disk_indices)} "
            "channels. Do NOT use this output file. Stop the run, save "
            "log.out, and send it to the data management team for "
            "investigation."
        )

    record_samples = sum(samples_per_record)
    header_bytes = 256 * (1 + n_signals_on_disk)

    # --- stream: one signal at a time, compare, free ---
    with open(clean_file_path, "rb") as f:
        with _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
            data = np.frombuffer(
                mm,
                dtype=np.int16,
                count=n_records * record_samples,
                offset=header_bytes,
            )
            records = data.reshape(n_records, record_samples)

            # The try/finally guarantees the mmap-backed views are
            # dropped before mmap.__exit__ runs, even when the audit
            # raises mid-loop. Without it, mmap.close() raises
            # BufferError because records/data still hold pointers
            # into its buffer — masking the real AUDIT FAILURE we want
            # the caller to see.
            try:
                for data_idx, disk_idx in enumerate(data_signal_disk_indices):
                    spr = samples_per_record[disk_idx]
                    col_offset = sum(samples_per_record[:disk_idx])
                    # .copy() materialises one channel (~20 MB per channel
                    # for a 178-channel NK export). Freed before next loop
                    # iteration so peak stays at one channel, not N.
                    clean_sig = records[:, col_offset:col_offset + spr].copy().ravel()

                    orig_sig = orig_signals[data_idx]
                    min_len = min(len(orig_sig), len(clean_sig))
                    if not np.array_equal(orig_sig[:min_len], clean_sig[:min_len]):
                        raise RuntimeError(
                            f"AUDIT FAILURE for {filename}: signal channel "
                            f"{data_idx} differs between the original file "
                            "(loaded into memory before de-identification) "
                            "and the de-identified file on disk. The "
                            "de-identification pipeline is supposed to only "
                            "modify header fields and annotations — signal "
                            "samples must remain bit-identical. Do NOT use "
                            "this output file. Save log.out and send it to "
                            "the data management team for investigation."
                        )
                    del clean_sig
            finally:
                del records
                del data

    n_data_signals = len(data_signal_disk_indices)

    # Independent-code-path cross-check: read ONE random non-annotation
    # channel via pyedflib's per-channel readSignal and compare to the
    # corresponding orig_signals[i]. Catches subtle layout bugs in the
    # mmap helper that would otherwise be hidden because both orig and
    # clean came from the same mmap code path (self-consistency would
    # mask the bug).
    #
    # pyedflib's readSignal does ~n_records small disk seeks for one
    # channel — slower than mmap but bounded to a single channel, so
    # cost is ~1-2 s on a 3.8 GB file. Negligible vs the rest of the
    # pipeline; worth it for the defensive cross-validation.
    if n_data_signals > 0:
        spot_idx = random.randrange(n_data_signals)
        with pyedflib.EdfReader(clean_file_path) as f:
            pyedflib_sig = f.readSignal(spot_idx, digital=digital)
        orig_sig = orig_signals[spot_idx]
        min_len = min(len(orig_sig), len(pyedflib_sig))
        if not np.array_equal(orig_sig[:min_len], pyedflib_sig[:min_len]):
            raise RuntimeError(
                f"AUDIT FAILURE for {filename} (pyedflib cross-check): "
                f"signal channel {spot_idx} appears unchanged when read "
                "via the fast mmap path, but disagrees with pyedflib's "
                "per-channel read of the same file's bytes. This means "
                "the two independent readers see different signal values "
                "in the de-identified file — most likely a bug in the "
                "fast loader. Do NOT use this output file. Save log.out "
                "and send it to the data management team for "
                "investigation."
            )

    # Success is intentionally silent — per-file confirmations used to
    # scroll for hundreds of lines on multi-file subjects, drowning
    # out the actually-important redaction warnings. The tqdm postfix
    # in clean_subject_edf_files carries running counts, and the
    # manifest records n_files_deidentified for post-hoc verification.
    # Failures still raise loudly (see the RuntimeErrors above).
    _ = (n_data_signals, spot_idx)


def convert_edfC_to_edfD(input_file: str):
    from clean_eeg.split_discontinuous_edf import overwrite_edfD_to_edfC
    from clean_eeg.load_eeg import is_edfC, is_edfD
    if is_edfD(input_file):
        overwrite_edfD_to_edfC(input_file, require_continuous_data=False)
        assert is_edfC(input_file)


def _load_edf_metadata(input_path: str,
                       load_method: str = "pyedflib",
                       verbosity: int = 1,
                       convert_to_edfC: bool = True,
                       repair_truncated: bool = True,
                       repair_phys_ranges: bool = True,
                       raise_errors: bool = False,
                       force_load_all: bool = False,
                       bench=None):
    from clean_eeg.repair_edf import (
        validate_edf_minimum_size,
        repair_main_header_numeric_fields,
        repair_degenerate_signal_ranges,
    )
    from clean_eeg.benchmark import BenchmarkCollector
    if bench is None:
        bench = BenchmarkCollector(enabled=False)
    EDF_meta_data = dict()
    failed_files: list[tuple[str, str]] = []  # (filename, error_message)
    consecutive_failures = 0
    # sorted() so the consecutive-failure counter and the tqdm progress
    # bar are deterministic — os.listdir order is filesystem-dependent
    # and makes load-cap behavior unpredictable across platforms.
    for filename in tqdm(sorted(os.listdir(input_path)),
                          desc="Loading EDF meta-data..."):
        if not filename.lower().endswith('.edf'):
            continue
        full_path = os.path.join(input_path, filename)
        try:
            validate_edf_minimum_size(full_path)
            if convert_to_edfC:
                with bench.step("convert_edfD_to_edfC", file=filename):
                    convert_edfC_to_edfD(full_path)
            if repair_truncated:
                # Single pass: repairs bytes_in_header, record_duration,
                # and n_records (truncation / sentinel / empty). n_signals
                # empty is surfaced as a ValueError here.
                with bench.step("repair_main_header_numeric_fields", file=filename):
                    repair_main_header_numeric_fields(full_path,
                                                       verbosity=verbosity)
            if repair_phys_ranges:
                with bench.step("repair_phys_ranges", file=filename):
                    repair_degenerate_signal_ranges(full_path, verbosity=verbosity)
            with bench.step("load_edf_metadata_only", file=filename):
                data = load_edf(full_path, load_method=load_method, preload=False)
            EDF_meta_data[filename] = {'data': data}
            consecutive_failures = 0  # a success resets the streak
        except Exception as e:
            if raise_errors:
                raise e
            failed_files.append((filename, f"{type(e).__name__}: {e}"))
            print(
                f"ERROR: Failed to load EDF file {filename}:\n\n"
                f"{e}\n\n"
                f"Stack trace (for the data team):\n"
                f"{traceback.format_exc().rstrip()}\n\n"
                f"Check if the file is corrupted. Skipping this file...\n"
            )
            _dump_edf_header_for_diagnosis(full_path)
            consecutive_failures += 1
            if (consecutive_failures >= MAX_CONSECUTIVE_LOAD_FAILURES
                    and not force_load_all):
                raise ConsecutiveLoadFailureLimit(
                    f"Aborting: {MAX_CONSECUTIVE_LOAD_FAILURES} EDF files "
                    f"in a row failed to load. This usually indicates a "
                    f"systematic issue with the input directory (wrong "
                    f"format, permissions, truncated export) rather than "
                    f"per-file corruption. Investigate the files above, "
                    f"then re-run with --force_load_all to attempt every "
                    f"remaining file regardless of failure streak."
                )
    if failed_files:
        print(
            f"\nWARNING: {len(failed_files)} EDF file(s) were skipped during "
            f"loading and will not be de-identified:"
        )
        for fname, err in failed_files:
            print(f"  - {fname}: {err}")
        print(
            "Please send the log file (log.out, in the EDF directory) to the "
            "data management team so these files can be investigated.\n"
        )
    return EDF_meta_data


def _get_start_time_earliest_recording(EDF_meta_data: dict, verbosity: int = 0) -> datetime:
    # compute the relative start times of all recordings with respect to the earliest recording
    start_times = list()
    for filename, edf in EDF_meta_data.items():
        data = edf['data']
        start_time = data['header']['startdate']
        if verbosity > 1:
            print(f"Start time for {filename}: {start_time}")
        start_times.append(start_time)
    min_start_time = min(start_times)
    if verbosity > -1:
        print(f"Earliest recording start time across all files: {min_start_time}")
    return min_start_time


def _validate_EDF_meta_data(EDF_meta_data: dict, subject_name: Union[PersonalName, None],
                            verbosity: int = 0, skip_header_name_check: bool = False):
    _check_recording_gaps(EDF_meta_data, verbosity=verbosity)
    if skip_header_name_check:
        print("Skipping EDF header subject-name consistency check "
              "(--skip_header_name_check). Name redaction will still run against all header fields.")
    else:
        _check_subject_name_consistency(EDF_meta_data, command_line_subject_name=subject_name,
                                        verbosity=verbosity)
    _check_signal_header_consistency(EDF_meta_data, verbosity=verbosity)


def _check_recording_gaps(EDF_meta_data: dict, verbosity: int = 0):
    # check for gaps between recordings greater than 1 hour
    start_times = list()
    end_times = dict()
    for filename, edf in EDF_meta_data.items():
        data = edf['data']
        start_time = data['header']['startdate']
        start_times.append((filename, start_time))
        file_duration_manual = data['header']['record_duration'] * data['header']['n_records']
        file_duration = data['header']['file_duration']
        if not np.isclose(file_duration, file_duration_manual, atol=0.5):
            print(f"WARNING: EDF file {filename} has inconsistent file duration (pyedflib duration: "
                  f"{file_duration} s vs. manual calculation: {file_duration_manual} s).")
        end_time = start_time + timedelta(seconds=file_duration)
        end_times[filename] = end_time
    start_times.sort(key=lambda x: x[1])  # sort by datetime
    continue_input = 'yes'
    confirm_continue = False
    for i in range(1, len(start_times)):
        prev_filename, _ = start_times[i-1]
        curr_filename, curr_start_time = start_times[i]
        gap = curr_start_time - end_times[prev_filename]
        end_time_prev = end_times[prev_filename]
        if gap.total_seconds() > MAX_RECORDING_GAP_SECONDS:
            print(f"WARNING: Gap of {gap} between neighboring recordings:\n"
                  f"{prev_filename} (end: {end_time_prev}) and\n"
                  f"{curr_filename} (start: {curr_start_time}).")
            print('This may indicate missing recording files. Double check no additional recording files are available.')
            confirm_continue = True
        elif gap.total_seconds() < MIN_RECORDING_GAP_WARNING_SECONDS:
            print(f"WARNING: Overlap of {abs(gap.total_seconds())} seconds between neighboring recordings:\n"
                  f"{prev_filename} (end: {end_time_prev}) and\n"
                  f"{curr_filename} (start: {curr_start_time}).")
            print('This may indicate corrupted EDF files. Check with the data analysis team.')
            if gap.total_seconds() < MIN_RECORDING_GAP_ERROR_SECONDS:
                confirm_continue = True
    if confirm_continue:
        continue_input = logged_input("Continue? yes/no: ")
    if continue_input.lower() not in ['yes', 'y']:
        raise RuntimeError("Aborting EDF de-identification conversion due to recording gap.")


def is_all_X_with_spaces(s: str) -> bool:
    return re.fullmatch(r"\s*X[\sX]*", s) is not None


def _check_subject_name_consistency(EDF_meta_data: dict, command_line_subject_name: Union[PersonalName, None],
                                    verbosity: int = 0):
    subject_names = dict()
    for filename, edf in EDF_meta_data.items():
        data = edf['data']
        header = data['header']
        subject_name = header.get('patientname', 'unknown')
        subject_names[filename] = subject_name
    unique_names = set(subject_names.values())
    if len(unique_names) > 1:
        print("WARNING: Multiple unique subject names found across EDF files:")
        for name in unique_names:
            files_with_name = [fname for fname, sname in subject_names.items() if sname == name]
            print(f'Subject name "{name}" found in files: {files_with_name}')
        print("This may indicate multiple subjects are included in the same EDF data folder, which should not be the case.")
        continue_input = logged_input("Continue? (only continue if names are indeed from the same subject for data integrity) yes/no: ")
        if continue_input.lower() not in ['yes', 'y']:
            raise RuntimeError("Aborting EDF de-identification conversion due to inconsistent subject names.")
    elif len(unique_names) < 1:
        raise RuntimeError("No subject names found in EDF files.")
    
    if command_line_subject_name is not None:
        command_line_subject_name_str = command_line_subject_name.get_full_name()
        continue_input = 'yes'
        if len(unique_names) == 1:
            subject_name = unique_names.pop()
            if (not is_all_X_with_spaces(subject_name)) and (subject_name != command_line_subject_name_str):
                continue_input = logged_input(f'Confirm that subject name in EDF files ("{subject_name}") matches '
                                       f'subject name specified by command line ("{command_line_subject_name_str}"): yes/no: ')
        elif (len(unique_names) > 1) and not all(is_all_X_with_spaces(subject_name) for subject_name in unique_names):
            continue_input = logged_input(f'Confirm that subject names in EDF files ({unique_names}) match '
                                   f'subject name specified by command line ("{command_line_subject_name_str}"): yes/no: ')
        if continue_input.lower() not in ['yes', 'y']:
            raise RuntimeError("Aborting EDF de-identification conversion due to inconsistent subject names.")


def _check_signal_header_consistency(EDF_meta_data: dict, verbosity: int = 0):
    signal_label_sets = dict()
    for filename, edf in EDF_meta_data.items():
        data = edf['data']
        signal_headers = data['signal_headers']
        signal_label_sets[filename] = tuple(signal_header['label']
                                            for signal_header in signal_headers)
    unique_label_sets = {*list(signal_label_sets.values())}
    if len(unique_label_sets) > 1:
        # Compact form: one line per unique signature with a channel
        # count and file count. Full per-signature label tuples used
        # to scroll for hundreds of lines on multi-montage NK subjects
        # — the audit tool has the detailed view via edf_audit.json.
        print(f"WARNING: {len(unique_label_sets)} unique signal-header "
              f"signatures across {len(signal_label_sets)} files.")
        for i, labels in enumerate(unique_label_sets):
            files_with_header = [fname for fname, label_keys in signal_label_sets.items()
                                 if label_keys == labels]
            preview = list(files_with_header[:3])
            more = f" (+{len(files_with_header)-3} more)" if len(files_with_header) > 3 else ""
            print(f"  signature {i+1}: {len(labels)} channels, "
                  f"{len(files_with_header)} file(s), e.g. {preview}{more}")
        print("This may indicate inconsistent EDF signal labels across recordings "
              "or multiple subjects across files in the EDF data folder.")
        print("Alternatively, this may be due to multiple recording montages during "
              "e.g., the same stay in the epilepsy monitoring unit.")
        print("Full labels are available via `audit-subject-eeg --print-edf-signal-header`.")
        continue_input = logged_input("Continue? (only continue if recordings have been confirmed as coming from the same subject and EMU stay for data integrity) yes/no: ")
        if continue_input.lower() not in ['yes', 'y']:
            raise RuntimeError("Aborting EDF de-identification conversion due to inconsistent signal headers.")


def get_clean_eeg_cli_arguments():
    import argparse
    import os

    def prompt_if_missing(args):
        """Prompt the user interactively for any missing required arguments."""

        # Required fields that must be non-empty
        required_fields = {
            "input_path":   "Enter path to all EDF files: ",
            "subject_code": "Enter subject code (e.g., R1755A): ",
            "first_name":   "Enter subject first name: ",
            "last_name":    "Enter subject last name: ",
        }

        # Prompt for required arguments
        for attr, prompt in required_fields.items():
            if getattr(args, attr) in (None, ""):
                value = logged_input(prompt).strip()
                setattr(args, attr, value)

        # Middle name: optional, but still prompt if missing.
        # --no_middle_name short-circuits the prompt entirely (the flag
        # is the cross-platform way to say "subject has no middle name"
        # since Windows cmd.exe drops empty quoted args).
        if getattr(args, "no_middle_name", False):
            args.middle_name = ""
        elif args.middle_name in (None, "", "NOT_SPECIFIED"):
            mn = logged_input(
                "Enter subject middle name(s) "
                "(use underscores between multiple names; press Enter to skip; "
                "or pass --no_middle_name on the command line for a "
                "non-interactive skip): "
            ).strip()
            if mn:  # Only override default if user typed something
                args.middle_name = mn

        return args

    parser = argparse.ArgumentParser(
        description="Rename and clean meta-data for clinical EEG EDF files "
                    "after mass export by Nihon Kohden."
    )

    # ---- DO NOT mark required=True; we prompt manually ----
    parser.add_argument("--input_path", type=str, default='',
                        help="Path to all EDF files (required)")
    parser.add_argument("--copy_path", type=str, default=None,
                        help="Write de-identified files to this directory instead "
                             "of modifying in place. If set without a value, "
                             "defaults to 'deidentified_eeg_files' within input_path.",
                        nargs='?', const='')
    parser.add_argument("--subject_code", type=str, default='',
                        help="Subject code (e.g., R1755A) (required)")
    parser.add_argument("--first_name", type=str, default='',
                        help="Subject first name (required)")
    parser.add_argument("--middle_name", type=str, default="NOT_SPECIFIED",
                        help='Subject middle name(s). Use underscores between '
                             'multiple middle names. If no middle name, pass '
                             '--no_middle_name (works on Windows cmd.exe, '
                             'which strips empty quoted args).')
    parser.add_argument("--no_middle_name", action="store_true",
                        help='Subject has no middle name. Cross-platform '
                             'alternative to --middle_name "" — needed on '
                             'shells that drop empty quoted arguments. '
                             'Mutually exclusive with --middle_name.')
    parser.add_argument("--last_name", type=str, default='',
                        help="Subject last name (required)")
    parser.add_argument("--raise_errors", action="store_true",
                        help="Raise errors instead of warnings for debugging")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="Enable verbose output")
    parser.add_argument("--skip_header_name_check", action="store_true",
                        help="Skip the EDF-header subject-name consistency check. Use when "
                             "header name fields have already been redacted but annotations "
                             "still need to be cleaned. Name redaction is still applied to "
                             "all header fields.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Print per-step wall time, RSS delta, and peak-RSS growth for "
                             "each EDF file. Useful for profiling the pipeline's time and "
                             "memory hot-spots.")
    parser.add_argument("--skip_audit", action="store_true",
                        help="Skip the post-write signal-integrity audit. In inplace mode the "
                             "audit is the only reason signals are loaded at all; this flag "
                             "avoids the per-channel interleaved read that pyedflib performs, "
                             "which can take minutes on multi-GB Nihon Kohden files. Headers "
                             "and annotations are still de-identified; only the cross-check "
                             "that signals survived byte-identical is skipped.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run de-identification even if deidentify.json already exists "
                             "in the output directory. Without --force, a pre-existing manifest "
                             "short-circuits straight to the transfer prompt.")
    parser.add_argument("--force_load_all", action="store_true",
                        help=f"Bypass the {MAX_CONSECUTIVE_LOAD_FAILURES}-consecutive-load-failure "
                             "abort. Use only after inspecting the failed files' errors and "
                             "confirming the remaining files are worth attempting.")
    parser.add_argument("--wipe-annotations", "--wipe_annotations",
                        dest="wipe_annotations", action="store_true",
                        help="DELETE all non-timekeeping annotations from the output EDF "
                             "instead of Presidio-redacting them. No separate '_annotations.edf' "
                             "sidecar is written. Requires per-subject confirmation (type the "
                             "subject code) unless suppressed via --approve-confirmations. "
                             "Preserves EDF+ mandatory record-timekeeping TALs.")
    parser.add_argument("--approve-confirmations", "--approve_confirmations",
                        dest="approve_confirmations", nargs="+", default=[],
                        choices=["wipe-annotations"],
                        help="List of destructive-operation confirmation prompts to auto-approve "
                             "(for headless / non-interactive runs). Each type must be listed "
                             "explicitly — there is no global --yes. New confirmation types must "
                             "be added to `choices` explicitly, forcing per-type opt-in.")

    args = parser.parse_args()

    # Mutually-exclusive guard. ``--middle_name`` keeps its
    # "NOT_SPECIFIED" sentinel default, so equality with the sentinel
    # is how we tell that the user didn't actually pass it.
    if args.no_middle_name and args.middle_name != "NOT_SPECIFIED":
        parser.error("--no_middle_name and --middle_name are mutually "
                     "exclusive; pick one")

    # Prompt for anything missing (including middle name)
    args = prompt_if_missing(args)

    # Resolve output_path based on mode
    if args.copy_path is not None:
        # Copy mode
        if not args.copy_path:
            args.output_path = os.path.join(args.input_path, "deidentified_eeg_files")
        else:
            args.output_path = args.copy_path
    else:
        # Inplace mode (default)
        args.output_path = args.input_path

    return args


def validate_cli_arguments(args):
    if not os.path.exists(args.input_path):
        raise ValueError(f"Input path does not exist: {args.input_path}")
    if args.copy_path is not None:
        if args.output_path == args.input_path:
            raise ValueError("With --copy_path, output path must differ from input path.")
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
    else:
        print(f"WARNING: De-identification will modify EDF files in place at:\n"
              f"  {args.input_path}\n"
              f"Original headers will be overwritten. Use --copy_path to write to a separate directory instead.")
        confirm = logged_input("Continue with in-place de-identification? yes/no: ")
        if confirm.lower() not in ['yes', 'y']:
            raise RuntimeError("Aborting. Re-run with --copy_path to write to a separate directory.")

    if args.middle_name == 'NOT_SPECIFIED':
        raise ValueError('Middle name must be specified. Pass --middle_name '
                         'with the name(s), or --no_middle_name if the '
                         'subject has none (the latter works on Windows '
                         'cmd.exe; --middle_name "" does not). '
                         'If subject has only a middle initial, provide the initial instead. '
                         'Separate multiple middle names with underscores (e.g., Paul_Angelina)')
    # First/last name backstop. prompt_if_missing handles the
    # interactive case, but a batch invocation passing e.g.
    # --first_name "" on POSIX would slip past it. Reject here so the
    # pipeline does not silently produce wrong output downstream.
    if not args.first_name.strip():
        raise ValueError('First name is required. Pass --first_name <name>. '
                         'Empty/whitespace-only values are not accepted.')
    if not args.last_name.strip():
        raise ValueError('Last name is required. Pass --last_name <name>. '
                         'Empty/whitespace-only values are not accepted.')

    print('Loading EDF files from path:', args.input_path)
    is_valid_subject_code(args.subject_code)


def redact_log_file(log_path: str, subject_name: PersonalName):
    """Run full name redaction on the log file to catch fuzzy matches and nicknames."""
    with open(log_path, "r") as f:
        content = f.read()
    redacted = redact_subject_name(content, subject_full_name=subject_name)
    with open(log_path, "w") as f:
        f.write(redacted)


LOG_FILENAME = "log.out"

if __name__ == "__main__":
    import tempfile
    # Start logging to a temp file so the log can capture interactive prompts
    # that run before args (and thus input_path) are known. Relocated into
    # input_path as soon as args are parsed.
    _tmp_fd, _tmp_log_path = tempfile.mkstemp(prefix="clean_eeg_log_", suffix=".out")
    os.close(_tmp_fd)
    log_path = _tmp_log_path
    logger = setup_logger(log_path)

    try:
        args = get_clean_eeg_cli_arguments()
        # Relocate the log into the subject's EDF directory now that we know it.
        if args.input_path and os.path.isdir(args.input_path):
            logger.relocate(os.path.join(args.input_path, LOG_FILENAME))
            log_path = logger.log_path
        validate_cli_arguments(args)

        # Register subject name parts as PHI first so the provenance
        # block that follows (which includes sys.argv) gets scrubbed on
        # write via the tee. rescrub() also cleans anything already
        # written (e.g., interactive-prompt captures) with the new patterns.
        for name_part in [args.first_name, args.last_name]:
            logger.add_phi(name_part)
        if args.middle_name and args.middle_name != "NOT_SPECIFIED":
            for mn in args.middle_name.split('_'):
                logger.add_phi(mn)
        logger.rescrub()

        # Full environment provenance: clean_eeg version + git SHA + dirty
        # flag + command line + python + OS + key-dep versions. Load-bearing
        # for reproducing an issue reported months after a subject shipped.
        from clean_eeg.provenance import log_environment_provenance
        log_environment_provenance(logger)

        logger.log_args(args)

        middle_names = [mn for mn in args.middle_name.split('_') if mn] if args.middle_name else []
        subject_name = PersonalName(
            first_name=args.first_name,
            middle_names=middle_names,
            last_name=args.last_name
        )

        if args.wipe_annotations:
            approved = set(args.approve_confirmations)
            if not confirm_wipe_annotations(args.subject_code, approved):
                raise SystemExit(1)

        clean_subject_edf_files(
            input_path=args.input_path,
            output_path=args.output_path,
            subject_code=args.subject_code,
            subject_name=subject_name,
            raise_errors=args.raise_errors,
            inplace=args.copy_path is None,
            verbosity=args.verbosity,
            skip_header_name_check=args.skip_header_name_check,
            benchmark=args.benchmark,
            skip_audit=args.skip_audit,
            force=args.force,
            force_load_all=args.force_load_all,
            wipe_annotations=args.wipe_annotations,
        )

    except Exception:
        import traceback
        traceback.print_exc()
        # Read the current log path from the logger (reflects any relocation).
        log_path = logger.log_path
        print(f"\nPlease send the log file to the data management team for debugging:")
        print(f"  {log_path}")
        raise SystemExit(1)

    finally:
        log_path = logger.log_path
        close_logger()
        # Run full name redaction on the log file (fuzzy matching, nicknames, etc.)
        _subject_name = locals().get('subject_name')
        if _subject_name is not None and os.path.exists(log_path):
            redact_log_file(log_path, _subject_name)
        # Copy log alongside output files for transfer (skip if it already lives there)
        if 'args' in locals() and hasattr(args, 'output_path') and args.output_path and os.path.isdir(args.output_path):
            dest = os.path.join(args.output_path, LOG_FILENAME)
            if os.path.abspath(dest) != os.path.abspath(log_path) and os.path.exists(log_path):
                shutil.copy(log_path, dest)
