import random
import re
import os
import shutil
import numpy as np
import pyedflib
from copy import deepcopy
from typing import Union
from datetime import datetime, timedelta
from tqdm import tqdm
from clean_eeg.anonymize import redact_subject_name, PersonalName, SubjectNameRedactor
from clean_eeg.load_eeg import load_edf, write_edf_pyedflib
from clean_eeg.log import logged_input, setup_logger, get_logger, close_logger
from clean_eeg.modify_edf_inplace import (
    update_edf_header_inplace,
    clear_edf_annotations_inplace,
    create_annotations_only_edf,
    validate_header_roundtrip,
)

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


def deidentify_edf(edf_data, subject_name, subject_code, earliest_recording_start_time,
                   redactor: Union[SubjectNameRedactor, None] = None):
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
    return {
        'header': deidentify_edf_header(edf_data['header'],
                                        subject_name=subject_name,
                                        subject_code=subject_code,
                                        earliest_recording_start_time=earliest_recording_start_time,
                                        redactor=redactor),
        'signal_headers': clean_signal_headers,
        'annotations': deidentify_edf_annotations(edf_data['annotations'],
                                                  subject_name=subject_name,
                                                  redactor=redactor),
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
                                redactor: Union[SubjectNameRedactor, None] = None):
    clean_start_times = list()
    clean_durations = list()
    clean_descriptions = list()
    for (start_time, duration, text) in zip(*annotations):
        assert isinstance(text, str)
        redacted_text = redact_string(str(text),
                                      field_name='annotation',
                                      subject_name=subject_name,
                                      alert=True,
                                      redactor=redactor)
        clean_start_times.append(start_time)
        clean_durations.append(duration)
        clean_descriptions.append(redacted_text)
        
    clean_annotations = (np.array(clean_start_times),
                         np.array(clean_durations), 
                         np.array(clean_descriptions))
    return clean_annotations


def is_valid_subject_code(subject_code,
                          pattern=r'^R1\d{3}[ACDEFHJMNPST]$',
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


def deidentify_start_date_time(recording_start_time, earliest_recording_start_time):
    shifted_time = recording_start_time - earliest_recording_start_time + BASE_START_DATE
    return shifted_time


# Matches empty, all-whitespace, pure-numeric (incl. sign and decimal), or
# EDF+ timekeeping-TAL-shaped strings like "+0.086" / "-12.5" / "+1234".
# These cannot contain PHI — skip the Presidio pass entirely.
_NON_PHI_TEXT_RE = re.compile(r"^\s*[+-]?\d*\.?\d*\s*$")


def redact_string(text: str, field_name: str, subject_name: PersonalName,
                  alert: bool = False,
                  redactor: Union[SubjectNameRedactor, None] = None) -> str:
    if _NON_PHI_TEXT_RE.match(text):
        # Empty, numeric, or timekeeping-shaped — cannot hold PHI; skip Presidio.
        return text
    redacted = redact_subject_name(text, subject_full_name=subject_name, redactor=redactor)
    redacted = remove_gendered_pronouns(redacted)
    if alert and text != redacted:
        # Print the *redacted* value (not the raw one) so the log stays
        # PHI-free while still showing what was flagged and what survived.
        print('Subject protected health information detected in EDF '
              f'{field_name}; redacted value: "{redacted}". '
              'Alert the data analysis team.')
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
):
    from clean_eeg.benchmark import BenchmarkCollector
    bench = BenchmarkCollector(enabled=benchmark)

    if inplace:
        assert input_path == output_path, "For inplace cleaning, input_path must equal output_path."
    EDF_meta_data = _load_edf_metadata(input_path=input_path,
                                       verbosity=verbosity,
                                       load_method=load_method,
                                       raise_errors=raise_errors,
                                       bench=bench)

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
    all_filenames = list(EDF_meta_data.keys())
    if skip_audit:
        audit_filenames: set = set()
    else:
        n_audit = min(2, len(all_filenames))
        audit_filenames = set(random.sample(all_filenames, n_audit))

    # Build Presidio once per subject and reuse across all redact_string calls.
    # This amortizes the spaCy-model + recognizer-registry construction cost.
    with bench.step("build_presidio_redactor"):
        redactor = SubjectNameRedactor(subject_name) if subject_name is not None else None

    # de-identify EDF files and save out
    print("Cleaning EDF files... Saving to output path:", output_path)
    failed_files: list[tuple[str, str]] = []
    for filename, _ in tqdm(EDF_meta_data.items()):
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
                )
            with bench.step("validate_header_roundtrip", file=filename):
                truncation_warnings = validate_header_roundtrip(
                    edf['header'], edf['signal_headers'])
            for warning in truncation_warnings:
                print(f"WARNING: {warning}")

            clean_start_time = edf['header']['startdate']
            filename_no_ext = os.path.splitext(filename)[0]
            subject_val = subject_code
            clean_filename = f"{filename_no_ext}_{subject_val}_{clean_start_time.strftime('%Y.%m.%d__%H.%M.%S')}.edf"
            clean_full_path = os.path.join(output_path, clean_filename)
            if inplace:
                with bench.step("write_inplace", file=filename):
                    shutil.move(input_file_path, clean_full_path)
                    clean_annotations_path = str(clean_full_path).replace('.edf', '_annotations.edf')
                    create_annotations_only_edf(clean_annotations_path,
                                                header=edf['header'],
                                                annotations=edf['annotations'])
                    update_edf_header_inplace(clean_full_path,
                                              header_updates=edf['header'],
                                              signal_header_updates=edf['signal_headers'])
                    clear_edf_annotations_inplace(clean_full_path)
            else:
                with bench.step("write_edf_pyedflib", file=filename):
                    write_edf_pyedflib(edf, clean_full_path, digital=read_digital)
            print(f"Cleaned EDF file at: {clean_filename}")

            # Audit signal integrity immediately after write
            if filename in audit_filenames:
                with bench.step("audit_signal_integrity", file=filename):
                    _audit_signal_integrity(orig_signals, clean_full_path, filename,
                                            inplace=inplace, digital=read_digital)
        except Exception as e:
            if raise_errors:
                raise e
            failed_files.append((filename, f"{type(e).__name__}: {e}"))
            print(f"\nERROR: Failed to de-identify EDF file {filename}:\n\n{e}\n\n"
                  f"Skipping this file and continuing...\n")

    print("Done cleaning EDF files. Saved to output path:", output_path)
    if benchmark:
        print(bench.report())
    if failed_files:
        print(
            f"\nWARNING: {len(failed_files)} EDF file(s) were skipped during "
            f"de-identification and will need to be handled manually:"
        )
        for fname, err in failed_files:
            print(f"  - {fname}: {err}")
        print(
            "Please send the log file (log.out, in the EDF directory) to the "
            "data management team so these files can be investigated.\n"
        )
    site_code = subject_code[-1]  # last character of subject code is site code
    site_code_incoming_folder = SITE_CODE_TO_INCOMING_FOLDER.get(site_code, 'UNKNOWN_SITE')
    remote_dir = f"/data10/RAM/incoming/{site_code_incoming_folder}/{subject_code}/all_clinical_eeg"
    print("Example commands to transfer cleaned EDF files to the CML rhino server (make sure to change USER to appropriate username):")
    print(f'ssh USER@rhino2.psych.upenn.edu "mkdir -p {remote_dir}"')
    print(f"scp {os.path.join(output_path, '*.edf')} USER@rhino2.psych.upenn.edu:{remote_dir}")


def _audit_signal_integrity(orig_signals: list, clean_file_path: str, filename: str,
                            inplace: bool = False, digital: bool = False):
    """Spot-check that signal data in the output file matches the original.

    For inplace mode, signals must be bit-identical since only headers are modified.
    For rewrite mode, pyedflib's digital/physical conversion introduces floating-point
    differences, so the audit is skipped (this is a known pyedflib limitation and the
    reason the in-place approach was developed).

    ``digital`` must match the mode used to read ``orig_signals``; when True, the
    clean file is also read in digital mode so the bit-comparison is meaningful.
    """
    if not inplace:
        return
    with pyedflib.EdfReader(clean_file_path) as f:
        n_signals = f.signals_in_file
        for i in range(n_signals):
            clean_signal = f.readSignal(i, digital=digital)
            orig_signal = orig_signals[i]
            min_len = min(len(orig_signal), len(clean_signal))
            if not np.array_equal(orig_signal[:min_len], clean_signal[:min_len]):
                raise RuntimeError(
                    f"AUDIT FAILURE: Signal {i} in {filename} was modified "
                    f"during in-place de-identification."
                )
    print(f"Audit passed for {filename}: all {n_signals} signals unchanged.")


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
                       bench=None):
    from clean_eeg.repair_edf import (
        repair_truncated_edf_header,
        repair_degenerate_signal_ranges,
    )
    from clean_eeg.benchmark import BenchmarkCollector
    if bench is None:
        bench = BenchmarkCollector(enabled=False)
    EDF_meta_data = dict()
    failed_files: list[tuple[str, str]] = []  # (filename, error_message)
    for filename in tqdm(os.listdir(input_path), desc="Loading EDF meta-data..."):
        if not filename.lower().endswith('.edf'):
            continue
        full_path = os.path.join(input_path, filename)
        try:
            if convert_to_edfC:
                with bench.step("convert_edfD_to_edfC", file=filename):
                    convert_edfC_to_edfD(full_path)
            if repair_truncated:
                with bench.step("repair_truncated_header", file=filename):
                    repair_truncated_edf_header(full_path, verbosity=verbosity)
            if repair_phys_ranges:
                with bench.step("repair_phys_ranges", file=filename):
                    repair_degenerate_signal_ranges(full_path, verbosity=verbosity)
            with bench.step("load_edf_metadata_only", file=filename):
                data = load_edf(full_path, load_method=load_method, preload=False)
            EDF_meta_data[filename] = {'data': data}
        except Exception as e:
            if raise_errors:
                raise e
            failed_files.append((filename, f"{type(e).__name__}: {e}"))
            print(f"ERROR: Failed to load EDF file {filename}:\n\n{e}\n\n"
                  f"Check if the file is corrupted. Skipping this file...\n")
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
        print("WARNING: Multiple unique sets of signal header labels found across EDF files:")
        for labels in unique_label_sets:
            files_with_header = [fname for fname, label_keys in signal_label_sets.items() if label_keys == labels]
            print(f'Signal header labels\n\n{labels}\n\nfound in files:\n{files_with_header}')
        print("\nThis may indicate inconsistent EDF signal labels across recordings or multiple subjects across files in the EDF data folder.")
        print('Alternatively, this may be due to multiple recording montages during e.g., the same stay in the epilepsy monitoring unit.')
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

        # Middle name: optional, but still prompt if missing
        # If user presses Enter, leave default "NOT_SPECIFIED"
        if args.middle_name in (None, "", "NOT_SPECIFIED"):
            mn = logged_input(
                "Enter subject middle name(s) "
                "(use underscores between multiple names; press Enter to skip): "
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
                             'multiple middle names. If no middle name, use ""')
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

    args = parser.parse_args()

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
        raise ValueError('Middle name must be specified with --middle-name argument. '
                         'If subject has no middle name, use --middle-name "" to leave blank. '
                         'If subject has only a middle initial, provide the initial instead. '
                         'Separate multiple middle names with underscores (e.g., Paul_Angelina)')

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

        # Log the installed version up-front so shared log.out files always
        # carry provenance for the code that produced them.
        try:
            from importlib.metadata import version as _pkg_version
            _installed_version = _pkg_version("clean_eeg")
        except Exception:
            _installed_version = "unknown"
        print(f"clean_eeg version: {_installed_version}")

        # Register subject name parts as PHI for log scrubbing
        for name_part in [args.first_name, args.last_name]:
            logger.add_phi(name_part)
        if args.middle_name and args.middle_name != "NOT_SPECIFIED":
            for mn in args.middle_name.split('_'):
                logger.add_phi(mn)
        logger.rescrub()
        logger.log_args(args)

        middle_names = [mn for mn in args.middle_name.split('_') if mn] if args.middle_name else []
        subject_name = PersonalName(
            first_name=args.first_name,
            middle_names=middle_names,
            last_name=args.last_name
        )

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
