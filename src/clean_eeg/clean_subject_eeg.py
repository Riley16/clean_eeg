import re
import os
import numpy as np
from copy import deepcopy
from typing import Union
from datetime import datetime
from tqdm import tqdm
from clean_eeg.anonymize import redact_subject_name, PersonalName
from clean_eeg.load_eeg import load_edf, write_edf_pyedflib

BASE_START_DATE = datetime(1985, 1, 1)
DEFAULT_REDACT_HEADER_KEYS = ['patientname', 'sex', 'gender',
                              'patient_additional',
                              'admincode','technician']
REDACT_REPLACEMENT = 'unknown'  # match pyedflib default for missing fields


def deidentify_edf(edf_data, subject_name, subject_code, earliest_recording_start_time):
    # remove protected health information (PHI) from EEG
    # accepts EDF data in 'pyedflib' format

    # de-identification operations:
    # 1) rename subject to subject code and remove meta-data fields for gender, birthdate, patient hospital code
    # 2) replace recording start time with time relative to the earliest recording start time
    # 3) remove any recording annotations containing regex patterns indicating PHI (name, gender, birthdate)
    # 4) save the modified EDF file with a new name in the format SUBJECT_CODE__RELATIVE.START.DATE_RELATIVE:START:TIME.edf
    #        RELATIVE.START.DATE_RELATIVE:START:TIME corresponds to YEAR.MONTH.DAY__HOUR:MINUTE:SECOND relative to the earliest recording start time
    #        relative times are offset by the EDF standard clipping date of 1985-01-01

    edf_data = deepcopy(edf_data)
    edf_data['header'] = deidentify_edf_header(edf_data['header'],
                                               subject_name=subject_name,
                                               subject_code=subject_code,
                                               earliest_recording_start_time=earliest_recording_start_time)
    clean_signal_headers = list()
    for signal_header in edf_data['signal_headers']:
        cleaned = deidentify_edf_header(signal_header,
                                        subject_name=subject_name,
                                        subject_code=subject_code,
                                        earliest_recording_start_time=None,  # signal headers do not have a start time
                                        redact_keys=list()  # check all
                                        )
        clean_signal_headers.append(cleaned)
    edf_data['signal_headers'] = clean_signal_headers
    edf_data['annotations'] = deidentify_edf_annotations(edf_data['annotations'], subject_name=subject_name)
    return edf_data


def deidentify_edf_header(header: dict,
                          subject_code: str,
                          subject_name: PersonalName,
                          earliest_recording_start_time: Union[datetime,None]=None,
                          redact_keys: list[str]=DEFAULT_REDACT_HEADER_KEYS):
    header = deepcopy(header)
    if earliest_recording_start_time is None:
        assert 'startdate' not in header
    else:
        header['startdate'] = deidentify_start_date_time(header['startdate'],
                                                         earliest_recording_start_time)
    header['birthdate'] = datetime(1900, 1, 1)
    for key in redact_keys:
        header[key] = REDACT_REPLACEMENT
    header['patientcode'] = subject_code
    # check for patient name, gendered pronouns in all other fields
    for key, val in header.items():
        if key in redact_keys:
            continue
        if isinstance(val, str):
            header[key] = redact_string(val,
                                        field_name=key,
                                        subject_name=subject_name)
        elif isinstance(val, (int, float, datetime)):
            pass
        else:
            raise ValueError(f'Unknown type in header field {key}: type: {type(val)}; value: {val}')
    return header


def deidentify_edf_annotations(annotations: tuple[np.ndarray], subject_name: PersonalName):
    clean_start_times = list()
    clean_durations = list()
    clean_descriptions = list()
    for (start_time, duration, text) in zip(*annotations):
        assert isinstance(text, str)
        redacted_text = redact_string(str(text),
                                      field_name='annotation',
                                      subject_name=subject_name)
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


def redact_string(text: str, field_name: str, subject_name: PersonalName) -> str:
    redacted = redact_subject_name(text, subject_full_name=subject_name)
    redacted = remove_gendered_pronouns(redacted)
    if text != redacted:
        print('Subject protected health information detected in EDF header field '
            f'"{field_name}" with value "{text}"... redacting. '
            'Alert the data analysis team.')
    return redacted


_GENDERED_PRONOUNS = [
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
]
REDACT_PRONOUN_REPLACEMENT = "[REDACTED_PRONOUN]"

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
    verbosity: int = 1,
):
    EDF_meta_data = _load_edf_metadata(input_path=input_path,
                                       load_method=load_method,
                                       verbosity=verbosity,
                                       raise_errors=raise_errors)

    min_start_time = _get_start_time_earliest_recording(EDF_meta_data, verbosity=verbosity)

    # de-identify EDF files and save out
    for filename, _ in tqdm(EDF_meta_data.items()):
        edf = load_edf(os.path.join(input_path, filename), load_method='pyedflib', preload=True)
        assert isinstance(edf, dict)
        edf = deidentify_edf(
            edf_data=edf,
            subject_name=subject_name,
            subject_code=subject_code,
            earliest_recording_start_time=min_start_time
        )
        clean_start_time = edf['header']['startdate']
        filename_no_ext = os.path.splitext(filename)[0]
        # subject argument is not always present, fallback to subject_code if not provided
        # subject_val = subject if subject is not None else subject_code
        subject_val = subject_code
        clean_filename = f"{filename_no_ext}_{subject_val}_{clean_start_time.strftime('%Y.%m.%d__%H:%M:%S')}.edf"
        clean_full_path = os.path.join(output_path, clean_filename)
        write_edf_pyedflib(edf, clean_full_path)
        print(f"Saved cleaned EDF file as {clean_filename}")
    print("Done cleaning EDF files.")


def convert_edfC_to_edfD(input_file: str):
    from clean_eeg.split_discontinuous_edf import overwrite_edfD_to_edfC
    from clean_eeg.load_eeg import is_edfC, is_edfD
    if is_edfD(input_file):
        overwrite_edfD_to_edfC(input_file, require_continuous_data=True)
        assert is_edfC(input_file)


def _load_edf_metadata(input_path: str,
                       load_method: str = "pyedflib",
                       verbosity: int = 1,
                       convert2edfC: bool = True,
                       raise_errors: bool = False):
    EDF_meta_data = dict()
    for filename in os.listdir(input_path):
        try:
            if filename.lower().endswith('.edf'):
                full_path = os.path.join(input_path, filename)
                if verbosity > 0:
                    print(f"Loading {filename}...")
                if convert2edfC:
                    convert_edfC_to_edfD(full_path)
                data = load_edf(full_path, load_method=load_method, preload=False)
                EDF_meta_data[filename] = {'data': data}
        except Exception as e:
            if raise_errors:
                raise e
            print(f"ERROR: Failed to load EDF file {filename}:\n\n{e}\n\nCheck if the file is corrupted. Skipping this file...\n")
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


def get_clean_eeg_cli_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Rename and clean meta-data for all clinical EEG EDF "
                                                 "files after mass export by Nihon Kohden.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to all EDF files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to all EDF files")
    parser.add_argument("--subject-code", type=str, required=True, help="Subject code (e.g., R1755A)")
    parser.add_argument("--first-name", type=str, required=True, help="Subject first name (e.g., John)")
    parser.add_argument("--middle-name", type=list, default='NOT_SPECIFIED',
                        help='Subject middle name or initial (e.g., Paul or P). Can be left blank with "".')
    parser.add_argument("--last-name", type=str, required=True, help="Subject last name (e.g., Smith)")
    parser.add_argument("--load-method", type=str, default="edfio",
                        help="Method to load EDF files: 'edfio', 'pyedflib', or 'mne'")
    parser.add_argument("--raise-errors", action="store_true",
                        help="Raise errors instead of warnings for debugging")
    parser.add_argument("--verbosity", type=int, default=1, help="Enable verbose output")
    args = parser.parse_args()
    return args


def validate_cli_arguments(args):
    if args.output_path == args.input_path:
        raise ValueError("Output path must be different from input path to avoid overwriting original EDF files.")
    if not os.path.exists(args.input_path):
        raise ValueError(f"Input path does not exist: {args.input_path}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.middle_name == 'NOT_SPECIFIED':
        raise ValueError('Middle name must be specified with --middle-name argument. '
                         'If subject has no middle name, use --middle-name "" to leave blank. '
                         'If subject has only a middle initial, provide the initial instead. '
                         'Separate multiple middle names with underscores (e.g., Paul_Angelina)')

    print('Loading EDF files from path:', args.input_path)
    is_valid_subject_code(args.subject_code)


if __name__ == "__main__":
    args = get_clean_eeg_cli_arguments()
    validate_cli_arguments(args)
    subject_name = PersonalName(
        first_name=args.first_name,
        middle_names=args.middle_name.split('_') if args.middle_name else [],
        last_name=args.last_name
    )

    clean_subject_edf_files(
        input_path=args.input_path,
        output_path=args.output_path,
        subject_code=args.subject_code,
        subject_name=subject_name,
        load_method=args.load_method,
        raise_errors=args.raise_errors,
        verbosity=args.verbosity,
    )
