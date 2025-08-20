import re
import os
from copy import deepcopy
import numpy as np
from datetime import datetime
from clean_eeg.anonymize import redact_subject_name

BASE_START_DATE = datetime(1985, 1, 1)

from clean_eeg.load_eeg import load_edf, get_edf_start_time_from_mne


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
                                        earliest_recording_start_time=earliest_recording_start_time,
                                        redact_keys=list()  # check all
                                        )
        clean_signal_headers.append(cleaned)
    edf_data['signal_headers'] = clean_signal_headers
    edf_data['annotations'] = deidentify_edf_annotations(edf_data['annotations'], subject_name=subject_name)
    return edf_data


def deidentify_edf_header(header,
                          subject_code,
                          earliest_recording_start_time,
                          subject_name,  # require manual entry in case patient name gets stored in non-standard fields
                          redact_keys=['patientname', 'sex', 'gender',
                                       'patient_additional', 'birthdate',
                                       'admincode','technician']):
    header = deepcopy(header)
    clean_start_time = deidentify_start_date_time(header['startdate'], earliest_recording_start_time)
    header['startdate'] = clean_start_time
    for key in redact_keys:
        header[key] = 'REDACTED'
    header['patientcode'] = subject_code
    # check for patient name, gendered pronouns in all other fields
    for key, val in header.items():
        if key in redact_keys:
            continue
        if isinstance(val, str):
            header[key] = redact_string(val,
                                        field_name=key,
                                        subject_name=subject_name)
        elif isinstance(val, (int, float)):
            pass
        else:
            raise ValueError(f'Unknown type in header field {key}: type: {type(val)}; value: {val}')
    return header


def deidentify_edf_annotations(annotations, subject_name):
    clean_start_times = list()
    clean_durations = list()
    clean_descriptions = list()
    for (start_time, duration, text) in zip(*annotations):
        assert isinstance(text, str)
        redacted_text = redact_string(text,
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


def redact_string(text, field_name, subject_name):
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

# \b-boundaries ensure we don't hit substrings (e.g., "her" in "other").
PRONOUN_RE = re.compile(r"\b(" + "|".join(map(re.escape, _GENDERED_PRONOUNS)) + r")\b",
                           flags=re.IGNORECASE | re.UNICODE)

def remove_gendered_pronouns(text: str, replacement: str = "REDACTED_PRONOUN") -> str:
    """
    Remove (or replace) gendered pronouns. Default behavior is deletion.
    Pass replacement='[REDACTED-PRONOUN]' if you prefer explicit redaction.
    """
    if replacement == "":
        return PRONOUN_RE.sub("", text)
    else:
        return PRONOUN_RE.sub(replacement, text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rename and clean meta-data for all clinical EEG EDF files after mass export by Nihon Kohden.")
    parser.add_argument("--path", type=str, required=True, help="Path to all EDF files")
    parser.add_argument("--subject-code", type=str, required=True, help="Subject code (e.g., R1755A)")
    parser.add_argument("--subject-name", type=str, required=True, help="Subject name (e.g., John P. Smith)")
    parser.add_argument("--load-method", type=str, default="edfio", help="Method to load EDF files: 'edfio', 'pyedflib', or 'mne'")
    parser.add_argument("--raise-errors", action="store_true", help="Raise errors instead of warnings for debugging")
    parser.add_argument("--verbosity", type=int, default=1, help="Enable verbose output")
    args = parser.parse_args()

    print('Loading EDF files from path:', args.path)
    is_valid_subject_code(args.subject_code)

    # load the meta-data for each EEG EDF file
    EDF_meta_data = dict()
    for filename in os.listdir(args.path):
        try:
            if filename.lower().endswith('.edf'):
                full_path = os.path.join(args.path, filename)
                if args.verbosity > 0:
                    print(f"Loading {filename}...")
                data = load_edf(full_path, load_method=args.load_method, preload=True)

                EDF_meta_data[filename] = {'data': data}
        except Exception as e:
            if args.raise_errors:
                raise e
            print(f"ERROR: Failed to load EDF file {filename}:\n\n{e}\n\nCheck if the file is corrupted. Skipping this file...\n")
    
    # compute the relative start times of all recordings with respect to the earliest recording
    # remove absolute timing information to maintain privacy while preserving relative timing information
    start_times = list()
    for filename, edf in EDF_meta_data.items():
        data = edf['data']
        start_time = data['header']['startdate']
        if args.verbosity > 1:
            print(f"Start time for {filename}: {start_time}")
        start_times.append(start_time)
    min_start_time = min(start_times)
    if args.verbosity > -1:
        print(f"Earliest recording start time across all files: {min_start_time}")
    
    # de-identify EDF files and save out
    for filename, edf in EDF_meta_data.items():
        edf['data'] = deidentify_edf(edf_data=edf['data'],
                                     subject_name=args.subject_name,
                                     subject_code=args.subject_code,
                                     earliest_recording_start_time=min_start_time)
        clean_start_time = edf['data']['header']['startdate']
        clean_filename = f"{args.subject}_{clean_start_time.strftime('%Y.%m.%d__%H:%M:%S')}.edf"
        new_path = os.path.join(args.path, clean_filename)
        print(f"Saved cleaned EDF file as {clean_filename}")
    
    # for filename, edf in EDF_meta_data.items():
    #     data = edf['data']
    #     print(filename)
    #     print(data)
    #     print(data.info)
