import re
import os
from datetime import datetime

from load_edf import load_edf, get_edf_start_time_from_mne


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rename and clean meta-data for all clinical EEG EDF files after mass export by Nihon Kohden.")
    parser.add_argument("--path", type=str, required=True, help="Path to all EDF files")
    parser.add_argument("--subject", type=str, required=True, help="Subject code (e.g., R1755A)")
    parser.add_argument("--load-method", type=str, default="edfio", help="Method to load EDF files: 'edfio', 'pyedflib', or 'mne'")
    parser.add_argument("--raise-errors", action="store_true", help="Raise errors instead of warnings for debugging")
    parser.add_argument("--verbosity", type=int, default=1, help="Enable verbose output")
    args = parser.parse_args()

    print('Loading EDF files from path:', args.path)
    is_valid_subject_code(args.subject)

    # load the meta-data for each EEG EDF file
    EDF_meta_data = dict()
    for filename in os.listdir(args.path):
        try:
            if filename.lower().endswith('.edf'):
                full_path = os.path.join(args.path, filename)
                if args.verbosity > 0:
                    print(f"Loading {filename}...")
                data = load_edf(full_path, load_method=args.load_method, preload=True)

                # check if the EDF file is in continuous or discontinuous format
                # assert that EDF file is contiguous even if in discontinuous format
                if args.load_method == 'edfio':
                    print(data)
                    print(data.__dir__())
                    import pdb; pdb.set_trace()
                    raise NotImplementedError("Loading EDF files with edfio is not fully implemented yet.")
                elif args.load_method == 'mne':
                    import mne
                    if isinstance(data, mne.io.edf.edf.RawEDF):
                        if data._data.shape[1] == 0:
                            raise ValueError(f"Empty data in {filename}.")
                    else:
                        raise ValueError(f"Unsupported data format for {filename}: {type(data)}.")
                EDF_meta_data[filename] = {'data': data}
        except Exception as e:
            if args.raise_errors:
                raise e
            print(f"ERROR: Failed to load EDF file {filename}:\n\n{e}\n\nCheck if the file is corrupted. Skipping this file...\n")
    
    # compute the relative start times of all recordings with respect to the earliest recording
    start_times = list()
    for filename, edf in EDF_meta_data.items():
        data = edf['data']
        if args.load_method == 'pyedflib':
            start_time = data['header']['starttime']
        elif args.load_method == 'mne':
            start_time = get_edf_start_time_from_mne(data)
        else:
            raise ValueError(f"Unsupported load method: {args.load_method}. Use 'pyedflib' or 'mne'.")
        if args.verbosity > 1:
            print(f"Start time for {filename}: {start_time}")
        edf['start_time'] = start_time
        start_times.append(start_time)
    if start_times:
        min_start_time = min(start_times)
        if args.verbosity > 0:
            print(f"Earliest recording start time across all files: {min_start_time}")
        for filename, edf in EDF_meta_data.items():
            edf['relative_start_time'] = edf['start_time'] - min_start_time
            if args.verbosity > 0:
                print(f"Start time relative to earliest recording for {filename}: {edf['relative_start_time']}")
    
    for filename, edf in EDF_meta_data.items():
        data = edf['data']
        print(filename)
        print(data)
        print(data.info)

    # Save out EDFs with meta-data cleaned of protected health information (PHI):
    # 1) rename subject to args.subject and remove meta-data fields for gender, birthdate, patient hospital code
    # 2) replace recording start time with time relative to the earliest recording start time
    # 3) remove any recording annotations containing regex patterns indicating PHI (name, gender, birthdate)
    # 4) save the modified EDF file with a new name in the format SUBJECT_CODE__RELATIVE.START.DATE_RELATIVE:START:TIME.edf
    #        RELATIVE.START.DATE_RELATIVE:START:TIME corresponds to YEAR.MONTH.DAY__HOUR:MINUTE:SECOND relative to the earliest recording start time
    #        relative times are offset by the EDF standard clipping date of 1985-01-01
    base_start_date = datetime.datetime(1985, 1, 1)

    for filename, edf in EDF_meta_data.items():
        data = edf['data']
        if args.load_method == 'pyedflib':
            # For pyedflib, we would need to implement saving logic here
            raise NotImplementedError("Saving modified EDF files with pyedflib is not implemented.")
        elif args.load_method == 'mne':
            # For MNE, we can save the modified raw object
            edf['shifted_start_date_time'] = edf['relative_start_time'] + base_start_date
            new_filename = f"{args.subject}_{edf['shifted_start_date_time'].strftime('%Y.%m.%d__%H:%M:%S')}.edf"
            new_path = os.path.join(args.path, new_filename)
            # data.save(new_path, overwrite=True)
            # print(f"Saved cleaned EDF file as {new_filename}")
        else:
            raise ValueError(f"Unsupported load method: {args.load_method}. Use 'pyedflib' or 'mne'.")
