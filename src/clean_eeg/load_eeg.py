import argparse
import datetime


def load_edf(filename, load_method='edfio', preload=False, read_digital=False):
    """
    Load an EDF file using either pyedflib, edfio, or MNE.

    Parameters:
        filename (str): Path to the EDF file.
        load_method (str): one of 'edfio', 'pyedflib', or 'mne'.
        preload (bool): If True, preload the EEG data into memory.
        read_digital (bool): If True, read digital signals if available.

    Returns:
        object: Loaded data object (depends on backend).
    """
    if load_method == 'edfio':
        import edfio
        edf = edfio.read_edf(filename)
        return edf
    elif load_method == 'pyedflib':
        import pyedflib
        reader = pyedflib.EdfReader(filename)
        if preload:
            signals = [reader.readSignal(i, digital=read_digital) for i in range(reader.signals_in_file)]
        else:
            signals = None
        header = reader.getHeader()
        signal_headers = reader.getSignalHeaders()
        annotations = reader.readAnnotations()
        reader.close()
        return {'signals': signals, 'header': header, 'signal_headers': signal_headers, 'annotations': annotations}
    elif load_method == 'mne':
        import mne
        raw = mne.io.read_raw_edf(filename, preload=preload, verbose='error')
        return raw
    else:
        raise ValueError("Invalid load method specified. Use 'edfio', 'pyedflib', 'mne'.")
    

def print_edf(data, load_method='edfio', verbosity=1):
    # Print the contents of an EDF file loaded by load_edf()
    if load_method == 'edfio':
        print_edf_edfio(data, verbosity)
    elif load_method == 'pyedflib':
        print_edf_pyedflib(data, verbosity)
    elif load_method == 'mne':
        print_edf_mne(data, verbosity)
    else:
        raise ValueError("Invalid load method specified. Use 'edfio', 'pyedflib', 'mne'.")


def print_edf_edfio(data, verbosity=1):
    # Print the contents of an EDF file loaded by load_edf() with edfio
    if verbosity > 0:
        print('Header info:')
        print(data.patient)
        print(data.recording)
    if verbosity > 1:
        print('Example signal header:')
        print(data.signals[0].__dict__)
    if verbosity > 2:
        print('Annotations:')
        print(data.annotations)


def print_edf_pyedflib(data, verbosity=1):
    # Print the contents of an EDF file loaded by load_edf() with pyedflib
    if verbosity > 0:
        print('Header info:')
        print(data['header'])
    if verbosity > 1:
        print('Example signal header:')
        print(data['signal_headers'][0])
    if verbosity > 2:
        print('Annotations:')
        n_annotations = len(data['annotations'][0])
        print(f'{n_annotations} annotations found:')
        print(data['annotations'])


def print_edf_mne(data, verbosity=1):
    # Print the contents of an EDF file loaded by load_edf() with MNE
    if verbosity > 0:
        print('Header info:')
        print(data.info)
    if verbosity > 1:
        print('Example signal header:')
        print(data.ch_names)
    if verbosity > 2:
        print('Annotations:')
        print(data.annotations)


def get_edf_start_time_from_mne(raw):
    """
    Get the start time of the EDF file using MNE.

    Parameters:
        raw (mne.io.Raw): MNE Raw object.

    Returns:
        float: Start time in seconds.
    """
    return raw.info['meas_date']


def offset_edf_start_time_first_recording(raw, offset_seconds):
    """
    Offset the start time of the first recording in an EDF file.

    Parameters:
        raw (mne.io.Raw): MNE Raw object.
        offset_seconds (float): Offset in seconds.

    Returns:
        mne.io.Raw: Modified MNE Raw object with updated start time.
    """
    new_start_time = get_edf_start_time_from_mne(raw) + offset_seconds
    raw.info['meas_date'] = datetime.datetime.fromtimestamp(new_start_time)
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load an EDF file using pyedflib or MNE.")
    parser.add_argument("--path", type=str, required=True, help="Path to EDF file")
    parser.add_argument("--load-method", type=str, default="edfio", help="Method to load EDF files: 'edfio', 'pyedflib', or 'mne'")
    parser.add_argument("--lazy-load", action="store_true", help="Preload EEG into memory")
    parser.add_argument("--raise-errors", action="store_true", help="Raise errors instead of warnings for debugging")
    parser.add_argument("--verbosity", type=int, default=1, help="Enable verbose output")
    args = parser.parse_args()

    data = load_edf(args.path, load_method=args.load_method, preload=not args.lazy_load)
    print(f"Loaded EDF file: {args.path} using method: {args.load_method}")

    print_edf(data, load_method=args.load_method, verbosity=args.verbosity)
