import argparse


RESERVED_FIELD_EDF_HEADER_BYTE_OFFSET = 192


def load_edf(filename, load_method='pyedflib', preload=False, read_digital=False):
    """
    Load an EDF file using pyedflib or lunapi.

    Parameters:
        filename (str): Path to the EDF file.
        load_method (str): one of 'pyedflib' or 'lunapi'.
        preload (bool): If True, preload the EEG data into memory.
        read_digital (bool): If True, read digital signals if available.

    Returns:
        object: Loaded data object (depends on backend).
    """
    validate_edf_file_path(filename)
    if load_method == 'pyedflib':
        import pyedflib
        reader = pyedflib.EdfReader(filename)
        if preload:
            signals = [reader.readSignal(i, digital=read_digital) for i in range(reader.signals_in_file)]
        else:
            signals = None
        header = reader.getHeader()
        header['record_duration'] = reader.datarecord_duration
        header['n_records'] = reader.datarecords_in_file
        header['file_duration'] = reader.file_duration
        signal_headers = reader.getSignalHeaders()
        annotations = reader.readAnnotations()
        reader.close()
        return {'signals': signals, 'header': header, 'signal_headers': signal_headers, 'annotations': annotations}
    elif load_method == 'lunapi':
        import lunapi as lp
        proj = lp.proj()
        inst = proj.inst("rec1")
        inst.attach_edf(filename)
        return inst
    else:
        raise ValueError("Invalid load method specified. Use 'pyedflib' or 'lunapi'.")
    

def write_edf_pyedflib(data, filename, digital: bool = False):
    """
    Write EDF data using pyedflib.

    Parameters:
        data (dict): Data dictionary containing 'signals', 'header', 'signal_headers', and 'annotations'.
        filename (str): Path to save the EDF file.
        digital (bool): If True, ``data['signals']`` is expected to hold int16
            digital samples (i.e. matching what was read with ``read_digital=True``).
            Avoids the float64<->int16 round-trip and the associated memory overhead.
    """
    import warnings
    import pyedflib
    with pyedflib.EdfWriter(filename, len(data['signals']), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader(data['header'])
        # Lock the writer to the original file's record_duration (captured by
        # load_edf into header['record_duration']). Without this, pyedflib
        # auto-calculates a record_duration from the sample frequencies —
        # typically 1.0s — which usually does not divide the total signal
        # length evenly for Nihon Kohden files (whose record_duration is
        # 0.086s). The resulting output file's claimed size then disagrees
        # with the bytes pyedflib actually writes, and the reopened file
        # fails the filesize compliance check.
        record_duration = data['header'].get('record_duration')
        if record_duration is not None and 0.001 <= record_duration <= 60:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.setDatarecordDuration(record_duration)
        f.setSignalHeaders(data['signal_headers'])
        f.writeSamples(data['signals'], digital=digital)
        for time, duration, text in zip(*data['annotations']):
            f.writeAnnotation(time, duration, text)
    

def print_edf(data, load_method='pyedflib', verbosity=1):
    # Print the contents of an EDF file loaded by load_edf()
    if load_method == 'pyedflib':
        print_edf_pyedflib(data, verbosity)
    else:
        raise ValueError("Invalid load method specified. Use 'pyedflib'.")


def print_edf_pyedflib(data, verbosity=1):
    # Print the contents of an EDF file loaded by load_edf() with pyedflib
    if verbosity > 0:
        print('Header:')
        print(data['header'])
    if verbosity > 1:
        print('Example signal header:')
        print(data['signal_headers'][0])
    if verbosity > 2:
        print('Annotations:')
        n_annotations = len(data['annotations'][0])
        print(f'{n_annotations} annotations found:')
        print(data['annotations'])


def validate_edf_file_path(input_file: str) -> None:
    import os
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"EDF file not found: {input_file}")
    if not input_file.lower().endswith('.edf'):
        raise ValueError(f"File does not have .edf extension: {input_file}")


def get_edf_reserved_field(input_file: str) -> str:
    # 'reserved' header field in EDF+ standard must start with 'EDF+C' or 'EDF+D' for continuous or discontinuous files
    validate_edf_file_path(input_file)
    with open(input_file, 'rb') as f:
        f.seek(RESERVED_FIELD_EDF_HEADER_BYTE_OFFSET)
        reserved_field_bytes = f.read(44)
        reserved_field = reserved_field_bytes.decode('ascii')
    return reserved_field


def is_edfD(input_file: str) -> bool:
    reserved_field = get_edf_reserved_field(input_file)
    return 'EDF+D' == reserved_field[:5]


def is_edfC(input_file: str) -> bool:
    reserved_field = get_edf_reserved_field(input_file)
    return 'EDF+C' == reserved_field[:5]


def is_edf_plus(input_file: str) -> bool:
    reserved_field = get_edf_reserved_field(input_file)
    return 'EDF+' == reserved_field[:4]


def is_edf_continuous(input_file: str) -> bool:
    validate_edf_file_path(input_file)
    from clean_eeg.split_discontinuous_edf import luna_open_and_segments
    _, segments = luna_open_and_segments(input_file)
    return len(segments) == 1


def print_edf_file_type(input_file: str) -> None:
    if is_edfC(input_file):
        print(f"{input_file} is an EDF+C (continuous) file.")
    elif is_edfD(input_file):
        print(f"{input_file} is an EDF+D (discontinuous) file.")
    elif is_edf_plus(input_file):
        print(f"{input_file} 'reserved field' starts with 'EDF+' but "
              "not specifically 'EDF+C' or 'EDF+D' and is EDF+ non-compliant.")
    else:
        print(f"{input_file} is in EDF (not EDF+) format.")

    if is_edf_continuous(input_file):
        print(f"{input_file} contains continuous data.")
    else:
        print(f"{input_file} contains discontinuous data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load an EDF file using pyedflib or MNE.")
    parser.add_argument("--path", type=str, required=True, help="Path to EDF file")
    parser.add_argument("--load-method", type=str, default="pyedflib", help="Method to load EDF files: 'pyedflib' or 'lunapi'")
    parser.add_argument("--lazy-load", action="store_true", help="Preload EEG into memory")
    parser.add_argument("--raise-errors", action="store_true", help="Raise errors instead of warnings for debugging")
    parser.add_argument("--verbosity", type=int, default=1, help="Enable verbose output")
    args = parser.parse_args()

    data = load_edf(args.path, load_method=args.load_method, preload=not args.lazy_load)
    print(f"Loaded EDF file: {args.path} using method: {args.load_method}")

    print_edf(data, load_method=args.load_method, verbosity=args.verbosity)
    print_edf_file_type(args.path)
