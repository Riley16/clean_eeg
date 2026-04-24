import argparse


RESERVED_FIELD_EDF_HEADER_BYTE_OFFSET = 192


def load_edf(filename, load_method='pyedflib', preload=False, read_digital=False,
             use_mmap=False):
    """
    Load an EDF file using pyedflib or lunapi.

    Parameters:
        filename (str): Path to the EDF file.
        load_method (str): one of 'pyedflib' or 'lunapi'.
        preload (bool): If True, preload the EEG data into memory.
        read_digital (bool): If True, read digital signals if available.
        use_mmap (bool): If True (and ``preload=True, read_digital=True``),
            use the record-based mmap signal reader instead of pyedflib's
            per-channel loop. Orders of magnitude faster on multi-GB NK
            files because it avoids pyedflib's interleaved-layout seeks.
            Headers / signal_headers / annotations still come from pyedflib.
            Ignored when ``preload=False`` or ``read_digital=False``
            (float / physical conversion is not supported by the mmap path).

    Returns:
        object: Loaded data object (depends on backend).
    """
    validate_edf_file_path(filename)
    if load_method == 'pyedflib':
        import pyedflib
        reader = pyedflib.EdfReader(filename)
        if preload:
            if use_mmap and read_digital:
                # Fast path — mmap-based record-deinterleaving reader.
                # Returns byte-identical int32 arrays to readSignal(digital=True).
                # Helper reads on-disk geometry (including annotation channels)
                # directly from the header bytes, then filters annotations out
                # of the output so the return value matches pyedflib's
                # signals_in_file count and ordering.
                #
                # Any failure (unexpected layout, truncation missed by
                # pyedflib, OS mmap limit, etc.) falls back to pyedflib's
                # per-channel readSignal loop with a warning, so the
                # pipeline never regresses below the baseline behaviour.
                try:
                    signals = _read_signals_via_mmap(filename)
                except Exception as e:
                    print(
                        f"WARNING: mmap signal loader failed for {filename}: "
                        f"{type(e).__name__}: {e}. "
                        "Falling back to pyedflib's per-channel readSignal loop "
                        "(slower but widely tested)."
                    )
                    signals = [reader.readSignal(i, digital=read_digital)
                               for i in range(reader.signals_in_file)]
            else:
                signals = [reader.readSignal(i, digital=read_digital)
                           for i in range(reader.signals_in_file)]
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


def _read_signals_via_mmap(filename: str) -> list:
    """Read all non-annotation signals from an EDF file via a single mmap pass.

    Returns a list of ``np.int32`` arrays — one per data (non-annotation)
    signal, in header order — matching the output of
    ``[pyedflib.EdfReader(filename).readSignal(i, digital=True) for i in
    range(signals_in_file)]`` byte-for-byte.

    Why this exists: EDF stores samples in an interleaved per-record
    layout (record 0: [sig0, sig1, ..., sigN], record 1: [sig0, sig1,
    ..., sigN], ...). pyedflib's ``readSignal(i)`` walks the file once
    per channel, which means ``N_signals * N_records`` small disk
    seeks to gather each channel's scattered samples. For a 3.8 GB NK
    file (~178 ch, ~62k records) that's ~11 million tiny I/O ops —
    minutes of wall time even on fast SSDs.

    This helper mmap's the file once, reshapes the data region to
    ``(n_records, record_samples)`` as a zero-copy ``int16`` view, and
    slices each signal's columns out. OS page cache handles the actual
    disk reads as one big sequential scan — seconds instead of minutes.

    The function parses on-disk header geometry itself (n_signals,
    n_records, samples_per_record, labels) from the EDF header bytes.
    It does NOT reuse pyedflib's ``signals_in_file``, because that
    count excludes EDF+ Annotations channels while the on-disk layout
    includes them — using the pyedflib count would miss the
    annotations-channel bytes in every record and corrupt every read.

    Annotation channels are identified by the EDF+ label ``"EDF
    Annotations"`` and filtered out of the returned list so the output
    matches pyedflib's ``readSignal`` contract.

    Output dtype is ``np.int32`` to stay drop-in compatible with
    ``readSignal(digital=True)``; the underlying data is int16 on disk
    and in the mmap, the int32 upcast is the only copy.
    """
    import mmap
    import os
    import numpy as np

    # --- parse on-disk geometry straight from the header bytes ---
    with open(filename, "rb") as f:
        main = f.read(256)
        n_signals = int(main[252:256].decode().strip())
        n_records = int(main[236:244].decode().strip())
        sig_header = f.read(256 * n_signals)

    if n_signals <= 0:
        return []
    if n_records <= 0:
        return []

    # Per-signal fields are stored FIELD-by-FIELD across all signals,
    # not per-signal. i.e. all N labels first, then all N transducers,
    # ..., then all N samples_per_record, then all N reserved.
    LABEL_OFFSET = 0
    LABEL_WIDTH = 16
    SPR_OFFSET = 216   # cumulative after label+transducer+dim+pmin+pmax+dmin+dmax+prefilter
    SPR_WIDTH = 8
    labels = []
    samples_per_record = []
    for i in range(n_signals):
        lab_b = sig_header[LABEL_OFFSET * n_signals + i * LABEL_WIDTH:
                           LABEL_OFFSET * n_signals + (i + 1) * LABEL_WIDTH]
        spr_b = sig_header[SPR_OFFSET * n_signals + i * SPR_WIDTH:
                           SPR_OFFSET * n_signals + (i + 1) * SPR_WIDTH]
        labels.append(lab_b.decode("ascii", errors="replace").rstrip())
        try:
            samples_per_record.append(int(spr_b.decode("ascii").strip()))
        except ValueError as e:
            raise ValueError(
                f"Unparseable samples_per_record for signal {i} "
                f"({lab_b!r}): {e}"
            ) from e

    if any(spr <= 0 for spr in samples_per_record):
        raise ValueError(
            f"samples_per_record contains non-positive value(s): "
            f"{samples_per_record}"
        )

    record_samples = sum(samples_per_record)
    total_samples = n_records * record_samples
    header_bytes = 256 * (1 + n_signals)

    file_size = os.path.getsize(filename)
    expected_data_bytes = total_samples * 2
    if file_size < header_bytes + expected_data_bytes:
        raise ValueError(
            f"File {filename} is smaller than expected: {file_size} bytes "
            f"< header({header_bytes}) + data({expected_data_bytes})."
        )

    # --- mmap and de-interleave all signals ---
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data = np.frombuffer(
                mm,
                dtype=np.int16,
                count=total_samples,
                offset=header_bytes,
            )
            records = data.reshape(n_records, record_samples)

            all_signals = []
            col_offset = 0
            for i in range(n_signals):
                spr = samples_per_record[i]
                # .astype(int32, copy=True) allocates a fresh buffer;
                # subsequent .ravel() returns a 1D view into that new
                # buffer, not into mmap.
                sig = records[:, col_offset:col_offset + spr].astype(
                    np.int32, copy=True)
                all_signals.append(sig.ravel())
                col_offset += spr

            # Drop mmap-backed views before the mmap context exits.
            # mmap.__exit__ raises BufferError if any ndarray still
            # references the buffer, even when all *data* has already
            # been copied into standalone arrays.
            del records
            del data

    # --- filter out annotation channels to match pyedflib's output ---
    data_signals = [
        sig for sig, lab in zip(all_signals, labels)
        if lab.strip().lower() != "edf annotations"
    ]
    return data_signals
    

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
