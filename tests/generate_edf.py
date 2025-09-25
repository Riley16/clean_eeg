import numpy as np
import os
import pyedflib
from datetime import datetime
import lunapi as lp

from clean_eeg.paths import TEST_SUBJECT_DATA_DIR
from clean_eeg.load_eeg import RESERVED_FIELD_EDF_HEADER_BYTE_OFFSET


def generate_test_edf(header, signal_headers, filename='sinusoidal_continuous_file.edf'):
    # generate EDF files for testing that include a few sinusoidal oscillations, values for all header meta-data, and a few annotations

    # generate a sinusoidal signal
    sample_rate_hz = signal_headers[0]['sample_frequency']
    duration_s = 5
    time_points = np.arange(0, duration_s, 1/sample_rate_hz)
    signals = list()
    
    n_signals = len(signal_headers)
    for i in range(n_signals):
        frequency = (i + 1) - 0.5
        signal = np.sin(2 * np.pi * frequency * time_points)  
        signals.append(signal)

    with pyedflib.EdfWriter(file_name=str(filename),
                            n_channels=n_signals,
                            file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader(header)
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)

        annotations = [
            (0.5, -1, "SEGMENT 1"),
            (1.0, -1, "SEGMENT 2"),
            (1.5, -1, "SEGMENT 3"),
            (3.5, -1, "SEGMENT 4"),
            (4.5, -1, "SEGMENT 5"),
        ]
        for time, duration, text in annotations:
            f.writeAnnotation(time, duration, text)


def generate_test_edf_from_config(config_dict, path, n_signals=2):
    """
    Generate a test EDF file based on the provided configuration dictionary.

    Parameters:
        config_dict (dict): Configuration dictionary containing 'header' and 'signal_headers'.

    Returns:
        str: Path to the generated EDF file.
    """
    formatted_config = format_edf_config_json(config_dict)
    header = formatted_config['header']
    signal_headers = formatted_config['signal_headers']
    
    if isinstance(signal_headers, dict):
        base_signal_header = signal_headers
        signal_headers = list()
        for _ in range(n_signals):
            signal_header = base_signal_header.copy()
            signal_header['label'] += f"_{len(signal_headers) + 1}"
            signal_headers.append(signal_header)
    else:
        assert isinstance(signal_headers, list), "signal_headers should be a list (of signal header dicts) or a single signal header dict to be duplicated."
    
    generate_test_edf(header, signal_headers, path)
    return None

def generate_discontinuous_edf_from_config(edf_config):
    assert edf_config['type'] == 'drop_intervals'

    proj = lp.proj()
    inst = proj.inst('temp')
    epoch_length_s = 1
    input_edf_path = str(TEST_DATA_DIR / edf_config['input_file'])
    inst.attach_edf(input_edf_path)
    inst.proc(f"EPOCH len={epoch_length_s}")
    luna_mask_command = edf_config['luna_mask_command']
    path = str(TEST_DATA_DIR / edf_config['filename'])
    file_no_extension = os.path.splitext(path)[0]
    if luna_mask_command:
        inst.eval(f"{luna_mask_command} & RE & WRITE edf={file_no_extension} EDF+D")
    else:
        inst.eval(f"WRITE edf={file_no_extension} EDF+D")
        # lunapi does not write continuous files with "EDF+D" in the file 
        # version header field even with the "EDF+D" option to force EDF+D, so overwrite the field manually
        with open(path, 'r+b') as f:
            f.seek(RESERVED_FIELD_EDF_HEADER_BYTE_OFFSET)
            f.write(b'EDF+D   ')

def format_edf_config_json(config_json):
    """
    Format the EDF configuration JSON to match the expected structure for pyedflib.

    Parameters:
        config_json (dict): The configuration JSON containing header and signal headers.

    Returns:
        dict: Formatted configuration with 'header' and 'signal_headers'.
    """
    formatted_config = {
        'header': config_json.get('pyedflib_header'),
        'signal_headers': config_json.get('pyedflib_signal_headers')
    }

    startdate = formatted_config['header']['startdate']
    timestamp_format = config_json['timestamp_format']
    formatted_config['header']['startdate'] = datetime.strptime(startdate, timestamp_format)
    return formatted_config


if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser(description="Generate simple EDF files for testing purposes.")
    parser.add_argument("--output", type=str, default='', help="Output path for the generated EDF file")
    args = parser.parse_args()

    from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE

    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)

    with open(TEST_CONFIG_FILE, 'r') as f:
        test_config = json.load(f)
    edf_config = test_config.get('basic_EDF+C')

    if args.output:
        print('Outputting basic EDF+C test file only... (do not specify --output to generate all test EDF files)')
        generate_test_edf_from_config(edf_config, args.output)
        print(f"EDF file generated at: {args.output}")
    else:
        print('Outputting all EDF test files...')

        path = TEST_DATA_DIR / edf_config['filename']
        generate_test_edf_from_config(edf_config, path=path)
        print(f"EDF file generated at: {path}")

        # merge existing EDF files into a test discontinuous EDF+D file
        generate_discontinuous_edf_from_config(edf_config=test_config.get('basic_EDF+D'))

        generate_discontinuous_edf_from_config(edf_config=test_config.get('continuous_EDF+D'))

        # generate subject-specific test EDF files representing multiple recordings from the same subject
        if not os.path.exists(TEST_SUBJECT_DATA_DIR):
            os.makedirs(TEST_SUBJECT_DATA_DIR)

        for config_key in ['subject_EDF+C_1', 'subject_EDF+C_2']:
            edf_config = test_config.get(config_key)
            path = TEST_SUBJECT_DATA_DIR / edf_config['filename']
            generate_test_edf_from_config(edf_config, path=path)

