import numpy as np
import os
import pyedflib
from datetime import datetime
import lunapi as lp

from clean_eeg.paths import TEST_DATA_DIR, TEST_SUBJECT_DATA_DIR, INCONSISTENT_SUBJECT_DATA_DIR
from clean_eeg.load_eeg import RESERVED_FIELD_EDF_HEADER_BYTE_OFFSET

DEFAULT_NUMBER_SIGNALS = 2


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


def generate_test_edf_from_config(config_dict, path, n_signals=DEFAULT_NUMBER_SIGNALS):
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

def generate_partial_record_edf(output_path, n_channels=2, sample_rate=100,
                                duration_sec=10, file_type=pyedflib.FILETYPE_EDFPLUS):
    """
    Generate an EDF file with a truncated final data record.

    Creates a valid EDF, then removes half the bytes of the last data record
    to simulate an NK export with a partial final recording block.
    The header's num_data_records field is left unchanged (claims more records
    than are fully present on disk).

    Returns (full_path, partial_path) where full_path is the intact reference.
    """
    full_path = str(output_path) + '.full_ref.edf'

    writer = pyedflib.EdfWriter(full_path, n_channels, file_type=file_type)
    for i in range(n_channels):
        writer.setSignalHeader(i, {
            'label': f'CH{i}', 'dimension': 'uV',
            'sample_frequency': sample_rate,
            'physical_max': 3200, 'physical_min': -3200,
            'digital_max': 32767, 'digital_min': -32768,
        })
    signals = [np.array([float(j + i * 1000 + 1)
                         for j in range(sample_rate * duration_sec)])
               for i in range(n_channels)]
    writer.writeSamples(signals)
    writer.close()

    # Read header geometry
    with open(full_path, 'rb') as f:
        f.seek(184)
        header_size = int(f.read(8).decode('ascii').strip())
        f.seek(236)
        n_records = int(f.read(8).decode('ascii').strip())

    full_size = os.path.getsize(full_path)
    bytes_per_record = (full_size - header_size) // n_records

    # Create truncated copy (remove half of last data record)
    import shutil
    partial_path = str(output_path)
    shutil.copy2(full_path, partial_path)
    with open(partial_path, 'r+b') as f:
        f.truncate(full_size - bytes_per_record // 2)

    return full_path, partial_path


def format_edf_config_json(config_json):
    """
    Format the EDF configuration JSON to match the expected structure for pyedflib.

    Parameters:
        config_json (dict): The configuration JSON containing header and signal headers.

    Returns:
        dict: Formatted configuration with 'header' and 'signal_headers'.
    """
    formatted_config = {
        'header': config_json['pyedflib_header'],
        'signal_headers': config_json['pyedflib_signal_headers']
    }
    startdate = formatted_config['header']['startdate']
    timestamp_format = config_json['timestamp_format']
    formatted_config['header']['startdate'] = datetime.strptime(startdate, timestamp_format)
    return formatted_config


def run_generate_test_edf(output=''):
    import json
    from clean_eeg.paths import TEST_DATA_DIR, TEST_CONFIG_FILE

    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)

    with open(TEST_CONFIG_FILE, 'r') as f:
        test_config = json.load(f)
    edf_config = test_config.get('basic_EDF+C')

    if output:
        print('Outputting basic EDF+C test file only... (do not specify --output to generate all test EDF files)')
        generate_test_edf_from_config(edf_config, output)
        print(f"EDF file generated at: {output}")
    else:
        print('Outputting all EDF test files...')

        path = TEST_DATA_DIR / edf_config['filename']
        generate_test_edf_from_config(edf_config, path=path)
        print(f"EDF file generated at: {path}")

        config_key = 'basic_EDF+C_modified'
        edf_config = test_config.get(config_key)
        path = TEST_DATA_DIR / edf_config['filename']
        generate_test_edf_from_config(edf_config, path=path)

        # merge existing EDF files into a test discontinuous EDF+D file
        generate_discontinuous_edf_from_config(edf_config=test_config.get('basic_EDF+D'))
        generate_discontinuous_edf_from_config(edf_config=test_config.get('continuous_EDF+D'))

        # generate subject-specific test EDF files representing multiple recordings from the same subject
        if not os.path.exists(TEST_SUBJECT_DATA_DIR):
            os.makedirs(TEST_SUBJECT_DATA_DIR)

        for config_key in ['subject_EDF+C_1',
                           'subject_EDF+C_2']:
            edf_config = test_config.get(config_key)
            path = TEST_SUBJECT_DATA_DIR / edf_config['filename']
            generate_test_edf_from_config(edf_config, path=path)

        if not os.path.exists(INCONSISTENT_SUBJECT_DATA_DIR):
            os.makedirs(INCONSISTENT_SUBJECT_DATA_DIR)

        for config_key in ['inconsistent_subject_EDF+C_1', 'inconsistent_subject_EDF+C_2']:
            edf_config = test_config.get(config_key)
            path = INCONSISTENT_SUBJECT_DATA_DIR / edf_config['filename']
            generate_test_edf_from_config(edf_config, path=path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simple EDF files for testing purposes.")
    parser.add_argument("--output", type=str, default='', help="Output path for the generated EDF file")
    args = parser.parse_args()
    run_generate_test_edf(output=args.output)
