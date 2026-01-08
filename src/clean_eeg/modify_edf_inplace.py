# edf_inplace.py
from typing import Dict, List, Union
from copy import deepcopy
import datetime
import os
import re
import numpy as np
import pyedflib


def update_edf_header_inplace(edf_path: str,
                              header_updates: dict,
                              signal_header_updates: Union[List[Dict], None] = None,
                              confirm_signals_unchanged: bool = False,
                              verbosity: int = 0):
    """Update EDF header fields in place without modifying signal data.
    
    Creates a temporary EDF file with updated header using pyedflib, then copies
    the updated header bytes back to the original file at the correct position.
    
    Args:
        edf_path: Path to the EDF file to update
        header_updates: Dictionary with same format as pyedflib header dict.
                        None values are treated as no-ops.
        signal_header_updates: List with same format as pyedflib signal header list. 
                        None values in signal dicts are no-ops.
        confirm_signals_unchanged: If True, verify that signal data remains 
                                 unchanged after header update
        verbosity: Verbosity level for output (0 = silent, >0 = verbose)
    
    Raises:
        ValueError: If confirm_signals_unchanged is True and signal data changes
    """
    # Read original header
    with pyedflib.EdfReader(edf_path) as f:
        orig_header = f.getHeader()
        orig_signal_headers = f.getSignalHeaders()
        if confirm_signals_unchanged:
            # load original signals for later comparison
            orig_signals = [f.readSignal(i) for i in range(f.signals_in_file)]
    
    # Apply updates to header fields
    updated_header = orig_header.copy()
    for field, value in header_updates.items():
        if value is not None:
            updated_header[field] = value
    # updated_header = clean_header(updated_header, raise_errors=False)

    if signal_header_updates is not None:
        if len(signal_header_updates) != len(orig_signal_headers):
            raise ValueError(f"Length of signal_header_updates ({len(signal_header_updates)}) must "
                                f"match number of signals in EDF ({len(orig_signal_headers)})")
        updated_signal_headers = list()
        for i, orig_sig_header in enumerate(orig_signal_headers):
            sig_header_update = signal_header_updates[i]
            updated_signal_headers.append(dict(orig_sig_header))
            for field, value in sig_header_update.items():
                if value is not None:
                    updated_signal_headers[i][field] = value
    
    # Write new EDF with updated header but no data
    temp_path = edf_path + ".tmp"
    with pyedflib.EdfWriter(temp_path, len(orig_signal_headers)) as f:
        if verbosity > 0:
            print("updated header written to temp EDF:")
            print(updated_header)
        f.setHeader(updated_header)
        if signal_header_updates is not None:
            f.setSignalHeaders(updated_signal_headers)

    # fix header fields since pyedflib only updates headers for signal
    # info once signals (but not signal headers) are added
    copy_overwrite_fields = ['header_bytes',
                             'num_signals',
                             'num_data_records',
                             'data_record_duration']
    for field in copy_overwrite_fields:
        copy_bytes(edf_path, temp_path,
                   *EDF_HEADER_FIELD_OFFSETS_LENGTHS[field][:2])

    # Copy updated header bytes back to original file
    with open(edf_path, "r+b") as orig_file, open(temp_path, "rb") as temp_file:
        n_copy_bytes = TOTAL_HEADER_BYTES
        if signal_header_updates is not None:
            n_copy_bytes += len(orig_signal_headers) * SIGNAL_HEADER_BYTES
        updated_header_bytes = temp_file.read(n_copy_bytes)
        orig_file.seek(0)
        orig_file.write(updated_header_bytes)

    if confirm_signals_unchanged:
        with pyedflib.EdfReader(edf_path) as f:
            for i in range(f.signals_in_file):
                updated_signal = f.readSignal(i)
                if not all(updated_signal == orig_signals[i]):
                    raise ValueError(f"Signal {i} changed after in-place header update")

    os.remove(temp_path)


# TODO account for multiple data fields in single header field
def clean_header(header: dict, raise_errors: bool) -> dict:
    header = deepcopy(header)
    str_fields = ['technician', 'recording_additional',
                  'patientname', 'patient_additional',
                  'patientcode', 'equipment', 'admincode',
                  'sex', 'birthdate', 'gender']
    for field in str_fields:
        assert field in header
        if not isinstance(header[field], str):
            raise ValueError(f"Header field '{field}' must be a string")
        max_length = EDF_HEADER_FIELD_OFFSETS_LENGTHS[field][1]
        if len(header[field]) > max_length:
            message = f"Header field '{field}' exceeds max length of {max_length} characters."
            if raise_errors:
                raise ValueError(message)
            print(message + " Truncating...")
            header[field] = header[field][:max_length]
    
    assert 'startdate' in header
    if not isinstance(header['startdate'], datetime.date):
        raise ValueError("Header field 'startdate' must be a datetime object")
    
    return header


def create_annotations_only_edf(path: str,
                                header: dict,
                                annotations: tuple[List[float], List[float], List[str]],
                                validate: bool = True) -> None:
    """Create a minimal EDF file containing only annotations."""
    with pyedflib.EdfWriter(file_name=path,
                            n_channels=0,
                            file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setHeader(header)
        for time, duration, text in zip(*annotations):
            f.writeAnnotation(time, duration, text)

    if validate:
        with pyedflib.EdfReader(path) as f:
            annotations_rewrite = f.readAnnotations()
            assert np.all(annotations_rewrite[0] == annotations[0]), "Annotation onsets mismatch after rewrite"
            assert np.all(annotations_rewrite[1] == annotations[1]), "Annotation durations mismatch after rewrite"
            assert np.all(np.logical_or(annotations_rewrite[2] == annotations[2], 
                                        [bool(re.match(r'^[Xx]+$', ann)) for ann in annotations_rewrite[2]])), "Annotation texts mismatch after rewrite"
            header_rewrite = f.getHeader()
            # Compare headers field by field, allowing 'x*' patterns to match empty strings
            ignore_fields = ['record_duration', 'n_records', 'file_duration']
            for field in header:
                if field in ignore_fields:
                    continue
                original_value = header[field]
                rewrite_value = header_rewrite[field]
                
                # For string fields, allow 'x*' pattern to match empty string
                if isinstance(original_value, str) and isinstance(rewrite_value, str):
                    if bool(re.match(r'^[Xx]+$', original_value)) and rewrite_value == '':
                        continue
                
                if original_value != rewrite_value:
                    raise ValueError(f"Header field '{field}' mismatch: original='{original_value}', rewrite='{rewrite_value}'")


def clear_edf_annotations_inplace(path, validate: bool = True):
    # blank out EDF annotation texts in-inplace

    ann_signal_index = get_annotation_signal_header_index(path)
    signal_record_lengths = get_signal_header_fields(path, field='num_samples')
    for i in range(len(signal_record_lengths)):
        signal_record_lengths[i] *= 2  # 2 bytes per sample
    ann_record_length = signal_record_lengths[ann_signal_index]
    ann_record_offset = sum(signal_record_lengths[:ann_signal_index])
    total_records_length = sum(signal_record_lengths)

    n_signals = len(signal_record_lengths)
    n_records = get_header_field(path, 'num_data_records')

    # get annotation record bytes
    with open(path, "r+b") as f:
        for i in range(n_records):
            annotation_record_offset = TOTAL_HEADER_BYTES + SIGNAL_HEADER_BYTES * n_signals + total_records_length * i + ann_record_offset
            f.seek(annotation_record_offset)
            ann_record_bytes = f.read(ann_record_length)

            # blank out annotation texts after time-keeping annotations
            # first annotation after time-keeping must be empty, so 2 x14 bytes in a row separate 
            # time-keeping from first annotation in each record
            EDF_TAL_TIMEKEEPING_DELIMITER = b'\x14\x14'
            time_keeping_offset = ann_record_bytes.find(EDF_TAL_TIMEKEEPING_DELIMITER) + len(EDF_TAL_TIMEKEEPING_DELIMITER)
            assert time_keeping_offset >= 4, "No time-keeping annotation found"
            blanked_record_bytes = (ann_record_bytes[:time_keeping_offset] +
                                    b'\x00' * (len(ann_record_bytes) - time_keeping_offset))
            f.seek(annotation_record_offset)
            f.write(blanked_record_bytes)
    
    # confirm the resulting EDF still loads and has no text annotations
    if validate:
        with pyedflib.EdfReader(path) as f:
            _, _, ann_texts = f.readAnnotations()
            assert len(ann_texts) == 0, "Annotations found after clearing"



TOTAL_HEADER_BYTES = 256
SIGNAL_HEADER_BYTES = 256

# Byte offsets and lengths for EDF header fields
EDF_HEADER_FIELD_OFFSETS_LENGTHS = {
    'version': (0, 8, int),
    'patient_id': (8, 80, str),
    'recording_id': (88, 80, str),
    'startdate': (168, 8, str),
    'starttime': (176, 8, str),
    'header_bytes': (184, 8, int),
    'reserved': (192, 44, str),
    'num_data_records': (236, 8, int),
    'data_record_duration': (244, 8, int),
    'num_signals': (252, 4, int),
}


# EDF signal headers are organized into contiguous blocks for each field across all signals
EDF_SIGNAL_HEADER_FIELD_OFFSETS_LENGTHS = {
    'label': (0, 16, str),
    'transducer_type': (16, 80, str),
    'physical_dimension': (96, 8, str),
    'physical_min': (104, 8, int),
    'physical_max': (112, 8, int),
    'digital_min': (120, 8, int),
    'digital_max': (128, 8, int),
    'prefiltering': (136, 80, str),
    'num_samples': (216, 8, int),
    'reserved': (224, 32, None),
}


def get_header_field(edf_path, field: str, return_raw_bytes: bool = False):
    with open(edf_path, "rb") as f:
        offset, length, field_type = EDF_HEADER_FIELD_OFFSETS_LENGTHS[field]
        f.seek(offset)
        field_bytes = f.read(length)
    if return_raw_bytes:
        return field_bytes
    value = format_field_from_bytes(field_bytes, field_type)
    return value


def get_signal_header_fields(edf_path, field: str):
    # load all signal header fields by raw bytes, including annotation signal
    with pyedflib.EdfReader(edf_path) as reader:
        n_signals = reader.signals_in_file

    field_attributes = EDF_SIGNAL_HEADER_FIELD_OFFSETS_LENGTHS[field]
    field_offset, field_bytes_length, field_type = field_attributes

    record_num_samples = list()
    with open(edf_path, "rb") as f:
        for i in range(n_signals + 1):
            f.seek(TOTAL_HEADER_BYTES + (n_signals + 1) * field_offset + i * field_bytes_length)
            field_bytes = f.read(field_bytes_length)
            value = format_field_from_bytes(field_bytes, field_type)
            record_num_samples.append(value)
    return record_num_samples


def format_field_from_bytes(bytes, field_type):
    if field_type == int:
        value = int(bytes.decode('ascii').strip())
    elif field_type == str:
        value = bytes.decode('ascii').strip()
    elif field_type == None:
        value = bytes
    else:
        raise ValueError(f"Unsupported field type: {field_type}")
    return value


def get_annotation_signal_header_index(edf_path):
    labels = get_signal_header_fields(edf_path, field='label')
    for i, label in enumerate(labels):
        if label.lower() == 'edf annotations':
            return i
    raise ValueError("No annotation signal found in EDF")


def read_header_raw_bytes(edf_path):
    with open(edf_path, "r+b") as f:
        header_bytes = f.read(TOTAL_HEADER_BYTES)
    return header_bytes


def copy_bytes(src_path, dest_path, offset, length):
    with open(src_path, "rb") as src_file:
        src_file.seek(offset)
        bytes_data = src_file.read(length)
    with open(dest_path, "r+b") as dest_file:
        dest_file.seek(offset)
        dest_file.write(bytes_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update EDF header in place.")
    parser.add_argument("--edf-path", type=str, required=True, help="Path to EDF file to update.")
    parser.add_argument("--test", action="store_true", help="Run in test mode, copying the input file to avoid modification.")
    args = parser.parse_args()

    if args.test:
        import shutil
        test_path = args.edf_path + ".test"
        shutil.copy(args.edf_path, test_path)
        args.edf_path = test_path

    updates = dict()
    update_edf_header_inplace(args.edf_path, updates)
