# edf_inplace.py
from copy import deepcopy
import datetime
import os
import pyedflib


def update_edf_header_inplace(edf_path,
                              updates: dict,
                              confirm_signals_unchanged: bool = False,
                              verbosity: int = 0):
    # write EDF file with no data but with updated header using pyedflib
    # then copy the updated header bytes into the original file at the correct position

    # Read original header
    with pyedflib.EdfReader(edf_path) as f:
        orig_header = f.getHeader()
        orig_signal_headers = f.getSignalHeaders()
        if confirm_signals_unchanged:
            # load original signals for later comparison
            orig_signals = [f.readSignal(i) for i in range(f.signals_in_file)]
    
    # Apply updates to header fields
    updated_header = orig_header.copy()
    for field, value in updates.items():
        if value is not None:
            updated_header[field] = value
    # updated_header = clean_header(updated_header, raise_errors=False)
    
    # Write new EDF with updated header but no data
    temp_path = edf_path + ".tmp"
    with pyedflib.EdfWriter(temp_path, 0) as f:
        if verbosity > 0:
            print("updated header written to temp EDF:")
            print(updated_header)
        f.setHeader(updated_header)
        f.setSignalHeaders(orig_signal_headers)

    # fix header fields since pyedflib only updates headers for signal
    # info once signals (but not signal headers) are added
    copy_overwrite_fields = ['header_bytes',
                             'num_signals',
                             'num_data_records',
                             'data_record_duration']
    for field in copy_overwrite_fields:
        copy_bytes(edf_path, temp_path,
                   *EDF_FIELD_OFFSETS_LENGTHS[field])

    # Copy updated header bytes back to original file
    with open(edf_path, "r+b") as orig_file, open(temp_path, "rb") as temp_file:
        updated_header_bytes = temp_file.read(256)
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
        max_length = EDF_FIELD_OFFSETS_LENGTHS[field][1]
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


def update_edf_annotations_inplace(edf_path,
                                   updated_annotations,
                                   verbosity: int = 0):
    updated_ann_onsets, updated_ann_durations, updated_ann_texts = updated_annotations
    # # Read original annotations
    # with pyedflib.EdfReader(edf_path) as f:
    #     ann_onsets, ann_durations, ann_texts = f.readAnnotations()

    # scan through EDF and overwrite annotation texts in place
    # (assuming/asserting lengths of onsets and durations remain unchanged)

    # get annotation signal index, number of records, and annotation record 
    # length from EDF header/signal headers
    with pyedflib.EdfReader(edf_path) as f:
        n_records = f.num_data_records
        ann_signal_index = f.getAnnotationSignalIndex()
        ann_signal_header = f.getSignalHeader(ann_signal_index)
        ann_record_length = ann_signal_header['record_length']

    print(f"EDF has {n_records} data records; "
          f"annotation signal index = {ann_signal_index}; "
          f"n_records = {n_records}; "
          f"annotation record length = {ann_record_length}")


# Byte offsets and lengths for EDF header fields
EDF_FIELD_OFFSETS_LENGTHS = {
    'version': (0, 8),
    'patient_id': (8, 80),
    'recording_id': (88, 80),
    'startdate': (168, 8),
    'starttime': (176, 8),
    'header_bytes': (184, 8),
    'reserved': (192, 44),
    'num_data_records': (236, 8),
    'data_record_duration': (244, 8),
    'num_signals': (252, 4),
}


def read_header_raw_bytes(edf_path):
    with open(edf_path, "r+b") as f:
        header_bytes = f.read(256)
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
