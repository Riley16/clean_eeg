# edf_inplace.py
from typing import Dict, List, Union
from copy import deepcopy
import datetime
import os
import re
import shutil
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
        # Preserve the original data-record duration so the temp writer computes
        # the correct samples_per_record per signal. Otherwise pyedflib auto-
        # derives a record_duration from the sample frequencies (typically 1.0s)
        # and writes samples_per_record = sample_frequency * 1.0, which mismatches
        # the orig's main-header data_record_duration we copy back below.
        orig_record_duration = f.datarecord_duration
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
        # Lock the temp writer to the orig's record duration so the
        # samples_per_record bytes in each signal header match what we'll
        # copy back. pyedflib's setDatarecordDuration emits a harmless
        # FutureWarning; we suppress it since the override is intentional.
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            f.setDatarecordDuration(orig_record_duration)
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

    # Preserve the original file's signal-header numeric fields. These
    # describe the actual on-disk data layout (physical/digital ranges and,
    # critically, samples_per_record per signal — which determines how many
    # bytes each signal occupies inside each data record). pyedflib's empty
    # temp writer recomputes these from whatever defaults it has and can
    # pick values that disagree with the bytes physically present on disk
    # (e.g. the "EDF Annotations" signal ends up with samples_per_record=57
    # instead of the orig's 172, shifting every record's layout). Copy the
    # orig's bytes into the temp before we swap headers back.
    on_disk_n_signals = get_header_field(edf_path, 'num_signals')
    preserve_signal_header_fields = [
        'physical_min', 'physical_max',
        'digital_min', 'digital_max',
        'num_samples',
    ]
    for field in preserve_signal_header_fields:
        field_offset, field_width, _ = EDF_SIGNAL_HEADER_FIELD_OFFSETS_LENGTHS[field]
        abs_offset = TOTAL_HEADER_BYTES + field_offset * on_disk_n_signals
        total_length = field_width * on_disk_n_signals
        copy_bytes(edf_path, temp_path, abs_offset, total_length)

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


def validate_header_roundtrip(header: dict, signal_headers: list = None) -> list[str]:
    """Check for pyedflib truncation by doing a dry-run header write.

    Returns a list of warning strings for any fields that would be truncated.
    pyedflib packs multiple fields into the 80-byte patient_id and recording_id
    EDF fields, so per-field byte limits can't be checked in isolation.
    """
    import tempfile
    import warnings as warnings_module
    result = []
    n_channels = len(signal_headers) if signal_headers else 0
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            with pyedflib.EdfWriter(tmp_path, n_channels) as w:
                w.setHeader(header)
                if signal_headers:
                    w.setSignalHeaders(signal_headers)
        for w in caught:
            result.append(str(w.message))
    finally:
        os.remove(tmp_path)
    return result


def create_annotations_only_edf(path: str,
                                header: dict,
                                annotations: tuple,
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



def _encode_tal(onset: float, duration: float, text: str) -> bytes:
    """Encode a single EDF+ Time-stamped Annotation List (TAL) entry.

    TAL format: +onset[\\x15duration]\\x14text\\x14\\x00
    The onset sign (+ or -) is mandatory per EDF+ spec.
    """
    onset_str = f"+{onset:.10g}" if onset >= 0 else f"{onset:.10g}"
    if duration > 0:
        dur_str = f"{duration:.10g}"
        tal = f"{onset_str}\x15{dur_str}\x14{text}\x14\x00"
    else:
        tal = f"{onset_str}\x14{text}\x14\x00"
    return tal.encode("utf-8")


def merge_annotation_stub_edf(data_edf_path: str,
                               stub_edf_path: str,
                               validate: bool = True) -> None:
    """Merge annotations from a stub EDF back into a data EDF.

    The data EDF is expected to have its annotation texts cleared
    (e.g., by clear_edf_annotations_inplace). The stub EDF is an
    annotations-only file created by create_annotations_only_edf.

    Uses atomic file replacement (os.replace) so the data EDF is never
    left in a partially-written state.

    Args:
        data_edf_path: Path to the data EDF (with cleared annotations).
        stub_edf_path: Path to the annotations-only stub EDF.
        validate: If True, verify merged annotations match the stub.
    """
    # Read annotations from stub
    with pyedflib.EdfReader(stub_edf_path) as f:
        ann_onsets, ann_durations, ann_texts = f.readAnnotations()

    if len(ann_onsets) == 0:
        return  # nothing to merge

    # Get annotation record layout from data EDF
    ann_signal_index = get_annotation_signal_header_index(data_edf_path)
    signal_record_lengths = get_signal_header_fields(data_edf_path, field='num_samples')
    for i in range(len(signal_record_lengths)):
        signal_record_lengths[i] *= 2  # 2 bytes per sample
    ann_record_length = signal_record_lengths[ann_signal_index]
    ann_record_offset = sum(signal_record_lengths[:ann_signal_index])
    total_records_length = sum(signal_record_lengths)

    n_signals = len(signal_record_lengths)
    n_records = get_header_field(data_edf_path, 'num_data_records')
    record_duration = float(get_header_field(data_edf_path, 'data_record_duration'))

    # Assign each annotation to the data record it belongs to
    record_annotations = {i: [] for i in range(n_records)}
    for onset, duration, text in zip(ann_onsets, ann_durations, ann_texts):
        if record_duration > 0:
            record_idx = min(int(onset / record_duration), n_records - 1)
        else:
            record_idx = 0
        record_annotations[record_idx].append((onset, duration, text))

    # Snapshot original signals and headers before any modification so
    # the post-merge integrity check can verify nothing was corrupted.
    with pyedflib.EdfReader(data_edf_path) as f:
        orig_header = f.getHeader()
        orig_signal_headers = [f.getSignalHeader(i)
                               for i in range(f.signals_in_file)]
        orig_signals = [f.readSignal(i) for i in range(f.signals_in_file)]

    # Work on a temp copy so the original is never partially written
    temp_path = data_edf_path + ".merge_tmp"
    shutil.copy2(data_edf_path, temp_path)
    try:
        header_size = TOTAL_HEADER_BYTES + SIGNAL_HEADER_BYTES * n_signals
        with open(temp_path, "r+b") as f:
            for record_idx in range(n_records):
                if not record_annotations[record_idx]:
                    continue  # no annotations for this record

                record_offset = (header_size
                                 + total_records_length * record_idx
                                 + ann_record_offset)
                f.seek(record_offset)
                ann_bytes = f.read(ann_record_length)

                # Find end of timekeeping TAL
                tk_delim = b"\x14\x14"
                tk_end = ann_bytes.find(tk_delim)
                if tk_end < 0:
                    raise ValueError(
                        f"No timekeeping TAL found in data record {record_idx}")
                tk_end += len(tk_delim)

                # Encode annotations for this record.
                # Leading \x00 terminates the timekeeping TAL before the
                # first annotation TAL begins.
                encoded = b"\x00"
                for onset, duration, text in record_annotations[record_idx]:
                    encoded += _encode_tal(onset, duration, text)

                available = ann_record_length - tk_end
                if len(encoded) > available:
                    raise ValueError(
                        f"Annotations for record {record_idx} "
                        f"({len(encoded)} bytes) exceed available space "
                        f"({available} bytes) in annotation signal")

                # Write annotation bytes + zero padding
                new_region = encoded + b"\x00" * (available - len(encoded))
                f.seek(record_offset + tk_end)
                f.write(new_region)
            f.flush()
            os.fsync(f.fileno())

        # ---- Integrity check: signals, headers, and annotations ----
        # The merge must not corrupt any part of the EDF. Verify all
        # three domains before committing the atomic replacement.
        _verify_merge_integrity(
            temp_path, orig_header, orig_signal_headers, orig_signals,
            ann_onsets, ann_durations, ann_texts)

        # Atomic replacement
        os.replace(temp_path, data_edf_path)
    except Exception as exc:
        print(f"ERROR: Annotation merge failed. The partially-written temp "
              f"file has been preserved for debugging:\n  {temp_path}")
        raise ValueError(
            f"Annotation merge failed; temp file preserved at {temp_path}"
        ) from exc


def _verify_merge_integrity(merged_path: str,
                            orig_header: dict,
                            orig_signal_headers: list,
                            orig_signals: list,
                            expected_onsets: np.ndarray,
                            expected_durations: np.ndarray,
                            expected_texts: np.ndarray) -> None:
    """Verify the merged EDF has identical signals/headers and correct annotations.

    Raises ValueError if any mismatch is detected, preventing the atomic
    replacement from proceeding.
    """
    with pyedflib.EdfReader(merged_path) as f:
        merged_header = f.getHeader()
        n_signals = f.signals_in_file
        merged_signal_headers = [f.getSignalHeader(i) for i in range(n_signals)]
        merged_signals = [f.readSignal(i) for i in range(n_signals)]
        m_onsets, m_durations, m_texts = f.readAnnotations()

    # --- Signals: must be bit-identical ---
    if len(merged_signals) != len(orig_signals):
        raise ValueError(
            f"Signal count changed: {len(orig_signals)} -> {len(merged_signals)}")
    for i, (orig, merged) in enumerate(zip(orig_signals, merged_signals)):
        if not np.array_equal(orig, merged):
            raise ValueError(f"Signal {i} data corrupted by merge")

    # --- Main header: every field must be unchanged ---
    for key in orig_header:
        if orig_header[key] != merged_header.get(key):
            raise ValueError(
                f"Header field '{key}' changed: "
                f"'{orig_header[key]}' -> '{merged_header.get(key)}'")

    # --- Signal headers: every field of every channel must be unchanged ---
    for i, (orig_sh, merged_sh) in enumerate(
            zip(orig_signal_headers, merged_signal_headers)):
        for key in orig_sh:
            if orig_sh[key] != merged_sh.get(key):
                raise ValueError(
                    f"Signal header {i} field '{key}' changed: "
                    f"'{orig_sh[key]}' -> '{merged_sh.get(key)}'")

    # --- Annotations: texts, onsets, and durations must match stub ---
    if len(m_texts) != len(expected_texts):
        raise ValueError(
            f"Annotation count mismatch: expected {len(expected_texts)}, "
            f"got {len(m_texts)}")
    for i, (exp, got) in enumerate(zip(expected_texts, m_texts)):
        if exp != got:
            raise ValueError(
                f"Annotation {i} text mismatch: expected '{exp}', got '{got}'")
    if not np.allclose(m_onsets, expected_onsets):
        raise ValueError("Annotation onsets mismatch after merge")
    if not np.allclose(m_durations, expected_durations):
        raise ValueError("Annotation durations mismatch after merge")


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
