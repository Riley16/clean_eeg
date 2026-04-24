"""EDF header repair for truncated Nihon Kohden exports.

Recordings that are stopped mid-session leave an EDF file whose main
header claims more data records than actually made it to disk.
pyedflib's strict filesize check rejects these files entirely
(``"the file is not EDF(+) or BDF(+) compliant (Filesize)"``). This
module provides a small repair pass that rewrites the header's
number-of-records field so pyedflib reads the file normally.

Approach:
- Parse n_records (bytes 236-244), n_signals (bytes 252-256), and
  samples_per_record from each signal header directly from the file
  bytes.
- Compute the number of *complete* data records actually present on
  disk: ``(filesize - header_bytes) // record_bytes``.
- Validate that at least one complete record is present and that the
  signal-header fields are non-degenerate.
- Overwrite bytes 236-244 with the new n_records.

Trailing bytes from a partial-trailing record are left in place —
pyedflib's filesize check is one-sided (rejects only on under-size),
so this is safe, and keeping bytes on disk avoids destructive writes
beyond what's strictly needed.
"""

import os


# EDF main header is always 256 bytes; each signal-header block adds 256.
MAIN_HEADER_BYTES = 256
SIGNAL_HEADER_BYTES_PER_SIGNAL = 256

# Byte offsets/widths within the main header
N_RECORDS_OFFSET = 236
N_RECORDS_WIDTH = 8
N_SIGNALS_OFFSET = 252
N_SIGNALS_WIDTH = 4

# Within the signal-header block, per-signal fields are stored
# field-by-field across all signals (label[ns], transducer[ns], ...).
# Field widths: label=16, transducer=80, phys_dim=8, phys_min=8,
# phys_max=8, dig_min=8, dig_max=8, prefilter=80, samples_per_record=8,
# reserved=32. samples_per_record begins at cumulative offset 216.
SAMPLES_PER_RECORD_FIELD_OFFSET = 16 + 80 + 8 + 8 + 8 + 8 + 8 + 80
SAMPLES_PER_RECORD_FIELD_WIDTH = 8


def _read_header_fields(edf_path: str) -> dict:
    """Parse the subset of EDF header fields needed for repair."""
    with open(edf_path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
        n_records = int(main[N_RECORDS_OFFSET:N_RECORDS_OFFSET + N_RECORDS_WIDTH].decode().strip())
        n_signals = int(main[N_SIGNALS_OFFSET:N_SIGNALS_OFFSET + N_SIGNALS_WIDTH].decode().strip())
        signal_header = f.read(SIGNAL_HEADER_BYTES_PER_SIGNAL * n_signals)

    base = SAMPLES_PER_RECORD_FIELD_OFFSET * n_signals
    samples_per_record = []
    for i in range(n_signals):
        start = base + i * SAMPLES_PER_RECORD_FIELD_WIDTH
        stop = start + SAMPLES_PER_RECORD_FIELD_WIDTH
        samples_per_record.append(int(signal_header[start:stop].decode().strip()))

    header_bytes = MAIN_HEADER_BYTES + SIGNAL_HEADER_BYTES_PER_SIGNAL * n_signals
    record_bytes = sum(samples_per_record) * 2  # EDF samples are int16
    return {
        "n_records": n_records,
        "n_signals": n_signals,
        "samples_per_record": samples_per_record,
        "header_bytes": header_bytes,
        "record_bytes": record_bytes,
    }


def is_edf_truncated(edf_path: str) -> bool:
    """Return True iff the file on disk is shorter than the header claims."""
    fields = _read_header_fields(edf_path)
    expected = fields["header_bytes"] + fields["n_records"] * fields["record_bytes"]
    return os.path.getsize(edf_path) < expected


def repair_truncated_edf_header(edf_path: str, verbosity: int = 1) -> int:
    """Rewrite the n_records field so pyedflib's filesize check passes.

    No-op on files that aren't truncated. Returns the n_records value that
    is now in the header. Raises ValueError when the file is too corrupt
    to repair safely.

    Parameters
    ----------
    edf_path : str
        Path to the EDF file. Modified in place.
    verbosity : int
        If >= 1, prints a one-line report of the repair.
    """
    fields = _read_header_fields(edf_path)
    record_bytes = fields["record_bytes"]
    header_bytes = fields["header_bytes"]
    old_n_records = fields["n_records"]

    # Validate that the signal-header-derived record size is sensible.
    if record_bytes <= 0:
        raise ValueError(
            f"Cannot repair {edf_path}: record_bytes derived from signal "
            f"headers is {record_bytes} (expected > 0). Signal header is "
            "malformed."
        )
    for i, spr in enumerate(fields["samples_per_record"]):
        if spr <= 0:
            raise ValueError(
                f"Cannot repair {edf_path}: signal {i} reports "
                f"samples_per_record={spr} (expected > 0)."
            )

    file_size = os.path.getsize(edf_path)
    data_bytes = file_size - header_bytes
    expected_file_size = header_bytes + old_n_records * record_bytes

    # Already valid — no repair needed.
    if file_size >= expected_file_size:
        return old_n_records

    if data_bytes <= 0:
        raise ValueError(
            f"Cannot repair {edf_path}: file contains no data bytes after "
            f"the header ({data_bytes} data bytes)."
        )

    new_n_records = data_bytes // record_bytes
    if new_n_records < 1:
        raise ValueError(
            f"Cannot repair {edf_path}: not even one complete data record "
            f"is present on disk ({data_bytes} data bytes, {record_bytes} "
            "bytes per record). File is too corrupt to read safely."
        )

    if verbosity >= 1:
        dropped = old_n_records - new_n_records
        leftover = data_bytes % record_bytes
        print(
            f"Repairing truncated EDF header: {edf_path}\n"
            f"  header claimed n_records={old_n_records}; "
            f"{new_n_records} complete records actually on disk "
            f"(dropping {dropped} missing records + a partial "
            f"{leftover}-byte trailing record)."
        )

    _write_n_records_field(edf_path, new_n_records)
    return new_n_records


def _write_n_records_field(edf_path: str, n_records: int) -> None:
    """Overwrite bytes 236-244 of the EDF main header with n_records as an
    ASCII decimal, space-padded on the right to exactly 8 bytes."""
    value = str(n_records)
    if len(value) > N_RECORDS_WIDTH:
        raise ValueError(
            f"n_records={n_records} does not fit in {N_RECORDS_WIDTH} ASCII "
            "bytes (EDF spec limit)."
        )
    encoded = value.ljust(N_RECORDS_WIDTH).encode("ascii")
    with open(edf_path, "r+b") as f:
        f.seek(N_RECORDS_OFFSET)
        f.write(encoded)
