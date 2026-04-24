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
# reserved=32.
SIG_LABEL_OFFSET = 0
SIG_LABEL_WIDTH = 16
SIG_PHYS_DIM_OFFSET = 16 + 80           # = 96
SIG_PHYS_DIM_WIDTH = 8
SIG_PHYS_MIN_OFFSET = 16 + 80 + 8       # = 104
SIG_PHYS_MIN_WIDTH = 8
SIG_PHYS_MAX_OFFSET = 16 + 80 + 8 + 8   # = 112
SIG_PHYS_MAX_WIDTH = 8
SIG_DIG_MIN_OFFSET = 16 + 80 + 8 + 8 + 8       # = 120
SIG_DIG_MIN_WIDTH = 8
SIG_DIG_MAX_OFFSET = 16 + 80 + 8 + 8 + 8 + 8   # = 128
SIG_DIG_MAX_WIDTH = 8
SAMPLES_PER_RECORD_FIELD_OFFSET = 16 + 80 + 8 + 8 + 8 + 8 + 8 + 80  # = 216
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


# Values written for a signal whose physical range could not be preserved.
# Per EDF+ spec (section 2.1.3 item 5): "In case of uncalibrated signals,
# physical dimension is left empty (that is 8 spaces), while 'Physical
# maximum' and 'Physical minimum' must still contain different values
# (this is to avoid 'division by 0' errors by some viewers)."
UNCALIBRATED_PHYS_MIN = "-1"
UNCALIBRATED_PHYS_MAX = "1"
UNCALIBRATED_PHYS_DIM = ""   # rendered as 8 spaces after ljust()

def repair_degenerate_signal_ranges(edf_path: str, verbosity: int = 1) -> list:
    """Detect signals with invalid EDF+ physical/digital range headers.

    Bad patterns detected:
    - ``phys_max == phys_min`` (triggers ``EDFLIB_FILE_ERRORS_PHYS_MAX``;
      linear digital-to-physical scaling is degenerate).
    - ``dig_max <= dig_min`` (triggers ``EDFLIB_FILE_ERRORS_DIG_MAX``;
      spec requires strict ``dig_max > dig_min``).
    - Either phys or dig field unparseable as a number.

    Note: ``phys_max < phys_min`` is **allowed** by the spec and left
    alone — it encodes a negative amplifier gain for the
    "negativity upward" Clinical Neurophysiology convention.

    Repair policy:
    - **Physical range issues ARE repaired** — the fix is honest because
      we overwrite with the EDF+ "uncalibrated signal" convention
      (phys_min=-1, phys_max=1, physical_dimension=8 spaces, section
      2.1.3 item 5). No downstream tool should trust the scaling.
    - **Digital range issues are NOT repaired**. Synthesising a digital
      range would silently re-scale the physical interpretation of the
      raw samples on disk (physical = phys_min + (dig - dig_min) *
      (phys_max - phys_min) / (dig_max - dig_min)), producing wrong
      amplitudes. Instead we emit a loud warning with the original
      values and let pyedflib reject the file on read — the existing
      per-file skip path in ``_load_edf_metadata`` then surfaces it to
      the operator as a manual-handling case.

    Returns a list of dicts describing every signal with a detected
    issue — keys include ``signal_idx``, ``label``, ``phys_issue``,
    ``dig_issue``, ``phys_repaired`` (always True when phys_issue is
    set), and the original field values.
    """
    with open(edf_path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
        n_signals = int(main[N_SIGNALS_OFFSET:N_SIGNALS_OFFSET + N_SIGNALS_WIDTH]
                        .decode().strip())
        sig_block = f.read(SIGNAL_HEADER_BYTES_PER_SIGNAL * n_signals)

    def _field_slice(field_offset: int, field_width: int, signal_idx: int) -> bytes:
        start = field_offset * n_signals + signal_idx * field_width
        return sig_block[start:start + field_width]

    repairs = []
    for i in range(n_signals):
        label_b = _field_slice(SIG_LABEL_OFFSET, SIG_LABEL_WIDTH, i)
        pmin_b = _field_slice(SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH, i)
        pmax_b = _field_slice(SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH, i)
        pdim_b = _field_slice(SIG_PHYS_DIM_OFFSET, SIG_PHYS_DIM_WIDTH, i)
        dmin_b = _field_slice(SIG_DIG_MIN_OFFSET, SIG_DIG_MIN_WIDTH, i)
        dmax_b = _field_slice(SIG_DIG_MAX_OFFSET, SIG_DIG_MAX_WIDTH, i)
        label = label_b.decode("ascii", errors="replace").rstrip()
        pmin_str = pmin_b.decode("ascii", errors="replace").rstrip()
        pmax_str = pmax_b.decode("ascii", errors="replace").rstrip()
        pdim_str = pdim_b.decode("ascii", errors="replace").rstrip()
        dmin_str = dmin_b.decode("ascii", errors="replace").rstrip()
        dmax_str = dmax_b.decode("ascii", errors="replace").rstrip()

        phys_reason = None
        try:
            pmin_val = float(pmin_str)
            pmax_val = float(pmax_str)
            if pmin_val == pmax_val:
                phys_reason = (f"phys_min == phys_max == {pmin_val} "
                               "(degenerate scaling would divide by zero)")
        except ValueError as e:
            phys_reason = (f"unparseable: phys_min={pmin_str!r} "
                           f"phys_max={pmax_str!r} ({e})")

        dig_reason = None
        try:
            dmin_val = int(dmin_str)
            dmax_val = int(dmax_str)
            # EDF+ spec requires strict dmax > dmin. edflib.c also rejects
            # equality: `if (dig_max < dig_min + 1) error`.
            if dmax_val <= dmin_val:
                dig_reason = (f"dig_max ({dmax_val}) <= dig_min ({dmin_val}); "
                              "EDF+ requires strict dig_max > dig_min")
        except ValueError as e:
            dig_reason = (f"unparseable: dig_min={dmin_str!r} "
                          f"dig_max={dmax_str!r} ({e})")

        if phys_reason is None and dig_reason is None:
            continue

        # Build a single warning covering whichever side(s) of the range
        # are invalid, for the log.
        msg_parts = [f"Signal {i} ({label!r}) has an invalid EDF+ header:"]
        if phys_reason:
            msg_parts.append(
                f"  physical range issue: {phys_reason}. Rewriting "
                f"phys_min={UNCALIBRATED_PHYS_MIN}, "
                f"phys_max={UNCALIBRATED_PHYS_MAX}, "
                f"units (physical_dimension)=<8 spaces> (EDF+ "
                f'"uncalibrated signal" convention, section 2.1.3 item 5).'
            )
        if dig_reason:
            msg_parts.append(
                f"  digital range issue: {dig_reason}. NOT REPAIRED — "
                "synthesising a digital range would silently re-scale "
                "the signal. pyedflib will reject this file with "
                '"(Digital Maximum)" and it will be reported in the '
                "skipped-files summary for manual handling."
            )
        msg_parts.append(
            f"  Original values: phys_min={pmin_str!r}, "
            f"phys_max={pmax_str!r}, units={pdim_str!r}, "
            f"dig_min={dmin_str!r}, dig_max={dmax_str!r}."
        )
        if verbosity >= 1:
            print("WARNING: " + "\n".join(msg_parts))

        if phys_reason:
            _write_signal_field(edf_path, n_signals, i,
                                SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH,
                                UNCALIBRATED_PHYS_MIN)
            _write_signal_field(edf_path, n_signals, i,
                                SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH,
                                UNCALIBRATED_PHYS_MAX)
            _write_signal_field(edf_path, n_signals, i,
                                SIG_PHYS_DIM_OFFSET, SIG_PHYS_DIM_WIDTH,
                                UNCALIBRATED_PHYS_DIM)
        # Dig range issues are detected-and-warned only; never rewritten.
        repairs.append({
            "signal_idx": i,
            "label": label,
            "phys_issue": phys_reason,
            "dig_issue": dig_reason,
            "phys_repaired": phys_reason is not None,
            "dig_repaired": False,
            "original_phys_min": pmin_str,
            "original_phys_max": pmax_str,
            "original_phys_dim": pdim_str,
            "original_dig_min": dmin_str,
            "original_dig_max": dmax_str,
        })
    return repairs


def _write_signal_field(edf_path: str,
                        n_signals: int,
                        signal_idx: int,
                        field_offset: int,
                        field_width: int,
                        value: str) -> None:
    """Overwrite one per-signal header field in place.

    The signal-header region is laid out field-by-field across signals, so
    signal i's field lives at absolute offset:
        MAIN_HEADER_BYTES + field_offset*n_signals + signal_idx*field_width
    """
    if len(value) > field_width:
        raise ValueError(
            f"value {value!r} does not fit in {field_width} ASCII bytes"
        )
    abs_offset = (MAIN_HEADER_BYTES
                  + field_offset * n_signals
                  + signal_idx * field_width)
    encoded = value.ljust(field_width).encode("ascii")
    with open(edf_path, "r+b") as f:
        f.seek(abs_offset)
        f.write(encoded)


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
