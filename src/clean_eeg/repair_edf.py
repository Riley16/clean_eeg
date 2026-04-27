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
BYTES_IN_HEADER_OFFSET = 184
BYTES_IN_HEADER_WIDTH = 8
N_RECORDS_OFFSET = 236
N_RECORDS_WIDTH = 8
RECORD_DURATION_OFFSET = 244
RECORD_DURATION_WIDTH = 8
N_SIGNALS_OFFSET = 252
N_SIGNALS_WIDTH = 4

# Default record_duration (seconds) used to repair an empty/blank field.
# 1.0 s is the EDF+ spec's most common convention. The repair pass emits
# a warning that includes the implied sample rate
# (samples_per_record_signal_0 / 1.0) so an operator can immediately see
# whether 1.0 is plausible for their recording setup.
DEFAULT_RECORD_DURATION_S = 1.0

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


def validate_edf_minimum_size(edf_path: str) -> None:
    """Reject 0-byte and sub-main-header files with a user-readable error
    before any parse is attempted."""
    size = os.path.getsize(edf_path)
    if size == 0:
        raise ValueError(
            f"EDF file is empty (0 bytes): {edf_path!r}. "
            f"Likely an interrupted copy or an EEG export that never "
            f"finalized. Verify the source file size and re-copy if needed."
        )
    if size < MAIN_HEADER_BYTES:
        raise ValueError(
            f"EDF file is too small to contain a header: "
            f"{edf_path!r} is {size} bytes (main header alone is "
            f"{MAIN_HEADER_BYTES} bytes)."
        )


def _parse_int_field(raw: bytes, field_name: str, signal_idx: int = -1,
                     edf_path: str = "") -> int | None:
    """Parse a numeric EDF header field.

    Returns ``int`` for a valid integer, ``None`` for an empty/whitespace-only
    field. Raises ``ValueError`` with the field name and the raw value on
    any other unparseable input.

    EDF spec only defines one missing-value sentinel (``n_records = -1``,
    "unknown / recording in progress"). For other numeric fields the spec
    requires a number; some Nihon Kohden devices nevertheless emit empty
    field-slots, so we surface the empty case as ``None`` and let the
    caller decide whether to repair or error.
    """
    s = raw.decode("ascii", errors="replace").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError as e:
        ctx = f" signal={signal_idx}" if signal_idx >= 0 else ""
        path_ctx = f" file={edf_path!r}" if edf_path else ""
        raise ValueError(
            f"EDF field {field_name!r}{ctx}{path_ctx} value={s!r}: "
            f"expected integer, got unparseable text"
        ) from e


def _empty_field_error(field_name: str, signal_idx: int = -1,
                       edf_path: str = "", *,
                       why_unrecoverable: str,
                       extra: str = "") -> ValueError:
    """Build a ``ValueError`` for a numeric field that is empty and that
    cannot be safely defaulted. Mirrors the ``_parse_int_field`` format
    so error text in log.out is consistent."""
    ctx = f" signal={signal_idx}" if signal_idx >= 0 else ""
    path_ctx = f" file={edf_path!r}" if edf_path else ""
    msg = (
        f"EDF field {field_name!r}{ctx}{path_ctx} value='': "
        f"empty/blank in source EDF (no spec-defined sentinel for this "
        f"field). Cannot proceed: {why_unrecoverable}"
    )
    if extra:
        msg += f"\n{extra}"
    return ValueError(msg)


def _read_header_fields(edf_path: str) -> dict:
    """Parse the subset of EDF header fields needed for repair.

    Returns a dict where ``n_records`` may be ``None`` if the source EDF
    left bytes 236-244 blank (Nihon Kohden anomaly); the truncation
    repair pass treats ``None`` as "compute from filesize". Empty
    ``n_signals`` or ``samples_per_record`` raise immediately — those
    govern byte offsets and have no safe fallback.
    """
    with open(edf_path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
        n_records = _parse_int_field(
            main[N_RECORDS_OFFSET:N_RECORDS_OFFSET + N_RECORDS_WIDTH],
            "n_records", edf_path=edf_path,
        )
        n_signals = _parse_int_field(
            main[N_SIGNALS_OFFSET:N_SIGNALS_OFFSET + N_SIGNALS_WIDTH],
            "n_signals", edf_path=edf_path,
        )
        if n_signals is None:
            raise _empty_field_error(
                "n_signals", edf_path=edf_path,
                why_unrecoverable=(
                    "every per-signal field-slice in the header depends on "
                    "knowing N. No safe default — file is unreadable as EDF."
                ),
            )
        signal_header = f.read(SIGNAL_HEADER_BYTES_PER_SIGNAL * n_signals)

    base = SAMPLES_PER_RECORD_FIELD_OFFSET * n_signals
    samples_per_record = []
    for i in range(n_signals):
        start = base + i * SAMPLES_PER_RECORD_FIELD_WIDTH
        stop = start + SAMPLES_PER_RECORD_FIELD_WIDTH
        spr = _parse_int_field(
            signal_header[start:stop],
            "samples_per_record", signal_idx=i, edf_path=edf_path,
        )
        if spr is None:
            raise _empty_field_error(
                "samples_per_record", signal_idx=i, edf_path=edf_path,
                why_unrecoverable=(
                    "this field determines the byte layout of every record "
                    "(record stride = sum of samples_per_record * 2 bytes). "
                    "No safe default — every read past this signal would "
                    "land at a wrong offset and silently corrupt the output."
                ),
            )
        samples_per_record.append(spr)

    header_bytes = MAIN_HEADER_BYTES + SIGNAL_HEADER_BYTES_PER_SIGNAL * n_signals
    record_bytes = sum(samples_per_record) * 2  # EDF samples are int16
    return {
        "n_records": n_records,  # may be None — caller must handle
        "n_signals": n_signals,
        "samples_per_record": samples_per_record,
        "header_bytes": header_bytes,
        "record_bytes": record_bytes,
    }


def is_edf_truncated(edf_path: str) -> bool:
    """Return True iff the file on disk is shorter than the header claims.

    Treats ``n_records=None`` (empty source field) and ``n_records < 0``
    (EDF+ 'recording in progress' sentinel) as "claim is unverifiable",
    not truncated — those are repair_truncated_edf_header's job.
    """
    fields = _read_header_fields(edf_path)
    n_rec = fields["n_records"]
    if n_rec is None or n_rec < 0:
        return False
    expected = fields["header_bytes"] + n_rec * fields["record_bytes"]
    return os.path.getsize(edf_path) < expected


def _repair_bytes_in_header(edf_path: str, main: bytes, n_signals: int,
                              verbosity: int):
    """Repair an empty ``bytes_in_header`` (bytes 184-192). No-op if the
    field is already populated. Returns the new value or ``None``."""
    bih_raw = main[BYTES_IN_HEADER_OFFSET:BYTES_IN_HEADER_OFFSET + BYTES_IN_HEADER_WIDTH]
    bih = _parse_int_field(bih_raw, "bytes_in_header", edf_path=edf_path)
    if bih is not None:
        return None
    new_bih = MAIN_HEADER_BYTES * (1 + n_signals)
    if verbosity >= 1:
        print(
            f"Repairing empty bytes_in_header in {edf_path}\n"
            f"  Field 'bytes_in_header' bytes 184-192 were empty/blank. "
            f"Writing {new_bih} (= 256 * (1 + n_signals={n_signals}))."
        )
    _write_main_header_string_field(
        edf_path, BYTES_IN_HEADER_OFFSET, BYTES_IN_HEADER_WIDTH, str(new_bih),
    )
    return new_bih


def _repair_record_duration(edf_path: str, main: bytes,
                             samples_per_record: list, verbosity: int):
    """Repair an empty ``record_duration`` (bytes 244-252). No-op if the
    field is already populated. On repair, emits a WARNING showing the
    sample rate that the default of 1.0 s would imply for signal 0 — so
    an operator can spot a wrong assumption at a glance. Returns the new
    value or ``None``."""
    rd_raw = main[RECORD_DURATION_OFFSET:RECORD_DURATION_OFFSET + RECORD_DURATION_WIDTH]
    if rd_raw.decode("ascii", errors="replace").strip():
        return None
    spr0 = samples_per_record[0] if samples_per_record else None
    new_rd = DEFAULT_RECORD_DURATION_S
    if verbosity >= 1:
        if isinstance(spr0, int) and spr0 > 0:
            rate_note = (
                f"Implied sample rate (samples_per_record signal 0 "
                f"= {spr0} / record_duration = {new_rd:g}) = "
                f"{spr0 / new_rd:g} Hz."
            )
        else:
            rate_note = (
                "Implied sample rate is unknown — samples_per_record "
                "for signal 0 was also empty/unparseable."
            )
        print(
            f"WARNING: Repairing empty record_duration in {edf_path}\n"
            f"  Field 'record_duration' bytes 244-252 were empty/blank "
            f"(EDF+ has no missing-value sentinel for this field). "
            f"Defaulting to {new_rd:g} seconds.\n"
            f"  {rate_note}\n"
            f"  If the implied rate is wrong, the time axis of the "
            f"de-identified output will be wrong. Verify against the "
            f"source data before sharing the cleaned file."
        )
    _write_main_header_string_field(
        edf_path, RECORD_DURATION_OFFSET, RECORD_DURATION_WIDTH,
        f"{new_rd:g}",
    )
    return new_rd


def _repair_n_records(edf_path: str, fields: dict, verbosity: int):
    """Repair ``n_records`` when it is empty, the EDF+ ``-1`` sentinel,
    or claims more records than the file holds (truncation). No-op when
    the on-disk count matches the header. Returns the new value or
    ``None``.

    Raises ``ValueError`` when the file is too corrupt to repair safely
    (no data bytes after the header, or fewer than one complete record)."""
    record_bytes = fields["record_bytes"]
    header_bytes = fields["header_bytes"]
    old_n_records = fields["n_records"]  # may be None for empty source field

    file_size = os.path.getsize(edf_path)
    data_bytes = file_size - header_bytes
    if data_bytes <= 0:
        raise ValueError(
            f"Cannot repair {edf_path}: file contains no data bytes after "
            f"the header ({data_bytes} data bytes)."
        )
    actual_n_records = data_bytes // record_bytes
    if actual_n_records < 1:
        raise ValueError(
            f"Cannot repair {edf_path}: not even one complete data record "
            f"is present on disk ({data_bytes} data bytes, {record_bytes} "
            "bytes per record). File is too corrupt to read safely."
        )

    if old_n_records is None:
        if verbosity >= 1:
            print(
                f"Repairing empty n_records field: {edf_path}\n"
                f"  EDF main header bytes 236-244 were empty/blank "
                f"(no spec-defined sentinel was written). "
                f"Computed n_records={actual_n_records} from filesize."
            )
    elif old_n_records < 0:
        if verbosity >= 1:
            print(
                f"Resolving n_records sentinel: {edf_path}\n"
                f"  Header n_records={old_n_records} (EDF+ 'unknown / "
                f"recording in progress'). "
                f"Computed n_records={actual_n_records} from filesize."
            )
    elif file_size < header_bytes + old_n_records * record_bytes:
        if verbosity >= 1:
            dropped = old_n_records - actual_n_records
            leftover = data_bytes % record_bytes
            print(
                f"Repairing truncated EDF header: {edf_path}\n"
                f"  header claimed n_records={old_n_records}; "
                f"{actual_n_records} complete records actually on disk "
                f"(dropping {dropped} missing records + a partial "
                f"{leftover}-byte trailing record)."
            )
    else:
        # Already valid — no repair needed.
        return None

    _write_n_records_field(edf_path, actual_n_records)
    return actual_n_records


def repair_main_header_numeric_fields(edf_path: str,
                                        verbosity: int = 1) -> dict:
    """Repair the EDF main-header numeric fields that pyedflib would
    otherwise reject — ``bytes_in_header``, ``record_duration``, and
    ``n_records``. Single pass over the file. Idempotent on already-valid
    headers.

    See ``_repair_bytes_in_header``, ``_repair_record_duration``, and
    ``_repair_n_records`` for per-field policy. ``n_signals`` empty (or
    any unparseable numeric field) is surfaced by ``_read_header_fields``
    with the field name and value in the error message.

    Returns a dict ``{field_name: new_value}`` listing the repairs made.
    """
    fields = _read_header_fields(edf_path)
    if fields["record_bytes"] <= 0:
        raise ValueError(
            f"Cannot repair {edf_path}: record_bytes derived from signal "
            f"headers is {fields['record_bytes']} (expected > 0)."
        )
    with open(edf_path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)

    repaired: dict = {}
    bih = _repair_bytes_in_header(edf_path, main, fields["n_signals"], verbosity)
    if bih is not None:
        repaired["bytes_in_header"] = bih
    rd = _repair_record_duration(edf_path, main,
                                  fields["samples_per_record"], verbosity)
    if rd is not None:
        repaired["record_duration"] = rd
    n_rec = _repair_n_records(edf_path, fields, verbosity)
    if n_rec is not None:
        repaired["n_records"] = n_rec
    return repaired


# Older fixtures, tests, and external callers import the
# truncation-only entry point. Forward to the merged repair pass.
def repair_truncated_edf_header(edf_path: str, verbosity: int = 1) -> int:
    """Backward-compatible alias for ``repair_main_header_numeric_fields``.

    Returns the n_records value now in the header (matching the legacy
    return contract). The merged function also repairs ``bytes_in_header``
    and ``record_duration`` if they were empty.
    """
    repair_main_header_numeric_fields(edf_path, verbosity=verbosity)
    return _read_header_fields(edf_path)["n_records"]


def _write_main_header_string_field(edf_path: str, offset: int,
                                     width: int, value: str) -> None:
    """Overwrite a fixed-width main-header field with ``value`` (ASCII
    space-padded on the right to ``width`` bytes)."""
    if len(value) > width:
        raise ValueError(
            f"value {value!r} does not fit in {width} ASCII bytes "
            f"(EDF spec limit for main-header field at offset {offset})."
        )
    encoded = value.ljust(width).encode("ascii")
    with open(edf_path, "r+b") as f:
        f.seek(offset)
        f.write(encoded)


# Values written for a signal whose physical range could not be preserved.
# Per EDF+ spec (section 2.1.3 item 5): "In case of uncalibrated signals,
# physical dimension is left empty (that is 8 spaces), while 'Physical
# maximum' and 'Physical minimum' must still contain different values
# (this is to avoid 'division by 0' errors by some viewers)."
UNCALIBRATED_PHYS_MIN = "-1"
UNCALIBRATED_PHYS_MAX = "1"
UNCALIBRATED_PHYS_DIM = ""   # rendered as 8 spaces after ljust()

# Full int16 range — EDF+ spec item 5 requires strict ``dig_max > dig_min``,
# and the spec calls out -32768 / +32767 as the canonical range for the
# EDF Annotations signal. A safe, spec-compliant default when we're
# already marking the channel as uncalibrated.
FALLBACK_DIG_MIN = "-32768"
FALLBACK_DIG_MAX = "32767"

def repair_degenerate_signal_ranges(edf_path: str, verbosity: int = 1) -> list:
    """Detect and repair signals with invalid EDF+ physical/digital range headers.

    Bad patterns detected:
    - ``phys_max == phys_min`` (triggers ``EDFLIB_FILE_ERRORS_PHYS_MAX``;
      linear digital-to-physical scaling is degenerate).
    - ``dig_max <= dig_min`` (triggers ``EDFLIB_FILE_ERRORS_DIG_MAX``;
      spec requires strict ``dig_max > dig_min``).
    - Either phys or dig field unparseable as a number.

    Note: ``phys_max < phys_min`` is **allowed** by the spec and left
    alone — it encodes a negative amplifier gain for the
    "negativity upward" Clinical Neurophysiology convention.

    Repair policy — all-or-nothing per signal to keep the header
    internally consistent:

    - If EITHER the phys range or the dig range is invalid on a signal,
      the signal is marked **uncalibrated** and BOTH sides are rewritten
      together:
        - phys_min -> -1, phys_max -> 1, physical_dimension -> 8 spaces
          (EDF+ spec section 2.1.3 item 5, "uncalibrated signal"
          convention)
        - dig_min -> -32768, dig_max -> 32767 (full int16 range)
      Raw sample bytes on disk are untouched; only the header's
      scaling metadata changes.

    - Rationale: physical value = phys_min + (dig - dig_min) *
      (phys_max - phys_min) / (dig_max - dig_min). Touching one side
      while keeping the other at its original (possibly valid) value
      would silently rescale the signal's physical interpretation.
      Marking the whole signal as uncalibrated is honest and keeps
      both sides of the mapping consistent.

    Returns a list of dicts describing every signal that was repaired;
    keys include signal_idx, label, phys_issue, dig_issue, and the
    original phys_min / phys_max / phys_dim / dig_min / dig_max values.
    """
    with open(edf_path, "rb") as f:
        main = f.read(MAIN_HEADER_BYTES)
        n_signals = _parse_int_field(
            main[N_SIGNALS_OFFSET:N_SIGNALS_OFFSET + N_SIGNALS_WIDTH],
            "n_signals", edf_path=edf_path,
        )
        if n_signals is None:
            raise _empty_field_error(
                "n_signals", edf_path=edf_path,
                why_unrecoverable=(
                    "every per-signal field-slice depends on knowing N. "
                    "No safe default — file is unreadable as EDF."
                ),
            )
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

        # If ANY range is invalid, mark the whole signal uncalibrated
        # (rewrite BOTH phys and dig together) so the header's scaling
        # mapping stays self-consistent.
        msg_parts = [f"Signal {i} ({label!r}) has an invalid EDF+ header:"]
        if phys_reason:
            msg_parts.append(f"  physical range issue: {phys_reason}.")
        if dig_reason:
            msg_parts.append(f"  digital range issue: {dig_reason}.")
        msg_parts.append(
            f"  Marking signal as uncalibrated (EDF+ spec section 2.1.3 "
            f"item 5): rewriting phys_min={UNCALIBRATED_PHYS_MIN}, "
            f"phys_max={UNCALIBRATED_PHYS_MAX}, "
            f"units (physical_dimension)=<8 spaces>, "
            f"dig_min={FALLBACK_DIG_MIN}, dig_max={FALLBACK_DIG_MAX}. "
            f"Raw sample bytes on disk are untouched."
        )
        msg_parts.append(
            f"  Original values: phys_min={pmin_str!r}, "
            f"phys_max={pmax_str!r}, units={pdim_str!r}, "
            f"dig_min={dmin_str!r}, dig_max={dmax_str!r}."
        )
        if verbosity >= 1:
            print("WARNING: " + "\n".join(msg_parts))

        # Rewrite both ranges together whenever either side is invalid.
        _write_signal_field(edf_path, n_signals, i,
                            SIG_PHYS_MIN_OFFSET, SIG_PHYS_MIN_WIDTH,
                            UNCALIBRATED_PHYS_MIN)
        _write_signal_field(edf_path, n_signals, i,
                            SIG_PHYS_MAX_OFFSET, SIG_PHYS_MAX_WIDTH,
                            UNCALIBRATED_PHYS_MAX)
        _write_signal_field(edf_path, n_signals, i,
                            SIG_PHYS_DIM_OFFSET, SIG_PHYS_DIM_WIDTH,
                            UNCALIBRATED_PHYS_DIM)
        _write_signal_field(edf_path, n_signals, i,
                            SIG_DIG_MIN_OFFSET, SIG_DIG_MIN_WIDTH,
                            FALLBACK_DIG_MIN)
        _write_signal_field(edf_path, n_signals, i,
                            SIG_DIG_MAX_OFFSET, SIG_DIG_MAX_WIDTH,
                            FALLBACK_DIG_MAX)
        repairs.append({
            "signal_idx": i,
            "label": label,
            "phys_issue": phys_reason,
            "dig_issue": dig_reason,
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
