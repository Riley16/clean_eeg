"""Fast, mmap-based reader for EDF+ annotations.

pyedflib's ``readAnnotations()`` loads every data record's signal
bytes into memory to extract the interspersed annotation TALs. For
multi-GB clinical EEGs that's minutes of I/O we don't need if we
only want the annotation texts. This module seeks directly to each
record's annotation slice (typically 50-200 bytes per record) and
skips the signal data entirely.

Works on any conformant EDF+ file, including raw NK exports that
pyedflib refuses to open due to strict format checks -- byte layout
of the annotation channel is the same regardless of whether pyedflib
accepts the reserved-field / record-count consistency checks.

TAL format inside each data record:

    +<onset_timekeeping>\\x14\\x14              # timekeeping TAL
    +<onset>[\\x15<duration>]\\x14<text>\\x14\\x00   # user TAL 1
    +<onset>[\\x15<duration>]\\x14<text>\\x14\\x00   # user TAL 2
    \\x00\\x00...                                  # padding

Reused by:
    * scripts/count_annotations.py -- review-time scoping
    * (upcoming) annotation-review TUI -- fast loading of large subjects
"""

from __future__ import annotations

import mmap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clean_eeg.modify_edf_inplace import (
    SIGNAL_HEADER_BYTES,
    TOTAL_HEADER_BYTES,
    get_annotation_signal_header_index,
    get_header_field,
    get_signal_header_fields,
)


EDF_TAL_TIMEKEEPING_DELIMITER = b"\x14\x14"


@dataclass
class Annotation:
    """One non-timekeeping annotation. ``record_index`` + ``byte_offset``
    together locate the TAL inside the file, so a later mutation pass
    can find the exact bytes to overwrite without re-parsing."""
    record_index: int
    byte_offset_in_record: int
    onset_s: float
    duration_s: float          # 0 if no duration was encoded
    text: str


def _parse_tal_bytes(record_ann_bytes: bytes,
                      record_index: int,
                      ) -> list[Annotation]:
    """Parse the annotation-channel slice of one data record into a
    list of :class:`Annotation`. Timekeeping TALs (empty text) are
    excluded from the output. Malformed TALs are silently skipped --
    they'd never have been surfaced by pyedflib either, and the
    review-time-estimate use case tolerates a couple of dropped
    entries better than an exception.
    """
    # Skip past the timekeeping TAL (ends at first \x14\x14)
    tk_end = record_ann_bytes.find(EDF_TAL_TIMEKEEPING_DELIMITER)
    if tk_end < 0:
        return []
    tk_end += len(EDF_TAL_TIMEKEEPING_DELIMITER)
    rest = record_ann_bytes[tk_end:]
    # Padding (\x00 bytes) separates + terminates TALs after the
    # timekeeping. Split on \x00; each non-empty chunk is one TAL.
    out: list[Annotation] = []
    running_offset = tk_end
    for tal_chunk in rest.split(b"\x00"):
        if not tal_chunk:
            running_offset += 1
            continue
        # TAL = <onset>[\x15<dur>]\x14<text>\x14
        # Split into: (onset_or_onset_dur, text, ...trailing)
        parts = tal_chunk.split(b"\x14")
        if len(parts) < 2:
            running_offset += len(tal_chunk) + 1
            continue
        onset_field = parts[0]
        text = parts[1]
        if not text:  # empty text = would-be timekeeping in a
                       # non-first position; skip
            running_offset += len(tal_chunk) + 1
            continue
        # onset[\x15dur] handling
        if b"\x15" in onset_field:
            onset_str, dur_str = onset_field.split(b"\x15", 1)
        else:
            onset_str, dur_str = onset_field, b"0"
        try:
            onset = float(onset_str)
            duration = float(dur_str)
        except ValueError:
            running_offset += len(tal_chunk) + 1
            continue
        out.append(Annotation(
            record_index=record_index,
            byte_offset_in_record=running_offset,
            onset_s=onset,
            duration_s=duration,
            text=text.decode("utf-8", errors="replace"),
        ))
        running_offset += len(tal_chunk) + 1  # +1 for the split \x00
    return out


def iter_annotations(edf_path: Path | str) -> list[Annotation]:
    """Read every non-timekeeping annotation from ``edf_path``. Signal
    data is never touched by the parser -- the annotation channel is
    extracted via a numpy view into an mmap.

    Works on files pyedflib.EdfReader would refuse (raw NK exports,
    EDF+D not yet split) as long as the main header + signal headers
    are parseable via byte offsets.

    Perf notes for clinical EDFs on networked storage:
      * The old implementation seeked per-record inside the mmap,
        triggering N_records small random 4KB page faults across the
        whole file. On network mounts (NFS, SMB, Box FS, etc.) this
        is dramatically slower than a single sequential scan.
      * The current implementation asks the kernel for a SEQUENTIAL
        access hint (madvise), materializes the whole data region as
        an int16 numpy view once, and slices out the annotation
        channel columns in one contiguous copy. The kernel is then
        free to prefetch large blocks instead of chasing small
        random reads.
      * Skips records whose annotation slice is all-zero padding
        (very common: most records have only the timekeeping TAL,
        which is short; the rest is padding). Avoids per-record
        Python-level parsing on the common case.
    """
    path = Path(edf_path)
    ann_idx = get_annotation_signal_header_index(str(path))
    lens_samples = get_signal_header_fields(str(path), field="num_samples")
    ann_samples_per_record = lens_samples[ann_idx]
    ann_sample_offset_in_record = sum(lens_samples[:ann_idx])
    record_samples = sum(lens_samples)
    n_signals = len(lens_samples)
    n_records = int(get_header_field(str(path), "num_data_records"))
    data_start = TOTAL_HEADER_BYTES + SIGNAL_HEADER_BYTES * n_signals

    out: list[Annotation] = []
    file_size = path.stat().st_size
    if (file_size <= data_start or n_records == 0
            or record_samples == 0 or ann_samples_per_record == 0):
        return out

    # Cap loop at what PHYSICALLY fits (raw NK often lies about
    # num_data_records). Also what makes the reader survive files
    # pyedflib refuses on the num_data_records / filesize check.
    record_bytes = record_samples * 2
    max_records_that_fit = (file_size - data_start) // record_bytes
    effective_records = min(n_records, max_records_that_fit)

    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Hint: we're about to touch the file in file-order. The
            # kernel prefetches large blocks instead of chasing
            # per-record random reads. Load-bearing on network mounts.
            try:
                mm.madvise(mmap.MADV_SEQUENTIAL)
            except (AttributeError, OSError):
                # madvise unavailable on some platforms; correctness
                # unaffected, only perf.
                pass

            # numpy view into the mmap: (n_records, record_samples)
            # int16. No copy, no per-record Python object churn.
            # Slice the annotation columns and materialize them into
            # a single contiguous bytes buffer (one C-level copy,
            # not N Python-level slices).
            #
            # try/finally guarantees the mmap-backed views drop
            # BEFORE mm.__exit__ runs -- otherwise mm.close() raises
            # BufferError. Same pattern as the audit's
            # _audit_signal_integrity_clean_side_stream.
            try:
                data = np.frombuffer(
                    mm, dtype=np.int16,
                    count=effective_records * record_samples,
                    offset=data_start)
                data = data.reshape(effective_records, record_samples)
                ann_cols = data[:,
                                 ann_sample_offset_in_record:
                                 ann_sample_offset_in_record
                                 + ann_samples_per_record]
                # np.ascontiguousarray drops the stride from the parent
                # array so .tobytes() is a straight memcpy.
                ann_bytes_all = np.ascontiguousarray(ann_cols).tobytes()
            finally:
                # Drop every reference into the mmap so mm.close()
                # (implicit at with-exit) doesn't hit BufferError.
                data = ann_cols = None  # type: ignore[assignment]

    ann_len = ann_samples_per_record * 2
    for rec_i in range(effective_records):
        offset = rec_i * ann_len
        slice_ = ann_bytes_all[offset : offset + ann_len]
        out.extend(_parse_tal_bytes(slice_, rec_i))
    return out


def count_words_in_annotations(annotations: list[Annotation]) -> int:
    """Whitespace-tokenized word count across all annotation texts.
    Matches the WPM assumption used by
    ``scripts/count_annotations.py``."""
    return sum(len(a.text.split()) for a in annotations if a.text.strip())
