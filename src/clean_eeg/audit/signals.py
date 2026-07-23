"""Windowed signal reader for the audit notebook's EEG snippet plots.

Reads only the first ``window_seconds`` of data via ``open/seek/read``
and numpy ``frombuffer``, staying independent of pyedflib so it keeps
working on files pyedflib refuses to open. Signal quality checks
(dead-channel detection, RMS, etc.) are out of scope for this PHI-
focused audit and will be handled separately at scale.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from clean_eeg.print_edf_header import (
    EDF_ANNOTATION_LABEL,
    MAIN_HEADER_BYTES,
    SIGNAL_HEADER_BYTES_PER_SIGNAL,
    read_main_header,
    read_signal_headers,
)


def read_signal_window(edf_path: str | Path,
                       *,
                       window_seconds: float = 10.0
                       ) -> dict[str, np.ndarray]:
    """Return ``{channel_label: int16 array}`` for the first
    ``window_seconds`` of data. Skips the ``EDF Annotations`` channel.
    Returns ``{}`` on any parse error (broken file, no data records,
    unparseable geometry).
    """
    p = Path(edf_path)
    header = read_main_header(str(p))
    n_signals = header.get("n_signals")
    n_records = header.get("n_records")
    rec_dur = header.get("record_duration")
    if (not isinstance(n_signals, int) or not isinstance(n_records, int)
            or not isinstance(rec_dur, float)
            or n_signals <= 0 or n_records <= 0 or rec_dur <= 0):
        return {}

    sigs = read_signal_headers(str(p), n_signals)
    spr = [s.get("samples_per_record") for s in sigs]
    if not all(isinstance(v, int) and v > 0 for v in spr):
        return {}

    total_spr = sum(spr)
    record_bytes = total_spr * 2
    n_window_records = max(1, min(n_records, math.ceil(window_seconds / rec_dur)))
    header_end = MAIN_HEADER_BYTES + n_signals * SIGNAL_HEADER_BYTES_PER_SIGNAL
    with open(p, "rb") as f:
        f.seek(header_end)
        buf = f.read(n_window_records * record_bytes)
    if len(buf) < n_window_records * record_bytes:
        return {}

    arr = np.frombuffer(buf, dtype="<i2").reshape(n_window_records, total_spr)
    out: dict[str, np.ndarray] = {}
    offset = 0
    for i, s in enumerate(sigs):
        label = str(s.get("label", f"ch{i}")).strip() or f"ch{i}"
        if label == EDF_ANNOTATION_LABEL:
            offset += spr[i]
            continue
        out[label] = arr[:, offset:offset + spr[i]].ravel().copy()
        offset += spr[i]
    return out


