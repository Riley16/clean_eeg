"""Save the PRE-Presidio annotation text to a sibling directory of the
cleaned-EDF output, so operators can audit what the pipeline actually
touched without re-running or reasoning backwards from the redacted
output.

Sibling naming convention:
    <parent>/clinical_eeg/                  -- cleaned EDFs (what transfers)
    <parent>/clinical_eeg_original_annotations/   -- raw annotation dump
                                                     (NEVER transfers)

The sibling directory holds one JSON file per EDF:
    {
      "source_edf": "<filename>.edf",
      "n_annotations": 42,
      "annotations": [
        {"onset": 0.5, "duration": -1.0, "text": "raw annotation text"},
        ...
      ]
    }

Content is RAW pre-redaction text and therefore CONTAINS PHI. The
transfer layer explicitly excludes this directory via both:
  1. sibling placement (transfer source is `.../clinical_eeg/`, sibling
     is outside)
  2. an rsync --exclude belt-and-suspenders in build_transfer_plan
  3. a hard-fail preflight assertion that catches any refactor putting
     the sibling INSIDE the transfer source
"""

from __future__ import annotations

import json
from pathlib import Path


# Suffix appended to the cleaned-EDF subdir's name to form the sibling.
# Parameterized rather than hardcoded to "clinical_eeg_original_annotations"
# so the convention works for any subfolder name; the user's workflow
# uses "clinical_eeg", but batch runs on other subfolders would get
# "<subfolder>_original_annotations" siblings.
ORIGINAL_ANNOTATIONS_SUFFIX = "_original_annotations"


def sibling_dir_for(output_path: str | Path) -> Path:
    """Return the sibling-directory path where the raw annotations for
    ``output_path`` should be saved. Does NOT create the directory.

    ``output_path`` is the cleaned-EDF subdir (typically
    ``<subject>/clinical_eeg/``). Sibling is
    ``<subject>/clinical_eeg_original_annotations/``.
    """
    p = Path(output_path).resolve()
    return p.parent / (p.name + ORIGINAL_ANNOTATIONS_SUFFIX)


def save_raw_annotations(output_path: str | Path,
                          source_filename: str,
                          annotations) -> Path:
    """Write one JSON file of the raw annotations for one source EDF.

    ``output_path`` is the cleaned-EDF subdir (the sibling is derived).
    ``source_filename`` is the basename of the source EDF (e.g.
    ``foo.edf``); the JSON file lands at
    ``<sibling>/<basename-without-.edf>.json``.
    ``annotations`` is the ``(onsets, durations, texts)`` tuple that
    ``load_edf`` returns.

    Overwrites any prior file at the destination -- on a --force
    re-clean, the fresh raw dump replaces the stale one. Returns the
    written path.
    """
    sibling = sibling_dir_for(output_path)
    sibling.mkdir(parents=True, exist_ok=True)
    dest = sibling / (Path(source_filename).stem + ".json")

    onsets, durations, texts = annotations
    payload = {
        "source_edf": source_filename,
        "n_annotations": int(len(texts)),
        "annotations": [
            {"onset": float(o), "duration": float(d), "text": str(t)}
            for o, d, t in zip(onsets, durations, texts)
        ],
    }
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return dest


def sibling_dir_inside(transfer_source: str | Path) -> Path | None:
    """Defensive check for the transfer layer: if the sibling directory
    has somehow ended up INSIDE the transfer source (e.g. a refactor
    changed the source path to the subject root instead of the
    clinical_eeg subdir), return the offending path so the caller can
    refuse to proceed. Returns None when the sibling is not inside the
    transfer source (the expected safe state).

    Recurses one level; the sibling directory is specifically named
    with the ``_original_annotations`` suffix so a bare glob is enough.
    """
    src = Path(transfer_source).resolve()
    if not src.is_dir():
        return None
    for entry in src.rglob("*" + ORIGINAL_ANNOTATIONS_SUFFIX):
        if entry.is_dir():
            return entry
    return None
