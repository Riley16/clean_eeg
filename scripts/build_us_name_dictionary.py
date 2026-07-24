"""Regenerate ``data/name_dictionary/us_names.txt.gz`` from the raw
``name_dataset`` US.csv.

Run once when the source dataset is refreshed. Requires the raw
dataset (10 GB extracted; download at https://github.com/philipperemy/name-dataset)
to be present at ``data/name_dataset/data/US.csv``.

    python scripts/build_us_name_dictionary.py

Writes ~15 MB of gzipped text. That file IS tracked in the repo (via a
``.gitignore`` exception under ``data/name_dictionary/``) so a fresh
clone can run the audit without provisioning the raw dataset.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd

from clean_eeg.audit.name_dictionary import NAME_DATA_PATH, US_NAMES_TXT_GZ


def main() -> int:
    src = NAME_DATA_PATH / "US.csv"
    dst = US_NAMES_TXT_GZ
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"ERROR: raw US.csv not found at {src}", file=sys.stderr)
        print("Provision the name_dataset first — see the module docstring.",
              file=sys.stderr)
        return 1

    print(f"Reading {src} ({src.stat().st_size / 1024**2:.1f} MB)...")
    df = pd.read_csv(src, names=["FirstName", "LastName", "Gender", "Country"])
    print(f"Rows: {len(df):,}")

    names = (set(df["FirstName"].dropna().unique().tolist())
             | set(df["LastName"].dropna().unique().tolist()))
    casefolded = sorted({n.casefold() for n in names
                          if isinstance(n, str) and n})
    print(f"Unique casefolded names: {len(casefolded):,}")

    # compresslevel=9 for smallest file (one-time build cost is trivial
    # vs. the bytes shipped forever in git history).
    with gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as f:
        for n in casefolded:
            f.write(n)
            f.write("\n")

    print(f"Wrote {dst} ({dst.stat().st_size / 1024**2:.2f} MB gzipped)")
    print(f"Compression ratio vs source: "
          f"{src.stat().st_size / dst.stat().st_size:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
