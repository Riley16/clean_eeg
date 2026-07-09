"""Extract column names from a CSV using pandas.

Emits a JSON array of column names (one per CSV column). Useful for
confirming the CSV header matches what the metadata expects before running
the full inspection.

Usage:
    python -m scripts.text_phi.inspect.extract_columns \\
        --input DATA.csv --output columns.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def extract_columns(csv_path: str | Path) -> list[str]:
    """Read only the header row of a CSV and return the column names."""
    df = pd.read_csv(csv_path, nrows=0)
    return list(df.columns)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to CSV file")
    p.add_argument("--output", required=True,
                   help="Path to write JSON array of column names")
    args = p.parse_args(argv)
    cols = extract_columns(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cols, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"Wrote {out} — {len(cols)} columns", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
