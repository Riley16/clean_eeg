"""Preprocess a REDCap CSV: propagate subject-level fields onto every row
of the same subject before the schema-driven redaction runs.

Why: with a longitudinal REDCap export, the patient_information row for
subject X carries `subject_name` and `implant_date`, but the
daily_testing_report rows for the same subject share `subject_number`
and leave those columns empty. To let per-row redaction:

  * scan every note field for the patient's name (needs `subject_name`)
  * shift every date relative to implant_date (needs `implant_date`)

… we fill those columns onto every row of the same subject. After
redaction, the propagated `subject_name` cells get blanked by
`constant_replace` and `implant_date` cells land on the base date — so
the propagation doesn't create new PHI leaks in the output.

Usage:
    python -m scripts.text_phi.inspect.preprocess_redcap \\
        --input DATA.csv --output DATA_preprocessed.csv \\
        [--subject-key subject_number] \\
        [--propagate subject_name,implant_date]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _first_nonempty(series: pd.Series) -> str:
    """Return the first non-empty string value in the series, or ''."""
    for v in series:
        if isinstance(v, str) and v.strip():
            return v
    return ""


def propagate_subject_fields(
    df: pd.DataFrame,
    subject_key: str,
    fields: list[str],
) -> pd.DataFrame:
    """Fill empty cells in `fields` with the same-subject value from any
    row where the field is populated. Non-empty cells are left as-is."""
    if subject_key not in df.columns:
        raise ValueError(f"subject key column {subject_key!r} not in CSV")
    for f in fields:
        if f not in df.columns:
            raise ValueError(f"propagate field {f!r} not in CSV")

    df = df.copy()
    for f in fields:
        first_per_subj = df.groupby(subject_key, sort=False)[f].transform(
            _first_nonempty
        )
        mask_empty = df[f].fillna("").astype(str).str.strip() == ""
        df.loc[mask_empty, f] = first_per_subj[mask_empty]
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to source CSV")
    p.add_argument("--output", required=True, help="Path to write preprocessed CSV")
    p.add_argument("--subject-key", default="subject_number",
                   help="Column that identifies a subject across rows "
                        "(default: subject_number)")
    p.add_argument("--propagate", default="subject_name,implant_date",
                   help="Comma-separated columns to fill down within each "
                        "subject (default: subject_name,implant_date)")
    args = p.parse_args(argv)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    fields = [f.strip() for f in args.propagate.split(",") if f.strip()]

    out_df = propagate_subject_fields(df, args.subject_key, fields)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    # Report fill deltas so the user can sanity-check.
    for f in fields:
        was = (df[f].fillna("").astype(str).str.strip() != "").sum()
        now = (out_df[f].fillna("").astype(str).str.strip() != "").sum()
        print(f"  {f}: {was} → {now} filled rows (+{now - was})",
              file=sys.stderr)
    print(f"Wrote {out} — {len(out_df)} rows.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
