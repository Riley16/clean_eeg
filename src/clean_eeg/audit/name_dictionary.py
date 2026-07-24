"""US name dictionary loader for the audit's annotation PHI scan.

Loads the derived set of unique casefolded first + last names from the
gzipped text file shipped in ``data/name_dictionary/us_names.txt.gz``.
That file is generated once from the raw ``name_dataset`` CSVs (see
``scripts/build_us_name_dictionary.py``) and committed to the repo so
the audit runs out-of-the-box on fresh installs — including cluster
deployments that never provisioned the ~10 GB raw dataset.

Format is deliberately simple: newline-delimited UTF-8 text, sorted,
casefolded, gzipped. Human-inspectable when gunzipped, cross-language
readable, no arbitrary-code-execution risk on load (unlike pickle).

The loader still supports building from raw CSVs when the caller
requests countries other than US or explicitly asks to regenerate;
that path uses :func:`build_name_set_from_csvs`.
"""

from __future__ import annotations

import functools
import gzip
from pathlib import Path

from clean_eeg.paths import DATA_DIR


# Shipped, ready-to-use derived artifact — the primary load path.
US_NAMES_TXT_GZ = DATA_DIR / "name_dictionary" / "us_names.txt.gz"

# Raw dataset location for the regeneration / other-country fallback
# path. This directory is gitignored — it holds ~10 GB of source CSVs
# that must be provisioned separately (see scripts/build_whitelist.py).
NAME_DATA_PATH = DATA_DIR / "name_dataset" / "data"


class NameDictionaryUnavailable(RuntimeError):
    """Raised when neither the shipped gzipped names file nor the raw
    CSV source is accessible for the requested countries. Distinct
    exception type so callers can degrade gracefully (skip the
    name-dictionary scan) rather than crashing the whole audit."""


def _load_from_gzip_text(path: Path) -> frozenset[str]:
    """Read the newline-delimited gzipped names file into a frozenset."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return frozenset(line.rstrip("\n") for line in f if line.strip())


def build_name_set_from_csvs(countries: tuple[str, ...]) -> frozenset[str]:
    """Union first + last names from the raw per-country CSVs
    (``FirstName, LastName, Gender, Country`` columns), casefolded.

    Used when the caller asks for countries other than US (the shipped
    gzip covers US only) or wants to regenerate the derived set.
    Requires ``pandas`` and the raw dataset on disk; raises
    :class:`NameDictionaryUnavailable` if the CSVs are missing.
    """
    import pandas as pd  # lazy: pandas is heavy and only this path needs it

    frames = []
    for c in countries:
        path = NAME_DATA_PATH / f"{c.upper()}.csv"
        if not path.exists():
            raise NameDictionaryUnavailable(
                f"Raw name-dataset CSV not found at {path}. The gzipped "
                "shipped file covers US only; other countries require "
                "the raw dataset. See scripts/build_whitelist.py."
            )
        frames.append(pd.read_csv(
            path, names=["FirstName", "LastName", "Gender", "Country"]))
    df = pd.concat(frames, ignore_index=True)
    names = set(df["FirstName"].dropna().unique().tolist()) \
        | set(df["LastName"].dropna().unique().tolist())
    return frozenset(n.casefold() for n in names
                     if isinstance(n, str) and n)


@functools.lru_cache(maxsize=4)
def load_us_name_dictionary(countries: tuple[str, ...] = ("US",)) -> frozenset[str]:
    """Return the lowercased union of first + last names for ``countries``.

    Fast path (default ``countries=("US",)``): decode the shipped gzip'd
    text file. ~0.5 s cold, memoized in-process for zero-cost reuse
    across subjects when auditing in ``--parent`` mode.

    Fallback path (other countries, or shipped file missing): rebuild
    from the raw CSVs via :func:`build_name_set_from_csvs`. Raises
    :class:`NameDictionaryUnavailable` if neither source is accessible.
    """
    if countries == ("US",) and US_NAMES_TXT_GZ.exists():
        return _load_from_gzip_text(US_NAMES_TXT_GZ)

    if countries == ("US",):
        # US requested but shipped file missing — degrade to CSV rebuild
        # so a repo cloned without the LFS/large-file blob still works
        # if the operator has the raw CSVs.
        try:
            return build_name_set_from_csvs(countries)
        except NameDictionaryUnavailable:
            raise NameDictionaryUnavailable(
                f"Neither {US_NAMES_TXT_GZ} nor the raw CSV at "
                f"{NAME_DATA_PATH / 'US.csv'} is available. Check that "
                "the shipped file made it through the git checkout "
                "(should be ~15 MB in data/name_dictionary/)."
            )

    return build_name_set_from_csvs(countries)
