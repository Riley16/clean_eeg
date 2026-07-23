"""Disk-cached loader for the US name dictionary.

The raw ``name_dataset`` build (see [scripts/build_whitelist.py]) reads
a 32M-row CSV and produces a ~4M-name set — ~23s on cold start every
audit run. Since the audit's name dictionary should be identical for
every subject in a given deployment, we cache the derived set to
``data/name_dictionary_cache/<countries>.pkl`` and rebuild only when
the source CSVs are newer than the cache.

The cache is a plain pickle of a ``frozenset[str]``. It's kept under
``data/`` (gitignored) because it's a derived artifact, one per machine.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from clean_eeg.paths import DATA_DIR


NAME_DATA_PATH = DATA_DIR / "name_dataset" / "data"
_CACHE_DIR = DATA_DIR / "name_dictionary_cache"


def _build_name_set(countries: tuple[str, ...]) -> frozenset[str]:
    """Load first + last name columns from the requested country CSVs
    (columns: ``FirstName, LastName, Gender, Country``), union them,
    and casefold. Same output as
    ``scripts.build_whitelist.load_names_dataset_names`` but reachable
    from an installed package (which cannot import top-level
    ``scripts``).
    """
    frames = []
    for c in countries:
        path = NAME_DATA_PATH / f"{c.upper()}.csv"
        frames.append(pd.read_csv(
            path, names=["FirstName", "LastName", "Gender", "Country"]))
    df = pd.concat(frames, ignore_index=True)
    names = set(df["FirstName"].dropna().unique().tolist()) \
        | set(df["LastName"].dropna().unique().tolist())
    return frozenset(n.casefold() for n in names
                     if isinstance(n, str) and n)


def _cache_path(countries: tuple[str, ...]) -> Path:
    key = "_".join(sorted(c.upper() for c in countries))
    return _CACHE_DIR / f"{key}.pkl"


def _cache_is_fresh(cache_path: Path, countries: tuple[str, ...]) -> bool:
    """True iff the cache exists and is newer than every source CSV for
    the requested countries. Missing source files → cache is not fresh
    (safest default: force rebuild)."""
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for c in countries:
        csv_path = Path(NAME_DATA_PATH) / f"{c.upper()}.csv"
        if not csv_path.exists():
            return False
        if csv_path.stat().st_mtime > cache_mtime:
            return False
    return True


def load_us_name_dictionary(countries: tuple[str, ...] = ("US",)) -> frozenset[str]:
    """Return the lowercased union of first + last names for ``countries``.

    Disk-cached: subsequent audits in the same deployment reuse the
    pickled set (~90 MB → ~1 s load). Delete
    ``data/name_dictionary_cache/`` to force rebuild.
    """
    cache_path = _cache_path(countries)
    if _cache_is_fresh(cache_path, countries):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    names = _build_name_set(countries)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(names, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)  # atomic swap so a killed run can't corrupt
    return names
