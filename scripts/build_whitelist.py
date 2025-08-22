import os
import json
import pandas as pd
from typing import Set
from functools import lru_cache

from clean_eeg.paths import DATA_DIR, AUTO_WORD_WHITELIST_PATH


@lru_cache(maxsize=10)
def load_subtletitlex_whitelist(path: str, top_n: int = 20000) -> Set[str]:
    """
    Build a whitelist of common words from SUBTLEX-US (Brysbaert & New, 2008).

    We sort by frequency (FREQcount if available; otherwise by Lg10WF descending),
    normalize via casefold, and keep alphabetic + internal apostrophes/hyphens.

    Citation:
    Brysbaert, M., New, B. Moving beyond Kučera and Francis: A critical evaluation of current 
    word frequency norms and the introduction of a new and improved word frequency measure for 
    American English. Behavior Research Methods 41, 977–990 (2009). https://doi.org/10.3758/BRM.41.4.977
    downloaded from https://osf.io/7wx25 on 2025-08-21

    """
    df = pd.read_excel(path)

    # Pick a sort key that exists in your file
    sort_cols = [c for c in ["FREQcount", "Lg10WF"] if c in df.columns]
    if not sort_cols:
        raise ValueError("Expected a frequency column like 'FREQcount' or 'Lg10WF' in SUBTLEX file.")

    df_sorted = df.sort_values(by=sort_cols[0], ascending=False)
    # If FREQcount is numeric, descending; if it's a string (rare), fall back to Lg10WF; adjust as needed.

    # Extract words
    if "Word" not in df.columns:
        raise ValueError("Could not find a 'Word' column.")

    words = []
    import re
    pat = re.compile(r"^[^\W\d_]+(?:[-'’][^\W\d_]+)*$")
    for w in df_sorted["Word"].astype(str).head(top_n):
        w = w.strip()
        if not w:
            continue
        # Keep alphabetic tokens with optional internal ' or -
        if pat.match(w):
            words.append(w.casefold())

    # Remove possessive "'s" and bare "'"
    wl = {re.sub(r"(?:['’]s)$", "", w) for w in words}
    return wl


NAME_LIST_PATH = os.path.join(DATA_DIR, 'name_list.json')

@lru_cache(maxsize=10)
def load_names(dataset_name: str = 'name_dataset') -> Set[str]:
    if dataset_name == 'nicknames':
        all_names = load_nicknames_dataset_names()
    elif dataset_name == 'name_dataset':
        all_names = load_names_dataset_names()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Use 'nicknames' or 'name_dataset'.")
    return all_names


def load_nicknames_dataset_names() -> Set[str]:
    # https://github.com/carltonnorthern/nicknames
    from nicknames import with_names_csv_path
    with with_names_csv_path() as f:
        print("Loading nicknames from:", f)
        names = pd.read_csv(f)
    all_names = set(list(names["name1"].unique()) + list(names["name2"].unique()))
    return all_names


NAME_DATA_PATH = DATA_DIR / 'name_dataset' / 'data'

@lru_cache(maxsize=10)
def load_names_dataset_names(countries=['US']) -> Set[str]:
    # https://github.com/philipperemy/name-dataset
    print('Building name list from:', NAME_DATA_PATH)
    # load names from CSV files in the name dataset
    names_df = list()
    for name_path in os.listdir(NAME_DATA_PATH):
        if os.path.splitext(name_path)[0] not in countries:
            continue
        if os.path.splitext(name_path)[1] != '.csv':
            continue
        path = os.path.join(NAME_DATA_PATH, name_path)
        df = pd.read_csv(path, names=["FirstName", "LastName", 'Gender', 'Country'])
        names_df.append(df)
    names_df = pd.concat(names_df, ignore_index=True)
    all_names = set(list(names_df["FirstName"].unique()) + list(names_df["LastName"].unique()))
    return all_names
    

if __name__ == "__main__":
    # build white list of words

    # build base white list of common words from SUBTITLEX-US corpus (Brysbaert & New, 2008)
    path = os.path.join(DATA_DIR, 'SUBTLEX-US_frequency_list_PoS_Zipf.xlsx')
    print('Building base white list from SUBTITLEX-US dataset from:', path)
    subtitlex_wl = load_subtletitlex_whitelist(path, top_n=50000)

    all_names = load_names('name_dataset')

    all_names = filter(lambda x: isinstance(x, str), all_names)
    all_names = set(map(str.lower, all_names))
    
    wl = subtitlex_wl - all_names
    print('Words before filtering:', len(subtitlex_wl))
    print('Names filtered:', len(all_names))
    print('Final whitelist size:', len(wl))

    with open(AUTO_WORD_WHITELIST_PATH, 'w') as f:
        json.dump(list(wl), f)
        print('Saved automatically generated whitelist to:', AUTO_WORD_WHITELIST_PATH)
