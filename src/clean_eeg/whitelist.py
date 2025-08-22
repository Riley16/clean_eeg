import json
import regex as re
from typing import Set

from clean_eeg.paths import DATA_DIR, AUTO_WORD_WHITELIST_PATH, MANUAL_WORD_WHITELIST_PATH


def load_whitelist() -> Set[str]:
    """
    Load a whitelist of common words from a JSON file.
    The file should contain a list of words.
    """
    with open(AUTO_WORD_WHITELIST_PATH, 'r', encoding='utf-8') as f:
        auto_words = json.load(f)
    with open(MANUAL_WORD_WHITELIST_PATH, 'r', encoding='utf-8') as f:
        manual_words = json.load(f)
    white_list = set(auto_words + manual_words)
    white_list = filter(lambda x: isinstance(x, str), white_list)
    white_list = set(map(str.lower, white_list))
    return white_list

NAME_WORD_RE = re.compile(r"\b\p{L}+(?:['’\-]\p{L}+)*\b", re.UNICODE)

def token_in_whitelist(token: str, whitelist: Set[str]) -> bool:
    """
    Returns True if token (or its punctuation-stripped variant) is in the whitelist.
    """
    t = token.lower()
    if t in whitelist:
        return True
    t2 = re.sub(r"[-'’]", "", t)
    return t2 in whitelist
