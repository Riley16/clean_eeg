"""Subject-name data structures.

Kept free of presidio/spaCy imports so lightweight, read-only tools (e.g.
``detect-edf-names``) can work with names without pulling in the NLP stack.
``anonymize`` re-exports these, so ``from clean_eeg.anonymize import
PersonalName`` keeps working.
"""

from typing import List


def normalize_name_token(name: str) -> str:
    # passthrough
    return name


class PersonalName():
    def __init__(self,
                 first_name: str,
                 middle_names: List[str],
                 last_name: str):
        self.first_name = first_name
        self.last_name = last_name
        self.middle_names = middle_names

    def get_full_name(self) -> str:
        """
        Get the full name as a string.
        """
        names = [self.first_name] + self.middle_names + [self.last_name]
        return " ".join(names).strip()

    def get_normalized_tokens(self) -> List[str]:
        """
        Get the normalized tokens of the full name.
        """
        return [normalize_name_token(self.first_name),
                *[normalize_name_token(m) for m in self.middle_names],
                normalize_name_token(self.last_name)]
