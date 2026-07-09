"""Session-scoped fixtures for the text-phi test suite.

Presidio + spaCy setup costs ~1s per analyzer. These fixtures build each
analyzer once per session so the full test file runs in seconds.
"""

from __future__ import annotations

import pytest

from clean_eeg.anonymize import PersonalName
from scripts.text_phi.detectors import build_generic_analyzer
from scripts.text_phi.redactor import TextRedactor


@pytest.fixture(scope="session")
def generic_analyzer():
    return build_generic_analyzer(enable_zip=True, enable_age=True, mrn_regex=None)


@pytest.fixture(scope="session")
def subject_pn() -> PersonalName:
    return PersonalName(first_name="John", middle_names=["P"], last_name="O'Connor")


@pytest.fixture(scope="session")
def subject_redactor(subject_pn) -> TextRedactor:
    return TextRedactor(mode="subject", subject_names=[subject_pn])


@pytest.fixture(scope="session")
def generic_redactor() -> TextRedactor:
    return TextRedactor(mode="generic")


@pytest.fixture(scope="session")
def both_redactor(subject_pn) -> TextRedactor:
    return TextRedactor(mode="both", subject_names=[subject_pn])
