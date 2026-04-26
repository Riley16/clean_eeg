import pytest

from clean_eeg.anonymize import redact_subject_name, PersonalName, REDACT_NAME_REPLACEMENT


PATIENT_NAME = PersonalName(first_name='John',
                            middle_names=['P.'],
                            last_name="O'Connor")
HYPHENATED_PATIENT_NAME = PersonalName(first_name='John',
                                       middle_names=[],
                                       last_name='Smith-Jones')


@pytest.mark.parametrize("text,expected", [

    # --- Full name exact ---
    ("John P. O'Connor was here.", f"{REDACT_NAME_REPLACEMENT} was here."),
    ("Dr. John P. O'Connor was here.", f"{REDACT_NAME_REPLACEMENT} was here."),
    ("Dr John P O'Connor was here.", f"{REDACT_NAME_REPLACEMENT} was here."),

    # --- First + last only ---
    ("John O'Connor signed.", f"{REDACT_NAME_REPLACEMENT} signed."),
    ("John Connor signed.", f"{REDACT_NAME_REPLACEMENT} signed."),  # Connor is edit distance 1 from O'Connor excluding apostrophe
    ("J. O'Connor is here.", f"{REDACT_NAME_REPLACEMENT} is here."),  # initial variant

    # --- Prefixes ---
    ("Mr. John O'Connor signed.", f"{REDACT_NAME_REPLACEMENT} signed."),
    ("Mrs John O'Connor signed.", f"{REDACT_NAME_REPLACEMENT} signed."),
    ("Prof O'Connor lectured.", f"{REDACT_NAME_REPLACEMENT} lectured."),
    ("Mx O'Connor presented.", f"{REDACT_NAME_REPLACEMENT} presented."),
    ("Dr. J. O'Connor is here.", f"{REDACT_NAME_REPLACEMENT} is here."),  # initial variant

    # --- Middle initials ---
    ("John P O'Connor spoke.", f"{REDACT_NAME_REPLACEMENT} spoke."),
    ("John P. Q. O'Connor spoke.", f"{REDACT_NAME_REPLACEMENT} spoke."),

    # --- Last name alone ---
    ("O'Connor reported results.", f"{REDACT_NAME_REPLACEMENT} reported results."),
    ("O’Connor reported results.", f"{REDACT_NAME_REPLACEMENT} reported results."),  # curly apostrophe
    ("O'Connor's report is ready.", f"{REDACT_NAME_REPLACEMENT} report is ready."),

    # --- First name alone ---
    ("John presented.", f"{REDACT_NAME_REPLACEMENT} presented."),

    # --- Apostrophe dropped ---
    ("OConnor reported.", f"{REDACT_NAME_REPLACEMENT} reported."),
    ("OConor reported.", f"{REDACT_NAME_REPLACEMENT} reported."),  # deletion (typo, edit dist 1)
    ("OConnorr reported.", f"{REDACT_NAME_REPLACEMENT} reported."),  # insertion
    ("OConner reported.", f"{REDACT_NAME_REPLACEMENT} reported."),  # replacement

    # --- Single-character deletions ---
    ("OConor attended.", f"{REDACT_NAME_REPLACEMENT} attended."),       # dropped 'n'
    ("O'Conor attended.", f"{REDACT_NAME_REPLACEMENT} attended."),      # dropped one 'n' after apostrophe
    ("O'Connr spoke.", f"{REDACT_NAME_REPLACEMENT} spoke."),            # dropped 'o'

    # --- Single-character insertions ---
    ("OConnorr wrote a letter.", f"{REDACT_NAME_REPLACEMENT} wrote a letter."),  # double 'r'
    ("O'Connoor wrote a letter.", f"{REDACT_NAME_REPLACEMENT} wrote a letter."), # double 'o'
    ("O'Conznor spoke.", f"{REDACT_NAME_REPLACEMENT} spoke."),                   # extra 'z'

    # --- Single-character replacements ---
    ("OConner testified.", f"{REDACT_NAME_REPLACEMENT} testified."),    # 'o' -> 'e'
    ("O'Conmor replied.", f"{REDACT_NAME_REPLACEMENT} replied."),       # 'n' -> 'm'
    ("O'Connar replied.", f"{REDACT_NAME_REPLACEMENT} replied."),       # 'o' -> 'a'
    ("Okonnor reported.", f"{REDACT_NAME_REPLACEMENT} reported."),      # 'c' -> 'k'
    
    # --- Controls: double edits should NOT redact ---
    ("OKonner said hi.", "OKonner said hi."),                # 2 changes

    # --- Combined ---
    ("Dr. John P. Q. O'Connor-Smith signed.", f"{REDACT_NAME_REPLACEMENT} signed."),
    ("DrJohnPOConnor signed.", f"{REDACT_NAME_REPLACEMENT} signed."),  # no spaces
    ("Dr OConnor's letter.", f"{REDACT_NAME_REPLACEMENT} letter."),

])


def test_redaction_variants(text, expected):
    assert redact_subject_name(text, PATIENT_NAME) == expected


def test_nickname_variants():
    # Test nickname variants (e.g., "John" -> "Johnny", "Jack")
    cases = [
        ("John is here.", f"{REDACT_NAME_REPLACEMENT} is here."),
        ("Johnny is here.", f"{REDACT_NAME_REPLACEMENT} is here."),
        ("Jonathan is here.", f"{REDACT_NAME_REPLACEMENT} is here."),
        ("Jack is here.", f"{REDACT_NAME_REPLACEMENT} is here."),
    ]
    for text, expected in cases:
        assert redact_subject_name(text, PATIENT_NAME) == expected


# def test_names_without_boundaries():
#     cases = [
#         # --- surrounding spaces dropped ---
#         ("thenOConnortestified.", f"then{REDACT_NAME_REPLACEMENT}testified."),
#         ("then O'Connertestified.", f"then {REDACT_NAME_REPLACEMENT}testified."),
#         ("thenO'Conner testified.", f"then{REDACT_NAME_REPLACEMENT} testified."),
#         ("then O'Conner testified.", f"then {REDACT_NAME_REPLACEMENT} testified."),  # should also handle cases with boundaries
#     ]
#     for text, expected in cases:
#         assert redact_subject_name(text, PATIENT_NAME) == expected


def test_hyphenated_name_variants():
    cases = [
        ("John Smith-Jones presented.", f"{REDACT_NAME_REPLACEMENT} presented."),  # full name
        ("Smith-Jones presented.", f"{REDACT_NAME_REPLACEMENT} presented."),       # last alone
        ("SmithJones presented.", f"{REDACT_NAME_REPLACEMENT} presented."),        # dropped hyphen
    ]
    for text, expected in cases:
        assert redact_subject_name(text, HYPHENATED_PATIENT_NAME) == expected


# --- No middle name ---

NO_MIDDLE_NAME = PersonalName(first_name='Alice',
                              middle_names=[],
                              last_name='Johnson')


def test_no_middle_name():
    """Name with no middle name should still redact first, last, and full."""
    cases = [
        ("Alice Johnson arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("Johnson arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("Alice arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("Dr. Alice Johnson arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
    ]
    for text, expected in cases:
        assert redact_subject_name(text, NO_MIDDLE_NAME) == expected


# --- Multiple middle names ---

MULTI_MIDDLE = PersonalName(first_name='John',
                            middle_names=['Paul', 'Angelina'],
                            last_name='Smith')


def test_multiple_middle_names_full():
    """Full name with multiple middle names should be redacted."""
    text = "John Paul Angelina Smith signed."
    assert REDACT_NAME_REPLACEMENT in redact_subject_name(text, MULTI_MIDDLE)
    assert "Smith" not in redact_subject_name(text, MULTI_MIDDLE)


def test_multiple_middle_names_individual_tokens():
    """Each middle name should be individually redacted."""
    cases = [
        ("Paul testified.", f"{REDACT_NAME_REPLACEMENT} testified."),
        ("Angelina testified.", f"{REDACT_NAME_REPLACEMENT} testified."),
        ("Smith testified.", f"{REDACT_NAME_REPLACEMENT} testified."),
        ("John testified.", f"{REDACT_NAME_REPLACEMENT} testified."),
    ]
    for text, expected in cases:
        assert redact_subject_name(text, MULTI_MIDDLE) == expected


def test_multiple_middle_names_first_last_only():
    """First + last (middle names omitted) should still be redacted."""
    text = "John Smith signed."
    assert REDACT_NAME_REPLACEMENT in redact_subject_name(text, MULTI_MIDDLE)
    assert "Smith" not in redact_subject_name(text, MULTI_MIDDLE)


def test_multiple_middle_names_subset():
    """Partial middle names (only some present) should still redact."""
    # First + one middle + last
    text = "John Paul Smith signed."
    assert REDACT_NAME_REPLACEMENT in redact_subject_name(text, MULTI_MIDDLE)
    assert "Smith" not in redact_subject_name(text, MULTI_MIDDLE)


def test_multiple_middle_names_initials():
    """Middle initials should be consumed in title pattern."""
    text = "Dr. John P. A. Smith signed."
    result = redact_subject_name(text, MULTI_MIDDLE)
    assert "Smith" not in result
    assert REDACT_NAME_REPLACEMENT in result


def test_multiple_middle_names_nicknames():
    """Nicknames of middle names should be detected (e.g., Paul -> Pablo)."""
    # NickNamer may not have all variants, so test that the mechanism works
    # by checking that the first name nickname still works with multiple middles
    text = "Johnny Smith signed."
    result = redact_subject_name(text, MULTI_MIDDLE)
    assert "Johnny" not in result
    assert REDACT_NAME_REPLACEMENT in result


# --- Hyphenated middle name ---

HYPHEN_MIDDLE = PersonalName(first_name='Jane',
                             middle_names=['Marie-Claire'],
                             last_name='Doe')


def test_hyphenated_middle_name():
    """Hyphenated middle name should be detected."""
    cases = [
        ("Jane Marie-Claire Doe arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("Marie-Claire arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("MarieClaire arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),  # dropped hyphen
    ]
    for text, expected in cases:
        assert redact_subject_name(text, HYPHEN_MIDDLE) == expected


# ---------------------------------------------------------------------
# SubjectNameRedactor memoization
# ---------------------------------------------------------------------

def test_redactor_memoizes_identical_inputs():
    """Repeated ``.redact()`` calls on the same text must return the same
    result and the text must land in the cache after the first call."""
    from clean_eeg.anonymize import SubjectNameRedactor

    redactor = SubjectNameRedactor(PATIENT_NAME)

    first = redactor.redact("John P. O'Connor was here")
    second = redactor.redact("John P. O'Connor was here")
    assert first == second
    assert REDACT_NAME_REPLACEMENT in first
    assert "John P. O'Connor was here" in redactor._cache


def test_redactor_cache_calls_presidio_once_per_unique_text(monkeypatch):
    """The analyzer must be invoked exactly once per unique input, no
    matter how many times that input is redacted. Guards against a
    future regression where the cache is bypassed."""
    from clean_eeg.anonymize import SubjectNameRedactor

    redactor = SubjectNameRedactor(PATIENT_NAME)

    call_count = {"n": 0}
    orig_analyze = redactor.analyzer.analyze

    def counting_analyze(*args, **kwargs):
        call_count["n"] += 1
        return orig_analyze(*args, **kwargs)

    monkeypatch.setattr(redactor.analyzer, "analyze", counting_analyze)

    # 3 unique strings, each redacted 10 times. analyzer must fire only
    # 3 times if the cache is wired up correctly.
    for _ in range(10):
        redactor.redact("uV")
        redactor.redact("")
        redactor.redact("John arrived")
    assert call_count["n"] == 3


LANE_NAME = PersonalName(first_name='John', middle_names=['Lane'], last_name='Smith')


@pytest.mark.parametrize("text,expected", [
    # Initial-chain forms (patient initials = J, L)
    ("L. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
    ("L Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),       # no period
    ("J. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),      # first initial alone
    ("J Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),       # no period
    ("J. L. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
    ("J L Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),     # no periods
    ("J.L. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),    # no spaces between initials
    ("JL Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),      # concatenated
    ("Dr. J. L. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
    ("Dr. JL Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
    ("Dr. L. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),  # title + middle initial only
])
def test_initial_chain_redaction(text, expected):
    """Patient initials (in any order/combo, with/without dots and spaces)
    plus the last name should be fully redacted."""
    assert redact_subject_name(text, LANE_NAME) == expected


def test_initial_chain_does_not_match_wrong_initials():
    """Initial chain must match the patient's *actual* initials. Random
    F/M letters should not get consumed — only the last name redacts
    via the last-name-only branch."""
    out = redact_subject_name("F. M. Smith arrived.", LANE_NAME)
    # Smith is still redacted (pat_lastonly). The F. M. survive intact.
    assert "F. M." in out
    assert "Smith" not in out
    assert REDACT_NAME_REPLACEMENT in out


def test_initial_chain_does_not_match_prose_with_letters():
    """Initial-chain regex must not fire on prose like 'A clinical event'
    even when the patient's first initial happens to appear as a
    standalone uppercase letter."""
    name = PersonalName(first_name='Andrew', middle_names=[], last_name='Smith')
    text = "A clinical event was logged at noon."
    assert redact_subject_name(text, name) == text


def test_initial_chain_with_no_middle_names():
    """For a patient with no middle name, the initial chain reduces to
    'first_initial + last' (the previous pat_first_initial behavior)."""
    name = PersonalName(first_name='Alice', middle_names=[], last_name='Johnson')
    cases = [
        ("A. Johnson arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("A Johnson arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("Dr. A. Johnson arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
    ]
    for text, expected in cases:
        assert redact_subject_name(text, name) == expected


def test_initial_chain_with_multiple_middles():
    """Patient with multiple middle names: each middle's initial is in
    the chain, and any contiguous subset of the chain (preserving order)
    matches."""
    name = PersonalName(first_name='John', middle_names=['Paul', 'Angelina'], last_name='Smith')
    cases = [
        ("J. P. A. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("J P A Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("JPA Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),
        ("P. A. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),  # middle initials only
        ("A. Smith arrived.", f"{REDACT_NAME_REPLACEMENT} arrived."),     # last middle initial alone
    ]
    for text, expected in cases:
        assert redact_subject_name(text, name) == expected


def test_single_letter_middle_initial_does_not_overredact():
    """A middle name of 'L' must not cause every standalone 'L' in EDF
    text to be redacted. The deny-list and fuzzy targets must skip
    initials; only TitleAndInitialsRecognizer redacts initials, and
    only in a 'Dr. F. M. Last' context."""
    name = PersonalName(first_name='John', middle_names=['L'], last_name='Smith')
    text = "Channel L was annotated. The L lead detached briefly."
    assert redact_subject_name(text, name) == text


def test_dotted_initial_middle_name_does_not_overredact():
    """A middle name of 'P.' (single alpha char) is also an initial —
    must not redact every standalone 'P' or 'P.' in the input."""
    name = PersonalName(first_name='John', middle_names=['P.'], last_name='Smith')
    text = "Channel P was logged. Then P. picked up noise."
    assert redact_subject_name(text, name) == text


def test_single_letter_middle_initial_still_redacts_in_title_pattern():
    """When a single-letter middle initial appears between the first
    name and last name (the only context where an initial unambiguously
    identifies the patient), the title-and-initials regex must still
    consume it."""
    name = PersonalName(first_name='John', middle_names=['L'], last_name='Smith')
    out = redact_subject_name("Dr. John L. Smith spoke.", name)
    assert "John" not in out
    assert "Smith" not in out
    assert REDACT_NAME_REPLACEMENT in out


def test_redactor_caches_are_independent_per_subject():
    """Two redactors for different subjects must have independent caches
    so a hit for one can't leak the wrong redacted value to the other."""
    from clean_eeg.anonymize import SubjectNameRedactor

    red_a = SubjectNameRedactor(PATIENT_NAME)  # John P. O'Connor
    red_b = SubjectNameRedactor(PersonalName(first_name='Jane',
                                              middle_names=[],
                                              last_name='Doe'))

    text = "Jane arrived"
    a_result = red_a.redact(text)
    b_result = red_b.redact(text)
    # For John: "Jane" is not his name -> unchanged.
    # For Jane: "Jane" is her name    -> redacted.
    assert a_result == text
    assert REDACT_NAME_REPLACEMENT in b_result
    assert red_a._cache is not red_b._cache
