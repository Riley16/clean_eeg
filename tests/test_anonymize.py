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
