import pytest

from clean_eeg.anonymize import redact_subject_name, PersonalName


PATIENT_NAME = PersonalName(first_name='John',
                            middle_names=['P.'],
                            last_name="O'Connor")
HYPHENATED_PATIENT_NAME = PersonalName(first_name='John',
                                       middle_names=[],
                                       last_name='Smith-Jones')


@pytest.mark.parametrize("text,expected", [

    # --- Full name exact ---
    ("John P. O'Connor was here.", "[REDACTED-NAME] was here."),
    ("Dr. John P. O'Connor was here.", "[REDACTED-NAME] was here."),
    ("Dr John P O'Connor was here.", "[REDACTED-NAME] was here."),

    # --- First + last only ---
    ("John O'Connor signed.", "[REDACTED-NAME] signed."),
    ("John Connor signed.", "[REDACTED-NAME] signed."),  # Connor is edit distance 1 from O'Connor excluding apostrophe
    ("J. O'Connor is here.", "[REDACTED-NAME] is here."),  # initial variant

    # --- Prefixes ---
    ("Mr. John O'Connor signed.", "[REDACTED-NAME] signed."),
    ("Mrs John O'Connor signed.", "[REDACTED-NAME] signed."),
    ("Prof O'Connor lectured.", "[REDACTED-NAME] lectured."),
    ("Mx O'Connor presented.", "[REDACTED-NAME] presented."),
    ("Dr. J. O'Connor is here.", "[REDACTED-NAME] is here."),  # initial variant

    # --- Middle initials ---
    ("John P O'Connor spoke.", "[REDACTED-NAME] spoke."),
    ("John P. Q. O'Connor spoke.", "[REDACTED-NAME] spoke."),

    # --- Last name alone ---
    ("O'Connor reported results.", "[REDACTED-NAME] reported results."),
    ("O’Connor reported results.", "[REDACTED-NAME] reported results."),  # curly apostrophe
    ("O'Connor's report is ready.", "[REDACTED-NAME] report is ready."),

    # --- First name alone ---
    ("John presented.", "[REDACTED-NAME] presented."),

    # --- Apostrophe dropped ---
    ("OConnor reported.", "[REDACTED-NAME] reported."),
    ("OConor reported.", "[REDACTED-NAME] reported."),  # deletion (typo, edit dist 1)
    ("OConnorr reported.", "[REDACTED-NAME] reported."),  # insertion
    ("OConner reported.", "[REDACTED-NAME] reported."),  # replacement

    # --- Single-character deletions ---
    ("OConor attended.", "[REDACTED-NAME] attended."),       # dropped 'n'
    ("O'Conor attended.", "[REDACTED-NAME] attended."),      # dropped one 'n' after apostrophe
    ("O'Connr spoke.", "[REDACTED-NAME] spoke."),            # dropped 'o'

    # --- Single-character insertions ---
    ("OConnorr wrote a letter.", "[REDACTED-NAME] wrote a letter."),  # double 'r'
    ("O'Connoor wrote a letter.", "[REDACTED-NAME] wrote a letter."), # double 'o'
    ("O'Conznor spoke.", "[REDACTED-NAME] spoke."),                   # extra 'z'

    # --- Single-character replacements ---
    ("OConner testified.", "[REDACTED-NAME] testified."),    # 'o' -> 'e'
    ("O'Conmor replied.", "[REDACTED-NAME] replied."),       # 'n' -> 'm'
    ("O'Connar replied.", "[REDACTED-NAME] replied."),       # 'o' -> 'a'
    ("Okonnor reported.", "[REDACTED-NAME] reported."),      # 'c' -> 'k'
    
    # --- Controls: double edits should NOT redact ---
    ("OKonner said hi.", "OKonner said hi."),                # 2 changes

    # --- Combined ---
    ("Dr. John P. Q. O'Connor-Smith signed.", "[REDACTED-NAME] signed."),
    ("DrJohnPOConnor signed.", "[REDACTED-NAME] signed."),  # no spaces
    ("Dr OConnor's letter.", "[REDACTED-NAME] letter."),

])


def test_redaction_variants(text, expected):
    assert redact_subject_name(text, PATIENT_NAME) == expected


def test_nickname_variants():
    # Test nickname variants (e.g., "John" -> "Johnny", "Jack")
    cases = [
        ("John is here.", "[REDACTED-NAME] is here."),
        ("Johnny is here.", "[REDACTED-NAME] is here."),
        ("Jonathan is here.", "[REDACTED-NAME] is here."),
        ("Jack is here.", "[REDACTED-NAME] is here."),
    ]
    for text, expected in cases:
        assert redact_subject_name(text, PATIENT_NAME) == expected


# def test_names_without_boundaries():
#     cases = [
#         # --- surrounding spaces dropped ---
#         ("thenOConnortestified.", "then[REDACTED-NAME]testified."),
#         ("then O'Connertestified.", "then [REDACTED-NAME]testified."),
#         ("thenO'Conner testified.", "then[REDACTED-NAME] testified."),
#         ("then O'Conner testified.", "then [REDACTED-NAME] testified."),  # should also handle cases with boundaries
#     ]
#     for text, expected in cases:
#         assert redact_subject_name(text, PATIENT_NAME) == expected


def test_hyphenated_name_variants():
    cases = [
        ("John Smith-Jones presented.", "[REDACTED-NAME] presented."),  # full name
        ("Smith-Jones presented.", "[REDACTED-NAME] presented."),       # last alone
        ("SmithJones presented.", "[REDACTED-NAME] presented."),        # dropped hyphen
    ]
    for text, expected in cases:
        assert redact_subject_name(text, HYPHENATED_PATIENT_NAME) == expected
