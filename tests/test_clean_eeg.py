from clean_eeg.clean_eeg import remove_gendered_pronouns, _GENDERED_PRONOUNS


def test_remove_gendered_pronouns_basic():
    input = ' asdf '.join(_GENDERED_PRONOUNS)
    output = ' asdf '.join(['REDACTED_PRONOUN'] * len(_GENDERED_PRONOUNS))
    assert remove_gendered_pronouns(input) == output

