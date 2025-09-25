import pathlib
PACKAGE_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = PACKAGE_DIR / 'data'
# automatically created by scripts/build_whitelist.py
AUTO_WORD_WHITELIST_PATH = DATA_DIR /'auto_word_whitelist.json'
# words added manually
MANUAL_WORD_WHITELIST_PATH = DATA_DIR / 'manual_word_whitelist.json'

TEST_DIR = PACKAGE_DIR / 'tests'
TEST_DATA_DIR = TEST_DIR / "test_data"
TEST_CONFIG_FILE = TEST_DIR / "config.json"
TEST_SUBJECT_DATA_DIR = TEST_DATA_DIR / "test_subject_data"
