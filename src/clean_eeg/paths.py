import pathlib
PACKAGE_DIR = pathlib.Path(__file__).parent.parent.parent
TEST_DIR = PACKAGE_DIR / 'tests'
TEST_DATA_DIR = TEST_DIR / "test_data"
TEST_CONFIG_FILE = TEST_DIR / "config.json"
