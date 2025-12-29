from pathlib import Path
import sys

import pytest

from test_helpers import load_config_template, write_config

SYMFLUENCE_CODE_DIR = Path(__file__).parent.parent.resolve()
if str(SYMFLUENCE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SYMFLUENCE_CODE_DIR))
TESTS_DIR = Path(__file__).parent.resolve()
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


@pytest.fixture(scope="session")
def symfluence_code_dir():
    return SYMFLUENCE_CODE_DIR


@pytest.fixture(scope="session")
def symfluence_data_root(symfluence_code_dir):
    data_root = symfluence_code_dir.parent / "SYMFLUENCE_data"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


@pytest.fixture()
def config_template(symfluence_code_dir):
    return load_config_template(symfluence_code_dir)
