"""
Root conftest.py - Session-scoped fixtures shared across all tests.

This file sets up the Python path and provides core fixtures for test discovery.
Additional fixtures are loaded via pytest_plugins from tests/fixtures/.
"""

from pathlib import Path
import sys

import pytest

# Add directories to path BEFORE importing local modules
SYMFLUENCE_CODE_DIR = Path(__file__).parent.parent.resolve()
if str(SYMFLUENCE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SYMFLUENCE_CODE_DIR))
TESTS_DIR = Path(__file__).parent.resolve()
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Import test utilities (backward compatibility)
from .test_helpers import load_config_template, write_config

# Also import from new location
from .utils import helpers, geospatial

# Load additional fixtures from fixture modules
pytest_plugins = [
    "fixtures.data_fixtures",
    "fixtures.domain_fixtures",
    "fixtures.model_fixtures",
]


@pytest.fixture(scope="session")
def symfluence_code_dir():
    """Path to SYMFLUENCE code directory."""
    return SYMFLUENCE_CODE_DIR


@pytest.fixture(scope="session")
def symfluence_data_root(symfluence_code_dir):
    """Path to SYMFLUENCE_data directory (shared test data)."""
    data_root = symfluence_code_dir.parent / "SYMFLUENCE_data"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


@pytest.fixture(scope="session")
def tests_dir():
    """Path to tests directory."""
    return TESTS_DIR


@pytest.fixture()
def config_template(symfluence_code_dir):
    """Load configuration template for tests."""
    return load_config_template(symfluence_code_dir)
