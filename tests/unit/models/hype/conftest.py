"""Shared fixtures for HYPE model tests."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from symfluence.core.config.models import SymfluenceConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def hype_config(temp_dir):
    """Create a HYPE-specific configuration."""
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(temp_dir / "data"),
        "SYMFLUENCE_CODE_DIR": str(temp_dir / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "hype_test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-12-31 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "HYPE",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 3600,
        "SETTINGS_HYPE_PATH": str(temp_dir / "data" / "domain_test_domain" / "settings" / "HYPE"),
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def setup_hype_directories(temp_dir, hype_config):
    """Set up directory structure for HYPE testing."""
    data_dir = hype_config.system.data_dir
    domain_dir = data_dir / f"domain_{hype_config.domain.name}"

    settings_dir = domain_dir / "settings" / "HYPE"
    simulations_dir = domain_dir / "simulations" / "HYPE"

    for d in [settings_dir, simulations_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create a fake HYPE executable
    hype_exe = data_dir / "installs" / "hype" / "bin" / "hype"
    hype_exe.parent.mkdir(parents=True, exist_ok=True)
    hype_exe.write_text("#!/bin/sh\necho hype")
    hype_exe.chmod(0o755)

    return {
        "data_dir": data_dir,
        "domain_dir": domain_dir,
        "settings_dir": settings_dir,
        "simulations_dir": simulations_dir,
        "hype_exe": hype_exe,
    }
