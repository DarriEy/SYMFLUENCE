"""
Domain Fixtures for SYMFLUENCE Tests

Provides fixtures for domain creation and management.
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="function")
def minimal_config_3hr(tmp_path, symfluence_code_dir):
    """
    Minimal 3-hour test configuration.

    Creates a configuration for quick smoke tests with:
    - 3-6 hour simulation period
    - Minimal domain (single HRU)
    - Basic settings

    Args:
        tmp_path: Pytest tmp_path fixture
        symfluence_code_dir: Path to SYMFLUENCE code directory

    Returns:
        tuple: (config_path, config_dict)
    """
    from ..utils.helpers import load_config_template, write_config

    config = load_config_template(symfluence_code_dir)

    # Minimal 3-hour settings
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-01 03:00"
    config["EXPERIMENT_ID"] = f"test_3hr_{tmp_path.name}"

    # Output to tmp directory
    cfg_path = tmp_path / "config_minimal_3hr.yaml"
    write_config(config, cfg_path)

    return cfg_path, config


@pytest.fixture(scope="function")
def standard_config_1month(tmp_path, symfluence_code_dir):
    """
    Standard 1-month test configuration.

    Creates a configuration for full tests with:
    - 1-month simulation period
    - Standard domain settings
    - Full preprocessing

    Args:
        tmp_path: Pytest tmp_path fixture
        symfluence_code_dir: Path to SYMFLUENCE code directory

    Returns:
        tuple: (config_path, config_dict)
    """
    from ..utils.helpers import load_config_template, write_config

    config = load_config_template(symfluence_code_dir)

    # Standard 1-month settings
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-31 23:00"
    config["CALIBRATION_PERIOD"] = "2010-01-05, 2010-01-19"
    config["EVALUATION_PERIOD"] = "2010-01-20, 2010-01-30"
    config["SPINUP_PERIOD"] = "2010-01-01, 2010-01-04"
    config["EXPERIMENT_ID"] = f"test_1month_{tmp_path.name}"

    # Output to tmp directory
    cfg_path = tmp_path / "config_standard_1month.yaml"
    write_config(config, cfg_path)

    return cfg_path, config


@pytest.fixture(scope="function")
def calibration_config(tmp_path, symfluence_code_dir):
    """
    Calibration test configuration.

    Creates a configuration for calibration tests with:
    - 1-month simulation period
    - Calibration settings
    - Minimal iterations for testing

    Args:
        tmp_path: Pytest tmp_path fixture
        symfluence_code_dir: Path to SYMFLUENCE code directory

    Returns:
        tuple: (config_path, config_dict)
    """
    from ..utils.helpers import load_config_template, write_config

    config = load_config_template(symfluence_code_dir)

    # Calibration settings
    config["EXPERIMENT_TIME_START"] = "2010-01-01 00:00"
    config["EXPERIMENT_TIME_END"] = "2010-01-31 23:00"
    config["CALIBRATION_PERIOD"] = "2010-01-05, 2010-01-19"
    config["EVALUATION_PERIOD"] = "2010-01-20, 2010-01-30"
    config["SPINUP_PERIOD"] = "2010-01-01, 2010-01-04"
    config["EXPERIMENT_ID"] = f"test_calib_{tmp_path.name}"

    # Minimal calibration for testing
    config["NUMBER_OF_ITERATIONS"] = 3
    config["RANDOM_SEED"] = 42

    # Output to tmp directory
    cfg_path = tmp_path / "config_calibration.yaml"
    write_config(config, cfg_path)

    return cfg_path, config
