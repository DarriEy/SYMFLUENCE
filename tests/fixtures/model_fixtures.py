"""
Model Fixtures for SYMFLUENCE Tests

Provides fixtures for model execution and validation.
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def available_models():
    """
    List of available hydrological models.

    Returns:
        list: List of model names available for testing
    """
    return ["SUMMA", "FUSE", "NGEN", "GR"]


@pytest.fixture(scope="function")
def summa_config(tmp_path, symfluence_code_dir):
    """
    SUMMA model configuration.

    Args:
        tmp_path: Pytest tmp_path fixture
        symfluence_code_dir: Path to SYMFLUENCE code directory

    Returns:
        tuple: (config_path, config_dict)
    """
    from ..utils.helpers import load_config_template, write_config

    config = load_config_template(symfluence_code_dir)
    config["HYDROLOGICAL_MODEL"] = "SUMMA"
    config["ROUTING_MODEL"] = "mizuRoute"
    config["EXPERIMENT_ID"] = f"test_summa_{tmp_path.name}"

    cfg_path = tmp_path / "config_summa.yaml"
    write_config(config, cfg_path)

    return cfg_path, config


@pytest.fixture(scope="function")
def fuse_config(tmp_path, symfluence_code_dir):
    """
    FUSE model configuration.

    Args:
        tmp_path: Pytest tmp_path fixture
        symfluence_code_dir: Path to SYMFLUENCE code directory

    Returns:
        tuple: (config_path, config_dict)
    """
    from ..utils.helpers import load_config_template, write_config

    config = load_config_template(symfluence_code_dir)
    config["HYDROLOGICAL_MODEL"] = "FUSE"
    config["EXPERIMENT_ID"] = f"test_fuse_{tmp_path.name}"

    cfg_path = tmp_path / "config_fuse.yaml"
    write_config(config, cfg_path)

    return cfg_path, config


@pytest.fixture(scope="function")
def ngen_config(tmp_path, symfluence_code_dir):
    """
    NGEN model configuration.

    Args:
        tmp_path: Pytest tmp_path fixture
        symfluence_code_dir: Path to SYMFLUENCE code directory

    Returns:
        tuple: (config_path, config_dict)
    """
    from ..utils.helpers import load_config_template, write_config

    config = load_config_template(symfluence_code_dir)
    config["HYDROLOGICAL_MODEL"] = "NGEN"
    config["EXPERIMENT_ID"] = f"test_ngen_{tmp_path.name}"

    cfg_path = tmp_path / "config_ngen.yaml"
    write_config(config, cfg_path)

    return cfg_path, config
