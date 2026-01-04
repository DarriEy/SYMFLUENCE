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
    from symfluence.core.config.models import SymfluenceConfig

    # Create config using SymfluenceConfig.from_minimal and then update
    config = SymfluenceConfig.from_minimal(
        domain_name="test_domain",
        model="SUMMA",
        experiment_id=f"test_summa_{tmp_path.name}",
        DOMAIN_DEFINITION_METHOD="lumped" # Explicitly set to lumped
    ).model_dump(by_alias=True)

    # Convert back to SymfluenceConfig, ensuring paths are handled correctly
    config_obj = SymfluenceConfig(**config)
    
    cfg_path = tmp_path / "config_summa.yaml"
    # Use write_config to serialize it for potential later loading
    from ..utils.helpers import write_config
    write_config(config_obj, cfg_path)

    return config_obj


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
    from symfluence.core.config.models import SymfluenceConfig

    config = SymfluenceConfig.from_minimal(
        domain_name="test_domain",
        model="FUSE",
        experiment_id=f"test_fuse_{tmp_path.name}",
        DOMAIN_DEFINITION_METHOD="lumped"
    ).model_dump(by_alias=True)

    config_obj = SymfluenceConfig(**config)

    cfg_path = tmp_path / "config_fuse.yaml"
    from ..utils.helpers import write_config
    write_config(config_obj, cfg_path)

    return config_obj


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
    from symfluence.core.config.models import SymfluenceConfig

    config = SymfluenceConfig.from_minimal(
        domain_name="test_domain",
        model="NGEN",
        experiment_id=f"test_ngen_{tmp_path.name}",
        DOMAIN_DEFINITION_METHOD="lumped"
    ).model_dump(by_alias=True)

    config_obj = SymfluenceConfig(**config)

    cfg_path = tmp_path / "config_ngen.yaml"
    from ..utils.helpers import write_config
    write_config(config_obj, cfg_path)

    return config_obj
