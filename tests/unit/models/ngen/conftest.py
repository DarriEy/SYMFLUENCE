"""Shared fixtures for NGEN model tests."""

import json
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
def ngen_config(temp_dir):
    """Create an NGEN-specific configuration."""
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(temp_dir / "data"),
        "SYMFLUENCE_CODE_DIR": str(temp_dir / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "ngen_test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-12-31 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "NGEN",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 3600,
        "NGEN_MODULES_TO_CALIBRATE": "CFE",
        "NGEN_CFE_PARAMS_TO_CALIBRATE": "bb,satdk",
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
def setup_ngen_directories(temp_dir, ngen_config):
    """Set up directory structure for NGEN testing."""
    data_dir = ngen_config.system.data_dir
    domain_dir = data_dir / f"domain_{ngen_config.domain.name}"

    settings_dir = domain_dir / "settings" / "NGEN"
    simulations_dir = domain_dir / "simulations" / "ngen_test" / "NGEN"

    for d in [settings_dir, simulations_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create fake NGEN executable
    ngen_exe = data_dir / "installs" / "ngen" / "cmake_build" / "ngen"
    ngen_exe.parent.mkdir(parents=True, exist_ok=True)
    ngen_exe.write_text("#!/bin/sh\necho ngen")
    ngen_exe.chmod(0o755)

    # Create placeholder GeoJSON and realization files
    catchment_geojson = settings_dir / "test_domain_catchment.geojson"
    catchment_geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"id": "cat-1"}, "geometry": {"type": "Point", "coordinates": [0, 0]}}]
    }))

    nexus_geojson = settings_dir / "test_domain_nexus.geojson"
    nexus_geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"id": "nex-1"}, "geometry": {"type": "Point", "coordinates": [0, 0]}}]
    }))

    realization_file = settings_dir / "realization_config.json"
    realization_file.write_text(json.dumps({
        "global": {"formulations": [{"name": "bmi_c", "params": {"model_type_name": "CFE"}}]}
    }))

    return {
        "data_dir": data_dir,
        "domain_dir": domain_dir,
        "settings_dir": settings_dir,
        "simulations_dir": simulations_dir,
        "ngen_exe": ngen_exe,
        "catchment_geojson": catchment_geojson,
        "nexus_geojson": nexus_geojson,
        "realization_file": realization_file,
    }
