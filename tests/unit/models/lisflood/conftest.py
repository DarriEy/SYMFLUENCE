# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared fixtures for LISFLOOD model tests."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def lisflood_config(tmp_path):
    """Create a LISFLOOD-specific SymfluenceConfig."""
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 01:00",
        "EXPERIMENT_TIME_END": "2005-03-01 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "LISFLOOD",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 86400,
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def lisflood_config_dict(tmp_path):
    """Create a plain dict LISFLOOD configuration (for calibration tests)."""
    return {
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "EXPERIMENT_TIME_START": "2005-01-01 01:00",
        "EXPERIMENT_TIME_END": "2005-03-01 23:00",
        "FORCING_TIME_STEP_SIZE": 86400,
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "HYDROLOGICAL_MODEL": "LISFLOOD",
    }


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
def real_logger():
    """Create a real logger for tests that need one."""
    return logging.getLogger("test_lisflood")


@pytest.fixture
def setup_lisflood_directories(tmp_path, lisflood_config):
    """Set up directory structure for LISFLOOD testing."""
    data_dir = lisflood_config.system.data_dir
    domain_dir = data_dir / f"domain_{lisflood_config.domain.name}"

    settings_dir = domain_dir / "settings" / "LISFLOOD"
    simulations_dir = domain_dir / "simulations" / "test_run" / "LISFLOOD"

    for d in [settings_dir, simulations_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": data_dir,
        "domain_dir": domain_dir,
        "settings_dir": settings_dir,
        "simulations_dir": simulations_dir,
    }


@pytest.fixture
def sample_forcing_dir(tmp_path):
    """Create minimal synthetic forcing NetCDF files for preprocessing tests."""
    forcing_dir = tmp_path / "domain_test_domain" / "forcing" / "basin_averaged_data"
    forcing_dir.mkdir(parents=True, exist_ok=True)

    times = xr.cftime_range("2005-01-01", "2005-03-01", freq="D")
    nt = len(times)

    precip = np.random.uniform(0, 10, nt)
    temp = np.random.uniform(-5, 20, nt)

    ds = xr.Dataset(
        {
            "pptrate": xr.DataArray(precip, dims=["time"], coords={"time": times}),
            "airtemp": xr.DataArray(temp, dims=["time"], coords={"time": times}),
        }
    )
    ds.to_netcdf(forcing_dir / "forcing.nc")
    return forcing_dir
