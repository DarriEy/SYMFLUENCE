# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for LISFLOOD calibration components."""

import logging
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import xarray as xr

from symfluence.models.lisflood.calibration.parameter_manager import LisfloodParameterManager


@pytest.fixture
def pm_logger():
    """Create a logger for parameter manager tests."""
    return logging.getLogger("test_lisflood_calibration")


@pytest.fixture
def pm_config(tmp_path):
    """Create a parameter manager config dict."""
    return {
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "LISFLOOD_PARAMS_TO_CALIBRATE": "ksat1,ksat2,ksat3,UpperZoneTimeConstant,GwPercValue",
    }


@pytest.fixture
def pm(pm_config, pm_logger, tmp_path):
    """Create a LisfloodParameterManager instance."""
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return LisfloodParameterManager(pm_config, pm_logger, settings_dir)


class TestParameterManagerMaps:
    """Tests for PARAM_FILE_MAP and XML_PARAMS dictionaries."""

    def test_param_file_map_is_dict(self):
        assert isinstance(LisfloodParameterManager.PARAM_FILE_MAP, dict)

    def test_param_file_map_has_soil_params(self):
        pfm = LisfloodParameterManager.PARAM_FILE_MAP
        for key in [
            "ksat1",
            "ksat2",
            "ksat3",
            "lambda1",
            "lambda2",
            "lambda3",
            "thetas1",
            "thetas2",
            "thetas3",
            "thetar1",
            "thetar2",
            "thetar3",
        ]:
            assert key in pfm, f"Missing PARAM_FILE_MAP key: {key}"

    def test_param_file_map_has_snow_params(self):
        pfm = LisfloodParameterManager.PARAM_FILE_MAP
        for key in ["snow_melt_coef", "temp_melt", "temp_snow"]:
            assert key in pfm

    def test_param_file_map_has_channel_params(self):
        pfm = LisfloodParameterManager.PARAM_FILE_MAP
        assert "chanman" in pfm
        assert "mannings" in pfm

    def test_xml_params_is_set(self):
        assert isinstance(LisfloodParameterManager.XML_PARAMS, set)

    def test_xml_params_has_expected_entries(self):
        xp = LisfloodParameterManager.XML_PARAMS
        expected = {
            "UpperZoneTimeConstant",
            "LowerZoneTimeConstant",
            "GwPercValue",
            "GwLoss",
            "LZThreshold",
            "b_Xinanjiang",
            "PowerPrefFlow",
            "CalChanMan",
            "SnowMeltCoef",
            "CalEvaporation",
            "SnowSeasonAdj",
        }
        assert expected.issubset(xp), f"Missing XML_PARAMS: {expected - xp}"


class TestDefaultBounds:
    """Tests for DEFAULT_BOUNDS coverage."""

    def test_default_bounds_covers_all_map_params(self):
        bounds = LisfloodParameterManager.DEFAULT_BOUNDS
        for param in LisfloodParameterManager.PARAM_FILE_MAP:
            assert param in bounds, f"DEFAULT_BOUNDS missing map param: {param}"

    def test_default_bounds_covers_all_xml_params(self):
        bounds = LisfloodParameterManager.DEFAULT_BOUNDS
        for param in LisfloodParameterManager.XML_PARAMS:
            assert param in bounds, f"DEFAULT_BOUNDS missing XML param: {param}"

    def test_bounds_min_less_than_max(self):
        for param, b in LisfloodParameterManager.DEFAULT_BOUNDS.items():
            assert b["min"] < b["max"], f"Bounds inverted for {param}: min={b['min']}, max={b['max']}"


class TestPreprocessorDefaults:
    """Tests for PREPROCESSOR_DEFAULTS coverage."""

    def test_preprocessor_defaults_covers_all_map_params(self):
        defaults = LisfloodParameterManager.PREPROCESSOR_DEFAULTS
        for param in LisfloodParameterManager.PARAM_FILE_MAP:
            assert param in defaults, f"PREPROCESSOR_DEFAULTS missing map param: {param}"

    def test_preprocessor_defaults_covers_all_xml_params(self):
        defaults = LisfloodParameterManager.PREPROCESSOR_DEFAULTS
        for param in LisfloodParameterManager.XML_PARAMS:
            assert param in defaults, f"PREPROCESSOR_DEFAULTS missing XML param: {param}"

    def test_preprocessor_defaults_within_bounds(self):
        defaults = LisfloodParameterManager.PREPROCESSOR_DEFAULTS
        bounds = LisfloodParameterManager.DEFAULT_BOUNDS
        for param, val in defaults.items():
            if param in bounds:
                assert bounds[param]["min"] <= val <= bounds[param]["max"], (
                    f"Default for {param}={val} outside bounds [{bounds[param]['min']}, {bounds[param]['max']}]"
                )


class TestValidateParameters:
    """Tests for validate_parameters."""

    def test_validate_clips_to_bounds(self, pm):
        params = {
            "ksat1": 999.0,  # Way above max of 100
            "ksat2": -10.0,  # Below min of 2
        }
        pm.validate_parameters(params)
        assert params["ksat1"] == pytest.approx(100.0)
        assert params["ksat2"] == pytest.approx(2.0)

    def test_validate_within_bounds_unchanged(self, pm):
        params = {
            "ksat1": 50.0,
            "ksat2": 30.0,
        }
        pm.validate_parameters(params)
        assert params["ksat1"] == pytest.approx(50.0)
        assert params["ksat2"] == pytest.approx(30.0)

    def test_validate_returns_true(self, pm):
        params = {"ksat1": 25.0}
        assert pm.validate_parameters(params) is True

    def test_validate_fixes_thetar_ge_thetas(self, pm_logger, tmp_path):
        """If thetar >= thetas, validate should fix it."""
        config = {
            "DOMAIN_NAME": "test",
            "EXPERIMENT_ID": "test",
            "SYMFLUENCE_DATA_DIR": str(tmp_path),
            "LISFLOOD_PARAMS_TO_CALIBRATE": "thetar1,thetas1",
        }
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir(parents=True, exist_ok=True)
        pm_local = LisfloodParameterManager(config, pm_logger, settings_dir)

        params = {"thetar1": 0.50, "thetas1": 0.45}
        pm_local.validate_parameters(params)
        # thetar should be adjusted below thetas
        assert params["thetar1"] < params["thetas1"]


class TestUpdateModelFiles:
    """Tests for update_model_files."""

    def test_update_netcdf_map_files(self, pm_logger, tmp_path):
        """Test that update_model_files modifies NetCDF map file values."""
        settings_dir = tmp_path / "settings"
        maps_dir = settings_dir / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        # Create a sample ksat1.nc file
        arr = np.full((3, 3), 25.0, dtype=np.float64)
        ds = xr.Dataset(
            {
                "ksat1": xr.DataArray(arr, dims=["y", "x"]),
            }
        )
        ds.to_netcdf(maps_dir / "ksat1.nc")

        config = {
            "DOMAIN_NAME": "test",
            "EXPERIMENT_ID": "test",
            "SYMFLUENCE_DATA_DIR": str(tmp_path),
            "LISFLOOD_PARAMS_TO_CALIBRATE": "ksat1",
        }
        pm_local = LisfloodParameterManager(config, pm_logger, settings_dir)

        result = pm_local.update_model_files({"ksat1": 50.0}, settings_dir)
        assert result is True

        # Verify the file was updated
        ds_updated = xr.open_dataset(maps_dir / "ksat1.nc")
        assert np.allclose(ds_updated["ksat1"].values, 50.0)
        ds_updated.close()

    def test_update_xml_scalar_parameters(self, pm_logger, tmp_path):
        """Test that update_model_files modifies XML scalar parameters."""
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal settings XML
        root = ET.Element("lfsettings")
        lfuser = ET.SubElement(root, "lfuser")
        textvar = ET.SubElement(lfuser, "textvar")
        textvar.set("name", "UpperZoneTimeConstant")
        textvar.set("value", "10")
        tree = ET.ElementTree(root)
        tree.write(settings_dir / "settings.xml", encoding="unicode", xml_declaration=True)

        config = {
            "DOMAIN_NAME": "test",
            "EXPERIMENT_ID": "test",
            "SYMFLUENCE_DATA_DIR": str(tmp_path),
            "LISFLOOD_PARAMS_TO_CALIBRATE": "UpperZoneTimeConstant",
            "LISFLOOD_SETTINGS_FILE": "settings.xml",
        }
        pm_local = LisfloodParameterManager(config, pm_logger, settings_dir)

        result = pm_local.update_model_files({"UpperZoneTimeConstant": 42.0}, settings_dir)
        assert result is True

        # Verify the XML was updated
        tree = ET.parse(settings_dir / "settings.xml")
        root = tree.getroot()
        for tv in root.iter("textvar"):
            if tv.get("name") == "UpperZoneTimeConstant":
                assert tv.get("value") == "42.0"


class TestParameterManagerInit:
    """Tests for parameter manager initialization."""

    def test_instantiation(self, pm):
        assert pm is not None

    def test_parameter_names(self, pm):
        names = pm._get_parameter_names()
        assert "ksat1" in names
        assert "ksat2" in names
        assert "ksat3" in names
        assert "UpperZoneTimeConstant" in names
        assert "GwPercValue" in names

    def test_load_bounds(self, pm):
        bounds = pm._load_parameter_bounds()
        assert "ksat1" in bounds
        assert bounds["ksat1"]["min"] == pytest.approx(5.0)
        assert bounds["ksat1"]["max"] == pytest.approx(100.0)

    def test_get_initial_parameters(self, pm):
        initial = pm.get_initial_parameters()
        assert isinstance(initial, dict)
        assert "ksat1" in initial
        # Default for ksat1 is 25.0, which is within bounds
        assert initial["ksat1"] == pytest.approx(25.0)


class TestOptimizerRegistration:
    """Tests for optimizer and worker registration."""

    def test_optimizer_is_registered(self):
        import symfluence.models.lisflood  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry

        cls = OptimizerRegistry.get_optimizer("LISFLOOD")
        assert cls is not None

    def test_worker_is_registered(self):
        import symfluence.models.lisflood  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry

        cls = OptimizerRegistry.get_worker("LISFLOOD")
        assert cls is not None
