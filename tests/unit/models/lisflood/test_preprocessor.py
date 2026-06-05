# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for LISFLOOD model preprocessor."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import xarray as xr

from symfluence.models.lisflood.preprocessor import LisfloodPreProcessor


class TestPreprocessorImport:
    """Basic import and instantiation tests."""

    def test_preprocessor_can_be_imported(self):
        assert LisfloodPreProcessor is not None

    def test_model_name(self):
        assert LisfloodPreProcessor.MODEL_NAME == "LISFLOOD"


class TestPreprocessorDirectoryStructure:
    """Tests for preprocessor directory creation."""

    def test_create_directory_structure(self, lisflood_config, mock_logger, setup_lisflood_directories):
        pp = LisfloodPreProcessor(lisflood_config, mock_logger)
        pp._create_directory_structure()
        assert pp.settings_dir.exists()
        assert pp.maps_dir.exists()
        assert pp.forcing_out_dir.exists()
        assert pp.tables_dir.exists()
        assert pp.inits_dir.exists()


class TestPreprocessorStaticMaps:
    """Tests for static map generation."""

    @pytest.fixture
    def preprocessor_with_dirs(self, lisflood_config, mock_logger, setup_lisflood_directories):
        pp = LisfloodPreProcessor(lisflood_config, mock_logger)
        pp._create_directory_structure()
        return pp

    def _mock_catchment_props(self):
        return {"lat": 51.17, "lon": -115.57, "area_m2": 2.21e9, "elev": 1500.0}

    def test_generates_static_maps(self, preprocessor_with_dirs):
        pp = preprocessor_with_dirs
        with patch.object(pp, "_get_catchment_properties", return_value=self._mock_catchment_props()):
            pp._generate_static_maps()
        # Check that maps directory has NetCDF files
        nc_files = list(pp.maps_dir.glob("*.nc"))
        assert len(nc_files) > 50, f"Expected >50 static map files, got {len(nc_files)}"

    def test_static_maps_have_3x3_grid(self, preprocessor_with_dirs):
        pp = preprocessor_with_dirs
        with patch.object(pp, "_get_catchment_properties", return_value=self._mock_catchment_props()):
            pp._generate_static_maps()
        ds = xr.open_dataset(pp.maps_dir / "dem.nc")
        assert ds["dem"].shape == (3, 3)
        ds.close()

    def test_static_maps_have_y_descending(self, preprocessor_with_dirs):
        pp = preprocessor_with_dirs
        with patch.object(pp, "_get_catchment_properties", return_value=self._mock_catchment_props()):
            pp._generate_static_maps()
        ds = xr.open_dataset(pp.maps_dir / "dem.nc")
        y_vals = ds["y"].values
        assert y_vals[0] > y_vals[-1], "y coordinates should be descending"
        ds.close()

    def test_ldd_map_has_proper_drainage_directions(self, preprocessor_with_dirs):
        pp = preprocessor_with_dirs
        with patch.object(pp, "_get_catchment_properties", return_value=self._mock_catchment_props()):
            pp._generate_static_maps()
        ds = xr.open_dataset(pp.maps_dir / "ldd.nc")
        ldd = ds["ldd"].values
        # Centre cell (y descending: row 1, col 1) should be pit = 5
        assert ldd[1, 1] == 5.0
        # All surrounding cells should drain toward centre (values 1-9, not 5)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                assert ldd[i, j] != 5.0, f"Cell ({i},{j}) should not be a pit"
                assert ldd[i, j] in [1, 2, 3, 4, 6, 7, 8, 9]
        ds.close()

    def test_outlet_map_has_one_at_centre(self, preprocessor_with_dirs):
        pp = preprocessor_with_dirs
        with patch.object(pp, "_get_catchment_properties", return_value=self._mock_catchment_props()):
            pp._generate_static_maps()
        ds = xr.open_dataset(pp.maps_dir / "outlet.nc")
        outlet = ds["outlet"].values
        assert outlet[1, 1] == 1.0
        # All other cells should be 0
        outlet_copy = outlet.copy()
        outlet_copy[1, 1] = 0.0
        assert np.all(outlet_copy == 0.0)
        ds.close()

    def test_gauges_map_has_one_at_centre(self, preprocessor_with_dirs):
        pp = preprocessor_with_dirs
        with patch.object(pp, "_get_catchment_properties", return_value=self._mock_catchment_props()):
            pp._generate_static_maps()
        ds = xr.open_dataset(pp.maps_dir / "Gauges.nc")
        gauges = ds["Gauges"].values
        assert gauges[1, 1] == 1.0
        gauges_copy = gauges.copy()
        gauges_copy[1, 1] = 0.0
        assert np.all(gauges_copy == 0.0)
        ds.close()

    def test_land_use_fractions_sum_to_one(self, preprocessor_with_dirs):
        pp = preprocessor_with_dirs
        with patch.object(pp, "_get_catchment_properties", return_value=self._mock_catchment_props()):
            pp._generate_static_maps()

        fraction_names = [
            "DirectRunoffFraction",
            "ForestFraction",
            "OtherFraction",
            "IrrigationFraction",
            "WaterFraction",
            "RiceFraction",
        ]
        total = 0.0
        for name in fraction_names:
            ds = xr.open_dataset(pp.maps_dir / f"{name}.nc")
            val = float(ds[name].values[0, 0])
            total += val
            ds.close()
        assert abs(total - 1.0) < 1e-10, f"Land use fractions sum to {total}, expected 1.0"


class TestPreprocessorLAIMaps:
    """Tests for LAI map generation."""

    @pytest.fixture
    def preprocessor_with_maps(self, lisflood_config, mock_logger, setup_lisflood_directories):
        pp = LisfloodPreProcessor(lisflood_config, mock_logger)
        pp._create_directory_structure()
        props = {"lat": 51.17, "lon": -115.57, "area_m2": 2.21e9, "elev": 1500.0}
        with patch.object(pp, "_get_catchment_properties", return_value=props):
            pp._generate_static_maps()
        return pp

    def test_lai_maps_created(self, preprocessor_with_maps):
        pp = preprocessor_with_maps
        lai_dir = pp.maps_dir / "lai"
        assert lai_dir.exists()
        for prefix in ["laio", "laif", "laii"]:
            assert (lai_dir / f"{prefix}.nc").exists(), f"Missing LAI file {prefix}.nc"

    def test_lai_maps_have_correct_dimensions(self, preprocessor_with_maps):
        pp = preprocessor_with_maps
        lai_dir = pp.maps_dir / "lai"
        ds = xr.open_dataset(lai_dir / "laio.nc")
        assert "time" in ds["laio"].dims
        assert "y" in ds["laio"].dims
        assert "x" in ds["laio"].dims
        ds.close()

    def test_lai_maps_have_36_time_steps(self, preprocessor_with_maps):
        pp = preprocessor_with_maps
        lai_dir = pp.maps_dir / "lai"
        ds = xr.open_dataset(lai_dir / "laio.nc")
        assert ds["laio"].shape[0] == 36
        ds.close()

    def test_lai_maps_spatial_3x3(self, preprocessor_with_maps):
        pp = preprocessor_with_maps
        lai_dir = pp.maps_dir / "lai"
        ds = xr.open_dataset(lai_dir / "laif.nc")
        assert ds["laif"].shape[1] == 3
        assert ds["laif"].shape[2] == 3
        ds.close()


class TestPreprocessorLookupTables:
    """Tests for look-up table generation."""

    @pytest.fixture
    def preprocessor_with_tables(self, lisflood_config, mock_logger, setup_lisflood_directories):
        pp = LisfloodPreProcessor(lisflood_config, mock_logger)
        pp._create_directory_structure()
        pp._generate_lookup_tables()
        return pp

    def test_laiofday_created(self, preprocessor_with_tables):
        pp = preprocessor_with_tables
        assert (pp.tables_dir / "laiofday.txt").exists()

    def test_firstdayofmonth_created(self, preprocessor_with_tables):
        pp = preprocessor_with_tables
        assert (pp.tables_dir / "firstdayofmonth.txt").exists()

    def test_laiofday_content(self, preprocessor_with_tables):
        pp = preprocessor_with_tables
        content = (pp.tables_dir / "laiofday.txt").read_text()
        assert "1 366" in content

    def test_firstdayofmonth_has_12_entries(self, preprocessor_with_tables):
        pp = preprocessor_with_tables
        content = (pp.tables_dir / "firstdayofmonth.txt").read_text()
        # Should have 12 data lines (months) plus comment line
        data_lines = [l for l in content.strip().split("\n") if not l.startswith("#")]
        assert len(data_lines) == 12


class TestPreprocessorSettingsXML:
    """Tests for settings XML generation."""

    @pytest.fixture
    def preprocessor_with_xml(self, lisflood_config, mock_logger, setup_lisflood_directories):
        pp = LisfloodPreProcessor(lisflood_config, mock_logger)
        pp._create_directory_structure()
        props = {"lat": 51.17, "lon": -115.57, "area_m2": 2.21e9, "elev": 1500.0}
        with patch.object(pp, "_get_catchment_properties", return_value=props):
            pp._generate_static_maps()
        pp._generate_settings_xml()
        return pp

    def test_settings_xml_created(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        assert (pp.settings_dir / "settings.xml").exists()

    def test_settings_xml_is_valid_xml(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        tree = ET.parse(pp.settings_dir / "settings.xml")
        root = tree.getroot()
        assert root.tag == "lfsettings"

    def test_settings_xml_has_lfuser_section(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        tree = ET.parse(pp.settings_dir / "settings.xml")
        root = tree.getroot()
        assert root.find("lfuser") is not None

    def test_settings_xml_has_lfbinding_section(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        tree = ET.parse(pp.settings_dir / "settings.xml")
        root = tree.getroot()
        assert root.find("lfbinding") is not None

    def test_settings_xml_has_lfoptions_section(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        tree = ET.parse(pp.settings_dir / "settings.xml")
        root = tree.getroot()
        assert root.find("lfoptions") is not None

    def test_settings_xml_has_prolog_section(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        tree = ET.parse(pp.settings_dir / "settings.xml")
        root = tree.getroot()
        assert root.find("prolog") is not None

    def test_lfbinding_has_many_bindings(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        tree = ET.parse(pp.settings_dir / "settings.xml")
        root = tree.getroot()
        lfbinding = root.find("lfbinding")
        textvars = lfbinding.findall("textvar")
        # The _build_lfbinding_dict produces 100+ entries
        assert len(textvars) >= 100, f"Expected >=100 lfbinding entries, got {len(textvars)}"

    def test_lfbinding_contains_key_bindings(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        tree = ET.parse(pp.settings_dir / "settings.xml")
        root = tree.getroot()
        lfbinding = root.find("lfbinding")
        names = {tv.get("name") for tv in lfbinding.findall("textvar")}

        key_bindings = [
            "MaskMap",
            "Ldd",
            "Outlets",
            "Dem",
            "CellArea",
            "Gauges",
            "PrecipitationMaps",
            "TavgMaps",
            "E0Maps",
            "ET0Maps",
            "LAIOtherMaps",
            "LAIForestMaps",
            "SoilDepth1",
            "SoilDepth2",
            "SoilDepth3",
            "ForestFraction",
            "OtherFraction",
            "Channels",
            "ChanGrad",
            "ChanMan",
            "MapThetaSat1",
            "MapKSat1",
            "DischargeMaps",
            "DischargeTS",
            "SnowMeltCoef",
            "TempMelt",
            "TempSnow",
            "UpperZoneTimeConstant",
            "LowerZoneTimeConstant",
        ]
        for binding in key_bindings:
            assert binding in names, f"Missing lfbinding: {binding}"

    def test_lfoptions_has_key_options(self, preprocessor_with_xml):
        pp = preprocessor_with_xml
        tree = ET.parse(pp.settings_dir / "settings.xml")
        root = tree.getroot()
        lfoptions = root.find("lfoptions")
        options = {opt.get("name") for opt in lfoptions.findall("setoption")}

        required_options = [
            "readNetcdfStack",
            "writeNetcdfStack",
            "writeNetcdf",
            "repDischargeTs",
            "repStateMaps",
            "repDischargeMaps",
            "simulateLakes",
            "simulateReservoirs",
        ]
        for opt in required_options:
            assert opt in options, f"Missing lfoption: {opt}"


class TestPreprocessorForcing:
    """Tests for forcing data generation."""

    @pytest.fixture
    def preprocessor_with_forcing(self, tmp_path, mock_logger, sample_forcing_dir):
        """Create a preprocessor with synthetic forcing data available."""
        from symfluence.core.config.models import SymfluenceConfig

        config_dict = {
            "SYMFLUENCE_DATA_DIR": str(tmp_path),
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
        config = SymfluenceConfig(**config_dict)
        pp = LisfloodPreProcessor(config, mock_logger)
        pp._create_directory_structure()
        return pp

    def test_forcing_generates_files(self, preprocessor_with_forcing):
        pp = preprocessor_with_forcing
        props = {"lat": 51.17, "lon": -115.57, "area_m2": 2.21e9, "elev": 1500.0}
        with patch.object(pp, "_get_catchment_properties", return_value=props):
            pp._generate_forcing()

        expected_files = ["pr.nc", "ta.nc", "et0.nc", "e0.nc", "es.nc"]
        for fname in expected_files:
            fpath = pp.forcing_out_dir / fname
            assert fpath.exists(), f"Missing forcing file: {fname}"

    def test_forcing_files_have_time_dimension(self, preprocessor_with_forcing):
        pp = preprocessor_with_forcing
        props = {"lat": 51.17, "lon": -115.57, "area_m2": 2.21e9, "elev": 1500.0}
        with patch.object(pp, "_get_catchment_properties", return_value=props):
            pp._generate_forcing()

        ds = xr.open_dataset(pp.forcing_out_dir / "pr.nc")
        assert "time" in ds.dims
        assert "y" in ds.dims
        assert "x" in ds.dims
        ds.close()

    def test_forcing_precipitation_non_negative(self, preprocessor_with_forcing):
        pp = preprocessor_with_forcing
        props = {"lat": 51.17, "lon": -115.57, "area_m2": 2.21e9, "elev": 1500.0}
        with patch.object(pp, "_get_catchment_properties", return_value=props):
            pp._generate_forcing()

        ds = xr.open_dataset(pp.forcing_out_dir / "pr.nc")
        assert float(ds["pr"].min()) >= 0.0
        ds.close()


class TestPreprocessorWriteMap:
    """Tests for the _write_map helper method."""

    def test_write_map_creates_file(self, lisflood_config, mock_logger, setup_lisflood_directories):
        pp = LisfloodPreProcessor(lisflood_config, mock_logger)
        pp._create_directory_structure()

        lat = np.array([50.0, 51.0, 52.0])
        lon = np.array([-116.0, -115.0, -114.0])
        pp._write_map("test_map", 42.0, lat, lon, "test_units")

        assert (pp.maps_dir / "test_map.nc").exists()

    def test_write_map_has_correct_value(self, lisflood_config, mock_logger, setup_lisflood_directories):
        pp = LisfloodPreProcessor(lisflood_config, mock_logger)
        pp._create_directory_structure()

        lat = np.array([50.0, 51.0, 52.0])
        lon = np.array([-116.0, -115.0, -114.0])
        pp._write_map("val_map", 99.5, lat, lon)

        ds = xr.open_dataset(pp.maps_dir / "val_map.nc")
        assert np.allclose(ds["val_map"].values, 99.5)
        ds.close()

    def test_write_map_y_descending(self, lisflood_config, mock_logger, setup_lisflood_directories):
        pp = LisfloodPreProcessor(lisflood_config, mock_logger)
        pp._create_directory_structure()

        lat = np.array([50.0, 51.0, 52.0])
        lon = np.array([-116.0, -115.0, -114.0])
        pp._write_map("ydesc_map", 1.0, lat, lon)

        ds = xr.open_dataset(pp.maps_dir / "ydesc_map.nc")
        y_vals = ds["y"].values
        # _write_map uses lat[::-1] so y should be descending
        assert y_vals[0] > y_vals[-1]
        ds.close()
