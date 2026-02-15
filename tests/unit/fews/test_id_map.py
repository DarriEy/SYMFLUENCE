"""Tests for FEWS variable ID mapper."""

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.config import FEWSConfig, IDMapEntry
from symfluence.fews.exceptions import IDMappingError
from symfluence.fews.id_map import IDMapper


class TestIDMapperForward:
    def test_explicit_mapping(self, sample_fews_config):
        mapper = IDMapper(sample_fews_config)
        name, factor, offset = mapper.fews_to_symfluence("P.obs")
        assert name == "pptrate"
        assert factor == 1.0
        assert offset == 0.0

    def test_auto_detect(self):
        cfg = FEWSConfig(auto_id_map=True)
        mapper = IDMapper(cfg)
        name, _, _ = mapper.fews_to_symfluence("P.obs")
        assert name == "pptrate"

    def test_pass_through(self):
        cfg = FEWSConfig(auto_id_map=False)
        mapper = IDMapper(cfg)
        name, factor, offset = mapper.fews_to_symfluence("UnknownVar")
        assert name == "UnknownVar"
        assert factor == 1.0

    def test_conversion_factor(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        name, factor, offset = mapper.fews_to_symfluence("P.obs")
        assert name == "pptrate"
        assert pytest.approx(factor) == 1.0 / 3600.0

    def test_conversion_offset(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        name, factor, offset = mapper.fews_to_symfluence("T.obs")
        assert name == "airtemp"
        assert pytest.approx(offset) == 273.15


class TestIDMapperReverse:
    def test_reverse_mapping(self, sample_fews_config):
        mapper = IDMapper(sample_fews_config)
        name, inv_factor, inv_offset = mapper.symfluence_to_fews("pptrate")
        assert name == "P.obs"
        assert inv_factor == 1.0

    def test_reverse_with_factor(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        name, inv_factor, inv_offset = mapper.symfluence_to_fews("pptrate")
        assert name == "P.obs"
        assert pytest.approx(inv_factor) == 3600.0

    def test_reverse_pass_through(self):
        cfg = FEWSConfig(auto_id_map=False)
        mapper = IDMapper(cfg)
        name, _, _ = mapper.symfluence_to_fews("custom_var")
        assert name == "custom_var"


class TestIDMapperYAML:
    def test_load_yaml(self, tmp_path):
        yaml_content = """
- fews_id: WL.obs
  symfluence_id: water_level
  fews_unit: m
  symfluence_unit: m
- fews_id: Q.sim
  symfluence_id: discharge
"""
        yaml_path = tmp_path / "id_map.yaml"
        yaml_path.write_text(yaml_content)

        cfg = FEWSConfig(id_map_file=str(yaml_path), auto_id_map=False)
        mapper = IDMapper(cfg)
        name, _, _ = mapper.fews_to_symfluence("WL.obs")
        assert name == "water_level"

    def test_missing_yaml(self, tmp_path):
        cfg = FEWSConfig(id_map_file=str(tmp_path / "nonexistent.yaml"), auto_id_map=False)
        # Should warn but not raise
        mapper = IDMapper(cfg)
        assert mapper is not None

    def test_invalid_yaml_format(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("key: value")  # Not a list
        cfg = FEWSConfig(id_map_file=str(yaml_path), auto_id_map=False)
        with pytest.raises(IDMappingError, match="must be a list"):
            IDMapper(cfg)


class TestIDMapperDataset:
    def test_rename_fews_to_sym(self, sample_fews_config):
        mapper = IDMapper(sample_fews_config)
        ds = xr.Dataset({
            "P.obs": ("time", [1.0, 2.0]),
            "T.obs": ("time", [280.0, 281.0]),
        })
        result = mapper.rename_dataset_fews_to_sym(ds)
        assert "pptrate" in result.data_vars
        assert "airtemp" in result.data_vars

    def test_rename_sym_to_fews(self, sample_fews_config):
        mapper = IDMapper(sample_fews_config)
        ds = xr.Dataset({
            "pptrate": ("time", [1.0, 2.0]),
            "airtemp": ("time", [280.0, 281.0]),
        })
        result = mapper.rename_dataset_sym_to_fews(ds)
        assert "P.obs" in result.data_vars
        assert "T.obs" in result.data_vars

    def test_conversion_applied(self, sample_fews_config_with_conversion):
        mapper = IDMapper(sample_fews_config_with_conversion)
        ds = xr.Dataset({
            "T.obs": ("time", [0.0, 10.0]),  # degC
        })
        result = mapper.rename_dataset_fews_to_sym(ds)
        # Should be K after conversion
        np.testing.assert_allclose(result["airtemp"].values, [273.15, 283.15], rtol=1e-5)
