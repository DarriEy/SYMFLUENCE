"""Tests for FEWS adapter configuration models."""

import pytest
from pydantic import ValidationError

from symfluence.fews.config import FEWSConfig, IDMapEntry


class TestIDMapEntry:
    def test_basic_entry(self):
        entry = IDMapEntry(fews_id="P.obs", symfluence_id="pptrate")
        assert entry.fews_id == "P.obs"
        assert entry.symfluence_id == "pptrate"
        assert entry.conversion_factor == 1.0
        assert entry.conversion_offset == 0.0

    def test_entry_with_conversion(self):
        entry = IDMapEntry(
            fews_id="T.obs",
            symfluence_id="airtemp",
            fews_unit="degC",
            symfluence_unit="K",
            conversion_offset=273.15,
        )
        assert entry.conversion_offset == 273.15
        assert entry.fews_unit == "degC"

    def test_frozen(self):
        entry = IDMapEntry(fews_id="P.obs", symfluence_id="pptrate")
        with pytest.raises(ValidationError):
            entry.fews_id = "Q.obs"


class TestFEWSConfig:
    def test_defaults(self):
        cfg = FEWSConfig()
        assert cfg.work_dir == "."
        assert cfg.data_format == "netcdf-cf"
        assert cfg.id_map == []
        assert cfg.auto_id_map is True
        assert cfg.missing_value == -999.0

    def test_with_inline_id_map(self, sample_fews_config):
        assert len(sample_fews_config.id_map) == 2
        assert sample_fews_config.id_map[0].fews_id == "P.obs"

    def test_frozen(self):
        cfg = FEWSConfig()
        with pytest.raises(ValidationError):
            cfg.work_dir = "/new/path"

    def test_alias_access(self):
        cfg = FEWSConfig(FEWS_WORK_DIR="/tmp/fews")
        assert cfg.work_dir == "/tmp/fews"

    def test_invalid_format(self):
        with pytest.raises(ValidationError):
            FEWSConfig(data_format="csv")

    def test_optional_fields_default_none(self):
        cfg = FEWSConfig()
        assert cfg.id_map_file is None
        assert cfg.state_dir is None
