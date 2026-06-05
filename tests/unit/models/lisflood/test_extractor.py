# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for LISFLOOD result extractor."""
from __future__ import annotations

import pytest

from symfluence.models.lisflood.extractor import LisfloodResultExtractor


@pytest.fixture
def extractor():
    """Create a LisfloodResultExtractor instance."""
    return LisfloodResultExtractor(model_name="LISFLOOD")


class TestExtractorImport:
    """Basic import tests."""

    def test_extractor_can_be_imported(self):
        assert LisfloodResultExtractor is not None

    def test_extractor_instantiation(self, extractor):
        assert extractor is not None


class TestGetOutputFilePatterns:
    """Tests for get_output_file_patterns."""

    def test_returns_dict(self, extractor):
        patterns = extractor.get_output_file_patterns()
        assert isinstance(patterns, dict)

    def test_has_streamflow_patterns(self, extractor):
        patterns = extractor.get_output_file_patterns()
        assert "streamflow" in patterns
        assert isinstance(patterns["streamflow"], list)
        assert len(patterns["streamflow"]) > 0

    def test_has_et_patterns(self, extractor):
        patterns = extractor.get_output_file_patterns()
        assert "et" in patterns

    def test_has_snow_patterns(self, extractor):
        patterns = extractor.get_output_file_patterns()
        assert "snow" in patterns

    def test_has_soil_moisture_patterns(self, extractor):
        patterns = extractor.get_output_file_patterns()
        assert "soil_moisture" in patterns

    def test_streamflow_patterns_include_dis(self, extractor):
        patterns = extractor.get_output_file_patterns()["streamflow"]
        has_dis = any("dis" in p for p in patterns)
        assert has_dis, f"No discharge pattern found in {patterns}"


class TestGetVariableNames:
    """Tests for get_variable_names."""

    def test_streamflow_variables(self, extractor):
        names = extractor.get_variable_names("streamflow")
        assert "dis" in names
        assert "discharge" in names

    def test_et_variables(self, extractor):
        names = extractor.get_variable_names("et")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_snow_variables(self, extractor):
        names = extractor.get_variable_names("snow")
        assert isinstance(names, list)
        assert "SnowMelt" in names or "SnowCover" in names

    def test_soil_moisture_variables(self, extractor):
        names = extractor.get_variable_names("soil_moisture")
        assert isinstance(names, list)
        assert any("theta" in v.lower() for v in names)

    def test_unknown_type_returns_type_as_list(self, extractor):
        names = extractor.get_variable_names("some_unknown_type")
        assert names == ["some_unknown_type"]


class TestSpatialAggregationMethod:
    """Tests for get_spatial_aggregation_method."""

    def test_streamflow_returns_max(self, extractor):
        method = extractor.get_spatial_aggregation_method("streamflow")
        assert method == "max"

    def test_et_returns_sum(self, extractor):
        method = extractor.get_spatial_aggregation_method("et")
        assert method == "sum"

    def test_precipitation_returns_sum(self, extractor):
        method = extractor.get_spatial_aggregation_method("precipitation")
        assert method == "sum"

    def test_other_returns_mean(self, extractor):
        method = extractor.get_spatial_aggregation_method("soil_moisture")
        assert method == "mean"


class TestRequiresUnitConversion:
    """Tests for requires_unit_conversion."""

    def test_streamflow_no_conversion(self, extractor):
        assert extractor.requires_unit_conversion("streamflow") is False

    def test_et_no_conversion(self, extractor):
        assert extractor.requires_unit_conversion("et") is False


class TestExtractVariable:
    """Tests for extract_variable with sample data."""

    def test_extract_streamflow_from_netcdf(self, extractor, tmp_path):
        import numpy as np
        import pandas as pd
        import xarray as xr

        times = pd.date_range("2005-01-01", periods=10, freq="D")
        dis_data = np.random.uniform(1.0, 50.0, (10, 3, 3))
        ds = xr.Dataset(
            {
                "dis": xr.DataArray(
                    dis_data,
                    dims=["time", "y", "x"],
                    coords={"time": times, "y": [2, 1, 0], "x": [0, 1, 2]},
                ),
            }
        )
        nc_path = tmp_path / "dis.nc"
        ds.to_netcdf(nc_path)

        result = extractor.extract_variable(nc_path, "streamflow")
        assert len(result) == 10
        # Should use 'max' aggregation for streamflow
        for t in range(10):
            expected_max = float(dis_data[t].max())
            assert result.iloc[t] == pytest.approx(expected_max, rel=1e-5)

    def test_extract_raises_for_missing_variable(self, extractor, tmp_path):
        import numpy as np
        import pandas as pd
        import xarray as xr

        times = pd.date_range("2005-01-01", periods=5, freq="D")
        ds = xr.Dataset(
            {
                "some_other_var": xr.DataArray(
                    np.zeros(5),
                    dims=["time"],
                    coords={"time": times},
                ),
            }
        )
        nc_path = tmp_path / "other.nc"
        ds.to_netcdf(nc_path)

        with pytest.raises(ValueError, match="No suitable variable found"):
            extractor.extract_variable(nc_path, "streamflow")

    def test_extract_streamflow_from_output_dir(self, extractor, tmp_path):
        import numpy as np
        import pandas as pd
        import xarray as xr

        times = pd.date_range("2005-01-01", periods=5, freq="D")
        ds = xr.Dataset(
            {
                "dis": xr.DataArray(
                    np.ones((5, 3, 3)),
                    dims=["time", "y", "x"],
                    coords={"time": times, "y": [2, 1, 0], "x": [0, 1, 2]},
                ),
            }
        )
        ds.to_netcdf(tmp_path / "dis.nc")

        result = extractor.extract_streamflow(tmp_path)
        assert len(result) == 5

    def test_extract_streamflow_raises_for_empty_dir(self, extractor, tmp_path):
        with pytest.raises(FileNotFoundError, match="No LISFLOOD output file"):
            extractor.extract_streamflow(tmp_path)
