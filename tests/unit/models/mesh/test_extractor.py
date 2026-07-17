"""
Tests for MESH result extractor.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest


class TestMESHResultExtractor:
    """Tests for MESH result extraction."""

    def test_extractor_can_be_imported(self):
        """Test that MESHResultExtractor can be imported."""
        from symfluence.models.mesh.extractor import MESHResultExtractor
        assert MESHResultExtractor is not None

    def test_extractor_initialization(self):
        """Test extractor initializes with model name."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        assert extractor is not None
        assert extractor.model_name == 'MESH'


class TestMESHOutputFilePatterns:
    """Tests for MESH output file patterns."""

    def test_get_output_file_patterns(self):
        """Test extractor returns correct output file patterns."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        patterns = extractor.get_output_file_patterns()

        assert 'streamflow' in patterns
        assert 'et' in patterns
        assert 'snow' in patterns

    def test_streamflow_patterns(self):
        """Test streamflow file patterns."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        patterns = extractor.get_output_file_patterns()

        streamflow_patterns = patterns['streamflow']
        assert 'MESH_output_streamflow.csv' in streamflow_patterns

    def test_et_patterns(self):
        """Test ET file patterns."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        patterns = extractor.get_output_file_patterns()

        et_patterns = patterns['et']
        assert 'GRU_water_balance.csv' in et_patterns


class TestMESHVariableNames:
    """Tests for MESH variable name mapping."""

    def test_get_streamflow_variable_names(self):
        """Test streamflow variable names."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        var_names = extractor.get_variable_names('streamflow')

        assert 'QOSIM' in var_names

    def test_get_et_variable_names(self):
        """Test ET variable names."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        var_names = extractor.get_variable_names('et')

        assert 'EVAP' in var_names or 'ET' in var_names

    def test_get_snow_variable_names(self):
        """Test snow variable names."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        var_names = extractor.get_variable_names('snow')

        assert 'SNOW' in var_names or 'SWE' in var_names

    def test_get_unknown_variable_type(self):
        """Test unknown variable type returns variable itself."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        var_names = extractor.get_variable_names('custom_var')

        assert 'custom_var' in var_names


class TestMESHVariableExtraction:
    """Tests for MESH variable extraction."""

    def test_extract_streamflow(self, sample_mesh_output_csv):
        """Test streamflow extraction from CSV."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        result = extractor.extract_variable(sample_mesh_output_csv, 'streamflow')

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_extract_streamflow_values(self, sample_mesh_output_csv):
        """Test extracted streamflow values are correct."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        result = extractor.extract_variable(sample_mesh_output_csv, 'streamflow')

        # First value should be 11.2 (QOSIM1)
        assert abs(result.iloc[0] - 11.2) < 0.01

    def test_extract_with_subbasin_index(self, sample_mesh_output_csv):
        """Test extraction with specific subbasin index."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        result = extractor.extract_variable(
            sample_mesh_output_csv, 'streamflow', subbasin_index=1
        )

        # Should extract QOSIM2, first value is 5.1
        assert abs(result.iloc[0] - 5.1) < 0.01

    def test_extract_rejects_non_csv(self, setup_mesh_directories):
        """Test extractor rejects non-CSV files."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')

        nc_file = setup_mesh_directories['forcing_dir'] / 'test.nc'
        nc_file.touch()

        with pytest.raises(ValueError, match="only supports CSV"):
            extractor.extract_variable(nc_file, 'streamflow')


class TestMESHJulianDateConversion:
    """Tests for Julian date conversion."""

    def test_julian_to_datetime(self):
        """Test Julian day to datetime conversion."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')

        # Day 1 of 2020 should be Jan 1, 2020
        row = pd.Series({'DAY': 1, 'YEAR': 2020})
        result = extractor._julian_to_datetime(row)

        assert result.year == 2020
        assert result.month == 1
        assert result.day == 1

    def test_julian_to_datetime_mid_year(self):
        """Test Julian day conversion for mid-year date."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')

        # Day 182 of 2020 (leap year) should be June 30, 2020
        row = pd.Series({'DAY': 182, 'YEAR': 2020})
        result = extractor._julian_to_datetime(row)

        assert result.year == 2020
        assert result.month == 6
        assert result.day == 30


class TestMESHExtractorProperties:
    """Tests for MESH extractor properties."""

    def test_requires_unit_conversion(self):
        """Test MESH outputs don't require unit conversion."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        assert extractor.requires_unit_conversion('streamflow') is False

    def test_get_spatial_aggregation_method(self):
        """Test spatial aggregation method is selection."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        extractor = MESHResultExtractor('MESH')
        method = extractor.get_spatial_aggregation_method('streamflow')

        assert method == 'selection'


class TestBasinAvgWaterBalanceStreamflow:
    """Streamflow reconstruction from the basin-average water balance.

    Guards the routing-vs-noroute total-discharge accounting: in a routing run
    the basin-average RFF already equals OVRFLW + LATFLW + LKG (total runoff),
    so RFF alone is the streamflow; adding LKG again double-counts baseflow. In
    a noroute run LKG == 0 and the deep drainage leaves via DRAINSOL, which RFF
    excludes, so DRAINSOL must be added.
    """

    def _write_wb(self, path, rff, lkg, drainsol):
        import pandas as pd
        n = len(rff)
        df = pd.DataFrame({
            'YEAR': [2004] * n,
            'JDAY': list(range(1, n + 1)),
            'RFF': rff,
            'OVRFLW': [0.0] * n,
            'LATFLW': [0.0] * n,
            'LKG': lkg,
            'DRAINSOL': drainsol,
        })
        df.to_csv(path, index=False)

    def test_routing_mode_uses_rff_alone_no_lkg_double_count(self, tmp_path):
        """LKG active => total == RFF (baseflow already inside RFF)."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        # RFF = OVRFLW + LATFLW + LKG already, so RFF is the total.
        wb = tmp_path / 'Basin_average_water_balance.csv'
        self._write_wb(wb, rff=[2.0, 3.0, 4.0], lkg=[1.5, 2.0, 2.5], drainsol=[1.5, 2.0, 2.5])
        s = MESHResultExtractor('MESH').extract_variable(
            wb, 'runoff', start_date=datetime(2004, 1, 1), aggregate='daily')
        # Must equal RFF, NOT RFF + LKG (which would be 3.5, 5.0, 6.5)
        assert list(s.round(4).values) == [2.0, 3.0, 4.0]

    def test_noroute_mode_adds_drainsol_as_baseflow(self, tmp_path):
        """LKG == 0 => total == RFF + DRAINSOL (deep drainage is the baseflow)."""
        from symfluence.models.mesh.extractor import MESHResultExtractor

        wb = tmp_path / 'Basin_average_water_balance.csv'
        self._write_wb(wb, rff=[0.05, 0.06, 0.07], lkg=[0.0, 0.0, 0.0], drainsol=[1.0, 1.1, 1.2])
        s = MESHResultExtractor('MESH').extract_variable(
            wb, 'runoff', start_date=datetime(2004, 1, 1), aggregate='daily')
        assert list(s.round(4).values) == [1.05, 1.16, 1.27]
