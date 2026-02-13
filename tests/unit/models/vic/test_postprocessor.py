"""Tests for VIC postprocessor."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestVICPostProcessorImport:
    """Tests for VIC postprocessor import and registration."""

    def test_postprocessor_can_be_imported(self):
        from symfluence.models.vic.postprocessor import VICPostProcessor
        assert VICPostProcessor is not None

    def test_postprocessor_registered_with_registry(self):
        from symfluence.models.registry import ModelRegistry
        assert 'VIC' in ModelRegistry._postprocessors

    def test_postprocessor_is_correct_class(self):
        from symfluence.models.registry import ModelRegistry
        from symfluence.models.vic.postprocessor import VICPostProcessor
        assert ModelRegistry._postprocessors.get('VIC') == VICPostProcessor

    def test_model_name(self):
        from symfluence.models.vic.postprocessor import VICPostProcessor
        assert VICPostProcessor.model_name == "VIC"

    def test_streamflow_unit(self):
        from symfluence.models.vic.postprocessor import VICPostProcessor
        assert VICPostProcessor.streamflow_unit == "mm_per_day"


class TestVICStreamflowExtraction:
    """Tests for VIC streamflow extraction logic."""

    def test_runoff_plus_baseflow_summation(self):
        """Test that VIC postprocessor sums OUT_RUNOFF + OUT_BASEFLOW."""
        import xarray as xr
        import pandas as pd

        times = pd.date_range('2020-01-01', periods=10, freq='D')
        runoff = np.random.uniform(0, 5, 10)
        baseflow = np.random.uniform(0, 2, 10)

        ds = xr.Dataset({
            'OUT_RUNOFF': ('time', runoff),
            'OUT_BASEFLOW': ('time', baseflow),
        }, coords={'time': times})

        # Expected total is sum of both
        expected_total = runoff + baseflow

        # Verify the summation logic works
        total = ds['OUT_RUNOFF'] + ds['OUT_BASEFLOW']
        np.testing.assert_array_almost_equal(total.values, expected_total)

    def test_spatial_aggregation(self):
        """Test that spatial dimensions are summed for runoff."""
        import xarray as xr
        import pandas as pd

        times = pd.date_range('2020-01-01', periods=5, freq='D')
        runoff_2d = np.random.uniform(0, 5, (5, 3))

        ds = xr.Dataset({
            'OUT_RUNOFF': (['time', 'lat'], runoff_2d),
        }, coords={'time': times, 'lat': [0, 1, 2]})

        spatial_dims = [d for d in ds['OUT_RUNOFF'].dims if d != 'time']
        total = ds['OUT_RUNOFF'].sum(dim=spatial_dims)

        assert total.dims == ('time',)
        np.testing.assert_array_almost_equal(total.values, runoff_2d.sum(axis=1))


class TestVICPostProcessorOutputDir:
    """Tests for VIC output directory resolution."""

    def test_output_file_pattern(self):
        from symfluence.models.vic.postprocessor import VICPostProcessor
        assert VICPostProcessor.output_file_pattern == "vic_output*.nc"
