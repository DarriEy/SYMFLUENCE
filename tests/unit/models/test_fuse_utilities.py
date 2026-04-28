"""
Tests for FUSE model utilities.

Tests FUSE-specific utility functions including mizuRoute conversion.
"""

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import xarray as xr

from symfluence.models.fuse.utilities import FuseToMizurouteConverter


class TestFuseToMizurouteConverter:
    """Test FuseToMizurouteConverter class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = FuseToMizurouteConverter()

    def test_converter_initialization(self):
        """Test converter can be instantiated."""
        assert self.converter is not None
        assert self.converter.logger is not None

    def test_fuse_runoff_vars_defined(self):
        """Test that FUSE runoff variable names are defined."""
        assert hasattr(FuseToMizurouteConverter, 'FUSE_RUNOFF_VARS')
        assert 'q_routed' in FuseToMizurouteConverter.FUSE_RUNOFF_VARS
        assert 'q_instnt' in FuseToMizurouteConverter.FUSE_RUNOFF_VARS
        assert 'total_discharge' in FuseToMizurouteConverter.FUSE_RUNOFF_VARS
        assert 'runoff' in FuseToMizurouteConverter.FUSE_RUNOFF_VARS

    def test_converter_has_convert_method(self):
        """Test converter has convert method."""
        assert hasattr(self.converter, 'convert')
        assert callable(self.converter.convert)


class TestEnsureRoutingVariable:
    """Test FUSERunner._ensure_routing_variable renames FUSE output vars."""

    def _make_runner_stub(self, routing_var='q_routed'):
        """Create a minimal object that has _ensure_routing_variable."""
        from symfluence.models.fuse.runner import FUSERunner
        stub = object.__new__(FUSERunner)
        stub.logger = MagicMock()
        type(stub).mizu_routing_var = PropertyMock(return_value=routing_var)
        return stub

    def _write_dataset(self, path, var_name='q_instnt'):
        ds = xr.Dataset({
            var_name: xr.DataArray(
                np.random.rand(3, 2),
                dims=('time', 'gru'),
            )
        })
        ds.to_netcdf(path)
        return ds

    def test_renames_q_instnt_to_q_routed(self, tmp_path):
        nc = tmp_path / 'test.nc'
        self._write_dataset(nc, 'q_instnt')
        runner = self._make_runner_stub('q_routed')

        runner._ensure_routing_variable(nc)

        ds = xr.open_dataset(nc)
        assert 'q_routed' in ds.data_vars
        assert 'q_instnt' not in ds.data_vars
        ds.close()

    def test_noop_when_variable_already_correct(self, tmp_path):
        nc = tmp_path / 'test.nc'
        self._write_dataset(nc, 'q_routed')
        runner = self._make_runner_stub('q_routed')

        runner._ensure_routing_variable(nc)

        ds = xr.open_dataset(nc)
        assert 'q_routed' in ds.data_vars
        ds.close()

    def test_noop_when_no_runoff_variable(self, tmp_path):
        nc = tmp_path / 'test.nc'
        ds = xr.Dataset({'temperature': xr.DataArray([1, 2, 3], dims=('time',))})
        ds.to_netcdf(nc)

        runner = self._make_runner_stub('q_routed')
        runner._ensure_routing_variable(nc)

        ds = xr.open_dataset(nc)
        assert 'temperature' in ds.data_vars
        assert 'q_routed' not in ds.data_vars
        ds.close()

    def test_renames_total_discharge_fallback(self, tmp_path):
        nc = tmp_path / 'test.nc'
        self._write_dataset(nc, 'total_discharge')
        runner = self._make_runner_stub('q_routed')

        runner._ensure_routing_variable(nc)

        ds = xr.open_dataset(nc)
        assert 'q_routed' in ds.data_vars
        assert 'total_discharge' not in ds.data_vars
        ds.close()


class TestFuseUtilitiesBackwardCompatibility:
    """Test backward compatibility with old import paths."""

    def test_import_from_optimization_utilities(self):
        """Test that old import path may still work if re-exported."""
        # Note: We didn't create a re-export for this, so this test
        # documents that the old path is deprecated
        with pytest.raises(ImportError):
            from symfluence.optimization.utilities.fuse_utilities import FuseToMizurouteConverter

    def test_import_from_models_fuse_utilities(self):
        """Test new import path."""
        from symfluence.models.fuse.utilities import FuseToMizurouteConverter as NewConverter

        converter = NewConverter()
        assert converter is not None
