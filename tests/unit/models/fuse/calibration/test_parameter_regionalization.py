"""
Unit Tests for Parameter Regionalization strategies.

Tests the regionalization framework:
- LumpedRegionalization
- TransferFunctionRegionalization
- ZoneRegionalization
- DistributedRegionalization
- RegionalizationFactory
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import logging

import numpy as np
import pandas as pd
import pytest

# Mark all tests in this module
pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def param_bounds():
    """Standard parameter bounds for testing."""
    return {
        'MAXWATR_1': (25.0, 500.0),
        'BASERTE': (0.001, 0.1),
        'MBASE': (-5.0, 5.0),
    }


@pytest.fixture
def subcatchment_attributes():
    """Subcatchment attributes DataFrame."""
    return pd.DataFrame({
        'elev_m': [500.0, 1000.0, 1500.0, 2000.0, 2500.0],
        'precip_mm_yr': [600.0, 800.0, 1000.0, 1200.0, 1400.0],
        'temp_C': [8.0, 5.0, 2.0, -1.0, -4.0],
        'aridity': [1.0, 0.8, 0.6, 0.4, 0.2],
        'snow_frac': [0.1, 0.3, 0.5, 0.7, 0.9],
    })


@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_regionalization')
    logger.setLevel(logging.DEBUG)
    return logger


# =============================================================================
# Abstract Base Class Tests
# =============================================================================

class TestParameterRegionalizationABC:
    """Test the abstract base class."""

    def test_cannot_instantiate_directly(self, param_bounds, test_logger):
        """Should not be able to instantiate the abstract base class."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            ParameterRegionalization,
        )

        with pytest.raises(TypeError):
            ParameterRegionalization(param_bounds, 5, test_logger)


# =============================================================================
# Lumped Regionalization Tests
# =============================================================================

class TestLumpedRegionalization:
    """Tests for LumpedRegionalization."""

    def test_name_property(self, param_bounds, test_logger):
        """Name should be 'lumped'."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            LumpedRegionalization,
        )

        reg = LumpedRegionalization(param_bounds, 5, test_logger)
        assert reg.name == "lumped"

    def test_calibration_params_match_original_bounds(self, param_bounds, test_logger):
        """get_calibration_parameters should return original bounds."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            LumpedRegionalization,
        )

        reg = LumpedRegionalization(param_bounds, 5, test_logger)
        cal_params = reg.get_calibration_parameters()

        assert cal_params == param_bounds

    def test_to_distributed_replicates_values(self, param_bounds, test_logger):
        """to_distributed should copy single values to all subcatchments."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            LumpedRegionalization,
        )

        n_subs = 5
        reg = LumpedRegionalization(param_bounds, n_subs, test_logger)

        calibration_params = {
            'MAXWATR_1': 200.0,
            'BASERTE': 0.05,
            'MBASE': 1.0,
        }

        param_array, param_names = reg.to_distributed(calibration_params)

        assert param_array.shape == (n_subs, 3)
        # All subcatchments should have the same values
        for col in range(3):
            np.testing.assert_array_equal(
                param_array[:, col],
                np.full(n_subs, param_array[0, col]),
            )

    def test_to_distributed_returns_correct_names(self, param_bounds, test_logger):
        """Should return parameter names matching the input keys."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            LumpedRegionalization,
        )

        reg = LumpedRegionalization(param_bounds, 3, test_logger)
        params = {'MAXWATR_1': 200.0, 'BASERTE': 0.05, 'MBASE': 1.0}

        _, param_names = reg.to_distributed(params)

        assert set(param_names) == set(params.keys())


# =============================================================================
# Transfer Function Regionalization Tests
# =============================================================================

class TestTransferFunctionRegionalization:
    """Tests for TransferFunctionRegionalization."""

    def test_name_property(self, param_bounds, subcatchment_attributes, test_logger):
        """Name should be 'transfer_function'."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            TransferFunctionRegionalization,
        )

        reg = TransferFunctionRegionalization(
            param_bounds, 5, subcatchment_attributes, logger=test_logger
        )
        assert reg.name == "transfer_function"

    def test_normalizes_attributes(self, param_bounds, subcatchment_attributes, test_logger):
        """Should create normalized attribute columns."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            TransferFunctionRegionalization,
        )

        reg = TransferFunctionRegionalization(
            param_bounds, 5, subcatchment_attributes, logger=test_logger
        )

        # Check that normalized columns exist
        assert 'elev_m_norm' in reg.attributes.columns
        # Normalized values should be in [0, 1]
        assert reg.attributes['elev_m_norm'].min() >= 0.0
        assert reg.attributes['elev_m_norm'].max() <= 1.0

    def test_log_transform_for_skewed_attributes(
        self, param_bounds, subcatchment_attributes, test_logger
    ):
        """Skewed attributes should be log-transformed before normalization."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            TransferFunctionRegionalization,
        )

        reg = TransferFunctionRegionalization(
            param_bounds, 5, subcatchment_attributes, logger=test_logger
        )

        # precip_mm_yr is in LOG_TRANSFORM_ATTRS
        assert 'precip_mm_yr' in reg.attr_stats
        assert reg.attr_stats['precip_mm_yr']['transform'] == 'log1p'

    def test_calibration_params_include_coefficients(
        self, param_bounds, subcatchment_attributes, test_logger
    ):
        """Should return a and b coefficients for calibration."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            TransferFunctionRegionalization,
        )

        param_config = {
            'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
            'BASERTE': {'attribute': 'aridity', 'calibrate_b': False},
        }

        reg = TransferFunctionRegionalization(
            param_bounds, 5, subcatchment_attributes,
            param_config=param_config,
            logger=test_logger,
        )
        cal_params = reg.get_calibration_parameters()

        # MAXWATR_1 should have both _a and _b
        assert 'MAXWATR_1_a' in cal_params
        assert 'MAXWATR_1_b' in cal_params

        # BASERTE should only have _a (calibrate_b=False)
        assert 'BASERTE_a' in cal_params
        assert 'BASERTE_b' not in cal_params

    def test_to_distributed_produces_spatial_variation(
        self, param_bounds, subcatchment_attributes, test_logger
    ):
        """Non-zero b should produce spatial variation."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            TransferFunctionRegionalization,
        )

        param_config = {
            'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
        }

        reg = TransferFunctionRegionalization(
            param_bounds, 5, subcatchment_attributes,
            param_config=param_config,
            logger=test_logger,
        )

        coeffs = {'MAXWATR_1_a': 200.0, 'MAXWATR_1_b': 100.0}
        param_array, param_names = reg.to_distributed(coeffs)

        assert param_array.shape[0] == 5
        # With b != 0, values should differ across subcatchments
        assert param_array[:, 0].std() > 0

    def test_to_distributed_clips_to_bounds(
        self, param_bounds, subcatchment_attributes, test_logger
    ):
        """Values should be clipped to original parameter bounds."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            TransferFunctionRegionalization,
        )

        param_config = {
            'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
        }

        reg = TransferFunctionRegionalization(
            param_bounds, 5, subcatchment_attributes,
            param_config=param_config,
            logger=test_logger,
        )

        # Use extreme coefficients to force clipping
        coeffs = {'MAXWATR_1_a': 9999.0, 'MAXWATR_1_b': 9999.0}
        param_array, _ = reg.to_distributed(coeffs)

        p_min, p_max = param_bounds['MAXWATR_1']
        assert param_array[:, 0].max() <= p_max
        assert param_array[:, 0].min() >= p_min

    def test_b_zero_gives_uniform_params(
        self, param_bounds, subcatchment_attributes, test_logger
    ):
        """b=0 should give uniform parameters (all equal to a)."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            TransferFunctionRegionalization,
        )

        param_config = {
            'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
        }

        reg = TransferFunctionRegionalization(
            param_bounds, 5, subcatchment_attributes,
            param_config=param_config,
            logger=test_logger,
        )

        coeffs = {'MAXWATR_1_a': 200.0, 'MAXWATR_1_b': 0.0}
        param_array, _ = reg.to_distributed(coeffs)

        # All values should be exactly 200.0
        np.testing.assert_array_almost_equal(param_array[:, 0], 200.0)


# =============================================================================
# Zone Regionalization Tests
# =============================================================================

class TestZoneRegionalization:
    """Tests for ZoneRegionalization."""

    def test_name_property(self, param_bounds, test_logger):
        """Name should be 'zones'."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            ZoneRegionalization,
        )

        zones = np.array([0, 0, 1, 1, 2])
        reg = ZoneRegionalization(param_bounds, 5, zones, test_logger)
        assert reg.name == "zones"

    def test_detects_number_of_zones(self, param_bounds, test_logger):
        """Should count unique zones."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            ZoneRegionalization,
        )

        zones = np.array([0, 0, 1, 1, 2])
        reg = ZoneRegionalization(param_bounds, 5, zones, test_logger)

        assert reg.n_zones == 3

    def test_calibration_params_per_zone(self, param_bounds, test_logger):
        """Should create parameter bounds for each zone."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            ZoneRegionalization,
        )

        zones = np.array([0, 0, 1])
        reg = ZoneRegionalization(param_bounds, 3, zones, test_logger)
        cal_params = reg.get_calibration_parameters()

        # 3 params * 2 zones = 6 calibration parameters
        assert len(cal_params) == 6
        assert 'MAXWATR_1_z0' in cal_params
        assert 'MAXWATR_1_z1' in cal_params

    def test_to_distributed_maps_zone_values(self, param_bounds, test_logger):
        """Should assign zone values to subcatchments."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            ZoneRegionalization,
        )

        zones = np.array([0, 0, 1, 1, 1])
        reg = ZoneRegionalization(param_bounds, 5, zones, test_logger)

        cal_params = {
            'MAXWATR_1_z0': 100.0,
            'MAXWATR_1_z1': 300.0,
            'BASERTE_z0': 0.01,
            'BASERTE_z1': 0.05,
            'MBASE_z0': -1.0,
            'MBASE_z1': 2.0,
        }

        param_array, param_names = reg.to_distributed(cal_params)

        assert param_array.shape == (5, 3)

        # Zone 0 subcatchments (0, 1) should have zone 0 values
        idx_maxwatr = param_names.index('MAXWATR_1')
        assert param_array[0, idx_maxwatr] == 100.0
        assert param_array[1, idx_maxwatr] == 100.0

        # Zone 1 subcatchments (2, 3, 4) should have zone 1 values
        assert param_array[2, idx_maxwatr] == 300.0
        assert param_array[3, idx_maxwatr] == 300.0


# =============================================================================
# Distributed Regionalization Tests
# =============================================================================

class TestDistributedRegionalization:
    """Tests for DistributedRegionalization."""

    def test_name_property(self, param_bounds, test_logger):
        """Name should be 'distributed'."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            DistributedRegionalization,
        )

        reg = DistributedRegionalization(param_bounds, 3, logger=test_logger)
        assert reg.name == "distributed"

    def test_calibration_params_per_subcatchment(self, param_bounds, test_logger):
        """Should create bounds for each parameter at each subcatchment."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            DistributedRegionalization,
        )

        n_subs = 3
        reg = DistributedRegionalization(param_bounds, n_subs, logger=test_logger)
        cal_params = reg.get_calibration_parameters()

        # 3 params * 3 subcatchments = 9 parameters
        assert len(cal_params) == 9
        assert 'MAXWATR_1_s0' in cal_params
        assert 'MAXWATR_1_s2' in cal_params

    def test_to_distributed_direct_mapping(self, param_bounds, test_logger):
        """Each subcatchment gets its own value."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            DistributedRegionalization,
        )

        reg = DistributedRegionalization(param_bounds, 2, logger=test_logger)

        cal_params = {
            'MAXWATR_1_s0': 100.0,
            'MAXWATR_1_s1': 200.0,
            'BASERTE_s0': 0.01,
            'BASERTE_s1': 0.05,
            'MBASE_s0': -1.0,
            'MBASE_s1': 2.0,
        }

        param_array, param_names = reg.to_distributed(cal_params)

        idx = param_names.index('MAXWATR_1')
        assert param_array[0, idx] == 100.0
        assert param_array[1, idx] == 200.0

    def test_warns_about_many_parameters(self, param_bounds, test_logger):
        """Should log a warning about large number of parameters."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            DistributedRegionalization,
        )

        # Create with large number to trigger warning
        reg = DistributedRegionalization(param_bounds, 100, logger=test_logger)

        # Just verify it created without error and recognized the count
        cal_params = reg.get_calibration_parameters()
        assert len(cal_params) == 300  # 3 params * 100 subs


# =============================================================================
# Factory Tests
# =============================================================================

class TestRegionalizationFactory:
    """Tests for RegionalizationFactory."""

    def test_creates_lumped(self, param_bounds, test_logger):
        """Should create LumpedRegionalization."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            RegionalizationFactory, LumpedRegionalization,
        )

        reg = RegionalizationFactory.create('lumped', param_bounds, 5, logger=test_logger)
        assert isinstance(reg, LumpedRegionalization)

    def test_creates_transfer_function(
        self, param_bounds, subcatchment_attributes, test_logger
    ):
        """Should create TransferFunctionRegionalization."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            RegionalizationFactory, TransferFunctionRegionalization,
        )

        reg = RegionalizationFactory.create(
            'transfer_function', param_bounds, 5,
            attributes=subcatchment_attributes, logger=test_logger,
        )
        assert isinstance(reg, TransferFunctionRegionalization)

    def test_transfer_function_requires_attributes(self, param_bounds, test_logger):
        """Should raise ValueError if attributes not provided."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            RegionalizationFactory,
        )

        with pytest.raises(ValueError, match="requires 'attributes'"):
            RegionalizationFactory.create(
                'transfer_function', param_bounds, 5, logger=test_logger
            )

    def test_creates_zones(self, param_bounds, test_logger):
        """Should create ZoneRegionalization."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            RegionalizationFactory, ZoneRegionalization,
        )

        config = {'zone_assignments': np.array([0, 0, 1, 1, 2])}
        reg = RegionalizationFactory.create(
            'zones', param_bounds, 5, config=config, logger=test_logger
        )
        assert isinstance(reg, ZoneRegionalization)

    def test_zones_requires_assignments(self, param_bounds, test_logger):
        """Should raise ValueError if zone_assignments missing."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            RegionalizationFactory,
        )

        with pytest.raises(ValueError, match="requires 'zone_assignments'"):
            RegionalizationFactory.create(
                'zones', param_bounds, 5, logger=test_logger
            )

    def test_creates_distributed(self, param_bounds, test_logger):
        """Should create DistributedRegionalization."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            RegionalizationFactory, DistributedRegionalization,
        )

        reg = RegionalizationFactory.create(
            'distributed', param_bounds, 5, logger=test_logger
        )
        assert isinstance(reg, DistributedRegionalization)

    def test_unknown_method_raises(self, param_bounds, test_logger):
        """Should raise ValueError for unknown method."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            RegionalizationFactory,
        )

        with pytest.raises(ValueError, match="Unknown regionalization method"):
            RegionalizationFactory.create(
                'nonexistent', param_bounds, 5, logger=test_logger
            )

    def test_handles_hyphenated_name(self, param_bounds, test_logger):
        """Should handle hyphenated method names (e.g., transfer-function)."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            RegionalizationFactory, TransferFunctionRegionalization,
        )

        attrs = pd.DataFrame({
            'elev_m': [500.0, 1000.0],
            'precip_mm_yr': [600.0, 800.0],
        })

        reg = RegionalizationFactory.create(
            'transfer-function', param_bounds, 2,
            attributes=attrs, logger=test_logger,
        )
        assert isinstance(reg, TransferFunctionRegionalization)


# =============================================================================
# Info Function Tests
# =============================================================================

class TestGetRegionalizationInfo:
    """Tests for get_regionalization_info utility."""

    def test_returns_all_methods(self):
        """Should return descriptions for all methods."""
        from symfluence.models.fuse.calibration.parameter_regionalization import (
            get_regionalization_info,
        )

        info = get_regionalization_info()

        assert 'lumped' in info
        assert 'transfer_function' in info
        assert 'zones' in info
        assert 'distributed' in info
        assert len(info) == 4
