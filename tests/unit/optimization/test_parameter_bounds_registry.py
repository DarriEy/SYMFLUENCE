"""
Unit tests for Parameter Bounds Registry

Tests that parameter bounds are correctly registered for all models.
"""

import pytest
from symfluence.optimization.core.parameter_bounds_registry import get_hype_bounds, get_mesh_bounds


class TestParameterBoundsRegistry:
    """Test that bounds are correctly registered."""

    def test_hype_bounds_exist(self):
        """Test that all HYPE parameters have bounds."""
        bounds = get_hype_bounds()

        expected_params = [
            'ttmp', 'cmlt', 'ttpi', 'cmrefr',  # Snow
            'cevp', 'lp', 'epotdist',  # ET
            'rrcs1', 'rrcs2', 'rrcs3', 'wcwp', 'wcfc', 'wcep', 'srrcs',  # Soil
            'rivvel', 'damp', 'qmean',  # Routing
            'ilratk', 'ilratp',  # Lakes
        ]

        for param in expected_params:
            assert param in bounds
            assert 'min' in bounds[param]
            assert 'max' in bounds[param]
            assert bounds[param]['min'] < bounds[param]['max']

    def test_mesh_bounds_exist(self):
        """Test that all MESH parameters have bounds."""
        bounds = get_mesh_bounds()

        expected_params = [
            'ZSNL', 'ZPLG', 'ZPLS', 'FRZTH', 'MANN',  # CLASS
            'RCHARG', 'DRAINFRAC', 'BASEFLW',  # Hydrology
            'DTMINUSR',  # Routing
        ]

        for param in expected_params:
            assert param in bounds
            assert 'min' in bounds[param]
            assert 'max' in bounds[param]
            assert bounds[param]['min'] < bounds[param]['max']
