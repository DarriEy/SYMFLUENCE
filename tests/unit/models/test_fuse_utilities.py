"""
Tests for FUSE model utilities.

Tests FUSE-specific utility functions including mizuRoute conversion.
"""

import pytest
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
