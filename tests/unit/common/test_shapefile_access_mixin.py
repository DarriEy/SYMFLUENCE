"""Unit tests for ShapefileAccessMixin."""

import pytest
from unittest.mock import Mock
from symfluence.core.mixins.shapefile import ShapefileAccessMixin


class MockPaths:
    """Mock paths object that raises AttributeError for missing attributes."""

    def __init__(self, config_dict):
        self._dict = config_dict

    def __getattr__(self, name):
        if name in self._dict:
            return self._dict[name]
        raise AttributeError(f"'MockPaths' object has no attribute '{name}'")


class MockConfig:
    """Mock config object for testing."""

    def __init__(self, config_dict):
        self._dict = config_dict
        self.paths = MockPaths(config_dict)

    def to_dict(self):
        return self._dict


class TestShapefileAccessMixin:
    """Test ShapefileAccessMixin properties."""

    def _make_test_class(self, config_dict):
        """Create a test class instance with mocked config."""
        class TestClass(ShapefileAccessMixin):
            pass

        obj = TestClass()
        obj._config = MockConfig(config_dict)
        return obj

    # =========================================================================
    # Catchment Shapefile Columns
    # =========================================================================

    def test_catchment_name_col_default(self):
        """Test catchment_name_col returns default value."""
        obj = self._make_test_class({})
        assert obj.catchment_name_col == 'HRU_ID'

    def test_catchment_name_col_custom(self):
        """Test catchment_name_col with custom value."""
        obj = self._make_test_class({'catchment_name': 'CUSTOM_ID'})
        assert obj.catchment_name_col == 'CUSTOM_ID'

    def test_catchment_hruid_col_default(self):
        """Test catchment_hruid_col returns default value."""
        obj = self._make_test_class({})
        assert obj.catchment_hruid_col == 'HRU_ID'

    def test_catchment_hruid_col_custom(self):
        """Test catchment_hruid_col with custom value."""
        obj = self._make_test_class({'catchment_hruid': 'MY_HRU_ID'})
        assert obj.catchment_hruid_col == 'MY_HRU_ID'

    def test_catchment_gruid_col_default(self):
        """Test catchment_gruid_col returns default value."""
        obj = self._make_test_class({})
        assert obj.catchment_gruid_col == 'GRU_ID'

    def test_catchment_gruid_col_custom(self):
        """Test catchment_gruid_col with custom value."""
        obj = self._make_test_class({'catchment_gruid': 'MY_GRU_ID'})
        assert obj.catchment_gruid_col == 'MY_GRU_ID'

    def test_catchment_area_col_default(self):
        """Test catchment_area_col returns default value."""
        obj = self._make_test_class({})
        assert obj.catchment_area_col == 'HRU_area'

    def test_catchment_area_col_custom(self):
        """Test catchment_area_col with custom value."""
        obj = self._make_test_class({'catchment_area': 'AREA_KM2'})
        assert obj.catchment_area_col == 'AREA_KM2'

    def test_catchment_lat_col_default(self):
        """Test catchment_lat_col returns default value."""
        obj = self._make_test_class({})
        assert obj.catchment_lat_col == 'center_lat'

    def test_catchment_lat_col_custom(self):
        """Test catchment_lat_col with custom value."""
        obj = self._make_test_class({'catchment_lat': 'LAT'})
        assert obj.catchment_lat_col == 'LAT'

    def test_catchment_lon_col_default(self):
        """Test catchment_lon_col returns default value."""
        obj = self._make_test_class({})
        assert obj.catchment_lon_col == 'center_lon'

    def test_catchment_lon_col_custom(self):
        """Test catchment_lon_col with custom value."""
        obj = self._make_test_class({'catchment_lon': 'LON'})
        assert obj.catchment_lon_col == 'LON'

    # =========================================================================
    # River Network Shapefile Columns
    # =========================================================================

    def test_river_network_name_col_default(self):
        """Test river_network_name_col returns default value."""
        obj = self._make_test_class({})
        assert obj.river_network_name_col == 'LINKNO'

    def test_river_network_name_col_custom(self):
        """Test river_network_name_col with custom value."""
        obj = self._make_test_class({'river_network_name': 'SEGMENT_ID'})
        assert obj.river_network_name_col == 'SEGMENT_ID'

    def test_river_segid_col_default(self):
        """Test river_segid_col returns default value."""
        obj = self._make_test_class({})
        assert obj.river_segid_col == 'LINKNO'

    def test_river_segid_col_custom(self):
        """Test river_segid_col with custom value."""
        obj = self._make_test_class({'river_network_segid': 'SEG_ID'})
        assert obj.river_segid_col == 'SEG_ID'

    def test_river_downsegid_col_default(self):
        """Test river_downsegid_col returns default value."""
        obj = self._make_test_class({})
        assert obj.river_downsegid_col == 'DSLINKNO'

    def test_river_downsegid_col_custom(self):
        """Test river_downsegid_col with custom value."""
        obj = self._make_test_class({'river_network_downsegid': 'DS_SEG_ID'})
        assert obj.river_downsegid_col == 'DS_SEG_ID'

    def test_river_length_col_default(self):
        """Test river_length_col returns default value."""
        obj = self._make_test_class({})
        assert obj.river_length_col == 'Length'

    def test_river_length_col_custom(self):
        """Test river_length_col with custom value."""
        obj = self._make_test_class({'river_network_length': 'LENGTH_M'})
        assert obj.river_length_col == 'LENGTH_M'

    def test_river_slope_col_default(self):
        """Test river_slope_col returns default value."""
        obj = self._make_test_class({})
        assert obj.river_slope_col == 'Slope'

    def test_river_slope_col_custom(self):
        """Test river_slope_col with custom value."""
        obj = self._make_test_class({'river_network_slope': 'SLOPE_PCT'})
        assert obj.river_slope_col == 'SLOPE_PCT'

    # =========================================================================
    # River Basin Shapefile Columns
    # =========================================================================

    def test_basin_name_col_default(self):
        """Test basin_name_col returns default value."""
        obj = self._make_test_class({})
        assert obj.basin_name_col == 'GRU_ID'

    def test_basin_name_col_custom(self):
        """Test basin_name_col with custom value."""
        obj = self._make_test_class({'river_basins_name': 'BASIN_ID'})
        assert obj.basin_name_col == 'BASIN_ID'

    def test_basin_gruid_col_default(self):
        """Test basin_gruid_col returns default value."""
        obj = self._make_test_class({})
        assert obj.basin_gruid_col == 'GRU_ID'

    def test_basin_gruid_col_custom(self):
        """Test basin_gruid_col with custom value."""
        obj = self._make_test_class({'river_basin_rm_gruid': 'MY_GRU'})
        assert obj.basin_gruid_col == 'MY_GRU'

    def test_basin_hru_to_seg_col_default(self):
        """Test basin_hru_to_seg_col returns default value."""
        obj = self._make_test_class({})
        assert obj.basin_hru_to_seg_col == 'gru_to_seg'

    def test_basin_hru_to_seg_col_custom(self):
        """Test basin_hru_to_seg_col with custom value."""
        obj = self._make_test_class({'river_basin_hru_to_seg': 'hru_segment_map'})
        assert obj.basin_hru_to_seg_col == 'hru_segment_map'

    def test_basin_area_col_default(self):
        """Test basin_area_col returns default value."""
        obj = self._make_test_class({})
        assert obj.basin_area_col == 'GRU_area'

    def test_basin_area_col_custom(self):
        """Test basin_area_col with custom value."""
        obj = self._make_test_class({'river_basin_area': 'BASIN_AREA_KM2'})
        assert obj.basin_area_col == 'BASIN_AREA_KM2'

    # =========================================================================
    # Multiple Inheritance
    # =========================================================================

    def test_multiple_inheritance(self):
        """Test that mixin works with multiple inheritance."""
        class BaseClass:
            def base_method(self):
                return "base"

        class TestClass(BaseClass, ShapefileAccessMixin):
            pass

        obj = TestClass()
        obj._config = MockConfig({'catchment_hruid': 'CUSTOM_HRU'})

        # Both base class and mixin methods should work
        assert obj.base_method() == "base"
        assert obj.catchment_hruid_col == 'CUSTOM_HRU'

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_all_properties_with_empty_config(self):
        """Test that all properties return defaults with empty config."""
        obj = self._make_test_class({})

        # Just verify all properties are accessible (no exceptions)
        assert isinstance(obj.catchment_name_col, str)
        assert isinstance(obj.catchment_hruid_col, str)
        assert isinstance(obj.catchment_gruid_col, str)
        assert isinstance(obj.catchment_area_col, str)
        assert isinstance(obj.catchment_lat_col, str)
        assert isinstance(obj.catchment_lon_col, str)
        assert isinstance(obj.river_network_name_col, str)
        assert isinstance(obj.river_segid_col, str)
        assert isinstance(obj.river_downsegid_col, str)
        assert isinstance(obj.river_length_col, str)
        assert isinstance(obj.river_slope_col, str)
        assert isinstance(obj.basin_name_col, str)
        assert isinstance(obj.basin_gruid_col, str)
        assert isinstance(obj.basin_hru_to_seg_col, str)
        assert isinstance(obj.basin_area_col, str)

    def test_all_properties_with_full_config(self):
        """Test that all properties use custom values when provided."""
        custom_config = {
            'catchment_name': 'C_NAME',
            'catchment_hruid': 'C_HRUID',
            'catchment_gruid': 'C_GRUID',
            'catchment_area': 'C_AREA',
            'catchment_lat': 'C_LAT',
            'catchment_lon': 'C_LON',
            'river_network_name': 'R_NAME',
            'river_network_segid': 'R_SEGID',
            'river_network_downsegid': 'R_DOWNSEGID',
            'river_network_length': 'R_LENGTH',
            'river_network_slope': 'R_SLOPE',
            'river_basins_name': 'B_NAME',
            'river_basin_rm_gruid': 'B_GRUID',
            'river_basin_hru_to_seg': 'B_HRU_TO_SEG',
            'river_basin_area': 'B_AREA',
        }
        obj = self._make_test_class(custom_config)

        # Verify all custom values are returned
        assert obj.catchment_name_col == 'C_NAME'
        assert obj.catchment_hruid_col == 'C_HRUID'
        assert obj.catchment_gruid_col == 'C_GRUID'
        assert obj.catchment_area_col == 'C_AREA'
        assert obj.catchment_lat_col == 'C_LAT'
        assert obj.catchment_lon_col == 'C_LON'
        assert obj.river_network_name_col == 'R_NAME'
        assert obj.river_segid_col == 'R_SEGID'
        assert obj.river_downsegid_col == 'R_DOWNSEGID'
        assert obj.river_length_col == 'R_LENGTH'
        assert obj.river_slope_col == 'R_SLOPE'
        assert obj.basin_name_col == 'B_NAME'
        assert obj.basin_gruid_col == 'B_GRUID'
        assert obj.basin_hru_to_seg_col == 'B_HRU_TO_SEG'
        assert obj.basin_area_col == 'B_AREA'
