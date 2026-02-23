"""
Unit tests for coordinate_utils module.

Tests bounding box parsing, coordinate normalization, and related utilities.
"""

import numpy as np
import pytest
from symfluence.geospatial.coordinate_utils import (
    BoundingBox,
    CoordinateUtilsMixin,
    create_coordinate_mask,
    normalize_longitude,
    parse_bbox,
)


class TestParseBBox:
    """Test parse_bbox() function."""

    def test_parse_bbox_default_format(self):
        """Test parsing with default system standard format (N/W/S/E)."""
        bbox = parse_bbox("60.0/-130.0/50.0/-120.0")
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == -130.0
        assert bbox['lon_max'] == -120.0

    def test_parse_bbox_system_standard_explicit(self):
        """Test parsing with explicit system standard format."""
        bbox = parse_bbox("60.0/-130.0/50.0/-120.0", format='lat_max/lon_min/lat_min/lon_max')
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == -130.0
        assert bbox['lon_max'] == -120.0

    def test_parse_bbox_ne_sw_format(self):
        """Test parsing with NE corner then SW corner format."""
        bbox = parse_bbox("60.0/-120.0/50.0/-130.0", format='lat_max/lon_max/lat_min/lon_min')
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == -130.0
        assert bbox['lon_max'] == -120.0

    def test_parse_bbox_sw_ne_format(self):
        """Test parsing with SW corner then NE corner format."""
        bbox = parse_bbox("50.0/-130.0/60.0/-120.0", format='lat_min/lon_min/lat_max/lon_max')
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == -130.0
        assert bbox['lon_max'] == -120.0

    def test_parse_bbox_lon_lat_format(self):
        """Test parsing with lon/lat order."""
        bbox = parse_bbox("-130.0/50.0/-120.0/60.0", format='lon_min/lat_min/lon_max/lat_max')
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == -130.0
        assert bbox['lon_max'] == -120.0

    def test_parse_bbox_empty_string(self):
        """Test that empty string returns empty dict."""
        assert parse_bbox("") == {}
        assert parse_bbox(None) == {}

    def test_parse_bbox_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        bbox = parse_bbox("60.0 / -130.0 / 50.0 / -120.0")
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0

    def test_parse_bbox_positive_coordinates(self):
        """Test with positive latitude and longitude (e.g., Eastern hemisphere)."""
        bbox = parse_bbox("60.0/10.0/50.0/20.0")  # N/W/S/E with positive W
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == 10.0
        assert bbox['lon_max'] == 20.0

    def test_parse_bbox_0_360_longitude(self):
        """Test with 0-360 longitude range."""
        bbox = parse_bbox("60.0/240.0/50.0/250.0")  # Pacific region in 0-360
        assert bbox['lon_min'] == 240.0
        assert bbox['lon_max'] == 250.0

    def test_parse_bbox_invalid_coordinate_count(self):
        """Test that incorrect number of coordinates raises ValueError."""
        with pytest.raises(ValueError, match="Expected 4 coordinates"):
            parse_bbox("60.0/-130.0/50.0")

        with pytest.raises(ValueError, match="Expected 4 coordinates"):
            parse_bbox("60.0/-130.0/50.0/-120.0/5.0")

    def test_parse_bbox_invalid_format(self):
        """Test that unknown format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown bbox format"):
            parse_bbox("60.0/-130.0/50.0/-120.0", format='unknown/format')

    def test_parse_bbox_non_numeric(self):
        """Test that non-numeric values raise ValueError."""
        with pytest.raises(ValueError):
            parse_bbox("60.0/invalid/50.0/-120.0")


class TestBoundingBox:
    """Test BoundingBox class."""

    def test_init_valid_coordinates(self):
        """Test initialization with valid coordinates."""
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)
        assert bbox.lat_min == 50.0
        assert bbox.lat_max == 60.0
        assert bbox.lon_min == -130.0
        assert bbox.lon_max == -120.0

    def test_init_swaps_reversed_latitudes(self):
        """Test that reversed lat_min/lat_max are automatically swapped."""
        bbox = BoundingBox(lat_min=60.0, lat_max=50.0, lon_min=-130.0, lon_max=-120.0)
        assert bbox.lat_min == 50.0
        assert bbox.lat_max == 60.0

    def test_init_invalid_latitude_range(self):
        """Test that latitudes outside [-90, 90] raise ValueError."""
        with pytest.raises(ValueError, match="Latitudes must be in"):
            BoundingBox(lat_min=95.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)

        with pytest.raises(ValueError, match="Latitudes must be in"):
            BoundingBox(lat_min=-95.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)
        d = bbox.to_dict()
        assert d == {
            'lat_min': 50.0,
            'lat_max': 60.0,
            'lon_min': -130.0,
            'lon_max': -120.0
        }

    def test_normalize_longitude_to_0_360(self):
        """Test longitude normalization to 0-360 range."""
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)
        normalized = bbox.normalize_longitude('0-360')
        assert normalized.lon_min == 230.0
        assert normalized.lon_max == 240.0
        # Latitudes should be unchanged
        assert normalized.lat_min == 50.0
        assert normalized.lat_max == 60.0

    def test_normalize_longitude_to_minus180_180(self):
        """Test longitude normalization to -180 to 180 range."""
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=230.0, lon_max=240.0)
        normalized = bbox.normalize_longitude('-180-180')
        assert normalized.lon_min == -130.0
        assert normalized.lon_max == -120.0

    def test_normalize_longitude_invalid_range(self):
        """Test that invalid target range raises ValueError."""
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)
        with pytest.raises(ValueError, match="Unknown target_range"):
            bbox.normalize_longitude('invalid')

    def test_crosses_meridian_no_crossing(self):
        """Test meridian crossing detection for bbox that doesn't cross."""
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)
        assert not bbox.crosses_meridian('0-360')
        assert not bbox.crosses_meridian('-180-180')

    def test_crosses_meridian_with_crossing(self):
        """Test meridian crossing detection for bbox that crosses antimeridian."""
        # Bbox that crosses antimeridian (e.g., from 170°E to -170°E = 170° to 190°)
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=170.0, lon_max=-170.0)
        assert bbox.crosses_meridian('-180-180')

    def test_get_sorted_coords(self):
        """Test getting sorted coordinates."""
        # Create with reversed coordinates
        bbox = BoundingBox(lat_min=60.0, lat_max=50.0, lon_min=-120.0, lon_max=-130.0)
        lat_min, lat_max, lon_min, lon_max = bbox.get_sorted_coords()
        assert lat_min <= lat_max
        assert lon_min <= lon_max

    def test_str_representation(self):
        """Test string representation."""
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)
        s = str(bbox)
        assert 'BBox' in s
        assert '50.0' in s
        assert '60.0' in s

    def test_repr_representation(self):
        """Test repr representation."""
        bbox = BoundingBox(lat_min=50.0, lat_max=60.0, lon_min=-130.0, lon_max=-120.0)
        r = repr(bbox)
        assert 'BoundingBox' in r
        assert 'lat_min=50.0' in r
        assert 'lat_max=60.0' in r


class TestNormalizeLongitude:
    """Test normalize_longitude() function."""

    def test_normalize_to_0_360_from_negative(self):
        """Test normalizing negative longitude to 0-360."""
        assert normalize_longitude(-120.0, '0-360') == 240.0
        assert normalize_longitude(-180.0, '0-360') == 180.0

    def test_normalize_to_0_360_from_positive(self):
        """Test normalizing positive longitude to 0-360."""
        assert normalize_longitude(120.0, '0-360') == 120.0
        assert normalize_longitude(360.0, '0-360') == 0.0

    def test_normalize_to_minus180_180_from_0_360(self):
        """Test normalizing 0-360 longitude to -180 to 180."""
        assert normalize_longitude(240.0, '-180-180') == -120.0
        # 180.0 normalized to -180-180 becomes -180.0 (same meridian)
        assert normalize_longitude(180.0, '-180-180') == -180.0

    def test_normalize_to_minus180_180_from_negative(self):
        """Test normalizing already in -180-180 range."""
        assert normalize_longitude(-120.0, '-180-180') == -120.0
        assert normalize_longitude(120.0, '-180-180') == 120.0

    def test_normalize_array(self):
        """Test normalizing numpy array of longitudes."""
        lons = np.array([-120.0, -180.0, 0.0, 120.0, 180.0])
        normalized = normalize_longitude(lons, '0-360')
        expected = np.array([240.0, 180.0, 0.0, 120.0, 180.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_invalid_range(self):
        """Test that invalid target range raises ValueError."""
        with pytest.raises(ValueError, match="Unknown target_range"):
            normalize_longitude(120.0, 'invalid')


class TestCreateCoordinateMask:
    """Test create_coordinate_mask() function."""

    def test_simple_mask(self):
        """Test creating a simple coordinate mask."""
        lat = np.array([45.0, 50.0, 55.0, 60.0, 65.0])
        lon = np.array([-140.0, -130.0, -120.0, -110.0, -100.0])
        bbox = {'lat_min': 50.0, 'lat_max': 60.0, 'lon_min': -130.0, 'lon_max': -110.0}

        mask = create_coordinate_mask(lat, lon, bbox)

        # Points at 50, 55, 60 should be in range for latitude
        # Points at -130, -120, -110 should be in range for longitude
        # Only points 1, 2, 3 (indices) meet both conditions
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_mask_with_2d_arrays(self):
        """Test mask creation with 2D coordinate arrays."""
        lat = np.array([[50.0, 55.0], [60.0, 65.0]])
        lon = np.array([[-130.0, -120.0], [-110.0, -100.0]])
        bbox = {'lat_min': 50.0, 'lat_max': 60.0, 'lon_min': -130.0, 'lon_max': -110.0}

        mask = create_coordinate_mask(lat, lon, bbox)

        # Should match the shape of input arrays
        assert mask.shape == lat.shape
        assert mask.shape == lon.shape

    def test_mask_with_0_360_longitude_range(self):
        """Test mask creation with 0-360 longitude range."""
        lat = np.array([50.0, 55.0, 60.0])
        lon = np.array([230.0, 240.0, 250.0])  # 0-360 range (equivalent to -130, -120, -110)
        bbox = {'lat_min': 50.0, 'lat_max': 60.0, 'lon_min': 230.0, 'lon_max': 250.0}

        mask = create_coordinate_mask(lat, lon, bbox, lon_range='0-360')

        # All points should be within the bounding box
        assert np.all(mask)


class TestBackwardCompatibility:
    """Test backward compatibility with previous format."""

    def test_old_default_format_still_works(self):
        """Test that old default format can still be used explicitly."""
        # Old default was 'lat_max/lon_max/lat_min/lon_min'
        bbox = parse_bbox("60.0/-120.0/50.0/-130.0", format='lat_max/lon_max/lat_min/lon_min')
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == -130.0
        assert bbox['lon_max'] == -120.0


class TestRealWorldExamples:
    """Test with real-world bounding boxes."""

    def test_north_america_bbox(self):
        """Test typical North America bounding box."""
        # Pacific Northwest: N=60, W=-130, S=40, E=-110
        bbox = parse_bbox("60.0/-130.0/40.0/-110.0")
        assert bbox['lat_min'] == 40.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == -130.0
        assert bbox['lon_max'] == -110.0

    def test_europe_bbox(self):
        """Test typical Europe bounding box."""
        # Western Europe: N=55, W=-5, S=45, E=15
        bbox = parse_bbox("55.0/-5.0/45.0/15.0")
        assert bbox['lat_min'] == 45.0
        assert bbox['lat_max'] == 55.0
        assert bbox['lon_min'] == -5.0
        assert bbox['lon_max'] == 15.0

    def test_global_bbox(self):
        """Test global bounding box."""
        bbox = parse_bbox("90.0/-180.0/-90.0/180.0")
        assert bbox['lat_min'] == -90.0
        assert bbox['lat_max'] == 90.0
        assert bbox['lon_min'] == -180.0
        assert bbox['lon_max'] == 180.0


class TestCoordinateUtilsMixin:
    """Test CoordinateUtilsMixin."""

    class MockClass(CoordinateUtilsMixin):
        pass

    def test_mixin_parse_bbox(self):
        """Test mixin _parse_bbox method."""
        mixin = self.MockClass()
        bbox = mixin._parse_bbox("60.0/-130.0/50.0/-120.0")
        assert bbox['lat_min'] == 50.0
        assert bbox['lat_max'] == 60.0
        assert bbox['lon_min'] == -130.0
        assert bbox['lon_max'] == -120.0

    def test_mixin_normalize_longitude(self):
        """Test mixin _normalize_longitude method."""
        mixin = self.MockClass()
        assert mixin._normalize_longitude(-120.0, '0-360') == 240.0

    def test_mixin_validate_bbox(self):
        """Test mixin _validate_bbox method."""
        mixin = self.MockClass()
        bbox = {'lat_min': 50.0, 'lat_max': 60.0, 'lon_min': -130.0, 'lon_max': -120.0}
        assert mixin._validate_bbox(bbox) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
