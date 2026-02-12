"""
Unit Tests for SpatialSubsetMixin.

Tests the spatial subsetting functionality:
- get_coord_names(): detect lat/lon, latitude/longitude, y/x, rlat/rlon
- subset_xarray_bbox(): basic subset, buffer, descending latitude, lon wrap
- bbox_to_geojson(): GeoJSON polygon conversion
- bbox_to_cds_area(): CDS [N, W, S, E] format
- bbox_to_wcs_params(): WCS SUBSET parameters
"""

from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def spatial_mixin(standard_bbox):
    """Create an instance of SpatialSubsetMixin for testing."""
    from symfluence.data.acquisition.mixins.spatial import SpatialSubsetMixin

    class TestableSpatialMixin(SpatialSubsetMixin):
        def __init__(self, bbox):
            self.bbox = bbox
            self.logger = MagicMock()

    return TestableSpatialMixin(standard_bbox)


@pytest.fixture
def dataset_lat_lon():
    """Dataset with 'lat' and 'lon' coordinate names."""
    lat = np.arange(45.0, 48.1, 0.25)
    lon = np.arange(7.0, 10.1, 0.25)
    data = np.random.rand(len(lat), len(lon))

    return xr.Dataset(
        {"var": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon}
    )


@pytest.fixture
def dataset_latitude_longitude():
    """Dataset with 'latitude' and 'longitude' coordinate names (ERA5 style)."""
    lat = np.arange(48.0, 44.9, -0.25)  # Descending
    lon = np.arange(7.0, 10.1, 0.25)
    data = np.random.rand(len(lat), len(lon))

    return xr.Dataset(
        {"var": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon}
    )


@pytest.fixture
def dataset_y_x():
    """Dataset with 'y' and 'x' coordinate names."""
    y = np.arange(45.0, 48.1, 0.25)
    x = np.arange(7.0, 10.1, 0.25)
    data = np.random.rand(len(y), len(x))

    return xr.Dataset(
        {"var": (["y", "x"], data)},
        coords={"y": y, "x": x}
    )


@pytest.fixture
def dataset_lon_360():
    """Dataset with 0-360 longitude convention."""
    lat = np.arange(45.0, 48.1, 0.25)
    lon = np.arange(350.0, 365.0, 0.25)  # Wraps through 0
    lon = np.where(lon >= 360, lon - 360, lon)
    lon = np.sort(np.concatenate([lon[lon >= 350], lon[lon < 10]]))
    data = np.random.rand(len(lat), len(lon))

    return xr.Dataset(
        {"var": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon}
    )


# =============================================================================
# Get Coordinate Names Tests
# =============================================================================

@pytest.mark.mixin_spatial
@pytest.mark.acquisition
class TestGetCoordNames:
    """Tests for get_coord_names method."""

    def test_detect_lat_lon(self, spatial_mixin, dataset_lat_lon):
        """Should detect 'lat' and 'lon' coordinate names."""
        lat_name, lon_name = spatial_mixin.get_coord_names(dataset_lat_lon)

        assert lat_name == "lat"
        assert lon_name == "lon"

    def test_detect_latitude_longitude(self, spatial_mixin, dataset_latitude_longitude):
        """Should detect 'latitude' and 'longitude' coordinate names."""
        lat_name, lon_name = spatial_mixin.get_coord_names(dataset_latitude_longitude)

        assert lat_name == "latitude"
        assert lon_name == "longitude"

    def test_detect_y_x(self, spatial_mixin, dataset_y_x):
        """Should detect 'y' and 'x' coordinate names."""
        lat_name, lon_name = spatial_mixin.get_coord_names(dataset_y_x)

        assert lat_name == "y"
        assert lon_name == "x"

    def test_detect_rlat_rlon(self, spatial_mixin):
        """Should detect 'rlat' and 'rlon' coordinate names."""
        ds = xr.Dataset(
            {"var": (["rlat", "rlon"], np.random.rand(10, 10))},
            coords={"rlat": np.arange(10), "rlon": np.arange(10)}
        )

        lat_name, lon_name = spatial_mixin.get_coord_names(ds)

        assert lat_name == "rlat"
        assert lon_name == "rlon"

    def test_custom_candidates(self, spatial_mixin):
        """Should use custom candidate names."""
        ds = xr.Dataset(
            {"var": (["row", "col"], np.random.rand(10, 10))},
            coords={"row": np.arange(10), "col": np.arange(10)}
        )

        lat_name, lon_name = spatial_mixin.get_coord_names(
            ds,
            lat_candidates=("row",),
            lon_candidates=("col",)
        )

        assert lat_name == "row"
        assert lon_name == "col"

    def test_returns_none_when_not_found(self, spatial_mixin):
        """Should return None when coordinates not found."""
        ds = xr.Dataset(
            {"var": (["a", "b"], np.random.rand(10, 10))},
            coords={"a": np.arange(10), "b": np.arange(10)}
        )

        lat_name, lon_name = spatial_mixin.get_coord_names(ds)

        assert lat_name is None
        assert lon_name is None


# =============================================================================
# Subset xarray Bbox Tests
# =============================================================================

@pytest.mark.mixin_spatial
@pytest.mark.acquisition
class TestSubsetXarrayBbox:
    """Tests for subset_xarray_bbox method."""

    def test_basic_subset(self, spatial_mixin, dataset_lat_lon, standard_bbox):
        """Basic bbox subsetting should work."""
        subset = spatial_mixin.subset_xarray_bbox(dataset_lat_lon)

        # Check dimensions reduced
        assert subset.sizes['lat'] < dataset_lat_lon.sizes['lat']
        assert subset.sizes['lon'] < dataset_lat_lon.sizes['lon']

        # Check values within bbox
        assert subset['lat'].min() >= standard_bbox['lat_min']
        assert subset['lat'].max() <= standard_bbox['lat_max']
        assert subset['lon'].min() >= standard_bbox['lon_min']
        assert subset['lon'].max() <= standard_bbox['lon_max']

    def test_subset_with_buffer(self, spatial_mixin, dataset_lat_lon, standard_bbox):
        """Subsetting with buffer should expand the selection."""
        subset_no_buffer = spatial_mixin.subset_xarray_bbox(
            dataset_lat_lon, buffer=0.0
        )
        subset_with_buffer = spatial_mixin.subset_xarray_bbox(
            dataset_lat_lon, buffer=0.5
        )

        # Buffer should result in more data
        assert subset_with_buffer.sizes['lat'] >= subset_no_buffer.sizes['lat']
        assert subset_with_buffer.sizes['lon'] >= subset_no_buffer.sizes['lon']

    def test_subset_descending_latitude(self, spatial_mixin, dataset_latitude_longitude, standard_bbox):
        """Should handle descending latitude (ERA5 style)."""
        subset = spatial_mixin.subset_xarray_bbox(dataset_latitude_longitude)

        # Should still work with descending latitude
        assert subset.sizes['latitude'] > 0
        assert subset.sizes['longitude'] > 0

    def test_subset_with_explicit_bbox(self, spatial_mixin, dataset_lat_lon):
        """Should accept explicit bbox parameter."""
        custom_bbox = {
            "lat_min": 46.0,
            "lat_max": 46.5,
            "lon_min": 8.0,
            "lon_max": 8.5
        }

        subset = spatial_mixin.subset_xarray_bbox(dataset_lat_lon, bbox=custom_bbox)

        assert subset['lat'].min() >= 46.0
        assert subset['lat'].max() <= 46.5

    def test_subset_with_explicit_coord_names(self, spatial_mixin, dataset_lat_lon):
        """Should accept explicit coordinate names."""
        subset = spatial_mixin.subset_xarray_bbox(
            dataset_lat_lon,
            lat_name='lat',
            lon_name='lon'
        )

        assert subset.sizes['lat'] > 0

    def test_subset_auto_detects_coords(self, spatial_mixin, dataset_latitude_longitude):
        """Should auto-detect coordinate names."""
        subset = spatial_mixin.subset_xarray_bbox(dataset_latitude_longitude)

        # Should work without specifying coord names
        assert 'latitude' in subset.coords
        assert 'longitude' in subset.coords

    def test_subset_with_time_slice(self, spatial_mixin):
        """Should apply time slice if specified."""
        time = pd.date_range("2020-01-01", periods=100, freq="D")
        lat = np.arange(45.0, 48.1, 0.5)
        lon = np.arange(7.0, 10.1, 0.5)
        data = np.random.rand(len(time), len(lat), len(lon))

        ds = xr.Dataset(
            {"var": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": lat, "lon": lon}
        )

        time_slice = (pd.Timestamp("2020-01-15"), pd.Timestamp("2020-01-30"))
        subset = spatial_mixin.subset_xarray_bbox(ds, time_slice=time_slice)

        assert subset.sizes['time'] == 16  # 15 days inclusive

    def test_subset_raises_without_bbox(self):
        """Should raise if no bbox provided and none set on instance."""
        from symfluence.data.acquisition.mixins.spatial import SpatialSubsetMixin

        class NoBboxMixin(SpatialSubsetMixin):
            def __init__(self):
                self.bbox = None
                self.logger = MagicMock()

        mixin = NoBboxMixin()
        ds = xr.Dataset({"var": (["lat", "lon"], np.random.rand(10, 10))})

        with pytest.raises(ValueError):
            mixin.subset_xarray_bbox(ds)

    def test_subset_raises_without_coords(self, spatial_mixin):
        """Should raise if coordinates cannot be found."""
        ds = xr.Dataset({"var": (["a", "b"], np.random.rand(10, 10))})

        with pytest.raises(ValueError) as exc_info:
            spatial_mixin.subset_xarray_bbox(ds)

        assert "Could not find lat/lon" in str(exc_info.value)


# =============================================================================
# Subset NumPy Mask Tests
# =============================================================================

@pytest.mark.mixin_spatial
@pytest.mark.acquisition
class TestSubsetNumpyMask:
    """Tests for subset_numpy_mask method."""

    def test_basic_mask_subset(self, spatial_mixin, standard_bbox):
        """Basic mask subsetting should work."""
        # Create 2D lat/lon arrays (curvilinear grid style)
        y = np.arange(10)
        x = np.arange(15)
        yy, xx = np.meshgrid(y, x, indexing='ij')

        lat = 45.0 + yy * 0.25
        lon = 7.0 + xx * 0.25

        ds = xr.Dataset(
            {
                "var": (["y", "x"], np.random.rand(10, 15)),
                "lat": (["y", "x"], lat),
                "lon": (["y", "x"], lon),
            }
        )

        subset = spatial_mixin.subset_numpy_mask(
            ds, bbox=standard_bbox, lat_name='lat', lon_name='lon'
        )

        # Should have smaller dimensions
        assert subset.sizes['y'] <= 10
        assert subset.sizes['x'] <= 15

    def test_mask_with_buffer(self, spatial_mixin, standard_bbox):
        """Should apply cell buffer."""
        y = np.arange(20)
        x = np.arange(20)
        lat = 45.0 + np.repeat(y[:, np.newaxis], 20, axis=1) * 0.1
        lon = 7.0 + np.repeat(x[np.newaxis, :], 20, axis=0) * 0.1

        ds = xr.Dataset(
            {
                "var": (["y", "x"], np.random.rand(20, 20)),
                "lat": (["y", "x"], lat),
                "lon": (["y", "x"], lon),
            }
        )

        subset_no_buffer = spatial_mixin.subset_numpy_mask(
            ds, bbox=standard_bbox, lat_name='lat', lon_name='lon',
            grid_dims=('y', 'x'), buffer_cells=0
        )

        subset_with_buffer = spatial_mixin.subset_numpy_mask(
            ds, bbox=standard_bbox, lat_name='lat', lon_name='lon',
            grid_dims=('y', 'x'), buffer_cells=2
        )

        # Buffer should result in larger subset
        assert subset_with_buffer.sizes['y'] >= subset_no_buffer.sizes['y']

    def test_mask_empty_returns_original(self, spatial_mixin):
        """Should return original if no cells in bbox."""
        ds = xr.Dataset(
            {
                "var": (["y", "x"], np.random.rand(5, 5)),
                "lat": (["y", "x"], np.ones((5, 5)) * 10.0),  # Far from bbox
                "lon": (["y", "x"], np.ones((5, 5)) * 10.0),
            }
        )

        subset = spatial_mixin.subset_numpy_mask(
            ds, lat_name='lat', lon_name='lon'
        )

        # Should return original when no overlap
        assert subset.sizes['y'] == 5


# =============================================================================
# Bbox Conversion Tests
# =============================================================================

@pytest.mark.mixin_spatial
@pytest.mark.acquisition
class TestBboxConversions:
    """Tests for bbox format conversion methods."""

    def test_bbox_to_geojson(self, spatial_mixin, standard_bbox):
        """Should convert bbox to GeoJSON FeatureCollection."""
        geojson = spatial_mixin.bbox_to_geojson()

        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == 1

        feature = geojson['features'][0]
        assert feature['type'] == 'Feature'
        assert feature['geometry']['type'] == 'Polygon'

        coords = feature['geometry']['coordinates'][0]
        assert len(coords) == 5  # Closed polygon

        # Verify coordinates match bbox
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        assert min(lons) == standard_bbox['lon_min']
        assert max(lons) == standard_bbox['lon_max']
        assert min(lats) == standard_bbox['lat_min']
        assert max(lats) == standard_bbox['lat_max']

    def test_bbox_to_geojson_custom_bbox(self, spatial_mixin):
        """Should accept custom bbox parameter."""
        custom_bbox = {"lat_min": 10, "lat_max": 20, "lon_min": 30, "lon_max": 40}

        geojson = spatial_mixin.bbox_to_geojson(bbox=custom_bbox)

        coords = geojson['features'][0]['geometry']['coordinates'][0]
        lons = [c[0] for c in coords]
        assert min(lons) == 30
        assert max(lons) == 40

    def test_bbox_to_cds_area(self, spatial_mixin, standard_bbox):
        """Should convert to CDS [North, West, South, East] format."""
        cds_area = spatial_mixin.bbox_to_cds_area()

        assert len(cds_area) == 4
        assert cds_area[0] == standard_bbox['lat_max']  # North
        assert cds_area[1] == standard_bbox['lon_min']  # West
        assert cds_area[2] == standard_bbox['lat_min']  # South
        assert cds_area[3] == standard_bbox['lon_max']  # East

    def test_bbox_to_cds_area_custom_bbox(self, spatial_mixin):
        """Should accept custom bbox parameter for CDS format."""
        custom_bbox = {"lat_min": 10, "lat_max": 20, "lon_min": -10, "lon_max": 10}

        cds_area = spatial_mixin.bbox_to_cds_area(bbox=custom_bbox)

        assert cds_area == [20, -10, 10, 10]

    def test_bbox_to_wcs_params(self, spatial_mixin, standard_bbox):
        """Should convert to WCS SUBSET parameters."""
        wcs_params = spatial_mixin.bbox_to_wcs_params()

        assert len(wcs_params) == 4

        # Should have SUBSETTINGCRS, OUTPUTCRS, and two SUBSET params
        param_keys = [p[0] for p in wcs_params]
        assert 'SUBSETTINGCRS' in param_keys
        assert 'OUTPUTCRS' in param_keys
        assert param_keys.count('SUBSET') == 2

        # Verify SUBSET values
        subset_params = [p[1] for p in wcs_params if p[0] == 'SUBSET']
        lat_subset = [s for s in subset_params if 'Lat' in s][0]
        lon_subset = [s for s in subset_params if 'Lon' in s][0]

        assert f"Lat({standard_bbox['lat_min']},{standard_bbox['lat_max']})" == lat_subset
        assert f"Lon({standard_bbox['lon_min']},{standard_bbox['lon_max']})" == lon_subset

    def test_bbox_to_wcs_params_custom_crs(self, spatial_mixin):
        """Should accept custom CRS for WCS params."""
        wcs_params = spatial_mixin.bbox_to_wcs_params(crs="EPSG/0/32632")

        crs_params = [p[1] for p in wcs_params if 'CRS' in p[0]]
        assert all("EPSG/0/32632" in p for p in crs_params)


# =============================================================================
# Longitude Handling Tests
# =============================================================================

@pytest.mark.mixin_spatial
@pytest.mark.acquisition
class TestLongitudeHandling:
    """Tests for longitude convention handling."""

    def test_convert_negative_to_360(self, standard_bbox):
        """Should convert -180-180 bbox to 0-360 when dataset uses 0-360."""
        from symfluence.data.acquisition.mixins.spatial import SpatialSubsetMixin

        # Bbox with negative longitude
        negative_bbox = {
            "lat_min": 46.0, "lat_max": 47.0,
            "lon_min": -5.0, "lon_max": 5.0
        }

        class TestMixin(SpatialSubsetMixin):
            def __init__(self):
                self.bbox = negative_bbox
                self.logger = MagicMock()

        mixin = TestMixin()

        # Dataset with 0-360 longitude
        lat = np.arange(45.0, 48.1, 0.5)
        lon = np.arange(350.0, 370.0, 0.5)
        lon = np.where(lon >= 360, lon - 360, lon)
        data = np.random.rand(len(lat), len(lon))

        ds = xr.Dataset(
            {"var": (["lat", "lon"], data)},
            coords={"lat": lat, "lon": lon}
        )

        # Should handle the conversion
        subset = mixin.subset_xarray_bbox(ds)
        assert subset.sizes['lon'] > 0

    def test_no_wrap_when_disabled(self, spatial_mixin, dataset_lat_lon):
        """Should not wrap longitude when handle_lon_wrap=False."""
        subset = spatial_mixin.subset_xarray_bbox(
            dataset_lat_lon,
            handle_lon_wrap=False
        )

        # Should still work for simple case
        assert subset.sizes['lon'] > 0


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.mixin_spatial
@pytest.mark.acquisition
class TestSpatialMixinEdgeCases:
    """Edge case tests for spatial subset mixin."""

    def test_inverted_bbox_values(self, spatial_mixin, dataset_lat_lon):
        """Should handle bbox with min > max values."""
        # Inverted bbox (min > max)
        inverted_bbox = {
            "lat_min": 47.0,
            "lat_max": 46.0,  # Inverted
            "lon_min": 9.0,
            "lon_max": 8.0,   # Inverted
        }

        # Should still work (method should normalize)
        subset = spatial_mixin.subset_xarray_bbox(
            dataset_lat_lon,
            bbox=inverted_bbox
        )

        assert subset.sizes['lat'] > 0
        assert subset.sizes['lon'] > 0

    def test_very_small_bbox(self, spatial_mixin, dataset_lat_lon):
        """Should handle very small bbox (single cell)."""
        small_bbox = {
            "lat_min": 46.0,
            "lat_max": 46.1,
            "lon_min": 8.0,
            "lon_max": 8.1,
        }

        subset = spatial_mixin.subset_xarray_bbox(
            dataset_lat_lon,
            bbox=small_bbox
        )

        # Should get at least some data (nearest neighbor)
        assert subset.sizes['lat'] >= 1
        assert subset.sizes['lon'] >= 1

    def test_geojson_polygon_is_closed(self, spatial_mixin):
        """GeoJSON polygon should be properly closed."""
        geojson = spatial_mixin.bbox_to_geojson()

        coords = geojson['features'][0]['geometry']['coordinates'][0]

        # First and last coordinate should be the same
        assert coords[0] == coords[-1]
