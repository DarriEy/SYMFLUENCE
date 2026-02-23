"""
Unit tests for the GeospatialUtilsMixin.

Tests centroid calculation with various CRS configurations,
geometric shapes, and edge cases.
"""

from unittest.mock import Mock

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon
from symfluence.geospatial.geometry_utils import GeospatialUtilsMixin


class MockClass(GeospatialUtilsMixin):
    """Mock class to test GeospatialUtilsMixin functionality."""

    def __init__(self):
        self.logger = Mock()


class TestGeospatialUtilsMixinCentroid:
    """Test suite for calculate_catchment_centroid method."""

    def test_centroid_simple_square_wgs84(self):
        """Test centroid calculation for a simple square in WGS84."""
        # Create a 1-degree square centered at (0, 0)
        coords = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should be centered at (0, 0)
        assert abs(lon - 0.0) < 0.01
        assert abs(lat - 0.0) < 0.01

    def test_centroid_rectangle_northern_hemisphere(self):
        """Test centroid for a rectangle in northern hemisphere."""
        # Rectangle from 45°N to 46°N, 10°W to 11°W
        coords = [(-11, 45), (-10, 45), (-10, 46), (-11, 46), (-11, 45)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should be approximately at center
        assert abs(lon - (-10.5)) < 0.1
        assert abs(lat - 45.5) < 0.1

    def test_centroid_rectangle_southern_hemisphere(self):
        """Test centroid for a rectangle in southern hemisphere."""
        # Rectangle from 45°S to 46°S, 10°E to 11°E
        coords = [(10, -46), (11, -46), (11, -45), (10, -45), (10, -46)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should be approximately at center
        assert abs(lon - 10.5) < 0.1
        assert abs(lat - (-45.5)) < 0.1

    def test_centroid_with_no_crs(self):
        """Test centroid calculation when CRS is not defined."""
        # Create polygon without CRS
        coords = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]})

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should assume EPSG:4326 and calculate centroid
        assert abs(lon - 0.0) < 0.01
        assert abs(lat - 0.0) < 0.01

        # Check that logger warning was called
        mixin.logger.warning.assert_called_once()

    def test_centroid_utm_zone_calculation_europe(self):
        """Test UTM zone calculation for European catchment."""
        # Catchment in central Europe (around 50°N, 10°E)
        coords = [(9.5, 49.5), (10.5, 49.5), (10.5, 50.5), (9.5, 50.5), (9.5, 49.5)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should be approximately at center
        assert abs(lon - 10.0) < 0.1
        assert abs(lat - 50.0) < 0.1

    def test_centroid_utm_zone_calculation_north_america(self):
        """Test UTM zone calculation for North American catchment."""
        # Catchment in western North America (around 45°N, 110°W)
        coords = [(-111, 44), (-109, 44), (-109, 46), (-111, 46), (-111, 44)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should be approximately at center
        assert abs(lon - (-110.0)) < 0.1
        assert abs(lat - 45.0) < 0.1

    def test_centroid_irregular_polygon(self):
        """Test centroid for an irregular polygon."""
        # Create an irregular shape
        coords = [(0, 0), (2, 0), (2, 1), (1, 2), (0, 1), (0, 0)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Centroid should be inside the polygon
        assert 0 < lon < 2
        assert 0 < lat < 2

    def test_centroid_projected_crs_input(self):
        """Test centroid calculation when input is already projected."""
        # Create polygon in UTM Zone 33N (EPSG:32633)
        # Approximately corresponds to central Europe
        coords = [(500000, 5000000), (600000, 5000000), (600000, 5100000), (500000, 5100000), (500000, 5000000)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:32633')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should return geographic coordinates (not projected)
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90

    def test_centroid_multipolygon_geodataframe(self):
        """Test centroid calculation for GeoDataFrame with multiple rows."""
        # Create two polygons
        poly1 = Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)])
        poly2 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)])
        gdf = gpd.GeoDataFrame({'geometry': [poly1, poly2]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should calculate centroid of entire geometry
        # (not necessarily the center of both polygons)
        assert -1 < lon < 2
        assert -1 < lat < 2

    def test_centroid_small_catchment(self):
        """Test centroid for a small catchment (few km²)."""
        # Small catchment: 0.01 degree square (~1 km at equator)
        coords = [(10.0, 45.0), (10.01, 45.0), (10.01, 45.01), (10.0, 45.01), (10.0, 45.0)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should be at approximate center
        assert abs(lon - 10.005) < 0.001
        assert abs(lat - 45.005) < 0.001

    def test_centroid_large_catchment(self):
        """Test centroid for a large catchment (many km²)."""
        # Large catchment: 5 degree square
        coords = [(10, 45), (15, 45), (15, 50), (10, 50), (10, 45)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should be approximately at center
        assert abs(lon - 12.5) < 0.5
        assert abs(lat - 47.5) < 0.5

    def test_centroid_near_antimeridian(self):
        """Test centroid calculation near the international date line."""
        # Catchment crossing 180° meridian (179°E to -179°W)
        coords = [(179, 0), (-179, 0), (-179, 1), (179, 1), (179, 0)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should handle antimeridian correctly
        assert -180 <= lon <= 180
        assert abs(lat - 0.5) < 0.1

    def test_centroid_near_poles(self):
        """Test centroid calculation near polar regions."""
        # Catchment in high northern latitude
        coords = [(10, 80), (11, 80), (11, 81), (10, 81), (10, 80)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Should handle high latitudes
        assert abs(lon - 10.5) < 0.5
        assert 80 < lat < 81

    def test_centroid_logging(self):
        """Test that centroid calculation logs information."""
        coords = [(10, 45), (11, 45), (11, 46), (10, 46), (10, 45)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        # Check that logger.info was called with centroid information
        assert mixin.logger.info.called
        call_args = str(mixin.logger.info.call_args)
        assert 'centroid' in call_args.lower()

    def test_calculate_feature_centroids_multiple(self):
        """Test calculate_feature_centroids with multiple features."""
        # Create two squares
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly2 = Polygon([(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)])
        gdf = gpd.GeoDataFrame({'geometry': [poly1, poly2]}, crs='EPSG:4326')

        mixin = MockClass()
        centroids = mixin.calculate_feature_centroids(gdf)

        assert len(centroids) == 2
        assert isinstance(centroids, gpd.GeoSeries)

        # Centroid 1 should be near (0.5, 0.5)
        assert abs(centroids.iloc[0].x - 0.5) < 0.05
        assert abs(centroids.iloc[0].y - 0.5) < 0.05

        # Centroid 2 should be near (10.5, 10.5)
        assert abs(centroids.iloc[1].x - 10.5) < 0.05
        assert abs(centroids.iloc[1].y - 10.5) < 0.05

    def test_calculate_feature_centroids_empty(self):
        """Test calculate_feature_centroids with empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        mixin = MockClass()
        centroids = mixin.calculate_feature_centroids(gdf)
        assert len(centroids) == 0


class TestGeospatialUtilsMixinAreaCalculation:
    """Test suite for calculate_catchment_area_km2 method."""

    def test_area_simple_square_at_equator(self):
        """Test area calculation for a 1-degree square at equator."""
        # 1-degree square at equator (~111 km x 111 km)
        coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        area_km2 = mixin.calculate_catchment_area_km2(gdf)

        # 1 degree at equator ≈ 111 km, so area ≈ 12,321 km²
        # Allow some tolerance for projection differences
        assert 11000 < area_km2 < 13000

    def test_area_small_catchment(self):
        """Test area for a small catchment."""
        # 0.01 degree square (~1 km x 1 km at equator)
        coords = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01), (0, 0)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        area_km2 = mixin.calculate_catchment_area_km2(gdf)

        # Should be approximately 1 km²
        assert 0.5 < area_km2 < 2.0

    def test_area_returns_positive_value(self):
        """Test that area calculation always returns positive value."""
        coords = [(10, 45), (11, 45), (11, 46), (10, 46), (10, 45)]
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        mixin = MockClass()
        area_km2 = mixin.calculate_catchment_area_km2(gdf)

        assert area_km2 > 0

    def test_area_larger_polygon_larger_area(self):
        """Test that larger polygons have larger areas."""
        # Small polygon
        small_coords = [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5), (0, 0)]
        small_poly = Polygon(small_coords)
        small_gdf = gpd.GeoDataFrame({'geometry': [small_poly]}, crs='EPSG:4326')

        # Large polygon
        large_coords = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
        large_poly = Polygon(large_coords)
        large_gdf = gpd.GeoDataFrame({'geometry': [large_poly]}, crs='EPSG:4326')

        mixin = MockClass()
        small_area = mixin.calculate_catchment_area_km2(small_gdf)
        large_area = mixin.calculate_catchment_area_km2(large_gdf)

        # Large polygon should have larger area
        assert large_area > small_area
        # Should be roughly 16x larger (2² vs 0.5²)
        assert 10 < (large_area / small_area) < 20


class TestGeospatialUtilsMixinEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_geodataframe(self):
        """Test behavior with empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')

        mixin = MockClass()

        # Should raise an error or handle gracefully
        with pytest.raises((IndexError, ValueError)):
            mixin.calculate_catchment_centroid(gdf)

    def test_point_geometry(self):
        """Test with point geometry instead of polygon."""
        point = Point(10, 45)
        gdf = gpd.GeoDataFrame({'geometry': [point]}, crs='EPSG:4326')

        mixin = MockClass()

        # Should handle point geometry (centroid of point is itself)
        lon, lat = mixin.calculate_catchment_centroid(gdf)

        assert abs(lon - 10) < 0.01
        assert abs(lat - 45) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
