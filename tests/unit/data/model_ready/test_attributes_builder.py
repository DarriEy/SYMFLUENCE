"""Tests for AttributesNetCDFBuilder."""

from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip('netCDF4')
gpd = pytest.importorskip('geopandas')

from shapely.geometry import box
from symfluence.data.model_ready.attributes_builder import AttributesNetCDFBuilder


def _create_catchment_shp(path: Path, n_hru: int = 3) -> None:
    """Create a minimal catchment shapefile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({
        'HRU_ID': [f'hru_{i}' for i in range(n_hru)],
        'HRU_area': np.random.uniform(1e6, 5e6, n_hru),
        'geometry': [box(i, 50, i + 1, 51) for i in range(n_hru)],
    }, crs='EPSG:4326')
    gdf.to_file(path)


def _create_dem_intersection_shp(path: Path, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({
        'elev_mean': np.random.uniform(500, 2000, n),
        'geometry': [box(i, 50, i + 1, 51) for i in range(n)],
    }, crs='EPSG:4326')
    gdf.to_file(path)


def _create_soil_intersection_shp(path: Path, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {f'USGS_{i}': np.random.uniform(0, 1, n) for i in range(5)}
    data['soil_pixel_count'] = np.random.randint(100, 1000, n)
    data['geometry'] = [box(i, 50, i + 1, 51) for i in range(n)]
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    gdf.to_file(path)


def _create_land_intersection_shp(path: Path, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {f'IGBP_{i}': np.random.uniform(0, 1, n) for i in range(5)}
    data['land_pixel_count'] = np.random.randint(100, 1000, n)
    data['geometry'] = [box(i, 50, i + 1, 51) for i in range(n)]
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    gdf.to_file(path)


class TestAttributesNetCDFBuilder:
    """Tests for attributes store construction."""

    def test_build_from_catchment(self, tmp_path):
        _create_catchment_shp(
            tmp_path / 'shapefiles' / 'catchment' / 'test_HRUs_lumped.shp'
        )

        builder = AttributesNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()
        assert result is not None
        assert result.exists()

        ds = netCDF4.Dataset(str(result), 'r')
        assert 'hru_identity' in ds.groups
        assert ds.groups['hru_identity'].dimensions['hru'].size == 3
        ds.close()

    def test_build_with_terrain(self, tmp_path):
        _create_catchment_shp(
            tmp_path / 'shapefiles' / 'catchment' / 'test_HRUs_lumped.shp'
        )
        _create_dem_intersection_shp(
            tmp_path / 'shapefiles' / 'catchment_intersection'
            / 'with_dem' / 'catchment_with_dem.shp'
        )

        builder = AttributesNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()

        ds = netCDF4.Dataset(str(result), 'r')
        assert 'terrain' in ds.groups
        v = ds.groups['terrain'].variables['elev_mean']
        assert v.standard_name == 'surface_altitude'
        ds.close()

    def test_build_with_soil_and_land(self, tmp_path):
        _create_catchment_shp(
            tmp_path / 'shapefiles' / 'catchment' / 'test_HRUs_lumped.shp'
        )
        isect = tmp_path / 'shapefiles' / 'catchment_intersection'
        _create_soil_intersection_shp(isect / 'with_soilgrids' / 'catchment_with_soilclass.shp')
        _create_land_intersection_shp(isect / 'with_landclass' / 'catchment_with_landclass.shp')

        builder = AttributesNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()

        ds = netCDF4.Dataset(str(result), 'r')
        assert 'soil' in ds.groups
        assert 'landcover' in ds.groups
        assert ds.groups['soil'].dimensions['soil_class'].size == 5
        assert ds.groups['landcover'].dimensions['land_class'].size == 5
        ds.close()

    def test_build_no_data(self, tmp_path):
        builder = AttributesNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()
        assert result is None

    def test_global_attrs(self, tmp_path):
        _create_catchment_shp(
            tmp_path / 'shapefiles' / 'catchment' / 'test_HRUs_lumped.shp'
        )
        builder = AttributesNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()

        ds = netCDF4.Dataset(str(result), 'r')
        assert ds.Conventions == 'CF-1.8'
        assert ds.domain_name == 'test'
        ds.close()

    def test_graceful_skip_missing_groups(self, tmp_path):
        """Only groups with available data are written."""
        _create_catchment_shp(
            tmp_path / 'shapefiles' / 'catchment' / 'test_HRUs_lumped.shp'
        )
        builder = AttributesNetCDFBuilder(
            project_dir=tmp_path, domain_name='test',
        )
        result = builder.build()

        ds = netCDF4.Dataset(str(result), 'r')
        assert 'hru_identity' in ds.groups
        # These should NOT be present since source data doesn't exist
        assert 'terrain' not in ds.groups
        assert 'soil' not in ds.groups
        ds.close()
