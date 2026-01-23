"""Unit tests for WMFire Ignition and Perimeter Validation classes."""
import numpy as np
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from symfluence.models.wmfire.ignition import (
    IgnitionPoint,
    IgnitionManager,
    FirePerimeterValidator,
)


class TestIgnitionPoint:
    """Tests for IgnitionPoint dataclass."""

    def test_basic_creation(self):
        """Test basic IgnitionPoint creation."""
        ign = IgnitionPoint(
            latitude=51.2096,
            longitude=-115.7539,
            name="test_ignition"
        )

        assert ign.latitude == 51.2096
        assert ign.longitude == -115.7539
        assert ign.name == "test_ignition"
        assert ign.date is None
        assert ign.source == "config"

    def test_with_date(self):
        """Test IgnitionPoint with date."""
        date = datetime(2014, 7, 15)
        ign = IgnitionPoint(
            latitude=51.2,
            longitude=-115.8,
            name="fire_2014",
            date=date
        )

        assert ign.date == date

    def test_to_dict(self):
        """Test to_dict method."""
        date = datetime(2014, 7, 15)
        ign = IgnitionPoint(
            latitude=51.2,
            longitude=-115.8,
            name="fire",
            date=date,
            source="shapefile"
        )

        d = ign.to_dict()
        assert d['latitude'] == 51.2
        assert d['longitude'] == -115.8
        assert d['name'] == "fire"
        assert d['date'] == "2014-07-15T00:00:00"
        assert d['source'] == "shapefile"


class TestIgnitionManager:
    """Tests for IgnitionManager class."""

    @pytest.fixture
    def mock_config_with_coords(self):
        """Config with ignition coordinates."""
        config = MagicMock()
        wmfire = MagicMock()
        wmfire.ignition_shapefile = None
        wmfire.ignition_point = "51.2096/-115.7539"
        wmfire.ignition_name = "test_ignition"
        wmfire.ignition_date = "2014-07-15"
        config.model.rhessys.wmfire = wmfire
        return config

    @pytest.fixture
    def mock_config_no_ignition(self):
        """Config without ignition."""
        config = MagicMock()
        wmfire = MagicMock()
        wmfire.ignition_shapefile = None
        wmfire.ignition_point = None
        config.model.rhessys.wmfire = wmfire
        return config

    def test_init(self, mock_config_with_coords):
        """Test IgnitionManager initialization."""
        mgr = IgnitionManager(mock_config_with_coords)
        assert mgr._wmfire_config is not None

    def test_parse_ignition_coords(self, mock_config_with_coords):
        """Test parsing ignition coordinates."""
        mgr = IgnitionManager(mock_config_with_coords)

        ign = mgr.parse_ignition_coords(
            "51.2096/-115.7539",
            name="test",
            date_str="2014-07-15"
        )

        assert ign.latitude == 51.2096
        assert ign.longitude == -115.7539
        assert ign.name == "test"
        assert ign.date == datetime(2014, 7, 15)
        assert ign.source == "config"

    def test_get_ignition_point_from_coords(self, mock_config_with_coords):
        """Test getting ignition from coordinates."""
        mgr = IgnitionManager(mock_config_with_coords)
        ign = mgr.get_ignition_point()

        assert ign is not None
        assert ign.latitude == 51.2096
        assert ign.longitude == -115.7539
        assert ign.name == "test_ignition"

    def test_get_ignition_point_none(self, mock_config_no_ignition):
        """Test getting ignition when none specified."""
        mgr = IgnitionManager(mock_config_no_ignition)
        ign = mgr.get_ignition_point()
        assert ign is None

    def test_write_ignition_shapefile(self, mock_config_with_coords, tmp_path):
        """Test writing ignition shapefile."""
        mgr = IgnitionManager(mock_config_with_coords)

        ign = IgnitionPoint(
            latitude=51.2096,
            longitude=-115.7539,
            name="test_fire",
            date=datetime(2014, 7, 15)
        )

        output_path = mgr.write_ignition_shapefile(ign, tmp_path)

        assert output_path is not None
        assert output_path.exists()

        # Verify content
        import geopandas as gpd
        gdf = gpd.read_file(output_path)
        assert len(gdf) == 1
        assert gdf.iloc[0]['name'] == "test_fire"
        assert gdf.iloc[0]['latitude'] == 51.2096

    def test_convert_to_grid_indices(self, mock_config_with_coords):
        """Test converting ignition to grid indices."""
        mgr = IgnitionManager(mock_config_with_coords)

        ign = IgnitionPoint(latitude=51.0, longitude=-116.0)

        # Create transform for UTM zone 11N
        # Origin at (500000, 5650000), 30m resolution
        transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 5650000.0)

        row, col = mgr.convert_to_grid_indices(
            ign,
            transform,
            'EPSG:32611',
            nrows=100,
            ncols=100
        )

        # Should return valid indices or -1 if out of bounds
        assert isinstance(row, int)
        assert isinstance(col, int)


class TestFirePerimeterValidator:
    """Tests for FirePerimeterValidator class."""

    @pytest.fixture
    def sample_perimeters(self, tmp_path):
        """Create sample perimeter shapefiles."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        # Create observed perimeter
        observed = Polygon([
            (500000, 5600000),
            (501000, 5600000),
            (501000, 5601000),
            (500000, 5601000),
        ])
        obs_gdf = gpd.GeoDataFrame({
            'name': ['observed'],
            'geometry': [observed]
        }, crs='EPSG:32611')

        obs_path = tmp_path / 'observed.shp'
        obs_gdf.to_file(obs_path)

        # Create simulated perimeter (overlapping)
        simulated = Polygon([
            (500500, 5600000),
            (501500, 5600000),
            (501500, 5601000),
            (500500, 5601000),
        ])
        sim_gdf = gpd.GeoDataFrame({
            'name': ['simulated'],
            'geometry': [simulated]
        }, crs='EPSG:32611')

        sim_path = tmp_path / 'simulated.shp'
        sim_gdf.to_file(sim_path)

        return obs_path, sim_path, obs_gdf, sim_gdf

    def test_load_perimeters_file(self, sample_perimeters):
        """Test loading perimeter from file."""
        obs_path, _, _, _ = sample_perimeters

        validator = FirePerimeterValidator()
        gdf = validator.load_perimeters(obs_path)

        assert gdf is not None
        assert len(gdf) == 1
        assert gdf.iloc[0]['name'] == 'observed'

    def test_load_perimeters_directory(self, sample_perimeters, tmp_path):
        """Test loading perimeters from directory."""
        validator = FirePerimeterValidator()
        gdf = validator.load_perimeters(tmp_path)

        assert gdf is not None
        assert len(gdf) == 2  # Both observed and simulated

    def test_compare_perimeters(self, sample_perimeters):
        """Test perimeter comparison metrics."""
        _, _, obs_gdf, sim_gdf = sample_perimeters

        validator = FirePerimeterValidator()
        metrics = validator.compare_perimeters(sim_gdf, obs_gdf)

        # Should have all expected metrics
        assert 'iou' in metrics
        assert 'dice' in metrics
        assert 'simulated_area_ha' in metrics
        assert 'observed_area_ha' in metrics
        assert 'commission_rate' in metrics
        assert 'omission_rate' in metrics

        # IoU should be between 0 and 1
        assert 0 <= metrics['iou'] <= 1
        assert 0 <= metrics['dice'] <= 1

        # Areas should be positive
        assert metrics['simulated_area_ha'] > 0
        assert metrics['observed_area_ha'] > 0

    def test_compare_perimeters_perfect_overlap(self):
        """Test comparison with perfect overlap."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        perim = Polygon([
            (0, 0), (100, 0), (100, 100), (0, 100)
        ])
        gdf = gpd.GeoDataFrame({
            'name': ['test'],
            'geometry': [perim]
        }, crs='EPSG:32611')

        validator = FirePerimeterValidator()
        metrics = validator.compare_perimeters(gdf, gdf)

        assert metrics['iou'] == pytest.approx(1.0, rel=0.01)
        assert metrics['dice'] == pytest.approx(1.0, rel=0.01)
        assert metrics['commission_rate'] == pytest.approx(0.0, abs=0.01)
        assert metrics['omission_rate'] == pytest.approx(0.0, abs=0.01)

    def test_compare_perimeters_no_overlap(self):
        """Test comparison with no overlap."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        perim1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        perim2 = Polygon([(200, 200), (300, 200), (300, 300), (200, 300)])

        gdf1 = gpd.GeoDataFrame({'geometry': [perim1]}, crs='EPSG:32611')
        gdf2 = gpd.GeoDataFrame({'geometry': [perim2]}, crs='EPSG:32611')

        validator = FirePerimeterValidator()
        metrics = validator.compare_perimeters(gdf1, gdf2)

        assert metrics['iou'] == pytest.approx(0.0, abs=0.01)
        assert metrics['intersection_area_ha'] == pytest.approx(0.0, abs=0.01)

    def test_create_comparison_map(self, sample_perimeters, tmp_path):
        """Test creating comparison map."""
        _, _, obs_gdf, sim_gdf = sample_perimeters

        validator = FirePerimeterValidator()
        output_path = tmp_path / 'comparison.png'

        result = validator.create_comparison_map(
            sim_gdf,
            obs_gdf,
            output_path,
            title="Test Comparison"
        )

        assert result is not None
        assert output_path.exists()


class TestIgnitionManagerIntegration:
    """Integration tests for IgnitionManager."""

    @pytest.fixture
    def real_ignition_shapefile(self):
        """Path to real ignition shapefile if available."""
        path = Path('/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/'
                   'domain_Bow_at_Banff_elevation/shapefiles/ignitions/Ignition_A.shp')
        if path.exists():
            return path
        pytest.skip("Real ignition shapefile not available")

    def test_load_real_ignition(self, real_ignition_shapefile):
        """Test loading real ignition shapefile."""
        config = MagicMock()
        wmfire = MagicMock()
        wmfire.ignition_shapefile = str(real_ignition_shapefile)
        wmfire.ignition_point = None
        config.model.rhessys.wmfire = wmfire

        mgr = IgnitionManager(config)
        ign = mgr.get_ignition_point()

        assert ign is not None
        assert ign.name == "Ignition_A"
        assert ign.source == "shapefile"
        # Check coordinates are reasonable for Bow watershed
        assert 50 < ign.latitude < 52
        assert -117 < ign.longitude < -115
