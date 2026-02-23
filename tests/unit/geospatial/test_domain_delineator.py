"""
Tests for DomainDelineator orchestrator class.

Tests domain delineation orchestration including:
- Method routing (point, lumped, semidistributed, distributed)
- Artifact tracking
- Configuration handling
- Pre-existing shapefile detection
"""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Import fixtures
from .conftest import requires_geopandas

# Skip all tests if geopandas not available
pytestmark = requires_geopandas


class TestDomainDelineatorInit:
    """Tests for DomainDelineator initialization."""

    def test_init_creates_all_delineators(self, mock_config_object, mock_logger):
        """Test that DomainDelineator creates all component delineators."""
        with patch('symfluence.geospatial.delineation.GeofabricDelineator'):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator'):
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter'):
                    with patch('symfluence.geospatial.delineation.PointDelineator'):
                        with patch('symfluence.geospatial.delineation.GridDelineator'):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)

                            assert delineator.delineator is not None
                            assert delineator.lumped_delineator is not None
                            assert delineator.subsetter is not None
                            assert delineator.point_delineator is not None
                            assert delineator.grid_delineator is not None


class TestDelineationArtifacts:
    """Tests for DelineationArtifacts dataclass."""

    def test_artifacts_default_values(self):
        """Test DelineationArtifacts has correct default values."""
        from symfluence.geospatial.delineation import DelineationArtifacts

        artifacts = DelineationArtifacts(method='lumped')

        assert artifacts.method == 'lumped'
        assert artifacts.river_basins_path is None
        assert artifacts.river_network_path is None
        assert artifacts.pour_point_path is None
        assert artifacts.original_basins_path is None
        assert artifacts.metadata == {}

    def test_artifacts_with_paths(self, tmp_path):
        """Test DelineationArtifacts stores paths correctly."""
        from symfluence.geospatial.delineation import DelineationArtifacts

        basins_path = tmp_path / "basins.shp"
        network_path = tmp_path / "network.shp"

        artifacts = DelineationArtifacts(
            method='semidistributed',
            river_basins_path=basins_path,
            river_network_path=network_path,
            metadata={'grid_cell_size': '1000.0'}
        )

        assert artifacts.method == 'semidistributed'
        assert artifacts.river_basins_path == basins_path
        assert artifacts.river_network_path == network_path
        assert artifacts.metadata['grid_cell_size'] == '1000.0'


class TestMethodRouting:
    """Tests for define_domain method routing."""

    def test_point_method_routing(self, mock_config_object, mock_logger):
        """Test that point method routes to PointDelineator."""
        mock_config_object.domain.definition_method = 'point'
        mock_config_object.paths.river_basins_name = 'default'

        mock_point_delineator = MagicMock()
        mock_point_delineator.create_point_domain_shapefile.return_value = Path('/test/output.shp')

        with patch('symfluence.geospatial.delineation.GeofabricDelineator'):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator'):
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter'):
                    with patch('symfluence.geospatial.delineation.PointDelineator', return_value=mock_point_delineator):
                        with patch('symfluence.geospatial.delineation.GridDelineator'):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)
                            result, artifacts = delineator.define_domain()

                            mock_point_delineator.create_point_domain_shapefile.assert_called_once()
                            assert artifacts.method == 'point'

    def test_lumped_method_routing(self, mock_config_object, mock_logger, tmp_path):
        """Test that lumped method routes to LumpedWatershedDelineator."""
        mock_config_object.domain.definition_method = 'lumped'
        mock_config_object.domain.subset_from_geofabric = False
        mock_config_object.paths.river_basins_name = 'default'

        network_path = tmp_path / "network.shp"
        basins_path = tmp_path / "basins.shp"

        mock_lumped_delineator = MagicMock()
        mock_lumped_delineator.delineate_lumped_watershed.return_value = (network_path, basins_path)

        with patch('symfluence.geospatial.delineation.GeofabricDelineator'):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator', return_value=mock_lumped_delineator):
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter'):
                    with patch('symfluence.geospatial.delineation.PointDelineator'):
                        with patch('symfluence.geospatial.delineation.GridDelineator'):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)
                            result, artifacts = delineator.define_domain()

                            mock_lumped_delineator.delineate_lumped_watershed.assert_called_once()
                            assert artifacts.method == 'lumped'
                            assert artifacts.river_basins_path == basins_path
                            assert artifacts.river_network_path == network_path

    def test_distributed_method_routing(self, mock_config_object, mock_logger, tmp_path):
        """Test that distributed method routes to GridDelineator."""
        mock_config_object.domain.definition_method = 'distributed'
        mock_config_object.domain.subset_from_geofabric = False
        mock_config_object.domain.grid_source = 'generate'
        mock_config_object.paths.river_basins_name = 'default'

        network_path = tmp_path / "network.shp"
        basins_path = tmp_path / "basins.shp"

        mock_grid_delineator = MagicMock()
        mock_grid_delineator.create_grid_domain.return_value = (network_path, basins_path)

        with patch('symfluence.geospatial.delineation.GeofabricDelineator'):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator'):
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter'):
                    with patch('symfluence.geospatial.delineation.PointDelineator'):
                        with patch('symfluence.geospatial.delineation.GridDelineator', return_value=mock_grid_delineator):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)
                            result, artifacts = delineator.define_domain()

                            mock_grid_delineator.create_grid_domain.assert_called_once()
                            assert artifacts.method == 'distributed'


class TestPreExistingShapefile:
    """Tests for pre-existing shapefile detection."""

    def test_skip_delineation_when_shapefile_provided(self, mock_config_object, mock_logger):
        """Test that delineation is skipped when shapefile already exists."""
        mock_config_object.domain.definition_method = 'lumped'
        mock_config_object.paths.river_basins_name = 'existing_basins.shp'  # Not 'default'

        with patch('symfluence.geospatial.delineation.GeofabricDelineator'):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator') as mock_lumped:
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter'):
                    with patch('symfluence.geospatial.delineation.PointDelineator'):
                        with patch('symfluence.geospatial.delineation.GridDelineator'):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)
                            result, artifacts = delineator.define_domain()

                            # Delineation should not be called
                            mock_lumped.return_value.delineate_lumped_watershed.assert_not_called()
                            assert result is None


class TestLumpedToDistributedRouting:
    """Tests for lumped-to-distributed routing workflow."""

    def test_lumped_with_river_network_routing(self, mock_config_object, mock_logger, tmp_path):
        """Test lumped domain with distributed routing creates subcatchments."""
        mock_config_object.domain.definition_method = 'lumped'
        mock_config_object.domain.subset_from_geofabric = False
        mock_config_object.domain.delineation.routing = 'river_network'
        mock_config_object.paths.river_basins_name = 'default'

        network_path = tmp_path / "network.shp"
        basins_path = tmp_path / "basins.shp"
        delineated_network = tmp_path / "delineated_network.shp"
        delineated_basins = tmp_path / "delineated_basins.shp"

        mock_lumped_delineator = MagicMock()
        mock_lumped_delineator.delineate_lumped_watershed.return_value = (network_path, basins_path)

        mock_geofabric_delineator = MagicMock()
        mock_geofabric_delineator.delineate_geofabric.return_value = (delineated_network, delineated_basins)

        with patch('symfluence.geospatial.delineation.GeofabricDelineator', return_value=mock_geofabric_delineator):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator', return_value=mock_lumped_delineator):
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter'):
                    with patch('symfluence.geospatial.delineation.PointDelineator'):
                        with patch('symfluence.geospatial.delineation.GridDelineator'):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)
                            result, artifacts = delineator.define_domain()

                            # Both lumped and distributed delineation should be called
                            mock_lumped_delineator.delineate_lumped_watershed.assert_called_once()
                            mock_geofabric_delineator.delineate_geofabric.assert_called_once()

                            # Metadata should include delineated paths
                            assert 'delineated_river_network_path' in artifacts.metadata
                            assert 'delineated_river_basins_path' in artifacts.metadata


class TestSubsetFromGeofabric:
    """Tests for geofabric subsetting workflows."""

    def test_lumped_subset_dissolves_basins(self, mock_config_object, mock_logger, tmp_path):
        """Test that lumped + subset dissolves basins to single polygon."""
        mock_config_object.domain.definition_method = 'lumped'
        mock_config_object.domain.subset_from_geofabric = True
        mock_config_object.domain.delineation.geofabric_type = 'merit_basins'
        mock_config_object.paths.river_basins_name = 'default'
        mock_config_object.paths.project_dir = tmp_path

        # Create mock subset data
        from fixtures.geospatial_fixtures import synthetic_river_network_gdf, synthetic_watershed_gdf
        basins = synthetic_watershed_gdf(n_basins=5)
        rivers = synthetic_river_network_gdf(basins)

        mock_subsetter = MagicMock()
        mock_subsetter.subset_geofabric.return_value = (basins, rivers)
        mock_subsetter.aggregate_to_lumped.return_value = basins.dissolve().reset_index(drop=True)

        with patch('symfluence.geospatial.delineation.GeofabricDelineator'):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator'):
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter', return_value=mock_subsetter):
                    with patch('symfluence.geospatial.delineation.PointDelineator'):
                        with patch('symfluence.geospatial.delineation.GridDelineator'):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)
                            # Set project_dir for path resolution
                            delineator._project_dir = tmp_path

                            result, artifacts = delineator.define_domain()

                            mock_subsetter.subset_geofabric.assert_called_once()
                            mock_subsetter.aggregate_to_lumped.assert_called_once()
                            assert artifacts.method == 'lumped'


class TestArtifactMetadata:
    """Tests for artifact metadata population."""

    def test_distributed_metadata_includes_grid_config(self, mock_config_object, mock_logger, tmp_path):
        """Test that distributed method includes grid configuration in metadata."""
        mock_config_object.domain.definition_method = 'distributed'
        mock_config_object.domain.subset_from_geofabric = False
        mock_config_object.domain.grid_source = 'generate'
        mock_config_object.domain.grid_cell_size = 1000.0
        mock_config_object.domain.clip_grid_to_watershed = True
        mock_config_object.paths.river_basins_name = 'default'

        network_path = tmp_path / "network.shp"
        basins_path = tmp_path / "basins.shp"

        mock_grid_delineator = MagicMock()
        mock_grid_delineator.create_grid_domain.return_value = (network_path, basins_path)

        with patch('symfluence.geospatial.delineation.GeofabricDelineator'):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator'):
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter'):
                    with patch('symfluence.geospatial.delineation.PointDelineator'):
                        with patch('symfluence.geospatial.delineation.GridDelineator', return_value=mock_grid_delineator):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)
                            result, artifacts = delineator.define_domain()

                            assert 'grid_cell_size' in artifacts.metadata
                            assert 'clip_to_watershed' in artifacts.metadata
                            assert 'grid_source' in artifacts.metadata


class TestUnknownMethod:
    """Tests for unknown delineation method handling."""

    def test_unknown_method_returns_none(self, mock_config_object, mock_logger):
        """Test that unknown method returns None with logged error."""
        mock_config_object.domain.definition_method = 'unknown_method'
        mock_config_object.paths.river_basins_name = 'default'

        with patch('symfluence.geospatial.delineation.GeofabricDelineator'):
            with patch('symfluence.geospatial.delineation.LumpedWatershedDelineator'):
                with patch('symfluence.geospatial.delineation.GeofabricSubsetter'):
                    with patch('symfluence.geospatial.delineation.PointDelineator'):
                        with patch('symfluence.geospatial.delineation.GridDelineator'):
                            from symfluence.geospatial.delineation import DomainDelineator

                            delineator = DomainDelineator(mock_config_object, mock_logger)
                            result, artifacts = delineator.define_domain()

                            assert result is None
                            # Error should be logged
                            mock_logger.error.assert_called()
