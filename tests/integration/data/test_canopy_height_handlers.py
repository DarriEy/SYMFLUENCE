"""
Integration tests for Tree Canopy Height acquisition and observation handlers.

Tests cover:
- GEDI L2A canopy height (NASA AppEEARS)
- Meta/WRI global canopy height (10m resolution)
- GLAD/UMD tree height (Landsat-based)
"""
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box
from symfluence.data.acquisition.registry import AcquisitionRegistry
from symfluence.data.data_manager import DataManager
from symfluence.data.observation.registry import ObservationRegistry

pytestmark = [pytest.mark.integration, pytest.mark.data]


@pytest.fixture
def canopy_config(tmp_path):
    """Base configuration for canopy height tests."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'canopy_test',
        'EXPERIMENT_ID': 'canopy_e2e',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_TIME_STEP_SIZE': 3600,
        'DOWNLOAD_USGS_DATA': False,
        'BOUNDING_BOX_COORDS': '40.5/-105.5/40.0/-105.0',
        'POUR_POINT_COORDS': '40.25/-105.25',
    }


@pytest.fixture
def mock_canopy_raster(tmp_path):
    """Create a mock canopy height GeoTIFF for testing."""
    height, width = 100, 100
    # Canopy height values 0-30m
    data = np.random.rand(height, width).astype(np.float32) * 30

    transform = from_bounds(-105.5, 40.0, -105.0, 40.5, width, height)

    raster_path = tmp_path / "mock_canopy.tif"
    with rasterio.open(
        raster_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(data, 1)

    return raster_path


@pytest.fixture
def mock_catchment_shapefile(tmp_path):
    """Create a mock catchment shapefile for basin statistics."""
    catchment_dir = tmp_path / "domain_canopy_test" / "shapefiles" / "catchment"
    catchment_dir.mkdir(parents=True, exist_ok=True)

    catchment_shp = catchment_dir / "canopy_test_catchment.shp"
    gdf = gpd.GeoDataFrame({
        'ID': [1],
        'geometry': [box(-105.4, 40.1, -105.1, 40.4)]
    }, crs='EPSG:4326')
    gdf.to_file(catchment_shp)

    return catchment_shp


# =============================================================================
# Registry Tests
# =============================================================================

class TestCanopyHeightRegistration:
    """Test that all canopy height handlers are properly registered."""

    def test_acquisition_handlers_registered(self):
        """Verify acquisition handlers are registered with correct names."""
        # GEDI handlers
        assert AcquisitionRegistry.is_registered('GEDI_CANOPY_HEIGHT')
        assert AcquisitionRegistry.is_registered('GEDI_L2A')

        # Meta/WRI handlers
        assert AcquisitionRegistry.is_registered('META_CANOPY_HEIGHT')
        assert AcquisitionRegistry.is_registered('WRI_CANOPY_HEIGHT')
        assert AcquisitionRegistry.is_registered('META_WRI_CANOPY')

        # GLAD handlers
        assert AcquisitionRegistry.is_registered('GLAD_TREE_HEIGHT')
        assert AcquisitionRegistry.is_registered('UMD_TREE_HEIGHT')
        assert AcquisitionRegistry.is_registered('GLAD_CANOPY')

    def test_observation_handlers_registered(self):
        """Verify observation handlers are registered with correct names."""
        # Unified handler
        assert ObservationRegistry.is_registered('canopy_height')
        assert ObservationRegistry.is_registered('tree_height')
        assert ObservationRegistry.is_registered('vegetation_height')

        # Source-specific handlers
        assert ObservationRegistry.is_registered('gedi_canopy_height')
        assert ObservationRegistry.is_registered('meta_canopy_height')
        assert ObservationRegistry.is_registered('wri_canopy_height')
        assert ObservationRegistry.is_registered('glad_tree_height')
        assert ObservationRegistry.is_registered('umd_tree_height')

    def test_case_insensitive_lookup(self):
        """Verify registry handles case-insensitive lookups."""
        # Test various case combinations
        assert AcquisitionRegistry.is_registered('gedi_canopy_height')
        assert AcquisitionRegistry.is_registered('GEDI_CANOPY_HEIGHT')
        assert AcquisitionRegistry.is_registered('Gedi_Canopy_Height')


# =============================================================================
# Acquisition Handler Tests (Mocked)
# =============================================================================

class TestGEDIAcquisitionMocked:
    """Test GEDI canopy height acquisition with mocked responses."""

    def test_gedi_handler_instantiation(self, canopy_config):
        """Test GEDI handler can be instantiated."""
        logger = logging.getLogger("test_gedi")

        handler = AcquisitionRegistry.get_handler(
            'GEDI_CANOPY_HEIGHT',
            canopy_config,
            logger
        )

        assert handler is not None
        assert hasattr(handler, 'download')
        assert hasattr(handler, 'bbox')

    @patch('requests.post')
    @patch('requests.get')
    def test_gedi_acquisition_workflow(self, mock_get, mock_post, canopy_config, tmp_path):
        """Test GEDI acquisition workflow with mocked AppEEARS API."""
        logger = logging.getLogger("test_gedi_workflow")

        # Mock login response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'token': 'test_token'}
        mock_post.return_value.raise_for_status = MagicMock()

        # Mock task submission and status
        task_response = MagicMock()
        task_response.status_code = 200
        task_response.json.return_value = {'task_id': 'test_task_123'}
        task_response.raise_for_status = MagicMock()

        status_response = MagicMock()
        status_response.status_code = 200
        status_response.json.return_value = {'status': 'done'}
        status_response.raise_for_status = MagicMock()

        bundle_response = MagicMock()
        bundle_response.status_code = 200
        bundle_response.json.return_value = {'files': []}
        bundle_response.raise_for_status = MagicMock()

        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: {'token': 'test_token'}),
            MagicMock(status_code=200, json=lambda: {'task_id': 'test_123'}),
            MagicMock(status_code=200),  # logout
        ]

        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {'status': 'done'}),
            MagicMock(status_code=200, json=lambda: {'files': []}),
        ]

        # Set credentials
        canopy_config['EARTHDATA_USERNAME'] = 'test_user'
        canopy_config['EARTHDATA_PASSWORD'] = 'test_pass'

        handler = AcquisitionRegistry.get_handler(
            'GEDI_CANOPY_HEIGHT',
            canopy_config,
            logger
        )

        # The handler will fail at consolidation since no files were downloaded,
        # but we've verified the API workflow is correct
        with pytest.raises(FileNotFoundError):
            handler.download(tmp_path / "output")


class TestMetaCanopyAcquisitionMocked:
    """Test Meta/WRI canopy height acquisition with mocked responses."""

    def test_meta_handler_instantiation(self, canopy_config):
        """Test Meta/WRI handler can be instantiated."""
        logger = logging.getLogger("test_meta")

        handler = AcquisitionRegistry.get_handler(
            'META_CANOPY_HEIGHT',
            canopy_config,
            logger
        )

        assert handler is not None
        assert hasattr(handler, 'download')

    @patch('symfluence.data.acquisition.handlers.canopy_height.create_robust_session')
    def test_meta_tile_download_workflow(self, mock_session, canopy_config, tmp_path, mock_canopy_raster):
        """Test Meta tile download with mocked session."""
        logger = logging.getLogger("test_meta_tiles")

        # Mock session that returns tile data
        mock_sess = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404  # Simulate no tiles available
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_sess.get.return_value = mock_response
        mock_session.return_value = mock_sess

        handler = AcquisitionRegistry.get_handler(
            'META_CANOPY_HEIGHT',
            canopy_config,
            logger
        )

        # Should raise FileNotFoundError since no tiles found (all 404s)
        with pytest.raises(FileNotFoundError):
            handler.download(tmp_path / "output")


class TestGLADAcquisitionMocked:
    """Test GLAD/UMD tree height acquisition with mocked responses."""

    def test_glad_handler_instantiation(self, canopy_config):
        """Test GLAD handler can be instantiated."""
        logger = logging.getLogger("test_glad")

        handler = AcquisitionRegistry.get_handler(
            'GLAD_TREE_HEIGHT',
            canopy_config,
            logger
        )

        assert handler is not None
        assert hasattr(handler, 'download')

    def test_glad_tile_naming(self, canopy_config):
        """Test GLAD tile naming convention."""
        logger = logging.getLogger("test_glad_naming")

        handler = AcquisitionRegistry.get_handler(
            'GLAD_TREE_HEIGHT',
            canopy_config,
            logger
        )

        # Verify snap_to_grid method
        assert handler._snap_to_grid(45.5, 10, 'floor') == 40
        assert handler._snap_to_grid(45.5, 10, 'ceil') == 50
        assert handler._snap_to_grid(-45.5, 10, 'floor') == -50
        assert handler._snap_to_grid(-45.5, 10, 'ceil') == -40


# =============================================================================
# Observation Handler Tests (Mocked)
# =============================================================================

class TestCanopyHeightObservationMocked:
    """Test canopy height observation handler with mocked data."""

    def test_observation_handler_instantiation(self, canopy_config):
        """Test observation handler can be instantiated."""
        logger = logging.getLogger("test_obs")

        handler = ObservationRegistry.get_handler(
            'canopy_height',
            canopy_config,
            logger
        )

        assert handler is not None
        assert handler.obs_type == "canopy_height"

    def test_process_canopy_raster(
        self, canopy_config, mock_canopy_raster, mock_catchment_shapefile, tmp_path
    ):
        """Test processing of canopy height raster to basin statistics."""
        logger = logging.getLogger("test_process")

        # Setup directory structure
        project_dir = tmp_path / "domain_canopy_test"
        canopy_dir = project_dir / "observations" / "vegetation" / "canopy_height" / "meta_wri"
        canopy_dir.mkdir(parents=True, exist_ok=True)

        # Copy mock raster to expected location
        import shutil
        output_raster = canopy_dir / "canopy_test_meta_canopy_height.tif"
        shutil.copy(mock_canopy_raster, output_raster)

        handler = ObservationRegistry.get_handler(
            'canopy_height',
            canopy_config,
            logger
        )

        # Process the data
        result_path = handler.process(project_dir / "observations" / "vegetation" / "canopy_height")

        assert result_path.exists()

        # Load and verify results
        df = pd.read_csv(result_path)
        assert 'source' in df.columns
        assert 'statistic' in df.columns
        assert 'value' in df.columns

        # Verify statistics are present
        stats = df[df['source'] == 'meta_wri']['statistic'].tolist()
        assert 'mean' in stats
        assert 'max' in stats
        assert 'std' in stats

    def test_basin_statistics_calculation(
        self, canopy_config, mock_canopy_raster, mock_catchment_shapefile, tmp_path
    ):
        """Test basin statistics are calculated correctly."""
        logger = logging.getLogger("test_stats")

        handler = ObservationRegistry.get_handler(
            'canopy_height',
            canopy_config,
            logger
        )

        # Load catchment for masking
        basin_gdf = gpd.read_file(mock_catchment_shapefile)

        # Extract statistics
        stats = handler._extract_basin_statistics(mock_canopy_raster, basin_gdf)

        assert stats is not None
        assert 'mean' in stats
        assert 'max' in stats
        assert 'min' in stats
        assert 'std' in stats
        assert 'p90' in stats
        assert 'coverage_fraction' in stats

        # Verify reasonable ranges
        assert 0 <= stats['mean'] <= 30
        assert 0 <= stats['max'] <= 30
        assert 0 <= stats['coverage_fraction'] <= 1


# =============================================================================
# Full E2E Tests (Mocked)
# =============================================================================

class TestCanopyHeightE2EMocked:
    """End-to-end tests with mocked data."""

    def test_full_acquisition_and_processing(
        self, canopy_config, mock_canopy_raster, mock_catchment_shapefile, tmp_path
    ):
        """Test full workflow from acquisition to processed statistics."""
        logger = logging.getLogger("test_e2e")

        # Setup complete directory structure
        project_dir = Path(canopy_config['SYMFLUENCE_DATA_DIR']) / "domain_canopy_test"
        canopy_dir = project_dir / "observations" / "vegetation" / "canopy_height" / "meta_wri"
        canopy_dir.mkdir(parents=True, exist_ok=True)

        # Copy mock raster
        import shutil
        output_raster = canopy_dir / "canopy_test_meta_canopy_height.tif"
        shutil.copy(mock_canopy_raster, output_raster)

        # Update config
        canopy_config['CANOPY_HEIGHT_SOURCE'] = 'meta'
        canopy_config['ADDITIONAL_OBSERVATIONS'] = 'canopy_height'

        # Create and run observation handler
        handler = ObservationRegistry.get_handler(
            'canopy_height',
            canopy_config,
            logger
        )

        # Process (acquisition is skipped since file exists)
        input_dir = project_dir / "observations" / "vegetation" / "canopy_height"
        result_path = handler.process(input_dir)

        # Verify outputs
        assert result_path.exists()

        df = pd.read_csv(result_path)
        assert len(df) > 0

        # Check for summary file
        summary_file = (
            project_dir / "observations" / "vegetation" / "preprocessed"
            / "canopy_test_canopy_height_summary.csv"
        )
        assert summary_file.exists()


# =============================================================================
# Live Tests (require network, marked as slow)
# =============================================================================

@pytest.mark.slow
@pytest.mark.requires_cloud
class TestCanopyHeightLive:
    """Live tests that actually download data (slow, require network)."""

    @pytest.mark.skip(reason="GEDI requires NASA Earthdata credentials")
    def test_gedi_live_acquisition(self, tmp_path):
        """Live test for GEDI data acquisition."""
        import yaml

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
            'DOMAIN_NAME': 'gedi_live_test',
            'EXPERIMENT_ID': 'gedi_live',
            'EXPERIMENT_TIME_START': '2020-06-01 00:00',
            'EXPERIMENT_TIME_END': '2020-06-30 23:00',
            'BOUNDING_BOX_COORDS': '40.1/-105.3/40.0/-105.2',  # Very small area
            'POUR_POINT_COORDS': '40.05/-105.25',
            'GEDI_METRIC': 'rh98',
            # ... other required config
        }

        logger = logging.getLogger("test_gedi_live")
        handler = AcquisitionRegistry.get_handler(
            'GEDI_CANOPY_HEIGHT',
            config,
            logger
        )

        output = handler.download(tmp_path / "gedi_output")
        assert output.exists()

    @pytest.mark.skip(reason="Meta tiles may not be available in test region")
    def test_meta_live_acquisition(self, tmp_path):
        """Live test for Meta/WRI data acquisition."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
            'DOMAIN_NAME': 'meta_live_test',
            'EXPERIMENT_ID': 'meta_live',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'BOUNDING_BOX_COORDS': '40.1/-105.3/40.0/-105.2',
            'POUR_POINT_COORDS': '40.05/-105.25',
        }

        logger = logging.getLogger("test_meta_live")
        handler = AcquisitionRegistry.get_handler(
            'META_CANOPY_HEIGHT',
            config,
            logger
        )

        output = handler.download(tmp_path / "meta_output")
        assert output.exists()

    def test_glad_live_acquisition(self, tmp_path):
        """
        Live test for GLAD tree cover acquisition.
        GLAD is freely available without authentication.
        """
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
            'DOMAIN_NAME': 'glad_live_test',
            'EXPERIMENT_ID': 'glad_live',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            # Required config keys
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'lumped',
            'FORCING_DATASET': 'ERA5',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            # Use coordinates in a forested area (Pacific Northwest)
            'BOUNDING_BOX_COORDS': '47.1/-122.1/47.0/-122.0',
            'POUR_POINT_COORDS': '47.05/-122.05',
            'GLAD_VERSION': 'GFC-2020-v1.8',
            'GLAD_VARIABLE': 'treecover2000',
        }

        logger = logging.getLogger("test_glad_live")
        handler = AcquisitionRegistry.get_handler(
            'GLAD_TREE_HEIGHT',
            config,
            logger
        )

        try:
            output = handler.download(tmp_path / "glad_output")
            assert output.exists()

            # Verify it's a valid GeoTIFF
            with rasterio.open(output) as src:
                assert src.crs is not None
                data = src.read(1)
                # Tree cover should be 0-100%
                assert data.min() >= 0
                assert data.max() <= 100

        except FileNotFoundError:
            # Some tiles may not exist (ocean, no forest, etc.)
            pytest.skip("GLAD tile not available for test region")


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestCanopyHeightDataValidation:
    """Tests to validate output data quality."""

    def test_canopy_height_range_validation(self, mock_canopy_raster, canopy_config):
        """Test that canopy height values are in valid range."""
        logger = logging.getLogger("test_validation")

        with rasterio.open(mock_canopy_raster) as src:
            data = src.read(1)

        # Canopy height should be 0-100m (generous upper bound)
        valid_data = data[(data > -9999) & (data < 9999)]
        assert valid_data.min() >= 0, "Canopy height should not be negative"
        assert valid_data.max() <= 100, "Canopy height should not exceed 100m"

    def test_statistics_consistency(self, mock_canopy_raster, canopy_config, mock_catchment_shapefile):
        """Test that computed statistics are internally consistent."""
        logger = logging.getLogger("test_consistency")

        handler = ObservationRegistry.get_handler(
            'canopy_height',
            canopy_config,
            logger
        )

        basin_gdf = gpd.read_file(mock_catchment_shapefile)
        stats = handler._extract_basin_statistics(mock_canopy_raster, basin_gdf)

        # Statistical consistency checks
        assert stats['min'] <= stats['mean'] <= stats['max']
        assert stats['p25'] <= stats['median'] <= stats['p75']
        assert stats['median'] <= stats['p90'] <= stats['max']
        assert stats['std'] >= 0
