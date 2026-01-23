"""
Integration tests for Hub'Eau (France) observation handlers.

Tests the French government hydrological data API integration:
- Station search functionality
- Streamflow data acquisition
- Data processing pipeline

IMPORTANT: The Hub'Eau API may be geo-restricted to French IP addresses.
Tests will be skipped if the API is not accessible.

These tests require internet access to the Hub'Eau API.
"""
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Skip all tests if requests not available
requests = pytest.importorskip("requests")


def hubeau_api_accessible():
    """Check if Hub'Eau API is accessible (not geo-restricted)."""
    try:
        from symfluence.data.observation.handlers.hubeau import (
            _hubeau_request, HUBEAU_STATIONS_URL, HubEauAPIError
        )
        _hubeau_request(HUBEAU_STATIONS_URL, {'size': 1})
        return True
    except (HubEauAPIError, Exception):
        return False


# Skip marker for tests requiring Hub'Eau API access
requires_hubeau_api = pytest.mark.skipif(
    not hubeau_api_accessible(),
    reason="Hub'Eau API not accessible (may be geo-restricted to French IPs)"
)


@requires_hubeau_api
class TestHubEauStationSearch:
    """Test Hub'Eau station search functionality."""

    def test_search_stations_by_river(self):
        """Search for stations on the Seine river."""
        from symfluence.data.observation.handlers.hubeau import search_hubeau_stations

        stations = search_hubeau_stations(river_name="Seine", limit=5)

        assert not stations.empty, "Should find stations on Seine river"
        assert 'code_station' in stations.columns
        assert 'libelle_station' in stations.columns
        assert len(stations) <= 5

    def test_search_stations_by_bbox(self):
        """Search for stations in Paris region bounding box."""
        from symfluence.data.observation.handlers.hubeau import search_hubeau_stations

        # Paris region bbox (lon_min, lat_min, lon_max, lat_max)
        bbox = (2.0, 48.5, 3.0, 49.0)
        stations = search_hubeau_stations(bbox=bbox, limit=10)

        assert not stations.empty, "Should find stations in Paris region"

    def test_search_stations_by_department(self):
        """Search for stations in Paris department (75)."""
        from symfluence.data.observation.handlers.hubeau import search_hubeau_stations

        stations = search_hubeau_stations(department="75", limit=5)

        # Paris has limited hydrometric stations, may be empty
        # Just verify the function runs without error
        assert isinstance(stations, pd.DataFrame)

    def test_get_station_info(self):
        """Get detailed info for a known station."""
        from symfluence.data.observation.handlers.hubeau import get_station_info

        # Seine at Paris (Austerlitz) - a well-known station
        station_id = "H5920010"
        info = get_station_info(station_id)

        assert info is not None
        assert info.get('code_station') == station_id
        assert 'libelle_station' in info
        assert 'libelle_cours_eau' in info

    def test_get_station_info_invalid(self):
        """Test error handling for invalid station ID."""
        from symfluence.data.observation.handlers.hubeau import get_station_info

        with pytest.raises(ValueError, match="Station not found"):
            get_station_info("INVALID_STATION_999")


@requires_hubeau_api
class TestHubEauStreamflowHandler:
    """Test Hub'Eau streamflow handler (requires API access)."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        temp_dir = tempfile.mkdtemp(prefix="hubeau_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_project_dir):
        """Create mock config for testing."""
        return {
            'SYMFLUENCE_DATA_DIR': str(temp_project_dir),
            'DOMAIN_NAME': 'test_seine',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-03-31 23:00',
            'BOUNDING_BOX_COORDS': '49.0/2.0/48.5/3.0',
            'evaluation': {
                'streamflow': {
                    'station_id': 'H5920010'  # Seine at Paris
                },
                'hubeau': {
                    'download': True,
                    'use_daily': True
                }
            },
            'data': {
                'download_hubeau_data': True
            },
            'forcing': {
                'time_step_size': 86400  # Daily
            }
        }

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        import logging
        logger = logging.getLogger("test_hubeau")
        logger.setLevel(logging.DEBUG)
        return logger

    def test_handler_initialization(self, mock_config, mock_logger):
        """Test handler can be initialized."""
        from symfluence.data.observation.handlers.hubeau import HubEauStreamflowHandler

        handler = HubEauStreamflowHandler(mock_config, mock_logger)

        assert handler.obs_type == "streamflow"
        assert handler.source_name == "Hub'Eau_Hydrometrie"

    def test_handler_acquire_downloads_data(self, mock_config, mock_logger, temp_project_dir):
        """Test handler can acquire data from Hub'Eau API."""
        from symfluence.data.observation.handlers.hubeau import HubEauStreamflowHandler

        mock_config['SYMFLUENCE_DATA_DIR'] = str(temp_project_dir)

        handler = HubEauStreamflowHandler(mock_config, mock_logger)
        raw_path = handler.acquire()

        assert raw_path.exists(), f"Raw file should exist: {raw_path}"
        assert raw_path.suffix == '.json'

        # Verify JSON content
        import json
        with open(raw_path) as f:
            data = json.load(f)

        assert 'data' in data
        assert 'source' in data
        assert len(data['data']) > 0, "Should have downloaded some records"

    def test_handler_process_converts_units(self, mock_config, mock_logger, temp_project_dir):
        """Test handler processes data and converts units correctly."""
        from symfluence.data.observation.handlers.hubeau import HubEauStreamflowHandler

        mock_config['SYMFLUENCE_DATA_DIR'] = str(temp_project_dir)

        handler = HubEauStreamflowHandler(mock_config, mock_logger)

        # First acquire data
        raw_path = handler.acquire()

        # Then process it
        processed_path = handler.process(raw_path)

        assert processed_path.exists(), f"Processed file should exist: {processed_path}"

        # Load and verify processed data
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

        assert not df.empty, "Processed DataFrame should not be empty"
        assert 'discharge_cms' in df.columns or df.columns[0] == 'discharge_cms' or len(df.columns) == 1

        # Values should be in m³/s (reasonable range for Seine at Paris)
        # Seine discharge typically 50-2500 m³/s
        values = df.values.flatten() if len(df.columns) == 1 else df['discharge_cms'].values
        valid_values = values[~pd.isna(values)]

        if len(valid_values) > 0:
            assert valid_values.min() >= 0, "Discharge should be non-negative"
            assert valid_values.max() < 10000, "Discharge should be reasonable (< 10000 m³/s)"

    def test_handler_creates_metadata(self, mock_config, mock_logger, temp_project_dir):
        """Test handler creates metadata JSON alongside processed data."""
        from symfluence.data.observation.handlers.hubeau import HubEauStreamflowHandler

        mock_config['SYMFLUENCE_DATA_DIR'] = str(temp_project_dir)

        handler = HubEauStreamflowHandler(mock_config, mock_logger)
        raw_path = handler.acquire()
        processed_path = handler.process(raw_path)

        # Check for metadata file
        meta_path = processed_path.with_suffix('.json')
        assert meta_path.exists(), f"Metadata file should exist: {meta_path}"

        import json
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta.get('source') == "Hub'Eau_Hydrometrie"
        assert meta.get('variable') == 'streamflow'
        assert meta.get('units') == 'm3/s'


@requires_hubeau_api
class TestHubEauWaterLevelHandler:
    """Test Hub'Eau water level handler (requires API access)."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        temp_dir = tempfile.mkdtemp(prefix="hubeau_wl_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_project_dir):
        """Create mock config for testing."""
        return {
            'SYMFLUENCE_DATA_DIR': str(temp_project_dir),
            'DOMAIN_NAME': 'test_wl',
            'EXPERIMENT_TIME_START': '2020-06-01 00:00',
            'EXPERIMENT_TIME_END': '2020-06-30 23:00',
            'BOUNDING_BOX_COORDS': '49.0/2.0/48.5/3.0',
            'evaluation': {
                'waterlevel': {
                    'station_id': 'H5920010'
                },
                'hubeau': {
                    'download': True
                }
            }
        }

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        import logging
        logger = logging.getLogger("test_hubeau_wl")
        logger.setLevel(logging.DEBUG)
        return logger

    def test_waterlevel_handler_initialization(self, mock_config, mock_logger):
        """Test water level handler can be initialized."""
        from symfluence.data.observation.handlers.hubeau import HubEauWaterLevelHandler

        handler = HubEauWaterLevelHandler(mock_config, mock_logger)

        assert handler.obs_type == "waterlevel"
        assert handler.source_name == "Hub'Eau_Hydrometrie"


class TestHubEauRegistration:
    """Test Hub'Eau handlers are properly registered."""

    def test_streamflow_handler_registered(self):
        """Test streamflow handler is in registry."""
        from symfluence.data.observation.registry import ObservationRegistry
        from symfluence.data.observation.handlers import HubEauStreamflowHandler  # Trigger registration

        assert ObservationRegistry.is_registered('hubeau_streamflow')

    def test_waterlevel_handler_registered(self):
        """Test water level handler is in registry."""
        from symfluence.data.observation.registry import ObservationRegistry
        from symfluence.data.observation.handlers import HubEauWaterLevelHandler  # Trigger registration

        assert ObservationRegistry.is_registered('hubeau_waterlevel')

    def test_get_handler_by_name(self):
        """Test handler can be retrieved by name."""
        from symfluence.data.observation.registry import ObservationRegistry
        from symfluence.data.observation.handlers import HubEauStreamflowHandler
        import logging
        import tempfile

        logger = logging.getLogger("test")

        mock_config = {
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test_exp',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'SYMFLUENCE_DATA_DIR': tempfile.gettempdir(),
            'SYMFLUENCE_CODE_DIR': tempfile.gettempdir(),
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'lumped',
            'FORCING_DATASET': 'ERA5',
            'HYDROLOGICAL_MODEL': 'SUMMA',
        }

        handler = ObservationRegistry.get_handler('hubeau_streamflow', mock_config, logger)

        assert handler is not None
        assert isinstance(handler, HubEauStreamflowHandler)


@pytest.mark.slow
@requires_hubeau_api
class TestHubEauLiveAPI:
    """Live API tests - marked slow and requires API access, skipped by default."""

    def test_fetch_one_year_data(self):
        """Fetch one year of data from Seine at Paris."""
        from symfluence.data.observation.handlers.hubeau import HubEauStreamflowHandler
        import logging
        import tempfile

        logger = logging.getLogger("live_test")
        temp_dir = tempfile.mkdtemp()

        config = {
            'SYMFLUENCE_DATA_DIR': temp_dir,
            'SYMFLUENCE_CODE_DIR': temp_dir,
            'DOMAIN_NAME': 'seine_paris',
            'EXPERIMENT_ID': 'seine_test',
            'EXPERIMENT_TIME_START': '2022-01-01 00:00',
            'EXPERIMENT_TIME_END': '2022-12-31 23:00',
            'BOUNDING_BOX_COORDS': '49.0/2.0/48.5/3.0',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'lumped',
            'FORCING_DATASET': 'ERA5',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'evaluation': {
                'streamflow': {'station_id': 'H5920010'},
                'hubeau': {'download': True, 'use_daily': True}
            },
            'forcing': {'time_step_size': 86400}
        }

        try:
            handler = HubEauStreamflowHandler(config, logger)
            raw_path = handler.acquire()
            processed_path = handler.process(raw_path)

            df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

            # Should have ~365 daily values
            assert len(df) > 300, f"Should have most of year's data, got {len(df)}"
            assert len(df) <= 366, "Should not exceed days in year"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
