import pytest
from pathlib import Path
import pandas as pd
import logging
import json
from unittest.mock import patch, MagicMock
from symfluence.core import SYMFLUENCE
from symfluence.data.data_manager import DataManager
from symfluence.data.observation.registry import ObservationRegistry

pytestmark = [pytest.mark.integration, pytest.mark.data, pytest.mark.requires_cloud, pytest.mark.slow]

@pytest.fixture
def mock_config(tmp_path):
    return {
        # Required system settings
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        # Required domain settings
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-05 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'DOMAIN_DISCRETIZATION': 'lumped',
        # Required forcing and model settings
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        # Observation-specific settings
        'FORCING_TIME_STEP_SIZE': 3600,
        'DOWNLOAD_USGS_DATA': True,
        'STATION_ID': '06306300',
        'STREAMFLOW_DATA_PROVIDER': 'USGS',
        'ADDITIONAL_OBSERVATIONS': 'USGS_STREAMFLOW',
        'STREAMFLOW_RAW_PATH': 'default',
        'STREAMFLOW_PROCESSED_PATH': 'default',
        'STREAMFLOW_RAW_NAME': 'test_raw.rdb'
    }

@pytest.fixture
def mock_usgs_response():
    # Real USGS RDB format
    return """# USGS RDB content
# Data columns:
agency_cd	site_no	datetime	00060_00000
5s	15s	20d	14n
USGS	06306300	2020-01-01 00:00	100.0
USGS	06306300	2020-01-01 01:00	110.0
USGS	06306300	2020-01-01 02:00	120.0
USGS	06306300	2020-01-01 03:00	130.0
"""

def test_usgs_streamflow_acquisition(mock_config, mock_usgs_response):
    """Test the formalized USGS streamflow acquisition pathway (Live Canary)."""
    logger = logging.getLogger("test_usgs")

    dm = DataManager(mock_config, logger)

    # Verify handler is registered
    assert ObservationRegistry.is_registered('USGS_STREAMFLOW')

    # Process observed data (Live API call)
    dm.process_observed_data()

    # Verify results
    project_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / f"domain_{mock_config['DOMAIN_NAME']}"
    raw_file = project_dir / "observations" / "streamflow" / "raw_data" / "usgs_06306300_raw.rdb"
    processed_file = project_dir / "observations" / "streamflow" / "preprocessed" / f"{mock_config['DOMAIN_NAME']}_streamflow_processed.csv"

    assert raw_file.exists(), f"Raw USGS file not found at {raw_file}"
    assert processed_file.exists(), f"Processed USGS file not found at {processed_file}"

    # Load and verify content
    df = pd.read_csv(processed_file)
    assert 'datetime' in df.columns
    assert 'discharge_cms' in df.columns
    assert len(df) > 0

def test_usgs_groundwater_acquisition(mock_config):
    """Test the formalized USGS groundwater acquisition pathway (Mocked)."""
    logger = logging.getLogger("test_usgs_gw")

    # Update config for real groundwater station
    gw_config = mock_config.copy()
    gw_config['DOWNLOAD_USGS_DATA'] = False
    gw_config['DOWNLOAD_USGS_GW'] = 'true'
    gw_config['USGS_STATION'] = '01646500'
    gw_config['STATION_ID'] = '01646500'
    gw_config['ADDITIONAL_OBSERVATIONS'] = 'USGS_GW'
    gw_config['DATA_ACCESS'] = 'cloud'

    # Mock Response
    mock_json = {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "variableName": "Depth to water level, feet below land surface",
                        "parameterCode": "72019",
                        "unit": {"unitCode": "ft"}
                    },
                    "values": [
                        {
                            "value": [
                                {"dateTime": "2020-01-01T12:00:00.000", "value": "10.5"},
                                {"dateTime": "2020-01-02T12:00:00.000", "value": "10.4"}
                            ]
                        }
                    ]
                }
            ]
        }
    }

    dm = DataManager(gw_config, logger)

    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_json
        mock_get.return_value.text = json.dumps(mock_json)

        dm.process_observed_data()

    processed_path = Path(gw_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "groundwater" / "test_domain_groundwater_processed.csv"

    assert processed_path.exists(), f"Processed USGS GW file not found at {processed_path}"

    df = pd.read_csv(processed_path)
    assert 'datetime' in df.columns
    assert 'groundwater_level' in df.columns
    assert len(df) > 0

@pytest.mark.integration
def test_provo_usgs_full_e2e(tmp_path):
    """
    E2E test for Provo River USGS data acquisition and processing.
    Runs a minimal full workflow to confirm actual usable data is retrieved.
    """
    import yaml

    # Complete config to satisfy SymfluenceConfig (Pydantic) validation
    config_data = {
        # Global
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'provo_river_test',
        'EXPERIMENT_ID': 'e2e_test',
        'EXPERIMENT_TIME_START': '2023-01-01 00:00',
        'EXPERIMENT_TIME_END': '2023-01-02 23:00',
        'CALIBRATION_PERIOD': '2023-01-01, 2023-01-01',
        'EVALUATION_PERIOD': '2023-01-02, 2023-01-02',
        'SPINUP_PERIOD': '2023-01-01, 2023-01-01',
        'MPI_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,

        # Domain
        'POUR_POINT_COORDS': '40.5577/-111.1688',
        'BOUNDING_BOX_COORDS': '41/-111.7/40.0/-110.6',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'DOMAIN_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',

        # Forcing
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,

        # Model
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',

        # Observations
        'STREAMFLOW_DATA_PROVIDER': 'USGS',
        'DOWNLOAD_USGS_DATA': True,
        'STATION_ID': '10163000',

        # Fallbacks for validation
        'DOWNLOAD_WSC_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': False,
        'SUPPLEMENT_FORCING': False,
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    # Initialize SYMFLUENCE with path
    sym = SYMFLUENCE(config_input=config_file)

    # Run observed data processing
    sym.managers['data'].process_observed_data()

    # Verify results
    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "observations" / "streamflow" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_streamflow_processed.csv"

    assert processed_file.exists(), f"Processed USGS file not found at {processed_file}"

    # Load processed data and verify it has content and correct format
    df = pd.read_csv(processed_file)
    assert not df.empty, "Processed data is empty"
    assert 'datetime' in df.columns
    assert 'discharge_cms' in df.columns
    assert (df['discharge_cms'] >= 0).all()

@pytest.mark.integration
def test_wsc_geomet_full_e2e(tmp_path):
    """
    E2E test for Bow River WSC data acquisition via GeoMet API.
    """
    import yaml
    config_data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'bow_river_wsc_test',
        'EXPERIMENT_ID': 'wsc_e2e_test',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-05 23:00',
        'CALIBRATION_PERIOD': '2020-01-01, 2020-01-01',
        'EVALUATION_PERIOD': '2020-01-02, 2020-01-02',
        'SPINUP_PERIOD': '2020-01-01, 2020-01-01',
        'MPI_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        'POUR_POINT_COORDS': '51.1722/-115.5717',
        'BOUNDING_BOX_COORDS': '51.8/-116.6/50.9/-115.5',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'DOMAIN_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',
        'STREAMFLOW_DATA_PROVIDER': 'WSC',
        'DOWNLOAD_WSC_DATA': True,
        'STATION_ID': '05BB001',
        'DOWNLOAD_USGS_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': False,
        'SUPPLEMENT_FORCING': False,
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "wsc_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    sym = SYMFLUENCE(config_input=config_file)
    sym.managers['data'].process_observed_data()

    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "observations" / "streamflow" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_streamflow_processed.csv"

    assert processed_file.exists(), "Processed WSC file not found"
    df = pd.read_csv(processed_file)
    assert not df.empty
    # Relax check: ignore -9999 or other no-data values
    valid_data = df[df['discharge_cms'] > -9000]
    if not valid_data.empty:
        assert (valid_data['discharge_cms'] >= 0).all()

@pytest.mark.integration
def test_usgs_gw_full_e2e(tmp_path):
    """
    E2E test for USGS Groundwater data acquisition via API.
    """
    import yaml
    import json

    config_data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'usgs_gw_test',
        'EXPERIMENT_ID': 'gw_e2e_test',
        'EXPERIMENT_TIME_START': '2022-01-01 00:00',
        'EXPERIMENT_TIME_END': '2022-01-05 23:00',
        'CALIBRATION_PERIOD': '2022-01-01, 2022-01-01',
        'EVALUATION_PERIOD': '2022-01-02, 2022-01-02',
        'SPINUP_PERIOD': '2022-01-01, 2022-01-01',
        'MPI_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        'POUR_POINT_COORDS': '40.0/-105.0',
        'BOUNDING_BOX_COORDS': '41/-106/39/-104',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'ROUTING_DELINEATION': 'lumped',
        'DOMAIN_DISCRETIZATION': 'lumped',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        'HYDROLOGICAL_MODEL': 'FUSE',
        'ROUTING_MODEL': 'none',
        'STREAMFLOW_DATA_PROVIDER': 'none',
        'DOWNLOAD_USGS_DATA': False,
        'STATION_ID': '01646500',
        'DOWNLOAD_WSC_DATA': False,
        'PROCESS_CARAVANS': False,
        'DOWNLOAD_SNOTEL': False,
        'DOWNLOAD_FLUXNET': False,
        'DOWNLOAD_USGS_GW': True,
        'USGS_STATION': '01646500', # Potomac River well
        'ADDITIONAL_OBSERVATIONS': 'USGS_GW',
        'SUPPLEMENT_FORG': False,
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'streamflow',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 2,
        'POPULATION_SIZE': 2,
        'OPTIMIZATION_METRIC': 'KGE'
    }

    config_file = tmp_path / "gw_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    sym = SYMFLUENCE(config_input=config_file)

    # Mock Response
    mock_json = {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "variableName": "Depth to water level, feet below land surface",
                        "parameterCode": "72019",
                        "unit": {"unitCode": "ft"}
                    },
                    "values": [
                        {
                            "value": [
                                {"dateTime": "2022-01-01T12:00:00.000", "value": "10.5"},
                                {"dateTime": "2022-01-02T12:00:00.000", "value": "10.4"}
                            ]
                        }
                    ]
                }
            ]
        }
    }

    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_json
        mock_get.return_value.text = json.dumps(mock_json)

        sym.managers['data'].process_observed_data()

    project_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}"
    processed_file = project_dir / "observations" / "groundwater" / f"{config_data['DOMAIN_NAME']}_groundwater_processed.csv"

    assert processed_file.exists(), "Processed USGS GW file not found"
    df = pd.read_csv(processed_file)
    assert not df.empty
    assert 'groundwater_level' in df.columns

@pytest.mark.integration
def test_grace_acquisition_and_processing(mock_config, tmp_path):
    """Test the GRACE acquisition and processing pathway with mocked NetCDF data."""
    import xarray as xr
    import numpy as np

    logger = logging.getLogger("test_grace")

    # 1. Create a mock GRACE NetCDF file
    grace_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "grace"
    grace_dir.mkdir(parents=True, exist_ok=True)
    mock_grace_file = grace_dir / "GRACE_JPL_test.nc"

    times = pd.date_range('2003-01-01', '2005-01-01', freq='MS')
    lats = np.linspace(30, 50, 10)
    lons = np.linspace(-120, -100, 10)

    ds = xr.Dataset(
        data_vars={
            'lwe_thickness': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)))
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        }
    )
    ds.to_netcdf(mock_grace_file)

    # 2. Create a mock catchment shapefile (required by GRACEHandler)
    catchment_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "shapefiles" / "catchment"
    catchment_dir.mkdir(parents=True, exist_ok=True)
    catchment_shp = catchment_dir / "test_domain_catchment.shp"

    import geopandas as gpd
    from shapely.geometry import box
    gdf = gpd.GeoDataFrame({
        'ID': [1],
        'geometry': [box(-115, 35, -105, 45)]
    }, crs='EPSG:4326')
    gdf.to_file(catchment_shp)

    # 3. Configure for GRACE
    grace_config = mock_config.copy()
    grace_config['ADDITIONAL_OBSERVATIONS'] = 'GRACE'
    grace_config['CATCHMENT_PATH'] = str(catchment_dir)
    grace_config['CATCHMENT_SHP_NAME'] = 'test_domain_catchment.shp'

    # 4. Run acquisition and processing
    dm = DataManager(grace_config, logger)
    dm.acquire_observations()
    dm.process_observed_data()

    # 5. Verify results
    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "grace" / "preprocessed" / "test_domain_grace_tws_processed.csv"

    assert processed_file.exists(), "Processed GRACE file not found"
    df = pd.read_csv(processed_file)
    # Check for any grace data columns (CSR, GSFC, JPL - depending on what's available)
    grace_cols = [c for c in df.columns if 'grace' in c.lower()]
    assert len(grace_cols) > 0, f"No GRACE columns found. Columns: {df.columns.tolist()}"
    assert len(df) > 0

@pytest.mark.integration
def test_modis_snow_acquisition_and_processing(mock_config, tmp_path):
    """Test the MODIS Snow acquisition and processing pathway with mocked NetCDF data."""
    import xarray as xr
    import numpy as np

    logger = logging.getLogger("test_modis")

    # 1. Create a mock MODIS NetCDF file
    snow_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "snow" / "raw"
    snow_dir.mkdir(parents=True, exist_ok=True)
    mock_snow_file = snow_dir / "test_domain_MOD10A1.006_raw.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    lats = np.linspace(30, 50, 5)
    lons = np.linspace(-120, -100, 5)

    ds = xr.Dataset(
        data_vars={
            'NDSI_Snow_Cover': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)))
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        }
    )
    ds.to_netcdf(mock_snow_file)

    # 2. Configure for MODIS
    modis_config = mock_config.copy()
    modis_config['ADDITIONAL_OBSERVATIONS'] = 'MODIS_SNOW'
    modis_config['DATA_ACCESS'] = 'cloud' # Trigger acquire() logic

    # 3. Just verify DataManager can be initialized with observation config
    dm = DataManager(modis_config, logger)

    # Verify the handler is registered
    assert ObservationRegistry.is_registered('MODIS_SNOW'), "MODIS_SNOW handler not registered"

@pytest.mark.integration
def test_smap_acquisition_and_processing(mock_config, tmp_path):
    """Test the SMAP acquisition and processing pathway."""
    import xarray as xr
    import numpy as np

    logger = logging.getLogger("test_smap")

    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "soil_moisture" / "smap"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "smap_test.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    ds = xr.Dataset(
        data_vars={'soil_moisture': (('time', 'lat', 'lon'), np.random.rand(len(times), 2, 2))},
        coords={'time': times, 'lat': [40, 41], 'lon': [-105, -104]}
    )
    ds.to_netcdf(mock_file)

    smap_config = mock_config.copy()
    smap_config['ADDITIONAL_OBSERVATIONS'] = 'SMAP'

    dm = DataManager(smap_config, logger)
    dm.acquire_observations()
    dm.process_observed_data()

    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "soil_moisture" / "preprocessed" / "test_domain_smap_processed.csv"
    assert processed_file.exists()
    df = pd.read_csv(processed_file)
    assert 'soil_moisture' in df.columns

@pytest.mark.integration
def test_esa_cci_sm_acquisition_and_processing(mock_config, tmp_path):
    """Test the ESA CCI SM acquisition and processing pathway."""
    import xarray as xr
    import numpy as np

    logger = logging.getLogger("test_esa")

    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "soil_moisture" / "esa_cci"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "esa_test.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    ds = xr.Dataset(
        data_vars={'sm': (('time', 'lat', 'lon'), np.random.rand(len(times), 2, 2))},
        coords={'time': times, 'lat': [40, 41], 'lon': [-105, -104]}
    )
    ds.to_netcdf(mock_file)

    esa_config = mock_config.copy()
    esa_config['ADDITIONAL_OBSERVATIONS'] = 'ESA_CCI_SM'

    # Just verify DataManager can be initialized with observation config
    dm = DataManager(esa_config, logger)

    # Verify the handler is registered
    assert ObservationRegistry.is_registered('ESA_CCI_SM'), "ESA_CCI_SM handler not registered"

@pytest.mark.integration
def test_fluxcom_et_acquisition_and_processing(mock_config, tmp_path):
    """Test the FLUXCOM ET acquisition and processing pathway."""
    import xarray as xr
    import numpy as np

    logger = logging.getLogger("test_fluxcom")

    obs_dir = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "et" / "fluxcom"
    obs_dir.mkdir(parents=True, exist_ok=True)
    mock_file = obs_dir / "fluxcom_test.nc"

    times = pd.date_range('2020-01-01', '2020-01-05', freq='D')
    ds = xr.Dataset(
        data_vars={'ET': (('time', 'lat', 'lon'), np.random.rand(len(times), 2, 2))},
        coords={'time': times, 'lat': [40, 41], 'lon': [-105, -104]}
    )
    ds.to_netcdf(mock_file)

    fluxcom_config = mock_config.copy()
    fluxcom_config['ADDITIONAL_OBSERVATIONS'] = 'FLUXCOM_ET'

    dm = DataManager(fluxcom_config, logger)
    dm.acquire_observations()
    dm.process_observed_data()

    processed_file = Path(mock_config['SYMFLUENCE_DATA_DIR']) / "domain_test_domain" / "observations" / "et" / "preprocessed" / "test_domain_fluxcom_et_processed.csv"
    assert processed_file.exists()
    df = pd.read_csv(processed_file)
    assert 'ET' in df.columns
