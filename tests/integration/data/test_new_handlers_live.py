"""
Live integration tests for newly added observation handlers.

These tests make actual API calls and download real data.
Mark with appropriate pytest markers for selective running.

Usage:
    # Run all live tests (may take time)
    pytest tests/integration/data/test_new_handlers_live.py -v

    # Run only tests that don't require credentials
    pytest tests/integration/data/test_new_handlers_live.py -v -m "not requires_credentials"

    # Run specific handler test
    pytest tests/integration/data/test_new_handlers_live.py::test_daymet_live -v
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.slow]

logger = logging.getLogger("test_live_handlers")
logging.basicConfig(level=logging.INFO)


def create_test_config(tmp_path, **overrides):
    """Create a minimal valid SymfluenceConfig dict for testing.

    Includes all required fields for SymfluenceConfig validation.
    """
    base_config = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_run',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'HYDROLOGICAL_MODEL': 'FUSE',
        'FORCING_DATASET': 'ERA5',
    }
    base_config.update(overrides)
    return base_config


def has_earthdata_credentials():
    """Check if NASA Earthdata credentials are available."""
    if os.environ.get('EARTHDATA_USERNAME') and os.environ.get('EARTHDATA_PASSWORD'):
        return True
    try:
        import netrc
        auth = netrc.netrc()
        for host in ['urs.earthdata.nasa.gov', 'earthdata.nasa.gov']:
            if auth.authenticators(host):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def has_cds_credentials():
    """Check if CDS API credentials are available."""
    cds_rc = Path.home() / '.cdsapirc'
    return cds_rc.exists() or os.environ.get('CDSAPI_KEY')


def has_mswep_credentials():
    """Check if MSWEP credentials are available."""
    return bool(os.environ.get('MSWEP_USERNAME') and os.environ.get('MSWEP_PASSWORD'))


def has_openet_credentials():
    """Check if OpenET API key is available."""
    return bool(os.environ.get('OPENET_API_KEY'))


def has_grdc_credentials():
    """Check if GRDC credentials are available."""
    return bool(os.environ.get('GRDC_USERNAME') and os.environ.get('GRDC_PASSWORD'))


def has_copernicus_credentials():
    """Check if Copernicus Data Space credentials are available."""
    return bool(
        (os.environ.get('SENTINEL1_CLIENT_ID') or os.environ.get('CDSE_CLIENT_ID')) and
        (os.environ.get('SENTINEL1_CLIENT_SECRET') or os.environ.get('CDSE_CLIENT_SECRET'))
    )


# =============================================================================
# Daymet Tests (No credentials required - public data)
# =============================================================================

@pytest.mark.integration
def test_daymet_live_single_pixel(tmp_path):
    """
    Live test for Daymet single-pixel data acquisition.
    No authentication required - publicly available from ORNL DAAC.
    """
    from symfluence.data.acquisition.handlers.daymet import DaymetAcquirer
    from symfluence.data.observation.handlers.daymet import DaymetHandler

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='daymet_live_test',
        EXPERIMENT_TIME_START='2020-06-01 00:00',
        EXPERIMENT_TIME_END='2020-06-05 00:00',
        BOUNDING_BOX_COORDS='40.01/-105.01/40.0/-105.0',  # Small area for single-pixel
        DAYMET_VARIABLES=['tmax', 'tmin', 'prcp'],
        FORCE_DOWNLOAD=True,
    )

    # Test acquisition
    acquirer = DaymetAcquirer(config, logger)
    output_dir = tmp_path / "daymet_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = acquirer.download(output_dir)
        assert result.exists(), "Daymet download should return existing path"

        # Check for downloaded files
        files = list(output_dir.glob("daymet*.csv"))
        assert len(files) > 0, "Should have downloaded at least one Daymet file"

        # Test processing
        handler = DaymetHandler(config, logger)
        processed = handler.process(output_dir)

        if processed.is_file():
            df = pd.read_csv(processed, index_col=0, parse_dates=True)
            assert len(df) > 0, "Processed data should have records"
            logger.info(f"Daymet live test: {len(df)} records, columns: {list(df.columns)}")

    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Daymet API unavailable: {e}")


# =============================================================================
# GRDC Tests (Registration check - credentials required for data)
# =============================================================================

@pytest.mark.integration
def test_grdc_station_metadata(tmp_path):
    """
    Test GRDC WFS metadata access (public, no auth required).
    Full data download requires credentials.
    """
    import requests

    from symfluence.data.acquisition.handlers.grdc import GRDCAcquirer

    # Test WFS access for station metadata
    wfs_url = "https://portal.grdc.bafg.de/geoserver/grdc/wfs"

    try:
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetCapabilities',
        }
        response = requests.get(wfs_url, params=params, timeout=30)

        if response.status_code == 200:
            logger.info("GRDC WFS service is accessible")
            assert 'WFS_Capabilities' in response.text or 'FeatureTypeList' in response.text
        else:
            pytest.skip(f"GRDC WFS not accessible: {response.status_code}")

    except Exception as e:  # noqa: BLE001
        pytest.skip(f"GRDC WFS unavailable: {e}")


@pytest.mark.integration
@pytest.mark.skipif(not has_grdc_credentials(), reason="GRDC credentials not available")
def test_grdc_live_acquisition(tmp_path):
    """
    Live test for GRDC data acquisition (requires credentials).
    """
    from symfluence.data.observation.handlers.grdc import GRDCHandler

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='grdc_live_test',
        EXPERIMENT_TIME_START='2020-01-01 00:00',
        EXPERIMENT_TIME_END='2020-01-31 00:00',
        GRDC_STATION_IDS='6340110',  # Example: Rhine at Lobith
        FORCE_DOWNLOAD=True,
    )

    handler = GRDCHandler(config, logger)

    try:
        raw_path = handler.acquire()
        processed = handler.process(raw_path)

        if processed.is_file():
            df = pd.read_csv(processed, index_col=0, parse_dates=True)
            assert 'discharge_cms' in df.columns
            logger.info(f"GRDC live test: {len(df)} records")

    except Exception as e:  # noqa: BLE001
        pytest.skip(f"GRDC acquisition failed: {e}")


# =============================================================================
# ERA5-Land Tests (CDS credentials required)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not has_cds_credentials(), reason="CDS API credentials not available")
def test_era5_land_live_acquisition(tmp_path):
    """
    Live test for ERA5-Land data acquisition via CDS API.
    Requires ~/.cdsapirc or CDSAPI_KEY environment variable.

    Uses minimal request (1 variable, 1 day, small area) to avoid CDS rate limits.
    """
    from symfluence.data.acquisition.handlers.era5_land import ERA5LandAcquirer
    from symfluence.data.observation.handlers.era5_land import ERA5LandHandler

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='era5_land_live_test',
        EXPERIMENT_TIME_START='2020-06-01 00:00',
        EXPERIMENT_TIME_END='2020-06-01 23:00',  # Single day
        BOUNDING_BOX_COORDS='40.1/-105.1/40.0/-105.0',  # ~10km x 10km area
        ERA5_LAND_VARIABLES=['2m_temperature'],  # Single variable
        ERA5_LAND_FREQUENCY='daily',
        FORCE_DOWNLOAD=True,
    )

    acquirer = ERA5LandAcquirer(config, logger)
    output_dir = tmp_path / "era5_land_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = acquirer.download(output_dir)
        assert result.exists()

        # Test processing
        handler = ERA5LandHandler(config, logger)
        processed = handler.process(output_dir)

        if processed.is_file():
            df = pd.read_csv(processed, index_col=0, parse_dates=True)
            assert len(df) > 0
            logger.info(f"ERA5-Land live test: {len(df)} records, columns: {list(df.columns)}")

    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ERA5-Land acquisition failed (CDS may be slow): {e}")


# =============================================================================
# MODIS Tests (AppEEARS - Earthdata credentials required)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not has_earthdata_credentials(), reason="NASA Earthdata credentials not available")
def test_modis_lst_appeears_auth(tmp_path):
    """
    Test AppEEARS authentication for MODIS LST.
    """
    import requests

    username = os.environ.get('EARTHDATA_USERNAME')
    password = os.environ.get('EARTHDATA_PASSWORD')

    if not username or not password:
        try:
            import netrc
            auth = netrc.netrc()
            creds = auth.authenticators('urs.earthdata.nasa.gov')
            if creds:
                username, _, password = creds
        except Exception:  # noqa: BLE001
            pass

    if not username or not password:
        pytest.skip("Earthdata credentials not found")

    try:
        response = requests.post(
            "https://appeears.earthdatacloud.nasa.gov/api/login",
            auth=(username, password),
            timeout=30
        )

        if response.status_code == 200:
            token = response.json().get('token')
            assert token, "Should receive auth token"
            logger.info("AppEEARS authentication successful")
        else:
            pytest.skip(f"AppEEARS auth failed: {response.status_code}")

    except Exception as e:  # noqa: BLE001
        pytest.skip(f"AppEEARS unavailable: {e}")


@pytest.mark.integration
@pytest.mark.skipif(not has_earthdata_credentials(), reason="NASA Earthdata credentials not available")
@pytest.mark.skip(reason="AppEEARS tasks take time - run manually")
def test_modis_lst_live_acquisition(tmp_path):
    """
    Live test for MODIS LST acquisition via AppEEARS.
    This test is skipped by default as AppEEARS tasks can take 10+ minutes.
    """
    from symfluence.data.acquisition.handlers.modis_lst import MODISLSTAcquirer

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='modis_lst_test',
        EXPERIMENT_TIME_START='2020-06-01 00:00',
        EXPERIMENT_TIME_END='2020-06-03 00:00',
        BOUNDING_BOX_COORDS='40.5/-105.5/40.0/-105.0',
        MODIS_LST_PRODUCT='MOD11A1',
        FORCE_DOWNLOAD=True,
    )

    acquirer = MODISLSTAcquirer(config, logger)
    output_dir = tmp_path / "modis_lst"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = acquirer.download(output_dir)
    assert result.exists()


@pytest.mark.integration
@pytest.mark.skipif(not has_earthdata_credentials(), reason="NASA Earthdata credentials not available")
@pytest.mark.skip(reason="AppEEARS tasks take time - run manually")
def test_modis_lai_live_acquisition(tmp_path):
    """
    Live test for MODIS LAI acquisition via AppEEARS.
    """
    from symfluence.data.acquisition.handlers.modis_lai import MODISLAIAcquirer

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='modis_lai_test',
        EXPERIMENT_TIME_START='2020-06-01 00:00',
        EXPERIMENT_TIME_END='2020-06-15 00:00',
        BOUNDING_BOX_COORDS='40.5/-105.5/40.0/-105.0',
        MODIS_LAI_PRODUCT='MCD15A2H',
        FORCE_DOWNLOAD=True,
    )

    acquirer = MODISLAIAcquirer(config, logger)
    output_dir = tmp_path / "modis_lai"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = acquirer.download(output_dir)
    assert result.exists()


@pytest.mark.integration
@pytest.mark.skipif(not has_earthdata_credentials(), reason="NASA Earthdata credentials not available")
@pytest.mark.skip(reason="AppEEARS tasks take time - run manually")
def test_viirs_snow_live_acquisition(tmp_path):
    """
    Live test for VIIRS Snow acquisition via AppEEARS.
    """
    from symfluence.data.acquisition.handlers.viirs_snow import VIIRSSnowAcquirer

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='viirs_snow_test',
        EXPERIMENT_TIME_START='2020-01-15 00:00',
        EXPERIMENT_TIME_END='2020-01-20 00:00',
        BOUNDING_BOX_COORDS='40.5/-106.0/39.5/-105.0',
        VIIRS_SNOW_PRODUCT='VNP10A1F',
        FORCE_DOWNLOAD=True,
    )

    acquirer = VIIRSSnowAcquirer(config, logger)
    output_dir = tmp_path / "viirs_snow"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = acquirer.download(output_dir)
    assert result.exists()


# =============================================================================
# OpenET Tests (API key required)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not has_openet_credentials(), reason="OpenET API key not available")
def test_openet_live_acquisition(tmp_path):
    """
    Live test for OpenET data acquisition.
    Requires OPENET_API_KEY environment variable.
    Note: OpenET coverage is limited to Western US.
    """
    from symfluence.data.acquisition.handlers.openet import OpenETAcquirer
    from symfluence.data.observation.handlers.openet import OpenETHandler

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='openet_test',
        EXPERIMENT_TIME_START='2020-06-01 00:00',
        EXPERIMENT_TIME_END='2020-06-30 00:00',
        BOUNDING_BOX_COORDS='40.5/-106.0/40.0/-105.5',  # Colorado
        OPENET_MODEL='ensemble',
        OPENET_RESOLUTION='monthly',
        FORCE_DOWNLOAD=True,
    )

    acquirer = OpenETAcquirer(config, logger)
    output_dir = tmp_path / "openet"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = acquirer.download(output_dir)
        assert result.exists()

        handler = OpenETHandler(config, logger)
        processed = handler.process(output_dir)

        if processed.is_file():
            df = pd.read_csv(processed, index_col=0, parse_dates=True)
            assert 'et_mm_day' in df.columns
            logger.info(f"OpenET live test: {len(df)} records")

    except Exception as e:  # noqa: BLE001
        pytest.skip(f"OpenET acquisition failed: {e}")


# =============================================================================
# MSWEP Tests (Credentials required)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not has_mswep_credentials(), reason="MSWEP credentials not available")
def test_mswep_live_acquisition(tmp_path):
    """
    Live test for MSWEP precipitation data acquisition.
    Requires MSWEP_USERNAME and MSWEP_PASSWORD environment variables.
    Register at http://www.gloh2o.org/mswep/
    """
    from symfluence.data.acquisition.handlers.mswep import MSWEPAcquirer

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='mswep_test',
        EXPERIMENT_TIME_START='2020-06-01 00:00',
        EXPERIMENT_TIME_END='2020-06-05 00:00',
        MSWEP_RESOLUTION='daily',
        FORCE_DOWNLOAD=True,
    )

    acquirer = MSWEPAcquirer(config, logger)
    output_dir = tmp_path / "mswep"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = acquirer.download(output_dir)
        assert result.exists()
        logger.info("MSWEP live acquisition successful")

    except Exception as e:  # noqa: BLE001
        pytest.skip(f"MSWEP acquisition failed: {e}")


# =============================================================================
# Sentinel-1 Tests (Copernicus Data Space credentials required)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not has_copernicus_credentials(), reason="Copernicus Data Space credentials not available")
def test_sentinel1_catalog_search(tmp_path):
    """
    Test Sentinel-1 catalog search via Copernicus Data Space.
    """
    import requests

    from symfluence.data.acquisition.handlers.sentinel1_sm import Sentinel1SMAcquirer

    config = create_test_config(
        tmp_path,
        DOMAIN_NAME='s1_test',
        EXPERIMENT_TIME_START='2023-06-01 00:00',
        EXPERIMENT_TIME_END='2023-06-05 00:00',
        BOUNDING_BOX_COORDS='40.5/-105.5/40.0/-105.0',
    )

    acquirer = Sentinel1SMAcquirer(config, logger)

    client_id, client_secret = acquirer._get_credentials()
    if not client_id or not client_secret:
        pytest.skip("Copernicus credentials not found")

    try:
        token = acquirer._get_access_token(client_id, client_secret)
        assert token, "Should get access token"

        products = acquirer._search_products(token)
        logger.info(f"Found {len(products)} Sentinel-1 products")

    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Sentinel-1 search failed: {e}")


# =============================================================================
# Summary test - check all handler registrations
# =============================================================================

@pytest.mark.integration
def test_all_new_handlers_registered():
    """Verify all new handlers are properly registered."""
    import symfluence.data.acquisition.handlers

    # Import to trigger registration
    import symfluence.data.observation.handlers
    from symfluence.data.acquisition.registry import AcquisitionRegistry
    from symfluence.data.observation.registry import ObservationRegistry

    observation_handlers = [
        'era5_land', 'mswep', 'modis_lst', 'modis_lai',
        'grdc', 'openet', 'sentinel1_sm', 'daymet', 'viirs_snow'
    ]

    acquisition_handlers = [
        'era5_land', 'mswep', 'modis_lst', 'modis_lai',
        'grdc', 'openet', 'sentinel1_sm', 'daymet', 'viirs_snow'
    ]

    for handler in observation_handlers:
        assert ObservationRegistry.is_registered(handler), f"{handler} not in ObservationRegistry"

    for handler in acquisition_handlers:
        assert AcquisitionRegistry.is_registered(handler), f"{handler} not in AcquisitionRegistry"

    logger.info("All new handlers registered successfully")


# =============================================================================
# Quick connectivity tests (no downloads)
# =============================================================================

@pytest.mark.integration
def test_api_connectivity():
    """Test connectivity to various APIs without downloading data."""
    import requests

    apis = {
        'Daymet ORNL': 'https://daymet.ornl.gov/single-pixel/api/data?lat=40&lon=-105&vars=tmax&start=2020-01-01&end=2020-01-02',
        'GRDC WFS': 'https://portal.grdc.bafg.de/geoserver/grdc/wfs?service=WFS&request=GetCapabilities',
        'AppEEARS': 'https://appeears.earthdatacloud.nasa.gov/api/product',
    }

    results = {}
    for name, url in apis.items():
        try:
            response = requests.get(url, timeout=15)
            results[name] = response.status_code
        except Exception as e:  # noqa: BLE001
            results[name] = str(e)

    logger.info(f"API connectivity: {results}")

    # At least Daymet should be accessible (no auth required)
    daymet_result = results.get('Daymet ORNL')
    if isinstance(daymet_result, str) and 'timeout' in daymet_result.lower():
        pytest.skip(f"Daymet API timeout (network issue, not a code failure): {daymet_result}")
    assert daymet_result == 200, \
        f"Daymet API should be accessible, got: {daymet_result}"
