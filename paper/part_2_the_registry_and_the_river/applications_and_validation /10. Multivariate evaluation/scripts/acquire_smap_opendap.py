#!/usr/bin/env python3
"""
Acquire SMAP L4 Soil Moisture via GES DISC OPeNDAP.

SMAP L4 is hosted at NASA GES DISC with OPeNDAP access.
Uses xarray with pydap for authenticated access.
"""

import os
import netrc
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
OUTPUT_DIR = DATA_DIR / "observations" / "soil_moisture" / "smap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Domain bounds for Bow at Banff
BBOX = {
    'lat_min': 50.95,
    'lat_max': 51.73,
    'lon_min': -116.55,
    'lon_max': -115.53
}

# Period
ANALYSIS_START = '2015-04-01'
ANALYSIS_END = '2017-12-31'


def get_earthdata_credentials():
    """Get Earthdata credentials."""
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    user = os.environ.get('EARTHDATA_USERNAME')
    password = os.environ.get('EARTHDATA_PASSWORD')
    if user and password:
        return user, password

    raise ValueError("No Earthdata credentials found")


def setup_opendap_session():
    """Setup authenticated OPeNDAP session."""
    try:
        from pydap.cas.urs import setup_session
        user, password = get_earthdata_credentials()
        # Create session for GES DISC
        session = setup_session(user, password, check_url="https://urs.earthdata.nasa.gov")
        return session
    except ImportError:
        logger.error("pydap not available. Install with: pip install pydap")
        return None
    except Exception as e:
        logger.error(f"Failed to setup OPeNDAP session: {e}")
        return None


def download_smap_l4_monthly(year, month, session):
    """Download SMAP L4 for one month via OPeNDAP."""
    import xarray as xr

    # GES DISC OPeNDAP URL for SMAP L4
    # Format: https://hydro1.gesdisc.eosdis.nasa.gov/opendap/SMAP_L4/SPL4SMGP.008/YYYY/DDD/
    base_url = "https://hydro1.gesdisc.eosdis.nasa.gov/opendap/SMAP_L4/SPL4SMGP.008"

    # Get first day of month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    monthly_data = []

    # Sample one file per week for efficiency
    current = start_date
    while current <= end_date:
        doy = current.timetuple().tm_yday

        # Construct URL for this day (first 3-hourly file)
        # File naming: SMAP_L4_SM_gph_YYYYDDDTHHMMSS_Vv5030_001.h5
        file_pattern = f"SMAP_L4_SM_gph_{year}{doy:03d}"
        day_url = f"{base_url}/{year}/{doy:03d}/"

        logger.info(f"  Trying {current.strftime('%Y-%m-%d')}...")

        try:
            # Try to access via OPeNDAP
            # This is a test - actual implementation would need file discovery
            response = session.get(day_url, timeout=30)
            if response.status_code == 200:
                # Parse for .h5 files
                import re
                files = re.findall(r'href="([^"]+\.h5)"', response.text)
                if files:
                    # Get first file of the day
                    file_url = f"{day_url}{files[0]}"
                    opendap_url = file_url.replace('/opendap/', '/opendap/hyrax/')

                    logger.info(f"    Found: {files[0]}")

                    # Open with xarray
                    try:
                        ds = xr.open_dataset(
                            opendap_url,
                            engine='pydap',
                            backend_kwargs={'session': session}
                        )

                        # Extract soil moisture for our bbox
                        if 'sm_surface' in ds:
                            sm = ds['sm_surface']
                            # Select bbox
                            if 'lat' in sm.dims and 'lon' in sm.dims:
                                sm_subset = sm.sel(
                                    lat=slice(BBOX['lat_min'], BBOX['lat_max']),
                                    lon=slice(BBOX['lon_min'], BBOX['lon_max'])
                                )
                                sm_mean = float(sm_subset.mean().values)
                                monthly_data.append({
                                    'date': current,
                                    'sm_surface': sm_mean
                                })
                        ds.close()
                    except Exception as e:
                        logger.debug(f"    OPeNDAP access failed: {e}")

        except Exception as e:
            logger.debug(f"    Error: {e}")

        current += timedelta(days=7)  # Sample weekly

    return monthly_data


def try_gesdisc_subset_api():
    """Try GES DISC subset/order API as alternative."""
    import requests

    user, password = get_earthdata_credentials()
    session = requests.Session()
    session.auth = (user, password)

    # GES DISC Subset API
    subset_url = "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi"

    # Test parameters
    params = {
        'FILENAME': '/data/SMAP_L4/SPL4SMGP.008/2016/001/SMAP_L4_SM_gph_20160101T013000_Vv5030_001.h5',
        'FORMAT': 'netCDF',
        'BBOX': f"{BBOX['lat_min']},{BBOX['lon_min']},{BBOX['lat_max']},{BBOX['lon_max']}",
        'LABEL': 'SMAP_L4_SM_subset.nc',
        'SHORTNAME': 'SPL4SMGP',
        'SERVICE': 'SUBSET_LATS4D',
        'DATASET_VERSION': '008',
        'VARIABLES': 'sm_surface,sm_rootzone'
    }

    logger.info("Testing GES DISC Subset API...")
    try:
        response = session.get(subset_url, params=params, timeout=60)
        logger.info(f"  Status: {response.status_code}")
        if response.status_code == 200:
            # Save test file
            test_file = OUTPUT_DIR / "smap_subset_test.nc"
            with open(test_file, 'wb') as f:
                f.write(response.content)
            logger.info(f"  Saved test subset to {test_file}")
            return True
        else:
            logger.info(f"  Response: {response.text[:200]}")
    except Exception as e:
        logger.info(f"  Error: {e}")

    return False


def try_podaac_harmony():
    """Try PODAAC Harmony API for SMAP."""
    import requests

    user, password = get_earthdata_credentials()
    session = requests.Session()
    session.auth = (user, password)

    # Harmony API endpoint
    harmony_url = "https://harmony.earthdata.nasa.gov"

    # SMAP collection ID on PODAAC
    collection_id = "C2251464909-POCLOUD"  # SPL3SMP Enhanced

    # Build subset request
    subset_params = {
        'subset': f'lat({BBOX["lat_min"]}:{BBOX["lat_max"]})',
        'subset': f'lon({BBOX["lon_min"]}:{BBOX["lon_max"]})',
        'temporal': f'{ANALYSIS_START}T00:00:00Z,2015-04-30T23:59:59Z',  # First month only for test
        'maxResults': 5
    }

    logger.info("Testing PODAAC Harmony API...")
    try:
        url = f"{harmony_url}/{collection_id}/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset"
        response = session.get(url, params=subset_params, timeout=60)
        logger.info(f"  Status: {response.status_code}")
        if response.status_code == 200:
            logger.info("  Harmony access successful!")
            return True
        elif response.status_code == 303:
            # Redirect to job status
            job_url = response.headers.get('Location')
            logger.info(f"  Job submitted: {job_url}")
            return True
        else:
            logger.info(f"  Response: {response.text[:200]}")
    except Exception as e:
        logger.info(f"  Error: {e}")

    return False


def main():
    logger.info("=" * 60)
    logger.info("SMAP L4 Acquisition via OPeNDAP/APIs")
    logger.info("=" * 60)
    logger.info(f"Period: {ANALYSIS_START} to {ANALYSIS_END}")
    logger.info("")

    # Try different methods
    methods_tried = []

    # Method 1: GES DISC Subset API
    if try_gesdisc_subset_api():
        methods_tried.append("GES DISC Subset")

    # Method 2: PODAAC Harmony
    if try_podaac_harmony():
        methods_tried.append("PODAAC Harmony")

    # Method 3: Direct OPeNDAP
    session = setup_opendap_session()
    if session:
        logger.info("\nTrying direct OPeNDAP access...")
        try:
            data = download_smap_l4_monthly(2016, 7, session)
            if data:
                logger.info(f"  Got {len(data)} samples via OPeNDAP")
                methods_tried.append("OPeNDAP")
        except Exception as e:
            logger.info(f"  OPeNDAP failed: {e}")

    if methods_tried:
        logger.info(f"\nWorking methods: {', '.join(methods_tried)}")
    else:
        logger.info("\nNo direct access methods worked.")
        logger.info("AppEEARS task is the best option - waiting for completion...")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
