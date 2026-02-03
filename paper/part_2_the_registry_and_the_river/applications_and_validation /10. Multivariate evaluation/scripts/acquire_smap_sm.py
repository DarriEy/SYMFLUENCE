#!/usr/bin/env python3
"""
Acquire SMAP Soil Moisture via NASA AppEEARS API.

SMAP L3 provides daily soil moisture at ~9km resolution.
"""

import os
import time
import requests
import logging
from pathlib import Path
from datetime import datetime
import netrc

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

# SMAP started in 2015, so adjust period
# For Bow at Banff, GRACE period is 2004-2017, but SMAP only 2015+
ANALYSIS_START = '2015-04-01'  # SMAP launch + commissioning
ANALYSIS_END = '2017-12-31'

# AppEEARS API
APPEEARS_API = "https://appeears.earthdatacloud.nasa.gov/api"


def get_earthdata_credentials():
    """Get Earthdata credentials from .netrc or environment."""
    # Try .netrc first
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    # Try environment variables
    user = os.environ.get('EARTHDATA_USERNAME')
    password = os.environ.get('EARTHDATA_PASSWORD')
    if user and password:
        return user, password

    raise ValueError("No Earthdata credentials found. Set EARTHDATA_USERNAME/PASSWORD or use .netrc")


def login_appeears():
    """Login to AppEEARS and get token."""
    user, password = get_earthdata_credentials()
    logger.info(f"Using Earthdata credentials for user: {user}")

    response = requests.post(
        f"{APPEEARS_API}/login",
        auth=(user, password)
    )
    response.raise_for_status()
    return response.json()['token']


def list_smap_products(token):
    """List available SMAP products in AppEEARS."""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f"{APPEEARS_API}/product", headers=headers)
    response.raise_for_status()

    products = response.json()
    smap_products = [p for p in products if 'SMAP' in p.get('ProductAndVersion', '').upper()
                     or 'SPL' in p.get('ProductAndVersion', '').upper()]

    logger.info(f"Found {len(smap_products)} SMAP products:")
    for p in smap_products:
        logger.info(f"  {p.get('ProductAndVersion')}: {p.get('Description', 'N/A')[:60]}")

    return smap_products


def submit_smap_task(token, product='SPL3SMP_E.005', layer='Soil_Moisture_Retrieval_Data_AM_soil_moisture'):
    """Submit AppEEARS area request for SMAP data."""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    task_name = f"SYMFLUENCE_Bow_at_Banff_SMAP_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    task = {
        "task_type": "area",
        "task_name": task_name,
        "params": {
            "dates": [
                {
                    "startDate": datetime.strptime(ANALYSIS_START, '%Y-%m-%d').strftime('%m-%d-%Y'),
                    "endDate": datetime.strptime(ANALYSIS_END, '%Y-%m-%d').strftime('%m-%d-%Y')
                }
            ],
            "layers": [
                {
                    "product": product,
                    "layer": layer
                }
            ],
            "geo": {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [BBOX['lon_min'], BBOX['lat_min']],
                            [BBOX['lon_max'], BBOX['lat_min']],
                            [BBOX['lon_max'], BBOX['lat_max']],
                            [BBOX['lon_min'], BBOX['lat_max']],
                            [BBOX['lon_min'], BBOX['lat_min']]
                        ]]
                    },
                    "properties": {"name": "Bow_at_Banff"}
                }]
            },
            "output": {
                "format": {"type": "netcdf4"},
                "projection": "geographic"
            }
        }
    }

    logger.info(f"Submitting AppEEARS task for {product} {layer}")
    response = requests.post(
        f"{APPEEARS_API}/task",
        headers=headers,
        json=task
    )

    if response.status_code != 202:
        logger.error(f"Task submission failed: {response.status_code}")
        logger.error(response.text)
        response.raise_for_status()

    task_id = response.json()['task_id']
    logger.info(f"Submitted task: {task_id}")
    return task_id


def check_existing_tasks(token):
    """Check for existing SMAP tasks."""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f"{APPEEARS_API}/task", headers=headers)
    response.raise_for_status()

    tasks = response.json()
    smap_tasks = [t for t in tasks if 'SMAP' in t.get('task_name', '').upper()
                  and 'Bow_at_Banff' in t.get('task_name', '')]

    if smap_tasks:
        logger.info(f"Found {len(smap_tasks)} existing SMAP tasks for this domain")
        for t in smap_tasks:
            logger.info(f"  {t['task_id']}: {t['task_name']} - {t['status']}")

    return smap_tasks


def wait_for_task(token, task_id, check_interval=30):
    """Wait for AppEEARS task to complete."""
    headers = {'Authorization': f'Bearer {token}'}

    logger.info(f"Waiting for task {task_id} to complete...")
    while True:
        response = requests.get(f"{APPEEARS_API}/task/{task_id}", headers=headers)
        response.raise_for_status()
        status = response.json()['status']

        logger.info(f"Task status: {status}")

        if status == 'done':
            return True
        elif status in ['error', 'deleted']:
            logger.error(f"Task failed with status: {status}")
            return False

        time.sleep(check_interval)


def download_results(token, task_id, output_dir):
    """Download completed task results."""
    headers = {'Authorization': f'Bearer {token}'}

    # Get file list
    response = requests.get(f"{APPEEARS_API}/bundle/{task_id}", headers=headers)
    response.raise_for_status()
    files = response.json()['files']

    nc_files = [f for f in files if f['file_name'].endswith('.nc')]
    logger.info(f"Found {len(nc_files)} NetCDF files to download")

    for file_info in nc_files:
        file_name = file_info['file_name']
        file_id = file_info['file_id']

        out_path = output_dir / file_name
        if out_path.exists():
            logger.info(f"Skipping {file_name} (exists)")
            continue

        logger.info(f"Downloading {file_name}...")
        response = requests.get(
            f"{APPEEARS_API}/bundle/{task_id}/{file_id}",
            headers=headers,
            stream=True
        )
        response.raise_for_status()

        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)

    logger.info(f"Downloaded {len(nc_files)} files to {output_dir}")
    return len(nc_files)


def try_nsidc_thredds():
    """Try direct NSIDC THREDDS access for SMAP."""
    logger.info("\n" + "=" * 50)
    logger.info("Trying NSIDC THREDDS access...")
    logger.info("=" * 50)

    user, password = get_earthdata_credentials()
    session = requests.Session()
    session.auth = (user, password)

    # SMAP L4 aggregated endpoint
    thredds_urls = [
        "https://n5eil01u.ecs.nsidc.org/thredds/ncss/grid/SMAP_L4_SM_gph_v4/aggregated.ncml",
        "https://n5eil01u.ecs.nsidc.org/thredds/ncss/grid/SMAP_L4_SM_gph_v5/aggregated.ncml",
    ]

    params = {
        "var": "sm_surface",
        "north": BBOX['lat_max'],
        "south": BBOX['lat_min'],
        "west": BBOX['lon_min'],
        "east": BBOX['lon_max'],
        "time_start": f"{ANALYSIS_START}T00:00:00Z",
        "time_end": f"{ANALYSIS_END}T23:59:59Z",
        "accept": "netcdf4"
    }

    for url in thredds_urls:
        logger.info(f"Trying: {url}")
        try:
            response = session.get(url, params=params, timeout=60)
            if response.status_code == 200:
                out_file = OUTPUT_DIR / "smap_thredds_subset.nc"
                with open(out_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Success! Downloaded to {out_file}")
                return True
            else:
                logger.warning(f"Status {response.status_code}: {response.reason}")
        except Exception as e:
            logger.warning(f"Failed: {e}")

    return False


def main():
    logger.info("=" * 60)
    logger.info("SMAP Soil Moisture Acquisition for Bow at Banff")
    logger.info("=" * 60)
    logger.info(f"Period: {ANALYSIS_START} to {ANALYSIS_END}")
    logger.info("Note: SMAP launched April 2015")
    logger.info("")

    # First try THREDDS (faster if it works)
    if try_nsidc_thredds():
        logger.info("THREDDS access successful!")
        return

    logger.info("\nTHREDDS failed, trying AppEEARS...")

    # Login to AppEEARS
    token = login_appeears()
    logger.info("Successfully logged into AppEEARS")

    # List available SMAP products
    smap_products = list_smap_products(token)

    # Check for existing tasks
    existing = check_existing_tasks(token)

    # If there's a completed task, download it
    for t in existing:
        if t['status'] == 'done':
            logger.info(f"Found completed task: {t['task_id']}")
            download_results(token, t['task_id'], OUTPUT_DIR)
            return

    # Submit new task - try different products (use current versions from listing)
    # SPL3SMP_E.006 is Enhanced L3 daily product
    products_to_try = [
        ('SPL3SMP_E.006', 'Soil_Moisture_Retrieval_Data_AM_soil_moisture'),
        ('SPL3SMP.009', 'Soil_Moisture_Retrieval_Data_AM_soil_moisture'),
        ('SPL4SMGP.008', 'Geophysical_Data_sm_surface'),
    ]

    task_id = None
    for product, layer in products_to_try:
        try:
            task_id = submit_smap_task(token, product, layer)
            break
        except Exception as e:
            logger.warning(f"Failed to submit task for {product}: {e}")
            continue

    if not task_id:
        logger.error("Could not submit any SMAP task")
        return

    # Wait for completion
    if wait_for_task(token, task_id):
        download_results(token, task_id, OUTPUT_DIR)

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
