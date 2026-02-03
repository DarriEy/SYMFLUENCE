#!/usr/bin/env python3
"""
Acquire SMAP L4 Soil Moisture directly from NSIDC DAAC Data Pool.

Uses direct HTTPS download with Earthdata authentication.
SMAP L4 provides 9km, 3-hourly global soil moisture estimates.
"""

import os
import re
import netrc
import requests
from datetime import datetime
import logging
from pathlib import Path
import numpy as np
import pandas as pd

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

# SMAP started April 2015
ANALYSIS_START = '2015-04-01'
ANALYSIS_END = '2017-12-31'

# NSIDC DAAC URLs
# SMAP L3 Enhanced: SPL3SMP_E (daily, 9km)
# SMAP L4 Global: SPL4SMGP (3-hourly, 9km)
NSIDC_BASE = "https://n5eil01u.ecs.nsidc.org"
SPL3SMP_E_PATH = "/SMAP/SPL3SMP_E.006"  # Enhanced L3 daily

# GES DISC for L4
GES_DISC_BASE = "https://hydro1.gesdisc.eosdis.nasa.gov/data"
SPL4SMGP_PATH = "/SMAP_L4/SPL4SMGP.008"  # L4 Global 3-hourly


def get_earthdata_session():
    """Create authenticated session for Earthdata."""
    session = requests.Session()

    # Try .netrc
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            session.auth = (auth[0], auth[2])
            return session
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    # Try environment
    user = os.environ.get('EARTHDATA_USERNAME')
    password = os.environ.get('EARTHDATA_PASSWORD')
    if user and password:
        session.auth = (user, password)
        return session

    raise ValueError("No Earthdata credentials found")


def list_nsidc_files(session, date):
    """List available SMAP files for a given date."""
    year = date.year
    month = date.month
    day = date.day

    # Try L3 Enhanced (SPL3SMP_E)
    url = f"{NSIDC_BASE}{SPL3SMP_E_PATH}/{year}.{month:02d}.{day:02d}/"

    try:
        response = session.get(url, timeout=30)
        if response.status_code == 200:
            # Parse HTML for .h5 files
            files = re.findall(r'href="([^"]+\.h5)"', response.text)
            return [f"{url}{f}" for f in files if f.endswith('.h5')]
        elif response.status_code == 404:
            logger.debug(f"No data for {date.strftime('%Y-%m-%d')}")
            return []
        else:
            logger.warning(f"HTTP {response.status_code} for {url}")
            return []
    except Exception as e:
        logger.warning(f"Error listing {url}: {e}")
        return []


def download_smap_file(session, url, output_dir):
    """Download a single SMAP file."""
    filename = url.split('/')[-1]
    output_path = output_dir / filename

    if output_path.exists():
        logger.debug(f"Skipping {filename} (exists)")
        return output_path

    logger.info(f"Downloading {filename}...")
    try:
        response = session.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)

        return output_path
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return None


def try_alternative_sources(session):
    """Try alternative SMAP data sources."""

    # Try PODAAC (Physical Oceanography DAAC)
    podaac_base = "https://opendap.earthdata.nasa.gov/collections"

    # Try subset via Earthdata Harmony
    harmony_base = "https://harmony.earthdata.nasa.gov"

    # Try LP DAAC (Land Processes)
    lpdaac_base = "https://e4ftl01.cr.usgs.gov/SMAP"

    sources = [
        (f"{lpdaac_base}/SPL3SMP_E.006/", "LP DAAC SPL3SMP_E"),
        (f"{lpdaac_base}/SPL4SMGP.008/", "LP DAAC SPL4SMGP"),
    ]

    for url, name in sources:
        logger.info(f"Trying {name}...")
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                logger.info(f"  {name} accessible!")
                return url, name
            else:
                logger.info(f"  {name}: HTTP {response.status_code}")
        except Exception as e:
            logger.info(f"  {name}: {e}")

    return None, None


def process_smap_h5(h5_files, bbox):
    """Process SMAP HDF5 files to extract catchment mean."""
    import h5py

    results = []

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, 'r') as f:
                # Navigate to soil moisture data
                # SPL3SMP_E structure
                if 'Soil_Moisture_Retrieval_Data_AM' in f:
                    group = f['Soil_Moisture_Retrieval_Data_AM']
                    sm = group['soil_moisture'][:]
                    lat = group['latitude'][:]
                    lon = group['longitude'][:]

                    # Create mask for bbox
                    mask = (
                        (lat >= bbox['lat_min']) & (lat <= bbox['lat_max']) &
                        (lon >= bbox['lon_min']) & (lon <= bbox['lon_max']) &
                        (sm != -9999.0)  # NoData
                    )

                    if np.any(mask):
                        sm_mean = np.mean(sm[mask])

                        # Extract date from filename
                        date_match = re.search(r'_(\d{8})T', h5_path.name)
                        if date_match:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y%m%d')
                            results.append({
                                'date': date,
                                'soil_moisture': sm_mean
                            })

        except Exception as e:
            logger.warning(f"Error processing {h5_path.name}: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        df.set_index('date', inplace=True)
        df = df.sort_index()
        return df

    return None


def main():
    logger.info("=" * 60)
    logger.info("SMAP Soil Moisture Direct Acquisition")
    logger.info("=" * 60)
    logger.info(f"Period: {ANALYSIS_START} to {ANALYSIS_END}")
    logger.info("")

    session = get_earthdata_session()
    logger.info("Earthdata session created")

    # Try alternative sources first
    alt_url, alt_name = try_alternative_sources(session)

    if alt_url:
        logger.info(f"\nUsing {alt_name}")
        # Continue with download from that source
    else:
        logger.info("\nTrying direct NSIDC access...")

    # Sample a few dates to test access
    test_dates = [
        datetime(2016, 7, 15),
        datetime(2017, 1, 15),
        datetime(2017, 7, 15),
    ]

    downloaded = 0
    for date in test_dates:
        files = list_nsidc_files(session, date)
        if files:
            logger.info(f"Found {len(files)} files for {date.strftime('%Y-%m-%d')}")
            for url in files[:1]:  # Just download one per day for testing
                result = download_smap_file(session, url, OUTPUT_DIR)
                if result:
                    downloaded += 1

    if downloaded > 0:
        logger.info(f"\nDownloaded {downloaded} test files")
        logger.info("Full download would proceed month by month...")

        # Process any downloaded files
        h5_files = list(OUTPUT_DIR.glob("*.h5"))
        if h5_files:
            logger.info(f"\nProcessing {len(h5_files)} files...")
            df = process_smap_h5(h5_files, BBOX)
            if df is not None:
                output_csv = OUTPUT_DIR / "smap_processed.csv"
                df.to_csv(output_csv)
                logger.info(f"Processed data saved to {output_csv}")
    else:
        logger.warning("No files could be downloaded via direct access")
        logger.info("Waiting for AppEEARS task to complete...")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
