#!/usr/bin/env python3
"""
Acquire SMAP L4 Soil Moisture via GES DISC with proper Earthdata authentication.

This handles the URS OAuth redirect flow properly.
"""

import os
import re
import netrc
import requests
from requests.auth import HTTPBasicAuth
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

# Period
ANALYSIS_START = datetime(2015, 4, 1)
ANALYSIS_END = datetime(2017, 12, 31)

# GES DISC data URLs
GES_DISC_DATA = "https://hydro1.gesdisc.eosdis.nasa.gov/data/SMAP_L4/SPL4SMGP.008"


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


def create_earthdata_session():
    """Create a session that handles Earthdata authentication redirects."""
    user, password = get_earthdata_credentials()

    session = requests.Session()

    # Pre-authenticate with URS
    urs_url = "https://urs.earthdata.nasa.gov"
    auth_url = f"{urs_url}/oauth/authorize"

    # Set up session to handle redirects and cookies
    session.auth = HTTPBasicAuth(user, password)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; SYMFLUENCE/1.0)'
    })

    # First, try to get a cookie by hitting URS
    try:
        response = session.get(urs_url, timeout=30)
        logger.debug(f"URS initial response: {response.status_code}")
    except Exception as e:
        logger.warning(f"Initial URS request failed: {e}")

    return session


def get_file_list_for_day(session, date):
    """Get list of SMAP L4 files for a specific day."""
    year = date.year
    doy = date.timetuple().tm_yday

    day_url = f"{GES_DISC_DATA}/{year}/{doy:03d}/"

    try:
        response = session.get(day_url, timeout=60, allow_redirects=True)

        # Check if we got HTML (directory listing) vs login page
        if response.status_code == 200 and 'html' in response.headers.get('content-type', '').lower():
            # Parse for .h5 files
            files = re.findall(r'href="([^"]+\.h5)"', response.text)
            if files:
                return [f"{day_url}{f}" for f in files]
            else:
                # Check if it's a login page
                if 'earthdata' in response.text.lower() and 'login' in response.text.lower():
                    logger.warning(f"Authentication redirect detected for {date}")
                    return []
        elif response.status_code == 401:
            logger.warning(f"401 Unauthorized for {date}")
            return []
        elif response.status_code == 404:
            logger.debug(f"No data for {date}")
            return []

    except Exception as e:
        logger.warning(f"Error accessing {date}: {e}")

    return []


def download_h5_file(session, url, output_dir):
    """Download a single HDF5 file."""
    filename = url.split('/')[-1]
    output_path = output_dir / filename

    if output_path.exists():
        logger.debug(f"Skipping {filename} (exists)")
        return output_path

    logger.info(f"Downloading {filename}...")

    try:
        response = session.get(url, stream=True, timeout=300, allow_redirects=True)

        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'html' in content_type.lower():
            logger.warning(f"Got HTML instead of HDF5 for {filename} - auth issue")
            return None

        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            return output_path
        else:
            logger.warning(f"HTTP {response.status_code} for {filename}")
            return None

    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return None


def process_smap_h5_files(h5_dir, bbox):
    """Process all SMAP HDF5 files to extract catchment mean."""
    import h5py

    h5_files = sorted(h5_dir.glob("SMAP_L4*.h5"))
    if not h5_files:
        logger.warning("No SMAP H5 files found")
        return None

    logger.info(f"Processing {len(h5_files)} files...")

    results = []
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, 'r') as f:
                # SMAP L4 structure
                geo_data = f.get('Geophysical_Data')
                if geo_data is None:
                    continue

                sm_surface = geo_data.get('sm_surface')
                sm_rootzone = geo_data.get('sm_rootzone')

                if sm_surface is None:
                    continue

                # Get coordinates
                lat = f['cell_lat'][:]
                lon = f['cell_lon'][:]

                # Mask for bbox
                mask = (
                    (lat >= bbox['lat_min']) & (lat <= bbox['lat_max']) &
                    (lon >= bbox['lon_min']) & (lon <= bbox['lon_max'])
                )

                sm_surface_data = sm_surface[:]
                valid_mask = mask & (sm_surface_data != -9999.0)

                if np.any(valid_mask):
                    # Extract date from filename
                    date_match = re.search(r'_(\d{8})T', h5_path.name)
                    if date_match:
                        date_str = date_match.group(1)
                        date = datetime.strptime(date_str, '%Y%m%d')

                        row = {
                            'date': date,
                            'sm_surface': np.mean(sm_surface_data[valid_mask])
                        }

                        if sm_rootzone is not None:
                            sm_root_data = sm_rootzone[:]
                            root_valid = valid_mask & (sm_root_data != -9999.0)
                            if np.any(root_valid):
                                row['sm_rootzone'] = np.mean(sm_root_data[root_valid])

                        results.append(row)

        except Exception as e:
            logger.warning(f"Error processing {h5_path.name}: {e}")
            continue

    if not results:
        return None

    df = pd.DataFrame(results)
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Daily mean if multiple timestamps per day
    df = df.groupby(df.index.date).mean()
    df.index = pd.to_datetime(df.index)

    return df


def main():
    logger.info("=" * 60)
    logger.info("SMAP L4 via GES DISC Direct Download")
    logger.info("=" * 60)
    logger.info(f"Period: {ANALYSIS_START.strftime('%Y-%m-%d')} to {ANALYSIS_END.strftime('%Y-%m-%d')}")
    logger.info("")

    session = create_earthdata_session()
    logger.info("Session created")

    # Sample a few dates across the period
    sample_dates = [
        datetime(2015, 7, 15),
        datetime(2016, 1, 15),
        datetime(2016, 7, 15),
        datetime(2017, 1, 15),
        datetime(2017, 7, 15),
    ]

    downloaded = 0
    for date in sample_dates:
        logger.info(f"\nChecking {date.strftime('%Y-%m-%d')}...")
        files = get_file_list_for_day(session, date)

        if files:
            logger.info(f"  Found {len(files)} files")
            # Download first file only for testing
            result = download_h5_file(session, files[0], OUTPUT_DIR)
            if result:
                downloaded += 1
        else:
            logger.info("  No files found")

    logger.info(f"\n\nDownloaded {downloaded} test files")

    if downloaded > 0:
        # Process downloaded files
        df = process_smap_h5_files(OUTPUT_DIR, BBOX)
        if df is not None:
            output_csv = OUTPUT_DIR / "smap_l4_processed.csv"
            df.to_csv(output_csv)
            logger.info(f"Processed data saved to {output_csv}")
            logger.info(f"  Records: {len(df)}")
            logger.info(f"  Surface SM range: {df['sm_surface'].min():.3f} - {df['sm_surface'].max():.3f}")
    else:
        logger.info("\nDirect download not working.")
        logger.info("Recommend waiting for AppEEARS task to complete.")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
