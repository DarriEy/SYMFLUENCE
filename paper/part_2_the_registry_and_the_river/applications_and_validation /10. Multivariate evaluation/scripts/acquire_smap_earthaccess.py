#!/usr/bin/env python3
"""
Acquire SMAP Soil Moisture using earthaccess (NASA official library).

This is the recommended way to access NASA Earthdata Cloud data.
"""

import earthaccess
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import h5py

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
BBOX = (
    -116.55,  # west
    50.95,    # south
    -115.53,  # east
    51.73     # north
)

# Period
ANALYSIS_START = '2015-04-01'
ANALYSIS_END = '2017-12-31'


def search_smap_granules():
    """Search for SMAP granules covering our domain and time period."""
    logger.info("Searching for SMAP granules...")

    # Login to Earthdata (uses .netrc)
    auth = earthaccess.login()
    if not auth:
        logger.error("Authentication failed")
        return []

    logger.info("Authenticated successfully")

    # Search for SMAP Enhanced L3 daily product
    results = earthaccess.search_data(
        short_name='SPL3SMP_E',
        version='006',
        temporal=(ANALYSIS_START, ANALYSIS_END),
        bounding_box=BBOX,
    )

    logger.info(f"Found {len(results)} granules")
    return results


def download_smap_granules(granules, max_files=None):
    """Download SMAP granules."""
    if max_files:
        granules = granules[:max_files]

    logger.info(f"Downloading {len(granules)} files...")

    # Download files
    files = earthaccess.download(
        granules,
        local_path=str(OUTPUT_DIR)
    )

    logger.info(f"Downloaded {len(files)} files")
    return files


def process_smap_h5(h5_path, bbox_dict):
    """Process a single SMAP H5 file to extract catchment mean."""
    try:
        with h5py.File(h5_path, 'r') as f:
            # SMAP Enhanced L3 structure
            group = f.get('Soil_Moisture_Retrieval_Data_AM')
            if group is None:
                group = f.get('Soil_Moisture_Retrieval_Data_PM')
            if group is None:
                return None

            sm = group['soil_moisture'][:]
            lat = group['latitude'][:]
            lon = group['longitude'][:]

            # Create mask for bbox
            mask = (
                (lat >= bbox_dict['lat_min']) & (lat <= bbox_dict['lat_max']) &
                (lon >= bbox_dict['lon_min']) & (lon <= bbox_dict['lon_max']) &
                (sm != -9999.0) &  # Fill value
                (sm > 0) & (sm < 1)  # Valid range
            )

            if np.any(mask):
                sm_mean = np.mean(sm[mask])

                # Extract date from filename
                import re
                date_match = re.search(r'_(\d{8})_', h5_path.name)
                if date_match:
                    date_str = date_match.group(1)
                    date = datetime.strptime(date_str, '%Y%m%d')
                    return {'date': date, 'soil_moisture': sm_mean}

    except Exception as e:
        logger.warning(f"Error processing {h5_path.name}: {e}")

    return None


def process_all_smap_files():
    """Process all downloaded SMAP files."""
    bbox_dict = {
        'lat_min': BBOX[1],
        'lat_max': BBOX[3],
        'lon_min': BBOX[0],
        'lon_max': BBOX[2]
    }

    h5_files = sorted(OUTPUT_DIR.glob("SMAP_L3*.h5"))
    logger.info(f"Processing {len(h5_files)} files...")

    results = []
    for h5_path in h5_files:
        result = process_smap_h5(h5_path, bbox_dict)
        if result:
            results.append(result)

    if not results:
        logger.warning("No data could be extracted")
        return None

    df = pd.DataFrame(results)
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Remove duplicates (keep first)
    df = df[~df.index.duplicated(keep='first')]

    return df


def main():
    logger.info("=" * 60)
    logger.info("SMAP Soil Moisture via earthaccess")
    logger.info("=" * 60)
    logger.info(f"Period: {ANALYSIS_START} to {ANALYSIS_END}")
    logger.info(f"Bbox: {BBOX}")
    logger.info("")

    # Search for granules
    granules = search_smap_granules()

    if not granules:
        logger.error("No granules found")
        return

    # Download (limit to sample for testing)
    logger.info("\nDownloading sample files (first 30 days)...")
    sample_granules = granules[:30]

    try:
        files = download_smap_granules(sample_granules)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return

    # Process downloaded files
    logger.info("\nProcessing downloaded files...")
    df = process_all_smap_files()

    if df is not None and len(df) > 0:
        # Save processed data
        output_csv = OUTPUT_DIR / "smap_processed.csv"
        df.to_csv(output_csv)
        logger.info(f"\nSaved processed data to {output_csv}")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"  SM range: {df['soil_moisture'].min():.3f} - {df['soil_moisture'].max():.3f} m³/m³")
    else:
        logger.warning("No valid data extracted")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
