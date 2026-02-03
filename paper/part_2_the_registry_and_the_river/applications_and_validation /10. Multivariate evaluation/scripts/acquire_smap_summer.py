#!/usr/bin/env python3
"""
Acquire SMAP Soil Moisture for summer months (snow-free period).

For the Bow at Banff catchment, meaningful soil moisture comparison
requires snow-free periods (typically June-September).
"""

import earthaccess
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import h5py
import re

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


def download_summer_smap(year):
    """Download SMAP for summer months of a given year."""
    logger.info(f"Downloading SMAP for summer {year}...")

    # Summer months: June-September
    start_date = f'{year}-06-01'
    end_date = f'{year}-09-30'

    auth = earthaccess.login()
    if not auth:
        logger.error("Authentication failed")
        return []

    # Search for granules
    results = earthaccess.search_data(
        short_name='SPL3SMP_E',
        version='006',
        temporal=(start_date, end_date),
        bounding_box=BBOX,
    )

    logger.info(f"  Found {len(results)} granules for summer {year}")

    if not results:
        return []

    # Download
    files = earthaccess.download(
        results,
        local_path=str(OUTPUT_DIR)
    )

    logger.info(f"  Downloaded {len(files)} files")
    return files


def process_smap_h5(h5_path, bbox_dict):
    """Process a single SMAP H5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            group = f.get('Soil_Moisture_Retrieval_Data_AM')
            if group is None:
                group = f.get('Soil_Moisture_Retrieval_Data_PM')
            if group is None:
                return None

            sm = group['soil_moisture'][:]
            lat = group['latitude'][:]
            lon = group['longitude'][:]

            mask = (
                (lat >= bbox_dict['lat_min']) & (lat <= bbox_dict['lat_max']) &
                (lon >= bbox_dict['lon_min']) & (lon <= bbox_dict['lon_max']) &
                (sm != -9999.0) & (sm > 0) & (sm < 1)
            )

            if np.any(mask):
                sm_mean = np.mean(sm[mask])
                date_match = re.search(r'_(\d{8})_', h5_path.name)
                if date_match:
                    date_str = date_match.group(1)
                    date = datetime.strptime(date_str, '%Y%m%d')
                    return {'date': date, 'soil_moisture': sm_mean}
    except Exception as e:
        logger.warning(f"Error processing {h5_path.name}: {e}")
    return None


def process_all_smap():
    """Process all SMAP files."""
    bbox_dict = {'lat_min': BBOX[1], 'lat_max': BBOX[3],
                 'lon_min': BBOX[0], 'lon_max': BBOX[2]}

    h5_files = sorted(OUTPUT_DIR.glob("SMAP_L3*.h5"))
    logger.info(f"Processing {len(h5_files)} total files...")

    results = []
    for h5_path in h5_files:
        result = process_smap_h5(h5_path, bbox_dict)
        if result:
            results.append(result)

    if not results:
        return None

    df = pd.DataFrame(results)
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    return df


def main():
    logger.info("=" * 60)
    logger.info("SMAP Summer Data Acquisition")
    logger.info("=" * 60)
    logger.info("Target: Snow-free periods (June-September)")
    logger.info("")

    # Download summer data for 2015, 2016, 2017
    for year in [2015, 2016, 2017]:
        try:
            download_summer_smap(year)
        except Exception as e:
            logger.error(f"Failed for {year}: {e}")

    # Process all downloaded files
    logger.info("\nProcessing all downloaded files...")
    df = process_all_smap()

    if df is not None and len(df) > 0:
        output_csv = OUTPUT_DIR / "smap_processed.csv"
        df.to_csv(output_csv)
        logger.info(f"\nSaved to {output_csv}")
        logger.info(f"  Total records: {len(df)}")
        logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

        # Summary by month
        logger.info("\nMonthly summary:")
        monthly = df.groupby(df.index.month)['soil_moisture'].agg(['count', 'mean'])
        for month, row in monthly.iterrows():
            logger.info(f"  Month {month}: {int(row['count'])} days, mean={row['mean']:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
