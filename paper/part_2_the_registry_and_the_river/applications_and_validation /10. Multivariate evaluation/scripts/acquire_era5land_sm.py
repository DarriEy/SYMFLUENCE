#!/usr/bin/env python3
"""
Acquire ERA5-Land Soil Moisture via CDS API.

ERA5-Land provides soil moisture at ~9km resolution, comparable to SMAP.
Variables: volumetric soil water layer 1 (0-7cm), layer 2 (7-28cm), etc.

Note: Not satellite data but reanalysis - can still be useful for validation
as it assimilates surface observations.
"""

import cdsapi
import logging
from pathlib import Path
import xarray as xr
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
OUTPUT_DIR = DATA_DIR / "observations" / "soil_moisture" / "era5_land"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Domain bounds for Bow at Banff (slightly expanded for interpolation)
BBOX = {
    'lat_min': 50.9,
    'lat_max': 51.8,
    'lon_min': -116.6,
    'lon_max': -115.5
}

# Analysis period
ANALYSIS_START = '2004-01-01'
ANALYSIS_END = '2017-12-31'


def download_era5_land_sm(year):
    """Download ERA5-Land soil moisture for one year."""
    client = cdsapi.Client()

    out_file = OUTPUT_DIR / f"era5_land_sm_{year}.nc"
    if out_file.exists():
        logger.info(f"  {year}: Already downloaded")
        return out_file

    logger.info(f"  Downloading {year}...")

    # Request monthly means for efficiency
    client.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                'volumetric_soil_water_layer_1',  # 0-7 cm
                'volumetric_soil_water_layer_2',  # 7-28 cm
                'volumetric_soil_water_layer_3',  # 28-100 cm
                'volumetric_soil_water_layer_4',  # 100-289 cm
            ],
            'year': str(year),
            'month': [f'{m:02d}' for m in range(1, 13)],
            'time': '00:00',
            'area': [BBOX['lat_max'], BBOX['lon_min'], BBOX['lat_min'], BBOX['lon_max']],
            'format': 'netcdf',
        },
        str(out_file)
    )

    return out_file


def process_era5_land_sm():
    """Process downloaded ERA5-Land files into a time series."""
    logger.info("Processing ERA5-Land soil moisture...")

    nc_files = sorted(OUTPUT_DIR.glob("era5_land_sm_*.nc"))
    if not nc_files:
        logger.error("No ERA5-Land files found")
        return None

    all_data = []

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)

        # Get catchment center coordinates
        target_lat = (BBOX['lat_max'] + BBOX['lat_min']) / 2
        target_lon = (BBOX['lon_max'] + BBOX['lon_min']) / 2

        # Select nearest point
        ds_point = ds.sel(latitude=target_lat, longitude=target_lon, method='nearest')

        # Extract time series
        times = pd.to_datetime(ds_point.time.values)

        for i, t in enumerate(times):
            row = {'date': t}

            # Layer 1: 0-7 cm (surface)
            if 'swvl1' in ds_point:
                row['sm_0_7cm'] = float(ds_point['swvl1'].values[i])

            # Layer 2: 7-28 cm (root zone upper)
            if 'swvl2' in ds_point:
                row['sm_7_28cm'] = float(ds_point['swvl2'].values[i])

            # Layer 3: 28-100 cm (root zone)
            if 'swvl3' in ds_point:
                row['sm_28_100cm'] = float(ds_point['swvl3'].values[i])

            # Layer 4: 100-289 cm (deep)
            if 'swvl4' in ds_point:
                row['sm_100_289cm'] = float(ds_point['swvl4'].values[i])

            all_data.append(row)

        ds.close()

    df = pd.DataFrame(all_data)
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Filter to analysis period
    df = df[(df.index >= ANALYSIS_START) & (df.index <= ANALYSIS_END)]

    # Save processed data
    processed_file = OUTPUT_DIR / "era5_land_sm_processed.csv"
    df.to_csv(processed_file)

    logger.info(f"Processed {len(df)} months of ERA5-Land SM")
    logger.info(f"  Surface SM (0-7cm) mean: {df['sm_0_7cm'].mean():.3f} m³/m³")

    return df


def main():
    logger.info("=" * 60)
    logger.info("ERA5-Land Soil Moisture Acquisition for Bow at Banff")
    logger.info("=" * 60)
    logger.info(f"Period: {ANALYSIS_START} to {ANALYSIS_END}")
    logger.info("")

    # Download yearly files
    start_year = int(ANALYSIS_START[:4])
    end_year = int(ANALYSIS_END[:4])

    logger.info("Downloading ERA5-Land monthly soil moisture...")
    for year in range(start_year, end_year + 1):
        try:
            download_era5_land_sm(year)
        except Exception as e:
            logger.error(f"  {year}: Failed - {e}")

    # Process into time series
    df = process_era5_land_sm()

    if df is not None:
        logger.info("\n" + "=" * 60)
        logger.info("Download complete!")
        logger.info("=" * 60)
        logger.info(f"Data saved to: {OUTPUT_DIR}")

    return df


if __name__ == "__main__":
    main()
