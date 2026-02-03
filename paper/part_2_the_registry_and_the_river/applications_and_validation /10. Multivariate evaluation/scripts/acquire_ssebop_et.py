"""
Acquire SSEBop Global ET data for Bow at Banff catchment.

SSEBop: USGS operational Simplified Surface Energy Balance
- No authentication required
- Global monthly product at 10km resolution
- Available from 2000-present
"""

import logging
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import zipfile

# Setup paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
ET_DIR = DATA_DIR / "observations/et/ssebop"
ET_DIR.mkdir(parents=True, exist_ok=True)

# Domain parameters
BBOX = {
    "lat_min": 50.95,
    "lat_max": 51.73,
    "lon_min": -116.55,
    "lon_max": -115.53
}
START_DATE = datetime(2004, 1, 1)
END_DATE = datetime(2017, 12, 31)
DOMAIN_NAME = "Bow_at_Banff_multivar"

# Shapefile for catchment masking
SHAPEFILE = DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp"

# SSEBop URLs - global data is flat directory with 'm' prefix
GLOBAL_BASE = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/monthly/eta/downloads"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_ssebop_month(year, month, output_dir):
    """Download SSEBop global monthly ET for a specific month."""
    # URL pattern: /m{YYYYMM}.zip (flat directory, 'm' prefix)
    url = f"{GLOBAL_BASE}/m{year}{month:02d}.zip"

    zip_file = output_dir / f"m{year}{month:02d}.zip"
    tif_file = output_dir / f"m{year}{month:02d}.tif"

    # Check if already downloaded
    if tif_file.exists():
        return tif_file

    logger.info(f"Downloading SSEBop {year}-{month:02d}...")

    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 404:
            logger.warning(f"SSEBop data not available for {year}-{month:02d}")
            return None
        response.raise_for_status()

        # Save zip
        with open(zip_file, 'wb') as f:
            f.write(response.content)

        # Extract
        with zipfile.ZipFile(zip_file, 'r') as zf:
            # Find the .tif file in the archive
            tif_names = [n for n in zf.namelist() if n.endswith('.tif')]
            if tif_names:
                zf.extract(tif_names[0], output_dir)
                extracted = output_dir / tif_names[0]
                if extracted != tif_file:
                    extracted.rename(tif_file)

        # Cleanup zip
        zip_file.unlink()

        return tif_file

    except Exception as e:
        logger.warning(f"Failed to download {year}-{month:02d}: {e}")
        if zip_file.exists():
            zip_file.unlink()
        return None


def extract_catchment_mean(tif_file, gdf):
    """Extract mean ET for catchment from GeoTIFF."""
    try:
        with rasterio.open(tif_file) as src:
            # Ensure same CRS
            gdf_reproj = gdf.to_crs(src.crs)

            # Mask to catchment
            out_image, out_transform = mask(src, gdf_reproj.geometry, crop=True)

            # Get values (first band)
            data = out_image[0]

            # Mask no-data
            nodata = src.nodata if src.nodata is not None else -9999
            data = np.ma.masked_equal(data, nodata)
            data = np.ma.masked_less_equal(data, 0)  # Remove invalid

            if data.count() == 0:
                return np.nan

            # SSEBop is in mm/month, convert to mm/day
            mean_mm_month = data.mean()

            # Get month from filename (m200401)
            fname = tif_file.stem  # m200401
            year = int(fname[1:5])
            month = int(fname[5:7])
            days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month

            mean_mm_day = mean_mm_month / days_in_month

            return mean_mm_day

    except Exception as e:
        logger.warning(f"Error extracting from {tif_file}: {e}")
        return np.nan


def main():
    """Main acquisition workflow."""
    logger.info("=" * 60)
    logger.info("SSEBop Global ET Acquisition for Bow at Banff")
    logger.info("=" * 60)
    logger.info(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    logger.info("")

    # Check for existing processed data
    processed_file = ET_DIR / f"{DOMAIN_NAME}_ssebop_et_processed.csv"
    if processed_file.exists():
        logger.info(f"Processed ET data already exists: {processed_file}")
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        logger.info(f"  Period: {df.index.min()} to {df.index.max()}")
        logger.info(f"  N observations: {len(df)}")
        return

    # Load catchment shapefile
    if not SHAPEFILE.exists():
        logger.error(f"Shapefile not found: {SHAPEFILE}")
        return
    gdf = gpd.read_file(SHAPEFILE)

    # Download all months
    all_data = []

    for year in range(START_DATE.year, END_DATE.year + 1):
        start_month = START_DATE.month if year == START_DATE.year else 1
        end_month = END_DATE.month if year == END_DATE.year else 12

        for month in range(start_month, end_month + 1):
            tif_file = download_ssebop_month(year, month, ET_DIR)

            if tif_file and tif_file.exists():
                et_mm_day = extract_catchment_mean(tif_file, gdf)

                if not np.isnan(et_mm_day):
                    date = pd.Timestamp(year=year, month=month, day=15)  # Mid-month
                    all_data.append({'date': date, 'et_mm_day': et_mm_day})
                    logger.info(f"  {year}-{month:02d}: {et_mm_day:.2f} mm/day")

    if not all_data:
        logger.error("No ET data could be downloaded")
        return

    # Create DataFrame
    df = pd.DataFrame(all_data)
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Save
    df.to_csv(processed_file)
    logger.info(f"\nSaved: {processed_file}")

    # Summary
    logger.info("\nSSEBop ET Summary:")
    logger.info(f"  Period: {df.index.min().date()} to {df.index.max().date()}")
    logger.info(f"  N months: {len(df)}")
    logger.info(f"  Mean: {df['et_mm_day'].mean():.2f} mm/day")
    logger.info(f"  Max: {df['et_mm_day'].max():.2f} mm/day")
    logger.info(f"  Annual total: {df['et_mm_day'].mean()*365:.0f} mm/year")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
