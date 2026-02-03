#!/usr/bin/env python
"""
Test script for CanSWE data pipeline for Bow at Banff domain.

This script tests the CanSWE observation handler by:
1. Loading the config file
2. Initializing the CanSWE handler
3. Downloading/locating the CanSWE data
4. Processing to extract stations within the domain
5. Reporting the results

Usage:
    python test_canswe_pipeline.py
"""

import sys
import logging
from pathlib import Path

# Add SYMFLUENCE to path
SYMFLUENCE_CODE = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE / "src"))

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data.observation.handlers.canswe import CanSWEHandler, NorSWEHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_canswe_pipeline():
    """Test the CanSWE data pipeline."""

    # Load config
    config_path = Path(__file__).parent / "configs" / "bow_grace_tws" / "config_Bow_SUMMA_calibrated_multivar.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    logger.info(f"Loading config from: {config_path}")
    config = SymfluenceConfig.from_file(config_path)

    # Extract key info - use to_dict() for backward compatible access
    config_dict = config.to_dict()
    domain_name = config_dict.get('DOMAIN_NAME')
    bbox = config_dict.get('BOUNDING_BOX_COORDS')
    start_date = config_dict.get('EXPERIMENT_TIME_START')
    end_date = config_dict.get('EXPERIMENT_TIME_END')

    logger.info(f"Domain: {domain_name}")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Time period: {start_date} to {end_date}")

    # Initialize CanSWE handler
    logger.info("Initializing CanSWE handler...")
    handler = CanSWEHandler(config, logger)

    # Acquire data
    logger.info("Acquiring CanSWE data...")
    try:
        raw_path = handler.acquire()
        logger.info(f"Raw data acquired: {raw_path}")
    except Exception as e:
        logger.error(f"Failed to acquire CanSWE data: {e}")
        return False

    # Process data
    logger.info("Processing CanSWE data for domain...")
    try:
        processed_path = handler.process(raw_path)
        logger.info(f"Processed data saved: {processed_path}")
    except Exception as e:
        logger.error(f"Failed to process CanSWE data: {e}")
        return False

    # Report results
    import pandas as pd
    df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')

    logger.info("\n" + "="*60)
    logger.info("CanSWE Data Summary for Bow at Banff")
    logger.info("="*60)
    logger.info(f"Total observations: {len(df)}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info("SWE statistics (mm):")
    logger.info(f"  Mean: {df['swe_mm'].mean():.2f}")
    logger.info(f"  Max:  {df['swe_mm'].max():.2f}")
    logger.info(f"  Min:  {df['swe_mm'].min():.2f}")
    if 'n_stations' in df.columns:
        logger.info(f"  Stations per day: {df['n_stations'].mean():.1f} (mean)")

    # Check overlap with experiment period
    overlap_start = max(df.index.min(), pd.Timestamp(start_date))
    overlap_end = min(df.index.max(), pd.Timestamp(end_date))
    df_overlap = df[(df.index >= overlap_start) & (df.index <= overlap_end)]

    logger.info(f"\nOverlap with experiment period ({start_date} to {end_date}):")
    logger.info(f"  Observations in overlap: {len(df_overlap)}")
    if len(df_overlap) > 0:
        logger.info(f"  Overlap date range: {df_overlap.index.min()} to {df_overlap.index.max()}")

    # Check station metadata
    station_file = processed_path.parent / f"{domain_name}_canswe_stations.csv"
    if station_file.exists():
        stations_df = pd.read_csv(station_file)
        logger.info("\nStation metadata:")
        logger.info(f"  Number of stations in domain: {len(stations_df)}")
        if 'lat' in stations_df.columns and 'lon' in stations_df.columns:
            logger.info(f"  Latitude range: {stations_df['lat'].min():.4f} to {stations_df['lat'].max():.4f}")
            logger.info(f"  Longitude range: {stations_df['lon'].min():.4f} to {stations_df['lon'].max():.4f}")
        if 'elevation_m' in stations_df.columns:
            logger.info(f"  Elevation range: {stations_df['elevation_m'].min():.0f} to {stations_df['elevation_m'].max():.0f} m")

    logger.info("\n" + "="*60)
    logger.info("CanSWE pipeline test PASSED")
    logger.info("="*60)

    return True


def test_norswe_pipeline():
    """Test the NorSWE data pipeline (optional - larger download)."""

    config_path = Path(__file__).parent / "configs" / "bow_grace_tws" / "config_Bow_SUMMA_calibrated_multivar.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    logger.info("Loading config for NorSWE test...")
    config = SymfluenceConfig.from_file(config_path)

    # Override to use NorSWE
    config_dict = config.to_dict()
    config_dict['DOWNLOAD_NORSWE'] = True

    logger.info("Initializing NorSWE handler...")
    handler = NorSWEHandler(config_dict, logger)

    try:
        raw_path = handler.acquire()
        logger.info(f"NorSWE raw data acquired: {raw_path}")

        processed_path = handler.process(raw_path)
        logger.info(f"NorSWE processed data saved: {processed_path}")

        logger.info("NorSWE pipeline test PASSED")
        return True

    except Exception as e:
        logger.error(f"NorSWE pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CanSWE/NorSWE data pipeline")
    parser.add_argument("--norswe", action="store_true", help="Also test NorSWE (larger download)")
    args = parser.parse_args()

    success = test_canswe_pipeline()

    if args.norswe:
        success = success and test_norswe_pipeline()

    sys.exit(0 if success else 1)
