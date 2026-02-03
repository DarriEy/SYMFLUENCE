#!/usr/bin/env python
"""
Test script for multivariate observation data acquisition for Bow at Banff domain.

Acquires:
1. GRACE TWS (Total Water Storage) anomaly data
2. SMAP soil moisture data
3. ISMN in-situ soil moisture stations (if any in domain)
4. GGMN groundwater stations (if any in domain)

Usage:
    python test_multivar_acquisitions.py
"""

import sys
import logging
from pathlib import Path

# Add SYMFLUENCE to path
SYMFLUENCE_CODE = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE / "src"))

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data.observation.handlers.grace import GRACEHandler
from symfluence.data.observation.handlers.soil_moisture import SMAPHandler, ISMNHandler
from symfluence.data.observation.handlers.ggmn import GGMNHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_grace_acquisition(config):
    """Test GRACE TWS acquisition."""
    logger.info("=" * 60)
    logger.info("GRACE TWS Acquisition")
    logger.info("=" * 60)

    try:
        handler = GRACEHandler(config, logger)

        # Acquire
        logger.info("Acquiring GRACE data...")
        raw_path = handler.acquire()
        logger.info(f"GRACE raw data: {raw_path}")

        # Process
        logger.info("Processing GRACE data for domain...")
        processed_path = handler.process(raw_path)
        logger.info(f"GRACE processed: {processed_path}")

        # Check output
        obs_dir = handler.project_dir / "observations" / "grace"
        if obs_dir.exists():
            files = list(obs_dir.rglob("*.csv")) + list(obs_dir.rglob("*.nc"))
            logger.info(f"GRACE output files: {len(files)}")
            for f in files[:5]:
                logger.info(f"  - {f.name}")

        logger.info("GRACE acquisition: SUCCESS")
        return True

    except Exception as e:
        logger.error(f"GRACE acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smap_acquisition(config):
    """Test SMAP soil moisture acquisition."""
    logger.info("=" * 60)
    logger.info("SMAP Soil Moisture Acquisition")
    logger.info("=" * 60)

    try:
        handler = SMAPHandler(config, logger)

        # Acquire
        logger.info("Acquiring SMAP data...")
        raw_path = handler.acquire()
        logger.info(f"SMAP raw data: {raw_path}")

        # Check if data was acquired
        if raw_path.exists():
            files = list(raw_path.glob("*.nc")) + list(raw_path.glob("*.h5"))
            if files:
                logger.info(f"SMAP files found: {len(files)}")

                # Process
                logger.info("Processing SMAP data for domain...")
                processed_path = handler.process(raw_path)
                logger.info(f"SMAP processed: {processed_path}")
                logger.info("SMAP acquisition: SUCCESS")
                return True
            else:
                logger.warning("No SMAP files found - may need Earthdata credentials")
                logger.info("SMAP acquisition: SKIPPED (no data)")
                return None
        else:
            logger.warning(f"SMAP directory does not exist: {raw_path}")
            return None

    except Exception as e:
        logger.error(f"SMAP acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ismn_stations(config):
    """Check for ISMN in-situ soil moisture stations in domain."""
    logger.info("=" * 60)
    logger.info("ISMN In-Situ Soil Moisture Stations")
    logger.info("=" * 60)

    try:
        handler = ISMNHandler(config, logger)

        # Get bounding box info
        bbox = handler.bbox
        logger.info(f"Searching ISMN stations in bbox: lat [{bbox['lat_min']:.2f}, {bbox['lat_max']:.2f}], "
                   f"lon [{bbox['lon_min']:.2f}, {bbox['lon_max']:.2f}]")

        # Acquire (this will trigger cloud acquisition if data_access is 'cloud')
        logger.info("Checking for ISMN data...")
        raw_path = handler.acquire()

        if raw_path.exists():
            files = list(raw_path.glob("*.csv")) + list(raw_path.glob("*.txt"))
            if files:
                logger.info(f"ISMN files found: {len(files)}")
                logger.info("ISMN acquisition: SUCCESS")
                return True
            else:
                logger.info("No ISMN stations found in domain")
                logger.info("ISMN acquisition: NO DATA IN DOMAIN")
                return None
        else:
            logger.info("ISMN directory not created - no stations in domain")
            return None

    except Exception as e:
        logger.error(f"ISMN check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ggmn_stations(config):
    """Check for GGMN groundwater stations in domain."""
    logger.info("=" * 60)
    logger.info("GGMN Groundwater Stations")
    logger.info("=" * 60)

    try:
        handler = GGMNHandler(config, logger)

        # Get bounding box info
        bbox = handler.bbox
        logger.info(f"Searching GGMN stations in bbox: lat [{bbox['lat_min']:.2f}, {bbox['lat_max']:.2f}], "
                   f"lon [{bbox['lon_min']:.2f}, {bbox['lon_max']:.2f}]")

        # Acquire (queries IGRAC WFS)
        logger.info("Querying GGMN WFS for groundwater wells...")
        raw_path = handler.acquire()

        if raw_path.exists() and raw_path.stat().st_size > 100:
            import json
            with open(raw_path) as f:
                data = json.load(f)
            features = data.get('features', [])
            if features:
                logger.info(f"GGMN stations found: {len(features)}")
                for feat in features[:5]:
                    props = feat.get('properties', {})
                    logger.info(f"  - {props.get('name', 'Unknown')} (ID: {props.get('id')})")
                if len(features) > 5:
                    logger.info(f"  ... and {len(features) - 5} more")
                logger.info("GGMN acquisition: SUCCESS")
                return True
            else:
                logger.info("No GGMN stations found in domain")
                return None
        else:
            logger.info("No GGMN stations with data in domain")
            return None

    except Exception as e:
        logger.error(f"GGMN check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all multivariate data acquisitions."""

    # Load config
    config_path = Path(__file__).parent / "configs" / "bow_grace_tws" / "config_Bow_SUMMA_calibrated_multivar.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    logger.info(f"Loading config from: {config_path}")
    config = SymfluenceConfig.from_file(config_path)

    # Extract key info
    config_dict = config.to_dict()
    domain_name = config_dict.get('DOMAIN_NAME')
    bbox = config_dict.get('BOUNDING_BOX_COORDS')
    start_date = config_dict.get('EXPERIMENT_TIME_START')
    end_date = config_dict.get('EXPERIMENT_TIME_END')

    logger.info(f"Domain: {domain_name}")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Time period: {start_date} to {end_date}")
    logger.info("")

    results = {}

    # Run acquisitions
    results['GRACE'] = test_grace_acquisition(config)
    logger.info("")

    results['SMAP'] = test_smap_acquisition(config)
    logger.info("")

    results['ISMN'] = test_ismn_stations(config)
    logger.info("")

    results['GGMN'] = test_ggmn_stations(config)
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for source, status in results.items():
        if status is True:
            logger.info(f"  {source}: SUCCESS")
        elif status is False:
            logger.info(f"  {source}: FAILED")
        else:
            logger.info(f"  {source}: NO DATA IN DOMAIN")

    # List all observation files
    obs_dir = Path(config_dict.get('SYMFLUENCE_DATA_DIR')) / f"domain_{domain_name}" / "observations"
    if obs_dir.exists():
        logger.info("")
        logger.info("Observation files created:")
        for f in sorted(obs_dir.rglob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                logger.info(f"  {f.relative_to(obs_dir)} ({size_kb:.1f} KB)")

    return all(r is not False for r in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
