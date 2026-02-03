#!/usr/bin/env python
"""
Test script for ESA CCI SM and ET data acquisition for Bow at Banff domain.

Acquires:
1. ESA CCI Soil Moisture (long-term satellite record)
2. MODIS ET (evapotranspiration)
3. GLEAM ET (global land evaporation)
"""

import sys
import logging
from pathlib import Path

# Add SYMFLUENCE to path
SYMFLUENCE_CODE = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE / "src"))

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data.observation.handlers.soil_moisture import ESACCISMHandler
from symfluence.data.observation.handlers.modis_et import MODISETHandler
from symfluence.data.observation.handlers.gleam import GLEAMETHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_esa_cci_sm(config):
    """Test ESA CCI Soil Moisture acquisition."""
    logger.info("=" * 60)
    logger.info("ESA CCI Soil Moisture Acquisition")
    logger.info("=" * 60)

    try:
        handler = ESACCISMHandler(config, logger)

        bbox = handler.bbox
        logger.info(f"Bounding box: lat [{bbox['lat_min']:.2f}, {bbox['lat_max']:.2f}], "
                   f"lon [{bbox['lon_min']:.2f}, {bbox['lon_max']:.2f}]")

        # Acquire
        logger.info("Acquiring ESA CCI SM data...")
        raw_path = handler.acquire()
        logger.info(f"ESA CCI SM raw data: {raw_path}")

        if raw_path.exists():
            files = list(raw_path.glob("*.nc")) + list(raw_path.glob("*.csv"))
            if files:
                logger.info(f"ESA CCI SM files found: {len(files)}")

                # Process
                logger.info("Processing ESA CCI SM data...")
                processed_path = handler.process(raw_path)
                logger.info(f"ESA CCI SM processed: {processed_path}")
                logger.info("ESA CCI SM acquisition: SUCCESS")
                return True
            else:
                logger.warning("No ESA CCI SM files found")
                return None
        return None

    except Exception as e:
        logger.error(f"ESA CCI SM acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modis_et(config):
    """Test MODIS ET acquisition."""
    logger.info("=" * 60)
    logger.info("MODIS ET Acquisition")
    logger.info("=" * 60)

    try:
        handler = MODISETHandler(config, logger)

        bbox = handler.bbox
        logger.info(f"Bounding box: lat [{bbox['lat_min']:.2f}, {bbox['lat_max']:.2f}], "
                   f"lon [{bbox['lon_min']:.2f}, {bbox['lon_max']:.2f}]")

        # Acquire
        logger.info("Acquiring MODIS ET data...")
        raw_path = handler.acquire()
        logger.info(f"MODIS ET raw data: {raw_path}")

        if raw_path.exists():
            files = list(raw_path.glob("*.nc")) + list(raw_path.glob("*.csv")) + list(raw_path.glob("*.hdf"))
            if files:
                logger.info(f"MODIS ET files found: {len(files)}")

                # Process
                logger.info("Processing MODIS ET data...")
                processed_path = handler.process(raw_path)
                logger.info(f"MODIS ET processed: {processed_path}")
                logger.info("MODIS ET acquisition: SUCCESS")
                return True
            else:
                logger.warning("No MODIS ET files found - may need to trigger download")
                # Try to trigger cloud acquisition
                logger.info("Attempting cloud acquisition for MODIS ET...")
                return None
        return None

    except Exception as e:
        logger.error(f"MODIS ET acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gleam_et(config):
    """Test GLEAM ET acquisition."""
    logger.info("=" * 60)
    logger.info("GLEAM ET Acquisition")
    logger.info("=" * 60)

    try:
        handler = GLEAMETHandler(config, logger)

        bbox = handler.bbox
        logger.info(f"Bounding box: lat [{bbox['lat_min']:.2f}, {bbox['lat_max']:.2f}], "
                   f"lon [{bbox['lon_min']:.2f}, {bbox['lon_max']:.2f}]")

        # Acquire
        logger.info("Acquiring GLEAM ET data...")
        raw_path = handler.acquire()
        logger.info(f"GLEAM ET raw data: {raw_path}")

        if raw_path.exists():
            files = list(raw_path.glob("*.nc")) + list(raw_path.glob("*.csv"))
            if files:
                logger.info(f"GLEAM ET files found: {len(files)}")

                # Process
                logger.info("Processing GLEAM ET data...")
                processed_path = handler.process(raw_path)
                logger.info(f"GLEAM ET processed: {processed_path}")
                logger.info("GLEAM ET acquisition: SUCCESS")
                return True
            else:
                logger.warning("No GLEAM ET files found")
                return None
        return None

    except Exception as e:
        logger.error(f"GLEAM ET acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run ESA SM and ET acquisitions."""

    # Load config
    config_path = Path(__file__).parent / "configs" / "bow_grace_tws" / "config_Bow_SUMMA_calibrated_multivar.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    logger.info(f"Loading config from: {config_path}")
    config = SymfluenceConfig.from_file(config_path)

    config_dict = config.to_dict()
    domain_name = config_dict.get('DOMAIN_NAME')
    logger.info(f"Domain: {domain_name}")
    logger.info("")

    results = {}

    # Run acquisitions
    results['ESA_CCI_SM'] = test_esa_cci_sm(config)
    logger.info("")

    results['MODIS_ET'] = test_modis_et(config)
    logger.info("")

    results['GLEAM_ET'] = test_gleam_et(config)
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
            logger.info(f"  {source}: NO DATA / NEEDS AUTH")

    # List new observation files
    obs_dir = Path(config_dict.get('SYMFLUENCE_DATA_DIR')) / f"domain_{domain_name}" / "observations"
    if obs_dir.exists():
        logger.info("")
        logger.info("All observation files:")
        for subdir in sorted(obs_dir.iterdir()):
            if subdir.is_dir():
                files = list(subdir.rglob("*"))
                data_files = [f for f in files if f.is_file() and not f.name.startswith('.')]
                if data_files:
                    total_size = sum(f.stat().st_size for f in data_files) / 1024 / 1024
                    logger.info(f"  {subdir.name}/: {len(data_files)} files ({total_size:.1f} MB)")

    return all(r is not False for r in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
