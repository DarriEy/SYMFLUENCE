"""
Acquire satellite data using SYMFLUENCE observation handlers.

Tests SMAP soil moisture and GLEAM ET handlers for Bow at Banff.
"""

import sys
import logging
from pathlib import Path

# Add SYMFLUENCE to path
sys.path.insert(0, "/Users/darrieythorsson/compHydro/code/SYMFLUENCE/src")

from symfluence.data.observation.registry import ObservationRegistry

# Setup paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")

# Minimal config dict for handlers
CONFIG = {
    'DOMAIN_NAME': 'Bow_at_Banff_multivar',
    'EXPERIMENT_TIME_START': '2004-01-01 00:00',
    'EXPERIMENT_TIME_END': '2017-12-31 23:00',
    'BOUNDING_BOX_COORDS': '51.73/-116.55/50.95/-115.53',  # lat_max/lon_min/lat_min/lon_max
    'DATA_DIR': str(DATA_DIR.parent),
    'DATA_ACCESS': 'cloud',  # Enable cloud acquisition
    'FORCE_DOWNLOAD': False,

    # SMAP config
    'SMAP_USE_OPENDAP': True,

    # GLEAM config - need a download URL
    # GLEAM data available from: https://www.gleam.eu/
    'GLEAM_ET_DOWNLOAD_URL': None,  # Set if you have access
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_available_handlers():
    """List all registered observation handlers."""
    logger.info("Available observation handlers:")
    # Access the registry to see what's registered
    from symfluence.data.observation.registry import ObservationRegistry

    # Get all registered handlers
    handlers = ObservationRegistry._handlers
    for name, handler_class in sorted(handlers.items()):
        logger.info(f"  {name}: {handler_class.__name__}")


def try_smap_handler():
    """Try to acquire SMAP data via the handler."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing SMAP Handler")
    logger.info("=" * 50)

    try:
        handler = ObservationRegistry.get_handler('smap', CONFIG, logger)
        logger.info(f"Handler: {handler.__class__.__name__}")

        # Try acquire
        logger.info("Calling acquire()...")
        result = handler.acquire()
        logger.info(f"Acquire result: {result}")

        # Check if any data was downloaded
        if result and result.exists():
            files = list(result.glob("*"))
            logger.info(f"Files in result directory: {len(files)}")
            for f in files[:5]:
                logger.info(f"  {f.name}")

    except Exception as e:
        logger.error(f"SMAP handler failed: {e}")
        import traceback
        traceback.print_exc()


def try_gleam_handler():
    """Try GLEAM ET handler."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing GLEAM Handler")
    logger.info("=" * 50)

    try:
        handler = ObservationRegistry.get_handler('gleam_et', CONFIG, logger)
        logger.info(f"Handler: {handler.__class__.__name__}")

        # GLEAM requires a download URL
        if not CONFIG.get('GLEAM_ET_DOWNLOAD_URL'):
            logger.warning("GLEAM_ET_DOWNLOAD_URL not set")
            logger.info("GLEAM data must be requested from: https://www.gleam.eu/")
            return

        result = handler.acquire()
        logger.info(f"Acquire result: {result}")

    except Exception as e:
        logger.error(f"GLEAM handler failed: {e}")


def check_existing_data():
    """Check what observation data already exists."""
    logger.info("\n" + "=" * 50)
    logger.info("Checking existing observation data")
    logger.info("=" * 50)

    obs_dir = DATA_DIR / "observations"

    for subdir in sorted(obs_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            files = list(subdir.rglob("*"))
            data_files = [f for f in files if f.is_file() and not f.name.startswith('.')]
            logger.info(f"  {subdir.name}: {len(data_files)} files")


def main():
    logger.info("=" * 60)
    logger.info("SYMFLUENCE Observation Handler Test")
    logger.info("=" * 60)

    # List available handlers
    list_available_handlers()

    # Check existing data
    check_existing_data()

    # Try SMAP
    try_smap_handler()

    # Try GLEAM (will fail without URL)
    try_gleam_handler()

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
