"""
Cloud Data Utilities for SYMFLUENCE
====================================

Registry-based access to cloud-hosted forcing datasets.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

from symfluence.utils.data.acquisition.registry import AcquisitionRegistry
import symfluence.utils.data.acquisition  # Trigger registration

class CloudForcingDownloader:
    """
    Main entry point for cloud data acquisition using the AcquisitionRegistry.
    """
    
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.dataset_name = config.get('FORCING_DATASET', '').upper()
        self.supplement_data = config.get('SUPPLEMENT_FORCING', False)

    def download_forcing_data(self, output_dir: Path) -> Path:
        """Download forcing data based on configured dataset."""
        self.logger.info(f"Starting cloud data acquisition for {self.dataset_name}")

        # Supplemental data handling (keeping original behavior)
        if self.supplement_data:
            self.logger.info('Supplementing data, downloading EM-Earth')
            em_handler = AcquisitionRegistry.get_handler('EM-EARTH', self.config, self.logger)
            em_handler.download(output_dir)

        # Main dataset download
        try:
            handler = AcquisitionRegistry.get_handler(self.dataset_name, self.config, self.logger)
            return handler.download(output_dir)
        except ValueError as e:
            self.logger.error(str(e))
            raise

    # Legacy methods kept for backward compatibility if needed by other components
    def download_soilgrids_soilclasses(self) -> Path:
        handler = AcquisitionRegistry.get_handler('SOILGRIDS', self.config, self.logger)
        return handler.download(Path(self.config.get('SYMFLUENCE_DATA_DIR')))

    def download_modis_landcover(self) -> Path:
        handler = AcquisitionRegistry.get_handler('MODIS_LANDCOVER', self.config, self.logger)
        return handler.download(Path(self.config.get('SYMFLUENCE_DATA_DIR')))

    def download_copernicus_dem(self) -> Path:
        handler = AcquisitionRegistry.get_handler('COPDEM30', self.config, self.logger)
        return handler.download(Path(self.config.get('SYMFLUENCE_DATA_DIR')))

def check_cloud_access_availability(dataset_name: str, logger) -> bool:
    """Check if a dataset is available for cloud access."""
    if AcquisitionRegistry.is_registered(dataset_name):
        logger.info(f"✓ {dataset_name} supports cloud data access")
        return True
    else:
        logger.warning(f"✗ {dataset_name} does not support cloud access.")
        return False

# Variable mappings (kept for legacy reasons or if needed by preprocessors)
def get_aorc_variable_mapping() -> Dict[str, str]:
    return {
        'APCP_surface': 'pptrate', 'TMP_2maboveground': 'airtemp', 'SPFH_2maboveground': 'spechum',
        'PRES_surface': 'airpres', 'DLWRF_surface': 'LWRadAtm', 'DSWRF_surface': 'SWRadAtm',
        'UGRD_10maboveground': 'wind_u', 'VGRD_10maboveground': 'wind_v'
    }

def get_era5_variable_mapping() -> Dict[str, str]:
    return {
        't2m': 'airtemp', 'u10': 'wind_u', 'v10': 'wind_v', 'sp': 'airpres',
        'd2m': 'dewpoint', 'q': 'spechum', 'tp': 'pptrate', 'ssrd': 'SWRadAtm', 'strd': 'LWRadAtm'
    }

def get_emearth_variable_mapping() -> Dict[str, str]:
    return {"prcp": "pptrate", "prcp_corrected": "pptrate", "tmean": "airtemp", "trange": "temp_range", "tdew": "dewpoint"}

def get_hrrr_variable_mapping() -> Dict[str, str]:
    return {
        'TMP': 'airtemp', 'SPFH': 'spechum', 'PRES': 'airpres', 'UGRD': 'wind_u',
        'VGRD': 'wind_v', 'DSWRF': 'SWRadAtm', 'DLWRF': 'LWRadAtm', 'APCP': 'pptrate'
    }

def get_conus404_variable_mapping() -> Dict[str, str]:
    return {
        'T2': 'airtemp', 'Q2': 'spechum', 'PSFC': 'airpres', 'U10': 'wind_u',
        'V10': 'wind_v', 'GLW': 'LWRadAtm', 'SWDOWN': 'SWRadAtm', 'RAINRATE': 'pptrate'
    }