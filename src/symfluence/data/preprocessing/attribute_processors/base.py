"""
Base class for attribute processors.

Provides shared infrastructure for all attribute processing modules including:
- Configuration management
- Path resolution
- Catchment shapefile access
- Common utilities
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import geopandas as gpd


class BaseAttributeProcessor:
    """Base class for all attribute processors."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize base attribute processor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.logger.info(f'data dir: {self.data_dir}')
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.logger.info(f'domain name: {self.domain_name}')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Get the catchment shapefile
        self.catchment_path = self._get_catchment_path()

        # Initialize results dictionary
        self.results = {}

    def _get_catchment_path(self) -> Path:
        """
        Get the path to the catchment shapefile.

        Returns:
            Path to catchment shapefile
        """
        catchment_path = self.config.get('CATCHMENT_PATH')
        self.logger.info(f'catchment path: {catchment_path}')

        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        self.logger.info(f'catchment name: {catchment_name}')

        if catchment_path == 'default':
            catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)

        if catchment_name == 'default':
            # Find the catchment shapefile based on domain discretization
            discretization = self.config.get('DOMAIN_DISCRETIZATION')
            catchment_file = f"{self.domain_name}_HRUs_{discretization}.shp"
        else:
            catchment_file = catchment_name

        return catchment_path / catchment_file

    def _get_data_path(self, config_key: str, default_subfolder: str) -> Path:
        """
        Resolve a data path from config with default fallback.

        Args:
            config_key: Configuration key for the path
            default_subfolder: Default subfolder under project_dir

        Returns:
            Resolved path
        """
        path_value = self.config.get(config_key)

        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subfolder

        return Path(path_value)

    def _is_lumped(self) -> bool:
        """
        Check if domain is lumped or distributed.

        Returns:
            True if lumped, False if distributed
        """
        return self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'

    def _get_hru_ids(self) -> Optional[list]:
        """
        Get list of HRU IDs from catchment shapefile (for distributed catchments).

        Returns:
            List of HRU IDs, or None for lumped catchments
        """
        if self._is_lumped():
            return None

        catchment = gpd.read_file(self.catchment_path)
        hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')

        return catchment[hru_id_field].tolist()

    def _format_results_for_hrus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results with HRU prefixes if needed.

        Args:
            results: Raw results dictionary

        Returns:
            Formatted results (with HRU_ prefixes for distributed catchments)
        """
        if self._is_lumped():
            return results

        # For distributed catchments, results are already formatted by
        # individual processors with HRU prefixes
        return results
