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
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import geopandas as gpd

from symfluence.core.config.coercion import coerce_config
from symfluence.core.mixins import ConfigMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseAttributeProcessor(ConfigMixin):
    """Base class for all attribute processors."""

    def __init__(self, config: Union['SymfluenceConfig', Dict[str, Any]], logger: logging.Logger):
        """
        Initialize base attribute processor.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance
        """
        self._config = coerce_config(config, warn=False)
        self.logger = logger

        # Use ConfigMixin properties and methods with dict fallback
        self.data_dir = Path(self._get_config_value(
            lambda: self.config.system.data_dir,
            dict_key='SYMFLUENCE_DATA_DIR'
        ))
        self.logger.info(f'data dir: {self.data_dir}')

        # Get domain_name with dict fallback (set as instance attribute for subclass access)
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key='DOMAIN_NAME'
        )
        self.logger.info(f'domain name: {self.domain_name}')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Get the catchment shapefile
        self.catchment_path = self._get_catchment_path()

        # Initialize results dictionary
        self.results: Dict[str, Any] = {}

    def _get_catchment_path(self) -> Path:
        """
        Get the path to the catchment shapefile.

        Returns:
            Path to catchment shapefile
        """
        catchment_path = self._get_config_value(
            lambda: self.config.paths.catchment_path,
            dict_key='CATCHMENT_PATH'
        )
        self.logger.info(f'catchment path: {catchment_path}')

        catchment_name = self._get_config_value(
            lambda: self.config.paths.catchment_shp_name,
            dict_key='CATCHMENT_SHP_NAME'
        )
        self.logger.info(f'catchment name: {catchment_name}')

        if catchment_path == 'default':
            catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)

        if catchment_name == 'default':
            # Find the catchment shapefile based on domain discretization
            discretization = self._get_config_value(
                lambda: self.config.domain.discretization,
                dict_key='SUB_GRID_DISCRETIZATION'
            )
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
        path_value = self._get_config_value(lambda: None, dict_key=config_key)

        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subfolder

        return Path(path_value)

    def _is_lumped(self) -> bool:
        """
        Check if domain is lumped or distributed.

        Returns:
            True if lumped, False if distributed
        """
        return self._get_config_value(
            lambda: self.config.domain.definition_method,
            dict_key='DOMAIN_DEFINITION_METHOD'
        ) == 'lumped'

    def _get_hru_ids(self) -> Optional[list]:
        """
        Get list of HRU IDs from catchment shapefile (for distributed catchments).

        Returns:
            List of HRU IDs, or None for lumped catchments
        """
        if self._is_lumped():
            return None

        catchment = gpd.read_file(self.catchment_path)
        hru_id_field = self._get_config_value(
            lambda: self.config.paths.catchment_hruid,
            default='HRU_ID',
            dict_key='CATCHMENT_SHP_HRUID'
        )

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
