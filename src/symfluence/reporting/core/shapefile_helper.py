# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Centralized shapefile loading utilities.

This module provides a unified interface for loading and caching geospatial
shapefiles used throughout the reporting module.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.constants import ConfigKeys
from symfluence.core.mixins import ConfigMixin

# Lazy import for geopandas
_gpd = None


def _get_geopandas():
    """Lazy import of geopandas."""
    global _gpd
    if _gpd is None:
        import geopandas as gpd
        _gpd = gpd
    return _gpd


def resolve_default_name(
    config: Dict[str, Any],
    config_key: str,
    pattern: str
) -> str:
    """
    Resolve config value, substituting 'default' with pattern.

    Args:
        config: Configuration dictionary
        config_key: Key to look up
        pattern: Format string with {domain} and {method} placeholders

    Returns:
        Resolved name string

    Example:
        >>> config = {'RIVER_BASINS_NAME': 'default', 'DOMAIN_NAME': 'test', 'DOMAIN_DEFINITION_METHOD': 'lumped'}
        >>> resolve_default_name(config, 'RIVER_BASINS_NAME', '{domain}_riverBasins_{method}.shp')
        'test_riverBasins_lumped.shp'
    """
    value = config.get(config_key, 'default')
    if value == 'default':
        value = pattern.format(
            domain=config.get(ConfigKeys.DOMAIN_NAME, 'domain'),
            method=config.get(ConfigKeys.DOMAIN_DEFINITION_METHOD, 'lumped'),
            discretization=config.get(ConfigKeys.SUB_GRID_DISCRETIZATION, 'GRUs')
        )
    return value


class ShapefileHelper(ConfigMixin):
    """
    Helper for loading and caching shapefiles.

    Provides centralized shapefile loading with caching to avoid
    repeated file I/O operations across different plotters and processors.

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        project_dir: Path to project directory
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        project_dir: Optional[Path] = None
    ):
        """
        Initialize the ShapefileHelper.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Optional project directory override
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger

        if project_dir is not None:
            self.project_dir = project_dir
        else:
            data_dir = self._get_config_value(
                lambda: self.config.system.data_dir,
                default='.',
                dict_key=ConfigKeys.SYMFLUENCE_DATA_DIR
            )
            domain_name = self._get_config_value(
                lambda: self.config.domain.name,
                default='unknown',
                dict_key=ConfigKeys.DOMAIN_NAME
            )
            self.project_dir = Path(data_dir) / f"domain_{domain_name}"

        self._cache: Dict[str, Any] = {}

    def clear_cache(self) -> None:
        """Clear the shapefile cache."""
        self._cache.clear()
        self.logger.debug("Shapefile cache cleared")

    def get_catchment_shapefile(self) -> Optional[Any]:
        """
        Load catchment/river basin shapefile with caching.

        Returns:
            GeoDataFrame of catchment boundaries, or None if not found
        """
        cache_key = 'catchment'
        if cache_key in self._cache:
            return self._cache[cache_key]

        _get_geopandas()
        name = resolve_default_name(
            self.config_dict,
            'RIVER_BASINS_NAME',
            '{domain}_riverBasins_{method}.shp'
        )

        shapefile_path = self._resolve_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', name)
        gdf = self._load_shapefile(shapefile_path)

        if gdf is not None:
            self._cache[cache_key] = gdf

        return gdf

    def get_river_network_shapefile(self) -> Optional[Any]:
        """
        Load river network shapefile with caching.

        Returns:
            GeoDataFrame of river network, or None if not found
        """
        cache_key = 'river_network'
        if cache_key in self._cache:
            return self._cache[cache_key]

        name = resolve_default_name(
            self.config_dict,
            'RIVER_NETWORK_SHP_NAME',
            '{domain}_riverNetwork_delineate.shp'
        )

        shapefile_path = self._resolve_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network', name)
        gdf = self._load_shapefile(shapefile_path)

        if gdf is not None:
            self._cache[cache_key] = gdf

        return gdf

    def get_pour_point_shapefile(self) -> Optional[Any]:
        """
        Load pour point shapefile with caching.

        Returns:
            GeoDataFrame of pour point, or None if not found
        """
        cache_key = 'pour_point'
        if cache_key in self._cache:
            return self._cache[cache_key]

        name = resolve_default_name(
            self.config_dict,
            'POUR_POINT_SHP_NAME',
            '{domain}_pourPoint.shp'
        )

        shapefile_path = self._resolve_path('POUR_POINT_SHP_PATH', 'shapefiles/pour_point', name)
        gdf = self._load_shapefile(shapefile_path)

        if gdf is not None:
            self._cache[cache_key] = gdf

        return gdf

    def get_hru_shapefile(self, discretization_method: Optional[str] = None) -> Optional[Any]:
        """
        Load HRU shapefile with caching.

        Args:
            discretization_method: Discretization method (uses config default if None)

        Returns:
            GeoDataFrame of HRUs, or None if not found
        """
        method = discretization_method or self._get_config_value(lambda: self.config.domain.discretization, default='GRUs', dict_key=ConfigKeys.SUB_GRID_DISCRETIZATION)
        cache_key = f'hru_{method}'

        if cache_key in self._cache:
            return self._cache[cache_key]

        name = resolve_default_name(
            self.config_dict,
            'CATCHMENT_SHP_NAME',
            f'{{domain}}_HRUs_{method}.shp'
        )

        shapefile_path = self._resolve_path('CATCHMENT_PATH', 'shapefiles/catchment', name)
        gdf = self._load_shapefile(shapefile_path)

        if gdf is not None:
            self._cache[cache_key] = gdf

        return gdf

    def get_basin_area(self, area_column: Optional[str] = None) -> Optional[float]:
        """
        Get total basin area from river basin shapefile.

        Args:
            area_column: Column name containing area values (uses config default if None)

        Returns:
            Basin area in m², or None if not available
        """
        try:
            basin_gdf = self.get_catchment_shapefile()
            if basin_gdf is None:
                return None

            area_col = area_column or self._get_config_value(lambda: self.config.paths.river_basin_area, default='GRU_area', dict_key=ConfigKeys.RIVER_BASIN_SHP_AREA)

            if area_col not in basin_gdf.columns:
                self.logger.warning(f"Area column '{area_col}' not found in basin shapefile")
                return None

            area_m2 = float(basin_gdf[area_col].sum())
            return area_m2

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.warning(f"Error getting basin area: {str(e)}")
            return None

    def _resolve_path(
        self,
        path_key: str,
        default_subpath: str,
        filename: str
    ) -> Path:
        """
        Resolve file path from config or use default.

        Args:
            path_key: Config key for path
            default_subpath: Default subdirectory under project_dir
            filename: File name

        Returns:
            Resolved Path object
        """
        config_path = self._get_config_value(lambda: None, default=None, dict_key=path_key)

        if config_path is None or config_path == 'default':
            return self.project_dir / default_subpath / filename
        else:
            return Path(config_path)

    def _load_shapefile(self, path: Path) -> Optional[Any]:
        """
        Load shapefile from path.

        Args:
            path: Path to shapefile

        Returns:
            GeoDataFrame or None if loading fails
        """
        gpd = _get_geopandas()

        if not path.exists():
            self.logger.debug(f"Shapefile not found: {path}")
            return None

        try:
            gdf = gpd.read_file(path)
            self.logger.debug(f"Loaded shapefile: {path}")
            return gdf
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.warning(f"Error loading shapefile {path}: {str(e)}")
            return None
