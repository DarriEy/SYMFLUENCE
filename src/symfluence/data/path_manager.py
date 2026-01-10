"""
PathManager - Centralized path resolution for SYMFLUENCE data pipeline.

This module provides a single source of truth for constructing file paths
based on configuration settings. It eliminates the duplicated path resolution
logic that was previously scattered across acquisition, preprocessing, and
observation modules.

Usage:
    from symfluence.data.path_manager import PathManager

    paths = PathManager(config)

    # Get project directory
    project_dir = paths.project_dir

    # Resolve paths with 'default' handling
    catchment_path = paths.resolve('CATCHMENT_PATH', 'shapefiles/catchment')
    dem_path = paths.resolve('DEM_PATH', 'attributes/elevation/dem', 'dem.tif')

    # Get standard subdirectories
    forcing_dir = paths.forcing_dir
    observations_dir = paths.observations_dir
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union


from symfluence.core import ConfigurableMixin


class PathManager(ConfigurableMixin):
    """
    Centralized path resolution for SYMFLUENCE data pipeline.

    Handles the common pattern of:
    1. Constructing project_dir from SYMFLUENCE_DATA_DIR and DOMAIN_NAME
    2. Resolving paths that can be either 'default' (use project_dir) or custom
    3. Providing standardized access to common subdirectories
    """

    def __init__(self, config: 'SymfluenceConfig'):
        """
        Initialize PathManager with configuration.

        Args:
            config: SymfluenceConfig instance
        """
        from symfluence.core.config.models import SymfluenceConfig
        if not isinstance(config, SymfluenceConfig):
            raise TypeError(
                f"config must be SymfluenceConfig, got {type(config).__name__}. "
                "Use SymfluenceConfig.from_file() to load configuration."
            )
        self._config = config

    # Standard subdirectories are now provided by ProjectContextMixin:
    # project_shapefiles_dir, project_attributes_dir, project_forcing_dir, 
    # project_observations_dir, project_simulations_dir, project_settings_dir, project_cache_dir

    @property
    def shapefiles_dir(self) -> Path:
        """Directory for shapefiles."""
        return self.project_shapefiles_dir

    @property
    def attributes_dir(self) -> Path:
        """Directory for attributes."""
        return self.project_attributes_dir

    @property
    def forcing_dir(self) -> Path:
        """Directory for forcing."""
        return self.project_forcing_dir

    @property
    def observations_dir(self) -> Path:
        """Directory for observations."""
        return self.project_observations_dir

    @property
    def simulations_dir(self) -> Path:
        """Directory for simulations."""
        return self.project_simulations_dir

    @property
    def settings_dir(self) -> Path:
        """Directory for settings."""
        return self.project_settings_dir

    @property
    def cache_dir(self) -> Path:
        """Directory for cache."""
        return self.project_cache_dir

    @property
    def catchment_dir(self) -> Path:
        """Directory for catchment shapefiles."""
        return self.shapefiles_dir / 'catchment'

    @property
    def forcing_shapefile_dir(self) -> Path:
        """Directory for forcing grid shapefiles."""
        return self.shapefiles_dir / 'forcing'

    @property
    def raw_forcing_dir(self) -> Path:
        """Directory for raw forcing data."""
        return self.forcing_dir / 'raw_data'

    @property
    def merged_forcing_dir(self) -> Path:
        """Directory for merged/processed forcing data."""
        return self.forcing_dir / 'merged_data'

    @property
    def streamflow_dir(self) -> Path:
        """Directory for streamflow observations."""
        return self.observations_dir / 'streamflow'

    @property
    def dem_dir(self) -> Path:
        """Directory for DEM data."""
        return self.attributes_dir / 'elevation' / 'dem'

    # -------------------------------------------------------------------------
    # Path resolution methods
    # -------------------------------------------------------------------------

    def resolve(
        self,
        config_key: str,
        default_subpath: str,
        filename: Optional[str] = None
    ) -> Path:
        """
        Resolve a path based on config, falling back to default if 'default'.

        This delegates to the standardized path resolution logic.

        Args:
            config_key: Configuration key to check (e.g., 'CATCHMENT_PATH')
            default_subpath: Default path relative to project_dir
            filename: Optional filename to append to the path

        Returns:
            Resolved Path object
        """
        from symfluence.core.path_resolver import resolve_path
        
        base_path = resolve_path(
            config=self.config_dict,
            config_key=config_key,
            project_dir=self.project_dir,
            default_subpath=default_subpath,
            logger=self.logger
        )

        if filename:
            return base_path / filename
        return base_path

    def resolve_with_name(
        self,
        path_key: str,
        name_key: str,
        default_subpath: str,
        default_name_pattern: Optional[str] = None
    ) -> Path:
        """
        Resolve a path with associated name from config.

        Common pattern where both path and filename are configurable.

        Args:
            path_key: Config key for the path
            name_key: Config key for the filename
            default_subpath: Default path relative to project_dir
            default_name_pattern: Pattern for default name (can include {domain})

        Returns:
            Full resolved path including filename
        """
        from symfluence.core.path_resolver import resolve_file_path
        
        default_name = default_name_pattern.format(domain=self.domain_name) if default_name_pattern else 'default'
        
        return resolve_file_path(
            config=self.config_dict,
            project_dir=self.project_dir,
            path_key=path_key,
            name_key=name_key,
            default_subpath=default_subpath,
            default_name=default_name,
            logger=self.logger
        )

    def ensure_dir(self, path: Path) -> Path:
        """
        Ensure a directory exists (delegates to FileUtilsMixin).
        """
        return super().ensure_dir(path)

    def get_forcing_dataset_dir(self, dataset_name: str, raw: bool = True) -> Path:
        """
        Get the directory for a specific forcing dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'ERA5', 'CARRA')
            raw: If True, return raw_data subdir; if False, return merged_data

        Returns:
            Path to the dataset directory
        """
        base = self.raw_forcing_dir if raw else self.merged_forcing_dir
        return base / dataset_name.upper()


class PathManagerMixin:
    """
    Mixin class to add PathManager capabilities to existing classes.

    Usage:
        class MyProcessor(PathManagerMixin):
            def __init__(self, config, logger):
                self.config = config
                self.logger = logger
                self._init_path_manager()

            def process(self):
                # Use paths directly
                dem_path = self.paths.resolve('DEM_PATH', 'attributes/elevation/dem')
    """

    _paths: Optional[PathManager] = None

    def _init_path_manager(self) -> None:
        """Initialize the path manager from self.config."""
        if hasattr(self, 'config'):
            self._paths = PathManager(self.config)
        else:
            raise AttributeError("PathManagerMixin requires self.config to be set")

    @property
    def paths(self) -> PathManager:
        """Access the PathManager instance."""
        if self._paths is None:
            self._init_path_manager()
        return self._paths

    # Convenience properties that delegate to PathManager
    @property
    def project_dir(self) -> Path:
        """Project directory (delegates to PathManager)."""
        return self.paths.project_dir

    @property
    def domain_name(self) -> str:
        """Domain name (delegates to PathManager)."""
        return self.paths.domain_name


def create_path_manager(config: Dict[str, Any]) -> PathManager:
    """
    Factory function to create a PathManager instance.

    Args:
        config: Configuration dictionary

    Returns:
        Configured PathManager instance
    """
    return PathManager(config)
