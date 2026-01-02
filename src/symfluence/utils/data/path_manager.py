"""
PathManager - Centralized path resolution for SYMFLUENCE data pipeline.

This module provides a single source of truth for constructing file paths
based on configuration settings. It eliminates the duplicated path resolution
logic that was previously scattered across acquisition, preprocessing, and
observation modules.

Usage:
    from symfluence.utils.data.path_manager import PathManager

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


class PathManager:
    """
    Centralized path resolution for SYMFLUENCE data pipeline.

    Handles the common pattern of:
    1. Constructing project_dir from SYMFLUENCE_DATA_DIR and DOMAIN_NAME
    2. Resolving paths that can be either 'default' (use project_dir) or custom
    3. Providing standardized access to common subdirectories
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PathManager with configuration.

        Args:
            config: Configuration dictionary containing at minimum:
                - SYMFLUENCE_DATA_DIR: Base data directory
                - DOMAIN_NAME: Name of the domain/project
        """
        self._config = config
        self._data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
        self._domain_name = config.get('DOMAIN_NAME', 'domain')
        self._project_dir = self._data_dir / f"domain_{self._domain_name}"

    @property
    def data_dir(self) -> Path:
        """Base SYMFLUENCE data directory."""
        return self._data_dir

    @property
    def domain_name(self) -> str:
        """Name of the current domain."""
        return self._domain_name

    @property
    def project_dir(self) -> Path:
        """Project directory: {data_dir}/domain_{domain_name}"""
        return self._project_dir

    # -------------------------------------------------------------------------
    # Standard subdirectories
    # -------------------------------------------------------------------------

    @property
    def shapefiles_dir(self) -> Path:
        """Directory for shapefiles."""
        return self._project_dir / 'shapefiles'

    @property
    def catchment_dir(self) -> Path:
        """Directory for catchment shapefiles."""
        return self.shapefiles_dir / 'catchment'

    @property
    def forcing_shapefile_dir(self) -> Path:
        """Directory for forcing grid shapefiles."""
        return self.shapefiles_dir / 'forcing'

    @property
    def attributes_dir(self) -> Path:
        """Directory for catchment attributes."""
        return self._project_dir / 'attributes'

    @property
    def forcing_dir(self) -> Path:
        """Directory for forcing data."""
        return self._project_dir / 'forcing'

    @property
    def raw_forcing_dir(self) -> Path:
        """Directory for raw forcing data."""
        return self.forcing_dir / 'raw_data'

    @property
    def merged_forcing_dir(self) -> Path:
        """Directory for merged/processed forcing data."""
        return self.forcing_dir / 'merged_data'

    @property
    def observations_dir(self) -> Path:
        """Directory for observation data."""
        return self._project_dir / 'observations'

    @property
    def streamflow_dir(self) -> Path:
        """Directory for streamflow observations."""
        return self.observations_dir / 'streamflow'

    @property
    def dem_dir(self) -> Path:
        """Directory for DEM data."""
        return self.attributes_dir / 'elevation' / 'dem'

    @property
    def cache_dir(self) -> Path:
        """Directory for cached data."""
        return self._project_dir / 'cache'

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

        This replaces the duplicated _get_file_path pattern found across modules.

        Args:
            config_key: Configuration key to check (e.g., 'CATCHMENT_PATH')
            default_subpath: Default path relative to project_dir (e.g., 'shapefiles/catchment')
            filename: Optional filename to append to the path

        Returns:
            Resolved Path object

        Examples:
            # Config has CATCHMENT_PATH = 'default'
            >>> paths.resolve('CATCHMENT_PATH', 'shapefiles/catchment')
            Path('/data/domain_test/shapefiles/catchment')

            # Config has CATCHMENT_PATH = '/custom/path'
            >>> paths.resolve('CATCHMENT_PATH', 'shapefiles/catchment')
            Path('/custom/path')

            # With filename
            >>> paths.resolve('DEM_PATH', 'attributes/elevation/dem', 'dem.tif')
            Path('/data/domain_test/attributes/elevation/dem/dem.tif')
        """
        config_value = self._config.get(config_key, 'default')

        if config_value == 'default':
            base_path = self._project_dir / default_subpath
        else:
            base_path = Path(config_value)

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
            path_key: Config key for the path (e.g., 'CATCHMENT_PATH')
            name_key: Config key for the filename (e.g., 'CATCHMENT_SHP_NAME')
            default_subpath: Default path relative to project_dir
            default_name_pattern: Pattern for default name using domain_name
                                 (e.g., '{domain}_HRUs_GRUs.shp')

        Returns:
            Full resolved path including filename

        Example:
            >>> paths.resolve_with_name(
            ...     'CATCHMENT_PATH', 'CATCHMENT_SHP_NAME',
            ...     'shapefiles/catchment',
            ...     '{domain}_HRUs_GRUs.shp'
            ... )
        """
        base_path = self.resolve(path_key, default_subpath)

        name_value = self._config.get(name_key, 'default')
        if name_value == 'default' and default_name_pattern:
            filename = default_name_pattern.format(domain=self._domain_name)
        else:
            filename = name_value

        return base_path / filename

    def ensure_dir(self, path: Path) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path to ensure exists

        Returns:
            The same path (for chaining)
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

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
