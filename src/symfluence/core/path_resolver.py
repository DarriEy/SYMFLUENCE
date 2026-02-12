"""
Path resolution utilities for SYMFLUENCE configuration management.

Provides standardized path resolution with default fallback handling to eliminate
code duplication across model preprocessors, managers, and utilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging


def resolve_path(
    config: Dict[str, Any],
    config_key: str,
    project_dir: Path,
    default_subpath: str,
    logger: Optional[logging.Logger] = None,
    must_exist: bool = False
) -> Path:
    """
    Resolve a path from config or use default relative to project directory.

    This function handles the common pattern of checking a config key for a path value,
    with special handling for 'default' keyword and None values which trigger use of
    a default path relative to the project directory.

    Args:
        config: Configuration dictionary
        config_key: Key to look up in configuration (e.g., 'FORCING_PATH')
        project_dir: Project base directory (typically data_dir/domain_{domain_name})
        default_subpath: Default path relative to project_dir (e.g., 'forcing/merged_data')
        logger: Optional logger for debug messages
        must_exist: If True, raise FileNotFoundError if resolved path doesn't exist

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist

    Examples:
        >>> config = {'FORCING_PATH': 'default'}
        >>> resolve_path(config, 'FORCING_PATH', Path('/data/domain_test'), 'forcing/merged_data')
        Path('/data/domain_test/forcing/merged_data')

        >>> config = {'FORCING_PATH': '/custom/forcing/path'}
        >>> resolve_path(config, 'FORCING_PATH', Path('/data/domain_test'), 'forcing/merged_data')
        Path('/custom/forcing/path')
    """
    path_value = config.get(config_key)

    # Handle 'default' or None values -> use default path relative to project_dir
    if path_value == 'default' or path_value is None:
        resolved = project_dir / default_subpath
        if logger:
            logger.debug(f"Using default path for {config_key}: {resolved}")
    else:
        resolved = Path(path_value)
        if logger:
            logger.debug(f"Using configured path for {config_key}: {resolved}")

    # Validate existence if required
    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Required path does not exist: {resolved} (config_key: {config_key})"
        )

    return resolved


def resolve_file_path(
    config: Dict[str, Any],
    project_dir: Path,
    path_key: str,
    name_key: str,
    default_subpath: str,
    default_name: str,
    logger: Optional[logging.Logger] = None,
    must_exist: bool = False
) -> Path:
    """
    Resolve complete file path from separate directory and filename config keys.

    This function handles the common pattern where a file path is specified via two
    config keys: one for the directory and one for the filename. Both support the
    'default' keyword for fallback values.

    Args:
        config: Configuration dictionary
        project_dir: Project base directory
        path_key: Config key for directory path (e.g., 'DEM_PATH')
        name_key: Config key for file name (e.g., 'DEM_NAME')
        default_subpath: Default directory relative to project_dir
        default_name: Default file name
        logger: Optional logger for debug messages
        must_exist: If True, raise FileNotFoundError if file doesn't exist

    Returns:
        Complete file path (Path object)

    Raises:
        FileNotFoundError: If must_exist=True and file doesn't exist

    Examples:
        >>> config = {'DEM_PATH': 'default', 'DEM_NAME': 'default'}
        >>> resolve_file_path(
        ...     config, Path('/data/domain_test'), 'DEM_PATH', 'DEM_NAME',
        ...     'attributes/elevation/dem', 'domain_test_elv.tif'
        ... )
        Path('/data/domain_test/attributes/elevation/dem/domain_test_elv.tif')

        >>> config = {'DEM_PATH': '/custom/dem', 'DEM_NAME': 'my_dem.tif'}
        >>> resolve_file_path(
        ...     config, Path('/data/domain_test'), 'DEM_PATH', 'DEM_NAME',
        ...     'attributes/elevation/dem', 'domain_test_elv.tif'
        ... )
        Path('/custom/dem/my_dem.tif')
    """
    # Get directory path using resolve_path
    dir_path = resolve_path(
        config=config,
        config_key=path_key,
        project_dir=project_dir,
        default_subpath=default_subpath,
        logger=logger,
        must_exist=False  # Check file existence below, not directory
    )

    # Get filename (handle 'default' or None)
    file_name = config.get(name_key)
    if file_name == 'default' or file_name is None:
        file_name = default_name

    # Construct complete file path
    file_path = dir_path / file_name

    # Validate file existence if required
    if must_exist and not file_path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {file_path} "
            f"(path_key: {path_key}, name_key: {name_key})"
        )

    return file_path


from .mixins import ConfigurableMixin


class PathResolverMixin(ConfigurableMixin):
    """
    Mixin providing path resolution methods for classes with config and project_dir.

    This mixin eliminates duplicate `_get_default_path()` implementations across
    model preprocessors, managers, and utilities by providing a standardized interface.

    Usage:
        class MyPreprocessor(BaseModelPreProcessor, PathResolverMixin):
            def __init__(self, config, logger):
                super().__init__(config, logger)

                # Use the mixin's methods
                self.forcing_path = self._get_default_path(
                    'FORCING_PATH', 'forcing/merged_data'
                )

    Examples:
        >>> class TestClass(PathResolverMixin):
        ...     def __init__(self):
        ...         self.config = {'MY_PATH': 'default', 'DOMAIN_NAME': 'test'}
        ...
        >>> obj = TestClass()
        >>> obj._get_default_path('MY_PATH', 'some/path')
        Path('./domain_test/some/path')
    """

    def _get_default_path(
        self,
        config_key: str,
        default_subpath: str,
        must_exist: bool = False
    ) -> Path:
        """
        Resolve path from config or use default (instance method).

        This method provides the standard interface that existing code expects,
        delegating to the standalone resolve_path() function.

        Args:
            config_key: Configuration key to lookup
            default_subpath: Default path relative to project_dir
            must_exist: If True, raise FileNotFoundError if path doesn't exist

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If must_exist=True and path doesn't exist
        """
        return resolve_path(
            config=self.config_dict,
            config_key=config_key,
            project_dir=self.project_dir,
            default_subpath=default_subpath,
            logger=self.logger,
            must_exist=must_exist
        )

    def resolve(
        self,
        config_key: str,
        default_subpath: str,
        filename: Optional[str] = None,
        must_exist: bool = False
    ) -> Path:
        """
        Resolve a path from config, with optional filename (alias for _get_default_path).

        Matches the PathManager.resolve() API for easier migration and consistency.
        """
        path = self._get_default_path(config_key, default_subpath, must_exist=must_exist)
        if filename:
            return path / filename
        return path

    def _get_file_path(
        self,
        path_key: str,
        name_key: str,
        default_subpath: str,
        default_name: str,
        must_exist: bool = False
    ) -> Path:
        """
        Resolve complete file path from directory and name keys (instance method).

        This method provides a convenient interface for resolving file paths that
        are specified via separate directory and filename config keys.

        Args:
            path_key: Config key for directory path
            name_key: Config key for file name
            default_subpath: Default directory relative to project_dir
            default_name: Default file name
            must_exist: If True, raise FileNotFoundError if file doesn't exist

        Returns:
            Complete file path (Path object)

        Raises:
            FileNotFoundError: If must_exist=True and file doesn't exist
        """
        return resolve_file_path(
            config=self.config_dict,
            project_dir=self.project_dir,
            path_key=path_key,
            name_key=name_key,
            default_subpath=default_subpath,
            default_name=default_name,
            logger=self.logger,
            must_exist=must_exist
        )

    def _get_catchment_file_path(
        self,
        catchment_name: Optional[str] = None
    ) -> Path:
        """
        Resolve catchment shapefile path with backward compatibility.

        Checks for existing file in the legacy location first, then returns
        the new organized path if not found.

        The new path structure includes domain_definition_method and experiment_id:
            shapefiles/catchment/{domain_definition_method}/{experiment_id}/

        For backward compatibility:
            - If file exists at old path (shapefiles/catchment/), use that
            - Otherwise use new organized path

        Args:
            catchment_name: Optional catchment shapefile name. If None,
                           constructs default name from domain and discretization.

        Returns:
            Full path to the catchment shapefile
        """
        # Construct catchment name if not provided
        if catchment_name is None or catchment_name == 'default':
            catchment_name = self.config_dict.get('CATCHMENT_SHP_NAME')

        if catchment_name is None or catchment_name == 'default':
            discretization = self.domain_discretization
            # Handle comma-separated attributes for output filename
            if "," in discretization:
                method_suffix = discretization.replace(",", "_")
            else:
                method_suffix = discretization
            catchment_name = f"{self.domain_name}_HRUs_{method_suffix}.shp"

        # Check old path first for backward compatibility
        old_path = self.project_dir / "shapefiles" / "catchment" / catchment_name
        if old_path.exists():
            return old_path

        # Return new organized path
        new_path = (
            self.project_dir / "shapefiles" / "catchment" /
            self.domain_definition_method / self.experiment_id / catchment_name
        )
        return new_path

    def _get_method_suffix(self) -> str:
        """
        Get delineation suffix encoding all relevant configuration.

        Returns unique suffixes based on definition method and associated options:

        | Method           | Subset | Grid Source | Suffix                              |
        |------------------|--------|-------------|-------------------------------------|
        | point            | —      | —           | point                               |
        | lumped           | false  | —           | lumped                              |
        | lumped           | true   | —           | lumped_subset_{geofabric}           |
        | semidistributed  | false  | —           | semidistributed                     |
        | semidistributed  | true   | —           | semidistributed_subset_{geofabric}  |
        | distributed      | false  | generate    | distributed_{cellsize}m             |
        | distributed      | true   | —           | distributed_subset_{geofabric}      |
        | distributed      | —      | native      | distributed_native_{dataset}        |
        """
        cfg = self.config
        if cfg is None:
            return 'semidistributed'

        # Handle both typed config objects and plain dicts
        if isinstance(cfg, dict):
            method = cfg.get('DOMAIN_DEFINITION_METHOD', cfg.get('DEFINITION_METHOD', 'semidistributed'))
            subset = cfg.get('SUBSET_FROM_GEOFABRIC', False)
            grid_source = cfg.get('GRID_SOURCE', 'generate')
            geofabric_type = cfg.get('GEOFABRIC_TYPE', 'na')
            grid_cell_size = int(cfg.get('GRID_CELL_SIZE', 1000))
            native_dataset = cfg.get('NATIVE_GRID_DATASET', 'era5')
        else:
            method = cfg.domain.definition_method or 'semidistributed'
            subset = getattr(cfg.domain, 'subset_from_geofabric', False)
            grid_source = getattr(cfg.domain, 'grid_source', 'generate')
            geofabric_type = cfg.domain.delineation.geofabric_type or 'na'
            grid_cell_size = int(cfg.domain.grid_cell_size)
            native_dataset = getattr(cfg.domain, 'native_grid_dataset', 'era5')

        # Point method - simple suffix
        if method == 'point':
            return 'point'

        # Lumped method
        if method == 'lumped':
            if subset:
                return f"lumped_subset_{geofabric_type}"
            return 'lumped'

        # Semi-distributed method
        if method == 'semidistributed':
            if subset:
                return f"semidistributed_subset_{geofabric_type}"
            return 'semidistributed'

        # Distributed method
        if method == 'distributed':
            if subset:
                return f"distributed_subset_{geofabric_type}"
            if grid_source == 'native':
                return f"distributed_native_{native_dataset}"
            return f"distributed_{grid_cell_size}m"

        return method
