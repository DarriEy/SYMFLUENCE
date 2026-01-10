"""
Core mixins for SYMFLUENCE modules.

Provides base mixins for logging, configuration, and project context that
other utilities and mixins can build upon.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, ContextManager, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class LoggingMixin:
    """
    Mixin providing standardized logger access.

    Ensures a logger is always available, defaulting to one named after the
    class if none is explicitly set.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        _logger = getattr(self, '_logger', None)
        if _logger is None:
            # Create a default logger if none exists
            module = self.__class__.__module__
            name = self.__class__.__name__
            self._logger = logging.getLogger(f"{module}.{name}")
            return self._logger
        return _logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        """Set the logger instance."""
        self._logger = value


class ConfigMixin:
    """
    Mixin for classes that use typed SymfluenceConfig configuration.

    Provides standardized access to configuration via the typed config object.
    The config_dict property provides a flattened dictionary view for legacy
    code that iterates over keys.
    """

    @property
    def config(self) -> 'SymfluenceConfig':
        """
        Get the typed configuration object.

        Returns:
            SymfluenceConfig instance
        """
        return getattr(self, '_config', None)

    @config.setter
    def config(self, value: 'SymfluenceConfig') -> None:
        """
        Set the typed configuration object.

        Args:
            value: SymfluenceConfig instance
        """
        self._config = value

    @property
    def config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a flattened dictionary.

        This property provides a flattened dict view of the typed config
        for code that needs to iterate over keys or use string-based access.
        The dict is cached internally by SymfluenceConfig for performance.

        Returns:
            Flattened configuration dictionary with uppercase keys
        """
        cfg = self.config
        if cfg is not None:
            # Handle both SymfluenceConfig and plain dict
            if isinstance(cfg, dict):
                return cfg
            return cfg.to_dict(flatten=True)
        return {}

    def _get_config_value(
        self,
        typed_accessor: Callable[[], Any],
        default: Any = None
    ) -> Any:
        """
        Get a configuration value from typed config with default fallback.

        Args:
            typed_accessor: Callable that accesses the typed config value,
                           e.g., lambda: self.config.domain.name
            default: Default value if accessor fails or returns None

        Returns:
            Configuration value or default

        Example:
            name = self._get_config_value(
                lambda: self.config.domain.name,
                default='unnamed'
            )
        """
        try:
            value = typed_accessor()
            return value if value is not None else default
        except (AttributeError, KeyError, TypeError):
            return default

    # =========================================================================
    # Convenience Properties for Common Config Values
    # =========================================================================

    @property
    def experiment_id(self) -> str:
        """Experiment identifier from config.domain.experiment_id."""
        return self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default='run_1'
        )

    @property
    def domain_definition_method(self) -> str:
        """Domain definition method from config.domain.definition_method."""
        return self._get_config_value(
            lambda: self.config.domain.definition_method,
            default='lumped'
        )

    @property
    def time_start(self) -> Optional[str]:
        """Experiment start time from config.domain.time_start."""
        return self._get_config_value(
            lambda: self.config.domain.time_start,
            default=None
        )

    @property
    def time_end(self) -> Optional[str]:
        """Experiment end time from config.domain.time_end."""
        return self._get_config_value(
            lambda: self.config.domain.time_end,
            default=None
        )

    @property
    def domain_discretization(self) -> str:
        """Domain discretization method from config.domain.discretization."""
        return self._get_config_value(
            lambda: self.config.domain.discretization,
            default='lumped'
        )

    @property
    def calibration_period(self) -> Optional[str]:
        """Calibration period string from config.domain.calibration_period."""
        _calibration_period = getattr(self, '_calibration_period', None)
        if _calibration_period is not None:
            return _calibration_period
        return self._get_config_value(
            lambda: self.config.domain.calibration_period,
            default=None
        )

    @calibration_period.setter
    def calibration_period(self, value) -> None:
        """Set the calibration period (can be string or tuple)."""
        self._calibration_period = value

    @property
    def spinup_period(self) -> Optional[str]:
        """Spinup period string from config.domain.spinup_period."""
        return self._get_config_value(
            lambda: self.config.domain.spinup_period,
            default=None
        )

    @property
    def evaluation_period(self) -> Optional[str]:
        """Evaluation period string from config.domain.evaluation_period."""
        _evaluation_period = getattr(self, '_evaluation_period', None)
        if _evaluation_period is not None:
            return _evaluation_period
        return self._get_config_value(
            lambda: self.config.domain.evaluation_period,
            default=None
        )

    @evaluation_period.setter
    def evaluation_period(self, value) -> None:
        """Set the evaluation period (can be string or tuple)."""
        self._evaluation_period = value

    @property
    def forcing_dataset(self) -> str:
        """Forcing dataset name from config.forcing.dataset."""
        # Allow override via _forcing_dataset_override attribute
        if hasattr(self, '_forcing_dataset_override') and self._forcing_dataset_override:
            return self._forcing_dataset_override.lower()
        return self._get_config_value(
            lambda: self.config.forcing.dataset,
            default=''
        ).lower()

    @forcing_dataset.setter
    def forcing_dataset(self, value: str) -> None:
        """Set forcing dataset override (for backward compatibility)."""
        self._forcing_dataset_override = value

    @property
    def forcing_time_step_size(self) -> int:
        """Forcing time step size in seconds from config.forcing.time_step_size."""
        return int(self._get_config_value(
            lambda: self.config.forcing.time_step_size,
            default=3600
        ))

    @property
    def hydrological_model(self) -> str:
        """Hydrological model name from config.model.hydrological_model."""
        model = self._get_config_value(
            lambda: self.config.model.hydrological_model,
            default=''
        )
        # Handle list case (multi-model)
        if isinstance(model, list):
            return model[0] if model else ''
        return model

    @property
    def routing_model(self) -> str:
        """Routing model name from config.model.routing_model."""
        return self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none'
        )

    @property
    def optimization_metric(self) -> str:
        """Optimization metric from config.optimization.metric."""
        return self._get_config_value(
            lambda: self.config.optimization.metric,
            default='KGE'
        )


class ShapefileAccessMixin(ConfigMixin):
    """
    Mixin providing standardized shapefile column name access.

    Provides properties for accessing shapefile column names from the typed
    config, with sensible defaults for common geofabric conventions.
    """

    # =========================================================================
    # Catchment Shapefile Columns
    # =========================================================================

    @property
    def catchment_name_col(self) -> str:
        """Name/ID column in catchment shapefile from config.paths.catchment_name."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_name,
            default='HRU_ID'
        )

    @property
    def catchment_hruid_col(self) -> str:
        """HRU ID column in catchment shapefile from config.paths.catchment_hruid."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_hruid,
            default='HRU_ID'
        )

    @property
    def catchment_gruid_col(self) -> str:
        """GRU ID column in catchment shapefile from config.paths.catchment_gruid."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_gruid,
            default='GRU_ID'
        )

    @property
    def catchment_area_col(self) -> str:
        """Area column in catchment shapefile from config.paths.catchment_area."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_area,
            default='HRU_area'
        )

    @property
    def catchment_lat_col(self) -> str:
        """Latitude column in catchment shapefile from config.paths.catchment_lat."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_lat,
            default='center_lat'
        )

    @property
    def catchment_lon_col(self) -> str:
        """Longitude column in catchment shapefile from config.paths.catchment_lon."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_lon,
            default='center_lon'
        )

    # =========================================================================
    # River Network Shapefile Columns
    # =========================================================================

    @property
    def river_network_name_col(self) -> str:
        """Name column in river network shapefile from config.paths.river_network_name."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_name,
            default='LINKNO'
        )

    @property
    def river_segid_col(self) -> str:
        """Segment ID column in river network from config.paths.river_network_segid."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_segid,
            default='LINKNO'
        )

    @property
    def river_downsegid_col(self) -> str:
        """Downstream segment ID column from config.paths.river_network_downsegid."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_downsegid,
            default='DSLINKNO'
        )

    @property
    def river_length_col(self) -> str:
        """Length column in river network from config.paths.river_network_length."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_length,
            default='Length'
        )

    @property
    def river_slope_col(self) -> str:
        """Slope column in river network from config.paths.river_network_slope."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_slope,
            default='Slope'
        )

    # =========================================================================
    # River Basin Shapefile Columns
    # =========================================================================

    @property
    def basin_name_col(self) -> str:
        """Name column in river basins shapefile from config.paths.river_basins_name."""
        return self._get_config_value(
            lambda: self.config.paths.river_basins_name,
            default='GRU_ID'
        )

    @property
    def basin_gruid_col(self) -> str:
        """GRU ID column in river basins from config.paths.river_basin_rm_gruid."""
        return self._get_config_value(
            lambda: self.config.paths.river_basin_rm_gruid,
            default='GRU_ID'
        )

    @property
    def basin_hru_to_seg_col(self) -> str:
        """HRU to segment mapping column from config.paths.river_basin_hru_to_seg."""
        return self._get_config_value(
            lambda: self.config.paths.river_basin_hru_to_seg,
            default='gru_to_seg'
        )

    @property
    def basin_area_col(self) -> str:
        """Area column in river basins from config.paths.river_basin_area."""
        return self._get_config_value(
            lambda: self.config.paths.river_basin_area,
            default='GRU_area'
        )


class ProjectContextMixin(ConfigMixin):
    """
    Mixin providing standard project context attributes.

    Extracts core project parameters (data_dir, domain_name, project_dir)
    from the typed SymfluenceConfig, providing a consistent interface across modules.
    """

    @property
    def data_dir(self) -> Path:
        """Root data directory from configuration."""
        _data_dir = getattr(self, '_data_dir', None)
        if _data_dir is not None:
            return Path(_data_dir)

        # Get from typed config
        cfg = self.config
        if cfg is not None:
            if isinstance(cfg, dict):
                return Path(cfg.get('SYMFLUENCE_DATA_DIR', '.'))
            return cfg.system.data_dir

        return Path('.')

    @data_dir.setter
    def data_dir(self, value: Union[str, Path]) -> None:
        """Set the data directory."""
        self._data_dir = Path(value)

    @data_dir.deleter
    def data_dir(self) -> None:
        """Delete the data directory override."""
        if hasattr(self, '_data_dir'):
            del self._data_dir

    @property
    def domain_name(self) -> str:
        """Domain name from configuration."""
        _domain_name = getattr(self, '_domain_name', None)
        if _domain_name is not None:
            return _domain_name

        # Get from typed config
        cfg = self.config
        if cfg is not None:
            if isinstance(cfg, dict):
                return cfg.get('DOMAIN_NAME', 'domain')
            return cfg.domain.name

        return 'domain'

    @domain_name.setter
    def domain_name(self, value: str) -> None:
        """Set the domain name."""
        self._domain_name = value

    @domain_name.deleter
    def domain_name(self) -> None:
        """Delete the domain name override."""
        if hasattr(self, '_domain_name'):
            del self._domain_name

    @property
    def project_dir(self) -> Path:
        """
        Resolved project directory: {data_dir}/domain_{domain_name}.
        """
        _project_dir = getattr(self, '_project_dir', None)
        if _project_dir is not None:
            return Path(_project_dir)

        return self.data_dir / f"domain_{self.domain_name}"

    @project_dir.setter
    def project_dir(self, value: Union[str, Path]) -> None:
        """Set the project directory."""
        self._project_dir = Path(value)

    @project_dir.deleter
    def project_dir(self) -> None:
        """Delete the project directory override."""
        if hasattr(self, '_project_dir'):
            del self._project_dir

    # Standard subdirectories (based on project convention)
    
    @property
    def project_shapefiles_dir(self) -> Path:
        """Directory for shapefiles: {project_dir}/shapefiles"""
        return self.project_dir / 'shapefiles'

    @property
    def project_attributes_dir(self) -> Path:
        """Directory for catchment attributes: {project_dir}/attributes"""
        return self.project_dir / 'attributes'

    @property
    def project_forcing_dir(self) -> Path:
        """Directory for forcing data: {project_dir}/forcing"""
        return self.project_dir / 'forcing'

    @property
    def project_observations_dir(self) -> Path:
        """Directory for observation data: {project_dir}/observations"""
        return self.project_dir / 'observations'

    @property
    def project_simulations_dir(self) -> Path:
        """Directory for model simulations: {project_dir}/simulations"""
        return self.project_dir / 'simulations'

    @property
    def project_settings_dir(self) -> Path:
        """Directory for model settings: {project_dir}/settings"""
        return self.project_dir / 'settings'

    @property
    def project_cache_dir(self) -> Path:
        """Directory for cached data: {project_dir}/cache"""
        return self.project_dir / 'cache'


class FileUtilsMixin:
    """
    Mixin providing standard file and directory operations.
    
    Requires self.logger to be available (e.g., from LoggingMixin).
    """

    def ensure_dir(
        self,
        path: Union[str, Path],
        parents: bool = True,
        exist_ok: bool = True
    ) -> Path:
        """Ensure a directory exists."""
        from .file_utils import ensure_dir
        logger = getattr(self, 'logger', None)
        return ensure_dir(path, logger=logger, parents=parents, exist_ok=exist_ok)

    def copy_file(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        preserve_metadata: bool = True
    ) -> Path:
        """Copy a file."""
        from .file_utils import copy_file
        logger = getattr(self, 'logger', None)
        return copy_file(src, dst, logger=logger, preserve_metadata=preserve_metadata)

    def copy_tree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        dirs_exist_ok: bool = True,
        ignore_patterns: Optional[List[str]] = None
    ) -> Path:
        """Copy a directory tree."""
        from .file_utils import copy_tree
        logger = getattr(self, 'logger', None)
        return copy_tree(
            src, dst, 
            logger=logger, 
            dirs_exist_ok=dirs_exist_ok, 
            ignore_patterns=ignore_patterns
        )

    def safe_delete(self, path: Union[str, Path], ignore_errors: bool = True) -> bool:
        """Safely delete a file or directory."""
        from .file_utils import safe_delete
        logger = getattr(self, 'logger', None)
        return safe_delete(path, logger=logger, ignore_errors=ignore_errors)


class ValidationMixin:
    """
    Mixin providing standard validation operations.
    
    Requires self.config_dict to be available (e.g., from ConfigMixin).
    """

    def validate_config(self, required_keys: List[str], operation: str) -> None:
        """Validate configuration keys."""
        from .validation import validate_config_keys
        config_dict = getattr(self, 'config_dict', {})
        validate_config_keys(config_dict, required_keys, operation)

    def validate_file(self, file_path: Union[str, Path], description: str = "file") -> Path:
        """Validate that a file exists."""
        from .validation import validate_file_exists
        return validate_file_exists(file_path, description)

    def validate_dir(self, dir_path: Union[str, Path], description: str = "directory") -> Path:
        """Validate that a directory exists."""
        from .validation import validate_directory_exists
        return validate_directory_exists(dir_path, description)


class TimingMixin:
    """
    Mixin providing timing and profiling utilities.
    
    Requires self.logger to be available.
    """

    @contextmanager
    def time_limit(self, task_name: str) -> ContextManager[None]:
        """
        Context manager to time a task and log the duration.
        """
        start_time = time.time()
        logger = getattr(self, 'logger', logging.getLogger(__name__))
        logger.debug(f"Starting task: {task_name}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"Completed task: {task_name} in {duration:.2f} seconds")


class ConfigurableMixin(LoggingMixin, ProjectContextMixin, FileUtilsMixin, ValidationMixin, TimingMixin):
    """
    Unified mixin for classes requiring logging, config, project context, file utils, validation, and timing.

    This is the recommended mixin for most SYMFLUENCE components that need
    to be aware of the project structure and configuration.
    """
    def _resolve_config_value(self, typed_accessor, dict_key=None, default=None):
        """
        Deprecated: Use _get_config_value instead.

        This method is kept for backward compatibility but delegates to _get_config_value.
        The dict_key parameter is ignored since we now use typed config only.

        Args:
            typed_accessor: Callable returning typed config value
            dict_key: Ignored - kept for backward compatibility
            default: Default to use if missing or 'default'
        """
        import warnings
        warnings.warn(
            "_resolve_config_value is deprecated, use _get_config_value instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self._get_config_value(typed_accessor, default)
