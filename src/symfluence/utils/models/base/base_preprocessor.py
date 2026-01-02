"""
Base class for model preprocessors.

Provides shared infrastructure for all model preprocessing modules including:
- Configuration management
- Path resolution with default fallbacks
- Directory creation
- Common directory structure
- Settings file copying
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import shutil

import pandas as pd
import xarray as xr

from symfluence.utils.common.path_resolver import PathResolverMixin
from symfluence.utils.common.constants import UnitConversion
from symfluence.utils.exceptions import (
    ModelExecutionError,
    ConfigurationError,
    FileOperationError,
    validate_config_keys,
    validate_directory_exists,
    symfluence_error_handler
)

# Import for type checking only (avoid circular imports)
try:
    from symfluence.utils.config.models import SymfluenceConfig
except ImportError:
    SymfluenceConfig = None


class BaseModelPreProcessor(ABC, PathResolverMixin):
    """
    Abstract base class for all model preprocessors.

    Provides common initialization, path management, and utility methods
    that are shared across different hydrological model preprocessors.

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        data_dir: Root data directory
        domain_name: Name of the domain
        project_dir: Project-specific directory
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'GR')
        setup_dir: Directory for model setup files
        forcing_dir: Directory for model-specific forcing inputs
        forcing_basin_path: Directory for basin-averaged forcing data
    """

    def __init__(self, config: Union[Dict[str, Any], 'SymfluenceConfig'], logger: Any):
        """
        Initialize base model preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig instance (Phase 2)
            logger: Logger instance

        Raises:
            ConfigurationError: If required configuration keys are missing
        """
        # Phase 2: Support both typed config and dict config for backward compatibility
        if SymfluenceConfig and isinstance(config, SymfluenceConfig):
            self.typed_config = config
            self.config = config.to_dict(flatten=True)
        else:
            self.typed_config = None
            self.config = config

        self.logger = logger

        # Validate required configuration keys
        self._validate_required_config()

        # Base paths
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Model-specific paths (subclasses should set model_name)
        self.model_name = self._get_model_name()
        self.setup_dir = self.project_dir / "settings" / self.model_name
        self.forcing_dir = self.project_dir / "forcing" / f"{self.model_name}_input"

        # Common forcing paths
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_raw_path = self._get_default_path('FORCING_RAW_PATH', 'forcing/raw_data')
        self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_data')

        # Common shapefile paths
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'
        self.intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'

        # Common configuration
        self.forcing_dataset = self.config.get('FORCING_DATASET', '').lower()
        self.forcing_time_step_size = int(self.config.get('FORCING_TIME_STEP_SIZE', 86400))

    def _validate_required_config(self) -> None:
        """
        Validate that all required configuration keys are present.

        Subclasses can override to add model-specific required keys.

        Raises:
            ConfigurationError: If required keys are missing
        """
        required_keys = [
            'SYMFLUENCE_DATA_DIR',
            'DOMAIN_NAME',
            'FORCING_DATASET',
        ]
        validate_config_keys(self.config, required_keys, f"{self._get_model_name()} preprocessing")

    @abstractmethod
    def _get_model_name(self) -> str:
        """
        Return the name of the model.

        Must be implemented by subclasses.

        Returns:
            Model name (e.g., 'SUMMA', 'FUSE', 'GR')
        """
        pass

    @abstractmethod
    def run_preprocessing(self):
        """
        Run model-specific preprocessing workflow.

        Must be implemented by subclasses to define the complete
        preprocessing pipeline for the specific model.

        **Best Practice**: Implementations should use the `symfluence_error_handler`
        context manager to ensure consistent error handling:

        Example:
            >>> def run_preprocessing(self):
            ...     with symfluence_error_handler(
            ...         f"{self.model_name} preprocessing",
            ...         self.logger,
            ...         error_type=ModelExecutionError
            ...     ):
            ...         # preprocessing steps here
            ...         self.create_directories()
            ...         self.prepare_forcing()
            ...         # ...

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails
        """
        pass


    def _get_file_path(self, file_type: str, path_key: str,
                       name_key: str, default_name: str) -> Path:
        """
        Resolve complete file path from config or defaults.

        Args:
            file_type: Description of file type (for logging)
            path_key: Config key for directory path
            name_key: Config key for file name
            default_name: Default file name if config value is 'default'

        Returns:
            Complete file path
        """
        # Get directory path
        dir_path = self.config.get(path_key)
        if dir_path == 'default' or dir_path is None:
            self.logger.warning(f"No {file_type} path specified, path resolution may fail")
            dir_path = self.project_dir
        else:
            dir_path = Path(dir_path)

        # Get file name
        file_name = self.config.get(name_key)
        if file_name == 'default' or file_name is None:
            file_name = default_name

        return dir_path / file_name

    def create_directories(self, additional_dirs: Optional[List[Path]] = None):
        """
        Create necessary directories for model setup.

        Creates standard directories (setup_dir, forcing_dir) plus any
        additional directories specified by the model.

        Args:
            additional_dirs: Optional list of additional directories to create

        Raises:
            FileOperationError: If directory creation fails
        """
        # Standard directories
        dirs_to_create = [
            self.setup_dir,
            self.forcing_dir
        ]

        # Add model-specific directories
        if additional_dirs:
            dirs_to_create.extend(additional_dirs)

        # Create all directories with error handling
        for dir_path in dirs_to_create:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                raise FileOperationError(
                    f"Failed to create directory {dir_path}: {e}"
                ) from e

    def copy_base_settings(self, source_dir: Optional[Path] = None,
                          file_patterns: Optional[List[str]] = None):
        """
        Copy base settings files from source to setup directory.

        Args:
            source_dir: Source directory containing base settings.
                       If None, uses default location based on model name.
            file_patterns: List of file patterns to copy (e.g., ['*.txt', '*.nc']).
                          If None, copies all files.

        Raises:
            FileOperationError: If settings cannot be copied
        """
        if source_dir is None:
            # Try to find base settings in config, then fall back to SYMFLUENCE_CODE_DIR
            try:
                base_settings_key = f'SETTINGS_{self.model_name}_BASE_PATH'
                source_dir = Path(self.config.get(base_settings_key, 'default'))
                if source_dir == Path('default') or not source_dir.exists():
                    fallback_dir = self.get_base_settings_source_dir()
                    if fallback_dir.exists():
                        source_dir = fallback_dir
                    else:
                        self.logger.warning(
                            f"Base settings directory not found for {self.model_name}. "
                            f"Skipping settings copy."
                        )
                        return
            except Exception as e:
                self.logger.warning(f"Could not locate base settings: {e}")
                return

        if source_dir is not None and not source_dir.exists():
            self.logger.warning(
                f"Base settings source directory does not exist: {source_dir}. "
                f"Skipping settings copy."
            )
            return

        self.logger.info(f"Copying base settings from {source_dir} to {self.setup_dir}")

        with symfluence_error_handler(
            f"copying base settings from {source_dir}",
            self.logger,
            error_type=FileOperationError
        ):
            if file_patterns is None:
                # Copy entire directory
                if self.setup_dir.exists():
                    shutil.rmtree(self.setup_dir)
                shutil.copytree(source_dir, self.setup_dir)
                self.logger.info(f"Copied all files from {source_dir}")
            else:
                # Copy specific file patterns
                self.setup_dir.mkdir(parents=True, exist_ok=True)
                for pattern in file_patterns:
                    for file_path in source_dir.glob(pattern):
                        if file_path.is_file():
                            dest_path = self.setup_dir / file_path.name
                            shutil.copy2(file_path, dest_path)
                            self.logger.info(f"Copied {file_path.name}")

    def get_catchment_path(self) -> Path:
        """
        Get path to catchment shapefile.

        Returns:
            Path to catchment shapefile
        """
        catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')

        if catchment_name == 'default' or catchment_name is None:
            discretization = self.config.get('DOMAIN_DISCRETIZATION')
            catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"

        return catchment_path / catchment_name

    def get_river_network_path(self) -> Path:
        """
        Get path to river network shapefile.

        Returns:
            Path to river network shapefile
        """
        river_path = self._get_default_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network')
        river_name = self.config.get('RIVER_NETWORK_SHP_NAME')

        if river_name == 'default' or river_name is None:
            river_name = f"{self.domain_name}_riverNetwork_delineate.shp"

        return river_path / river_name

    def _is_lumped(self) -> bool:
        """
        Check if domain is configured as lumped.

        Returns:
            True if lumped, False if distributed
        """
        return self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'

    def get_dem_path(self) -> Path:
        """
        Get path to DEM file.

        Returns:
            Path to DEM file
        """
        dem_name = self.config.get('DEM_NAME')
        if dem_name == "default" or dem_name is None:
            dem_name = f"domain_{self.domain_name}_elv.tif"
        return self._get_default_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")

    def get_timestep_config(self) -> Dict[str, Any]:
        """
        Get timestep configuration based on FORCING_TIME_STEP_SIZE.

        Provides standardized configuration for time-related parameters used
        across different models for data processing and unit conversions.

        Returns:
            Dict with keys:
                - resample_freq: Pandas resample frequency string (e.g., 'h', 'D')
                - time_units: NetCDF time units string
                - time_unit: Pandas timedelta unit
                - conversion_factor: Factor to convert from cms to mm/timestep
                - time_label: Human-readable label
                - timestep_seconds: Timestep in seconds
        """
        timestep_seconds = self.forcing_time_step_size

        if timestep_seconds == 3600:  # Hourly
            return {
                'resample_freq': 'h',
                'time_units': 'hours since 1970-01-01',
                'time_unit': 'h',
                'conversion_factor': UnitConversion.MM_HOUR_TO_CMS,  # cms to mm/hour
                'time_label': 'hourly',
                'timestep_seconds': 3600
            }
        elif timestep_seconds == 86400:  # Daily
            return {
                'resample_freq': 'D',
                'time_units': 'days since 1970-01-01',
                'time_unit': 'D',
                'conversion_factor': UnitConversion.MM_DAY_TO_CMS,  # cms to mm/day
                'time_label': 'daily',
                'timestep_seconds': 86400
            }
        else:
            # Generic case - calculate based on seconds
            hours = timestep_seconds / 3600
            if hours < 24:
                return {
                    'resample_freq': f'{int(hours)}h',
                    'time_units': 'hours since 1970-01-01',
                    'time_unit': 'h',
                    'conversion_factor': 3.6 * hours,
                    'time_label': f'{int(hours)}-hourly',
                    'timestep_seconds': timestep_seconds
                }
            else:
                days = timestep_seconds / 86400
                return {
                    'resample_freq': f'{int(days)}D',
                    'time_units': 'days since 1970-01-01',
                    'time_unit': 'D',
                    'conversion_factor': 86.4 * days,
                    'time_label': f'{int(days)}-daily',
                    'timestep_seconds': timestep_seconds
                }

    def get_base_settings_source_dir(self) -> Path:
        """
        Get the source directory for base settings files.

        Returns:
            Path to base settings directory for this model
        """
        code_dir = Path(self.config.get('SYMFLUENCE_CODE_DIR'))
        return code_dir / '0_base_settings' / self.model_name

    # =========================================================================
    # Time Window Utilities
    # =========================================================================

    def get_simulation_time_window(self) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Get simulation start/end times from typed config or dict config.

        Handles both SymfluenceConfig and dict-based configuration with
        consistent parsing and error handling.

        Returns:
            Tuple of (start_time, end_time) as pandas Timestamps, or None if
            time window cannot be determined.
        """
        if self.typed_config:
            start_raw = self.typed_config.domain.time_start
            end_raw = self.typed_config.domain.time_end
        else:
            start_raw = self.config.get('EXPERIMENT_TIME_START')
            end_raw = self.config.get('EXPERIMENT_TIME_END')

        if not start_raw or not end_raw:
            return None

        try:
            return pd.to_datetime(start_raw), pd.to_datetime(end_raw)
        except Exception as exc:
            self.logger.warning(f"Unable to parse simulation time window: {exc}")
            return None

    def subset_to_simulation_time(
        self,
        ds: xr.Dataset,
        label: str = "Data"
    ) -> xr.Dataset:
        """
        Subset dataset to the configured simulation time window.

        Args:
            ds: Dataset with a 'time' coordinate
            label: Label for logging messages (e.g., "Forcing", "Observations")

        Returns:
            Dataset subset to simulation window, or original if subsetting fails
        """
        time_window = self.get_simulation_time_window()
        if time_window is None or "time" not in ds.coords:
            return ds

        start_time, end_time = time_window
        try:
            subset = ds.sel(time=slice(start_time, end_time))
        except Exception as exc:
            self.logger.warning(f"Unable to subset {label} to simulation window: {exc}")
            return ds

        if len(subset.time) == 0:
            self.logger.warning(
                f"{label} has no records in simulation window; using full dataset"
            )
            return ds

        self.logger.info(
            f"{label} subset to simulation window: "
            f"{subset.time.min().values} to {subset.time.max().values}"
        )
        return subset

    def align_datasets_to_period(
        self,
        datasets: Dict[str, xr.Dataset],
        start_time: Union[datetime, pd.Timestamp],
        end_time: Union[datetime, pd.Timestamp],
        freq: str = 'D'
    ) -> Tuple[Dict[str, xr.Dataset], pd.DatetimeIndex]:
        """
        Align multiple datasets to a common time period with reindexing.

        This is useful when combining forcing data, observations, and other
        time series that may have slightly different time ranges.

        Args:
            datasets: Dict mapping names to xr.Dataset objects
            start_time: Start of the alignment period
            end_time: End of the alignment period
            freq: Pandas frequency string for the time index (e.g., 'D', 'h')

        Returns:
            Tuple of (aligned_datasets dict, time_index)
        """
        time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        aligned = {}

        for name, ds in datasets.items():
            try:
                aligned[name] = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
            except Exception as exc:
                self.logger.warning(f"Could not align {name} to time period: {exc}")
                aligned[name] = ds

        return aligned, time_index

    # =========================================================================
    # Template Method Pattern for Preprocessing
    # =========================================================================

    def run_preprocessing_template(self) -> bool:
        """
        Template method for preprocessing workflow.

        Provides a standard preprocessing workflow that models can use by
        overriding the hook methods. This ensures consistent error handling
        and logging across all model preprocessors.

        The workflow is:
        1. _pre_setup() - Model-specific pre-setup (optional)
        2. create_directories() - Create necessary directories
        3. copy_base_settings() - Copy base settings files
        4. _prepare_forcing() - Prepare forcing data (model-specific)
        5. _create_model_configs() - Create model config files (model-specific)
        6. _post_setup() - Model-specific post-setup (optional)

        Returns:
            True if preprocessing completed successfully

        Raises:
            ModelExecutionError: If any step fails
        """
        with symfluence_error_handler(
            f"{self.model_name} preprocessing",
            self.logger,
            error_type=ModelExecutionError
        ):
            self._pre_setup()
            self.create_directories()
            self.copy_base_settings()
            self._prepare_forcing()
            self._create_model_configs()
            self._post_setup()
            self.logger.info(f"{self.model_name} preprocessing completed successfully")
            return True

    def _pre_setup(self) -> None:
        """
        Hook for model-specific pre-setup tasks.

        Override in subclass to perform any setup needed before directory
        creation and settings copy. Default implementation does nothing.
        """
        pass

    def _prepare_forcing(self) -> None:
        """
        Hook for model-specific forcing data preparation.

        Override in subclass to implement forcing data processing.
        Default implementation does nothing.
        """
        pass

    def _create_model_configs(self) -> None:
        """
        Hook for model-specific configuration file creation.

        Override in subclass to create model-specific config files
        (e.g., file managers, parameter files). Default implementation
        does nothing.
        """
        pass

    def _post_setup(self) -> None:
        """
        Hook for model-specific post-setup tasks.

        Override in subclass to perform any cleanup or finalization
        after main preprocessing. Default implementation does nothing.
        """
        pass
