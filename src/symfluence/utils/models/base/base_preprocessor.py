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
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil


class BaseModelPreProcessor(ABC):
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

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize base model preprocessor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        # Core configuration
        self.config = config
        self.logger = logger

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
        """
        pass

    def _get_default_path(self, config_key: str, default_subpath: str) -> Path:
        """
        Get path from config or use default relative to project directory.

        Args:
            config_key: Key to look up in configuration
            default_subpath: Default path relative to project_dir

        Returns:
            Resolved path
        """
        path_value = self.config.get(config_key)

        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath

        return Path(path_value)

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
        """
        # Standard directories
        dirs_to_create = [
            self.setup_dir,
            self.forcing_dir
        ]

        # Add model-specific directories
        if additional_dirs:
            dirs_to_create.extend(additional_dirs)

        # Create all directories
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def copy_base_settings(self, source_dir: Optional[Path] = None,
                          file_patterns: Optional[List[str]] = None):
        """
        Copy base settings files from source to setup directory.

        Args:
            source_dir: Source directory containing base settings.
                       If None, uses default location based on model name.
            file_patterns: List of file patterns to copy (e.g., ['*.txt', '*.nc']).
                          If None, copies all files.
        """
        if source_dir is None:
            # Try to find base settings in package
            from importlib import resources
            try:
                # Try to get base settings from package data
                base_settings_key = f'SETTINGS_{self.model_name}_BASE_PATH'
                source_dir = Path(self.config.get(base_settings_key, 'default'))

                if source_dir == Path('default') or not source_dir.exists():
                    self.logger.warning(
                        f"Base settings directory not found for {self.model_name}. "
                        f"Skipping settings copy."
                    )
                    return
            except Exception as e:
                self.logger.warning(f"Could not locate base settings: {e}")
                return

        if not source_dir.exists():
            self.logger.warning(f"Source directory does not exist: {source_dir}")
            return

        self.logger.info(f"Copying base settings from {source_dir} to {self.setup_dir}")

        try:
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

        except Exception as e:
            self.logger.error(f"Error copying base settings: {e}")
            raise

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
