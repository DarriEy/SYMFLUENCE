"""
Base class for model runners.

Provides shared infrastructure for all model execution modules including:
- Configuration management
- Path resolution with default fallbacks
- Directory creation for outputs and logs
- Common experiment structure
- Settings file backup utilities
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

from symfluence.utils.common.path_resolver import PathResolverMixin
from symfluence.utils.exceptions import (
    ModelExecutionError,
    ConfigurationError,
    validate_config_keys
)


class BaseModelRunner(ABC, PathResolverMixin):
    """
    Abstract base class for all model runners.

    Provides common initialization, path management, and utility methods
    that are shared across different hydrological model runners.

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        data_dir: Root data directory
        domain_name: Name of the domain
        project_dir: Project-specific directory
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'GR')
        output_dir: Directory for model outputs (created if specified)
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize base model runner.

        Args:
            config: Configuration dictionary
            logger: Logger instance

        Raises:
            ConfigurationError: If required configuration keys are missing
        """
        # Core configuration
        self.config = config
        self.logger = logger

        # Validate required configuration keys
        self._validate_required_config()

        # Base paths (standard naming)
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Model-specific initialization
        self.model_name = self._get_model_name()

        # Allow subclasses to perform custom setup before output dir creation
        self._setup_model_specific_paths()

        # Create output directory if configured to do so
        if self._should_create_output_dir():
            self.output_dir = self._get_output_dir()
            self.output_dir.mkdir(parents=True, exist_ok=True)

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
        ]
        validate_config_keys(
            self.config,
            required_keys,
            f"{self._get_model_name()} runner initialization"
        )

    @abstractmethod
    def _get_model_name(self) -> str:
        """
        Return the name of the model.

        Must be implemented by subclasses.

        Returns:
            Model name (e.g., 'SUMMA', 'FUSE', 'GR')
        """
        pass

    def _setup_model_specific_paths(self) -> None:
        """
        Hook for subclasses to set up model-specific paths.

        Called after base paths are initialized but before output_dir creation.
        Override this method to add model-specific path attributes.

        Example:
            def _setup_model_specific_paths(self):
                self.setup_dir = self.project_dir / "settings" / self.model_name
                self.forcing_path = self.project_dir / 'forcing' / f'{self.model_name}_input'
        """
        pass

    def _should_create_output_dir(self) -> bool:
        """
        Determine if output directory should be created in __init__.

        Default behavior is to create it. Subclasses can override.

        Returns:
            True if output_dir should be created, False otherwise
        """
        return True

    def _get_output_dir(self) -> Path:
        """
        Get the output directory path for this model run.

        Default implementation uses EXPERIMENT_ID from config.
        Subclasses can override for custom behavior.

        Returns:
            Path to output directory
        """
        experiment_id = self.config.get('EXPERIMENT_ID')
        return self.project_dir / 'simulations' / experiment_id / self.model_name

    def backup_settings(self, source_dir: Path, backup_subdir: str = "run_settings") -> None:
        """
        Backup settings files to the output directory for reproducibility.

        Args:
            source_dir: Source directory containing settings to backup
            backup_subdir: Subdirectory name within output_dir for backups

        Raises:
            FileOperationError: If backup fails
        """
        if not hasattr(self, 'output_dir'):
            self.logger.warning("Cannot backup settings: output_dir not initialized")
            return

        backup_path = self.output_dir / backup_subdir
        backup_path.mkdir(parents=True, exist_ok=True)

        try:
            # Copy all files from source to backup
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, backup_path / item.name)
                elif item.is_dir() and not item.name.startswith('.'):
                    shutil.copytree(item, backup_path / item.name, dirs_exist_ok=True)

            self.logger.info(f"Settings backed up to {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to backup settings: {e}")
            raise

    def get_log_path(self, log_subdir: str = "logs") -> Path:
        """
        Get or create log directory path for this model run.

        Args:
            log_subdir: Subdirectory name for logs

        Returns:
            Path to log directory (created if it doesn't exist)
        """
        if hasattr(self, 'output_dir'):
            log_path = self.output_dir / log_subdir
        else:
            # Fallback if output_dir not set
            experiment_id = self.config.get('EXPERIMENT_ID', 'default')
            log_path = self.project_dir / 'simulations' / experiment_id / self.model_name / log_subdir

        log_path.mkdir(parents=True, exist_ok=True)
        return log_path
