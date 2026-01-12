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
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING
import shutil
import subprocess
import os

from symfluence.core.path_resolver import PathResolverMixin
from symfluence.core.mixins import ShapefileAccessMixin
from symfluence.core.exceptions import (
    ModelExecutionError,
    ConfigurationError,
    validate_config_keys
)

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseModelRunner(ABC, PathResolverMixin, ShapefileAccessMixin):
    """
    Abstract base class for all model runners.

    Provides common initialization, path management, and utility methods
    that are shared across different hydrological model runners.

    Inheritance Structure:
        BaseModelRunner (this class)
        ├── ABC: Abstract base class functionality
        ├── PathResolverMixin: Path resolution utilities
        │   ├── Inherits from: ConfigurableMixin → LoggingMixin + ProjectContextMixin
        │   ├── Methods:
        │   │   - _get_default_path(config_key, default_subpath, must_exist=False)
        │   │   - _get_file_path(path_key, name_key, default_subpath, default_name, must_exist=False)
        │   └── Provides consistent handling of "default" keyword in config
        └── ShapefileAccessMixin: Shapefile column name properties
            ├── Inherits from: ConfigMixin
            └── Properties (~20 properties):
                - Catchment: catchment_name_col, catchment_hruid_col, catchment_gruid_col, etc.
                - River network: river_network_name_col, river_segid_col, river_downsegid_col, etc.
                - Pour point: pour_point_name_col, pour_point_gruid_col, etc.

    Usage Example:
        class MyModelRunner(BaseModelRunner):
            def __init__(self, config, logger):
                super().__init__(config, logger)

                # Use PathResolverMixin methods
                self.forcing_path = self._get_default_path(
                    'FORCING_PATH', 'forcing/data', must_exist=True
                )

                # Use ShapefileAccessMixin properties
                hru_column = self.catchment_hruid_col  # From config or default 'HRU_ID'

                # Use get_install_path for executables
                self.model_exe = self.get_model_executable(
                    'MY_MODEL_INSTALL_PATH',
                    'installs/mymodel/bin',
                    'MY_MODEL_EXE',
                    'mymodel.exe',
                    must_exist=True
                )

            def _get_model_name(self) -> str:
                return 'MyModel'

    Attributes:
        config: SymfluenceConfig instance (typed config object)
        logger: Logger instance
        data_dir: Root data directory
        domain_name: Name of the domain
        project_dir: Project-specific directory
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'GR')
        output_dir: Directory for model outputs (created if specified)

    Abstract Methods:
        Subclasses must implement:
        - _get_model_name(): Return model name string

    Optional Hooks:
        Subclasses may override:
        - _setup_model_specific_paths(): Setup paths after base initialization
        - _should_create_output_dir(): Control output directory creation
        - _get_output_dir(): Customize output directory location
        - _validate_required_config(): Add model-specific config validation
    """

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize base model runner.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance
            reporting_manager: ReportingManager instance

        Raises:
            ConfigurationError: If required configuration keys are missing
        """
        # Import here to avoid circular imports at module level
        from symfluence.core.config.models import SymfluenceConfig

        # Auto-convert dict to typed config for backward compatibility
        if isinstance(config, dict):
            self._config = SymfluenceConfig(**config)
        else:
            self._config = config

        self.logger = logger
        self.reporting_manager = reporting_manager

        # Validate required configuration keys
        self._validate_required_config()

        # Base paths - direct typed access
        self.data_dir = self.config.system.data_dir
        self.code_dir = self._get_config_value(
            lambda: self.config.system.code_dir,
            default=None
        )
        if self.code_dir:
            self.code_dir = Path(self.code_dir)
        self.domain_name = self.config.domain.name
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Model-specific initialization
        self.model_name = self._get_model_name()

        # Allow subclasses to perform custom setup before output dir creation
        self._setup_model_specific_paths()

        # Create output directory if configured to do so
        if self._should_create_output_dir():
            self.output_dir = self._get_output_dir()
            self.ensure_dir(self.output_dir)

    # =========================================================================
    # Inherited Methods from Mixins
    # =========================================================================
    # The following methods and properties are provided by parent mixins:
    #
    # From PathResolverMixin (src/symfluence/core/path_resolver.py):
    #   - _get_default_path(config_key, default_subpath, must_exist=False)
    #       Resolves paths with "default" keyword support
    #   - _get_file_path(path_key, name_key, default_subpath, default_name, must_exist=False)
    #       Resolves file paths (directory + filename)
    #
    # From ShapefileAccessMixin (src/symfluence/core/mixins/shapefile.py):
    #   Catchment shapefile columns:
    #     - catchment_name_col, catchment_hruid_col, catchment_gruid_col
    #     - catchment_area_col, catchment_lat_col, catchment_lon_col
    #     - catchment_elev_col, catchment_slope_col
    #   River network columns:
    #     - river_network_name_col, river_segid_col, river_downsegid_col
    #     - river_slope_col, river_length_col, river_topo_col
    #   Pour point columns:
    #     - pour_point_name_col, pour_point_gruid_col
    #
    # From ConfigurableMixin (inherited via PathResolverMixin):
    #   - _get_config_value(accessor, default)
    #       Safe typed config access with fallback
    #   - validate_config(required_keys, context)
    #       Validate required configuration keys
    #
    # From ConfigMixin (inherited via ShapefileAccessMixin):
    #   - config (property): Returns typed or dict config
    #   - config_dict (property): Returns dict representation of config
    #
    # From LoggingMixin and ProjectContextMixin (inherited transitively):
    #   - ensure_dir(path): Create directory if it doesn't exist
    #   - copy_file(src, dst): Copy file with logging
    #   - copy_tree(src, dst): Copy directory tree with logging
    #   - run_command(command, **kwargs): Execute shell command
    # =========================================================================

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
        self.validate_config(
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
        experiment_id = self.config.domain.experiment_id
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
        self.ensure_dir(backup_path)

        # Copy all files from source to backup using copy_file and copy_tree
        for item in source_dir.iterdir():
            if item.is_file():
                self.copy_file(item, backup_path / item.name)
            elif item.is_dir() and not item.name.startswith('.'):
                self.copy_tree(item, backup_path / item.name)

        self.logger.info(f"Settings backed up to {backup_path}")

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
            experiment_id = self.config_dict.get('EXPERIMENT_ID', 'default')
            log_path = self.project_dir / 'simulations' / experiment_id / self.model_name / log_subdir

        return self.ensure_dir(log_path)

    def get_install_path(
        self,
        config_key: str,
        default_subpath: str,
        relative_to: str = 'data_dir',
        must_exist: bool = False,
        typed_accessor: Optional[Any] = None
    ) -> Path:
        """
        Resolve model installation path from config or use default.

        Args:
            config_key: Configuration key (e.g., 'SUMMA_INSTALL_PATH')
            default_subpath: Default path relative to base (e.g., 'installs/summa/bin')
            relative_to: Base directory ('data_dir' or 'project_dir')
            must_exist: If True, raise FileNotFoundError if path doesn't exist
            typed_accessor: Optional lambda to access typed config directly

        Returns:
            Path to installation directory

        Raises:
            FileNotFoundError: If must_exist=True and path doesn't exist

        Example:
            self.summa_exe = self.get_install_path(
                'SUMMA_INSTALL_PATH',
                'installs/summa/bin',
                must_exist=True,
                typed_accessor=lambda: self.config.model.summa.install_path
            ) / 'summa.exe'
        """
        self.logger.debug(f"Resolving install path for key: {config_key}, default: {default_subpath}, relative_to: {relative_to}")

        # Get install path from typed config or config_dict
        if typed_accessor:
            install_path = self._get_config_value(typed_accessor, default='default')
        else:
            # Fallback to config_dict for legacy keys
            install_path = self.config_dict.get(config_key, 'default')

        if install_path == 'default' or install_path is None:
            if relative_to == 'data_dir':
                path = self.data_dir / default_subpath
                # Fallback search if not found in current data_dir
                if not path.exists():
                    # 1. Try code_dir
                    if self.code_dir:
                        fallback_path = self.code_dir / default_subpath
                        if fallback_path.exists():
                            self.logger.debug(f"Default path not found in data_dir, using fallback from code_dir: {fallback_path}")
                            path = fallback_path
                        else:
                            # 2. Try default sibling data directory (SYMFLUENCE_data)
                            sibling_data = self.code_dir.parent / 'SYMFLUENCE_data'
                            fallback_path = sibling_data / default_subpath
                            if fallback_path.exists():
                                self.logger.debug(f"Default path not found in data_dir or code_dir, using fallback from sibling data dir: {fallback_path}")
                                path = fallback_path
            elif relative_to == 'code_dir':
                path = self.code_dir / default_subpath if self.code_dir else self.data_dir / default_subpath
                # Fallback search if not found in code_dir
                if self.code_dir and not path.exists():
                    # Try default sibling data directory (SYMFLUENCE_data)
                    sibling_data = self.code_dir.parent / 'SYMFLUENCE_data'
                    fallback_path = sibling_data / default_subpath
                    if fallback_path.exists():
                        self.logger.debug(f"Default path not found in code_dir, using fallback from sibling data dir: {fallback_path}")
                        path = fallback_path
            else:
                path = self.project_dir / default_subpath
            self.logger.debug(f"Resolved default install path: {path}")
        else:
            path = Path(install_path)
            self.logger.debug(f"Using custom install path: {path}")

        # Optional validation
        if must_exist and not path.exists():
            raise FileNotFoundError(
                f"Installation path not found: {path}\n"
                f"Config key: {config_key}"
            )

        return path

    def get_model_executable(
        self,
        install_path_key: str,
        default_install_subpath: str,
        exe_name_key: Optional[str] = None,
        default_exe_name: Optional[str] = None,
        typed_exe_accessor: Optional[Any] = None,
        relative_to: str = 'data_dir',
        must_exist: bool = False
    ) -> Path:
        """
        Resolve complete model executable path (install dir + exe name).

        Standardizes the common pattern of:
        1. Resolving installation directory from config
        2. Resolving executable name from config
        3. Combining them into full executable path

        Args:
            install_path_key: Config key for install directory (e.g., 'FUSE_INSTALL_PATH')
            default_install_subpath: Default install dir (e.g., 'installs/fuse/bin')
            exe_name_key: Config key for exe name (e.g., 'FUSE_EXE')
            default_exe_name: Default exe name (e.g., 'fuse.exe')
            typed_exe_accessor: Optional lambda for typed config exe name
            relative_to: Base directory ('data_dir' or 'project_dir')
            must_exist: If True, raise FileNotFoundError if executable doesn't exist

        Returns:
            Complete path to model executable

        Raises:
            FileNotFoundError: If must_exist=True and executable doesn't exist

        Example:
            >>> # Simple case
            >>> self.fuse_exe = self.get_model_executable(
            ...     'FUSE_INSTALL_PATH',
            ...     'installs/fuse/bin',
            ...     'FUSE_EXE',
            ...     'fuse.exe'
            ... )

            >>> # With typed config accessor
            >>> self.mesh_exe = self.get_model_executable(
            ...     'MESH_INSTALL_PATH',
            ...     'installs/MESH-DEV',
            ...     'MESH_EXE',
            ...     'sa_mesh',
            ...     typed_exe_accessor=lambda: self.config.model.mesh.exe if self.config.model.mesh else None
            ... )
        """
        # Get installation directory
        install_dir = self.get_install_path(
            install_path_key,
            default_install_subpath,
            relative_to=relative_to,
            must_exist=False  # We'll check exe existence instead
        )

        # Get executable name
        if typed_exe_accessor:
            exe_name = self._get_config_value(typed_exe_accessor, default=default_exe_name)
        elif exe_name_key:
            exe_name = self.config_dict.get(exe_name_key, default_exe_name)
        else:
            exe_name = default_exe_name

        # Combine into full path
        exe_path = install_dir / exe_name

        # Optional validation
        if must_exist and not exe_path.exists():
            raise FileNotFoundError(
                f"Model executable not found: {exe_path}\n"
                f"Install path key: {install_path_key}\n"
                f"Exe name key: {exe_name_key}"
            )

        return exe_path

    def execute_model_subprocess(
        self,
        command: Union[List[str], str],
        log_file: Path,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = False,
        check: bool = True,
        timeout: Optional[int] = None,
        success_message: str = "Model execution completed successfully",
        success_log_level: int = logging.INFO,
        error_context: Optional[Dict[str, Any]] = None
    ) -> subprocess.CompletedProcess:
        """
        Execute model subprocess with standardized error handling and logging.

        Args:
            command: Command to execute (list or string)
            log_file: Path to log file for stdout/stderr
            cwd: Working directory for command execution
            env: Environment variables (merged with os.environ)
            shell: Whether to use shell execution
            check: Whether to raise CalledProcessError on non-zero exit
            timeout: Optional timeout in seconds
            success_message: Message to log on success
            success_log_level: Log level for success message (default: logging.INFO)
            error_context: Additional context to log on error (e.g., paths, env vars)

        Returns:
            CompletedProcess object with result information

        Raises:
            subprocess.CalledProcessError: If execution fails and check=True
            subprocess.TimeoutExpired: If timeout is exceeded
        """
        try:
            # Merge environment variables
            run_env = os.environ.copy()
            if env:
                run_env.update(env)

            # Ensure log directory exists
            self.ensure_dir(log_file.parent)

            # Execute subprocess
            self.logger.debug(f"Executing command: {' '.join(command)}")
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=check,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    env=run_env,
                    shell=shell,
                    text=True,
                    timeout=timeout
                )

            if result.returncode == 0:
                self.logger.log(success_log_level, success_message)
            else:
                self.logger.warning(f"Process exited with code {result.returncode}")

            return result

        except subprocess.CalledProcessError as e:
            error_msg = f"Model execution failed with return code {e.returncode}"
            self.logger.error(error_msg)

            # Log error context if provided
            if error_context:
                for key, value in error_context.items():
                    self.logger.error(f"{key}: {value}")

            self.logger.error(f"See log file for details: {log_file}")
            raise

        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Process timeout after {timeout} seconds")
            self.logger.error(f"See log file for details: {log_file}")
            raise

    def verify_required_files(
        self,
        files: Union[Path, List[Path]],
        context: str = "model execution"
    ) -> None:
        """
        Verify that required files exist, raise FileNotFoundError if missing.

        Args:
            files: Single path or list of paths to verify
            context: Description of what these files are for (used in error message)

        Raises:
            FileNotFoundError: If any required file is missing
        """
        # Normalize to list
        if isinstance(files, Path):
            files = [files]

        # Check existence
        missing_files = [f for f in files if not f.exists()]

        if missing_files:
            error_msg = f"Required files for {context} not found:\n"
            error_msg += "\n".join(f"  - {f}" for f in missing_files)
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.debug(f"Verified {len(files)} required file(s) for {context}")

    def get_config_path(
        self,
        config_key: str,
        default_subpath: str,
        must_exist: bool = False
    ) -> Path:
        """
        Resolve configuration path with default fallback.

        This is a convenience wrapper around PathResolverMixin._get_default_path
        with consistent naming for model runners.

        Args:
            config_key: Configuration key to look up
            default_subpath: Default path relative to project_dir
            must_exist: Whether to raise error if path doesn't exist

        Returns:
            Resolved Path object
        """
        return self._get_default_path(config_key, default_subpath, must_exist)

    def verify_model_outputs(
        self,
        expected_files: Union[str, List[str]],
        output_dir: Optional[Path] = None
    ) -> bool:
        """
        Verify that expected model output files exist.

        Args:
            expected_files: Single filename or list of expected output filenames
            output_dir: Directory to check (defaults to self.output_dir)

        Returns:
            True if all files exist, False otherwise
        """
        if isinstance(expected_files, str):
            expected_files = [expected_files]

        check_dir = output_dir or self.output_dir

        missing_files = []
        for filename in expected_files:
            if not (check_dir / filename).exists():
                missing_files.append(filename)

        if missing_files:
            self.logger.error(
                f"Missing {len(missing_files)} expected output file(s) in {check_dir}:\n" +
                "\n".join(f"  - {f}" for f in missing_files)
            )
            return False

        self.logger.debug(f"Verified {len(expected_files)} output file(s) in {check_dir}")
        return True

    def get_experiment_output_dir(
        self,
        experiment_id: Optional[str] = None
    ) -> Path:
        """
        Get the experiment-specific output directory for this model.

        Standard pattern: {project_dir}/simulations/{experiment_id}/{model_name}

        Args:
            experiment_id: Experiment identifier (defaults to config.domain.experiment_id)

        Returns:
            Path to experiment output directory
        """
        exp_id = experiment_id or self.config.domain.experiment_id
        return self.project_dir / 'simulations' / exp_id / self.model_name

    def setup_path_aliases(self, aliases: Dict[str, str]) -> None:
        """
        Set up legacy path aliases for backward compatibility.

        Args:
            aliases: Dictionary mapping alias name to source attribute
                     Example: {'root_path': 'data_dir', 'result_dir': 'output_dir'}
        """
        for alias, source_attr in aliases.items():
            if hasattr(self, source_attr):
                setattr(self, alias, getattr(self, source_attr))
                self.logger.debug(f"Set legacy alias: {alias} -> {source_attr}")
            else:
                self.logger.warning(
                    f"Cannot create alias '{alias}': source attribute '{source_attr}' not found"
                )
