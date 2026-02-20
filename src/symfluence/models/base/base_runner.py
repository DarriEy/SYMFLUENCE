"""
Base class for model runners.

Provides shared infrastructure for all model execution modules including:
- Configuration management
- Path resolution with default fallbacks
- Directory creation for outputs and logs
- Common experiment structure
- Settings file backup utilities
- Subprocess and SLURM execution
- Spatial mode validation (Phase 3)
"""

from abc import ABC, abstractmethod
import logging
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, TYPE_CHECKING

from symfluence.core.path_resolver import PathResolverMixin
from symfluence.core.mixins import ShapefileAccessMixin
from symfluence.models.mixins import ModelComponentMixin
from symfluence.models.execution.model_executor import (
    augment_conda_library_paths,
    ExecutionResult,
    ExecutionMode,
    SlurmJobConfig,
)
from symfluence.models.spatial_modes import (
    get_spatial_mode_from_config,
    validate_spatial_mode
)

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseModelRunner(ABC, ModelComponentMixin, PathResolverMixin, ShapefileAccessMixin):  # type: ignore[misc]
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

    Error Handling Pattern:
        All model runners should follow this consistent pattern:

        1. Top-level run methods (run(), run_fuse(), run_hbv(), etc.):
           Use symfluence_error_handler context manager::

               with symfluence_error_handler("Model execution", self.logger,
                                              error_type=ModelExecutionError):
                   # execution code

        2. Internal methods returning bool:
           Use try/except with return values::

               def _internal_step(self) -> bool:
                   try:
                       # step logic
                       return True
                   except Exception as e:
                       self.logger.error(f"Step failed: {e}")
                       return False
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
        # Common initialization via mixin
        self._init_model_component(config, logger, reporting_manager)

        # Runner-specific: code_dir handling
        self.code_dir = self._get_config_value(
            lambda: self.config.system.code_dir,
            default=None
        )
        if self.code_dir:
            self.code_dir = Path(self.code_dir)

        # Allow subclasses to perform custom setup before output dir creation
        self._setup_model_specific_paths()

        # Validate spatial mode compatibility (Phase 3)
        self._validate_spatial_mode()

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

    def _validate_spatial_mode(self) -> None:
        """
        Validate spatial mode compatibility for this model.

        Called during initialization to ensure the model supports the configured
        spatial mode and that routing is properly configured if required.

        Phase 3 Addition: Centralized spatial mode validation across all models.

        Logs warnings for suboptimal configurations but does not raise exceptions
        by default to maintain backward compatibility.
        """
        try:
            # Get current spatial mode from config
            spatial_mode = get_spatial_mode_from_config(self.config_dict)

            # Check if routing is configured
            routing_model = self.config_dict.get('ROUTING_MODEL', 'none')
            has_routing = routing_model and routing_model.lower() not in ('none', 'default', '')

            # Validate against model capabilities
            is_valid, message = validate_spatial_mode(
                self.model_name,
                spatial_mode,
                has_routing_configured=has_routing
            )

            if message:
                if is_valid:
                    # It's a warning, not an error
                    self.logger.warning(f"Spatial mode validation: {message}")
                else:
                    # Configuration is invalid
                    self.logger.error(f"Spatial mode validation error: {message}")
                    # Don't raise to maintain backward compatibility
                    # Subclasses can override to raise if needed

        except Exception as e:
            # Don't let validation failures prevent model initialization
            self.logger.debug(f"Spatial mode validation skipped: {e}")

    def _has_routing_configured(self) -> bool:
        """
        Check if a routing model is configured.

        Returns:
            True if routing model is configured, False otherwise
        """
        routing_model = self.config_dict.get('ROUTING_MODEL', 'none')
        return routing_model and routing_model.lower() not in ('none', 'default', '')

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

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute the model using the registered run method.

        Provides a uniform interface for all model runners by delegating
        to the model-specific run method (run_fuse, run_hbv, etc.).

        Args:
            **kwargs: Arguments passed to the model-specific run method

        Returns:
            Path to output directory on success, None on failure

        Raises:
            NotImplementedError: If model doesn't implement the run method
        """
        from symfluence.models.registry import ModelRegistry

        method_name = ModelRegistry.get_runner_method(self.model_name)
        if method_name is None:
            method_name = f"run_{self.model_name.lower()}"

        run_method = getattr(self, method_name, None)
        if run_method is None:
            raise NotImplementedError(
                f"Model {self.model_name} does not implement {method_name}()"
            )

        return run_method(**kwargs)

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
        must_exist: bool = False,
        candidates: Optional[List[str]] = None
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
            candidates: Optional list of subdirectory candidates to try.
                        The method tries each candidate in order and returns
                        the first existing path. Use '' for the root install dir.
                        e.g., ['', 'cmake_build', 'bin'] for NGEN

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

            >>> # With candidates (search multiple subdirectories)
            >>> self.ngen_exe = self.get_model_executable(
            ...     'NGEN_INSTALL_PATH',
            ...     'installs/ngen',
            ...     default_exe_name='ngen',
            ...     candidates=['', 'cmake_build', 'bin'],
            ...     must_exist=True
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

        # Handle candidates: try each subdirectory in order
        if candidates:
            for candidate in candidates:
                if candidate:
                    candidate_path = install_dir / candidate / exe_name
                else:
                    candidate_path = install_dir / exe_name

                if candidate_path.exists():
                    self.logger.debug(f"Found executable at: {candidate_path}")
                    return candidate_path

            # No candidate found - use first candidate (or root) as default path for error message
            exe_path = install_dir / (candidates[0] if candidates[0] else '') / exe_name
        else:
            # Standard behavior - combine into full path
            exe_path = install_dir / exe_name

        # Optional validation
        if must_exist and not exe_path.exists():
            if candidates:
                searched_paths = [
                    str(install_dir / (c if c else '') / exe_name)
                    for c in candidates
                ]
                raise FileNotFoundError(
                    "Model executable not found in any candidate location.\n"
                    "Searched paths:\n  " + "\n  ".join(searched_paths) + "\n"
                    f"Install path key: {install_path_key}"
                )
            else:
                raise FileNotFoundError(
                    f"Model executable not found: {exe_path}\n"
                    f"Install path key: {install_path_key}\n"
                    f"Exe name key: {exe_name_key}"
                )

        return exe_path

    # =========================================================================
    # Subprocess Execution (absorbed from ModelExecutor)
    # =========================================================================

    def execute_subprocess(
        self,
        command: Union[List[str], str],
        log_file: Path,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        shell: bool = False,
        check: bool = True,
        capture_output: bool = False,
        success_message: Optional[str] = None,
        error_context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute a subprocess with standardized logging and error handling.

        This is the canonical execution method for all model runners. It returns
        an ``ExecutionResult`` with structured information about the run.

        Args:
            command: Command to execute (list or string)
            log_file: Path to write stdout/stderr
            cwd: Working directory for execution
            env: Environment variables (merged with os.environ)
            timeout: Timeout in seconds (None = no timeout)
            shell: Whether to use shell execution
            check: Raise exception on non-zero exit code
            capture_output: Return stdout/stderr in result metadata
            success_message: Custom message to log on success
            error_context: Additional context to include in error logs

        Returns:
            ExecutionResult with success status, return code, and metadata

        Raises:
            subprocess.TimeoutExpired: If timeout is exceeded
            subprocess.CalledProcessError: If check=True and process fails
        """
        start_time = time.time()

        # Merge environment variables
        run_env = os.environ.copy()
        augment_conda_library_paths(run_env)
        if env:
            run_env.update(env)

        # Ensure log directory exists
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Log execution start
        cmd_str = command if isinstance(command, str) else ' '.join(command)
        self.logger.debug(f"Executing: {cmd_str}")
        if cwd:
            self.logger.debug(f"Working directory: {cwd}")

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                result = subprocess.run(
                    command,
                    check=False,  # We handle the check ourselves
                    stdout=f if not capture_output else subprocess.PIPE,
                    stderr=subprocess.STDOUT if not capture_output else subprocess.PIPE,
                    cwd=cwd,
                    env=run_env,
                    shell=shell,  # nosec B602 - shell mode required for model executables
                    text=True,
                    timeout=timeout
                )

            duration = time.time() - start_time

            # Build result
            exec_result = ExecutionResult(
                success=(result.returncode == 0),
                return_code=result.returncode,
                log_file=log_file,
                duration_seconds=duration,
                metadata={}
            )

            if capture_output:
                exec_result.metadata['stdout'] = result.stdout
                exec_result.metadata['stderr'] = result.stderr

            # Log outcome
            if result.returncode == 0:
                msg = success_message or f"Process completed successfully in {duration:.1f}s"
                self.logger.info(msg)
            else:
                self.logger.warning(f"Process exited with code {result.returncode}")
                exec_result.error_message = f"Exit code: {result.returncode}"

                # Log error context
                if error_context:
                    for key, value in error_context.items():
                        self.logger.error(f"  {key}: {value}")

                self.logger.error(f"See log file: {log_file}")

                if check:
                    raise subprocess.CalledProcessError(
                        result.returncode, command
                    )

            return exec_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.logger.error(f"Process timed out after {timeout}s")
            return ExecutionResult(
                success=False,
                return_code=-1,
                log_file=log_file,
                duration_seconds=duration,
                error_message=f"Timeout after {timeout}s"
            )

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

        .. deprecated::
            Use :meth:`execute_subprocess` instead, which returns an
            ``ExecutionResult`` with richer metadata. This method is a
            backward-compatible wrapper that converts the result to
            ``subprocess.CompletedProcess``.

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
        warnings.warn(
            "execute_model_subprocess() is deprecated, use execute_subprocess() instead. "
            "execute_subprocess() returns an ExecutionResult with richer metadata.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            # Merge environment variables
            run_env = os.environ.copy()
            augment_conda_library_paths(run_env)
            if env:
                run_env.update(env)

            # Ensure log directory exists
            self.ensure_dir(log_file.parent)

            # Execute subprocess
            self.logger.debug(f"Executing command: {' '.join(command)}")
            with open(log_file, 'w', encoding='utf-8') as f:
                result = subprocess.run(  # nosec B602 - shell mode for trusted model executables
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
                self.logger.debug(f"Process exited with code {result.returncode}")

            return result

        except subprocess.CalledProcessError as e:
            error_msg = f"Model execution failed with return code {e.returncode}"
            self.logger.error(error_msg)

            # Log error context if provided
            if error_context:
                for key, value in error_context.items():
                    self.logger.error(f"{key}: {value}")

            # Read and include last lines from log file for better diagnostics
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_lines = f.readlines()
                        last_lines = log_lines[-20:] if len(log_lines) > 20 else log_lines
                        if last_lines:
                            self.logger.error("Last 20 lines from log file:")
                            for line in last_lines:
                                self.logger.error(f"  {line.rstrip()}")
                except Exception as read_error:
                    self.logger.error(f"Could not read log file: {read_error}")

            self.logger.error(f"Full log available at: {log_file}")
            raise

        except subprocess.TimeoutExpired:
            self.logger.error(f"Process timeout after {timeout} seconds")
            self.logger.error(f"See log file for details: {log_file}")
            raise

    # =========================================================================
    # SLURM Job Management (absorbed from ModelExecutor)
    # =========================================================================

    def is_slurm_available(self) -> bool:
        """Check if SLURM is available on this system."""
        return shutil.which("sbatch") is not None

    def create_slurm_script(
        self,
        config: SlurmJobConfig,
        commands: List[str],
        setup_commands: Optional[List[str]] = None,
        cleanup_commands: Optional[List[str]] = None
    ) -> str:
        """
        Generate a SLURM batch script.

        Args:
            config: SlurmJobConfig with job parameters
            commands: Main commands to execute
            setup_commands: Commands to run before main commands
            cleanup_commands: Commands to run after main commands

        Returns:
            Complete SLURM batch script as string
        """
        lines = ["#!/bin/bash"]

        # Core directives
        lines.append(f"#SBATCH --job-name={config.job_name}")
        lines.append(f"#SBATCH --time={config.time_limit}")
        lines.append(f"#SBATCH --mem={config.memory}")
        lines.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")

        # Optional directives
        if config.partition:
            lines.append(f"#SBATCH --partition={config.partition}")
        if config.account:
            lines.append(f"#SBATCH --account={config.account}")
        if config.array_size is not None:
            lines.append(f"#SBATCH --array=0-{config.array_size}")
        if config.output_pattern:
            lines.append(f"#SBATCH --output={config.output_pattern}")
        if config.error_pattern:
            lines.append(f"#SBATCH --error={config.error_pattern}")

        # Additional directives
        for key, value in config.additional_directives.items():
            lines.append(f"#SBATCH --{key}={value}")

        lines.append("")

        # Job info for debugging
        lines.extend([
            "# Print job info for debugging",
            'echo "Starting job at $(date)"',
            'echo "Running on host: $(hostname)"',
            'echo "SLURM Job ID: $SLURM_JOB_ID"',
        ])

        if config.array_size is not None:
            lines.append('echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"')

        lines.append("")

        # Setup commands
        if setup_commands:
            lines.append("# Setup")
            lines.extend(setup_commands)
            lines.append("")

        # Main commands
        lines.append("# Main execution")
        lines.extend(commands)
        lines.append("")

        # Cleanup commands
        if cleanup_commands:
            lines.append("# Cleanup")
            lines.extend(cleanup_commands)
            lines.append("")

        lines.append('echo "Job completed at $(date)"')

        return "\n".join(lines)

    def submit_slurm_job(
        self,
        script_path: Path,
        wait: bool = False,
        poll_interval: int = 60,
        max_wait_time: int = 3600
    ) -> ExecutionResult:
        """
        Submit a SLURM job and optionally wait for completion.

        Args:
            script_path: Path to SLURM batch script
            wait: If True, monitor job until completion
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait before returning

        Returns:
            ExecutionResult with job_id and completion status

        Raises:
            RuntimeError: If sbatch command not found or submission fails
        """
        if not self.is_slurm_available():
            raise RuntimeError("SLURM 'sbatch' command not found")

        start_time = time.time()

        try:
            # Submit job
            result = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True
            )

            # Extract job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            self.logger.info(f"Submitted SLURM job: {job_id}")

            if not wait:
                return ExecutionResult(
                    success=True,
                    return_code=0,
                    job_id=job_id,
                    metadata={'status': 'submitted'}
                )

            # Monitor job
            return self.monitor_slurm_job(
                job_id=job_id,
                poll_interval=poll_interval,
                max_wait_time=max_wait_time
            )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"SLURM submission failed: {e.stderr}")
            return ExecutionResult(
                success=False,
                return_code=e.returncode,
                error_message=e.stderr,
                duration_seconds=time.time() - start_time
            )

    def monitor_slurm_job(
        self,
        job_id: str,
        poll_interval: int = 60,
        max_wait_time: int = 3600,
        on_status_change: Optional[Callable[[str, str], None]] = None
    ) -> ExecutionResult:
        """
        Monitor a SLURM job until completion or timeout.

        Args:
            job_id: SLURM job ID to monitor
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait
            on_status_change: Callback(job_id, new_status) on status change

        Returns:
            ExecutionResult with final job status
        """
        start_time = time.time()
        last_status = None

        self.logger.info(f"Monitoring SLURM job {job_id}")

        while (time.time() - start_time) < max_wait_time:
            try:
                # Check if job is still in queue
                queue_result = subprocess.run(
                    ["squeue", "-j", job_id, "-h"],
                    capture_output=True,
                    text=True
                )

                if not queue_result.stdout.strip():
                    # Job no longer in queue, check final status
                    status = self._get_slurm_job_status(job_id)

                    duration = time.time() - start_time

                    if status == "COMPLETED":
                        self.logger.info(f"Job {job_id} completed successfully")
                        return ExecutionResult(
                            success=True,
                            return_code=0,
                            job_id=job_id,
                            duration_seconds=duration,
                            metadata={'slurm_status': status}
                        )
                    elif status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                        self.logger.error(f"Job {job_id} ended with status: {status}")
                        return ExecutionResult(
                            success=False,
                            return_code=1,
                            job_id=job_id,
                            duration_seconds=duration,
                            error_message=f"SLURM status: {status}",
                            metadata={'slurm_status': status}
                        )
                    else:
                        # Unknown status, consider it done
                        self.logger.warning(f"Job {job_id} has unknown status: {status}")
                        return ExecutionResult(
                            success=True,
                            return_code=0,
                            job_id=job_id,
                            duration_seconds=duration,
                            metadata={'slurm_status': status}
                        )
                else:
                    # Job still running - parse status
                    current_status = self._parse_squeue_output(queue_result.stdout)

                    if current_status != last_status:
                        self.logger.info(f"Job {job_id}: {current_status}")
                        if on_status_change:
                            on_status_change(job_id, current_status)
                        last_status = current_status

            except subprocess.SubprocessError as e:
                self.logger.warning(f"Error checking job status: {e}")

            time.sleep(poll_interval)

        # Timeout
        self.logger.warning(f"Timeout waiting for job {job_id}")
        return ExecutionResult(
            success=False,
            return_code=-1,
            job_id=job_id,
            duration_seconds=time.time() - start_time,
            error_message=f"Timeout after {max_wait_time}s"
        )

    def _get_slurm_job_status(self, job_id: str) -> str:
        """Get final status of a completed SLURM job."""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-o", "State", "-n", "--parsable2"],
                capture_output=True,
                text=True
            )
            # First line contains the status
            status = result.stdout.strip().split('\n')[0].split('|')[0]
            return status.strip()
        except (subprocess.SubprocessError, OSError, IndexError):
            return "UNKNOWN"

    def _parse_squeue_output(self, output: str) -> str:
        """Parse squeue output to get human-readable status."""
        lines = output.strip().split('\n')
        pending = sum(1 for line in lines if 'PENDING' in line or ' PD ' in line)
        running = sum(1 for line in lines if 'RUNNING' in line or ' R ' in line)
        return f"{running} running, {pending} pending"

    # =========================================================================
    # Parallel Execution Helpers (absorbed from ModelExecutor)
    # =========================================================================

    def create_gru_parallel_script(
        self,
        model_exe: Path,
        file_manager: Path,
        log_dir: Path,
        total_grus: int,
        grus_per_job: int,
        job_name: str,
        time_limit: str = "03:00:00",
        memory: str = "4G"
    ) -> str:
        """
        Create a SLURM script for GRU-parallel execution.

        Args:
            model_exe: Path to model executable
            file_manager: Path to file manager/control file
            log_dir: Directory for log files
            total_grus: Total number of GRUs to process
            grus_per_job: Number of GRUs per array task
            job_name: SLURM job name
            time_limit: Time limit per task
            memory: Memory per task

        Returns:
            Complete SLURM batch script
        """
        n_array_jobs = max(1, -(-total_grus // grus_per_job)) - 1  # Ceiling div, 0-based

        slurm_config = SlurmJobConfig(
            job_name=job_name,
            time_limit=time_limit,
            memory=memory,
            array_size=n_array_jobs,
            output_pattern=str(log_dir / f"{job_name}_%A_%a.out"),
            error_pattern=str(log_dir / f"{job_name}_%A_%a.err")
        )

        setup = [
            f"mkdir -p {log_dir}",
            "",
            "# Calculate GRU range for this task",
            f"gru_start=$(( ({grus_per_job} * $SLURM_ARRAY_TASK_ID) + 1 ))",
            f"gru_end=$(( gru_start + {grus_per_job} - 1 ))",
            "",
            f"if [ $gru_end -gt {total_grus} ]; then",
            f"    gru_end={total_grus}",
            "fi",
            "",
            'echo "Processing GRUs $gru_start to $gru_end"',
        ]

        commands = [
            "for gru in $(seq $gru_start $gru_end); do",
            '    echo "Starting GRU $gru"',
            f"    {model_exe} -g $gru 1 -m {file_manager}",
            "    exit_code=$?",
            "    if [ $exit_code -ne 0 ]; then",
            '        echo "Model failed for GRU $gru with exit code $exit_code"',
            "        exit 1",
            "    fi",
            '    echo "Completed GRU $gru"',
            "done",
        ]

        return self.create_slurm_script(
            config=slurm_config,
            commands=commands,
            setup_commands=setup
        )

    def estimate_optimal_grus_per_job(
        self,
        total_grus: int,
        min_jobs: int = 10,
        max_jobs: int = 500,
        ideal_grus_per_job: int = 50
    ) -> int:
        """
        Estimate optimal number of GRUs per SLURM job.

        Args:
            total_grus: Total GRUs in domain
            min_jobs: Minimum number of jobs to create
            max_jobs: Maximum jobs (prevent queue flooding)
            ideal_grus_per_job: Target GRUs per job

        Returns:
            Optimal GRUs per job
        """
        if total_grus <= min_jobs:
            return 1

        if total_grus <= ideal_grus_per_job * min_jobs:
            return max(1, total_grus // min_jobs)

        ideal_jobs = total_grus // ideal_grus_per_job

        if ideal_jobs <= max_jobs:
            grus_per_job = ideal_grus_per_job
        else:
            grus_per_job = -(-total_grus // max_jobs)

        # Scale up for very large domains
        if total_grus > 10000:
            scale_factor = min(3.0, total_grus / 10000)
            grus_per_job = int(grus_per_job * scale_factor)

        return min(grus_per_job, total_grus)

    # =========================================================================
    # Convenience Execution Methods (absorbed from ModelExecutor)
    # =========================================================================

    def run_with_retry(
        self,
        command: Union[List[str], str],
        log_file: Path,
        max_attempts: int = 3,
        retry_delay: int = 5,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute subprocess with automatic retry on failure.

        Args:
            command: Command to execute
            log_file: Path for log file
            max_attempts: Maximum retry attempts
            retry_delay: Seconds between retries
            **kwargs: Additional arguments for execute_subprocess

        Returns:
            ExecutionResult from final attempt
        """
        # Initialize last_result to handle case where loop might not execute
        last_result = ExecutionResult(
            success=False,
            return_code=-1,
            error_message="Execution failed to start or all attempts failed",
            metadata={'command': str(command)}
        )

        for attempt in range(1, max_attempts + 1):
            self.logger.debug(f"Attempt {attempt}/{max_attempts}")

            # Add attempt number to log file name
            attempt_log = log_file.with_suffix(f".attempt{attempt}.log")

            result = self.execute_subprocess(
                command=command,
                log_file=attempt_log,
                check=False,
                **kwargs
            )

            if result.success:
                return result

            last_result = result

            if attempt < max_attempts:
                self.logger.warning(f"Attempt {attempt} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        return last_result

    def execute_in_mode(
        self,
        mode: ExecutionMode,
        command: Union[List[str], str],
        log_file: Path,
        slurm_config: Optional[SlurmJobConfig] = None,
        wait_for_slurm: bool = True,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute command in specified mode (local or SLURM).

        Args:
            mode: Execution mode (LOCAL or SLURM)
            command: Command to execute
            log_file: Path for log file
            slurm_config: SLURM configuration (required for SLURM mode)
            wait_for_slurm: Wait for SLURM job completion
            **kwargs: Additional arguments for execute_subprocess

        Returns:
            ExecutionResult
        """
        if mode == ExecutionMode.LOCAL:
            return self.execute_subprocess(command, log_file, **kwargs)

        elif mode in (ExecutionMode.SLURM, ExecutionMode.SLURM_ARRAY):
            if not slurm_config:
                raise ValueError("slurm_config required for SLURM execution")

            if not self.is_slurm_available():
                self.logger.warning("SLURM not available, falling back to local execution")
                return self.execute_subprocess(command, log_file, **kwargs)

            # Create script
            cmd_list = command if isinstance(command, list) else [command]
            script_content = self.create_slurm_script(
                config=slurm_config,
                commands=cmd_list
            )

            # Write script
            script_path = log_file.parent / f"{slurm_config.job_name}.sh"
            script_path.write_text(script_content, encoding='utf-8')
            if os.name != 'nt':
                script_path.chmod(0o755)

            # Submit
            return self.submit_slurm_job(
                script_path=script_path,
                wait=wait_for_slurm
            )

        else:
            raise ValueError(f"Unknown execution mode: {mode}")

    # =========================================================================
    # File Verification
    # =========================================================================

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
