"""
Core mixins for SYMFLUENCE modules.

Provides base mixins for logging, configuration, and project context that
other utilities and mixins can build upon.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, ContextManager
from contextlib import contextmanager


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
    Mixin for classes that use configuration settings.

    Provides a standardized way to access configuration regardless of whether
    it's a dictionary or a typed SymfluenceConfig object.
    """

    @property
    def config_dict(self) -> Dict[str, Any]:
        """
        Standardized access to configuration as a dictionary.

        Handles both dict-based and SymfluenceConfig-based configurations.
        """
        # Prioritize _config_dict if set directly (backward compatibility)
        if hasattr(self, '_config_dict'):
            return self._config_dict

        # Try self.typed_config or self.config, skipping None values
        cfg = getattr(self, 'typed_config', None)
        if cfg is None:
            cfg = getattr(self, 'config', None)
        
        if cfg is not None:
            # If it has a to_dict method, use it
            to_dict_func = getattr(cfg, 'to_dict', None)
            if callable(to_dict_func):
                try:
                    # Prefer flattened output if supported (standard for SymfluenceConfig)
                    return to_dict_func(flatten=True)
                except (TypeError, ValueError, AttributeError):
                    try:
                        return to_dict_func()
                    except (TypeError, ValueError, AttributeError):
                        pass
            
            # Fallback to direct dict check
            if isinstance(cfg, dict):
                return cfg

        return {}

    @config_dict.setter
    def config_dict(self, value: Dict[str, Any]) -> None:
        """Set the configuration dictionary (backward compatibility)."""
        self._config_dict = value

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with a default.

        Handles both dictionary-based and object-based configuration.
        """
        # Try typed_config first if it exists
        typed_cfg = getattr(self, 'typed_config', None)
        if typed_cfg is not None:
            # If it's a Pydantic model (which SymfluenceConfig is)
            if hasattr(typed_cfg, key.lower()):
                return getattr(typed_cfg, key.lower())

        # Fallback to config_dict
        return self.config_dict.get(key, default)

    def _resolve_config_value(self, typed_accessor: Any, dict_key: str,
                              default: Any = None) -> Any:
        """
        Resolve configuration value from typed or dict config.

        Phase 3: Prioritizes typed config access, falls back to dict only for backward compatibility.

        Args:
            typed_accessor: Callable or value from typed config (e.g.,
                           lambda: self.config.domain.time_start)
            dict_key: Key to use with dict config (fallback)
            default: Default value if key not found

        Returns:
            Resolved configuration value
        """
        val = None
        typed_cfg = getattr(self, 'typed_config', getattr(self, 'config', None))
        
        # Check if typed_cfg is actually a SymfluenceConfig (or similar object)
        # and not a dict
        if typed_cfg is not None and not isinstance(typed_cfg, dict):
            # Handle callable (lambda) or direct value
            if callable(typed_accessor):
                try:
                    val = typed_accessor()
                except (AttributeError, KeyError):
                    val = None
            else:
                val = typed_accessor
        
        # If val is None (either missing from typed config or typed config not used),
        # fall back to dict config (deprecated path)
        if val is None:
            return self.config_dict.get(dict_key, default)
        
        return val


class ProjectContextMixin(ConfigMixin):
    """
    Mixin providing standard project context attributes.

    Extracts core project parameters (data_dir, domain_name, project_dir)
    from the configuration, providing a consistent interface across modules.
    """

    @property
    def data_dir(self) -> Path:
        """Root data directory from configuration."""
        _data_dir = getattr(self, '_data_dir', None)
        if _data_dir is not None:
            return Path(_data_dir)
        
        # Prioritize typed config
        typed_cfg = getattr(self, 'typed_config', None)
        if typed_cfg is not None and not isinstance(typed_cfg, dict):
            try:
                return typed_cfg.system.data_dir
            except AttributeError:
                pass
        
        return Path(self.config_dict.get('SYMFLUENCE_DATA_DIR', '.'))

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
            
        # Prioritize typed config
        typed_cfg = getattr(self, 'typed_config', None)
        if typed_cfg is not None and not isinstance(typed_cfg, dict):
            try:
                return typed_cfg.domain.name
            except AttributeError:
                pass
            
        return self.config_dict.get('DOMAIN_NAME', 'domain')

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
        logger.info(f"Starting task: {task_name}")
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
    pass

