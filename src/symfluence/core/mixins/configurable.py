"""
Configurable mixin for SYMFLUENCE modules.

Provides a unified mixin combining logging, config, project context,
file utils, validation, and timing capabilities.

This is the recommended mixin for most SYMFLUENCE components.

Mixin Hierarchy
---------------
::

    ConfigurableMixin
    ├── LoggingMixin          # self.logger property
    ├── ProjectContextMixin   # project paths (inherits ConfigMixin)
    │   └── ConfigMixin       # self.config + convenience properties
    ├── FileUtilsMixin        # ensure_dir, copy_file, safe_delete
    ├── ValidationMixin       # validate_config, validate_file, validate_dir
    └── TimingMixin           # time_limit context manager

Example
-------
>>> from symfluence.core.mixins import ConfigurableMixin
>>>
>>> class MyProcessor(ConfigurableMixin):
...     def __init__(self, config):
...         self.config = config  # Required: set config before using properties
...
...     def process(self):
...         self.logger.info(f"Processing {self.domain_name}")
...         output_dir = self.project_dir / "output"
...         self.ensure_dir(output_dir)
...         with self.time_limit("processing"):
...             # do work...
...             pass
"""

import warnings

from .logging import LoggingMixin
from .project import ProjectContextMixin
from .file_utils import FileUtilsMixin
from .validation import ValidationMixin
from .timing import TimingMixin


class ConfigurableMixin(LoggingMixin, ProjectContextMixin, FileUtilsMixin, ValidationMixin, TimingMixin):
    """
    Unified mixin for classes requiring logging, config, project context,
    file utils, validation, and timing.

    This is the **recommended mixin** for most SYMFLUENCE components that need
    to be aware of the project structure and configuration.

    Provides
    --------
    From LoggingMixin:
        - self.logger : logging.Logger
            Module-specific logger instance

    From ConfigMixin (via ProjectContextMixin):
        - self.config : SymfluenceConfig
            Typed configuration object (must be set by subclass)
        - self.config_dict : Dict[str, Any]
            Flattened config dictionary for legacy code
        - Convenience properties: experiment_id, domain_definition_method,
          time_start, time_end, calibration_period, evaluation_period,
          forcing_dataset, hydrological_model, routing_model, optimization_metric

    From ProjectContextMixin:
        - self.data_dir : Path
            Root data directory
        - self.domain_name : str
            Domain identifier
        - self.project_dir : Path
            Project directory ({data_dir}/domain_{domain_name})
        - Standard subdirectories: project_shapefiles_dir, project_forcing_dir,
          project_observations_dir, project_simulations_dir, project_settings_dir

    From FileUtilsMixin:
        - ensure_dir(path) : Path
            Create directory if it doesn't exist
        - copy_file(src, dst) : Path
            Copy file with logging
        - copy_tree(src, dst) : Path
            Copy directory tree
        - safe_delete(path) : bool
            Delete file/directory with error handling

    From ValidationMixin:
        - validate_config(required_keys) : bool
            Validate required config keys exist
        - validate_file(path) : bool
            Validate file exists
        - validate_dir(path) : bool
            Validate directory exists

    From TimingMixin:
        - time_limit(task_name) : ContextManager
            Log execution time of code block

    Requirements
    ------------
    Subclasses must set ``self.config`` to a SymfluenceConfig instance
    before accessing config-dependent properties.

    Example
    -------
    >>> class MyProcessor(ConfigurableMixin):
    ...     def __init__(self, config):
    ...         self.config = config
    ...
    ...     def run(self):
    ...         self.logger.info(f"Running for {self.domain_name}")
    ...         with self.time_limit("main processing"):
    ...             self.ensure_dir(self.project_dir / "output")
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
        warnings.warn(
            "_resolve_config_value is deprecated, use _get_config_value instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self._get_config_value(typed_accessor, default)
