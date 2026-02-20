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
    """Unified mixin: logging + config + project paths + file utils + validation + timing.

    Recommended mixin for most SYMFLUENCE components. Subclasses must set
    ``self.config`` to a SymfluenceConfig before using config-dependent properties.
    See parent mixins for available methods and properties.
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
