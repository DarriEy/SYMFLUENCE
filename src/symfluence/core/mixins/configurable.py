"""
Configurable mixin for SYMFLUENCE modules.

Provides a unified mixin combining logging, config, project context,
file utils, validation, and timing capabilities.
"""

import warnings

from .logging import LoggingMixin
from .project import ProjectContextMixin
from .file_utils import FileUtilsMixin
from .validation import ValidationMixin
from .timing import TimingMixin


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
        warnings.warn(
            "_resolve_config_value is deprecated, use _get_config_value instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self._get_config_value(typed_accessor, default)
