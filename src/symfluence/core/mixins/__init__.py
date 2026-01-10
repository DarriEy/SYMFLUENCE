"""
Core mixins for SYMFLUENCE modules.

Provides base mixins for logging, configuration, and project context that
other utilities and mixins can build upon.

Usage:
    from symfluence.core.mixins import ConfigurableMixin

    class MyProcessor(ConfigurableMixin):
        def process(self):
            self.logger.info(f"Processing {self.domain_name}")
            self.ensure_dir(self.project_dir)
"""

from .logging import LoggingMixin
from .config import ConfigMixin
from .shapefile import ShapefileAccessMixin
from .project import ProjectContextMixin
from .file_utils import FileUtilsMixin
from .validation import ValidationMixin
from .timing import TimingMixin
from .configurable import ConfigurableMixin

__all__ = [
    # Base mixins
    "LoggingMixin",
    "ConfigMixin",
    "ShapefileAccessMixin",
    "ProjectContextMixin",
    "FileUtilsMixin",
    "ValidationMixin",
    "TimingMixin",
    # Combined mixin (recommended for most use cases)
    "ConfigurableMixin",
]
