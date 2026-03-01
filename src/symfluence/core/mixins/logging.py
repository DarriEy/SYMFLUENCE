# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Logging mixin for SYMFLUENCE modules.

Provides standardized logger access for classes.
"""

import logging


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
