# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Logging mixin for SYMFLUENCE modules.

Provides standardized logger access for classes.
"""
from __future__ import annotations

import logging


def _class_logger_name(cls: type) -> str:
    """Logger name for ``cls``, rooted at the ``symfluence`` hierarchy.

    ``symfluence.data.foo.Bar`` for a class ``Bar`` in ``symfluence.data.foo``;
    non-symfluence modules (e.g. test helpers) are nested under
    ``symfluence.`` so their records still flow to the configured handlers.
    """
    module = cls.__module__ or ''
    if module == 'symfluence':
        return f"symfluence.{cls.__name__}"
    if module.startswith('symfluence.'):
        return f"{module}.{cls.__name__}"
    return f"symfluence.{module}.{cls.__name__}"


class LoggingMixin:
    """
    Mixin providing standardized logger access.

    Ensures a logger is always available, defaulting to one named after the
    class (rooted at the ``symfluence`` logger hierarchy) if none is
    explicitly set.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        _logger = getattr(self, '_logger', None)
        if _logger is None:
            # Create a default logger (rooted at 'symfluence') if none exists
            self._logger = logging.getLogger(_class_logger_name(self.__class__))
            return self._logger
        return _logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        """Set the logger instance."""
        self._logger = value
