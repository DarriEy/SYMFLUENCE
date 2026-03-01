# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
State management exceptions.

Provides specific exception types for model state operations.
Reuses StateExchangeError from fews.exceptions for I/O failures.
"""

from symfluence.core.exceptions import SYMFLUENCEError


class StateError(SYMFLUENCEError):
    """Base exception for state management errors."""
    pass


class StateValidationError(StateError):
    """Raised when a state is incompatible with the target model."""
    pass
