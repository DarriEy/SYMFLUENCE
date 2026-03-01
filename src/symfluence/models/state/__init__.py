# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
State management for SYMFLUENCE models.

Provides a unified interface for saving and restoring model state,
enabling ensemble forecasting, warm-starting, and FEWS operational cycling.

Public API:
    - StateFormat: Supported state storage formats
    - StateMetadata: Immutable state descriptor
    - ModelState: State data container
    - StateCapableMixin: Opt-in mixin for model runners
    - StateManager: Orchestration and serialization
    - StateError, StateValidationError: Exceptions
"""

from .exceptions import StateError, StateValidationError
from .manager import StateManager
from .mixin import StateCapableMixin
from .types import ModelState, StateFormat, StateMetadata

__all__ = [
    "StateFormat",
    "StateMetadata",
    "ModelState",
    "StateCapableMixin",
    "StateManager",
    "StateError",
    "StateValidationError",
]
