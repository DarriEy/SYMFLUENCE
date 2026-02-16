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

from .types import StateFormat, StateMetadata, ModelState
from .mixin import StateCapableMixin
from .manager import StateManager
from .exceptions import StateError, StateValidationError

__all__ = [
    "StateFormat",
    "StateMetadata",
    "ModelState",
    "StateCapableMixin",
    "StateManager",
    "StateError",
    "StateValidationError",
]
