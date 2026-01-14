"""Data handling utilities."""

from typing import Any

# Lazy imports to avoid circular import issues and allow submodule access
# even if some imports fail
try:
    from .path_manager import PathManager, PathManagerMixin, create_path_manager
    from .base_registry import BaseRegistry, HandlerRegistry
except ImportError:
    # Allow package to load even if these fail - submodules like utils
    # should still be accessible
    PathManager: Any = None  # type: ignore[no-redef]
    PathManagerMixin: Any = None  # type: ignore[no-redef]
    create_path_manager: Any = None  # type: ignore[no-redef]
    BaseRegistry: Any = None  # type: ignore[no-redef]
    HandlerRegistry: Any = None  # type: ignore[no-redef]

__all__ = [
    'PathManager',
    'PathManagerMixin',
    'create_path_manager',
    'BaseRegistry',
    'HandlerRegistry',
]
