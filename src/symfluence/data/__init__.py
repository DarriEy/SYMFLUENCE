"""Data handling utilities."""

# Lazy imports to avoid circular import issues and allow submodule access
# even if some imports fail
try:
    from .path_manager import PathManager, PathManagerMixin, create_path_manager
    from .base_registry import BaseRegistry, HandlerRegistry
except ImportError:
    # Allow package to load even if these fail - submodules like utils
    # should still be accessible
    PathManager = None
    PathManagerMixin = None
    create_path_manager = None
    BaseRegistry = None
    HandlerRegistry = None

__all__ = [
    'PathManager',
    'PathManagerMixin',
    'create_path_manager',
    'BaseRegistry',
    'HandlerRegistry',
]