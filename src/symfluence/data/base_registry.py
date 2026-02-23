"""
BaseRegistry - Standardized registry pattern for SYMFLUENCE.

This module provides a consistent registry pattern used across:
- AcquisitionRegistry: Data acquisition handlers
- DatasetRegistry: Dataset preprocessing handlers
- ObservationRegistry: Observation data handlers

All registries use lowercase keys internally for consistency.

Phase 4 delegation shim: internal ``_handlers`` dicts are retained only as
a fallback for subclasses that have not yet set ``_r_registry_name``.
Subclasses that *do* set the attribute delegate all state to
``R.<_r_registry_name>``.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar('T')


class BaseRegistry(ABC, Generic[T]):
    """
    Abstract base class for handler registries.

    Provides consistent API for:
    - Registering handlers via decorator
    - Retrieving handler instances
    - Listing available handlers
    - Checking handler availability

    All keys are normalized to lowercase internally.

    Subclasses should set ``_r_registry_name`` to the corresponding attribute
    name on ``R`` (e.g. ``"acquisition_handlers"``).  When set, all state is
    delegated to the unified registry and the internal ``_handlers`` dict is
    unused.
    """

    _handlers: Dict[str, Type[T]] = {}

    # Subclasses set this to the R.* attribute name they delegate to.
    # When ``None``, the base class falls back to ``_handlers``.
    _r_registry_name: Optional[str] = None

    @classmethod
    def _get_r_registry(cls):
        """Return the unified ``R.<name>`` Registry instance, or ``None``."""
        if cls._r_registry_name is None:
            return None
        from symfluence.core.registries import R
        return getattr(R, cls._r_registry_name)

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Normalize registry key to lowercase."""
        return key.lower()

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a handler class.

        Args:
            name: Name to register the handler under

        Returns:
            Decorator function

        Example:
            @MyRegistry.register('era5')
            class ERA5Handler(BaseHandler):
                pass
        """
        def decorator(handler_class: Type[T]) -> Type[T]:
            r_reg = cls._get_r_registry()
            if r_reg is not None:
                warnings.warn(
                    f"{cls.__name__}.register() is deprecated; "
                    "use R.{}.add() or model_manifest() instead.".format(
                        cls._r_registry_name
                    ),
                    DeprecationWarning,
                    stacklevel=2,
                )
                r_reg.add(name, handler_class)
            else:
                # Fallback for subclasses without _r_registry_name
                normalized_name = cls._normalize_key(name)
                cls._handlers[normalized_name] = handler_class
            return handler_class
        return decorator

    @classmethod
    @abstractmethod
    def get_handler(cls, name: str, *args, **kwargs) -> T:
        """
        Get an instance of the appropriate handler.

        Args:
            name: Handler name
            *args: Positional arguments for handler constructor
            **kwargs: Keyword arguments for handler constructor

        Returns:
            Handler instance

        Raises:
            ValueError: If handler not found
        """
        pass

    @classmethod
    def _get_handler_class(cls, name: str) -> Type[T]:
        """
        Get the handler class for a given name.

        Args:
            name: Handler name

        Returns:
            Handler class

        Raises:
            ValueError: If handler not found
        """
        normalized_name = cls._normalize_key(name)
        r_reg = cls._get_r_registry()

        if r_reg is not None:
            handler = r_reg.get(normalized_name)
            if handler is None:
                available = ', '.join(sorted(r_reg.keys()))
                raise ValueError(
                    f"Unknown handler: '{name}'. Available: {available}"
                )
            return handler

        # Fallback for subclasses without _r_registry_name
        if normalized_name not in cls._handlers:
            available = ', '.join(sorted(cls._handlers.keys()))
            raise ValueError(
                f"Unknown handler: '{name}'. Available: {available}"
            )

        return cls._handlers[normalized_name]

    @classmethod
    def list_handlers(cls) -> List[str]:
        """
        List all registered handler names.

        Returns:
            Sorted list of handler names
        """
        r_reg = cls._get_r_registry()
        if r_reg is not None:
            return sorted(r_reg.keys())
        return sorted(cls._handlers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a handler is registered.

        Args:
            name: Handler name to check

        Returns:
            True if registered, False otherwise
        """
        r_reg = cls._get_r_registry()
        if r_reg is not None:
            return cls._normalize_key(name) in r_reg
        return cls._normalize_key(name) in cls._handlers

    @classmethod
    def clear(cls) -> None:
        """Clear all registered handlers (mainly for testing)."""
        r_reg = cls._get_r_registry()
        if r_reg is not None:
            r_reg.clear()
        cls._handlers.clear()


class HandlerRegistry(BaseRegistry[T]):
    """
    Concrete registry implementation with standard get_handler.

    Use this for simple registries where handlers have a consistent
    constructor signature.
    """

    @classmethod
    def get_handler(
        cls,
        name: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        **kwargs
    ) -> T:
        """
        Get an instance of the appropriate handler.

        Args:
            name: Handler name
            config: Configuration dictionary
            logger: Logger instance
            **kwargs: Additional arguments for handler constructor

        Returns:
            Handler instance
        """
        handler_class = cls._get_handler_class(name)
        # Cast to Any to allow calling constructor with standard args
        # as Mypy doesn't know the exact signature of the registered Type[T]
        from typing import cast
        return cast(Any, handler_class)(config, logger, **kwargs)
