"""
BaseRegistry - Standardized registry pattern for SYMFLUENCE.

This module provides a consistent registry pattern used across:
- AcquisitionRegistry: Data acquisition handlers
- DatasetRegistry: Dataset preprocessing handlers
- ObservationRegistry: Observation data handlers

All registries use lowercase keys internally for consistency.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, List, Optional, TypeVar, Generic
import logging


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
    """

    _handlers: Dict[str, Type[T]] = {}

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
        return cls._normalize_key(name) in cls._handlers

    @classmethod
    def clear(cls) -> None:
        """Clear all registered handlers (mainly for testing)."""
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
