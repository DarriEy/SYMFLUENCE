# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
BaseRegistry - Standardized registry pattern for SYMFLUENCE.

This module provides a consistent lookup facade used across:
- AcquisitionRegistry: Data acquisition handlers
- DatasetRegistry: Dataset preprocessing handlers
- ObservationRegistry: Observation data handlers

All registries use lowercase keys internally for consistency.

Registration goes through the unified registry facade
(``R.<registry>.add()``); subclasses set ``_r_registry_name`` to the
corresponding attribute name on ``R`` (e.g. ``"acquisition_handlers"``)
and all state lives there.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar('T')


class BaseRegistry(ABC, Generic[T]):
    """
    Abstract base class for handler registry lookup facades.

    Provides consistent API for:
    - Retrieving handler instances
    - Listing available handlers
    - Checking handler availability

    All keys are normalized to lowercase internally.

    Subclasses must set ``_r_registry_name`` to the corresponding attribute
    name on ``R`` (e.g. ``"acquisition_handlers"``); all state is delegated
    to that unified registry.
    """

    # Subclasses set this to the R.* attribute name they delegate to.
    _r_registry_name: Optional[str] = None

    @classmethod
    def _get_r_registry(cls):
        """Return the unified ``R.<name>`` Registry instance."""
        if cls._r_registry_name is None:
            raise TypeError(
                f"{cls.__name__} must set _r_registry_name to the R.* "
                "registry attribute it delegates to"
            )
        from symfluence.core.registries import R
        return getattr(R, cls._r_registry_name)

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Normalize registry key to lowercase."""
        return key.lower()

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

        handler = r_reg.get(normalized_name)
        if handler is None:
            available = ', '.join(sorted(r_reg.keys()))
            raise ValueError(
                f"Unknown handler: '{name}'. Available: {available}"
            )
        return handler

    @classmethod
    def list_handlers(cls) -> List[str]:
        """
        List all registered handler names.

        Returns:
            Sorted list of handler names
        """
        return sorted(cls._get_r_registry().keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a handler is registered.

        Args:
            name: Handler name to check

        Returns:
            True if registered, False otherwise
        """
        return cls._normalize_key(name) in cls._get_r_registry()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered handlers (mainly for testing)."""
        cls._get_r_registry().clear()


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
