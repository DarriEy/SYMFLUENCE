"""Observation Registry for SYMFLUENCE

Provides a central registry for observational data handlers (GRACE, MODIS, etc.).

This module implements a plugin-style registry pattern that allows observation handlers
to self-register and be dynamically instantiated by type string. This decouples handler
implementations from the core acquisition system and enables easy addition of new
data sources without modifying the registry code.

Example:
    Register a custom handler:

    >>> @ObservationRegistry.register('CUSTOM_SENSOR')
    ... class CustomHandler(BaseObservationHandler):
    ...     def acquire(self): ...
    ...     def process(self, input_path): ...

    Get a handler instance:

    >>> handler = ObservationRegistry.get_handler('CUSTOM_SENSOR', config, logger)
    >>> raw_data = handler.acquire()
    >>> processed = handler.process(raw_data)
"""
from typing import Dict, Type, Any, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class ObservationRegistry:
    """Plugin registry for observation data handlers.

    This registry uses the plugin pattern to manage observation handler classes.
    Handlers register themselves using the @register() class method decorator,
    enabling dynamic instantiation without hardcoded imports.

    Class Attributes:
        _handlers (dict): Maps observation type strings (uppercase) to handler classes.
    """

    _handlers: Dict[str, Type] = {}

    @classmethod
    def register(cls, observation_type: str):
        """Decorator to register an observation handler class.

        This decorator registers a handler class for a given observation type string.
        The type is converted to uppercase for case-insensitive lookups. The decorated
        class must implement BaseObservationHandler interface.

        Args:
            observation_type: Case-insensitive type identifier for the observation
                (e.g., 'GRACE', 'MODIS_SNOW', 'USGS'). Will be stored in uppercase.

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Example:
            >>> @ObservationRegistry.register('GRACE')
            ... class GRACEHandler(BaseObservationHandler):
            ...     pass
        """
        def decorator(handler_class):
            cls._handlers[observation_type.upper()] = handler_class
            return handler_class
        return decorator

    @classmethod
    def get_handler(
        cls,
        observation_type: str,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger
    ):
        """
        Get an instance of the appropriate observation handler.

        Args:
            observation_type: Type of observation (case-insensitive)
            config: Configuration (SymfluenceConfig or dict for backward compatibility)
            logger: Logger instance

        Returns:
            Handler instance
        """
        obs_type_upper = observation_type.upper()
        if obs_type_upper not in cls._handlers:
            available = ', '.join(cls._handlers.keys())
            raise ValueError(
                f"Unknown observation type: '{observation_type}'. "
                f"Available: {available}"
            )

        handler_class = cls._handlers[obs_type_upper]
        return handler_class(config, logger)

    @classmethod
    def list_observations(cls) -> list:
        """Get sorted list of all registered observation types.

        Returns:
            list: Registered observation type strings in uppercase, sorted alphabetically.

        Example:
            >>> ObservationRegistry.list_observations()
            ['GLEAM', 'GRACE', 'MODIS_ET', 'MODIS_SNOW', 'USGS']
        """
        return sorted(list(cls._handlers.keys()))

    @classmethod
    def is_registered(cls, observation_type: str) -> bool:
        """Check if an observation type is registered.

        Args:
            observation_type: Case-insensitive type identifier.

        Returns:
            bool: True if the type is registered, False otherwise.

        Example:
            >>> ObservationRegistry.is_registered('GRACE')
            True
            >>> ObservationRegistry.is_registered('unknown')
            False
        """
        return observation_type.upper() in cls._handlers
