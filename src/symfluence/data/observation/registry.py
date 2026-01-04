"""
Observation Registry for SYMFLUENCE

Provides a central registry for observational data handlers (GRACE, MODIS, etc.).
"""
from typing import Dict, Type, Any
from pathlib import Path

class ObservationRegistry:
    _handlers: Dict[str, Type] = {}

    @classmethod
    def register(cls, observation_type: str):
        """Decorator to register an observation handler."""
        def decorator(handler_class):
            cls._handlers[observation_type.upper()] = handler_class
            return handler_class
        return decorator

    @classmethod
    def get_handler(cls, observation_type: str, config: Dict[str, Any], logger):
        """Get an instance of the appropriate observation handler."""
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
        return sorted(list(cls._handlers.keys()))

    @classmethod
    def is_registered(cls, observation_type: str) -> bool:
        return observation_type.upper() in cls._handlers
