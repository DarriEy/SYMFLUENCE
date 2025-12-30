"""
Objective Registry for SYMFLUENCE

Provides a central registry for objective functions used in calibration.
"""
from typing import Dict, Type, Any, Optional
from pathlib import Path

class ObjectiveRegistry:
    _handlers: Dict[str, Type] = {}

    @classmethod
    def register(cls, objective_type: str):
        """Decorator to register an objective function handler."""
        def decorator(handler_class):
            cls._handlers[objective_type.upper()] = handler_class
            return handler_class
        return decorator

    @classmethod
    def get_objective(cls, objective_type: str, config: Dict[str, Any], logger):
        """Get an instance of the appropriate objective handler."""
        obj_type_upper = objective_type.upper()
        if obj_type_upper not in cls._handlers:
            return None
        
        handler_class = cls._handlers[obj_type_upper]
        return handler_class(config, logger)

    @classmethod
    def list_objectives(cls) -> list:
        return sorted(list(cls._handlers.keys()))
