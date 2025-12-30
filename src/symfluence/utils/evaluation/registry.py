"""
Evaluation Registry for SYMFLUENCE

Provides a central registry for performance evaluation handlers.
"""
from typing import Dict, Type, Any, Optional
from pathlib import Path

class EvaluationRegistry:
    _handlers: Dict[str, Type] = {}

    @classmethod
    def register(cls, variable_type: str):
        """Decorator to register an evaluation handler."""
        def decorator(handler_class):
            cls._handlers[variable_type.upper()] = handler_class
            return handler_class
        return decorator

    @classmethod
    def get_evaluator(cls, variable_type: str, config: Dict[str, Any], logger):
        """Get an instance of the appropriate evaluation handler."""
        var_type_upper = variable_type.upper()
        if var_type_upper not in cls._handlers:
            return None
        
        handler_class = cls._handlers[var_type_upper]
        return handler_class(config, logger)

    @classmethod
    def list_evaluators(cls) -> list:
        return sorted(list(cls._handlers.keys()))
