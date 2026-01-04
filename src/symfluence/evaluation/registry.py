"""
Evaluation Registry for SYMFLUENCE

Provides a central registry for performance evaluation handlers.
"""
from typing import Dict, Type, Any, Optional
from pathlib import Path
import logging

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
    def get_evaluator(
        cls,
        variable_type: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        project_dir: Optional[Path] = None,
    ):
        """Get an instance of the appropriate evaluation handler."""
        var_type_upper = variable_type.upper()
        if var_type_upper not in cls._handlers:
            return None
        
        handler_class = cls._handlers[var_type_upper]
        handler_logger = logger or logging.getLogger(handler_class.__name__)
        handler_project_dir = project_dir or Path(".")
        return handler_class(config, handler_project_dir, handler_logger)

    @classmethod
    def list_evaluators(cls) -> list:
        return sorted(list(cls._handlers.keys()))
