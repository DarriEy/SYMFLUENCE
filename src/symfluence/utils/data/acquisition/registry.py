"""
Acquisition Registry for SYMFLUENCE

Provides a central registry for data acquisition handlers.
"""
from typing import Dict, Type, Any
from pathlib import Path

class AcquisitionRegistry:
    _handlers: Dict[str, Type] = {}

    @classmethod
    def register(cls, dataset_name: str):
        """Decorator to register an acquisition handler."""
        def decorator(handler_class):
            cls._handlers[dataset_name.upper()] = handler_class
            return handler_class
        return decorator

    @classmethod
    def get_handler(cls, dataset_name: str, config: Dict[str, Any], logger):
        """Get an instance of the appropriate acquisition handler."""
        dataset_name_upper = dataset_name.upper()
        if dataset_name_upper not in cls._handlers:
            available = ', '.join(cls._handlers.keys())
            raise ValueError(
                f"Unknown acquisition dataset: '{dataset_name}'. "
                f"Available datasets: {available}"
            )
        
        handler_class = cls._handlers[dataset_name_upper]
        return handler_class(config, logger)

    @classmethod
    def list_datasets(cls) -> list:
        return sorted(list(cls._handlers.keys()))

    @classmethod
    def is_registered(cls, dataset_name: str) -> bool:
        return dataset_name.upper() in cls._handlers
