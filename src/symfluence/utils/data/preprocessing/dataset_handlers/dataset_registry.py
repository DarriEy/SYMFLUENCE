"""
Dataset Registry for SYMFLUENCE
"""

from typing import Dict, Type
from pathlib import Path


class DatasetRegistry:
    _handlers: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, dataset_name: str):
        def decorator(handler_class):
            cls._handlers[dataset_name.lower()] = handler_class
            return handler_class
        return decorator
    
    @classmethod
    def get_handler(cls, dataset_name: str, config: Dict, logger, project_dir: Path):
        dataset_name_lower = dataset_name.lower()
        
        if dataset_name_lower not in cls._handlers:
            available = ', '.join(cls._handlers.keys())
            raise ValueError(f"Unknown forcing dataset: '{dataset_name}'. Available: {available}")
        
        handler_class = cls._handlers[dataset_name_lower]
        
        # Pull standard forcing parameters
        kwargs = {
            "forcing_timestep_seconds": config.get("FORCING_TIME_STEP_SIZE", 3600)
        }
        
        return handler_class(config, logger, project_dir, **kwargs)
    
    @classmethod
    def list_datasets(cls) -> list:
        return list(cls._handlers.keys())
    
    @classmethod
    def is_registered(cls, dataset_name: str) -> bool:
        return dataset_name.lower() in cls._handlers