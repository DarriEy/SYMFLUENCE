"""
Dataset Registry for SYMFLUENCE

Provides a central registry for dataset preprocessing handlers.
Uses standardized BaseRegistry pattern with lowercase key normalization.
"""

from typing import Dict, Type, Any, List
from pathlib import Path
import logging

from symfluence.utils.data.base_registry import BaseRegistry


class DatasetRegistry(BaseRegistry):
    """
    Registry for dataset preprocessing handlers.

    Handlers are registered using the @register decorator and retrieved
    using get_handler(). All keys are normalized to lowercase.
    """

    _handlers: Dict[str, Type] = {}

    @classmethod
    def get_handler(
        cls,
        dataset_name: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        project_dir: Path,
        **kwargs
    ):
        """
        Get an instance of the appropriate dataset handler.

        Args:
            dataset_name: Name of the dataset (case-insensitive)
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Project directory path
            **kwargs: Additional handler arguments

        Returns:
            Handler instance

        Raises:
            ValueError: If handler not found
        """
        handler_class = cls._get_handler_class(dataset_name)

        # Merge standard forcing parameters with any provided kwargs
        handler_kwargs = {
            "forcing_timestep_seconds": config.get("FORCING_TIME_STEP_SIZE", 3600),
            **kwargs
        }

        return handler_class(config, logger, project_dir, **handler_kwargs)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names (alias for list_handlers)."""
        return cls.list_handlers()