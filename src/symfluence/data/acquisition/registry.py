"""
Acquisition Registry for SYMFLUENCE

Provides a central registry for data acquisition handlers.
Uses standardized BaseRegistry pattern with lowercase key normalization.
"""
from typing import Dict, Type, Any, List
import logging

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.data.base_registry import BaseRegistry


class AcquisitionRegistry(BaseRegistry):
    """
    Registry for data acquisition handlers.

    Handlers are registered using the @register decorator and retrieved
    using get_handler(). All keys are normalized to lowercase.
    """

    _handlers: Dict[str, Type] = {}

    @classmethod
    def get_handler(
        cls,
        dataset_name: str,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Get an instance of the appropriate acquisition handler.

        Args:
            dataset_name: Name of the dataset (case-insensitive)
            config: Configuration dictionary
            logger: Logger instance

        Returns:
            Handler instance

        Raises:
            DataAcquisitionError: If handler not found
        """
        try:
            handler_class = cls._get_handler_class(dataset_name)
            return handler_class(config, logger)
        except ValueError as e:
            raise DataAcquisitionError(str(e))

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names (alias for list_handlers)."""
        return cls.list_handlers()
