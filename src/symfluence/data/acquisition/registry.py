"""
Acquisition Registry for SYMFLUENCE

Provides a central registry for data acquisition handlers.
Uses standardized BaseRegistry pattern with lowercase key normalization.
"""
from typing import Dict, Type, Any, List, Union, TYPE_CHECKING
import logging

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.data.base_registry import BaseRegistry

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class AcquisitionRegistry(BaseRegistry):
    """
    Registry for data acquisition handlers.

    Implements the Registry Pattern to enable pluggable data sources. Each
    dataset type (ERA5, CARRA, RDRS, etc.) registers a handler that knows
    how to download and preprocess that specific data source.

    Registered Handlers (typical):
        - ERA5: Global reanalysis from ECMWF (via CDS or ARCO)
        - CARRA: Arctic regional reanalysis
        - CERRA: European regional reanalysis
        - RDRS: Canadian regional reanalysis
        - AORC: NOAA Analysis of Record for Calibration
        - CONUS404: High-resolution US forcing data
        - HRRR: High-Resolution Rapid Refresh

    Usage:
        # Handler registration (in handler module):
        @AcquisitionRegistry.register('ERA5')
        class ERA5Acquirer(BaseAcquisitionHandler):
            ...

        # Handler retrieval (in acquisition service):
        handler = AcquisitionRegistry.get_handler('ERA5', config, logger)
        output_path = handler.download(output_dir)

    All dataset names are normalized to lowercase for case-insensitive matching.
    """

    _handlers: Dict[str, Type] = {}

    @classmethod
    def get_handler(  # type: ignore[override]
        cls,
        dataset_name: str,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger
    ):
        """
        Get an instance of the appropriate acquisition handler.

        Args:
            dataset_name: Name of the dataset (case-insensitive)
            config: Configuration (SymfluenceConfig or dict for backward compatibility)
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
