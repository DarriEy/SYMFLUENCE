"""
Base Observation Handler for SYMFLUENCE
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
import pandas as pd
import geopandas as gpd
from symfluence.core import ConfigurableMixin
from symfluence.geospatial.coordinate_utils import CoordinateUtilsMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseObservationHandler(ABC, ConfigurableMixin, CoordinateUtilsMixin):
    """Abstract base class for observation data handlers.

    This class defines the interface for acquiring and processing observational data
    (e.g., GRACE water storage, MODIS snow cover, streamflow observations). Subclasses
    implement handlers for specific data sources.

    The handler is responsible for two main tasks:
    1. **Acquisition**: Downloading or locating raw data files from remote or local sources
    2. **Processing**: Converting raw data into SYMFLUENCE-standard formats (e.g., gridded
       NetCDF with standardized variable names and spatial/temporal coordinates)

    Attributes:
        bbox (dict): Bounding box coordinates for spatial filtering (parsed from config)
        start_date (pd.Timestamp): Experiment start date for temporal filtering
        end_date (pd.Timestamp): Experiment end date for temporal filtering
        logger: Logger instance for diagnostic and error messages
    """

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger
    ):
        """Initialize the observation handler.

        Args:
            config: SYMFLUENCE configuration object or dict. If dict, will be converted
                to SymfluenceConfig for type safety and validation.
            logger: Python logger instance for recording acquisition/processing events.

        Raises:
            ValueError: If config dict cannot be converted to SymfluenceConfig.
        """
        from symfluence.core.config.models import SymfluenceConfig

        # Auto-convert dict to typed config for backward compatibility
        if isinstance(config, dict):
            self._config = SymfluenceConfig(**config)
        else:
            self._config = config

        self.logger = logger

        # Standard attributes use config_dict (from ConfigMixin) for compatibility
        self.bbox = self._parse_bbox(self.config_dict.get('BOUNDING_BOX_COORDS'))
        self.start_date = pd.to_datetime(self.config_dict.get('EXPERIMENT_TIME_START'))
        self.end_date = pd.to_datetime(self.config_dict.get('EXPERIMENT_TIME_END'))

    @abstractmethod
    def acquire(self) -> Path:
        """Acquire raw data from the source (download or locate local files).

        Subclasses must implement this method to retrieve raw observational data
        from their respective data source (e.g., GRACE server, USGS database).
        Implementations should handle authentication, error handling, and logging.

        Returns:
            Path: Local filesystem path to the acquired raw data file(s).

        Raises:
            IOError: If data cannot be retrieved from the source.
            ValueError: If data for the specified spatial/temporal bounds doesn't exist.
        """
        pass

    @abstractmethod
    def process(self, input_path: Path) -> Path:
        """Process raw data into SYMFLUENCE-standard formats.

        Subclasses must implement this method to transform raw data into a
        standardized format (typically gridded NetCDF) with:
        - Standardized variable names (e.g., 'SWE', 'ET', 'streamflow')
        - Proper spatial coordinates (lat/lon or projected CRS)
        - Proper temporal coordinates (datetime)
        - Required metadata (units, source, processing steps)

        Args:
            input_path: Path to raw data file(s) from acquire().

        Returns:
            Path: Local filesystem path to processed data file in standard format.

        Raises:
            IOError: If input file cannot be read.
            ValueError: If data cannot be processed into standard format.
        """
        pass
