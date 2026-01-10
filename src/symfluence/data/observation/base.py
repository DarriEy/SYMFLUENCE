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
    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger
    ):
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
        """Acquire raw data (download or locate)."""
        pass

    @abstractmethod
    def process(self, input_path: Path) -> Path:
        """Process raw data into SYMFLUENCE-standard formats."""
        pass
