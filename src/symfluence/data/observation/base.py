"""
Base Observation Handler for SYMFLUENCE
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import geopandas as gpd
from symfluence.core import ConfigurableMixin
from symfluence.geospatial.coordinate_utils import CoordinateUtilsMixin

class BaseObservationHandler(ABC, ConfigurableMixin, CoordinateUtilsMixin):
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        
        # Standard attributes are now provided as properties by ConfigurableMixin:
        # self.domain_name, self.data_dir, self.project_dir, self.config_dict, self.logger
        
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
