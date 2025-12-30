"""
Base Observation Handler for SYMFLUENCE
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import geopandas as gpd

class BaseObservationHandler(ABC):
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME', 'domain')
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.bbox = self._parse_bbox(config.get('BOUNDING_BOX_COORDS'))
        self.start_date = pd.to_datetime(config.get('EXPERIMENT_TIME_START'))
        self.end_date = pd.to_datetime(config.get('EXPERIMENT_TIME_END'))

    def _parse_bbox(self, bbox_string: Optional[str]) -> Dict[str, float]:
        if not bbox_string:
            return {}
        coords = bbox_string.split('/')
        return {
            'lat_min': float(coords[2]),
            'lat_max': float(coords[0]),
            'lon_min': float(coords[3]),
            'lon_max': float(coords[1])
        }

    @abstractmethod
    def acquire(self) -> Path:
        """Acquire raw data (download or locate)."""
        pass

    @abstractmethod
    def process(self, input_path: Path) -> Path:
        """Process raw data into SYMFLUENCE-standard formats."""
        pass
