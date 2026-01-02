"""
Base Acquisition Handler for SYMFLUENCE
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import pandas as pd

class BaseAcquisitionHandler(ABC):
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME', 'domain')
        self.bbox = self._parse_bbox(config.get('BOUNDING_BOX_COORDS'))
        self.start_date = pd.to_datetime(config.get('EXPERIMENT_TIME_START'))
        self.end_date = pd.to_datetime(config.get('EXPERIMENT_TIME_END'))

    def _parse_bbox(self, bbox_string: str) -> Dict[str, float]:
        if not bbox_string:
            return {}
        coords = bbox_string.split('/')
        lat1 = float(coords[0])
        lon1 = float(coords[1])
        lat2 = float(coords[2])
        lon2 = float(coords[3])
        return {
            'lat_min': min(lat1, lat2),
            'lat_max': max(lat1, lat2),
            'lon_min': min(lon1, lon2),
            'lon_max': max(lon1, lon2)
        }

    @property
    def domain_dir(self) -> Path:
        base = Path(self.config.get("SYMFLUENCE_DATA_DIR"))
        d = base / f"domain_{self.domain_name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _attribute_dir(self, subdir: str) -> Path:
        d = self.domain_dir / "attributes" / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    @abstractmethod
    def download(self, output_dir: Path) -> Path:
        pass
