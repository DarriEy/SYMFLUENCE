"""
Base Acquisition Handler for SYMFLUENCE
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from symfluence.utils.common.coordinate_utils import CoordinateUtilsMixin

class BaseAcquisitionHandler(ABC, CoordinateUtilsMixin):
    def __init__(self, config: Dict[str, Any], logger, reporting_manager: Any = None):
        self.config = config
        self.logger = logger
        self.reporting_manager = reporting_manager
        self.domain_name = config.get('DOMAIN_NAME', 'domain')
        self.bbox = self._parse_bbox(config.get('BOUNDING_BOX_COORDS'))
        self.start_date = pd.to_datetime(config.get('EXPERIMENT_TIME_START'))
        self.end_date = pd.to_datetime(config.get('EXPERIMENT_TIME_END'))

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

    def plot_diagnostics(self, file_path: Path):
        """
        Create diagnostic plots for the acquired data.
        Can be overridden by subclasses for specific plotting needs.
        """
        if not self.reporting_manager:
            return

        try:
            if file_path.suffix in ['.tif', '.nc']:
                # Raster data
                self.reporting_manager.visualize_spatial_coverage(
                    file_path, 
                    variable_name=file_path.stem, 
                    stage='acquisition'
                )
            elif file_path.suffix == '.csv':
                # Tabular data - try to read and plot distribution
                df = pd.read_csv(file_path)
                # Assume numeric columns are interesting
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    if 'id' not in col.lower() and 'date' not in col.lower():
                        self.reporting_manager.visualize_data_distribution(
                            df[col], 
                            variable_name=col, 
                            stage='acquisition'
                        )
        except Exception as e:
            self.logger.warning(f"Failed to create diagnostic plots for {file_path}: {e}")
