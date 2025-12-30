import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('MODIS_ET')
class MODISETHandler(BaseObservationHandler):
    """
    Handles MODIS 8-day Evapotranspiration (ET) data.
    """

    def acquire(self) -> Path:
        """Locate MODIS ET data."""
        et_dir = Path(self.config.get('MODIS_ET_DIR', self.project_dir / "observations" / "et"))
        if not et_dir.exists():
            et_dir.mkdir(parents=True, exist_ok=True)
        return et_dir

    def process(self, input_path: Path) -> Path:
        """Process MODIS ET CSV files (8-day to daily conversion)."""
        self.logger.info(f"Processing MODIS ET for domain: {self.domain_name}")
        
        # Look for CSV files
        csv_files = list(input_path.glob("ET8D_Basin_*.csv"))
        if not csv_files:
            self.logger.warning("No MODIS ET CSV files found")
            return input_path
            
        df = pd.read_csv(csv_files[0], parse_dates=['date']).set_index('date')
        if 'mean_et_mm' in df.columns:
            # Convert 8-day cumulative/mean to daily mean
            df['et_daily_mm'] = df['mean_et_mm'] / 8.0
            
        output_dir = self.project_dir / "observations" / "et" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_modis_et_processed.csv"
        df.to_csv(output_file)
        
        self.logger.info(f"MODIS ET processing complete: {output_file}")
        return output_file
