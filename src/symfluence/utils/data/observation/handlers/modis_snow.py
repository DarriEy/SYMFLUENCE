import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('MODIS_SNOW')
class MODISSnowHandler(BaseObservationHandler):
    """
    Handles MODIS Snow Cover Area (SCA) data.
    """

    def acquire(self) -> Path:
        """Locate MODIS snow data."""
        snow_dir = Path(self.config.get('MODIS_SNOW_DIR', self.project_dir / "observations" / "snow"))
        if not snow_dir.exists():
            snow_dir.mkdir(parents=True, exist_ok=True)
        return snow_dir

    def process(self, input_path: Path) -> Path:
        """Process MODIS SCA CSV files."""
        self.logger.info(f"Processing MODIS Snow for domain: {self.domain_name}")
        
        # Look for CSV files
        csv_files = list(input_path.glob("SCA_Daily_Basin_*.csv"))
        if not csv_files:
            self.logger.warning("No MODIS SCA CSV files found")
            return input_path
            
        # Standard filter: at least 100 valid pixels
        min_pixels = self.config.get('MODIS_MIN_PIXELS', 100)
        
        # For now, just copy and filter the first matching file
        df = pd.read_csv(csv_files[0], parse_dates=['date']).set_index('date')
        if 'valid_pixels' in df.columns:
            df = df[df['valid_pixels'] >= min_pixels]
            
        output_dir = self.project_dir / "observations" / "snow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_modis_snow_processed.csv"
        df.to_csv(output_file)
        
        self.logger.info(f"MODIS snow processing complete: {output_file}")
        return output_file
