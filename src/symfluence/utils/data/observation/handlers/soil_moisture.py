import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('SMAP')
class SMAPHandler(BaseObservationHandler):
    """
    Handles SMAP Soil Moisture data.
    """

    def acquire(self) -> Path:
        """Locate SMAP data."""
        smap_dir = Path(self.config.get('SMAP_PATH', self.project_dir / "observations" / "soil_moisture" / "smap"))
        if not smap_dir.exists():
            smap_dir.mkdir(parents=True, exist_ok=True)
        return smap_dir

    def process(self, input_path: Path) -> Path:
        """Process SMAP NetCDF data."""
        self.logger.info(f"Processing SMAP Soil Moisture for domain: {self.domain_name}")
        
        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No SMAP NetCDF files found")
            return input_path
            
        # Strategy: spatial average over bounding box if multiple pixels
        # For simplicity in this implementation, we take the mean of the first file
        results = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                # SMAP variables often named 'soil_moisture' or 'sm_surface'
                var_name = [v for v in ds.data_vars if 'soil_moisture' in v.lower() or 'sm_surface' in v.lower()]
                if not var_name:
                    continue
                var_name = var_name[0]
                
                # Spatial average
                mean_sm = ds[var_name].mean(dim=[d for d in ds[var_name].dims if d != 'time'])
                df_ts = mean_sm.to_dataframe().reset_index()
                results.append(df_ts)
        
        if not results:
            self.logger.warning("No SMAP data could be extracted")
            return input_path
            
        df = pd.concat(results).sort_values('time').set_index('time')
        
        output_dir = self.project_dir / "observations" / "soil_moisture" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_smap_processed.csv"
        df.to_csv(output_file)
        
        self.logger.info(f"SMAP processing complete: {output_file}")
        return output_file

@ObservationRegistry.register('ESA_CCI_SM')
class ESACCISMHandler(BaseObservationHandler):
    """
    Handles ESA CCI Soil Moisture data.
    """

    def acquire(self) -> Path:
        """Locate ESA CCI SM data."""
        esa_dir = Path(self.config.get('ESA_CCI_SM_PATH', self.project_dir / "observations" / "soil_moisture" / "esa_cci"))
        if not esa_dir.exists():
            esa_dir.mkdir(parents=True, exist_ok=True)
        return esa_dir

    def process(self, input_path: Path) -> Path:
        """Process ESA CCI SM NetCDF data."""
        self.logger.info(f"Processing ESA CCI Soil Moisture for domain: {self.domain_name}")
        
        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No ESA CCI SM NetCDF files found")
            return input_path
            
        results = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                # ESA CCI SM variable is usually 'sm'
                if 'sm' not in ds.data_vars:
                    continue
                
                # Spatial average
                mean_sm = ds['sm'].mean(dim=[d for d in ds['sm'].dims if d != 'time'])
                df_ts = mean_sm.to_dataframe().reset_index()
                results.append(df_ts)
        
        if not results:
            self.logger.warning("No ESA CCI SM data could be extracted")
            return input_path
            
        df = pd.concat(results).sort_values('time').set_index('time')
        
        output_dir = self.project_dir / "observations" / "soil_moisture" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_esa_cci_sm_processed.csv"
        df.to_csv(output_file)
        
        self.logger.info(f"ESA CCI SM processing complete: {output_file}")
        return output_file
