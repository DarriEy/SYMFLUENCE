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
        """Locate or download MODIS snow data."""
        data_access = self.config.get('DATA_ACCESS', 'local').lower()
        snow_dir = Path(self.config.get('MODIS_SNOW_DIR', self.project_dir / "observations" / "snow"))
        
        if not snow_dir.exists():
            snow_dir.mkdir(parents=True, exist_ok=True)

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for MODIS snow")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('MODIS_SNOW', self.config, self.logger)
            return acquirer.download(snow_dir)
            
        return snow_dir

    def process(self, input_path: Path) -> Path:
        """Process MODIS SCA data (CSV or NetCDF)."""
        self.logger.info(f"Processing MODIS Snow for domain: {self.domain_name}")
        
        # Determine if we are processing a file or a directory
        if input_path.is_file() and input_path.suffix == '.nc':
            return self._process_netcdf(input_path)
            
        # Legacy/Directory-based CSV processing
        csv_files = list(input_path.glob("SCA_Daily_Basin_*.csv"))
        if not csv_files:
            self.logger.warning(f"No MODIS SCA files found in {input_path}")
            return input_path
            
        # Standard filter: at least 100 valid pixels
        min_pixels = self.config.get('MODIS_MIN_PIXELS', 100)
        
        # For now, just copy and filter the first matching file
        df = pd.read_csv(csv_files[0], parse_dates=['date']).set_index('date')
        if 'valid_pixels' in df.columns:
            df = df[df['valid_pixels'] >= min_pixels]
            
        return self._save_processed(df)

    def _process_netcdf(self, nc_path: Path) -> Path:
        """Extract basin average snow cover from NetCDF."""
        self.logger.info(f"Extracting basin average from MODIS NetCDF: {nc_path}")
        
        try:
            # Try netcdf4 engine explicitly first
            ds = xr.open_dataset(nc_path, engine='netcdf4')
        except Exception as e:
            self.logger.warning(f"Failed to open with netcdf4, trying h5netcdf: {e}")
            try:
                ds = xr.open_dataset(nc_path, engine='h5netcdf')
            except Exception as e2:
                self.logger.error(f"Failed to open NetCDF with any engine: {e2}")
                # Check if it's an error page
                if nc_path.stat().st_size < 10000:
                    with open(nc_path, 'r', errors='ignore') as f:
                        self.logger.error(f"File snippet: {f.read(500)}")
                raise
        
        with ds:
            # Assume NDSI_Snow_Cover variable
            var_name = "NDSI_Snow_Cover"
            if var_name not in ds.data_vars:
                # Find first suitable variable
                suitable_vars = [v for v in ds.data_vars if 'snow' in v.lower()]
                if not suitable_vars:
                    raise ValueError(f"No snow variables found in {nc_path}. Vars: {list(ds.data_vars)}")
                var_name = suitable_vars[0]
            
            # Spatial averaging
            spatial_dims = [d for d in ds[var_name].dims if d != 'time']
            if spatial_dims:
                sca = ds[var_name].mean(dim=spatial_dims)
            else:
                sca = ds[var_name]
            
            df = pd.DataFrame({
                'sca': sca.values
            }, index=pd.to_datetime(ds.time.values))
            
            return self._save_processed(df)

    def _save_processed(self, df: pd.DataFrame) -> Path:
        """Save processed dataframe to standard location."""
        output_dir = self.project_dir / "observations" / "snow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_modis_snow_processed.csv"
        
        df.to_csv(output_file)
        self.logger.info(f"MODIS snow processing complete: {output_file}")
        return output_file
