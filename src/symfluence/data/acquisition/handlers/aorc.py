import xarray as xr
import pandas as pd
import s3fs
from pathlib import Path
from typing import Dict, Any
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('AORC')
class AORCAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        self.logger.info("Downloading AORC data from S3")
        fs = s3fs.S3FileSystem(anon=True)
        years = range(self.start_date.year, self.end_date.year + 1)
        datasets = []
        for year in years:
            try:
                store = s3fs.S3Map(f'noaa-nws-aorc-v1-1-1km/{year}.zarr', s3=fs)
                ds = xr.open_zarr(store)
                lon1, lon2 = sorted([self.bbox['lon_min'], self.bbox['lon_max']])
                # Convert to 0-360 if dataset uses that convention
                if float(ds['longitude'].max()) > 180.0:
                    lon_min, lon_max = (lon1 + 360.0) % 360.0, (lon2 + 360.0) % 360.0
                else:
                    lon_min, lon_max = lon1, lon2
                ds_subset = ds.sel(latitude=slice(self.bbox['lat_min'], self.bbox['lat_max']), longitude=slice(lon_min, lon_max))
                ds_subset = ds_subset.sel(time=slice(max(self.start_date, pd.Timestamp(f'{year}-01-01')), min(self.end_date, pd.Timestamp(f'{year}-12-31 23:59:59'))))
                if len(ds_subset.time) > 0: datasets.append(ds_subset)
            except Exception as e:
                self.logger.error(f"Error processing year {year}: {e}")
                raise
        if not datasets: raise ValueError("No data extracted for the specified time period")
        ds_combined = xr.concat(datasets, dim='time')
        ds_combined.attrs.update({'source': 'NOAA AORC v1.1', 'bbox': str(self.bbox)})
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_AORC_{self.start_date.year}-{self.end_date.year}.nc"
        ds_combined.to_netcdf(output_file)
        return output_file
