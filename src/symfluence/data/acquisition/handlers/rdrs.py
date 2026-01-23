"""
Acquisition handler for RDRS (Regional Deterministic Reanalysis System) datasets.

Provides cloud-based acquisition for RDRS v3.1 and v2.1 from ECCC's HPFX server.
Supports parallel downloads and automatic merging.
"""

import xarray as xr
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Optional
import concurrent.futures
from datetime import datetime

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


@AcquisitionRegistry.register('RDRS')
@AcquisitionRegistry.register('RDRS_v3.1')
class RDRSAcquirer(BaseAcquisitionHandler):
    """
    Acquisition handler for RDRS v3.1 data.

    Prefers the MSC Open Data S3 Zarr pathway for efficient spatial subsetting,
    falling back to hourly NetCDF downloads from HPFX if necessary.
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process RDRS data for the configured time period."""
        # Setup output files
        output_dir.mkdir(parents=True, exist_ok=True)
        final_file = output_dir / f"domain_{self.domain_name}_RDRS_{self.start_date.year}_{self.end_date.year}.nc"

        if final_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            return final_file

        # Try S3 Zarr pathway first (much faster)
        from importlib.util import find_spec
        if find_spec("s3fs"):
            try:
                return self._download_s3(final_file)
            except Exception as e:
                self.logger.warning(f"S3 Zarr pathway failed: {e}. Falling back to HTTP.")
                return self._download_http(output_dir, final_file)
        else:
            self.logger.warning("s3fs not installed. Falling back to HTTP.")
            return self._download_http(output_dir, final_file)

    def _download_s3(self, final_file: Path) -> Path:
        """Download RDRS 3.1 using S3 Zarr pathway."""
        import s3fs
        self.logger.info("Accessing RDRS v3.1 via MSC Open Data S3 (Zarr)")

        fs = s3fs.S3FileSystem(anon=True)
        # MSC Open Data RDRS 3.1 Zarr path
        bucket = "msc-open-data"
        zarr_path = f"{bucket}/reanalysis/rdrs/v3.1/zarr"

        # Open the store
        store = s3fs.S3Map(zarr_path, s3=fs)
        ds = xr.open_zarr(store, consolidated=True)

        # Spatial subsetting
        if self.bbox:
            # Handle RDRS rotated pole grid
            mask = (
                (ds.lat >= self.bbox['lat_min']) & (ds.lat <= self.bbox['lat_max']) &
                (ds.lon >= self.bbox['lon_min']) & (ds.lon <= self.bbox['lon_max'])
            )

            y_indices, x_indices = np.where(mask.values)

            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()

                # Add buffer
                y_min = max(0, y_min - 2)
                y_max = min(ds.dims['rlat'] - 1, y_max + 2)
                x_min = max(0, x_min - 2)
                x_max = min(ds.dims['rlon'] - 1, x_max + 2)

                ds = ds.isel(rlat=slice(y_min, y_max + 1), rlon=slice(x_min, x_max + 1))
                self.logger.info(f"Spatially subsetted Zarr to {ds.dims['rlat']}x{ds.dims['rlon']} grid")

        # Temporal subsetting
        ds = ds.sel(time=slice(self.start_date, self.end_date))

        if ds.time.size == 0:
            raise ValueError(f"No RDRS data found for time range {self.start_date} to {self.end_date}")

        # Load and save to NetCDF
        self.logger.info("Loading subsetted data and saving to NetCDF...")
        ds.to_netcdf(final_file)
        return final_file

    def _download_http(self, output_dir: Path, final_file: Path) -> Path:
        """Fallback HTTP download method."""
        # ... existing HTTP logic ...
        version = self.config_dict.get('RDRS_VERSION', "v3.1")
        if version == "v2.1":
            default_url = "https://hpfx.collab.science.gc.ca/~rlarocque/RDRS_v2.1/"
        else:
            default_url = "https://hpfx.collab.science.gc.ca/~rlarocque/RDRS_v3.1/"

        base_url = self.config_dict.get('RDRS_BASE_URL', default_url)

        # Generate list of hours
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='h')

        max_workers = min(len(date_range), 4)
        downloaded_files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(self._download_hour, dt, base_url, output_dir): dt
                for dt in date_range
            }
            for future in concurrent.futures.as_completed(future_to_date):
                try:
                    f = future.result()
                    if f: downloaded_files.append(f)
                except Exception as e:
                    self.logger.error(f"HTTP download failed: {e}")

        if not downloaded_files:
            raise RuntimeError("HTTP fallback failed: No RDRS data downloaded")

        downloaded_files.sort()
        with xr.open_mfdataset(downloaded_files, combine='by_coords') as ds:
            # Simple spatial subset for fallback
            if self.bbox:
                mask = (ds.lat >= self.bbox['lat_min']) & (ds.lat <= self.bbox['lat_max']) & \
                       (ds.lon >= self.bbox['lon_min']) & (ds.lon <= self.bbox['lon_max'])
                y_idx, x_idx = np.where(mask.values)
                if len(y_idx) > 0:
                    ds = ds.isel(rlat=slice(y_idx.min(), y_idx.max()+1), rlon=slice(x_idx.min(), x_idx.max()+1))
            ds.to_netcdf(final_file)

        for f in downloaded_files: f.unlink(missing_ok=True)
        return final_file

    def _download_hour(self, dt: datetime, base_url: str, output_dir: Path) -> Optional[Path]:
        """Download a single hourly file from HPFX."""
        year_str = dt.strftime("%Y")
        file_name = dt.strftime("%Y%m%d%H.nc")
        url = f"{base_url.rstrip('/')}/{year_str}/{file_name}"
        dest_path = output_dir / f"temp_rdrs_{file_name}"
        if dest_path.exists():
            return dest_path
        try:
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return dest_path
            else:
                self.logger.debug(f"RDRS download returned status {response.status_code} for {file_name}")
        except requests.exceptions.Timeout:
            self.logger.debug(f"RDRS download timed out for {file_name}")
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"RDRS download failed for {file_name}: {e}")
        return None
