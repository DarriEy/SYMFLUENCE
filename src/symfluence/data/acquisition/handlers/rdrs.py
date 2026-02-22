"""
Acquisition handler for RDRS/CaSR (Canadian Surface Reanalysis) datasets.

Provides cloud-based acquisition for CaSR v3.2 (successor to RDRS v3.1) and
RDRS v2.1 via PAVICS THREDDS OPeNDAP for efficient remote subsetting, with
HTTP fallback to ECCC's GPSC-C server for daily NetCDF files.

Data sources:
  - Primary: PAVICS THREDDS OPeNDAP (CaSR v3.2, 1980-2024, hourly)
  - Fallback: ECCC GPSC-C daily NetCDF files (CaSR v3.2)
  - Legacy:   PAVICS THREDDS OPeNDAP (RDRS v2.1, 1980-2018, hourly)
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
from ..utils import create_robust_session

# PAVICS THREDDS OPeNDAP endpoints
PAVICS_OPENDAP_CASR_V32 = (
    "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/dodsC/"
    "datasets/reanalyses/1hr_NAM_GovCan_CaSR_v32_198001-202412.ncml"
)
PAVICS_OPENDAP_RDRS_V21 = (
    "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/dodsC/"
    "datasets/reanalyses/1hr_RDRSv2.1_NAM.ncml"
)

# ECCC GPSC-C HTTP endpoint for CaSR v3.2 daily NetCDF files
GPSCC_CASR_V32_BASE_URL = (
    "https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/data/CaSRv3.2/netcdf"
)


@AcquisitionRegistry.register('RDRS')
@AcquisitionRegistry.register('RDRS_v3.1')
class RDRSAcquirer(BaseAcquisitionHandler):
    """
    Acquisition handler for CaSR/RDRS reanalysis data.

    RDRS has been renamed to CaSR (Canadian Surface Reanalysis) by ECCC.
    This handler acquires CaSR v3.2 by default (successor to v3.1), falling
    back to RDRS v2.1 if configured. The primary pathway uses PAVICS THREDDS
    OPeNDAP for efficient server-side spatial and temporal subsetting.
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process RDRS/CaSR data for the configured time period."""
        output_dir.mkdir(parents=True, exist_ok=True)
        final_file = output_dir / f"domain_{self.domain_name}_RDRS_{self.start_date.year}_{self.end_date.year}.nc"

        if final_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            return final_file

        # Try OPeNDAP pathway first (efficient remote subsetting)
        try:
            return self._download_opendap(final_file)
        except Exception as e:
            self.logger.warning(f"OPeNDAP pathway failed: {e}. Falling back to HTTP.")
            return self._download_http(output_dir, final_file)

    def _download_opendap(self, final_file: Path) -> Path:
        """Download CaSR/RDRS data via PAVICS THREDDS OPeNDAP."""
        version = self.config_dict.get('RDRS_VERSION', 'v3.2')

        if version == 'v2.1':
            opendap_url = PAVICS_OPENDAP_RDRS_V21
            self.logger.info("Accessing RDRS v2.1 via PAVICS THREDDS OPeNDAP")
        else:
            opendap_url = PAVICS_OPENDAP_CASR_V32
            self.logger.info("Accessing CaSR v3.2 (successor to RDRS v3.1) via PAVICS THREDDS OPeNDAP")

        # Allow user override of OPeNDAP URL
        opendap_url = self.config_dict.get('RDRS_OPENDAP_URL', opendap_url)

        self.logger.info(f"Opening OPeNDAP dataset: {opendap_url}")
        ds = xr.open_dataset(opendap_url)

        # Spatial subsetting using lat/lon coordinates on rotated pole grid
        if self.bbox:
            self.logger.info("Computing spatial mask from OPeNDAP coordinates...")
            lat_vals = ds.lat.values
            lon_vals = ds.lon.values

            mask = (
                (lat_vals >= self.bbox['lat_min']) & (lat_vals <= self.bbox['lat_max']) &
                (lon_vals >= self.bbox['lon_min']) & (lon_vals <= self.bbox['lon_max'])
            )

            y_indices, x_indices = np.where(mask)

            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()
            else:
                # Bounding box smaller than grid resolution (e.g. point-scale domain).
                # Fall back to nearest grid cell.
                center_lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
                center_lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2
                dist = np.sqrt((lat_vals - center_lat)**2 + (lon_vals - center_lon)**2)
                y_near, x_near = np.unravel_index(dist.argmin(), dist.shape)
                self.logger.info(
                    f"Bounding box smaller than grid resolution, using nearest grid cell "
                    f"at ({lat_vals[y_near, x_near]:.4f}, {lon_vals[y_near, x_near]:.4f})"
                )
                y_min = y_max = y_near
                x_min = x_max = x_near

            # Add buffer for safety
            y_min = max(0, y_min - 2)
            y_max = min(ds.sizes['rlat'] - 1, y_max + 2)
            x_min = max(0, x_min - 2)
            x_max = min(ds.sizes['rlon'] - 1, x_max + 2)

            ds = ds.isel(rlat=slice(y_min, y_max + 1), rlon=slice(x_min, x_max + 1))
            self.logger.info(f"Spatially subsetted to {ds.sizes['rlat']}x{ds.sizes['rlon']} grid")

        # Temporal subsetting
        ds = ds.sel(time=slice(self.start_date, self.end_date))

        if ds.time.size == 0:
            raise ValueError(f"No data found for time range {self.start_date} to {self.end_date}")

        time_steps = ds.time.size
        self.logger.info(f"Downloading {time_steps} time steps via OPeNDAP...")

        # Load data from remote server and save locally
        ds_loaded = ds.load()

        # Use chunked encoding for better memory management
        encoding = {}
        for var in ds_loaded.data_vars:
            shape = ds_loaded[var].shape
            if len(shape) == 3:  # time, rlat, rlon
                encoding[var] = {'chunksizes': (min(168, shape[0]), shape[1], shape[2])}

        ds_loaded.to_netcdf(final_file, encoding=encoding)
        self.logger.info(f"Successfully saved data to {final_file.name}")
        return final_file

    def _download_http(self, output_dir: Path, final_file: Path) -> Path:
        """Fallback HTTP download from ECCC GPSC-C daily NetCDF files."""
        # CaSR v3.2 daily files from GPSC-C (format: YYYYMMDD12.nc)
        base_url = self.config_dict.get('RDRS_BASE_URL', GPSCC_CASR_V32_BASE_URL)
        self.logger.info(f"Downloading CaSR daily files from {base_url}")

        # Generate list of days (files are daily at 12 UTC)
        date_range = pd.date_range(start=self.start_date.normalize(), end=self.end_date.normalize(), freq='D')
        total_files = len(date_range)
        self.logger.info(f"Downloading {total_files} daily CaSR files via HTTP fallback")

        # Use robust session with connection pooling and retry logic
        session = create_robust_session(max_retries=3, backoff_factor=1.0)

        max_workers = min(total_files, 4)
        downloaded_files = []
        failed_dates = []
        completed = 0
        log_interval = max(1, total_files // 20)  # Log every ~5%

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(self._download_daily_file, dt, base_url, output_dir, session): dt
                for dt in date_range
            }
            for future in concurrent.futures.as_completed(future_to_date):
                dt = future_to_date[future]
                completed += 1
                try:
                    f = future.result()
                    if f:
                        downloaded_files.append(f)
                    else:
                        failed_dates.append(dt)
                except Exception as e:
                    self.logger.warning(f"HTTP download failed for {dt.strftime('%Y-%m-%d')}: {e}")
                    failed_dates.append(dt)

                if completed % log_interval == 0 or completed == total_files:
                    self.logger.info(
                        f"HTTP download progress: {completed}/{total_files} "
                        f"({len(downloaded_files)} ok, {len(failed_dates)} failed)"
                    )

        # Retry failed downloads sequentially
        if failed_dates:
            self.logger.info(f"Retrying {len(failed_dates)} failed downloads...")
            retry_session = create_robust_session(max_retries=5, backoff_factor=2.0)
            for dt in failed_dates:
                f = self._download_daily_file(dt, base_url, output_dir, retry_session, timeout=300)
                if f:
                    downloaded_files.append(f)
                else:
                    self.logger.warning(f"Retry also failed for {dt.strftime('%Y-%m-%d')}")

        final_failed = total_files - len(downloaded_files)
        if final_failed > 0:
            self.logger.warning(f"{final_failed}/{total_files} daily files failed to download after retries")

        if not downloaded_files:
            raise RuntimeError(
                "HTTP fallback failed: No CaSR data downloaded. "
                "The GPSC-C server may not yet have data for your time period. "
                "Check https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/download.html "
                "for current data availability."
            )

        downloaded_files.sort()
        self.logger.info(f"Merging {len(downloaded_files)} daily CaSR files...")
        with xr.open_mfdataset(downloaded_files, combine='by_coords', chunks={'time': 24}, data_vars='minimal', coords='minimal', compat='override') as ds:
            if self.bbox:
                self.logger.info("Computing spatial mask for HTTP-downloaded files...")
                lat_vals = ds.lat.compute()
                lon_vals = ds.lon.compute()

                mask = (lat_vals >= self.bbox['lat_min']) & (lat_vals <= self.bbox['lat_max']) & \
                       (lon_vals >= self.bbox['lon_min']) & (lon_vals <= self.bbox['lon_max'])
                y_idx, x_idx = np.where(mask.values)

                if len(y_idx) > 0:
                    y_min, y_max = y_idx.min(), y_idx.max()
                    x_min, x_max = x_idx.min(), x_idx.max()
                else:
                    # Nearest grid cell fallback for point-scale domains
                    center_lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
                    center_lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2
                    dist = np.sqrt((lat_vals.values - center_lat)**2 + (lon_vals.values - center_lon)**2)
                    y_near, x_near = np.unravel_index(dist.argmin(), dist.shape)
                    self.logger.info(
                        f"Bounding box smaller than grid resolution, using nearest grid cell "
                        f"at ({lat_vals.values[y_near, x_near]:.4f}, {lon_vals.values[y_near, x_near]:.4f})"
                    )
                    y_min = y_max = y_near
                    x_min = x_max = x_near

                self.logger.info("Subsetting spatial domain")
                ds = ds.isel(rlat=slice(y_min, y_max+1), rlon=slice(x_min, x_max+1))

            self.logger.info(f"Saving merged data to {final_file.name}...")
            ds.to_netcdf(final_file)

        self.logger.info("Cleaning up temporary daily files...")
        for f in downloaded_files:
            f.unlink(missing_ok=True)
        return final_file

    def _download_daily_file(
        self, dt: datetime, base_url: str, output_dir: Path,
        session: requests.Session, timeout: int = 120
    ) -> Optional[Path]:
        """Download a single daily CaSR file from GPSC-C (format: YYYYMMDD12.nc)."""
        file_name = dt.strftime("%Y%m%d12.nc")
        url = f"{base_url.rstrip('/')}/{file_name}"
        dest_path = output_dir / f"temp_casr_{file_name}"
        if dest_path.exists():
            return dest_path
        try:
            response = session.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        f.write(chunk)
                return dest_path
            else:
                self.logger.debug(f"CaSR download returned status {response.status_code} for {file_name}")
        except requests.exceptions.Timeout:
            self.logger.warning(f"CaSR download timed out for {file_name}")
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"CaSR download failed for {file_name}: {e}")
        return None
