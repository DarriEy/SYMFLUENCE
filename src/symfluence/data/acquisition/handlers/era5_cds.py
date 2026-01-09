"""
ERA5 CDS Data Acquisition Handler for SYMFLUENCE.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from symfluence.core.constants import UnitConversion

@AcquisitionRegistry.register('ERA5_CDS')
class ERA5CDSAcquirer(BaseAcquisitionHandler):
    """
    ERA5 data acquisition handler using the Copernicus Climate Data Store (CDS) API.
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process ERA5 data from CDS in monthly chunks."""
        if not HAS_CDSAPI:
            raise ImportError(
                "cdsapi package is required for ERA5 CDS downloads. "
                "Install it with 'pip install cdsapi'."
            )

        self.logger.info(f"Downloading ERA5 data from CDS for {self.domain_name}...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine year-month combinations to process (same pattern as CARRA/CERRA)
        dates = pd.date_range(self.start_date, self.end_date, freq='MS')
        if dates.empty:
            # Handle case where range is within a single month
            ym_range = [(self.start_date.year, self.start_date.month)]
        else:
            ym_range = [(d.year, d.month) for d in dates]
            # Ensure the end date's month is included if not already
            if (self.end_date.year, self.end_date.month) not in ym_range:
                ym_range.append((self.end_date.year, self.end_date.month))

        chunk_files = []

        # Use ThreadPoolExecutor for downloads
        # CDS has strict rate limits - use 1 worker for sequential downloads to avoid conflicts
        max_workers = 1

        self.logger.info(f"Starting ERA5 download for {len(ym_range)} months (sequential)...")

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ym = {
                executor.submit(self._download_and_process_month, year, month, output_dir): (year, month)
                for year, month in ym_range
            }

            for future in concurrent.futures.as_completed(future_to_ym):
                year, month = future_to_ym[future]
                try:
                    chunk_path = future.result()
                    if chunk_path:
                        chunk_files.append(chunk_path)
                        self.logger.info(f"✓ Completed processing for {year}-{month:02d}")
                except Exception as exc:
                    self.logger.error(f"Processing for {year}-{month:02d} failed: {exc}")
                    # Cancel remaining and raise
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise exc

        # Merge all monthly chunks
        chunk_files.sort()

        if not chunk_files:
            raise RuntimeError("No ERA5 data downloaded")

        self.logger.info(f"Merging {len(chunk_files)} monthly chunks...")

        # Open all chunks and merge
        with xr.open_mfdataset(chunk_files, combine='by_coords') as ds_merged:
            # Trim to exact requested time range
            ds_merged = ds_merged.sel(time=slice(self.start_date, self.end_date))

            # Save final dataset
            final_f = output_dir / f"domain_{self.domain_name}_ERA5_CDS_{self.start_date.year}_{self.end_date.year}.nc"
            ds_merged.to_netcdf(final_f)
            self.logger.info(f"ERA5 CDS data saved to {final_f}")

        # Cleanup monthly chunks
        for chunk_file in chunk_files:
            if chunk_file.exists():
                try:
                    chunk_file.unlink()
                except OSError:
                    pass

        return final_f

    def _download_and_process_month(self, year: int, month: int, output_dir: Path) -> Path:
        """Download and process a single month of ERA5 data (executed in thread)."""
        # Create a thread-local CDS client
        try:
            c = cdsapi.Client()
        except Exception as e:
            self.logger.error(f"Failed to initialize CDS client: {e}")
            self.logger.error("Ensure ~/.cdsapirc exists or CDSAPI_URL/CDSAPI_KEY env vars are set.")
            raise

        self.logger.info(f"Downloading ERA5 for {year}-{month:02d}...")

        # Build temporal parameters for this month
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)

        # Restrict to actual requested date range
        month_start = max(month_start, self.start_date)
        month_end = min(month_end, self.end_date)

        dates = pd.date_range(month_start, month_end, freq='h')
        days = sorted(list(set([f"{d.day:02d}" for d in dates])))
        times = sorted(list(set([f"{d.hour:02d}:00" for d in dates])))

        # Bounding box for CDS (North, West, South, East)
        area = [
            self.bbox['lat_max'],
            self.bbox['lon_min'],
            self.bbox['lat_min'],
            self.bbox['lon_max']
        ]

        # Temp files for this month (analysis + forecast, like CARRA/CERRA)
        analysis_file = output_dir / f"{self.domain_name}_era5_analysis_{year}{month:02d}_temp.nc"
        forecast_file = output_dir / f"{self.domain_name}_era5_forecast_{year}{month:02d}_temp.nc"

        try:
            # Request 1: Analysis variables (instantaneous)
            analysis_vars = [
                '2m_temperature',
                'surface_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                '2m_dewpoint_temperature'
            ]

            analysis_request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': analysis_vars,
                'year': [str(year)],
                'month': [f"{month:02d}"],
                'day': days,
                'time': times,
                'area': area,
            }

            # Request 2: Accumulated variables (precipitation, radiation)
            # ERA5 has these in 'reanalysis' product, splitting to reduce request size
            forecast_vars = [
                'total_precipitation',
                'surface_solar_radiation_downwards',
                'surface_thermal_radiation_downwards'
            ]

            forecast_request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': forecast_vars,
                'year': [str(year)],
                'month': [f"{month:02d}"],
                'day': days,
                'time': times,
                'area': area,
            }

            # Download both products
            self.logger.info(f"Downloading ERA5 analysis data for {year}-{month:02d}...")
            self._retrieve_with_retry(c, 'reanalysis-era5-single-levels', analysis_request, str(analysis_file))

            self.logger.info(f"Downloading ERA5 forecast data for {year}-{month:02d}...")
            self._retrieve_with_retry(c, 'reanalysis-era5-single-levels', forecast_request, str(forecast_file))

            # Process and merge datasets (like CARRA/CERRA)
            ds_chunk = self._process_and_merge_datasets(analysis_file, forecast_file)

            # Save chunk to disk
            chunk_file = output_dir / f"{self.domain_name}_era5_cds_processed_{year}{month:02d}_temp.nc"
            ds_chunk.to_netcdf(chunk_file)

            self.logger.info(f"✓ Processed ERA5 chunk for {year}-{month:02d}")
            return chunk_file

        finally:
            # Cleanup raw downloads for this month
            if analysis_file.exists():
                analysis_file.unlink()
            if forecast_file.exists():
                forecast_file.unlink()

    def _process_and_merge_datasets(self, analysis_file: Path, forecast_file: Path) -> xr.Dataset:
        """Process and merge ERA5 analysis and forecast files (similar to CARRA/CERRA)."""
        with xr.open_dataset(analysis_file, engine='netcdf4') as dsa, \
             xr.open_dataset(forecast_file, engine='netcdf4') as dsf:

            # Handle dimension standardization
            if 'valid_time' in dsa.dims:
                dsa = dsa.rename({'valid_time': 'time'})
            if 'valid_time' in dsf.dims:
                dsf = dsf.rename({'valid_time': 'time'})

            # Handle expver dimension if present
            for ds_name, ds in [('analysis', dsa), ('forecast', dsf)]:
                if 'expver' in ds.dims:
                    if ds.sizes['expver'] > 1:
                        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
                    else:
                        ds = ds.isel(expver=0)
                    if ds_name == 'analysis':
                        dsa = ds
                    else:
                        dsf = ds

            # Handle ensemble members if present (forecast file)
            if 'number' in dsf.dims:
                self.logger.info("Ensemble data detected. Selecting first member.")
                dsf = dsf.isel(number=0)

            # Sort by time
            dsa = dsa.sortby('time')
            dsf = dsf.sortby('time')

            self.logger.info(f"Analysis variables: {list(dsa.data_vars)}")
            self.logger.info(f"Forecast variables: {list(dsf.data_vars)}")

            # Merge analysis and forecast (inner join on time)
            dsm = xr.merge([dsa, dsf], join='inner')
            self.logger.info(f"Merged variables: {list(dsm.data_vars)}")

            # Now process variables (rename, convert units, derive, etc.)
            dsm = self._process_era5_variables(dsm)

            return dsm.load()

    def _process_era5_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Process ERA5 variables: rename, convert units, derive variables."""
        self.logger.info(f"Processing ERA5 variables...")

        processed_vars = {}

        # Find precipitation
        v_tp = next((v for v in ds.variables if any(x in v.lower() for x in ['total_precip', 'tp'])), None)
        if v_tp:
            self.logger.info(f"Processing precipitation from {v_tp}")
            # ERA5 precip is in meters, convert to mm/s
            pptrate = (ds[v_tp] * 1000.0 / UnitConversion.SECONDS_PER_HOUR).astype('float32')
            pptrate.attrs = {'units': 'kg m-2 s-1', 'long_name': 'precipitation rate'}
            processed_vars['pptrate'] = pptrate

        # Shortwave radiation - ERA5 accumulated, needs time-differencing
        v_ssrd = next((v for v in ds.variables if any(x in v.lower() for x in ['solar_radiation_down', 'ssrd'])), None)
        if v_ssrd:
            self.logger.info(f"Processing shortwave from {v_ssrd}")
            val = ds[v_ssrd]
            dt = (ds['time'].diff('time') / np.timedelta64(1, 's')).astype('float32')
            sw_diff = val.diff('time').where(val.diff('time') >= 0, 0)
            sw_rad = (sw_diff / dt).clip(min=0).astype('float32')
            sw_rad = xr.concat([sw_rad.isel(time=0), sw_rad], dim='time')
            sw_rad.attrs = {'units': 'W m-2', 'long_name': 'shortwave radiation'}
            processed_vars['SWRadAtm'] = sw_rad

        # Longwave radiation - ERA5 accumulated, needs time-differencing
        v_strd = next((v for v in ds.variables if any(x in v.lower() for x in ['thermal_radiation_down', 'strd'])), None)
        if v_strd:
            self.logger.info(f"Processing longwave from {v_strd}")
            val = ds[v_strd]
            self.logger.debug(f"Raw LW range: min={val.min().values:.1f}, max={val.max().values:.1f}")
            # ERA5 downward flux is typically positive; only flip if data are negative.
            val_positive = -val if float(val.min()) < 0.0 else val
            dt = (ds['time'].diff('time') / np.timedelta64(1, 's')).astype('float32')
            lw_diff = val_positive.diff('time')
            lw_diff = lw_diff.where(lw_diff >= 0, val_positive.isel(time=slice(1, None)))
            lw_rad = (lw_diff / dt).clip(min=50, max=600).astype('float32')
            lw_rad = xr.concat([lw_rad.isel(time=0), lw_rad], dim='time')
            if lw_rad.mean() < 200:
                raise ValueError(f"LW radiation too low: {lw_rad.mean().values:.1f} W/m²")
            self.logger.info(f"✓ LW radiation: mean={lw_rad.mean().values:.1f} W/m²")
            lw_rad.attrs = {'units': 'W m-2', 'long_name': 'longwave radiation'}
            processed_vars['LWRadAtm'] = lw_rad

        # Temperatures and Pressure
        rename_map = {
            't2m': 'airtemp', '2m_temperature': 'airtemp',
            'sp': 'airpres', 'surface_pressure': 'airpres',
            'd2m': 'dewpoint', '2m_dewpoint_temperature': 'dewpoint',
            'u10': 'wind_u', '10m_u_component_of_wind': 'wind_u',
            'v10': 'wind_v', '10m_v_component_of_wind': 'wind_v'
        }
        for src, target in rename_map.items():
            if src in ds.variables:
                processed_vars[target] = ds[src].astype('float32')

        # Derived variables
        if 'wind_u' in processed_vars and 'wind_v' in processed_vars:
            processed_vars['windspd'] = np.sqrt(processed_vars['wind_u']**2 + processed_vars['wind_v']**2).astype('float32')
            processed_vars['windspd'].attrs = {'units': 'm s-1', 'long_name': 'wind speed'}

        if 'dewpoint' in processed_vars and 'airpres' in processed_vars:
            Td_C = processed_vars['dewpoint'] - 273.15
            P = processed_vars['airpres']
            e = 611.2 * np.exp((17.67 * Td_C) / (Td_C + 243.5))
            processed_vars['spechum'] = (0.622 * e / (P - 0.378 * e)).astype('float32')
            processed_vars['spechum'].attrs = {'units': 'kg kg-1', 'long_name': 'specific humidity'}

        # Final assembly
        final_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']
        ds_final = xr.Dataset(
            data_vars={v: processed_vars[v] for v in final_vars if v in processed_vars},
            coords={c: ds.coords[c] for c in ['time', 'latitude', 'longitude'] if c in ds.coords},
            attrs=ds.attrs
        )
        ds_final = ds_final.transpose('time', 'latitude', 'longitude', missing_dims='ignore')
        return ds_final

    def _retrieve_with_retry(
        self, client, dataset_name: str, request: Dict[str, Any], target_path: str,
        max_retries: int = 3, base_delay: int = 60
    ):
        """
        Retrieve data from CDS with retry logic for transient errors.

        Args:
            client: CDS API client
            dataset_name: Name of the dataset to retrieve
            request: Request parameters dictionary
            target_path: Path to save the retrieved data
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds between retries (default: 60)

        Raises:
            Exception: If all retries fail
        """
        import time

        for attempt in range(max_retries + 1):
            try:
                client.retrieve(dataset_name, request, target_path)
                return  # Success
            except Exception as e:
                error_msg = str(e)

                # Check if it's a 403 error
                is_403 = "403" in error_msg or "Forbidden" in error_msg

                # Check if it's a "request too large" error (should not retry)
                is_too_large = "too large" in error_msg.lower() or "cost limits exceeded" in error_msg.lower()

                # Check if it's worth retrying
                should_retry = (is_403 or "temporarily" in error_msg.lower() or "maintenance" in error_msg.lower()) and not is_too_large

                if attempt < max_retries and should_retry:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"CDS request failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}"
                    )
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    # Last attempt or non-retryable error
                    if is_403:
                        self.logger.error(
                            f"CDS API returned 403 Forbidden error. This may indicate:\n"
                            f"  1. Temporary service maintenance (retry later)\n"
                            f"  2. Dataset license not accepted (visit https://cds.climate.copernicus.eu/datasets/{dataset_name})\n"
                            f"  3. API credentials issue (check ~/.cdsapirc)\n"
                            f"  4. Rate limiting (too many requests)"
                        )
                    raise
