"""
CDS Dataset Handlers for Regional Reanalysis Products.

Provides acquisition handlers for CARRA (Arctic) and CERRA (European) datasets
from the Copernicus Climate Data Store, using a shared base class to eliminate
code duplication.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod
import time
import concurrent.futures

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from symfluence.data.utilities import VariableStandardizer


class CDSRegionalReanalysisHandler(BaseAcquisitionHandler, ABC):
    """
    Abstract base handler for CDS regional reanalysis products.

    Implements common dual-product download strategy (analysis + forecast),
    time alignment, spatial subsetting, unit conversions, and variable derivations.

    Subclasses must implement abstract methods to specify dataset-specific
    configurations such as temporal resolution, variable lists, and spatial handling.
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process regional reanalysis data in parallel."""
        if not HAS_CDSAPI:
            raise ImportError(
                f"cdsapi package is required for {self._get_dataset_id()} downloads."
            )

        # Setup output files
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine year-month combinations to process
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
        
        # Use ThreadPoolExecutor for parallel downloads
        # Monthly chunks are smaller, so we can potentially use more workers,
        # but CDS still has per-user limits on active requests.
        max_workers = 2
        
        logging.info(f"Starting parallel download for {len(ym_range)} months with {max_workers} workers...")
        
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
                        logging.info(f"Completed processing for {year}-{month:02d}")
                except Exception as exc:
                    logging.error(f"Processing for {year}-{month:02d} generated an exception: {exc}")
                    # Cancel remaining and raise
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise exc

        # Merge all monthly chunks
        chunk_files.sort()
        
        try:
            if not chunk_files:
                raise RuntimeError("No data downloaded")

            # Check if aggregation is disabled
            if not self.config.get("AGGREGATE_FORCING_FILES", False):
                logging.info("Skipping aggregation of forcing files as per configuration")
                
                final_files = []
                for chunk_file in chunk_files:
                    # Rename from ..._processed_YYYYMM_temp.nc to ..._YYYYMM.nc
                    # Example: domain_CARRA_processed_201501_temp.nc -> domain_CARRA_201501.nc
                    new_name = chunk_file.name.replace("_processed_", "_").replace("_temp.nc", ".nc")
                    final_path = output_dir / new_name
                    
                    if final_path.exists():
                        final_path.unlink()
                        
                    chunk_file.rename(final_path)
                    final_files.append(final_path)
                    logging.info(f"Saved monthly file: {final_path.name}")
                
                # Return the directory containing the files
                return output_dir

            logging.info(f"Merging {len(chunk_files)} monthly chunks...")
            # open_mfdataset creates a dask-backed dataset, good for memory
            with xr.open_mfdataset(chunk_files, combine='by_coords') as ds_final:
                # Save final dataset
                final_f = self._save_final_dataset(ds_final, output_dir)

            # Validate that all required variables are present
            self._validate_required_variables(final_f)

        except Exception as e:
            logging.error(f"Error during merge: {e}")
            raise e
        finally:
            # Cleanup processed chunks (only if they still exist)
            for f in chunk_files:
                if f.exists():
                    try:
                        f.unlink()
                    except OSError:
                        pass

        return final_f

    def _download_and_process_month(self, year: int, month: int, output_dir: Path) -> Path:
        """Helper to download and process a single month (executed in thread)."""
        # Create a thread-local client
        c = cdsapi.Client()
        
        logging.info(f"Processing {self._get_dataset_id()} for {year}-{month:02d}...")
        
        current_months = [f"{month:02d}"]
        current_years = [str(year)]
        
        # Days (all days, API handles invalid dates like Feb 31)
        days = [f"{d:02d}" for d in range(1, 32)]
        hours = self._get_time_hours()

        # Temp files for this month
        af = output_dir / f"{self.domain_name}_{self._get_dataset_id()}_analysis_{year}{month:02d}_temp.nc"
        ff = output_dir / f"{self.domain_name}_{self._get_dataset_id()}_forecast_{year}{month:02d}_temp.nc"

        try:
            # Build requests
            analysis_req = self._build_analysis_request(current_years, current_months, days, hours)
            forecast_req = self._build_forecast_request(current_years, current_months, days, hours)

            # Debug: Log what variables we're requesting
            logging.info(f"Requesting forecast variables: {forecast_req.get('variable', [])}")

            # Download both products
            logging.info(f"Downloading {self._get_dataset_id()} analysis data for {self.domain_name} ({year}-{month:02d})...")
            self._retrieve_with_retry(c, self._get_dataset_name(), analysis_req, str(af))

            logging.info(f"Downloading {self._get_dataset_id()} forecast data for {self.domain_name} ({year}-{month:02d})...")
            self._retrieve_with_retry(c, self._get_dataset_name(), forecast_req, str(ff))

            # Debug: Check what variables were actually downloaded
            with xr.open_dataset(ff) as dsf_debug:
                logging.info(f"Forecast file variables: {list(dsf_debug.data_vars)}")

            # Process and merge this month's data
            ds_chunk = self._process_and_merge_datasets(af, ff)

            # Debug: Check what variables are in the processed chunk
            logging.info(f"Processed chunk variables: {list(ds_chunk.data_vars)}")

            # Save chunk to disk
            chunk_path = output_dir / f"{self.domain_name}_{self._get_dataset_id()}_processed_{year}{month:02d}_temp.nc"
            ds_chunk.to_netcdf(chunk_path)
            ds_chunk.close()
            
            return chunk_path
            
        finally:
            # Cleanup raw downloads for this month
            # Temporarily keep first month's files for debugging
            if year == 2015 and month == 1:
                logging.info(f"DEBUG: Keeping raw files for inspection: {af}, {ff}")
            else:
                if af.exists(): af.unlink()
                if ff.exists(): ff.unlink()

    def _download_and_process_year(self, year: int, output_dir: Path) -> Path:
        """Deprecated: Use _download_and_process_month instead."""
        # Kept as a placeholder to avoid breaking potential external calls,
        # but internally we now use monthly chunks.
        raise NotImplementedError("Use _download_and_process_month instead")

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
        for attempt in range(max_retries + 1):
            try:
                client.retrieve(dataset_name, request, target_path)
                return  # Success
            except Exception as e:
                error_msg = str(e)

                # Check if it's a 403 error
                is_403 = "403" in error_msg or "Forbidden" in error_msg

                # Check if it's worth retrying
                should_retry = is_403 or "temporarily" in error_msg.lower() or "maintenance" in error_msg.lower()

                if attempt < max_retries and should_retry:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logging.warning(
                        f"CDS request failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}"
                    )
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    # Last attempt or non-retryable error
                    if is_403:
                        logging.error(
                            f"CDS API returned 403 Forbidden error. This may indicate:\n"
                            f"  1. Temporary service maintenance (retry later)\n"
                            f"  2. Dataset license not accepted (visit https://cds.climate.copernicus.eu/datasets/{dataset_name})\n"
                            f"  3. API credentials issue (check ~/.cdsapirc)\n"
                            f"  4. Rate limiting (too many requests)"
                        )
                    raise

    def _get_time_hours(self) -> List[str]:
        """Generate hourly time strings based on temporal resolution."""
        resolution = self._get_temporal_resolution()
        if not self.start_date or not self.end_date:
            return [f"{h:02d}:00" for h in range(0, 24, resolution)]
        times = pd.date_range(self.start_date, self.end_date, freq=f"{resolution}h")
        if times.empty:
            return [f"{self.start_date:%H}:00"]
        hours = sorted({t.strftime("%H:00") for t in times})
        return hours

    def _build_analysis_request(
        self, years: List[int], months: List[str], days: List[str], hours: List[str]
    ) -> Dict[str, Any]:
        """Build CDS API request for analysis product."""
        request = {
            "level_type": "surface_or_atmosphere",
            "product_type": "analysis",
            "variable": self._get_analysis_variables(),
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf"
        }

        # Add domain if applicable (CARRA-specific)
        domain = self._get_domain()
        if domain:
            request["domain"] = domain

        # Add area if bbox is available to reduce download size
        if hasattr(self, 'bbox') and self.bbox:
            # Use a conservative 0.1 degree buffer to ensure enough coverage for grid points
            n = min(90, self.bbox['lat_max'] + 0.1)
            w = self.bbox['lon_min'] - 0.1
            s = max(-90, self.bbox['lat_min'] - 0.1)
            e = self.bbox['lon_max'] + 0.1
            request["area"] = self._get_cds_area(n, w, s, e)

        # Add subclass-specific parameters (e.g., CERRA's data_type)
        request.update(self._get_additional_request_params())

        return request

    def _build_forecast_request(
        self, years: List[int], months: List[str], days: List[str], hours: List[str]
    ) -> Dict[str, Any]:
        """Build CDS API request for forecast product."""
        request = {
            "level_type": "surface_or_atmosphere",
            "product_type": "forecast",
            "leadtime_hour": [self._get_leadtime_hour()],
            "variable": self._get_forecast_variables(),
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf"
        }

        # Add domain if applicable
        domain = self._get_domain()
        if domain:
            request["domain"] = domain

        # Add area if bbox is available
        if hasattr(self, 'bbox') and self.bbox:
            n = min(90, self.bbox['lat_max'] + 0.1)
            w = self.bbox['lon_min'] - 0.1
            s = max(-90, self.bbox['lat_min'] - 0.1)
            e = self.bbox['lon_max'] + 0.1
            request["area"] = self._get_cds_area(n, w, s, e)

        # Add subclass-specific parameters
        request.update(self._get_additional_request_params())

        return request

    def _process_and_merge(
        self, analysis_file: Path, forecast_file: Path, output_dir: Path
    ) -> Path:
        """Process, merge, subset, and save final dataset."""
        dsm = self._process_and_merge_datasets(analysis_file, forecast_file)

        # Save final dataset
        final_f = self._save_final_dataset(dsm, output_dir)

        return final_f

    def _process_and_merge_datasets(
        self, analysis_file: Path, forecast_file: Path
    ) -> xr.Dataset:
        """Process, merge, and subset to return an in-memory dataset."""
        with xr.open_dataset(analysis_file) as dsa, xr.open_dataset(forecast_file) as dsf:
            # Standardize time dimension names
            dsa = self._standardize_time_dimension(dsa)
            dsf = self._standardize_time_dimension(dsf)

            logging.info(
                f"{self._get_dataset_id()} analysis time range: "
                f"{self._format_time_range(dsa)}"
            )
            logging.info(
                f"{self._get_dataset_id()} forecast time range (pre-leadtime): "
                f"{self._format_time_range(dsf)}"
            )

            # Align forecast time (correct for leadtime offset)
            leadtime_hours = int(self._get_leadtime_hour())
            dsf["time"] = dsf["time"] - pd.Timedelta(hours=leadtime_hours)
            logging.info(
                f"{self._get_dataset_id()} forecast time range (post-leadtime): "
                f"{self._format_time_range(dsf)}"
            )

            # Merge datasets
            dsm = xr.merge([dsa, dsf], join="inner")
            logging.info(
                f"{self._get_dataset_id()} merged time range: "
                f"{self._format_time_range(dsm)}"
            )
            logging.info(f"Variables after merge: {list(dsm.data_vars)}")

            # Spatial subsetting
            if hasattr(self, "bbox") and self.bbox:
                dsm = self._spatial_subset(dsm)

            # Rename to SUMMA standards
            dsm = self._rename_variables(dsm)
            logging.info(f"Variables after rename: {list(dsm.data_vars)}")

            # Calculate derived variables
            dsm = self._calculate_derived_variables(dsm)

            # Unit conversions
            dsm = self._convert_units(dsm)

            # Temporal subsetting
            dsm = dsm.sel(time=slice(self.start_date, self.end_date))

            return dsm.load()

    def _format_time_range(self, ds: xr.Dataset) -> str:
        if "time" not in ds or ds["time"].size == 0:
            return "empty"
        times = pd.to_datetime(ds["time"].values)
        return f"{times.min()} -> {times.max()} ({len(times)} steps)"

    def _expected_times(self) -> Optional[pd.DatetimeIndex]:
        resolution = self._get_temporal_resolution()
        if not resolution:
            return None
        freq = f"{resolution}h"
        return pd.date_range(self.start_date, self.end_date, freq=freq)

    def _get_time_len(self, dataset_path: Path) -> int:
        try:
            with xr.open_dataset(dataset_path) as ds:
                if "time" not in ds:
                    return 0
                return len(ds["time"])
        except Exception as exc:
            logging.warning(f"Failed to read time dimension from {dataset_path}: {exc}")
            return 0

    def _download_per_timestep(
        self, output_dir: Path, expected_times: pd.DatetimeIndex
    ) -> Path:
        if self._get_dataset_id() == "CARRA":
            raise RuntimeError(
                "CARRA CDS requests reject per-timestep retrieval; "
                "use the multi-hour request only."
            )
        c = cdsapi.Client()
        datasets = []
        domain_name = self.domain_name
        dataset_id = self._get_dataset_id()

        for ts in expected_times:
            years = [str(ts.year)]
            months = [ts.strftime("%m")]
            days = [ts.strftime("%d")]
            hours = [f"{ts:%H}:00"]

            analysis_req = self._build_analysis_request(years, months, days, hours)
            forecast_req = self._build_forecast_request(years, months, days, hours)

            af = output_dir / f"{domain_name}_{dataset_id}_analysis_{ts:%Y%m%d%H}.nc"
            ff = output_dir / f"{domain_name}_{dataset_id}_forecast_{ts:%Y%m%d%H}.nc"

            logging.info(
                f"Downloading {dataset_id} timestep {ts:%Y-%m-%d %H:%M} (analysis/forecast)"
            )
            self._retrieve_with_retry(c, self._get_dataset_name(), analysis_req, str(af))
            self._retrieve_with_retry(c, self._get_dataset_name(), forecast_req, str(ff))

            ds = self._process_and_merge_datasets(af, ff)
            datasets.append(ds)

            for f in [af, ff]:
                if f.exists():
                    f.unlink()

        combined = xr.concat(datasets, dim="time").sortby("time")
        if hasattr(combined, "get_index"):
            time_index = combined.get_index("time")
            combined = combined.sel(time=~time_index.duplicated())

        return self._save_final_dataset(combined, output_dir)

    def _standardize_time_dimension(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename time dimension to standard 'time'."""
        time_name = 'valid_time' if 'valid_time' in ds.dims else 'time'
        return ds.rename({time_name: 'time'})

    def _spatial_subset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply spatial subsetting based on bounding box."""
        # Handle 1D regular lat/lon coordinates (e.g. from 'grid' interpolation)
        if 'latitude' in ds.dims and 'longitude' in ds.dims and \
           ds.latitude.ndim == 1 and ds.longitude.ndim == 1:
            # Data is already on a regular grid and likely subsetted by 'area' in request
            # We can use xarray's .sel() for additional precision or just return if satisfied
            lat = ds.latitude.values
            lon = ds.longitude.values
            
            # Subclasses expect matching shapes for lat/lon in _create_spatial_mask
            # For 1D coordinates, we create a 2D meshgrid for masking
            lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
            mask = self._create_spatial_mask(lat_2d, lon_2d)
            
            y_idx, x_idx = np.where(mask)
            if len(y_idx) > 0:
                ds = ds.isel(latitude=slice(y_idx.min(), y_idx.max() + 1),
                            longitude=slice(x_idx.min(), x_idx.max() + 1))
                logging.info(f"Spatially subsetted 1D grid to {ds.dims['latitude']}x{ds.dims['longitude']}")
            return ds

        # Handle native 2D grid (usually with 'x' and 'y' dimensions)
        lat = ds.latitude.values
        lon = ds.longitude.values

        # Create spatial mask (subclass-specific longitude handling)
        mask = self._create_spatial_mask(lat, lon)

        # np.where returns a tuple of arrays, one for each dimension
        indices = np.where(mask)
        if len(indices) < 2:
             logging.warning("Mask is not 2D, skipping spatial subsetting")
             return ds
             
        y_idx, x_idx = indices
        if len(y_idx) > 0:
            # Add buffer (subclass can override)
            buffer = self._get_spatial_buffer()
            
            # Determine dimension names (often 'y'/'x' or 'rlat'/'rlon')
            y_dim = 'y' if 'y' in ds.dims else ('rlat' if 'rlat' in ds.dims else None)
            x_dim = 'x' if 'x' in ds.dims else ('rlon' if 'rlon' in ds.dims else None)
            
            if y_dim and x_dim:
                y_min = max(0, y_idx.min() - buffer)
                y_max = min(ds.dims[y_dim] - 1, y_idx.max() + buffer)
                x_min = max(0, x_idx.min() - buffer)
                x_max = min(ds.dims[x_dim] - 1, x_idx.max() + buffer)

                ds = ds.isel({y_dim: slice(y_min, y_max + 1), x_dim: slice(x_min, x_max + 1)})
                logging.info(f"Spatially subsetted to {ds.dims[y_dim]}x{ds.dims[x_dim]} grid")
            else:
                logging.warning(f"Could not find x/y dimensions for subsetting in {list(ds.dims)}")
        else:
            logging.warning(f"No grid points found in bbox {self.bbox}, keeping full domain")

        return ds

    def _rename_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename variables to SUMMA standards using centralized VariableStandardizer."""
        standardizer = VariableStandardizer()
        dataset_id = self._get_dataset_id()
        return standardizer.standardize(ds, dataset_id)

    def _calculate_derived_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate derived meteorological variables."""
        # Wind speed from components (if not already present)
        if 'windspd_u' in ds and 'windspd_v' in ds and 'windspd' not in ds:
            ds['windspd'] = np.sqrt(ds['windspd_u']**2 + ds['windspd_v']**2)

        # Specific humidity from relative humidity
        if 'relhum' in ds and 'airtemp' in ds and 'airpres' in ds:
            ds['spechum'] = self._calculate_specific_humidity(
                ds['airtemp'], ds['relhum'], ds['airpres']
            )

        return ds

    def _calculate_specific_humidity(
        self, T: xr.DataArray, RH: xr.DataArray, P: xr.DataArray
    ) -> xr.DataArray:
        """
        Calculate specific humidity from temperature, RH, and pressure.

        Uses Magnus formula for saturation vapor pressure. Subclasses can
        override _get_magnus_denominator() for dataset-specific formulas.
        """
        # Saturation vapor pressure (Magnus formula)
        T_celsius = T - 273.15
        denominator = self._get_magnus_denominator(T_celsius)
        es = 611.2 * np.exp(17.67 * T_celsius / denominator)

        # Actual vapor pressure
        e = (RH / 100.0) * es

        # Specific humidity
        return (0.622 * e) / (P - 0.378 * e)

    def _detect_temporal_resolution_seconds(self, ds: xr.Dataset) -> Optional[float]:
        """
        Detect temporal resolution from dataset by analyzing time coordinate.

        Returns:
            Resolution in seconds, or None if detection fails.
        """
        if 'time' not in ds.dims or ds.sizes['time'] < 2:
            return None

        try:
            times = pd.to_datetime(ds['time'].values)
            diffs = np.diff(times)
            median_seconds = float(np.median(diffs) / np.timedelta64(1, 's'))
            logging.debug(f"Detected temporal resolution: {median_seconds} seconds ({median_seconds/3600:.1f} hours)")
            return median_seconds
        except Exception as e:
            logging.warning(f"Failed to detect temporal resolution: {e}")
            return None

    def _convert_units(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert units to SUMMA standards."""
        # Try to detect resolution from data first, fall back to hardcoded value
        detected_seconds = self._detect_temporal_resolution_seconds(ds)
        if detected_seconds is not None:
            resolution_seconds = detected_seconds
            logging.info(f"Using detected temporal resolution: {resolution_seconds/3600:.1f} hours")
        else:
            resolution_hours = self._get_temporal_resolution()
            resolution_seconds = resolution_hours * 3600
            logging.info(f"Using default temporal resolution: {resolution_hours} hours")

        # Precipitation: kg/m2 per leadtime -> m/s
        if 'pptrate' in ds:
            ds['pptrate'] = (ds['pptrate'] * 0.001) / resolution_seconds

        # Radiation: J/m2 per leadtime -> W/m2
        if 'SWRadAtm' in ds:
            ds['SWRadAtm'] = ds['SWRadAtm'] / resolution_seconds

        if 'LWRadAtm' in ds:
            ds['LWRadAtm'] = ds['LWRadAtm'] / resolution_seconds

        return ds

    def _save_final_dataset(self, ds: xr.Dataset, output_dir: Path) -> Path:
        """Save final processed dataset."""
        final_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm',
                     'windspd', 'spechum', 'LWRadAtm']
        available_vars = [v for v in final_vars if v in ds.variables]

        final_f = output_dir / (
            f"{self.domain_name}_{self._get_dataset_id()}_"
            f"{self.start_date.year}-{self.end_date.year}.nc"
        )
        ds[available_vars].to_netcdf(final_f)

        return final_f

    def _validate_required_variables(self, file_path: Path) -> None:
        """
        Validate that all required variables are present in the downloaded file.

        Raises:
            ValueError: If required variables are missing
        """
        required_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm',
                        'windspd', 'spechum', 'LWRadAtm']

        with xr.open_dataset(file_path) as ds:
            missing_vars = [v for v in required_vars if v not in ds.variables]

            if missing_vars:
                logging.warning(
                    f"{self._get_dataset_id()} download is missing required variables: {missing_vars}. "
                    f"Available variables: {list(ds.variables)}"
                )
                logging.warning(
                    "This may indicate an issue with the CDS API request. "
                    "Check that forecast variables are being requested correctly."
                )
                raise ValueError(
                    f"Downloaded {self._get_dataset_id()} file is missing required variables: {missing_vars}"
                )

            logging.info(f"Validated {self._get_dataset_id()} file has all required variables: {required_vars}")

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    def _get_dataset_name(self) -> str:
        """Return CDS dataset name (e.g., 'reanalysis-carra-single-levels')."""
        pass

    @abstractmethod
    def _get_dataset_id(self) -> str:
        """Return short dataset ID for filenames (e.g., 'CARRA')."""
        pass

    @abstractmethod
    def _get_domain(self) -> Optional[str]:
        """Return domain identifier or None if not applicable."""
        pass

    @abstractmethod
    def _get_temporal_resolution(self) -> int:
        """Return temporal resolution in hours (e.g., 1 for hourly, 3 for 3-hourly)."""
        pass

    @abstractmethod
    def _get_analysis_variables(self) -> List[str]:
        """Return list of analysis variables to download."""
        pass

    @abstractmethod
    def _get_forecast_variables(self) -> List[str]:
        """Return list of forecast variables to download."""
        pass

    @abstractmethod
    def _get_leadtime_hour(self) -> str:
        """Return leadtime hour as string (e.g., '1')."""
        pass

    @abstractmethod
    def _get_additional_request_params(self) -> Dict[str, Any]:
        """Return additional dataset-specific request parameters."""
        pass

    @abstractmethod
    def _create_spatial_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Create spatial mask for subsetting.

        Must handle dataset-specific longitude conventions (0-360 vs -180-180).
        """
        pass

    # Optional methods with sensible defaults (can be overridden)

    def _get_spatial_buffer(self) -> int:
        """Return number of grid cells to add as buffer (default: 0)."""
        return 0

    def _get_cds_area(self, n: float, w: float, s: float, e: float) -> List[float]:
        """
        Return [North, West, South, East] area for CDS request.
        Subclasses can override this for dataset-specific longitude handling.
        """
        return [n, w, s, e]

    def _get_magnus_denominator(self, T_celsius: xr.DataArray) -> xr.DataArray:
        """Return Magnus formula denominator (default: standard formula T + 243.5)."""
        return T_celsius + 243.5


@AcquisitionRegistry.register('CARRA')
class CARRAAcquirer(CDSRegionalReanalysisHandler):
    """
    CARRA (Copernicus Arctic Regional Reanalysis) data acquisition handler.

    Hourly data covering the Arctic region with special longitude handling (0-360°).
    """

    def _get_dataset_name(self) -> str:
        return "reanalysis-carra-single-levels"

    def _get_dataset_id(self) -> str:
        return "CARRA"

    def _get_domain(self) -> Optional[str]:
        return self.config.get("CARRA_DOMAIN", "west_domain")

    def _get_temporal_resolution(self) -> int:
        return 3  # 3-hourly (CARRA native resolution)

    def _get_analysis_variables(self) -> List[str]:
        return [
            "2m_temperature",
            "2m_relative_humidity",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure"
        ]

    def _get_forecast_variables(self) -> List[str]:
        return [
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "thermal_surface_radiation_downwards"  # Correct name: thermal comes BEFORE surface
        ]

    def _get_leadtime_hour(self) -> str:
        return "1"

    def _get_additional_request_params(self) -> Dict[str, Any]:
        return {"grid": [0.025, 0.025]}  # Force interpolation to allow 'area' cropping

    def _create_spatial_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Create mask with CARRA longitude handling (0-360 degrees)."""
        # Normalize bbox to [0, 360]
        target_lon_min = self.bbox['lon_min'] % 360
        target_lon_max = self.bbox['lon_max'] % 360

        # Handle wrapping around prime meridian
        if target_lon_min > target_lon_max:
            lon_mask = (lon >= target_lon_min) | (lon <= target_lon_max)
        else:
            lon_mask = (lon >= target_lon_min) & (lon <= target_lon_max)

        mask = (
            (lat >= self.bbox['lat_min']) & (lat <= self.bbox['lat_max']) &
            lon_mask
        )

        return mask

    def _get_spatial_buffer(self) -> int:
        return 2  # CARRA uses 2-cell buffer

    def _get_cds_area(self, n: float, w: float, s: float, e: float) -> List[float]:
        """Return normalized area for CARRA (0-360 longitude)."""
        # CARRA data is natively 0-360. CDS 'area' parameter for CARRA
        # works best when matching the native convention.
        return [n, w % 360, s, e % 360]

    def _get_magnus_denominator(self, T_celsius: xr.DataArray) -> xr.DataArray:
        return T_celsius - 29.65  # CARRA-specific formula


@AcquisitionRegistry.register('CERRA')
class CERRAAcquirer(CDSRegionalReanalysisHandler):
    """
    CERRA (Copernicus European Regional Reanalysis) data acquisition handler.

    3-hourly data covering Europe with standard longitude handling (-180 to 180°).
    """

    def _get_dataset_name(self) -> str:
        return "reanalysis-cerra-single-levels"

    def _get_dataset_id(self) -> str:
        return "CERRA"

    def _get_domain(self) -> Optional[str]:
        return None  # CERRA doesn't use domain parameter

    def _get_temporal_resolution(self) -> int:
        return 3  # 3-hourly

    def _get_analysis_variables(self) -> List[str]:
        return [
            "2m_temperature",
            "2m_relative_humidity",
            "surface_pressure",
            "10m_wind_speed"  # CERRA provides combined wind speed
        ]

    def _get_forecast_variables(self) -> List[str]:
        return [
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "thermal_surface_radiation_downwards"  # Correct name: thermal comes BEFORE surface
        ]

    def _get_leadtime_hour(self) -> str:
        return "1"

    def _get_additional_request_params(self) -> Dict[str, Any]:
        return {
            "data_type": "reanalysis",
            "grid": [0.05, 0.05]  # Force interpolation to allow 'area' cropping
        }

    def _create_spatial_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Create mask with CERRA longitude handling (-180 to 180 degrees)."""
        # Standard longitude handling for European domain
        mask = (
            (lat >= self.bbox['lat_min']) & (lat <= self.bbox['lat_max']) &
            (lon >= self.bbox['lon_min']) & (lon <= self.bbox['lon_max'])
        )

        return mask

    # Uses default implementations for:
    # - _get_spatial_buffer (0)
    # - _get_magnus_denominator (standard T + 243.5)
