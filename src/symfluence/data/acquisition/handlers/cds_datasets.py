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
from symfluence.data.utils import VariableStandardizer


class CDSRegionalReanalysisHandler(BaseAcquisitionHandler, ABC):
    """
    Abstract base handler for CDS regional reanalysis products.

    Implements the common workflow for downloading high-resolution regional
    reanalysis data from the Copernicus Climate Data Store (CDS). Handles
    the complexity of combining analysis and forecast products, which is
    required for complete forcing variable coverage in CARRA/CERRA.

    Key features:
    - Dual-product strategy: Analysis products (hourly) + Forecast products (3-hourly)
    - Parallel monthly chunk downloads with configurable workers
    - Automatic time alignment and coordinate standardization
    - Spatial subsetting to bounding box with native grid preservation
    - Unit conversions and variable derivations (wind speed from U/V components)
    - Optional aggregation of monthly files into single dataset

    Subclasses (CARRAAcquirer, CERRAAcquirer) implement:
    - _get_dataset_id(): CDS dataset identifier
    - _get_analysis_variables(): Variables from analysis product
    - _get_forecast_variables(): Variables from forecast product
    - _get_temporal_resolution(): Native dataset time step
    - _transform_coordinates(): Spatial coordinate handling
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
        """Download and process a single month of data (executed in thread pool).

        This method is called from the parallel executor. Each thread gets its own
        CDS API client to avoid thread safety issues. The method:
        1. Downloads both analysis and forecast products for the month
        2. Merges them accounting for forecast leadtime offset
        3. Performs spatial/temporal subsetting and unit conversions
        4. Saves the processed chunk and cleans up raw downloads

        The dual-product approach is necessary because:
        - Analysis products have some variables not in forecasts (e.g., analysis wind)
        - Forecast products have variables not in analysis (e.g., precipitation accumulations)
        - Merging requires careful time alignment since forecasts are offset by leadtime

        Args:
            year: Year to download (e.g., 2015)
            month: Month to download (1-12)
            output_dir: Directory where temporary and output files are saved

        Returns:
            Path: Location of processed monthly chunk file (temporary .nc file)

        Side Effects:
            - Creates temporary files in output_dir (cleaned up in finally block)
            - For debugging: keeps first month's raw files for inspection
        """
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
        """Generate hourly time strings for CDS request based on dataset resolution.

        This method creates the list of hours to request from the CDS API. It handles:
        - Extracting unique hours from the full date range
        - Respecting the dataset's native temporal resolution (1h, 3h, etc.)
        - Returning sorted, unique hour strings for the API

        The CDS API requires explicit hour strings (e.g., ['00:00', '03:00', '06:00']).
        Rather than hardcoding hours, this method derives them from the date range
        to ensure only relevant times are requested.

        Returns:
            List[str]: Hours as strings like ['00:00', '03:00', '06:00', ...]

        Example:
            For CARRA (3-hourly) from 2015-01-05 to 2015-01-06:
            Returns: ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
        """
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
        """Build CDS API request dictionary for analysis product.

        Constructs the parameter dictionary required by the CDS API to download
        the analysis product. The analysis product contains variables that are
        analyzed from observations (e.g., temperature, pressure, wind components).

        The method handles:
        - Standard parameters common to all regional reanalysis (product_type, time, etc.)
        - Spatial subsetting via bounding box (adds 'area' parameter to reduce download size)
        - Dataset-specific parameters via _get_additional_request_params() (domain, data_type)
        - Domain specification (CARRA-specific)

        Why separate analysis and forecast?
            Analysis products: hourly, contain wind components, temperature
            Forecast products: 3-hourly, contain accumulated fields like precipitation
            Both are needed to get complete forcing dataset

        Args:
            years: List of year strings ['2015', '2016']
            months: List of month strings ['01', '02']
            days: List of day strings ['01', '02', ..., '31']
            hours: List of hour strings ['00:00', '03:00', ...]

        Returns:
            Dict: CDS API request parameters
        """
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
        """Process, merge, and subset analysis and forecast datasets.

        This is the core processing method that handles the complex workflow of
        combining two separate CDS products into a single coherent dataset:

        Workflow:
            1. **Time standardization**: Rename time dimensions to 'time' (CDS uses 'valid_time')
            2. **Leadtime correction**: Forecast data includes a leadtime offset that must
               be removed so forecast times align with analysis times
            3. **Merging**: Inner join on time dimension to keep only overlapping times
            4. **Spatial subsetting**: Keep only grid points within bounding box
            5. **Variable renaming**: Map dataset-specific names to SYMFLUENCE standard names
            6. **Derived variables**: Calculate wind speed from U/V components, specific humidity
            7. **Unit conversions**: Convert accumulated fields to rates (kg/m2 -> m/s)
            8. **Temporal subsetting**: Keep only requested date range

        Why Leadtime Correction?
            Forecast products in CDS include a 'leadtime' offset (e.g., forecast for
            2015-01-05 12:00 issued at 2015-01-05 11:00). This offset must be subtracted
            from the forecast time dimension to align with analysis times.

        Args:
            analysis_file: Path to raw analysis product NetCDF
            forecast_file: Path to raw forecast product NetCDF

        Returns:
            xr.Dataset: Merged, processed dataset in memory with standard variable names

        Side Effects:
            - Loads both files into memory (can be large for full months)
            - Logs time ranges and variable names for debugging
        """
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
        """Calculate specific humidity from temperature, relative humidity, and pressure.

        This method uses the Magnus approximation formula for saturation vapor pressure,
        which is accurate to ~0.1% for meteorological applications. Specific humidity
        is required for many hydrological models but not always directly available from
        reanalysis products.

        The Magnus formula provides saturation vapor pressure as:
            e_s = 611.2 * exp(17.67 * T_c / (T_c + D))
        where T_c is temperature in Celsius and D is a denominator (default: 243.5).
        Some datasets (e.g., CARRA) use different denominators empirically tuned
        for better accuracy in their regions.

        Then specific humidity is:
            q = (0.622 * e) / (P - 0.378 * e)
        where e is actual vapor pressure and P is pressure in Pa.

        Args:
            T: Temperature as xarray DataArray (Kelvin)
            RH: Relative humidity as DataArray (percent, 0-100)
            P: Pressure as DataArray (Pa)

        Returns:
            xr.DataArray: Specific humidity (kg/kg) with same dimensions as inputs

        Note:
            Subclasses can override _get_magnus_denominator() for dataset-specific
            empirical formulas (e.g., CARRA uses T_c - 29.65 instead of T_c + 243.5)
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
        """Return CDS dataset name for API.

        This is the exact string used in CDS API client.retrieve() calls.
        Examples:
            - 'reanalysis-carra-single-levels' for CARRA
            - 'reanalysis-cerra-single-levels' for CERRA

        Returns:
            str: Official CDS dataset identifier
        """
        pass

    @abstractmethod
    def _get_dataset_id(self) -> str:
        """Return short dataset ID for filenames and logging.

        This is used in output filenames, logging messages, and parameter names.
        Should be uppercase and 3-6 characters.
        Examples:
            - 'CARRA'
            - 'CERRA'

        Returns:
            str: Short identifier for display and filenames
        """
        pass

    @abstractmethod
    def _get_domain(self) -> Optional[str]:
        """Return domain identifier for CDS API request or None if not applicable.

        Some regional reanalysis datasets (e.g., CARRA) require a domain parameter
        to specify which region to download. This should be read from configuration
        or hardcoded if fixed.
        Examples:
            - 'west_domain' for CARRA Arctic west
            - 'east_domain' for CARRA Arctic east
            - None for CERRA (no domain parameter needed)

        Returns:
            Optional[str]: Domain string for API request, or None
        """
        pass

    @abstractmethod
    def _get_temporal_resolution(self) -> int:
        """Return temporal resolution in hours.

        Indicates how frequently data is available (e.g., hourly, 3-hourly).
        Used to generate time hour lists and detect temporal resolution from data.
        Examples:
            - 1 for hourly data
            - 3 for 3-hourly data

        Returns:
            int: Hours between timesteps
        """
        pass

    @abstractmethod
    def _get_analysis_variables(self) -> List[str]:
        """Return list of analysis variables to download from CDS.

        Analysis products are generated from observations and include variables
        like temperature, pressure, and wind components. These variable names
        are the exact strings used in CDS API requests.
        Examples for CARRA:
            - '2m_temperature'
            - '10m_u_component_of_wind'

        Returns:
            List[str]: Variable names as used in CDS API
        """
        pass

    @abstractmethod
    def _get_forecast_variables(self) -> List[str]:
        """Return list of forecast variables to download from CDS.

        Forecast products include accumulated/derived variables not in analysis,
        particularly precipitation and radiation. These variable names are
        the exact strings used in CDS API requests.
        Examples for CARRA:
            - 'total_precipitation'
            - 'surface_solar_radiation_downwards'

        Returns:
            List[str]: Variable names as used in CDS API
        """
        pass

    @abstractmethod
    def _get_leadtime_hour(self) -> str:
        """Return leadtime hour as string for forecast product request.

        Forecast products are issued at a given time with a leadtime offset.
        This specifies which leadtime to request. Most datasets use '1' (1-hour
        forecast) which provides the best temporal alignment.

        Returns:
            str: Leadtime hour as string (e.g., '1', '3')
        """
        pass

    @abstractmethod
    def _get_additional_request_params(self) -> Dict[str, Any]:
        """Return dataset-specific CDS API request parameters.

        Different datasets require different parameters. This method allows
        subclasses to add dataset-specific options without modifying the
        base request building logic.
        Examples:
            - {'grid': [0.025, 0.025]} to specify output grid resolution
            - {'data_type': 'reanalysis'} for CERRA
            - {'domain': ...} handled separately in _get_domain()

        Returns:
            Dict[str, Any]: Additional parameters to add to CDS API requests
        """
        pass

    @abstractmethod
    def _create_spatial_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Create spatial mask for subsetting to bounding box.

        Creates a boolean 2D mask indicating which grid points fall within
        the domain's bounding box. Must handle dataset-specific longitude
        conventions (0-360 vs -180-180 degrees).

        Args:
            lat: Latitude array (1D or 2D) in degrees
            lon: Longitude array (1D or 2D) in degrees.
                 May be in [0, 360] or [-180, 180] depending on dataset.

        Returns:
            np.ndarray: Boolean mask (same shape as lat/lon) where True indicates
                       points within bounding box

        Example:
            For a 100x100 grid with 5 points in bbox:
            Returns: array of shape (100, 100) with 5 True values and rest False
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
    """CARRA (Copernicus Arctic Regional Reanalysis) data acquisition handler.

    Handles download and processing of CARRA data: a high-resolution (2.5 km) Arctic
    reanalysis covering 1980-present. Key characteristics:
    - Temporal: 3-hourly
    - Domain: Arctic region (configurable: 'west_domain' or 'east_domain')
    - Longitude: 0-360° convention (requires special handling at dateline)
    - Analysis + Forecast products to get complete variable set

    Special Handling:
    - Longitude normalization: CARRA uses [0, 360] instead of [-180, 180]
    - Dateline wrapping: Domains spanning the prime meridian need special masking
    - Magnus formula: Uses T_c - 29.65 for specific humidity calculation
    - Spatial buffer: 2-cell buffer for lat/lon subsetting to ensure coverage
    - Grid interpolation: 0.025° forced to enable spatial subsetting via 'area' param

    Typical Configuration:
        CARRA_DOMAIN: 'west_domain'  # or 'east_domain'
        AGGREGATE_FORCING_FILES: True  # Merge monthly chunks
    """

    def _get_dataset_name(self) -> str:
        return "reanalysis-carra-single-levels"

    def _get_dataset_id(self) -> str:
        return "CARRA"

    def _get_domain(self) -> Optional[str]:
        """Return CARRA domain configuration.

        CARRA Arctic coverage is split into west and east domains to manage
        download and processing complexity. Configuration selects which to download.

        Returns:
            str: 'west_domain' (default) or 'east_domain' or other configured value
        """
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
        """Return CARRA-specific request parameters.

        Forces grid interpolation to 0.025° (native is 2.5 km ≈ 0.023°) to enable
        spatial subsetting via the 'area' parameter in CDS requests. Without this,
        CDS returns the full domain regardless of 'area' specification.

        Returns:
            Dict with 'grid' and 'domain' keys
        """
        return {"grid": [0.025, 0.025]}  # Force interpolation to allow 'area' cropping

    def _create_spatial_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Create mask with CARRA longitude handling (0-360 degrees).

        CARRA uses 0-360° longitude convention (unlike standard -180-180°).
        This requires special handling:
        1. Normalize bbox to [0, 360]
        2. Handle wrapping at prime meridian (e.g., [350, 10] wraps around 0°)
        3. Apply mask using either AND or OR depending on whether span crosses dateline

        Args:
            lat: Latitude array
            lon: Longitude array in [0, 360] convention

        Returns:
            np.ndarray: Boolean mask for grid points in bounding box
        """
        # Normalize bbox to [0, 360]
        target_lon_min = self.bbox['lon_min'] % 360
        target_lon_max = self.bbox['lon_max'] % 360

        # Handle wrapping around prime meridian
        # Example: lon_min=350, lon_max=10 should match [350-360] OR [0-10]
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
        """Return grid cell buffer for spatial subsetting.

        CARRA data often has small discontinuities at domain edges. A 2-cell
        buffer ensures continuous coverage and prevents edge artifacts.

        Returns:
            int: Number of grid cells to add around masked region
        """
        return 2  # CARRA uses 2-cell buffer

    def _get_cds_area(self, n: float, w: float, s: float, e: float) -> List[float]:
        """Return normalized area for CARRA (0-360 longitude).

        CARRA data is natively 0-360°. CDS 'area' parameter for CARRA works best
        when matching the native convention, so we normalize all longitudes.

        Args:
            n, w, s, e: North, West, South, East bounds in various conventions

        Returns:
            List[float]: [North, West, South, East] normalized to [0, 360]
        """
        # CARRA data is natively 0-360. CDS 'area' parameter for CARRA
        # works best when matching the native convention.
        return [n, w % 360, s, e % 360]

    def _get_magnus_denominator(self, T_celsius: xr.DataArray) -> xr.DataArray:
        """Return Magnus formula denominator for CARRA.

        CARRA uses an empirically-tuned Magnus formula for better accuracy
        in high latitudes. The standard formula (T + 243.5) is less accurate
        above 60°N.

        Args:
            T_celsius: Temperature in Celsius

        Returns:
            xr.DataArray: Denominator for Magnus formula (T - 29.65)
        """
        return T_celsius - 29.65  # CARRA-specific formula for Arctic accuracy


@AcquisitionRegistry.register('CERRA')
class CERRAAcquirer(CDSRegionalReanalysisHandler):
    """CERRA (Copernicus European Regional Reanalysis) data acquisition handler.

    Handles download and processing of CERRA data: a high-resolution (5.5 km)
    European reanalysis covering 1985-present. Key characteristics:
    - Temporal: 3-hourly
    - Domain: Europe (fixed, no domain parameter needed)
    - Longitude: Standard [-180, 180]° convention (simple masking)
    - Analysis + Forecast products to get complete variable set

    Key Differences from CARRA:
    - No domain parameter (covers all of Europe automatically)
    - Standard latitude/longitude handling (no prime meridian wrapping)
    - Wind speed provided directly (not decomposed U/V in analysis)
    - Coarser grid: 0.05° ≈ 5.5 km (vs CARRA 0.025° ≈ 2.5 km)
    - No spatial buffer needed (finer interpolation issues not present)
    - Standard Magnus formula for specific humidity

    Typical Configuration:
        AGGREGATE_FORCING_FILES: True  # Merge monthly chunks into single file
    """

    def _get_dataset_name(self) -> str:
        return "reanalysis-cerra-single-levels"

    def _get_dataset_id(self) -> str:
        return "CERRA"

    def _get_domain(self) -> Optional[str]:
        """Return CERRA domain parameter.

        CERRA covers all of Europe and doesn't require a domain parameter
        (unlike CARRA which has west/east split). Returns None to omit
        from API request.

        Returns:
            None: No domain parameter for CERRA
        """
        return None  # CERRA doesn't use domain parameter

    def _get_temporal_resolution(self) -> int:
        return 3  # 3-hourly

    def _get_analysis_variables(self) -> List[str]:
        """Return CERRA analysis variables.

        Unlike CARRA, CERRA provides wind speed directly rather than U/V
        components. The base class _calculate_derived_variables will still
        attempt to calculate wind speed from U/V if present, which is handled
        gracefully.

        Returns:
            List[str]: Variable names from CERRA analysis product
        """
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
        """Return CERRA-specific request parameters.

        CERRA requires 'data_type': 'reanalysis' to distinguish from other
        European datasets. Also forces grid interpolation to 0.05° to enable
        spatial subsetting via 'area' parameter (similar to CARRA).

        Returns:
            Dict with 'data_type' and 'grid' keys
        """
        return {
            "data_type": "reanalysis",
            "grid": [0.05, 0.05]  # Force interpolation to allow 'area' cropping
        }

    def _create_spatial_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Create mask with CERRA longitude handling (-180 to 180 degrees).

        CERRA uses standard longitude convention [-180, 180]°, so masking is
        straightforward: no wrapping at dateline needed.

        Args:
            lat: Latitude array in degrees [-90, 90]
            lon: Longitude array in degrees [-180, 180]

        Returns:
            np.ndarray: Boolean mask for grid points in bounding box
        """
        # Standard longitude handling for European domain
        mask = (
            (lat >= self.bbox['lat_min']) & (lat <= self.bbox['lat_max']) &
            (lon >= self.bbox['lon_min']) & (lon <= self.bbox['lon_max'])
        )

        return mask

    # Uses default implementations for:
    # - _get_spatial_buffer (0) - CERRA doesn't need buffer due to interpolation method
    # - _get_magnus_denominator (standard T + 243.5) - standard formula works for Europe
    # - _get_cds_area (standard [N, W, S, E]) - no longitude normalization needed
