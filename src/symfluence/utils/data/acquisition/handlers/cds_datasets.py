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

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


class CDSRegionalReanalysisHandler(BaseAcquisitionHandler, ABC):
    """
    Abstract base handler for CDS regional reanalysis products.

    Implements common dual-product download strategy (analysis + forecast),
    time alignment, spatial subsetting, unit conversions, and variable derivations.

    Subclasses must implement abstract methods to specify dataset-specific
    configurations such as temporal resolution, variable lists, and spatial handling.
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process regional reanalysis data."""
        if not HAS_CDSAPI:
            raise ImportError(
                f"cdsapi package is required for {self._get_dataset_id()} downloads."
            )

        # Initialize CDS client
        c = cdsapi.Client()

        # Build temporal parameters
        years = list(range(self.start_date.year, self.end_date.year + 1))
        months = [f"{m:02d}" for m in range(self.start_date.month, self.end_date.month + 1)]
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        days = sorted(list(set([d.strftime('%d') for d in dates])))
        hours = self._get_time_hours()

        # Setup output files
        output_dir.mkdir(parents=True, exist_ok=True)
        af = output_dir / f"{self.domain_name}_{self._get_dataset_id()}_analysis_temp.nc"
        ff = output_dir / f"{self.domain_name}_{self._get_dataset_id()}_forecast_temp.nc"

        # Build requests
        analysis_req = self._build_analysis_request(years, months, days, hours)
        forecast_req = self._build_forecast_request(years, months, days, hours)

        # Download both products
        logging.info(f"Downloading {self._get_dataset_id()} analysis data for {self.domain_name}...")
        c.retrieve(self._get_dataset_name(), analysis_req, str(af))

        logging.info(f"Downloading {self._get_dataset_id()} forecast data for {self.domain_name}...")
        c.retrieve(self._get_dataset_name(), forecast_req, str(ff))

        # Process and merge
        final_f = self._process_and_merge(af, ff, output_dir)

        expected_times = self._expected_times()
        if expected_times is not None and self._get_dataset_id() != "CARRA":
            actual_len = self._get_time_len(final_f)
            if actual_len < len(expected_times):
                logging.warning(
                    f"{self._get_dataset_id()} returned {actual_len} timesteps for "
                    f"{len(expected_times)} requested. Re-downloading per timestep."
                )
                try:
                    final_f.unlink()
                except OSError:
                    pass
                final_f = self._download_per_timestep(output_dir, expected_times)

        # Cleanup temp files
        for f in [af, ff]:
            if f.exists():
                f.unlink()

        return final_f

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

            # Spatial subsetting
            if hasattr(self, "bbox") and self.bbox:
                dsm = self._spatial_subset(dsm)

            # Rename to SUMMA standards
            dsm = self._rename_variables(dsm)

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
            c.retrieve(self._get_dataset_name(), analysis_req, str(af))
            c.retrieve(self._get_dataset_name(), forecast_req, str(ff))

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
        lat = ds.latitude.values
        lon = ds.longitude.values

        # Create spatial mask (subclass-specific longitude handling)
        mask = self._create_spatial_mask(lat, lon)

        y_idx, x_idx = np.where(mask)
        if len(y_idx) > 0:
            # Add buffer (subclass can override)
            buffer = self._get_spatial_buffer()
            y_min = max(0, y_idx.min() - buffer)
            y_max = min(ds.dims['y'] - 1, y_idx.max() + buffer)
            x_min = max(0, x_idx.min() - buffer)
            x_max = min(ds.dims['x'] - 1, x_idx.max() + buffer)

            ds = ds.isel(y=slice(y_min, y_max + 1), x=slice(x_min, x_max + 1))
            logging.info(f"Spatially subsetted to {ds.dims['y']}x{ds.dims['x']} grid")
        else:
            logging.warning(f"No grid points found in bbox {self.bbox}, keeping full domain")

        return ds

    def _rename_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename variables to SUMMA standards."""
        rename_map = {
            't2m': 'airtemp',
            'sp': 'airpres',
            'u10': 'wind_u',
            'v10': 'wind_v',
            'si10': 'windspd',  # CERRA provides this directly
            'tp': 'pptrate',
            'ssrd': 'SWRadAtm',
            'strd': 'LWRadAtm'
        }
        return ds.rename({k: v for k, v in rename_map.items() if k in ds.variables})

    def _calculate_derived_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate derived meteorological variables."""
        # Wind speed from components (if not already present)
        if 'wind_u' in ds and 'wind_v' in ds and 'windspd' not in ds:
            ds['windspd'] = np.sqrt(ds['wind_u']**2 + ds['wind_v']**2)

        # Specific humidity from relative humidity
        if 'r2' in ds and 'airtemp' in ds and 'airpres' in ds:
            ds['spechum'] = self._calculate_specific_humidity(
                ds['airtemp'], ds['r2'], ds['airpres']
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

    def _convert_units(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert units to SUMMA standards."""
        resolution_hours = self._get_temporal_resolution()
        resolution_seconds = resolution_hours * 3600

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
        return 1  # Hourly

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
            "surface_thermal_radiation_downwards"
        ]

    def _get_leadtime_hour(self) -> str:
        return "1"

    def _get_additional_request_params(self) -> Dict[str, Any]:
        return {}  # CARRA doesn't need data_type parameter

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
            "surface_thermal_radiation_downwards"
        ]

    def _get_leadtime_hour(self) -> str:
        return "1"

    def _get_additional_request_params(self) -> Dict[str, Any]:
        return {"data_type": "reanalysis"}  # CERRA requires this

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
