"""
RHESSys Climate File Generator

Handles generation of RHESSys-compatible climate input files from forcing data.
Extracted from RHESSysPreprocessor for better organization and testability.

RHESSys uses text-based climate files with format:
    year month day hour value
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from symfluence.data.utils.variable_utils import VariableHandler

logger = logging.getLogger(__name__)


class RHESSysClimateGenerator:
    """
    Generates RHESSys-compatible climate input files.

    Handles:
    - Loading forcing data from various sources
    - Converting units and aggregating to daily
    - Computing derived variables (relative humidity from specific humidity)
    - Writing base station and climate files in RHESSys format
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.project_dir = Path(project_dir)
        self.domain_name = domain_name
        self.logger = logger or logging.getLogger(__name__)

        # Setup paths
        self.rhessys_input_dir = self.project_dir / "RHESSys_input"
        self.climate_dir = self.rhessys_input_dir / "clim"
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_raw_path = self.project_dir / 'forcing' / 'raw_data'

        # Get forcing dataset info
        forcing_ds = config.get('FORCING_DATASET', 'ERA5')
        self.forcing_dataset = forcing_ds.upper() if forcing_ds else 'ERA5'

    def generate_climate_files(
        self,
        start_date: datetime,
        end_date: datetime,
        catchment_path: Optional[Path] = None
    ) -> bool:
        """
        Generate all climate files for the simulation period.

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
            catchment_path: Path to catchment shapefile (for coordinates)

        Returns:
            True if successful
        """
        self.logger.info("Generating climate files...")
        self.climate_dir.mkdir(parents=True, exist_ok=True)

        try:
            ds = self._load_forcing_data(start_date, end_date)
        except FileNotFoundError as e:
            self.logger.warning(f"Could not load forcing data: {e}")
            self.logger.info("Creating synthetic climate data for testing")
            self._create_synthetic_climate(start_date, end_date)
            return True

        # Initialize variable handler
        VariableHandler(self.config, self.logger, self.forcing_dataset, 'RHESSys')

        # Get variable names from dataset
        precip_var = self._find_variable(ds, ['pr', 'precipitation', 'PREC', 'precip', 'pptrate'])
        temp_var = self._find_variable(ds, ['tas', 't2m', 'temp', 'airtemp', 'TEMP', 'temperature'])
        tmax_var = self._find_variable(ds, ['tasmax', 'tmax', 'TMAX'])
        tmin_var = self._find_variable(ds, ['tasmin', 'tmin', 'TMIN'])

        # Additional variables for ET calculation
        swrad_var = self._find_variable(ds, ['SWRadAtm', 'rsds', 'swdown', 'ssrd', 'shortwave_radiation', 'Kdown'])
        lwrad_var = self._find_variable(ds, ['LWRadAtm', 'rlds', 'lwdown', 'strd', 'longwave_radiation', 'Ldown'])
        wind_var = self._find_variable(ds, ['windspd', 'wind', 'sfcWind', 'wind_speed', 'ws', 'u10', 'v10'])
        spechum_var = self._find_variable(ds, ['spechum', 'huss', 'specific_humidity', 'q'])
        airpres_var = self._find_variable(ds, ['airpres', 'ps', 'sp', 'air_pressure', 'pressure'])

        # Extract data
        time_coord = ds['time'].values
        dates = pd.to_datetime(time_coord)

        # Process each variable
        precip = self._process_precipitation(ds, precip_var, dates)
        temp, tmax, tmin = self._process_temperature(ds, temp_var, tmax_var, tmin_var, dates)

        # Process optional variables
        wind = self._process_wind(ds, wind_var, dates)
        rh = self._process_relative_humidity(ds, spechum_var, airpres_var, temp_var, dates)
        kdown = self._process_radiation(ds, swrad_var, dates, 'shortwave')
        ldown = self._process_radiation(ds, lwrad_var, dates, 'longwave')

        # Aggregate to daily
        df = pd.DataFrame({
            'precip': precip,
            'temp': temp,
            'tmax': tmax,
            'tmin': tmin,
            'wind': wind,
            'rh': rh,
            'kdown': kdown,
            'ldown': ldown
        }, index=dates)

        daily_df = self._aggregate_to_daily(df)

        # Write files
        base_name = f"{self.domain_name}_base"
        self._write_base_station_file(base_name, 1, daily_df.index[0], catchment_path)

        self._write_climate_file(f"{base_name}.rain", daily_df.index, daily_df['precip'].values)
        self._write_climate_file(f"{base_name}.tmax", daily_df.index, daily_df['tmax'].values)
        self._write_climate_file(f"{base_name}.tmin", daily_df.index, daily_df['tmin'].values)
        self._write_climate_file(f"{base_name}.tavg", daily_df.index, daily_df['temp'].values)

        # Write optional climate files if data available
        if wind is not None and not np.all(np.isnan(daily_df['wind'])):
            self._write_climate_file(f"{base_name}.wind", daily_df.index, daily_df['wind'].values)
        if rh is not None and not np.all(np.isnan(daily_df['rh'])):
            self._write_climate_file(f"{base_name}.relative_humidity", daily_df.index, daily_df['rh'].values)
        if kdown is not None and not np.all(np.isnan(daily_df['kdown'])):
            self._write_climate_file(f"{base_name}.Kdown_direct", daily_df.index, daily_df['kdown'].values)
        if ldown is not None and not np.all(np.isnan(daily_df['ldown'])):
            self._write_climate_file(f"{base_name}.Ldown", daily_df.index, daily_df['ldown'].values)

        ds.close()
        self.logger.info(f"Climate files written to {self.climate_dir}")
        return True

    def _load_forcing_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> xr.Dataset:
        """Load basin-averaged forcing data from available sources."""
        # Search multiple potential locations
        search_paths = [
            self.forcing_basin_path,
            self.project_dir / 'forcing' / 'merged_path',
            self.project_dir / 'forcing' / 'SUMMA_input',
            self.forcing_raw_path,
        ]

        forcing_files = []
        for path in search_paths:
            if path.exists():
                files = list(path.glob("*.nc"))
                if files:
                    self.logger.info(f"Found {len(files)} forcing files in {path}")
                    forcing_files = files
                    break

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in any of: {search_paths}")

        self.logger.info(f"Loading forcing from {len(forcing_files)} files")

        try:
            ds = xr.open_mfdataset(forcing_files, combine='by_coords')
        except ValueError as e:
            self.logger.warning(f"Failed with combine='by_coords': {e}. Retrying...")
            try:
                ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time')
            except (FileNotFoundError, OSError, ValueError, KeyError):
                self.logger.warning("Failed to concat. Attempting merge...")
                datasets = [xr.open_dataset(f) for f in forcing_files]
                ds = xr.merge(datasets)

        # Subset to simulation period
        ds = ds.sel(time=slice(start_date, end_date))

        return ds

    def _find_variable(self, ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
        """Find first matching variable name in dataset."""
        for var in candidates:
            if var in ds.data_vars:
                return var
        return None

    def _basin_average(self, data_array) -> np.ndarray:
        """Average across all spatial dimensions (HRU, GRU, etc.)"""
        values = data_array.values
        if values.ndim > 1:
            spatial_axes = tuple(range(1, values.ndim))
            values = np.nanmean(values, axis=spatial_axes)
        return values.flatten()

    def _process_precipitation(
        self,
        ds: xr.Dataset,
        precip_var: Optional[str],
        dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Process precipitation variable.

        Converts all source units to meters per timestep (hour) so that
        daily aggregation (sum) results in meters per day, which is what
        RHESSys expects in climate station files.
        """
        if precip_var:
            precip = self._basin_average(ds[precip_var])
            units = str(ds[precip_var].attrs.get('units', '')).lower()

            # If it's a rate (e.g. kg/m2/s, mm/s, m/s)
            if 's-1' in units or 's^-1' in units or '/s' in units:
                # If units are kg/m2/s or mm/s, they are effectively mm/s
                # We also treat 'm s-1' as 'mm s-1' because meteorological
                # precipitation rates are almost always in mm/s or kg/m2/s in NetCDF.
                # True m/s would be 1000x larger than typical rain.
                if 'kg' in units or 'mm' in units or 'm s-1' in units or 'm/s' in units:
                    # Convert mm/s to m/hour (3600 s/hr / 1000 mm/m)
                    precip = precip * 3.6
                else:
                    # Assume other rates are also mm/s for safety
                    precip = precip * 3.6
            # If it's already a depth per timestep (e.g. ERA5 'm')
            elif 'm' in units:
                if 'mm' in units:
                    # Convert mm to m
                    precip = precip / 1000.0
                else:
                    # Assume m (already correct for hourly accumulation)
                    pass

            return precip
        else:
            self.logger.warning("No precipitation variable found, using zeros")
            return np.zeros(len(dates))

    def _process_temperature(
        self,
        ds: xr.Dataset,
        temp_var: Optional[str],
        tmax_var: Optional[str],
        tmin_var: Optional[str],
        dates: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process temperature variables (mean, max, min)."""
        # Get temperature
        if temp_var:
            temp = self._basin_average(ds[temp_var])
            # Convert K to C if needed
            if np.nanmean(temp) > 100:
                temp = temp - 273.15
        else:
            self.logger.warning("No temperature variable found")
            temp = np.full(len(dates), 15.0)

        # Get tmax
        if tmax_var:
            tmax = self._basin_average(ds[tmax_var])
            if np.nanmean(tmax) > 100:
                tmax = tmax - 273.15
        else:
            tmax = temp + 5

        # Get tmin
        if tmin_var:
            tmin = self._basin_average(ds[tmin_var])
            if np.nanmean(tmin) > 100:
                tmin = tmin - 273.15
        else:
            tmin = temp - 5

        return temp, tmax, tmin

    def _process_wind(
        self,
        ds: xr.Dataset,
        wind_var: Optional[str],
        dates: pd.DatetimeIndex
    ) -> Optional[np.ndarray]:
        """Process wind speed variable."""
        if wind_var:
            return self._basin_average(ds[wind_var])
        return None

    def _process_relative_humidity(
        self,
        ds: xr.Dataset,
        spechum_var: Optional[str],
        airpres_var: Optional[str],
        temp_var: Optional[str],
        dates: pd.DatetimeIndex
    ) -> Optional[np.ndarray]:
        """Calculate relative humidity from specific humidity if available."""
        if spechum_var and airpres_var and temp_var:
            try:
                q = self._basin_average(ds[spechum_var])
                p = self._basin_average(ds[airpres_var])
                t = self._basin_average(ds[temp_var])

                # Convert to Celsius if needed
                if np.nanmean(t) > 100:
                    t = t - 273.15

                # Calculate saturation vapor pressure (Tetens formula)
                es = 6.112 * np.exp(17.67 * t / (t + 243.5)) * 100  # Pa

                # Calculate actual vapor pressure from specific humidity
                # q = 0.622 * e / (p - 0.378 * e)
                e = q * p / (0.622 + 0.378 * q)

                # Relative humidity
                rh = 100 * e / es
                rh = np.clip(rh, 0, 100)

                return rh
            except Exception as e:
                self.logger.warning(f"Could not calculate relative humidity: {e}")
        return None

    def _process_radiation(
        self,
        ds: xr.Dataset,
        rad_var: Optional[str],
        dates: pd.DatetimeIndex,
        rad_type: str
    ) -> Optional[np.ndarray]:
        """Process radiation variable."""
        if rad_var:
            rad = self._basin_average(ds[rad_var])
            # Convert J/m2 to W/m2 if needed (ERA5 often in J/m2)
            if np.nanmean(rad) > 10000:
                timestep_hours = 1  # Assume hourly data
                rad = rad / (3600 * timestep_hours)
            return rad
        return None

    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sub-daily data to daily values."""
        daily = df.resample('D').agg({
            'precip': 'sum',  # Sum precipitation
            'temp': 'mean',   # Mean temperature
            'tmax': 'max',    # Max of max temps
            'tmin': 'min',    # Min of min temps
            'wind': 'mean',   # Mean wind
            'rh': 'mean',     # Mean relative humidity
            'kdown': 'mean',  # Mean radiation
            'ldown': 'mean',
        })
        return daily

    def _write_base_station_file(
        self,
        base_name: str,
        station_id: int,
        start_date: pd.Timestamp,
        catchment_path: Optional[Path] = None
    ) -> None:
        """Write RHESSys base station file."""
        base_file = self.climate_dir / f"{base_name}"

        # Get centroid coordinates from basin shapefile
        lon, lat, elev = -115.0, 51.0, 1500.0
        if catchment_path and catchment_path.exists():
            try:
                gdf = gpd.read_file(catchment_path)
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y
                elev = float(gdf.get('elev_mean', [1000])[0]) if 'elev_mean' in gdf.columns else 1000.0
            except (FileNotFoundError, KeyError, IndexError, ValueError):
                pass

        # Full path to climate file prefix
        climate_prefix = self.climate_dir / base_name

        # Build list of non-critical daily sequences
        non_critical_sequences = []
        for suffix in ['wind', 'relative_humidity', 'Kdown_direct', 'Ldown', 'tavg']:
            if (self.climate_dir / f"{base_name}.{suffix}").exists():
                non_critical_sequences.append(suffix)

        num_sequences = len(non_critical_sequences)
        sequence_lines = "\n".join(non_critical_sequences) if non_critical_sequences else ""

        content = f"""{station_id}\tbase_station_id
{lon:.4f}\tx_coordinate
{lat:.4f}\ty_coordinate
{elev:.1f}\tz_coordinate
3.5\teffective_lai
2.0\tscreen_height
none\tannual_climate_prefix
0\tnumber_non_critical_annual_sequences
none\tmonthly_climate_prefix
0\tnumber_non_critical_monthly_sequences
{climate_prefix}\tdaily_climate_prefix
{num_sequences}\tnumber_non_critical_daily_sequences
{sequence_lines}
none\thourly_climate_prefix
0\tnumber_non_critical_hourly_sequences
"""
        base_file.write_text(content)
        self.logger.info(f"Base station file written: {base_file}")

    def _write_climate_file(
        self,
        filename: str,
        dates: pd.DatetimeIndex,
        values: np.ndarray
    ) -> None:
        """
        Write a single RHESSys climate file.

        Format:
        - Line 1: start date (year month day hour)
        - Lines 2+: one value per line
        """
        filepath = self.climate_dir / filename

        with open(filepath, 'w') as f:
            start_date = dates[0]
            f.write(f"{start_date.year} {start_date.month} {start_date.day} 1\n")

            for value in values:
                if np.isnan(value):
                    f.write("0.0000\n")
                else:
                    f.write(f"{value:.4f}\n")

        self.logger.debug(f"Climate file written: {filepath}")

    def _create_synthetic_climate(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Create synthetic climate data for testing."""
        self.climate_dir.mkdir(parents=True, exist_ok=True)

        dates = pd.date_range(start_date, end_date, freq='D')

        # Simple synthetic data
        precip = np.random.exponential(2, len(dates))
        temp = 10 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates))
        tmax = temp + 5 + np.random.normal(0, 1, len(dates))
        tmin = temp - 5 + np.random.normal(0, 1, len(dates))

        base_name = f"{self.domain_name}_base"
        self._write_base_station_file(base_name, 1, dates[0])
        self._write_climate_file(f"{base_name}.rain", dates, precip)
        self._write_climate_file(f"{base_name}.tmax", dates, tmax)
        self._write_climate_file(f"{base_name}.tmin", dates, tmin)
        self._write_climate_file(f"{base_name}.tavg", dates, temp)

        self.logger.info("Synthetic climate files created")
