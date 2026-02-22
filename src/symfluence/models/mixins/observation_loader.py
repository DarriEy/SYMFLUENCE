"""
Observation Loading Mixin for streamflow data.

Provides standardized loading, parsing, and unit conversion for streamflow
observations across preprocessing and calibration workflows. Eliminates code
duplication and provides a consistent interface.
"""

from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConversion
from symfluence.geospatial.geometry_utils import calculate_catchment_area_km2
from symfluence.core.exceptions import DataAcquisitionError

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


class ObservationLoaderMixin:
    """
    Mixin providing streamflow observation loading functionality.

    Handles:
    - Multiple path resolution strategies with fallbacks
    - Flexible column detection (datetime, discharge)
    - Robust datetime parsing with multiple strategies
    - Unit conversions (cms ↔ mm/timestep, cfs → cms)
    - Multiple output formats (xarray, pandas Series/DataFrame)
    - Graceful error handling and logging

    Requires the following attributes in the class:
        - self.config: Dict[str, Any]
        - self.project_dir: Path
        - self.domain_name: str
        - self.logger: logging.Logger

    Optional (for unit conversion):
        - self.forcing_time_step_size: int (seconds)
    """

    def load_streamflow_observations(
        self,
        output_format: Literal['xarray', 'series', 'dataframe'] = 'xarray',
        target_units: Literal['cms', 'mm_per_timestep', 'mm_per_day', 'mm_per_hour'] = 'cms',
        resample_freq: Optional[str] = None,
        time_slice: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        catchment_area_km2: Optional[float] = None,
        return_none_on_error: bool = True
    ) -> Union[xr.Dataset, pd.Series, pd.DataFrame, None]:
        """
        Load streamflow observations with flexible output and units.

        Args:
            output_format: 'xarray', 'series', or 'dataframe'
            target_units: Desired output units
            resample_freq: Pandas resample frequency (e.g., 'D', 'h', '3h')
            time_slice: Tuple of (start, end) timestamps for filtering
            catchment_area_km2: Catchment area for unit conversion (auto-detected if None)
            return_none_on_error: If True, return None on error; if False, raise

        Returns:
            Streamflow observations in requested format and units, or None if error

        Raises:
            FileNotFoundError: If observation file not found and return_none_on_error=False
            ValueError: If required data cannot be parsed and return_none_on_error=False
        """
        try:
            # Find observation file
            obs_file = self._find_observation_file()
            if obs_file is None:
                if return_none_on_error:
                    self.logger.warning("Streamflow observations not found, returning None")
                    return None
                raise FileNotFoundError("Streamflow observation file not found")

            self.logger.debug(f"Loading streamflow observations from: {obs_file}")

            # Read and parse file
            df = self._read_observation_file(obs_file)

            # Extract time series
            series = self._extract_streamflow_series(df)

            # Resample if requested
            if resample_freq:
                series = series.resample(resample_freq).mean()

            # Apply time slice
            if time_slice:
                start, end = time_slice
                series = series.loc[slice(start, end)]

            # Unit conversion
            series = self._convert_units(
                series,
                source_units='cms',  # Assume cms after initial parsing
                target_units=target_units,
                catchment_area_km2=catchment_area_km2
            )

            # Format output
            return self._format_output(series, output_format, target_units)

        except Exception as e:
            if return_none_on_error:
                self.logger.error(f"Error loading observations: {e}")
                return None
            raise

    def _find_observation_file(self) -> Optional[Path]:
        """
        Find observation file using multiple strategies.

        Priority:
        1. config['OBSERVATIONS_PATH'] (explicit path)
        2. config['observations']['streamflow']['path'] (nested config)
        3. project_dir / 'observations/streamflow/preprocessed/*_streamflow_processed.csv'
        4. project_dir / 'observations/streamflow/preprocessed/*.csv'
        5. project_dir / 'observations/streamflow/*_streamflow_obs.csv'
        """
        candidates = []

        # Strategy 0: Model-ready observations store (grouped NetCDF)
        model_ready_obs = (
            self.project_dir / 'data' / 'model_ready' / 'observations'
            / f'{self.domain_name}_observations.nc'
        )
        if model_ready_obs.exists():
            try:
                # Data lives in the 'streamflow' group, not the root
                with xr.open_dataset(model_ready_obs, group='streamflow') as _ds:
                    if _ds.data_vars:
                        candidates.append(model_ready_obs)
            except Exception:
                self.logger.debug(f"Model-ready observations file has no streamflow group: {model_ready_obs}")

        # Strategy 1: Explicit path in config
        obs_path = self.config_dict.get('OBSERVATIONS_PATH')
        if obs_path and obs_path != 'default':
            candidates.append(Path(obs_path))

        # Strategy 2: Nested config
        obs_nested = self.config_dict.get('observations', {}).get('streamflow', {}).get('path')
        if obs_nested:
            candidates.append(Path(obs_nested))

        # Strategy 3-5: Standard locations
        obs_dir = self.project_observations_dir / 'streamflow'
        preprocessed_dir = obs_dir / 'preprocessed'

        if preprocessed_dir.exists():
            # Prefer processed files
            candidates.extend(preprocessed_dir.glob(f"{self.domain_name}_streamflow_processed.csv"))
            candidates.extend(preprocessed_dir.glob("*_streamflow_processed.csv"))
            candidates.extend(preprocessed_dir.glob("*.csv"))

        if obs_dir.exists():
            candidates.extend(obs_dir.glob(f"{self.domain_name}_streamflow_obs.csv"))
            candidates.extend(obs_dir.glob("*_streamflow_obs.csv"))

        # Return first existing file
        for candidate in candidates:
            if candidate.exists():
                return candidate

        self.logger.debug(f"Observation file search tried: {[str(p) for p in candidates[:5]]}")
        return None

    def _read_observation_file(self, file_path: Path) -> pd.DataFrame:
        """Read observation file with format detection (CSV or NetCDF).

        For grouped NetCDF files (model_ready store), reads the
        'streamflow' group and reshapes to a flat DataFrame.
        """
        if file_path.suffix.lower() in ('.nc', '.nc4', '.netcdf'):
            # Try streamflow group first (model_ready observations store)
            for group in ('streamflow', None):
                try:
                    with xr.open_dataset(file_path, group=group) as ds:
                        if ds.data_vars:
                            return ds.to_dataframe().reset_index()
                except Exception:
                    continue
            raise ValueError(f"No readable data in NetCDF file: {file_path}")
        return pd.read_csv(file_path)

    def _extract_streamflow_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract streamflow time series from DataFrame with flexible column detection.

        Handles:
        - Multiple datetime column names
        - Multiple discharge column names
        - Unit detection (cms, cfs, mm/day, etc.)
        - Automatic conversion to cms
        """
        # Find datetime column
        datetime_col = self._find_datetime_column(df)
        if datetime_col is None:
            raise DataAcquisitionError(f"Could not identify datetime column. Columns: {list(df.columns)}")

        # Find discharge column
        discharge_col, source_units = self._find_discharge_column(df)
        if discharge_col is None:
            raise DataAcquisitionError(f"Could not identify discharge column. Columns: {list(df.columns)}")

        # Parse datetime and drop timezone to avoid tz-aware/naive comparisons
        # Note: dayfirst=False (default) is correct for ISO-format dates (YYYY-MM-DD)
        # Using dayfirst=True silently misparses dates where day > 12, causing massive data loss
        times = pd.to_datetime(df[datetime_col], utc=True, errors='coerce').dt.tz_convert(None)

        # Create series
        series = pd.Series(
            df[discharge_col].astype(float).values,
            index=times,
            name='streamflow'
        ).sort_index()

        # Remove duplicates and NaN
        series = series[~series.index.duplicated(keep='first')].dropna()

        # Convert to cms if needed
        if source_units == 'cfs':
            series = series * UnitConversion.CFS_TO_CMS  # ft³/s -> m³/s
            self.logger.info("Converted streamflow from cfs to cms")

        return series

    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find datetime column with flexible matching."""
        # Common datetime column names (priority order)
        datetime_candidates = [
            'datetime', 'time', 'date_time', 'DateTime',
            'date', 'timestamp', 'Date', 'Time'
        ]

        # Exact match
        for col in datetime_candidates:
            if col in df.columns:
                return col

        # Case-insensitive fuzzy match
        import re
        for col in df.columns:
            if re.search(r'date|time', col, re.IGNORECASE):
                return col

        return None

    def _find_discharge_column(self, df: pd.DataFrame) -> Tuple[Optional[str], str]:
        """
        Find discharge column and detect units.

        Returns:
            Tuple of (column_name, units) where units is 'cms', 'cfs', or 'unknown'
        """
        # CMS variants
        cms_candidates = [
            'discharge_cms', 'flow_cms', 'Q_cms', 'cms',
            'streamflow_cms', 'discharge', 'flow', 'Q'
        ]

        # CFS variants
        cfs_candidates = [
            'discharge_cfs', 'flow_cfs', 'Q_cfs', 'cfs',
            'streamflow_cfs'
        ]

        # Check CMS
        for col in cms_candidates:
            if col in df.columns:
                return col, 'cms'

        # Check CFS
        for col in cfs_candidates:
            if col in df.columns:
                return col, 'cfs'

        # Fallback: first numeric column (assume cms)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.logger.warning(
                    f"Could not identify discharge units, assuming cms for column: {col}"
                )
                return col, 'cms'

        return None, 'unknown'

    def _convert_units(
        self,
        series: pd.Series,
        source_units: str,
        target_units: str,
        catchment_area_km2: Optional[float] = None
    ) -> pd.Series:
        """
        Convert streamflow units.

        Supported conversions:
        - cms → mm/timestep (requires area and timestep)
        - cms → mm/day (requires area)
        - cms → mm/hour (requires area)
        - mm/timestep → cms (requires area and timestep)
        - mm/day → cms (requires area)
        - No conversion if source == target
        """
        if source_units == target_units:
            return series

        # Get catchment area if needed
        if 'mm' in target_units or 'mm' in source_units:
            if catchment_area_km2 is None:
                catchment_area_km2 = self._get_catchment_area()
            if catchment_area_km2 is None:
                raise DataAcquisitionError("Catchment area required for mm ↔ cms conversion")

        # Conversions
        if source_units == 'cms':
            if target_units == 'mm_per_day':
                # Q(mm/day) = Q(cms) * 86.4 / Area(km2)
                return series * UnitConversion.MM_DAY_TO_CMS / catchment_area_km2

            elif target_units == 'mm_per_hour':
                # Q(mm/hour) = Q(cms) * 3.6 / Area(km2)
                return series * UnitConversion.MM_HOUR_TO_CMS / catchment_area_km2

            elif target_units == 'mm_per_timestep':
                # Get timestep from config
                timestep_seconds = getattr(self, 'forcing_time_step_size', 86400)
                conversion_factor = UnitConversion.mm_per_timestep_to_cms_factor(timestep_seconds)
                return series * conversion_factor / catchment_area_km2

            else:
                raise DataAcquisitionError(
                    f"Unsupported unit conversion: {source_units} → {target_units}"
                )

        elif source_units == 'mm_per_day' and target_units == 'cms':
            return series * catchment_area_km2 / UnitConversion.MM_DAY_TO_CMS

        elif source_units == 'mm_per_hour' and target_units == 'cms':
            return series * catchment_area_km2 / UnitConversion.MM_HOUR_TO_CMS

        else:
            raise DataAcquisitionError(
                f"Unsupported unit conversion: {source_units} → {target_units}"
            )

    def _get_catchment_area(self) -> Optional[float]:
        """
        Estimate catchment area from shapefiles.

        Priority:
        1. River basins shapefile (GRU_area field)
        2. Catchment shapefile (geometry area via calculate_catchment_area_km2)
        """
        if not HAS_GEOPANDAS:
            self.logger.warning("geopandas not available, cannot determine catchment area")
            return None

        try:
            # Try river basins first (pre-calculated area field)
            basin_name = self.config_dict.get('RIVER_BASINS_NAME')
            if basin_name == 'default' or basin_name is None:
                basin_name = f"{self.domain_name}_riverBasins_{self.config_dict.get('DOMAIN_DEFINITION_METHOD')}.shp"

            basin_path = self.project_dir / 'shapefiles' / 'river_basins' / basin_name

            if basin_path.exists():
                basin_gdf = gpd.read_file(basin_path)
                if 'GRU_area' in basin_gdf.columns:
                    area_km2 = basin_gdf['GRU_area'].sum() / 1e6  # m2 to km2
                    self.logger.debug(f"Using catchment area from river basins: {area_km2:.2f} km2")
                    return area_km2

            # Fallback to catchment shapefile using shared geospatial utility
            catchment_path = self._get_catchment_path()
            if catchment_path and catchment_path.exists():
                catchment_gdf = gpd.read_file(catchment_path)
                area_km2 = calculate_catchment_area_km2(catchment_gdf, logger=self.logger)
                self.logger.debug(f"Using estimated catchment area: {area_km2:.2f} km2")
                return area_km2

        except Exception as e:
            self.logger.error(f"Could not determine catchment area: {e}")

        return None

    def _get_catchment_path(self) -> Optional[Path]:
        """Get catchment shapefile path with backward compatibility (subclass can override)."""
        # Use backward-compatible path resolution if available
        if hasattr(self, '_get_catchment_file_path'):
            catchment_path = self._get_catchment_file_path()
            return catchment_path if catchment_path.exists() else None

        # Fallback for classes without PathResolverMixin
        catchment_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if catchment_name == 'default' or catchment_name is None:
            discretization = self.config_dict.get('SUB_GRID_DISCRETIZATION')
            catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"

        # Check old path first
        old_path = self.project_dir / 'shapefiles' / 'catchment' / catchment_name
        if old_path.exists():
            return old_path

        # Check new organized path
        definition_method = self.config_dict.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        experiment_id = self.config_dict.get('EXPERIMENT_ID', 'run_1')
        new_path = self.project_dir / 'shapefiles' / 'catchment' / definition_method / experiment_id / catchment_name
        return new_path if new_path.exists() else None

    def _format_output(
        self,
        series: pd.Series,
        output_format: str,
        units: str
    ) -> Union[xr.Dataset, pd.Series, pd.DataFrame]:
        """Format output based on requested format."""
        if output_format == 'series':
            return series

        elif output_format == 'dataframe':
            df = series.reset_index()
            df.columns = ['datetime', 'streamflow_cms' if units == 'cms' else 'streamflow']
            return df

        elif output_format == 'xarray':
            # Determine time units based on target units
            if 'hour' in units:
                time_values = ((series.index - pd.Timestamp('1970-01-01')).total_seconds() / UnitConversion.SECONDS_PER_HOUR).values
                time_units = 'hours since 1970-01-01'
            else:
                time_values = (series.index - pd.Timestamp('1970-01-01')).days.values.astype(float)
                time_units = 'days since 1970-01-01'

            # Create dataset
            ds = xr.Dataset(
                {
                    'q_obs': xr.DataArray(
                        series.values,
                        dims=['time'],
                        coords={'time': time_values},
                        attrs={
                            'units': units.replace('_', ' '),
                            'long_name': 'Observed streamflow',
                            'standard_name': 'water_volume_transport_in_river_channel'
                        }
                    )
                },
                coords={
                    'time': xr.DataArray(
                        time_values,
                        dims=['time'],
                        attrs={'units': time_units, 'long_name': 'time'}
                    )
                }
            )
            return ds

        else:
            raise DataAcquisitionError(f"Unknown output format: {output_format}")
