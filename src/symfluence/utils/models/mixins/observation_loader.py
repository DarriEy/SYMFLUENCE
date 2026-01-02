"""
Observation Loading Mixin for streamflow data.

Provides standardized loading, parsing, and unit conversion for streamflow
observations across preprocessing and calibration workflows. Eliminates code
duplication and provides a consistent interface.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal, Tuple
import pandas as pd
import xarray as xr
import numpy as np
import logging

from symfluence.utils.common.constants import UnitConversion
from symfluence.utils.exceptions import DataAcquisitionError

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

            self.logger.info(f"Loading streamflow observations from: {obs_file}")

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

        # Strategy 1: Explicit path in config
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path and obs_path != 'default':
            candidates.append(Path(obs_path))

        # Strategy 2: Nested config
        obs_nested = self.config.get('observations', {}).get('streamflow', {}).get('path')
        if obs_nested:
            candidates.append(Path(obs_nested))

        # Strategy 3-5: Standard locations
        obs_dir = self.project_dir / 'observations' / 'streamflow'
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
        """Read CSV file with flexible parsing."""
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
            series = series * 0.028316846592  # ft³/s -> m³/s
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
                conversion_factor = timestep_seconds / 1000  # seconds to mm conversion
                return series * conversion_factor / catchment_area_km2

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
        2. Catchment shapefile (geometry area)
        """
        if not HAS_GEOPANDAS:
            self.logger.warning("geopandas not available, cannot determine catchment area")
            return None

        try:
            # Try river basins
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default' or basin_name is None:
                basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"

            basin_path = self.project_dir / 'shapefiles' / 'river_basins' / basin_name

            if basin_path.exists():
                basin_gdf = gpd.read_file(basin_path)
                if 'GRU_area' in basin_gdf.columns:
                    area_km2 = basin_gdf['GRU_area'].sum() / 1e6  # m2 to km2
                    self.logger.info(f"Using catchment area from river basins: {area_km2:.2f} km2")
                    return area_km2

            # Fallback to catchment shapefile
            catchment_path = self._get_catchment_path()
            if catchment_path and catchment_path.exists():
                catchment = gpd.read_file(catchment_path)
                catchment_proj = catchment.to_crs(catchment.estimate_utm_crs())
                area_km2 = catchment_proj.geometry.area.sum() / 1e6
                self.logger.warning(f"Using estimated catchment area: {area_km2:.2f} km2")
                return area_km2

        except Exception as e:
            self.logger.error(f"Could not determine catchment area: {e}")

        return None

    def _get_catchment_path(self) -> Optional[Path]:
        """Get catchment shapefile path (subclass can override)."""
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if catchment_name == 'default' or catchment_name is None:
            discretization = self.config.get('DOMAIN_DISCRETIZATION')
            catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"

        catchment_path = self.project_dir / 'shapefiles' / 'catchment' / catchment_name
        return catchment_path if catchment_path.exists() else None

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
                time_values = ((series.index - pd.Timestamp('1970-01-01')).total_seconds() / 3600).values
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
