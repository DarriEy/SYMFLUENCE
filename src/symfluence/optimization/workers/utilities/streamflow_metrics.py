"""
Streamflow Metrics Utility

Shared utility for streamflow metric calculation across all optimization workers.
Consolidates duplicate implementations from GR, FUSE, SUMMA, MESH, HYPE, and NGEN workers.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import pandas as pd

from symfluence.core.constants import UnitConversion
from symfluence.evaluation.metrics import kge, nse, rmse, mae, kge_prime

logger = logging.getLogger(__name__)


class StreamflowMetrics:
    """
    Shared utility for streamflow metric calculation across all workers.

    Provides standardized methods for:
    - Loading observation data
    - Getting catchment area from various sources (shapefile, NetCDF, text)
    - Converting runoff units to discharge
    - Aligning time series
    - Calculating performance metrics
    """

    def __init__(self, penalty_score: float = -9999.0):
        """
        Initialize StreamflowMetrics.

        Args:
            penalty_score: Value to return when metric calculation fails
        """
        self.penalty_score = penalty_score

    def load_observations(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        resample_freq: Optional[str] = 'D',
        discharge_col: str = 'discharge_cms'
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
        """
        Load observation data from standard location.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            domain_name: Domain name for path construction
            resample_freq: Resampling frequency (e.g., 'D' for daily, None for no resampling)
            discharge_col: Name of discharge column in CSV

        Returns:
            Tuple of (values, datetime_index), or (None, None) on error
        """
        try:
            obs_file = config.get('OBSERVATIONS_PATH', 'default')
            if obs_file == 'default' or not obs_file:
                obs_file = project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv"
            else:
                obs_file = Path(obs_file)

            if not obs_file.exists():
                logger.error(f"Observations file not found: {obs_file}")
                return None, None

            # Read CSV with datetime index
            df_obs = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # Get discharge column
            if discharge_col not in df_obs.columns:
                # Try to find a suitable column
                discharge_cols = [c for c in df_obs.columns if 'discharge' in c.lower() or 'flow' in c.lower()]
                if discharge_cols:
                    discharge_col = discharge_cols[0]
                    logger.debug(f"Using discharge column: {discharge_col}")
                else:
                    logger.error(f"No discharge column found in {obs_file}")
                    return None, None

            observed = df_obs[discharge_col]

            # Resample if requested
            if resample_freq:
                observed = observed.resample(resample_freq).mean()

            return observed.values, observed.index

        except Exception as e:
            logger.error(f"Error loading observations: {e}")
            return None, None

    def get_catchment_area(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        source: str = 'shapefile',
        settings_dir: Optional[Path] = None,
        default_area: float = 1000.0
    ) -> float:
        """
        Get catchment area in km2 from various sources.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            domain_name: Domain name
            source: Source type: 'shapefile', 'netcdf', 'geodata'
            settings_dir: Settings directory (required for 'netcdf' and 'geodata' sources)
            default_area: Default area to return on error (km2)

        Returns:
            Catchment area in km2
        """
        try:
            if source == 'shapefile':
                return self._get_area_from_shapefile(config, project_dir, domain_name, default_area)
            elif source == 'netcdf':
                return self._get_area_from_netcdf(settings_dir, domain_name, default_area)
            elif source == 'geodata':
                return self._get_area_from_geodata(settings_dir, default_area)
            else:
                logger.warning(f"Unknown area source: {source}. Using default {default_area} km2")
                return default_area
        except Exception as e:
            logger.warning(f"Error getting catchment area from {source}: {e}. Using default {default_area} km2")
            return default_area

    def _get_area_from_shapefile(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        default_area: float
    ) -> float:
        """Get catchment area from shapefile/geopackage."""
        import geopandas as gpd

        # Resolve catchment path
        catchment_path = config.get('CATCHMENT_PATH', 'default')
        if catchment_path == 'default' or not catchment_path:
            catchment_path = project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)

        # Resolve catchment filename
        discretization = config.get('DOMAIN_DISCRETIZATION', 'elevation')
        catchment_name = config.get('CATCHMENT_SHP_NAME', 'default')
        if catchment_name == 'default' or not catchment_name:
            catchment_name = f"{domain_name}_HRUs_{discretization}.shp"

        catchment_file = catchment_path / catchment_name

        if not catchment_file.exists():
            logger.warning(f"Catchment file not found: {catchment_file}. Using default {default_area} km2")
            return default_area

        gdf = gpd.read_file(catchment_file)

        # Priority 1: Use GRU_area column if available (already in m2)
        if 'GRU_area' in gdf.columns:
            area_km2 = gdf['GRU_area'].sum() / 1e6
            logger.debug(f"Catchment area from GRU_area: {area_km2:.2f} km2")
            return area_km2

        # Priority 2: Calculate from geometry
        if gdf.crs and not gdf.crs.is_geographic:
            area_m2 = gdf.geometry.area.sum()
        else:
            gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
            area_m2 = gdf_utm.geometry.area.sum()

        area_km2 = area_m2 / 1e6
        logger.debug(f"Catchment area from geometry: {area_km2:.2f} km2")
        return area_km2

    def _get_area_from_netcdf(
        self,
        settings_dir: Optional[Path],
        domain_name: str,
        default_area: float
    ) -> float:
        """Get catchment area from SUMMA attributes.nc."""
        import xarray as xr

        if settings_dir is None:
            logger.warning("settings_dir required for netcdf area source")
            return default_area

        attrs_file = settings_dir / 'attributes.nc'
        if not attrs_file.exists():
            # Try SUMMA subdirectory
            attrs_file = settings_dir / 'SUMMA' / 'attributes.nc'

        if not attrs_file.exists():
            logger.warning(f"Attributes file not found: {attrs_file}")
            return default_area

        with xr.open_dataset(attrs_file) as attrs:
            if 'HRUarea' in attrs.data_vars:
                area_m2 = float(attrs['HRUarea'].values.sum())
                area_km2 = area_m2 / 1e6
                logger.debug(f"Catchment area from HRUarea: {area_km2:.2f} km2")
                return area_km2

        logger.warning("HRUarea variable not found in attributes.nc")
        return default_area

    def _get_area_from_geodata(
        self,
        settings_dir: Optional[Path],
        default_area: float
    ) -> float:
        """Get catchment area from HYPE GeoData.txt."""
        if settings_dir is None:
            logger.warning("settings_dir required for geodata area source")
            return default_area

        geodata_file = settings_dir / 'GeoData.txt'
        if not geodata_file.exists():
            logger.warning(f"GeoData.txt not found: {geodata_file}")
            return default_area

        geodata = pd.read_csv(geodata_file, sep='\t')
        if 'area' in geodata.columns:
            # HYPE area is in m2
            area_km2 = geodata['area'].sum() / 1e6
            logger.debug(f"Catchment area from GeoData.txt: {area_km2:.2f} km2")
            return area_km2

        logger.warning("'area' column not found in GeoData.txt")
        return default_area

    def convert_runoff_to_discharge(
        self,
        runoff: Union[np.ndarray, pd.Series],
        area_km2: float,
        input_units: str = 'mm/day'
    ) -> Union[np.ndarray, pd.Series]:
        """
        Convert runoff to discharge (m3/s).

        Args:
            runoff: Runoff values
            area_km2: Catchment area in km2
            input_units: Input units: 'mm/day', 'm/s', 'm3/s'

        Returns:
            Discharge in m3/s (same type as input)
        """
        if input_units == 'mm/day':
            # Q(m3/s) = Q(mm/day) * Area(km2) / 86.4
            return runoff * area_km2 / UnitConversion.MM_DAY_TO_CMS
        elif input_units == 'm/s':
            # Q(m3/s) = Q(m/s) * Area(m2) = Q(m/s) * Area(km2) * 1e6
            return runoff * area_km2 * 1e6
        elif input_units == 'm3/s':
            # Already in correct units
            return runoff
        else:
            logger.warning(f"Unknown input units: {input_units}. Assuming mm/day")
            return runoff * area_km2 / UnitConversion.MM_DAY_TO_CMS

    def align_timeseries(
        self,
        sim: pd.Series,
        obs: pd.Series,
        calibration_period: Optional[Tuple[str, str]] = None,
        min_overlap: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align simulation and observation time series.

        Args:
            sim: Simulation series with datetime index
            obs: Observation series with datetime index
            calibration_period: Optional (start_date, end_date) tuple for filtering
            min_overlap: Minimum number of overlapping points required

        Returns:
            Tuple of (obs_aligned, sim_aligned) as numpy arrays

        Raises:
            ValueError: If insufficient overlapping data
        """
        # Find common index
        common_idx = sim.index.intersection(obs.index)

        if len(common_idx) < min_overlap:
            raise ValueError(f"Insufficient overlapping data: {len(common_idx)} points (min: {min_overlap})")

        # Align to common index
        sim_aligned = sim.loc[common_idx]
        obs_aligned = obs.loc[common_idx]

        # Filter to calibration period if specified
        if calibration_period:
            start_date, end_date = calibration_period
            mask = (sim_aligned.index >= start_date) & (sim_aligned.index <= end_date)
            sim_aligned = sim_aligned[mask]
            obs_aligned = obs_aligned[mask]

            if len(sim_aligned) < min_overlap:
                raise ValueError(f"Insufficient data in calibration period: {len(sim_aligned)} points")

        # Remove NaN pairs
        valid_mask = ~(np.isnan(obs_aligned.values) | np.isnan(sim_aligned.values))
        obs_clean = obs_aligned.values[valid_mask]
        sim_clean = sim_aligned.values[valid_mask]

        if len(obs_clean) < min_overlap:
            raise ValueError(f"Insufficient valid data after NaN removal: {len(obs_clean)} points")

        return obs_clean, sim_clean

    def calculate_metrics(
        self,
        obs: np.ndarray,
        sim: np.ndarray,
        metrics: List[str] = ['kge', 'nse']
    ) -> Dict[str, float]:
        """
        Calculate performance metrics with error handling.

        Args:
            obs: Observed values (numpy array)
            sim: Simulated values (numpy array)
            metrics: List of metrics to calculate ('kge', 'nse', 'rmse', 'mae', 'kge_prime')

        Returns:
            Dictionary of metric values, or penalty_score on error
        """
        try:
            # Check for zero variance in simulation (common during spinup)
            if np.std(sim) == 0:
                logger.warning("Zero variance in simulation - returning penalty scores")
                return {m: -1.0 for m in metrics}  # Bad but not penalty

            result = {}
            for metric in metrics:
                try:
                    if metric == 'kge':
                        val = kge(obs, sim, transfo=1)
                    elif metric == 'nse':
                        val = nse(obs, sim, transfo=1)
                    elif metric == 'rmse':
                        val = rmse(obs, sim, transfo=1)
                    elif metric == 'mae':
                        val = mae(obs, sim, transfo=1)
                    elif metric == 'kge_prime':
                        val = kge_prime(obs, sim, transfo=1)
                    else:
                        logger.warning(f"Unknown metric: {metric}")
                        val = self.penalty_score

                    # Handle NaN/Inf values
                    if pd.isna(val) or np.isinf(val):
                        val = self.penalty_score

                    result[metric] = float(val)

                except Exception as e:
                    logger.warning(f"Error calculating {metric}: {e}")
                    result[metric] = self.penalty_score

            return result

        except Exception as e:
            logger.error(f"Error in metric calculation: {e}")
            return {'kge': self.penalty_score}

    def evaluate_streamflow(
        self,
        sim_series: pd.Series,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        area_km2: Optional[float] = None,
        area_source: str = 'shapefile',
        settings_dir: Optional[Path] = None,
        input_units: str = 'mm/day',
        metrics: List[str] = ['kge', 'nse'],
        resample_freq: Optional[str] = 'D',
        calibration_period: Optional[Tuple[str, str]] = None
    ) -> Dict[str, float]:
        """
        Full evaluation pipeline: load obs, convert units, align, calculate metrics.

        Args:
            sim_series: Simulation time series (pandas Series with datetime index)
            config: Configuration dictionary
            project_dir: Project directory path
            domain_name: Domain name
            area_km2: Catchment area in km2 (if None, will be loaded from source)
            area_source: Source for area if not provided: 'shapefile', 'netcdf', 'geodata'
            settings_dir: Settings directory (for netcdf/geodata area sources)
            input_units: Units of simulation data: 'mm/day', 'm/s', 'm3/s'
            metrics: List of metrics to calculate
            resample_freq: Resampling frequency for observations
            calibration_period: Optional (start_date, end_date) for filtering

        Returns:
            Dictionary of metric values
        """
        try:
            # Load observations
            obs_values, obs_index = self.load_observations(
                config, project_dir, domain_name, resample_freq
            )
            if obs_values is None:
                return {'kge': self.penalty_score}

            obs_series = pd.Series(obs_values, index=obs_index)

            # Get catchment area if not provided
            if area_km2 is None:
                area_km2 = self.get_catchment_area(
                    config, project_dir, domain_name, area_source, settings_dir
                )

            # Convert simulation to discharge if needed
            if input_units != 'm3/s':
                sim_discharge = self.convert_runoff_to_discharge(sim_series, area_km2, input_units)
            else:
                sim_discharge = sim_series

            # Ensure sim_discharge is a Series
            if not isinstance(sim_discharge, pd.Series):
                sim_discharge = pd.Series(sim_discharge, index=sim_series.index)

            # Align time series
            obs_aligned, sim_aligned = self.align_timeseries(
                sim_discharge, obs_series, calibration_period
            )

            # Calculate metrics
            return self.calculate_metrics(obs_aligned, sim_aligned, metrics)

        except ValueError as e:
            logger.warning(f"Alignment error: {e}")
            return {'kge': self.penalty_score}
        except Exception as e:
            logger.error(f"Error in streamflow evaluation: {e}")
            return {'kge': self.penalty_score}


# Module-level instance for convenience
_streamflow_metrics = StreamflowMetrics()


def load_observations(
    config: Dict[str, Any],
    project_dir: Path,
    domain_name: str,
    resample_freq: Optional[str] = 'D'
) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """Convenience function. See StreamflowMetrics.load_observations."""
    return _streamflow_metrics.load_observations(config, project_dir, domain_name, resample_freq)


def get_catchment_area(
    config: Dict[str, Any],
    project_dir: Path,
    domain_name: str,
    source: str = 'shapefile',
    settings_dir: Optional[Path] = None
) -> float:
    """Convenience function. See StreamflowMetrics.get_catchment_area."""
    return _streamflow_metrics.get_catchment_area(config, project_dir, domain_name, source, settings_dir)


def calculate_metrics(
    obs: np.ndarray,
    sim: np.ndarray,
    metrics: List[str] = ['kge', 'nse']
) -> Dict[str, float]:
    """Convenience function. See StreamflowMetrics.calculate_metrics."""
    return _streamflow_metrics.calculate_metrics(obs, sim, metrics)
