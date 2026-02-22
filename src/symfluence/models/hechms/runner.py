"""
HEC-HMS Model Runner.

Handles HEC-HMS model execution, output processing, and result saving.
Lumped mode with JAX/NumPy backend support.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base import BaseModelRunner
from symfluence.models.registry import ModelRegistry
from symfluence.models.execution import SpatialOrchestrator
from symfluence.models.mixins import SpatialModeDetectionMixin, ObservationLoaderMixin
from symfluence.geospatial.geometry_utils import calculate_catchment_area_km2
from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

# Lazy JAX import
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


@ModelRegistry.register_runner('HECHMS', method_name='run_hechms')
class HecHmsRunner(  # type: ignore[misc]
    BaseModelRunner,
    SpatialOrchestrator,
    SpatialModeDetectionMixin,
    ObservationLoaderMixin
):
    """
    Runner class for the HEC-HMS hydrological model.

    Supports:

    - Lumped mode (single catchment simulation)
    - JAX backend for autodiff/JIT compilation
    - NumPy fallback when JAX unavailable
    """

    MODEL_NAME = "HECHMS"

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None,
        settings_dir: Optional[Path] = None
    ):
        """Initialize HEC-HMS runner."""
        self.settings_dir = Path(settings_dir) if settings_dir else None
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self._external_params: Optional[Dict[str, float]] = None

        # Spatial mode
        self.spatial_mode = self.detect_spatial_mode('HECHMS')

        # Backend configuration
        self.backend = self._get_config_value(
            lambda: self.config.model.hechms.backend if self.config.model and self.config.model.hechms else None,
            'jax' if HAS_JAX else 'numpy'
        )

        if self.backend == 'jax' and not HAS_JAX:
            self.logger.warning("JAX not available, falling back to NumPy backend")
            self.backend = 'numpy'

        # Warmup configuration
        self.warmup_days = self._get_config_value(
            lambda: self.config.model.hechms.warmup_days if self.config.model and self.config.model.hechms else None,
            365
        )

        # Lazy model function
        self._simulate_fn = None

    def _setup_model_specific_paths(self) -> None:
        """Set up HEC-HMS-specific paths."""
        if hasattr(self, 'settings_dir') and self.settings_dir:
            self.hechms_setup_dir = self.settings_dir
        else:
            self.hechms_setup_dir = self.project_dir / "settings" / "HECHMS"

        self.hechms_forcing_dir = self.project_forcing_dir / 'HECHMS_input'

    def _get_output_dir(self) -> Path:
        """HEC-HMS output directory."""
        return self.get_experiment_output_dir()

    def _get_catchment_area(self) -> float:
        """Get total catchment area in m2."""
        try:
            import geopandas as gpd
            catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
            discretization = self._get_config_value(
                lambda: self.config.domain.discretization,
                'GRUs'
            )

            possible_paths = [
                catchment_dir / f"{self.domain_name}_HRUs_{discretization}.shp",
                catchment_dir / self.spatial_mode / self.experiment_id / f"{self.domain_name}_HRUs_{discretization}.shp",
                catchment_dir / self.spatial_mode / f"{self.domain_name}_HRUs_{discretization}.shp",
            ]

            for path in possible_paths:
                if path.exists():
                    gdf = gpd.read_file(path)
                    try:
                        area_km2 = calculate_catchment_area_km2(gdf, logger=self.logger)
                        return float(area_km2 * 1e6)
                    except Exception:
                        area_cols = [c for c in gdf.columns if 'area' in c.lower()]
                        if area_cols:
                            return float(gdf[area_cols[0]].sum() * 1e6)

        except (ImportError, FileNotFoundError, OSError, KeyError, ValueError):
            pass

        # Fall back to config
        area_km2 = self._get_config_value(
            lambda: self.config.domain.catchment_area_km2,
            None
        )
        if area_km2:
            return area_km2 * 1e6

        raise ValueError(
            "Catchment area could not be determined. Provide via shapefile or CATCHMENT_AREA_KM2 config."
        )

    def _get_default_params(self) -> Dict[str, float]:
        """Get default HEC-HMS parameters from config or built-in defaults."""
        from .parameters import DEFAULT_PARAMS

        params = {}
        for param_name in DEFAULT_PARAMS.keys():
            config_key = f'default_{param_name}'
            params[param_name] = self._get_config_value(
                lambda pn=config_key: getattr(  # type: ignore[misc]
                    self.config.model.hechms, pn, None)
                if self.config.model and self.config.model.hechms else None,
                DEFAULT_PARAMS[param_name]
            )

        return params

    def run_hechms(self, params: Optional[Dict[str, float]] = None) -> Optional[Path]:
        """
        Run the HEC-HMS model.

        Args:
            params: Optional parameter dictionary for calibration.

        Returns:
            Path to output directory if successful, None otherwise.
        """
        self.logger.info(f"Starting HEC-HMS model run in {self.spatial_mode} mode (backend: {self.backend})")

        if params:
            self.logger.info(f"Using external parameters: {params}")
            self._external_params = params

        with symfluence_error_handler(
            "HEC-HMS model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            success = self._execute_lumped()

            if success:
                self.logger.info("HEC-HMS model run completed successfully")
                return self.output_dir
            else:
                self.logger.error("HEC-HMS model run failed")
                return None

    def _execute_lumped(self) -> bool:
        """Execute HEC-HMS in lumped mode."""
        self.logger.info("Running lumped HEC-HMS simulation")

        try:
            from .model import simulate, HAS_JAX as MODEL_HAS_JAX

            # Load forcing data
            forcing = self._load_forcing()
            if forcing is None:
                return False

            precip = forcing['precip'].flatten()
            temp = forcing['temp'].flatten()
            pet = forcing['pet'].flatten()
            time_index = forcing['time']

            # Get parameters
            params = self._external_params if self._external_params else self._get_default_params()

            use_jax = self.backend == 'jax' and MODEL_HAS_JAX

            if use_jax:
                precip = jnp.array(precip)
                temp = jnp.array(temp)
                pet = jnp.array(pet)

            # Determine day of year start
            doy_start = 1
            if len(time_index) > 0:
                doy_start = time_index[0].timetuple().tm_yday

            # Run simulation
            self.logger.info(f"Running simulation for {len(precip)} timesteps")

            runoff, final_state = simulate(
                precip, temp, pet,
                params=params,
                warmup_days=self.warmup_days,
                use_jax=use_jax,
                day_of_year_start=doy_start,
            )

            # Convert output to numpy
            if use_jax:
                runoff = np.array(runoff)

            # Save results
            self._save_lumped_results(runoff, time_index)

            return True

        except FileNotFoundError as e:
            self.logger.error(f"Missing forcing data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"HEC-HMS simulation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_forcing(self) -> Optional[Dict[str, Any]]:
        """Load forcing data from HECHMS_input directory."""
        forcing_dir = self.hechms_forcing_dir
        domain_name = self.domain_name

        nc_patterns = [
            forcing_dir / f"{domain_name}_hechms_forcing.nc",
            forcing_dir / f"{domain_name}_forcing.nc",
        ]

        for nc_file in nc_patterns:
            if nc_file.exists():
                try:
                    ds = xr.open_dataset(nc_file)
                    forcing = {}

                    var_map = {'precip': 'pr', 'temp': 'temp', 'pet': 'pet'}
                    for std_name, var_name in var_map.items():
                        if var_name in ds.variables:
                            forcing[std_name] = ds[var_name].values.flatten()
                        elif std_name in ds.variables:
                            forcing[std_name] = ds[std_name].values.flatten()

                    if 'time' in ds.coords:
                        forcing['time'] = pd.to_datetime(ds.time.values)

                    ds.close()

                    if len(forcing) >= 3:
                        self.logger.info(f"Loaded forcing from {nc_file.name}: {len(forcing['precip'])} timesteps")
                        return forcing
                except Exception as e:
                    self.logger.warning(f"Error loading {nc_file}: {e}")

        # Try CSV fallback
        csv_patterns = [
            forcing_dir / f"{domain_name}_hechms_forcing.csv",
        ]

        for csv_file in csv_patterns:
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, parse_dates=['datetime'])
                    return {
                        'precip': df['pr'].values,
                        'temp': df['temp'].values,
                        'pet': df['pet'].values,
                        'time': pd.to_datetime(df['datetime']),
                    }
                except Exception as e:
                    self.logger.warning(f"Error loading {csv_file}: {e}")

        self.logger.error(f"No forcing file found in {forcing_dir}")
        return None

    def _save_lumped_results(self, runoff: np.ndarray, time_index: pd.DatetimeIndex) -> None:
        """Save lumped simulation results."""
        n = min(len(runoff), len(time_index))
        runoff = runoff[:n]
        time_index = time_index[:n]

        # Convert mm/day to m3/s
        try:
            area_m2 = self._get_catchment_area()
            area_km2 = area_m2 / 1e6
            streamflow_cms = runoff * area_m2 / 1000.0 / 86400.0
        except ValueError:
            self.logger.warning("Catchment area not available; saving runoff in mm/day only")
            area_km2 = None
            streamflow_cms = runoff  # fallback

        # Save CSV
        csv_data = {
            'datetime': time_index,
            'streamflow_mm_day': runoff,
        }
        if area_km2 is not None:
            csv_data['streamflow_cms'] = streamflow_cms

        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / f"{self.domain_name}_hechms_output.csv"
        df.to_csv(csv_path, index=False)

        # Save NetCDF
        data_vars = {
            'runoff': (['time'], runoff.astype(np.float64), {'units': 'mm/day', 'long_name': 'Total runoff'}),
        }
        if area_km2 is not None:
            data_vars['streamflow'] = (
                ['time'], streamflow_cms.astype(np.float64),
                {'units': 'm3/s', 'long_name': 'Streamflow'}
            )

        ds = xr.Dataset(
            data_vars,
            coords={'time': time_index},
            attrs={
                'model': 'HEC-HMS',
                'description': 'HEC-HMS native Python/JAX simulation output',
                'catchment_area_km2': float(area_km2) if area_km2 else 'unknown',
            }
        )
        nc_path = self.output_dir / f"{self.domain_name}_hechms_output.nc"
        encoding = create_netcdf_encoding(ds)
        ds.to_netcdf(nc_path, encoding=encoding)

        self.logger.info(f"Saved results: {csv_path.name}, {nc_path.name}")
