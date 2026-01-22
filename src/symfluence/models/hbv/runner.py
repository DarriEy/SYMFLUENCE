"""
HBV Model Runner.

Handles HBV-96 model execution, state management, and output processing.
Supports both lumped and distributed spatial modes with optional mizuRoute routing.
"""

from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base import BaseModelRunner
from symfluence.models.registry import ModelRegistry
from symfluence.models.execution import UnifiedModelExecutor
from symfluence.models.mizuroute.mixins import MizuRouteConfigMixin
from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.constants import UnitConversion

# Lazy JAX import
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


@ModelRegistry.register_runner('HBV', method_name='run_hbv')
class HBVRunner(BaseModelRunner, UnifiedModelExecutor, MizuRouteConfigMixin):
    """
    Runner class for the HBV-96 hydrological model.

    Supports:
    - Lumped mode (single catchment simulation)
    - Distributed mode (per-HRU simulation with mizuRoute routing)
    - JAX backend for autodiff/JIT compilation
    - NumPy fallback when JAX unavailable

    Attributes:
        config: Configuration dictionary or SymfluenceConfig object
        logger: Logger instance
        spatial_mode: 'lumped' or 'distributed'
        backend: 'jax' or 'numpy'
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None,
        settings_dir: Optional[Path] = None
    ):
        """
        Initialize HBV runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance
            reporting_manager: Optional reporting manager for visualization
            settings_dir: Optional override for settings directory
        """
        # Set settings_dir BEFORE super().__init__() so it's available in _setup_model_specific_paths
        self.settings_dir = Path(settings_dir) if settings_dir else None

        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Instance variables for external parameters during calibration
        self._external_params: Optional[Dict[str, float]] = None

        # Determine spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.hbv.spatial_mode if self.config.model and self.config.model.hbv else None,
            'auto'
        )

        if configured_mode in (None, 'auto', 'default'):
            if self.domain_definition_method == 'delineate':
                self.spatial_mode = 'distributed'
            else:
                self.spatial_mode = 'lumped'
        else:
            self.spatial_mode = configured_mode

        # Backend configuration
        self.backend = self._get_config_value(
            lambda: self.config.model.hbv.backend if self.config.model and self.config.model.hbv else None,
            'jax' if HAS_JAX else 'numpy'
        )

        if self.backend == 'jax' and not HAS_JAX:
            self.logger.warning("JAX not available, falling back to NumPy backend")
            self.backend = 'numpy'

        self.use_gpu = self._get_config_value(
            lambda: self.config.model.hbv.use_gpu if self.config.model and self.config.model.hbv else None,
            False
        )

        self.jit_compile = self._get_config_value(
            lambda: self.config.model.hbv.jit_compile if self.config.model and self.config.model.hbv else None,
            True
        )

        # Initial state configuration
        self.warmup_days = self._get_config_value(
            lambda: self.config.model.hbv.warmup_days if self.config.model and self.config.model.hbv else None,
            365
        )

        self.initial_snow = self._get_config_value(
            lambda: self.config.model.hbv.initial_snow if self.config.model and self.config.model.hbv else None,
            0.0
        )

        self.initial_sm = self._get_config_value(
            lambda: self.config.model.hbv.initial_sm if self.config.model and self.config.model.hbv else None,
            150.0
        )

        self.initial_suz = self._get_config_value(
            lambda: self.config.model.hbv.initial_suz if self.config.model and self.config.model.hbv else None,
            10.0
        )

        self.initial_slz = self._get_config_value(
            lambda: self.config.model.hbv.initial_slz if self.config.model and self.config.model.hbv else None,
            10.0
        )

        # Routing requirements
        self.needs_routing = self._check_routing_requirements()

        # Lazy-loaded model functions
        self._simulate_fn = None
        self._loss_fn = None
        self._grad_fn = None

    def _get_model_name(self) -> str:
        """Return model name for HBV."""
        return "HBV"

    def _setup_model_specific_paths(self) -> None:
        """Set up HBV-specific paths."""
        if hasattr(self, 'settings_dir') and self.settings_dir:
            self.hbv_setup_dir = self.settings_dir
        else:
            self.hbv_setup_dir = self.project_dir / "settings" / "HBV"

        self.hbv_forcing_dir = self.project_dir / 'forcing' / 'HBV_input'

    def _get_output_dir(self) -> Path:
        """HBV output directory."""
        return self.get_experiment_output_dir()

    def _get_catchment_area(self) -> float:
        """Get total catchment area in m²."""
        # Try to get from shapefile
        try:
            import geopandas as gpd
            # Construct catchment path directly since runner doesn't have get_catchment_path
            catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
            discretization = self._get_config_value(
                lambda: self.config.domain.discretization,
                'GRUs'
            )
            catchment_path = catchment_dir / f"{self.domain_name}_HRUs_{discretization}.shp"
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                # Look for area column
                area_cols = [c for c in gdf.columns if 'area' in c.lower()]
                if area_cols:
                    total_area = gdf[area_cols[0]].sum()
                    self.logger.info(f"Catchment area from shapefile: {total_area/1e6:.2f} km²")
                    return float(total_area)
            else:
                self.logger.debug(f"Catchment shapefile not found at: {catchment_path}")
        except Exception as e:
            self.logger.debug(f"Could not read catchment area from shapefile: {e}")

        # Fall back to config
        area_km2 = self._get_config_value(
            lambda: self.config.domain.catchment_area_km2,
            None
        )
        if area_km2:
            return area_km2 * 1e6

        # Default fallback
        self.logger.warning("Could not determine catchment area, using default 1000 km²")
        return 1000.0 * 1e6

    def _check_routing_requirements(self) -> bool:
        """Check if distributed routing is needed."""
        routing_integration = self._get_config_value(
            lambda: self.config.model.hbv.routing_integration if self.config.model and self.config.model.hbv else None,
            'none'
        )

        global_routing = self.routing_model

        if routing_integration and routing_integration.lower() == 'mizuroute':
            if self.spatial_mode == 'distributed':
                self.logger.info("HBV routing enabled via HBV_ROUTING_INTEGRATION: mizuRoute")
                return True

        if global_routing and global_routing.lower() == 'mizuroute':
            if self.spatial_mode == 'distributed':
                self.logger.info("HBV routing auto-enabled: ROUTING_MODEL=mizuRoute with distributed mode")
                return True

        return False

    def _get_default_params(self) -> Dict[str, float]:
        """Get default HBV parameters from config or built-in defaults."""
        from .model import DEFAULT_PARAMS

        params = {}
        for param_name in DEFAULT_PARAMS.keys():
            config_key = f'default_{param_name}'
            params[param_name] = self._get_config_value(
                lambda pn=config_key: getattr(self.config.model.hbv, pn, None)
                if self.config.model and self.config.model.hbv else None,
                DEFAULT_PARAMS[param_name]
            )

        return params

    def run_hbv(self, params: Optional[Dict[str, float]] = None) -> Optional[Path]:
        """
        Run the HBV-96 model.

        Args:
            params: Optional parameter dictionary. If provided, uses these
                    instead of defaults. Used during calibration.

        Returns:
            Path to output directory if successful, None otherwise.
        """
        # Emit experimental warning on first use
        from symfluence.models.hbv import _warn_experimental
        _warn_experimental()

        self.logger.info(f"Starting HBV model run in {self.spatial_mode} mode (backend: {self.backend})")

        # Store provided parameters
        if params:
            self.logger.info(f"Using external parameters: {params}")
            self._external_params = params

        with symfluence_error_handler(
            "HBV model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Execute model
            if self.spatial_mode == 'lumped':
                success = self._execute_lumped()
            else:
                success = self._execute_distributed()

            # Run routing if needed
            if success and self.needs_routing:
                self.logger.info("Running distributed routing with mizuRoute")
                success = self._run_distributed_routing()

            if success:
                self.logger.info("HBV model run completed successfully")
                self._calculate_and_log_metrics()
                return self.output_dir
            else:
                self.logger.error("HBV model run failed")
                return None

    def _execute_lumped(self) -> bool:
        """Execute HBV in lumped mode."""
        self.logger.info("Running lumped HBV simulation")

        try:
            # Import model functions
            from .model import (
                simulate, create_initial_state,
                HAS_JAX as MODEL_HAS_JAX
            )

            # Load forcing data
            forcing, obs = self._load_forcing()

            precip = forcing['precip'].flatten()
            temp = forcing['temp'].flatten()
            pet = forcing['pet'].flatten()
            time_index = forcing['time']

            # Get parameters
            params = self._external_params if self._external_params else self._get_default_params()

            # Convert to JAX/numpy arrays
            use_jax = self.backend == 'jax' and MODEL_HAS_JAX

            if use_jax:
                precip = jnp.array(precip)
                temp = jnp.array(temp)
                pet = jnp.array(pet)

            # Create initial state
            initial_state = create_initial_state(
                initial_snow=self.initial_snow,
                initial_sm=self.initial_sm,
                initial_suz=self.initial_suz,
                initial_slz=self.initial_slz,
                use_jax=use_jax
            )

            # Run simulation
            self.logger.info(f"Running simulation for {len(precip)} timesteps")

            runoff, final_state = simulate(
                precip, temp, pet,
                params=params,
                initial_state=initial_state,
                warmup_days=self.warmup_days,
                use_jax=use_jax
            )

            # Convert output to numpy if needed
            if use_jax:
                runoff = np.array(runoff)

            # Save results
            self._save_lumped_results(runoff, time_index)

            return True

        except Exception as e:
            self.logger.error(f"Error in lumped HBV execution: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _execute_distributed(self) -> bool:
        """Execute HBV in distributed mode (per-HRU)."""
        self.logger.info("Running distributed HBV simulation")

        try:
            from .model import (
                simulate, create_initial_state,
                HAS_JAX as MODEL_HAS_JAX
            )

            # Load distributed forcing
            forcing_file = self.hbv_forcing_dir / f"{self.domain_name}_hbv_forcing_distributed.nc"
            if not forcing_file.exists():
                self.logger.error(f"Distributed forcing not found: {forcing_file}")
                return False

            ds = xr.open_dataset(forcing_file)

            precip = ds['pr'].values  # (time, hru)
            temp = ds['temp'].values
            pet = ds['pet'].values
            time_index = pd.to_datetime(ds.time.values)
            hru_ids = ds['hru_id'].values if 'hru_id' in ds else np.arange(ds.dims['hru']) + 1

            n_times, n_hrus = precip.shape
            self.logger.info(f"Running simulation for {n_times} timesteps x {n_hrus} HRUs")

            # Get parameters
            params = self._external_params if self._external_params else self._get_default_params()

            use_jax = self.backend == 'jax' and MODEL_HAS_JAX

            # Run simulation for each HRU
            all_runoff = np.zeros((n_times, n_hrus))

            for hru_idx in range(n_hrus):
                hru_precip = precip[:, hru_idx]
                hru_temp = temp[:, hru_idx]
                hru_pet = pet[:, hru_idx]

                if use_jax:
                    hru_precip = jnp.array(hru_precip)
                    hru_temp = jnp.array(hru_temp)
                    hru_pet = jnp.array(hru_pet)

                initial_state = create_initial_state(
                    initial_snow=self.initial_snow,
                    initial_sm=self.initial_sm,
                    initial_suz=self.initial_suz,
                    initial_slz=self.initial_slz,
                    use_jax=use_jax
                )

                runoff, _ = simulate(
                    hru_precip, hru_temp, hru_pet,
                    params=params,
                    initial_state=initial_state,
                    warmup_days=self.warmup_days,
                    use_jax=use_jax
                )

                if use_jax:
                    runoff = np.array(runoff)

                all_runoff[:, hru_idx] = runoff

            # Save distributed results
            self._save_distributed_results(all_runoff, time_index, hru_ids)

            return True

        except Exception as e:
            self.logger.error(f"Error in distributed HBV execution: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_forcing(self) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """Load forcing data from preprocessed files."""
        # Try NetCDF first
        nc_file = self.hbv_forcing_dir / f"{self.domain_name}_hbv_forcing.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            forcing = {
                'precip': ds['pr'].values,
                'temp': ds['temp'].values,
                'pet': ds['pet'].values,
                'time': pd.to_datetime(ds.time.values),
            }
            ds.close()
        else:
            # Try CSV
            csv_file = self.hbv_forcing_dir / f"{self.domain_name}_hbv_forcing.csv"
            if not csv_file.exists():
                raise FileNotFoundError(f"No forcing file found at {nc_file} or {csv_file}")

            df = pd.read_csv(csv_file)
            forcing = {
                'precip': df['pr'].values,
                'temp': df['temp'].values,
                'pet': df['pet'].values,
                'time': pd.to_datetime(df['time']),
            }

        # Load observations if available
        obs_file = self.hbv_forcing_dir / f"{self.domain_name}_observations.csv"
        if obs_file.exists():
            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            obs = obs_df.iloc[:, 0].values
        else:
            obs = None

        return forcing, obs

    def _save_lumped_results(self, runoff: np.ndarray, time_index: pd.DatetimeIndex) -> None:
        """Save lumped simulation results."""
        # Get catchment area for unit conversion
        area_m2 = self._get_catchment_area()

        # Convert mm/day to m³/s: Q = runoff * area / (1000 * 86400)
        streamflow_cms = runoff * area_m2 / (1000.0 * UnitConversion.SECONDS_PER_DAY)

        # Create DataFrame with both units
        results_df = pd.DataFrame({
            'datetime': time_index,
            'streamflow_mm_day': runoff,
            'streamflow_cms': streamflow_cms,
        })

        # Save CSV
        csv_file = self.output_dir / f"{self.domain_name}_hbv_output.csv"
        results_df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved lumped results to: {csv_file}")

        # Save NetCDF with streamflow in m³/s (standard unit)
        ds = xr.Dataset(
            data_vars={
                'streamflow': (['time'], streamflow_cms),
                'runoff': (['time'], runoff),
            },
            coords={
                'time': time_index,
            },
            attrs={
                'model': 'HBV-96',
                'spatial_mode': 'lumped',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'catchment_area_m2': area_m2,
            }
        )
        ds['streamflow'].attrs = {'units': 'm3/s', 'long_name': 'Streamflow'}
        ds['runoff'].attrs = {'units': 'mm/day', 'long_name': 'Runoff depth'}

        nc_file = self.output_dir / f"{self.domain_name}_hbv_output.nc"
        encoding = create_netcdf_encoding(ds, compression=True)
        ds.to_netcdf(nc_file, encoding=encoding)
        self.logger.info(f"Saved NetCDF output to: {nc_file}")

    def _save_distributed_results(
        self,
        runoff: np.ndarray,
        time_index: pd.DatetimeIndex,
        hru_ids: np.ndarray
    ) -> None:
        """Save distributed simulation results for mizuRoute."""
        n_hrus = runoff.shape[1]

        # Create time coordinate in seconds since 1970
        time_seconds = (time_index - pd.Timestamp('1970-01-01')).total_seconds().values

        # Convert runoff from mm/day to m/s for mizuRoute
        runoff_ms = runoff / (1000.0 * UnitConversion.SECONDS_PER_DAY)

        # Get routing variable name
        routing_var = self.mizu_routing_var or 'q_routed'

        # Create Dataset
        ds = xr.Dataset(
            data_vars={
                'gruId': (['gru'], hru_ids.astype(np.int32)),
                routing_var: (['time', 'gru'], runoff_ms),
            },
            coords={
                'time': ('time', time_seconds),
                'gru': ('gru', np.arange(n_hrus)),
            },
            attrs={
                'model': 'HBV-96',
                'spatial_mode': 'distributed',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'n_hrus': n_hrus,
            }
        )

        ds['gruId'].attrs = {
            'long_name': 'ID of grouped response unit',
            'units': '-'
        }

        ds[routing_var].attrs = {
            'long_name': 'HBV-96 runoff for mizuRoute routing',
            'units': 'm/s',
        }

        ds.time.attrs = {
            'units': 'seconds since 1970-01-01 00:00:00',
            'calendar': 'standard',
        }

        # Save
        output_file = self.output_dir / f"{self.domain_name}_{self.experiment_id}_runs_def.nc"
        encoding = create_netcdf_encoding(ds, compression=True, int_vars={'gruId': 'int32'})
        ds.to_netcdf(output_file, encoding=encoding)
        self.logger.info(f"Saved distributed results to: {output_file}")

    def _run_distributed_routing(self) -> bool:
        """Run mizuRoute routing for distributed output."""
        self.logger.info("Starting mizuRoute routing for distributed HBV")

        self._setup_hbv_mizuroute_config()

        mizu_settings_dir = self.mizu_settings_path
        mizu_control = self.mizu_control_file or 'mizuRoute_control_HBV.txt'

        create_control = True
        if mizu_settings_dir:
            control_path = Path(mizu_settings_dir) / mizu_control
            if control_path.exists():
                self.logger.debug(f"MizuRoute control file exists at {control_path}")
                create_control = False

        spatial_config = self.get_spatial_config('HBV')
        result = self._run_mizuroute(spatial_config, model_name='hbv', create_control_file=create_control)

        return result is not None

    def _setup_hbv_mizuroute_config(self):
        """Update configuration for HBV-mizuRoute integration."""
        self.config_dict['MIZU_FROM_MODEL'] = 'HBV'

        if not self.mizu_control_file:
            self.config_dict['SETTINGS_MIZU_CONTROL_FILE'] = 'mizuRoute_control_HBV.txt'

        current_models = self.hydrological_model
        if 'MIZUROUTE' not in current_models.upper():
            self.config_dict['HYDROLOGICAL_MODEL'] = f"{current_models},MIZUROUTE"

    def _calculate_and_log_metrics(self) -> None:
        """Calculate and log performance metrics."""
        try:
            from symfluence.evaluation.metrics import kge, nse

            # Load simulation (now in m³/s)
            output_file = self.output_dir / f"{self.domain_name}_hbv_output.nc"
            if output_file.exists():
                ds = xr.open_dataset(output_file)
                sim = ds['streamflow'].values  # Already in m³/s
                sim_time = pd.to_datetime(ds.time.values)
                ds.close()
            else:
                # Try CSV
                csv_file = self.output_dir / f"{self.domain_name}_hbv_output.csv"
                if not csv_file.exists():
                    self.logger.warning("No output file found for metrics calculation")
                    return
                df = pd.read_csv(csv_file)
                sim = df['streamflow_cms'].values  # Use m³/s column
                sim_time = pd.to_datetime(df['datetime'])

            # Load observations (in m³/s)
            obs_file = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
            if not obs_file.exists():
                self.logger.warning("Observations not found for metrics")
                return

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # Align time series
            sim_series = pd.Series(sim, index=sim_time)
            obs_series = obs_df.iloc[:, 0]  # Already in m³/s (discharge_cms)

            # Skip warmup
            if len(sim_series) > self.warmup_days:
                sim_series = sim_series.iloc[self.warmup_days:]

            # Find common dates
            common_idx = sim_series.index.intersection(obs_series.index)
            if len(common_idx) < 10:
                self.logger.warning(f"Insufficient common dates ({len(common_idx)}) for metrics")
                return

            sim_aligned = sim_series.loc[common_idx].values
            obs_aligned = obs_series.loc[common_idx].values

            # Remove NaN
            valid_mask = ~(np.isnan(sim_aligned) | np.isnan(obs_aligned))
            sim_aligned = sim_aligned[valid_mask]
            obs_aligned = obs_aligned[valid_mask]

            if len(sim_aligned) == 0:
                self.logger.warning("No valid data pairs for metrics")
                return

            # Calculate metrics
            kge_val = kge(obs_aligned, sim_aligned, transfo=1)
            nse_val = nse(obs_aligned, sim_aligned, transfo=1)

            self.logger.info("=" * 40)
            self.logger.info(f"HBV Model Performance ({self.spatial_mode})")
            self.logger.info(f"   KGE: {kge_val:.4f}")
            self.logger.info(f"   NSE: {nse_val:.4f}")
            self.logger.info(f"   Output: {self.output_dir}")
            self.logger.info("=" * 40)

        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {e}")
            self.logger.debug("Traceback:", exc_info=True)

    # =========================================================================
    # Calibration Support
    # =========================================================================

    def get_loss_function(self, metric: str = 'kge') -> Callable:
        """
        Get differentiable loss function for calibration.

        Args:
            metric: 'kge' or 'nse'

        Returns:
            Loss function that takes (params_dict, precip, temp, pet, obs) -> loss
        """
        from .model import nse_loss, kge_loss

        if metric.lower() == 'nse':
            return nse_loss
        return kge_loss

    def get_gradient_function(self, metric: str = 'kge') -> Optional[Callable]:
        """
        Get gradient function for gradient-based calibration.

        Args:
            metric: 'kge' or 'nse'

        Returns:
            Gradient function or None if JAX unavailable.
        """
        if not HAS_JAX:
            self.logger.warning("JAX not available for gradient computation")
            return None

        from .model import get_nse_gradient_fn, get_kge_gradient_fn

        # Load forcing
        forcing, obs = self._load_forcing()

        precip = jnp.array(forcing['precip'].flatten())
        temp = jnp.array(forcing['temp'].flatten())
        pet = jnp.array(forcing['pet'].flatten())

        if obs is None:
            self.logger.error("Observations required for gradient calibration")
            return None

        obs = jnp.array(obs)

        if metric.lower() == 'nse':
            return get_nse_gradient_fn(precip, temp, pet, obs, self.warmup_days)
        return get_kge_gradient_fn(precip, temp, pet, obs, self.warmup_days)

    def evaluate_parameters(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> float:
        """
        Evaluate a parameter set.

        Args:
            params: Parameter dictionary
            metric: Evaluation metric

        Returns:
            Metric value (higher is better)
        """
        from .model import kge_loss, nse_loss

        forcing, obs = self._load_forcing()

        if obs is None:
            self.logger.error("Observations required for evaluation")
            return -999.0

        use_jax = self.backend == 'jax' and HAS_JAX

        if use_jax:
            precip = jnp.array(forcing['precip'].flatten())
            temp = jnp.array(forcing['temp'].flatten())
            pet = jnp.array(forcing['pet'].flatten())
            obs = jnp.array(obs)
        else:
            precip = forcing['precip'].flatten()
            temp = forcing['temp'].flatten()
            pet = forcing['pet'].flatten()

        if metric.lower() == 'nse':
            loss = nse_loss(params, precip, temp, pet, obs, self.warmup_days, use_jax)
        else:
            loss = kge_loss(params, precip, temp, pet, obs, self.warmup_days, use_jax)

        # Return positive metric (loss is negative)
        return -float(loss)
