"""
jFUSE Model Runner.

Handles jFUSE model execution, state management, and output processing.
Supports both lumped and distributed spatial modes with optional Muskingum-Cunge routing.

jFUSE is a JAX-based implementation of the FUSE (Framework for Understanding Structural
Errors) model, enabling automatic differentiation for gradient-based calibration.
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
from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.constants import UnitConversion

# Lazy jFUSE and JAX import
try:
    import jfuse
    from jfuse import FUSEModel, ModelConfig, PARAM_BOUNDS
    HAS_JFUSE = True
except ImportError:
    HAS_JFUSE = False
    jfuse = None
    FUSEModel = None
    ModelConfig = None
    PARAM_BOUNDS = {}

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


@ModelRegistry.register_runner('JFUSE', method_name='run_jfuse')
class JFUSERunner(BaseModelRunner, UnifiedModelExecutor):
    """
    Runner class for the jFUSE hydrological model.

    Supports:
    - Lumped mode (single catchment simulation)
    - Distributed mode (per-HRU simulation with optional routing)
    - JAX backend for autodiff/JIT compilation
    - Multiple model structures (PRMS, Sacramento, TOPMODEL, VIC)

    Attributes:
        config: Configuration dictionary or SymfluenceConfig object
        logger: Logger instance
        spatial_mode: 'lumped' or 'distributed'
        model_config_name: jFUSE model structure name
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None,
        settings_dir: Optional[Path] = None
    ):
        """
        Initialize jFUSE runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance
            reporting_manager: Optional reporting manager for visualization
            settings_dir: Optional override for settings directory
        """
        self.settings_dir = Path(settings_dir) if settings_dir else None

        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Check jFUSE availability
        if not HAS_JFUSE:
            self.logger.warning("jFUSE not installed. Install with: pip install jfuse")

        # Instance variables for external parameters during calibration
        self._external_params: Optional[Dict[str, float]] = None

        # Determine spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.jfuse.spatial_mode if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            'auto'
        )

        if configured_mode in (None, 'auto', 'default'):
            if self.domain_definition_method == 'delineate':
                self.spatial_mode = 'distributed'
            else:
                self.spatial_mode = 'lumped'
        else:
            self.spatial_mode = configured_mode

        # Model structure configuration
        self.model_config_name = self._get_config_value(
            lambda: self.config.model.jfuse.model_config_name if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            'prms'
        )

        # Snow configuration
        self.enable_snow = self._get_config_value(
            lambda: self.config.model.jfuse.enable_snow if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            True
        )

        # Routing configuration
        self.enable_routing = self._get_config_value(
            lambda: self.config.model.jfuse.enable_routing if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            False
        )

        # JIT configuration
        self.jit_compile = self._get_config_value(
            lambda: self.config.model.jfuse.jit_compile if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            True
        )

        self.use_gpu = self._get_config_value(
            lambda: self.config.model.jfuse.use_gpu if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            False
        )

        # Initial state configuration
        self.warmup_days = self._get_config_value(
            lambda: self.config.model.jfuse.warmup_days if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            365
        )

        self.initial_s1 = self._get_config_value(
            lambda: self.config.model.jfuse.initial_s1 if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            0.0
        )

        self.initial_s2 = self._get_config_value(
            lambda: self.config.model.jfuse.initial_s2 if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            50.0
        )

        self.initial_snow = self._get_config_value(
            lambda: self.config.model.jfuse.initial_snow if self.config.model and hasattr(self.config.model, 'jfuse') and self.config.model.jfuse else None,
            0.0
        )

        # Lazy-loaded model
        self._model = None
        self._coupled_model = None

    def _get_model_name(self) -> str:
        """Return model name for jFUSE."""
        return "JFUSE"

    def _setup_model_specific_paths(self) -> None:
        """Set up jFUSE-specific paths."""
        if hasattr(self, 'settings_dir') and self.settings_dir:
            self.jfuse_setup_dir = self.settings_dir
        else:
            self.jfuse_setup_dir = self.project_dir / "settings" / "JFUSE"

        self.jfuse_forcing_dir = self.project_dir / 'forcing' / 'JFUSE_input'

    def _get_output_dir(self) -> Path:
        """jFUSE output directory."""
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
            catchment_path = catchment_dir / f"{self.domain_name}_HRUs_{discretization}.shp"
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                area_cols = [c for c in gdf.columns if 'area' in c.lower()]
                if area_cols:
                    total_area = gdf[area_cols[0]].sum()
                    self.logger.info(f"Catchment area from shapefile: {total_area/1e6:.2f} km2")
                    return float(total_area)
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
        self.logger.warning("Could not determine catchment area, using default 1000 km2")
        return 1000.0 * 1e6

    def _get_default_params(self) -> Dict[str, float]:
        """Get default jFUSE parameters."""
        if not HAS_JFUSE:
            return {}

        try:
            from jfuse import get_default_params
            return get_default_params(self.model_config_name)
        except (ImportError, AttributeError):
            # Fall back to manual defaults
            return {
                'S1_max': 100.0,
                'S2_max': 500.0,
                'ku': 0.5,
                'ki': 0.1,
                'ks': 0.01,
                'n': 1.0,
                'v': 0.5,
                'Ac_max': 0.5,
                'T_melt': 0.0,
                'melt_rate': 3.0,
            }

    def run_jfuse(self, params: Optional[Dict[str, float]] = None) -> Optional[Path]:
        """
        Run the jFUSE model.

        Args:
            params: Optional parameter dictionary. If provided, uses these
                    instead of defaults. Used during calibration.

        Returns:
            Path to output directory if successful, None otherwise.
        """
        if not HAS_JFUSE:
            self.logger.error("jFUSE not installed. Cannot run model.")
            return None

        self.logger.info(f"Starting jFUSE model run in {self.spatial_mode} mode (structure: {self.model_config_name})")

        # Store provided parameters
        if params:
            self.logger.info(f"Using external parameters: {params}")
            self._external_params = params

        with symfluence_error_handler(
            "jFUSE model execution",
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

            if success:
                self.logger.info("jFUSE model run completed successfully")
                self._calculate_and_log_metrics()
                return self.output_dir
            else:
                self.logger.error("jFUSE model run failed")
                return None

    def _execute_lumped(self) -> bool:
        """Execute jFUSE in lumped mode."""
        self.logger.info("Running lumped jFUSE simulation")

        try:
            # Load forcing data
            forcing, obs = self._load_forcing()

            precip = forcing['precip'].flatten()
            temp = forcing['temp'].flatten()
            pet = forcing['pet'].flatten()
            time_index = forcing['time']

            # Get parameters
            params = self._external_params if self._external_params else self._get_default_params()

            # Create jFUSE model
            model_config = ModelConfig(
                model_name=self.model_config_name,
                enable_snow=self.enable_snow,
            )
            model = FUSEModel(model_config)

            # Convert to JAX arrays
            precip_jax = jnp.array(precip)
            temp_jax = jnp.array(temp)
            pet_jax = jnp.array(pet)

            # Run simulation
            self.logger.info(f"Running simulation for {len(precip)} timesteps")

            runoff, final_state = model.simulate(
                precip_jax, temp_jax, pet_jax,
                params=params,
                warmup_steps=self.warmup_days,
            )

            # Convert output to numpy
            runoff = np.array(runoff)

            # Save results
            self._save_lumped_results(runoff, time_index)

            return True

        except Exception as e:
            self.logger.error(f"Error in lumped jFUSE execution: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _execute_distributed(self) -> bool:
        """Execute jFUSE in distributed mode (per-HRU)."""
        self.logger.info("Running distributed jFUSE simulation")

        try:
            # Load distributed forcing
            forcing_file = self.jfuse_forcing_dir / f"{self.domain_name}_jfuse_forcing_distributed.nc"
            if not forcing_file.exists():
                self.logger.error(f"Distributed forcing not found: {forcing_file}")
                return False

            ds = xr.open_dataset(forcing_file)

            precip = ds['precip'].values  # (time, hru)
            temp = ds['temp'].values
            pet = ds['pet'].values
            time_index = pd.to_datetime(ds.time.values)
            hru_ids = ds['hru_id'].values if 'hru_id' in ds else np.arange(ds.dims['hru']) + 1

            n_times, n_hrus = precip.shape
            self.logger.info(f"Running simulation for {n_times} timesteps x {n_hrus} HRUs")

            # Get parameters
            params = self._external_params if self._external_params else self._get_default_params()

            # Create jFUSE model
            model_config = ModelConfig(
                model_name=self.model_config_name,
                enable_snow=self.enable_snow,
            )
            model = FUSEModel(model_config)

            # Run simulation for each HRU
            all_runoff = np.zeros((n_times, n_hrus))

            for hru_idx in range(n_hrus):
                hru_precip = jnp.array(precip[:, hru_idx])
                hru_temp = jnp.array(temp[:, hru_idx])
                hru_pet = jnp.array(pet[:, hru_idx])

                runoff, _ = model.simulate(
                    hru_precip, hru_temp, hru_pet,
                    params=params,
                    warmup_steps=self.warmup_days,
                )

                all_runoff[:, hru_idx] = np.array(runoff)

            # If routing enabled, run Muskingum-Cunge routing
            if self.enable_routing:
                self.logger.info("Routing not yet implemented for jFUSE distributed mode")
                # TODO: Implement routing when jFUSE supports CoupledModel

            # Save distributed results
            self._save_distributed_results(all_runoff, time_index, hru_ids)

            return True

        except Exception as e:
            self.logger.error(f"Error in distributed jFUSE execution: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_forcing(self) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """Load forcing data from preprocessed files."""
        # Try NetCDF first
        nc_file = self.jfuse_forcing_dir / f"{self.domain_name}_jfuse_forcing.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            forcing = {
                'precip': ds['precip'].values,
                'temp': ds['temp'].values,
                'pet': ds['pet'].values,
                'time': pd.to_datetime(ds.time.values),
            }
            ds.close()
        else:
            # Try CSV
            csv_file = self.jfuse_forcing_dir / f"{self.domain_name}_jfuse_forcing.csv"
            if not csv_file.exists():
                raise FileNotFoundError(f"No forcing file found at {nc_file} or {csv_file}")

            df = pd.read_csv(csv_file)
            forcing = {
                'precip': df['precip'].values,
                'temp': df['temp'].values,
                'pet': df['pet'].values,
                'time': pd.to_datetime(df['time']),
            }

        # Load observations if available
        obs_file = self.jfuse_forcing_dir / f"{self.domain_name}_observations.csv"
        if obs_file.exists():
            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            obs = obs_df.iloc[:, 0].values
        else:
            obs = None

        return forcing, obs

    def _save_lumped_results(self, runoff: np.ndarray, time_index: pd.DatetimeIndex) -> None:
        """Save lumped simulation results."""
        area_m2 = self._get_catchment_area()

        # Convert mm/day to m3/s
        streamflow_cms = runoff * area_m2 / (1000.0 * UnitConversion.SECONDS_PER_DAY)

        # Create DataFrame
        results_df = pd.DataFrame({
            'datetime': time_index,
            'streamflow_mm_day': runoff,
            'streamflow_cms': streamflow_cms,
        })

        # Save CSV
        csv_file = self.output_dir / f"{self.domain_name}_jfuse_output.csv"
        results_df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved lumped results to: {csv_file}")

        # Save NetCDF
        ds = xr.Dataset(
            data_vars={
                'streamflow': (['time'], streamflow_cms),
                'runoff': (['time'], runoff),
            },
            coords={
                'time': time_index,
            },
            attrs={
                'model': 'jFUSE',
                'model_config': self.model_config_name,
                'spatial_mode': 'lumped',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'catchment_area_m2': area_m2,
            }
        )
        ds['streamflow'].attrs = {'units': 'm3/s', 'long_name': 'Streamflow'}
        ds['runoff'].attrs = {'units': 'mm/day', 'long_name': 'Runoff depth'}

        nc_file = self.output_dir / f"{self.domain_name}_jfuse_output.nc"
        encoding = create_netcdf_encoding(ds, compression=True)
        ds.to_netcdf(nc_file, encoding=encoding)
        self.logger.info(f"Saved NetCDF output to: {nc_file}")

    def _save_distributed_results(
        self,
        runoff: np.ndarray,
        time_index: pd.DatetimeIndex,
        hru_ids: np.ndarray
    ) -> None:
        """Save distributed simulation results."""
        n_hrus = runoff.shape[1]

        # Create time coordinate in seconds since 1970
        time_seconds = (time_index - pd.Timestamp('1970-01-01')).total_seconds().values

        # Convert runoff from mm/day to m/s for routing
        runoff_ms = runoff / (1000.0 * UnitConversion.SECONDS_PER_DAY)

        # Create Dataset
        ds = xr.Dataset(
            data_vars={
                'gruId': (['gru'], hru_ids.astype(np.int32)),
                'runoff': (['time', 'gru'], runoff_ms),
            },
            coords={
                'time': ('time', time_seconds),
                'gru': ('gru', np.arange(n_hrus)),
            },
            attrs={
                'model': 'jFUSE',
                'model_config': self.model_config_name,
                'spatial_mode': 'distributed',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'n_hrus': n_hrus,
            }
        )

        ds['gruId'].attrs = {'long_name': 'ID of grouped response unit', 'units': '-'}
        ds['runoff'].attrs = {'long_name': 'jFUSE runoff', 'units': 'm/s'}
        ds.time.attrs = {'units': 'seconds since 1970-01-01 00:00:00', 'calendar': 'standard'}

        # Save
        output_file = self.output_dir / f"{self.domain_name}_{self.experiment_id}_runs_def.nc"
        encoding = create_netcdf_encoding(ds, compression=True, int_vars={'gruId': 'int32'})
        ds.to_netcdf(output_file, encoding=encoding)
        self.logger.info(f"Saved distributed results to: {output_file}")

    def _calculate_and_log_metrics(self) -> None:
        """Calculate and log performance metrics."""
        try:
            from symfluence.evaluation.metrics import kge, nse

            # Load simulation
            output_file = self.output_dir / f"{self.domain_name}_jfuse_output.nc"
            if output_file.exists():
                ds = xr.open_dataset(output_file)
                sim = ds['streamflow'].values
                sim_time = pd.to_datetime(ds.time.values)
                ds.close()
            else:
                csv_file = self.output_dir / f"{self.domain_name}_jfuse_output.csv"
                if not csv_file.exists():
                    self.logger.warning("No output file found for metrics calculation")
                    return
                df = pd.read_csv(csv_file)
                sim = df['streamflow_cms'].values
                sim_time = pd.to_datetime(df['datetime'])

            # Load observations
            obs_file = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
            if not obs_file.exists():
                self.logger.warning("Observations not found for metrics")
                return

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # Align time series
            sim_series = pd.Series(sim, index=sim_time)
            obs_series = obs_df.iloc[:, 0]

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
            self.logger.info(f"jFUSE Model Performance ({self.spatial_mode})")
            self.logger.info(f"   Model structure: {self.model_config_name}")
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
        if not HAS_JFUSE:
            raise ImportError("jFUSE not installed")

        from jfuse import nse_loss, kge_loss

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
        if not HAS_JAX or not HAS_JFUSE:
            self.logger.warning("JAX or jFUSE not available for gradient computation")
            return None

        # Load forcing
        forcing, obs = self._load_forcing()

        precip = jnp.array(forcing['precip'].flatten())
        temp = jnp.array(forcing['temp'].flatten())
        pet = jnp.array(forcing['pet'].flatten())

        if obs is None:
            self.logger.error("Observations required for gradient calibration")
            return None

        obs = jnp.array(obs)

        model_config = ModelConfig(
            model_name=self.model_config_name,
            enable_snow=self.enable_snow,
        )
        model = FUSEModel(model_config)

        def loss_fn(params):
            runoff, _ = model.simulate(precip, temp, pet, params=params, warmup_steps=self.warmup_days)
            if metric.lower() == 'nse':
                return -jfuse.nse(obs[self.warmup_days:], runoff)
            return -jfuse.kge(obs[self.warmup_days:], runoff)

        return jax.grad(loss_fn)

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
        if not HAS_JFUSE:
            self.logger.error("jFUSE not installed")
            return -999.0

        forcing, obs = self._load_forcing()

        if obs is None:
            self.logger.error("Observations required for evaluation")
            return -999.0

        precip = jnp.array(forcing['precip'].flatten())
        temp = jnp.array(forcing['temp'].flatten())
        pet = jnp.array(forcing['pet'].flatten())

        model_config = ModelConfig(
            model_name=self.model_config_name,
            enable_snow=self.enable_snow,
        )
        model = FUSEModel(model_config)

        runoff, _ = model.simulate(precip, temp, pet, params=params, warmup_steps=self.warmup_days)
        runoff = np.array(runoff)

        # Skip warmup for observations
        sim = runoff
        obs_arr = obs[self.warmup_days:] if len(obs) > self.warmup_days else obs

        # Align lengths
        min_len = min(len(sim), len(obs_arr))
        sim = sim[:min_len]
        obs_arr = obs_arr[:min_len]

        # Remove NaN
        valid_mask = ~(np.isnan(sim) | np.isnan(obs_arr))
        sim = sim[valid_mask]
        obs_arr = obs_arr[valid_mask]

        if len(sim) < 10:
            return -999.0

        from symfluence.evaluation.metrics import kge, nse

        if metric.lower() == 'nse':
            return float(nse(obs_arr, sim, transfo=1))
        return float(kge(obs_arr, sim, transfo=1))
