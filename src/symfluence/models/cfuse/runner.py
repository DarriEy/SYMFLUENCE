"""
cFUSE Model Runner.

Handles cFUSE model execution, state management, and output processing.
Supports both lumped and distributed spatial modes with optional routing.

cFUSE is a PyTorch/Enzyme AD implementation of the FUSE (Framework for Understanding
Structural Errors) model, enabling automatic differentiation for gradient-based calibration.
"""

from typing import Dict, Any, Optional, Tuple
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

# Lazy cFUSE and PyTorch import
try:
    import cfuse
    from cfuse import (
        PARAM_BOUNDS, DEFAULT_PARAMS, PARAM_NAMES,
        VIC_CONFIG, TOPMODEL_CONFIG, PRMS_CONFIG, SACRAMENTO_CONFIG, ARNO_CONFIG
    )
    HAS_CFUSE = True
except ImportError:
    HAS_CFUSE = False
    cfuse = None
    PARAM_BOUNDS = {}
    DEFAULT_PARAMS = {}
    PARAM_NAMES = []

try:
    import cfuse_core
    HAS_CFUSE_CORE = True
except ImportError:
    HAS_CFUSE_CORE = False
    cfuse_core = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def _get_model_config(structure: str) -> dict:
    """Get model configuration dictionary for a structure."""
    if not HAS_CFUSE:
        return {}

    configs = {
        'vic': VIC_CONFIG,
        'topmodel': TOPMODEL_CONFIG,
        'prms': PRMS_CONFIG,
        'sacramento': SACRAMENTO_CONFIG,
        'arno': ARNO_CONFIG,
    }

    structure_lower = structure.lower()
    if structure_lower in configs:
        return configs[structure_lower].to_dict()
    return PRMS_CONFIG.to_dict()  # Default to PRMS for better gradient support


@ModelRegistry.register_runner('CFUSE', method_name='run_cfuse')
class CFUSERunner(BaseModelRunner, UnifiedModelExecutor):
    """
    Runner class for the cFUSE hydrological model.

    Supports:
    - Lumped mode (single catchment simulation)
    - Distributed mode (per-HRU simulation with optional routing)
    - PyTorch backend for autodiff/gradient computation
    - Multiple model structures (PRMS, Sacramento, TOPMODEL, VIC, ARNO)

    Attributes:
        config: Configuration dictionary or SymfluenceConfig object
        logger: Logger instance
        spatial_mode: 'lumped' or 'distributed'
        model_structure: cFUSE model structure name
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None,
        settings_dir: Optional[Path] = None
    ):
        """
        Initialize cFUSE runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance
            reporting_manager: Optional reporting manager for visualization
            settings_dir: Optional override for settings directory
        """
        self.settings_dir = Path(settings_dir) if settings_dir else None

        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Check cFUSE availability
        if not HAS_CFUSE_CORE:
            self.logger.warning("cFUSE core not installed. Install from: https://github.com/DarriEy/cFUSE")

        # Instance variables for external parameters during calibration
        self._external_params: Optional[Dict[str, float]] = None

        # Determine spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.cfuse.spatial_mode if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            'auto'
        )

        if configured_mode in (None, 'auto', 'default'):
            if self.domain_definition_method == 'delineate':
                self.spatial_mode = 'distributed'
            else:
                self.spatial_mode = 'lumped'
        else:
            self.spatial_mode = configured_mode

        # Model structure configuration - default to prms for better gradient support
        self.model_structure = self._get_config_value(
            lambda: self.config.model.cfuse.model_structure if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            'prms'
        )

        # Get model configuration dictionary (use _model_config to avoid conflict
        # with the read-only config_dict property from ConfigMixin)
        self._model_config = _get_model_config(self.model_structure)

        # Snow configuration
        self.enable_snow = self._get_config_value(
            lambda: self.config.model.cfuse.enable_snow if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            True
        )
        if self.enable_snow:
            self._model_config['enable_snow'] = True

        # Routing configuration
        self.enable_routing = self._get_config_value(
            lambda: self.config.model.cfuse.enable_routing if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            False
        )

        # Timestep configuration
        self.timestep_days = self._get_config_value(
            lambda: self.config.model.cfuse.timestep_days if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            1.0
        )

        # Initial state configuration
        self.warmup_days = self._get_config_value(
            lambda: self.config.model.cfuse.warmup_days if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            365
        )

        self.initial_s1 = self._get_config_value(
            lambda: self.config.model.cfuse.initial_s1 if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            50.0
        )

        self.initial_s2 = self._get_config_value(
            lambda: self.config.model.cfuse.initial_s2 if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            200.0
        )

        self.initial_snow = self._get_config_value(
            lambda: self.config.model.cfuse.initial_snow if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            0.0
        )

        # Get number of states
        self._n_states = 10  # Default
        if HAS_CFUSE_CORE:
            try:
                self._n_states = cfuse_core.get_num_active_states(self._model_config)
            except Exception as e:
                self.logger.debug(f"Could not get state count from cfuse_core, using default: {e}")

    def _get_model_name(self) -> str:
        """Return model name for cFUSE."""
        return "CFUSE"

    def _setup_model_specific_paths(self) -> None:
        """Set up cFUSE-specific paths."""
        if hasattr(self, 'settings_dir') and self.settings_dir:
            self.cfuse_setup_dir = self.settings_dir
        else:
            self.cfuse_setup_dir = self.project_dir / "settings" / "CFUSE"

        self.cfuse_forcing_dir = self.project_dir / 'forcing' / 'CFUSE_input'

    def _get_output_dir(self) -> Path:
        """cFUSE output directory."""
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
        """Get default cFUSE parameters."""
        if HAS_CFUSE:
            return DEFAULT_PARAMS.copy()
        return {}

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array in correct order."""
        if not HAS_CFUSE:
            return np.array(list(params.values()), dtype=np.float32)

        full_params = DEFAULT_PARAMS.copy()
        full_params.update(params)
        return np.array([full_params.get(name, 0.0) for name in PARAM_NAMES], dtype=np.float32)

    def _get_initial_states(self, n_hrus: int = 1) -> np.ndarray:
        """Get initial state array."""
        states = np.zeros((n_hrus, self._n_states), dtype=np.float32)
        states[:, 0] = self.initial_s1
        if self._n_states > 1:
            states[:, 1] = 20.0  # Upper free storage
        if self._n_states > 2:
            states[:, 2] = self.initial_s2
        return states

    def run_cfuse(self, params: Optional[Dict[str, float]] = None) -> Optional[Path]:
        """
        Run the cFUSE model.

        Args:
            params: Optional parameter dictionary. If provided, uses these
                    instead of defaults. Used during calibration.

        Returns:
            Path to output directory if successful, None otherwise.
        """
        # Emit experimental warning on first use
        from symfluence.models.cfuse import _warn_experimental
        _warn_experimental()

        if not HAS_CFUSE_CORE:
            self.logger.error("cFUSE core not installed. Cannot run model.")
            return None

        self.logger.info(f"Starting cFUSE model run in {self.spatial_mode} mode (structure: {self.model_structure})")

        # Store provided parameters
        if params:
            self.logger.info(f"Using external parameters: {params}")
            self._external_params = params

        with symfluence_error_handler(
            "cFUSE model execution",
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
                self.logger.info("cFUSE model run completed successfully")
                self._calculate_and_log_metrics()
                return self.output_dir
            else:
                self.logger.error("cFUSE model run failed")
                return None

    def _execute_lumped(self) -> bool:
        """Execute cFUSE in lumped mode."""
        self.logger.info("Running lumped cFUSE simulation")

        try:
            # Load forcing data
            forcing, obs = self._load_forcing()

            precip = forcing['precip'].flatten()
            temp = forcing['temp'].flatten()
            pet = forcing['pet'].flatten()
            time_index = forcing['time']

            n_times = len(precip)
            self.logger.info(f"Running simulation for {n_times} timesteps")

            # Get parameters
            params = self._external_params if self._external_params else self._get_default_params()
            params_array = self._params_to_array(params)

            # Prepare forcing in batch format: [time, hru, 3]
            forcing_array = np.stack([precip, pet, temp], axis=-1)[:, np.newaxis, :].astype(np.float32)

            # Initial states: [hru, states]
            initial_states = self._get_initial_states(n_hrus=1)

            # Run simulation
            final_states, runoff = cfuse_core.run_fuse_batch(
                initial_states,
                forcing_array,
                params_array,
                self._model_config,
                float(self.timestep_days)
            )

            # Handle warmup
            if self.warmup_days > 0 and len(runoff) > self.warmup_days:
                runoff = runoff[self.warmup_days:]
                time_index = time_index[self.warmup_days:]

            # Flatten output
            runoff = runoff.flatten()

            # Save results
            self._save_lumped_results(runoff, time_index)

            return True

        except Exception as e:
            self.logger.error(f"Error in lumped cFUSE execution: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _execute_distributed(self) -> bool:
        """Execute cFUSE in distributed mode (per-HRU batch)."""
        self.logger.info("Running distributed cFUSE simulation")

        try:
            # Load distributed forcing
            forcing_file = self.cfuse_forcing_dir / f"{self.domain_name}_cfuse_forcing_distributed.nc"
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
            params_array = self._params_to_array(params)

            # Prepare forcing in batch format: [time, hru, 3]
            forcing_array = np.stack([precip, pet, temp], axis=-1).astype(np.float32)

            # Initial states: [hru, states]
            initial_states = self._get_initial_states(n_hrus=n_hrus)

            # Run batch simulation (all HRUs at once)
            final_states, runoff = cfuse_core.run_fuse_batch(
                initial_states,
                forcing_array,
                params_array,
                self._model_config,
                float(self.timestep_days)
            )

            # Handle warmup
            if self.warmup_days > 0 and len(runoff) > self.warmup_days:
                runoff = runoff[self.warmup_days:]
                time_index = time_index[self.warmup_days:]

            # Save distributed results
            self._save_distributed_results(runoff, time_index, hru_ids)

            return True

        except Exception as e:
            self.logger.error(f"Error in distributed cFUSE execution: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_forcing(self) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """Load forcing data from preprocessed files."""
        # Try NetCDF first
        nc_file = self.cfuse_forcing_dir / f"{self.domain_name}_cfuse_forcing.nc"
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
            csv_file = self.cfuse_forcing_dir / f"{self.domain_name}_cfuse_forcing.csv"
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
        obs_file = self.cfuse_forcing_dir / f"{self.domain_name}_observations.csv"
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
        csv_file = self.output_dir / f"{self.domain_name}_cfuse_output.csv"
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
                'model': 'cFUSE',
                'model_structure': self.model_structure,
                'spatial_mode': 'lumped',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'catchment_area_m2': area_m2,
            }
        )
        ds['streamflow'].attrs = {'units': 'm3/s', 'long_name': 'Streamflow'}
        ds['runoff'].attrs = {'units': 'mm/day', 'long_name': 'Runoff depth'}

        nc_file = self.output_dir / f"{self.domain_name}_cfuse_output.nc"
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
                'model': 'cFUSE',
                'model_structure': self.model_structure,
                'spatial_mode': 'distributed',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'n_hrus': n_hrus,
            }
        )

        ds['gruId'].attrs = {'long_name': 'ID of grouped response unit', 'units': '-'}
        ds['runoff'].attrs = {'long_name': 'cFUSE runoff', 'units': 'm/s'}
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
            output_file = self.output_dir / f"{self.domain_name}_cfuse_output.nc"
            if output_file.exists():
                ds = xr.open_dataset(output_file)
                sim = ds['streamflow'].values
                sim_time = pd.to_datetime(ds.time.values)
                ds.close()
            else:
                csv_file = self.output_dir / f"{self.domain_name}_cfuse_output.csv"
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
            self.logger.info(f"cFUSE Model Performance ({self.spatial_mode})")
            self.logger.info(f"   Model structure: {self.model_structure}")
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
        if not HAS_CFUSE_CORE:
            self.logger.error("cFUSE core not installed")
            return -999.0

        forcing, obs = self._load_forcing()

        if obs is None:
            self.logger.error("Observations required for evaluation")
            return -999.0

        try:
            precip = forcing['precip'].flatten()
            temp = forcing['temp'].flatten()
            pet = forcing['pet'].flatten()

            params_array = self._params_to_array(params)
            forcing_array = np.stack([precip, pet, temp], axis=-1)[:, np.newaxis, :].astype(np.float32)
            initial_states = self._get_initial_states(n_hrus=1)

            _, runoff = cfuse_core.run_fuse_batch(
                initial_states,
                forcing_array,
                params_array,
                self._model_config,
                float(self.timestep_days)
            )

            # Handle warmup
            if self.warmup_days > 0 and len(runoff) > self.warmup_days:
                runoff = runoff[self.warmup_days:]
                obs = obs[self.warmup_days:] if len(obs) > self.warmup_days else obs

            runoff = runoff.flatten()

            # Align lengths
            min_len = min(len(runoff), len(obs))
            sim = runoff[:min_len]
            obs_arr = obs[:min_len]

            # Remove NaN
            valid_mask = ~(np.isnan(sim) | np.isnan(obs_arr))
            sim = sim[valid_mask]
            obs_arr = obs_arr[valid_mask]

            if len(sim) < 10:
                return -999.0

            from symfluence.evaluation.metrics import kge as calc_kge, nse as calc_nse

            if metric.lower() == 'nse':
                return float(calc_nse(obs_arr, sim, transfo=1))
            return float(calc_kge(obs_arr, sim, transfo=1))

        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return -999.0
