"""
HBV Calibration Worker.

Worker implementation for HBV-96 model optimization with support for
both evolutionary and gradient-based calibration.
"""

import os
import sys
import signal
import random
import time
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.metrics import kge, nse
from symfluence.core.constants import ModelDefaults

# Lazy JAX import
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


@OptimizerRegistry.register_worker('HBV')
class HBVWorker(BaseWorker):
    """
    Worker for HBV-96 model calibration.

    Supports:
    - Standard evolutionary optimization (evaluate -> apply -> run -> metrics)
    - Gradient-based optimization with JAX autodiff
    - Efficient in-memory simulation (no file I/O during calibration)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize HBV worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Lazy-loaded model components
        self._forcing = None
        self._observations = None
        self._simulate_fn = None
        self._use_jax = HAS_JAX

        # Configuration
        self.warmup_days = 365
        if config:
            self.warmup_days = config.get('HBV_WARMUP_DAYS', 365)

        # Catchment area for unit conversion
        self._catchment_area = None

    def supports_native_gradients(self) -> bool:
        """
        Check if native gradient computation is available.

        HBV supports native gradients via JAX autodiff when JAX is installed.
        This enables ~15x faster gradient computation compared to finite differences.

        Returns:
            True if JAX is available and gradients can be computed via autodiff.
        """
        return HAS_JAX

    def _get_catchment_area(self, config: Dict[str, Any], project_dir: Path) -> float:
        """Get total catchment area in m² for unit conversion."""
        # Try to get from shapefile
        try:
            import geopandas as gpd
            domain_name = config.get('DOMAIN_NAME', '')
            discretization = config.get('SUB_GRID_DISCRETIZATION', 'GRUs')
            catchment_path = project_dir / 'shapefiles' / 'catchment' / f"{domain_name}_HRUs_{discretization}.shp"
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                area_cols = [c for c in gdf.columns if 'area' in c.lower()]
                if area_cols:
                    total_area = gdf[area_cols[0]].sum()
                    self.logger.info(f"Catchment area from shapefile: {total_area/1e6:.2f} km²")
                    return float(total_area)
            else:
                self.logger.debug(f"Catchment shapefile not found at: {catchment_path}")
        except Exception as e:
            self.logger.debug(f"Could not read catchment area: {e}")

        # Fall back to config
        area_km2 = config.get('CATCHMENT_AREA_KM2')
        if area_km2:
            return area_km2 * 1e6

        # Default fallback
        self.logger.warning("Could not determine catchment area, using default 1000 km²")
        return 1000.0 * 1e6

    def _save_output_files(
        self,
        runoff: np.ndarray,
        time_index: pd.DatetimeIndex,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> None:
        """
        Save simulation output to files for final evaluation.

        Args:
            runoff: Runoff timeseries in mm/day
            time_index: Time coordinate
            output_dir: Directory to save output files
            config: Configuration dictionary
        """
        import xarray as xr
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        domain_name = config.get('DOMAIN_NAME', 'unknown')

        # Get catchment area for unit conversion
        project_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.')) / f"domain_{domain_name}"
        area_m2 = self._get_catchment_area(config, project_dir)

        # Convert mm/day to m³/s
        seconds_per_day = 86400.0
        streamflow_cms = runoff * area_m2 / (1000.0 * seconds_per_day)

        # Create DataFrame
        results_df = pd.DataFrame({
            'datetime': time_index,
            'streamflow_mm_day': runoff,
            'streamflow_cms': streamflow_cms,
        })

        # Save CSV
        csv_path = output_dir / f"{domain_name}_hbv_streamflow.csv"
        results_df.to_csv(csv_path, index=False)

        # Create NetCDF - use "streamflow" in filename so OutputFileLocator can find it
        ds = xr.Dataset(
            {
                'streamflow': (['time'], streamflow_cms.astype(np.float32)),
                'runoff': (['time'], runoff.astype(np.float32)),
            },
            coords={'time': time_index}
        )
        ds['streamflow'].attrs = {'units': 'm3/s', 'long_name': 'Streamflow'}
        ds['runoff'].attrs = {'units': 'mm/day', 'long_name': 'Runoff'}

        nc_path = output_dir / f"{domain_name}_hbv_streamflow.nc"
        ds.to_netcdf(nc_path)
        ds.close()

        self.logger.info(f"Saved HBV output to: {output_dir}")

    def _load_data(self) -> bool:
        """
        Load forcing and observation data for in-memory simulation.

        Returns:
            True if data loaded successfully.
        """
        if self._forcing is not None:
            return True  # Already loaded

        try:
            # Get paths from config
            if self.config is None:
                self.logger.error("Config not set for HBV worker")
                return False

            data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = self.config.get('DOMAIN_NAME', 'unknown')
            project_dir = data_dir / f"domain_{domain_name}"
            forcing_dir = project_dir / 'forcing' / 'HBV_input'

            # Load forcing
            nc_file = forcing_dir / f"{domain_name}_hbv_forcing.nc"
            if nc_file.exists():
                import xarray as xr
                ds = xr.open_dataset(nc_file)
                self._forcing = {
                    'precip': ds['pr'].values.flatten(),
                    'temp': ds['temp'].values.flatten(),
                    'pet': ds['pet'].values.flatten(),
                    'time': pd.to_datetime(ds.time.values),
                }
                ds.close()
            else:
                csv_file = forcing_dir / f"{domain_name}_hbv_forcing.csv"
                if not csv_file.exists():
                    self.logger.error(f"Forcing file not found: {nc_file} or {csv_file}")
                    return False

                df = pd.read_csv(csv_file)
                self._forcing = {
                    'precip': df['pr'].values,
                    'temp': df['temp'].values,
                    'pet': df['pet'].values,
                    'time': pd.to_datetime(df['time']),
                }

            # Load observations
            obs_file = project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv"
            if obs_file.exists():
                obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
                obs_cms = obs_df.iloc[:, 0]  # Observations in m³/s as Series

                # Get catchment area for unit conversion
                area_m2 = self._get_catchment_area(self.config, project_dir)
                self._catchment_area = area_m2

                # Get forcing time index
                forcing_time = self._forcing['time']

                # Determine temporal resolutions
                obs_freq = pd.infer_freq(obs_cms.index[:100])  # Check first 100 points
                forcing_freq = pd.infer_freq(forcing_time[:100]) if len(forcing_time) > 1 else None

                self.logger.debug(f"Observation frequency: {obs_freq}, Forcing frequency: {forcing_freq}")
                self.logger.debug(f"Observations: {len(obs_cms)} timesteps, Forcing: {len(forcing_time)} timesteps")

                # Resample observations to daily if needed (hourly -> daily mean)
                if obs_freq and ('H' in str(obs_freq) or 'h' in str(obs_freq) or 'T' in str(obs_freq)):
                    self.logger.info("Resampling hourly observations to daily mean for HBV")
                    obs_daily = obs_cms.resample('D').mean()
                else:
                    obs_daily = obs_cms

                # Convert observations from m³/s to mm/day
                # Q (mm/day) = Q (m³/s) * 86400 / area_m² * 1000
                seconds_per_day = 86400.0
                obs_mm_day = obs_daily * seconds_per_day / area_m2 * 1000.0

                # Align observations with forcing time
                # Normalize forcing dates to midnight for comparison
                forcing_dates = pd.to_datetime(forcing_time).normalize()

                # Reindex observations to match forcing dates
                obs_aligned = obs_mm_day.reindex(forcing_dates)

                # Log alignment statistics
                n_valid = (~obs_aligned.isna()).sum()
                self.logger.info(f"Aligned observations: {n_valid}/{len(forcing_dates)} valid timesteps")

                self._observations = obs_aligned.values

            else:
                self.logger.warning("Observations not found, calibration will fail")
                self._observations = None

            self.logger.info(f"Loaded HBV data: {len(self._forcing['precip'])} timesteps")
            return True

        except Exception as e:
            self.logger.error(f"Error loading HBV data: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _ensure_simulate_fn(self) -> bool:
        """
        Ensure simulation function is loaded.

        Returns:
            True if function is available.
        """
        if self._simulate_fn is not None:
            return True

        try:
            from symfluence.models.hbv.model import simulate, HAS_JAX as MODEL_HAS_JAX
            self._simulate_fn = simulate
            self._use_jax = MODEL_HAS_JAX and HAS_JAX
            return True
        except ImportError as e:
            self.logger.error(f"Failed to import HBV model: {e}")
            return False

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters (no-op for HBV as simulation is in-memory).

        HBV doesn't need to write parameter files - parameters are passed
        directly to the simulation function.

        Args:
            params: Parameter values
            settings_dir: Settings directory (unused for HBV)
            **kwargs: Additional arguments

        Returns:
            True (always succeeds for HBV)
        """
        # Store parameters for run_model
        self._current_params = params
        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run HBV model simulation.

        For calibration, runs in-memory simulation instead of full runner.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if model ran successfully.
        """
        try:
            # Load data if not already loaded
            if not self._load_data():
                return False

            if not self._ensure_simulate_fn():
                return False

            # Get parameters
            params = getattr(self, '_current_params', None)
            if params is None:
                self.logger.error("No parameters set for HBV run")
                return False

            # Run in-memory simulation
            from symfluence.models.hbv.model import create_initial_state

            precip = self._forcing['precip']
            temp = self._forcing['temp']
            pet = self._forcing['pet']

            if self._use_jax:
                precip = jnp.array(precip)
                temp = jnp.array(temp)
                pet = jnp.array(pet)

            initial_state = create_initial_state(use_jax=self._use_jax)

            runoff, _ = self._simulate_fn(
                precip, temp, pet,
                params=params,
                initial_state=initial_state,
                warmup_days=self.warmup_days,
                use_jax=self._use_jax
            )

            # Store results for metric calculation
            if self._use_jax:
                self._last_runoff = np.array(runoff)
            else:
                self._last_runoff = runoff

            # If output_dir is provided and save_output=True, save results
            save_output = kwargs.get('save_output', False)
            if save_output and output_dir:
                self._save_output_files(
                    self._last_runoff,
                    self._forcing['time'],
                    output_dir,
                    config
                )

            return True

        except Exception as e:
            self.logger.error(f"Error running HBV: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from HBV output.

        Uses in-memory results from run_model.

        Args:
            output_dir: Output directory (unused)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values.
        """
        try:
            # Get stored results
            runoff = getattr(self, '_last_runoff', None)
            if runoff is None:
                return {'kge': self.penalty_score, 'error': 'No simulation results'}

            if self._observations is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            # Skip warmup
            sim = runoff[self.warmup_days:]
            obs = self._observations[self.warmup_days:]

            # Align lengths
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs = obs[:min_len]

            # Remove NaN
            valid_mask = ~(np.isnan(sim) | np.isnan(obs))
            sim = sim[valid_mask]
            obs = obs[valid_mask]

            if len(sim) < 10:
                return {'kge': self.penalty_score, 'error': 'Insufficient data'}

            # Calculate metrics
            kge_val = kge(obs, sim, transfo=1)
            nse_val = nse(obs, sim, transfo=1)

            # Handle NaN
            if np.isnan(kge_val):
                kge_val = self.penalty_score
            if np.isnan(nse_val):
                nse_val = self.penalty_score

            return {
                'kge': float(kge_val),
                'nse': float(nse_val),
                'n_points': len(sim)
            }

        except Exception as e:
            self.logger.error(f"Error calculating HBV metrics: {e}")
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradient of loss with respect to parameters.

        Uses JAX autodiff for efficient gradient computation.

        Args:
            params: Current parameter values
            metric: Metric to compute gradient for ('kge' or 'nse')

        Returns:
            Dictionary of parameter gradients, or None if JAX unavailable.
        """
        if not HAS_JAX:
            self.logger.warning("JAX not available for gradient computation")
            return None

        if not self._load_data():
            return None

        try:
            from symfluence.models.hbv.model import (
                kge_loss, nse_loss
            )

            precip = jnp.array(self._forcing['precip'])
            temp = jnp.array(self._forcing['temp'])
            pet = jnp.array(self._forcing['pet'])
            obs = jnp.array(self._observations)

            # Define loss function for gradient
            def loss_fn(params_array, param_names):
                params_dict = dict(zip(param_names, params_array))
                if metric.lower() == 'nse':
                    return nse_loss(params_dict, precip, temp, pet, obs,
                                   self.warmup_days, use_jax=True)
                return kge_loss(params_dict, precip, temp, pet, obs,
                               self.warmup_days, use_jax=True)

            # Get gradient function
            grad_fn = jax.grad(loss_fn)

            # Convert params to array
            param_names = list(params.keys())
            param_values = jnp.array([params[k] for k in param_names])

            # Compute gradient
            grad_values = grad_fn(param_values, param_names)

            # Convert back to dict
            return dict(zip(param_names, np.array(grad_values)))

        except Exception as e:
            self.logger.error(f"Error computing gradient: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> tuple:
        """
        Evaluate loss and compute gradient in single pass (efficient).

        Args:
            params: Parameter values
            metric: Metric to evaluate

        Returns:
            Tuple of (loss_value, gradient_dict)
        """
        if not HAS_JAX:
            # Fallback to separate calls
            loss = self._evaluate_loss(params, metric)
            return loss, None

        if not self._load_data():
            return self.penalty_score, None

        try:
            from symfluence.models.hbv.model import kge_loss, nse_loss

            precip = jnp.array(self._forcing['precip'])
            temp = jnp.array(self._forcing['temp'])
            pet = jnp.array(self._forcing['pet'])
            obs = jnp.array(self._observations)

            def loss_fn(params_array, param_names):
                params_dict = dict(zip(param_names, params_array))
                if metric.lower() == 'nse':
                    return nse_loss(params_dict, precip, temp, pet, obs,
                                   self.warmup_days, use_jax=True)
                return kge_loss(params_dict, precip, temp, pet, obs,
                               self.warmup_days, use_jax=True)

            # Get value and gradient together
            value_and_grad_fn = jax.value_and_grad(loss_fn)

            param_names = list(params.keys())
            param_values = jnp.array([params[k] for k in param_names])

            loss_val, grad_values = value_and_grad_fn(param_values, param_names)

            gradient = dict(zip(param_names, np.array(grad_values)))
            return float(loss_val), gradient

        except Exception as e:
            self.logger.error(f"Error in evaluate_with_gradient: {e}")
            return self.penalty_score, None

    def _evaluate_loss(self, params: Dict[str, float], metric: str) -> float:
        """Helper to evaluate loss without gradient."""
        if not self._load_data():
            return self.penalty_score

        if not self._ensure_simulate_fn():
            return self.penalty_score

        try:
            from symfluence.models.hbv.model import create_initial_state

            precip = self._forcing['precip']
            temp = self._forcing['temp']
            pet = self._forcing['pet']

            if self._use_jax:
                precip = jnp.array(precip)
                temp = jnp.array(temp)
                pet = jnp.array(pet)

            initial_state = create_initial_state(use_jax=self._use_jax)

            runoff, _ = self._simulate_fn(
                precip, temp, pet,
                params=params,
                initial_state=initial_state,
                warmup_days=self.warmup_days,
                use_jax=self._use_jax
            )

            if self._use_jax:
                runoff = np.array(runoff)

            # Calculate metric
            sim = runoff[self.warmup_days:]
            obs = self._observations[self.warmup_days:]
            min_len = min(len(sim), len(obs))
            sim, obs = sim[:min_len], obs[:min_len]

            valid_mask = ~(np.isnan(sim) | np.isnan(obs))
            sim, obs = sim[valid_mask], obs[valid_mask]

            if metric.lower() == 'nse':
                val = nse(obs, sim, transfo=1)
            else:
                val = kge(obs, sim, transfo=1)

            # Return negative (for minimization)
            return -float(val) if not np.isnan(val) else self.penalty_score

        except Exception as e:
            self.logger.error(f"Error evaluating loss: {e}")
            return self.penalty_score

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_hbv_parameters_worker(task_data)


def _evaluate_hbv_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary containing params, config, etc.

    Returns:
        Result dictionary with score and metrics.
    """
    # Set up signal handler for clean termination
    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass  # Signal handling not available

    # Force single-threaded execution
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    })

    # Small random delay to prevent process contention when spawning parallel workers
    # Not used for security/cryptographic purposes - just jitter to reduce race conditions
    time.sleep(random.uniform(0.05, 0.2))  # nosec B311

    try:
        worker = HBVWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'HBV worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
