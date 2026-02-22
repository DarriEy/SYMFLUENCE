"""
HBV Calibration Worker.

Worker implementation for HBV-96 model optimization with support for
both evolutionary and gradient-based calibration.

Refactored to use InMemoryModelWorker base class for common functionality.
"""

import os
import sys
import signal
import random
import time
import traceback
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.optimization.workers.inmemory_worker import InMemoryModelWorker, HAS_JAX
from symfluence.optimization.workers.base_worker import WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.core.constants import ModelDefaults
from symfluence.models.hbv.time_utils import warmup_timesteps
from symfluence.core.mixins.project import resolve_data_subdir

# Lazy JAX import
if HAS_JAX:
    import jax
    import jax.numpy as jnp


@OptimizerRegistry.register_worker('HBV')
class HBVWorker(InMemoryModelWorker):
    """Worker for HBV-96 model calibration.

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
        """Initialize HBV worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Model-specific components
        self._simulate_fn = None
        self._use_jax = HAS_JAX

        # Timestep configuration (hours) - maps to config.model.hbv.timestep_hours
        self.timestep_hours = 24  # Default to daily
        if config:
            self.timestep_hours = int(self._cfg('HBV_TIMESTEP_HOURS', self._cfg('TIMESTEP_HOURS', 24)))

    # =========================================================================
    # InMemoryModelWorker Abstract Method Implementations
    # =========================================================================

    def _get_model_name(self) -> str:
        """Return the model identifier."""
        return 'HBV'

    def _get_forcing_subdir(self) -> str:
        """Return the forcing subdirectory name."""
        return 'HBV_input'

    def _get_forcing_variable_map(self) -> Dict[str, str]:
        """Return mapping from standard names to HBV variable names."""
        return {
            'precip': 'pr',
            'temp': 'temp',
            'pet': 'pet',
        }

    def _load_forcing(self, task=None) -> bool:
        """Load forcing data with timestep-aware file selection.

        For hourly simulations (timestep_hours < 24), looks for files with
        the timestep suffix (e.g., _hbv_forcing_1h.nc for hourly).

        Returns:
            True if loading successful
        """
        if self._forcing is not None:
            return True

        try:
            import xarray as xr
        except ImportError:
            self.logger.error("xarray required for loading forcing")
            return False

        forcing_dir = self._get_forcing_dir(task)
        domain_name = self._get_domain_name()
        var_map = self._get_forcing_variable_map()

        # Build list of files to try, prioritizing timestep-specific files
        nc_patterns = []
        if self.timestep_hours < 24:
            # For sub-daily, try timestep-specific file first
            nc_patterns.append(forcing_dir / f"{domain_name}_hbv_forcing_{self.timestep_hours}h.nc")
        nc_patterns.extend([
            forcing_dir / f"{domain_name}_hbv_forcing_{24}h.nc",  # Daily fallback
            forcing_dir / f"{domain_name}_hbv_forcing.nc",
            forcing_dir / f"{domain_name}_forcing.nc",
        ])

        for nc_file in nc_patterns:
            if nc_file.exists():
                try:
                    ds = xr.open_dataset(nc_file)
                    self._forcing = {}

                    for std_name, var_name in var_map.items():
                        if var_name in ds.variables:
                            self._forcing[std_name] = ds[var_name].values.flatten()
                        elif std_name in ds.variables:
                            self._forcing[std_name] = ds[std_name].values.flatten()
                        else:
                            self.logger.warning(f"Variable {var_name} not found in {nc_file}")
                            continue

                    if 'time' in ds.coords:
                        import pandas as pd
                        self._time_index = pd.to_datetime(ds.time.values)

                    ds.close()

                    if len(self._forcing) >= 3:
                        self.logger.info(
                            f"Loaded forcing from {nc_file.name}: "
                            f"{len(self._forcing['precip'])} timesteps"
                        )
                        return True
                except (OSError, RuntimeError, KeyError) as e:
                    self.logger.warning(f"Error loading {nc_file}: {e}")

        self.logger.error(f"No forcing file found in {forcing_dir}")
        return False

    def _load_observations(self, task=None) -> bool:
        """Load observations with timestep-aware resolution handling.

        For hourly simulations, keeps observations at hourly resolution.
        For daily simulations, resamples to daily.

        Returns:
            True if loading successful
        """
        if self._observations is not None:
            return True

        import pandas as pd
        from pathlib import Path

        domain_name = self._get_domain_name()
        data_dir = Path(self._get_config_value(lambda: str(self.config.system.data_dir), default='.'))
        project_dir = data_dir / f"domain_{domain_name}"

        obs_patterns = [
            resolve_data_subdir(project_dir, 'observations') / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv",
        ]

        for obs_file in obs_patterns:
            if obs_file.exists():
                try:
                    obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

                    if not isinstance(obs_df.index, pd.DatetimeIndex):
                        obs_df.index = pd.to_datetime(obs_df.index)

                    obs_cms = obs_df.iloc[:, 0]

                    # Resample based on model timestep
                    if len(obs_cms) > 1:
                        time_diff = obs_cms.index[1] - obs_cms.index[0]
                        target_freq = f'{self.timestep_hours}h' if self.timestep_hours < 24 else 'D'

                        if self.timestep_hours == 24:
                            # Daily model - resample obs to daily
                            if time_diff < pd.Timedelta(days=1):
                                self.logger.info(f"Resampling observations from {time_diff} to daily")
                                obs_cms = obs_cms.resample('D').mean().dropna()
                        else:
                            # Sub-daily model - resample to model timestep
                            target_timedelta = pd.Timedelta(hours=self.timestep_hours)
                            if time_diff != target_timedelta:
                                self.logger.info(f"Resampling observations from {time_diff} to {target_freq}")
                                obs_cms = obs_cms.resample(target_freq).mean().dropna()

                    # Convert m³/s to mm/timestep
                    area_km2 = self.get_catchment_area()
                    # mm/timestep = m³/s × seconds_per_timestep / (area_m² × 0.001)
                    seconds_per_timestep = self.timestep_hours * 3600
                    conversion_factor = seconds_per_timestep / (area_km2 * 1e6 * 0.001)
                    obs_mm = obs_cms * conversion_factor

                    # Align with forcing time if available
                    if self._time_index is not None:
                        obs_aligned = obs_mm.reindex(self._time_index)
                        n_valid = (~obs_aligned.isna()).sum()
                        self.logger.info(f"Aligned observations: {n_valid}/{len(self._time_index)} valid")
                        self._observations = obs_aligned.values
                    else:
                        self._observations = obs_mm.values

                    self.logger.debug(f"Loaded observations from {obs_file}")
                    return True

                except (FileNotFoundError, ValueError, KeyError) as e:
                    self.logger.warning(f"Error loading {obs_file}: {e}")

        self.logger.warning("No observation file found")
        return False

    def calculate_streamflow_metrics(
        self,
        sim_mm: np.ndarray,
        obs_mm: np.ndarray,
        skip_warmup: bool = True,
        time_index: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """Calculate streamflow metrics with timestep-aware warmup handling.

        Args:
            sim_mm: Simulated runoff in mm/timestep
            obs_mm: Observed runoff in mm/timestep
            skip_warmup: Whether to skip warmup period
            time_index: Optional time index for period filtering

        Returns:
            Dictionary with 'kge', 'nse', 'n_points'
        """
        from symfluence.evaluation.metrics import kge, nse

        try:
            # Get time index for filtering
            if time_index is None:
                time_index = self._time_index

            # Convert warmup days to timesteps
            warmup_steps = warmup_timesteps(self.warmup_days, self.timestep_hours)

            # Skip warmup if requested
            if skip_warmup:
                sim = sim_mm[warmup_steps:]
                obs = obs_mm[warmup_steps:]
                if time_index is not None and len(time_index) > warmup_steps:
                    time_index = time_index[warmup_steps:]
            else:
                sim = sim_mm
                obs = obs_mm

            # Align lengths
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs = obs[:min_len]
            if time_index is not None and len(time_index) > min_len:
                time_index = time_index[:min_len]

            # Filter to calibration period if specified
            cal_period = self._cfg('CALIBRATION_PERIOD', self._cfg('EXPERIMENT_CALIBRATION_PERIOD', ''))
            if cal_period and time_index is not None:
                try:
                    dates = [d.strip() for d in cal_period.split(',')]
                    if len(dates) >= 2:
                        start_date = pd.Timestamp(dates[0])
                        end_date = pd.Timestamp(dates[1])

                        # Ensure time_index is DatetimeIndex
                        if not isinstance(time_index, pd.DatetimeIndex):
                            time_index = pd.DatetimeIndex(time_index)

                        # Create mask for calibration period
                        cal_mask = (time_index >= start_date) & (time_index <= end_date)
                        sim = sim[cal_mask]
                        obs = obs[cal_mask]
                        self.logger.debug(
                            f"Filtered to calibration period {start_date} - {end_date}: "
                            f"{cal_mask.sum()} points"
                        )
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Could not filter to calibration period: {e}")

            # Remove NaN
            valid_mask = ~(np.isnan(sim) | np.isnan(obs))
            sim = sim[valid_mask]
            obs = obs[valid_mask]

            if len(sim) < 10:
                return {
                    'kge': self.penalty_score,
                    'nse': self.penalty_score,
                    'n_points': len(sim),
                    'error': 'Insufficient valid data points'
                }

            # Calculate metrics
            kge_val = float(kge(obs, sim, transfo=1))
            nse_val = float(nse(obs, sim, transfo=1))

            if np.isnan(kge_val):
                kge_val = self.penalty_score
            if np.isnan(nse_val):
                nse_val = self.penalty_score

            return {
                'kge': kge_val,
                'nse': nse_val,
                'n_points': len(sim),
                'mean_sim': float(np.mean(sim)),
                'mean_obs': float(np.mean(obs)),
            }

        except (ValueError, ZeroDivisionError, KeyError) as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {
                'kge': self.penalty_score,
                'nse': self.penalty_score,
                'error': str(e)
            }

    def _run_simulation(
        self,
        forcing: Dict[str, np.ndarray],
        params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """Run HBV model simulation.

        Args:
            forcing: Dictionary with 'precip', 'temp', 'pet' arrays
            params: Parameter dictionary
            **kwargs: Additional arguments

        Returns:
            Runoff array in mm/timestep

        Raises:
            ValueError: If recession ordering constraint k0 > k1 > k2 is violated.
        """
        # Enforce recession coefficient ordering: k0 > k1 > k2
        # This is a physical constraint — fast flow must recede faster than
        # interflow, which must recede faster than baseflow.
        k0 = params.get('k0', 0.3)
        k1 = params.get('k1', 0.1)
        k2 = params.get('k2', 0.01)
        if not (k0 > k1 > k2):
            raise ValueError(
                f"Recession ordering violated: k0={k0:.4f}, k1={k1:.4f}, k2={k2:.4f} "
                f"(require k0 > k1 > k2)"
            )

        if not self._ensure_simulate_fn():
            raise RuntimeError("HBV simulation function not available")

        from symfluence.models.hbv.model import create_initial_state

        # Inject smoothing config into parameters
        if self.config:
            # Check for boolean flag (handles "True", "true", True, 1)
            smoothing_enabled = self._cfg('HBV_SMOOTHING', self._cfg('SMOOTHING', False))
            if isinstance(smoothing_enabled, str):
                smoothing_enabled = smoothing_enabled.lower() in ('true', '1', 'yes', 'on')

            params['smoothing_enabled'] = bool(smoothing_enabled)

            # Check for custom smoothing factor
            smoothing_factor = self._cfg('HBV_SMOOTHING_FACTOR', self._cfg('SMOOTHING_FACTOR'))
            if smoothing_factor is not None:
                try:
                    params['smoothing'] = float(smoothing_factor)
                except (ValueError, TypeError) as e:
                    self.logger.debug(
                        f"Could not parse HBV_SMOOTHING_FACTOR "
                        f"'{smoothing_factor}': {e}"
                    )

        precip = forcing['precip']
        temp = forcing['temp']
        pet = forcing['pet']

        if self._use_jax:
            precip = jnp.array(precip)
            temp = jnp.array(temp)
            pet = jnp.array(pet)

        initial_state = create_initial_state(
            use_jax=self._use_jax,
            timestep_hours=self.timestep_hours
        )

        assert self._simulate_fn is not None
        runoff, _ = self._simulate_fn(
            precip, temp, pet,
            params=params,
            initial_state=initial_state,
            warmup_days=self.warmup_days,
            use_jax=self._use_jax,
            timestep_hours=self.timestep_hours
        )

        if self._use_jax:
            return np.array(runoff)
        return runoff

    # =========================================================================
    # Model-Specific Methods
    # =========================================================================

    def _ensure_simulate_fn(self) -> bool:
        """Ensure simulation function is loaded.

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

    def _initialize_model(self) -> bool:
        """Initialize HBV model components."""
        return self._ensure_simulate_fn()

    # =========================================================================
    # Native Gradient Support (JAX autodiff)
    # =========================================================================

    def supports_native_gradients(self) -> bool:
        """Check if native gradient computation is available.

        HBV supports native gradients via JAX autodiff when JAX is installed.

        Returns:
            True if JAX is available.
        """
        return HAS_JAX and self._use_jax

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """Compute gradient of loss with respect to parameters.

        Uses JAX autodiff for efficient gradient computation.

        Args:
            params: Current parameter values
            metric: Metric to compute gradient for ('kge' or 'nse')

        Returns:
            Dictionary of parameter gradients, or None if JAX unavailable.
        """
        if not HAS_JAX or not self._use_jax:
            return None

        if not self._initialized:
            if not self.initialize():
                return None

        try:
            from symfluence.models.hbv.model import kge_loss, nse_loss

            assert self._forcing is not None
            precip = jnp.array(self._forcing['precip'])
            temp = jnp.array(self._forcing['temp'])
            pet = jnp.array(self._forcing['pet'])
            obs = jnp.array(self._observations)
            timestep_hours = self.timestep_hours

            def loss_fn(params_array, param_names):
                params_dict = dict(zip(param_names, params_array))
                if metric.lower() == 'nse':
                    return nse_loss(params_dict, precip, temp, pet, obs,
                                   self.warmup_days, use_jax=True,
                                   timestep_hours=timestep_hours)
                return kge_loss(params_dict, precip, temp, pet, obs,
                               self.warmup_days, use_jax=True,
                               timestep_hours=timestep_hours)

            grad_fn = jax.grad(loss_fn)
            param_names = list(params.keys())
            param_values = jnp.array([params[k] for k in param_names])
            grad_values = grad_fn(param_values, param_names)

            return dict(zip(param_names, np.array(grad_values)))

        except Exception as e:
            self.logger.error(f"Error computing gradient: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """Evaluate loss and compute gradient in single pass.

        Uses JAX value_and_grad for efficient computation.

        Args:
            params: Parameter values
            metric: Metric to evaluate

        Returns:
            Tuple of (loss_value, gradient_dict)
        """
        if not HAS_JAX or not self._use_jax:
            loss = self._evaluate_loss(params, metric)
            return loss, None

        if not self._initialized:
            if not self.initialize():
                return self.penalty_score, None

        try:
            from symfluence.models.hbv.model import kge_loss, nse_loss

            assert self._forcing is not None
            precip = jnp.array(self._forcing['precip'])
            temp = jnp.array(self._forcing['temp'])
            pet = jnp.array(self._forcing['pet'])
            obs = jnp.array(self._observations)
            timestep_hours = self.timestep_hours

            def loss_fn(params_array, param_names):
                params_dict = dict(zip(param_names, params_array))
                if metric.lower() == 'nse':
                    return nse_loss(params_dict, precip, temp, pet, obs,
                                   self.warmup_days, use_jax=True,
                                   timestep_hours=timestep_hours)
                return kge_loss(params_dict, precip, temp, pet, obs,
                               self.warmup_days, use_jax=True,
                               timestep_hours=timestep_hours)

            value_and_grad_fn = jax.value_and_grad(loss_fn)
            param_names = list(params.keys())
            param_values = jnp.array([params[k] for k in param_names])
            loss_val, grad_values = value_and_grad_fn(param_values, param_names)

            gradient = dict(zip(param_names, np.array(grad_values)))
            return float(loss_val), gradient

        except Exception as e:
            self.logger.error(f"Error in evaluate_with_gradient: {e}")
            return self.penalty_score, None

    # =========================================================================
    # Static Worker Function for Process Pool
    # =========================================================================

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_hbv_parameters_worker(task_data)


def _evaluate_hbv_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution.

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

    # Small random delay to prevent process contention
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
