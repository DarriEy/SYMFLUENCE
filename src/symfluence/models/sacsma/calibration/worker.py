"""
SAC-SMA Calibration Worker.

Worker implementation for SAC-SMA + Snow-17 model optimization.
Uses InMemoryModelWorker base class for common functionality.
"""

import logging
import os
import random
import signal
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.constants import ModelDefaults
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.workers.base_worker import WorkerTask
from symfluence.optimization.workers.inmemory_worker import HAS_JAX, InMemoryModelWorker

# Lazy JAX import
if HAS_JAX:
    import jax
    import jax.numpy as jnp


@OptimizerRegistry.register_worker('SACSMA')
class SacSmaWorker(InMemoryModelWorker):
    """Worker for SAC-SMA + Snow-17 model calibration.

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
        super().__init__(config, logger)
        self._simulate_fn = None
        self._use_jax = HAS_JAX

        # Configuration
        self.latitude = 45.0
        self.si = 100.0
        self.snow_module = 'snow17'
        if config:
            self.latitude = float(config.get('SACSMA_LATITUDE', 45.0))
            self.si = float(config.get('SACSMA_SI', 100.0))
            self.snow_module = str(config.get('SACSMA_SNOW_MODULE', 'snow17'))

    def _get_model_name(self) -> str:
        return 'SACSMA'

    def _get_forcing_subdir(self) -> str:
        return 'SACSMA_input'

    def _get_forcing_variable_map(self) -> Dict[str, str]:
        var_map = {
            'precip': 'pr',
            'pet': 'pet',
        }
        if self.snow_module == 'snow17':
            var_map['temp'] = 'temp'
        return var_map

    def _load_forcing(self, task=None) -> bool:
        """Load SAC-SMA forcing data."""
        if self._forcing is not None:
            return True

        try:
            import xarray as xr
        except ImportError:
            self.logger.error("xarray required for loading forcing")
            return False

        forcing_dir = self._get_forcing_dir(task)
        domain_name = self._get_config_value(lambda: self.config.domain.name, default='domain', dict_key='DOMAIN_NAME')
        var_map = self._get_forcing_variable_map()

        nc_patterns = [
            forcing_dir / f"{domain_name}_sacsma_forcing.nc",
            forcing_dir / f"{domain_name}_forcing.nc",
        ]

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

                    if 'time' in ds.coords:
                        self._time_index = pd.to_datetime(ds.time.values)

                    ds.close()

                    if len(self._forcing) >= 3:
                        self.logger.info(
                            f"Loaded SAC-SMA forcing from {nc_file.name}: "
                            f"{len(self._forcing['precip'])} timesteps"
                        )
                        return True
                except (OSError, RuntimeError, KeyError) as e:
                    self.logger.warning(f"Error loading {nc_file}: {e}")

        self.logger.error(f"No SAC-SMA forcing file found in {forcing_dir}")
        return False

    def _load_observations(self, task=None) -> bool:
        """Load observations."""
        if self._observations is not None:
            return True

        from pathlib import Path

        domain_name = self._get_config_value(lambda: self.config.domain.name, default='domain', dict_key='DOMAIN_NAME')
        data_dir = Path(self._get_config_value(lambda: str(self.config.system.data_dir), default='.', dict_key='DATA_DIR'))
        project_dir = data_dir / f"domain_{domain_name}"

        obs_file = (resolve_data_subdir(project_dir, 'observations') / 'streamflow' / 'preprocessed' /
                    f"{domain_name}_streamflow_processed.csv")

        if obs_file.exists():
            try:
                obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
                if not isinstance(obs_df.index, pd.DatetimeIndex):
                    obs_df.index = pd.to_datetime(obs_df.index)

                obs_cms = obs_df.iloc[:, 0]

                # Resample to daily
                if len(obs_cms) > 1:
                    time_diff = obs_cms.index[1] - obs_cms.index[0]
                    if time_diff < pd.Timedelta(days=1):
                        obs_cms = obs_cms.resample('D').mean().dropna()

                # Convert m³/s to mm/day
                area_km2 = self.get_catchment_area()
                conversion_factor = 86400.0 / (area_km2 * 1e6 * 0.001)
                obs_mm = obs_cms * conversion_factor

                # Align with forcing time
                if self._time_index is not None:
                    obs_aligned = obs_mm.reindex(self._time_index)
                    self._observations = obs_aligned.values
                else:
                    self._observations = obs_mm.values

                return True
            except (FileNotFoundError, ValueError, KeyError) as e:
                self.logger.warning(f"Error loading observations: {e}")

        self.logger.warning("No observation file found for SAC-SMA")
        return False

    def _run_simulation(
        self,
        forcing: Dict[str, np.ndarray],
        params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """Run SAC-SMA + Snow-17 simulation.

        Returns runoff in mm/day.
        """
        if not self._ensure_simulate_fn():
            raise RuntimeError("SAC-SMA simulation function not available")

        precip = forcing['precip']
        temp = forcing['temp']
        pet = forcing['pet']

        # Create day_of_year from time index
        day_of_year = None
        if self._time_index is not None:
            day_of_year = self._time_index.dayofyear.values

        assert self._simulate_fn is not None

        # For standalone mode, temp may not be in forcing
        if self.snow_module == 'none' and 'temp' not in forcing:
            temp = np.zeros_like(precip)

        runoff, _ = self._simulate_fn(
            precip, temp, pet,
            params=params,
            day_of_year=day_of_year,
            warmup_days=self.warmup_days,
            latitude=self.latitude,
            si=self.si,
            use_jax=False,
            snow_module=self.snow_module,
        )

        return runoff

    def _ensure_simulate_fn(self) -> bool:
        """Ensure simulation function is loaded."""
        if self._simulate_fn is not None:
            return True

        try:
            from symfluence.models.sacsma.model import simulate
            self._simulate_fn = simulate
            return True
        except ImportError as e:
            self.logger.error(f"Failed to import SAC-SMA model: {e}")
            return False

    def _initialize_model(self) -> bool:
        return self._ensure_simulate_fn()

    # =========================================================================
    # Native Gradient Support (JAX autodiff)
    # =========================================================================

    def supports_native_gradients(self) -> bool:
        """Check if native gradient computation is available.

        SAC-SMA supports native gradients via JAX autodiff when JAX is
        installed. The coupled Snow-17 + SAC-SMA lax.scan simulation is
        fully differentiable end-to-end.

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

        Uses JAX autodiff through the coupled Snow-17 + SAC-SMA
        simulation for efficient gradient computation.

        Args:
            params: Current parameter values (Snow-17 + SAC-SMA combined).
            metric: Metric to compute gradient for ('kge' or 'nse').

        Returns:
            Dictionary of parameter gradients, or None if JAX unavailable.
        """
        if not HAS_JAX or not self._use_jax:
            return None

        if not self._initialized:
            if not self.initialize():
                return None

        try:
            from symfluence.models.sacsma.losses import kge_loss, nse_loss

            assert self._forcing is not None
            precip = jnp.array(self._forcing['precip'])
            temp = jnp.array(self._forcing['temp'])
            pet = jnp.array(self._forcing['pet'])
            obs = jnp.array(self._observations)

            # Day of year from time index
            day_of_year = None
            if self._time_index is not None:
                day_of_year = jnp.array(self._time_index.dayofyear.values,
                                        dtype=float)

            latitude = self.latitude
            si = self.si
            snow_module = self.snow_module
            warmup = self.warmup_days

            def loss_fn(params_array, param_names):
                params_dict = dict(zip(param_names, params_array))
                if metric.lower() == 'nse':
                    return nse_loss(
                        params_dict, precip, temp, pet, obs,
                        warmup, use_jax=True,
                        day_of_year=day_of_year,
                        latitude=latitude, si=si,
                        snow_module=snow_module,
                    )
                return kge_loss(
                    params_dict, precip, temp, pet, obs,
                    warmup, use_jax=True,
                    day_of_year=day_of_year,
                    latitude=latitude, si=si,
                    snow_module=snow_module,
                )

            grad_fn = jax.grad(loss_fn)
            param_names = list(params.keys())
            param_values = jnp.array([params[k] for k in param_names])
            grad_values = grad_fn(param_values, param_names)

            return dict(zip(param_names, np.array(grad_values)))

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error computing gradient: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """Evaluate loss and compute gradient in single pass.

        Uses JAX value_and_grad for efficient computation through the
        coupled Snow-17 + SAC-SMA simulation.

        Args:
            params: Parameter values (Snow-17 + SAC-SMA combined).
            metric: Metric to evaluate.

        Returns:
            Tuple of (loss_value, gradient_dict).
        """
        if not HAS_JAX or not self._use_jax:
            loss = self._evaluate_loss(params, metric)
            return loss, None

        if not self._initialized:
            if not self.initialize():
                return self.penalty_score, None

        try:
            from symfluence.models.sacsma.losses import kge_loss, nse_loss

            assert self._forcing is not None
            precip = jnp.array(self._forcing['precip'])
            temp = jnp.array(self._forcing['temp'])
            pet = jnp.array(self._forcing['pet'])
            obs = jnp.array(self._observations)

            # Day of year from time index
            day_of_year = None
            if self._time_index is not None:
                day_of_year = jnp.array(self._time_index.dayofyear.values,
                                        dtype=float)

            latitude = self.latitude
            si = self.si
            snow_module = self.snow_module
            warmup = self.warmup_days

            def loss_fn(params_array, param_names):
                params_dict = dict(zip(param_names, params_array))
                if metric.lower() == 'nse':
                    return nse_loss(
                        params_dict, precip, temp, pet, obs,
                        warmup, use_jax=True,
                        day_of_year=day_of_year,
                        latitude=latitude, si=si,
                        snow_module=snow_module,
                    )
                return kge_loss(
                    params_dict, precip, temp, pet, obs,
                    warmup, use_jax=True,
                    day_of_year=day_of_year,
                    latitude=latitude, si=si,
                    snow_module=snow_module,
                )

            value_and_grad_fn = jax.value_and_grad(loss_fn)
            param_names = list(params.keys())
            param_values = jnp.array([params[k] for k in param_names])
            loss_val, grad_values = value_and_grad_fn(param_values, param_names)

            gradient = dict(zip(param_names, np.array(grad_values)))
            return float(loss_val), gradient

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in evaluate_with_gradient: {e}")
            return self.penalty_score, None

    # =========================================================================
    # Static Worker Function for Process Pool
    # =========================================================================

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_sacsma_parameters_worker(task_data)


def _evaluate_sacsma_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for process pool execution."""
    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    })

    time.sleep(random.uniform(0.05, 0.2))  # nosec B311

    try:
        worker = SacSmaWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'SAC-SMA worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
