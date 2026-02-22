"""
Xinanjiang Calibration Worker.

Worker implementation for Xinanjiang model optimization.
Uses InMemoryModelWorker base class for common functionality.
Supports native gradient computation via JAX autodiff.
"""

import os
import sys
import signal
import random
import time
import traceback
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.optimization.workers.inmemory_worker import InMemoryModelWorker
from symfluence.optimization.workers.base_worker import WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.core.constants import ModelDefaults
from symfluence.core.mixins.project import resolve_data_subdir


@OptimizerRegistry.register_worker('XINANJIANG')
class XinanjiangWorker(InMemoryModelWorker):
    """Worker for Xinanjiang model calibration.

    Supports both evolutionary optimization (DDS, PSO) and gradient-based
    optimization via JAX autodiff when JAX is available.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        self._simulate_fn = None

    def _get_model_name(self) -> str:
        return 'XINANJIANG'

    def _get_forcing_subdir(self) -> str:
        return 'XINANJIANG_input'

    def _get_forcing_variable_map(self) -> Dict[str, str]:
        var_map = {
            'precip': 'pr',
            'pet': 'pet',
        }
        if self.config and self.config.get('XINANJIANG_SNOW_MODULE') == 'snow17':
            var_map['temp'] = 'tas'
        return var_map

    @property
    def use_jax(self) -> bool:
        """Whether to use JAX backend for simulation."""
        if self.config:
            return str(self.config.get('XINANJIANG_BACKEND', 'numpy')).lower() == 'jax'
        return False

    @property
    def snow_module(self) -> str:
        """Get snow module setting from config."""
        if self.config:
            return str(self.config.get('XINANJIANG_SNOW_MODULE', 'none'))
        return 'none'

    @property
    def latitude(self) -> float:
        """Get latitude from config."""
        if self.config:
            return float(self.config.get(
                'XINANJIANG_LATITUDE',
                self.config.get('LATITUDE', 45.0),
            ))
        return 45.0

    @property
    def si(self) -> float:
        """Get SI from config."""
        if self.config:
            return float(self.config.get('XINANJIANG_SI', 100.0))
        return 100.0

    def _load_forcing(self, task=None) -> bool:
        """Load Xinanjiang forcing data."""
        if self._forcing is not None:
            return True

        try:
            import xarray as xr
        except ImportError:
            self.logger.error("xarray required for loading forcing")
            return False

        forcing_dir = self._get_forcing_dir(task)
        domain_name = self.config.get('DOMAIN_NAME', 'domain')
        var_map = self._get_forcing_variable_map()

        nc_patterns = [
            forcing_dir / f"{domain_name}_xinanjiang_forcing.nc",
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
                        # Fallback aliases for temperature
                        elif std_name == 'temp':
                            for alias in ('tas', 'temp', 'temperature', 'airtemp'):
                                if alias in ds.variables:
                                    self._forcing[std_name] = ds[alias].values.flatten()
                                    break

                    if 'time' in ds.coords:
                        self._time_index = pd.to_datetime(ds.time.values)

                    ds.close()

                    if len(self._forcing) >= 2:
                        self.logger.info(
                            f"Loaded Xinanjiang forcing from {nc_file.name}: "
                            f"{len(self._forcing['precip'])} timesteps"
                        )
                        return True
                except (OSError, RuntimeError, KeyError) as e:
                    self.logger.warning(f"Error loading {nc_file}: {e}")

        self.logger.error(f"No Xinanjiang forcing file found in {forcing_dir}")
        return False

    def _load_observations(self, task=None) -> bool:
        """Load observations."""
        if self._observations is not None:
            return True

        from pathlib import Path

        domain_name = self.config.get('DOMAIN_NAME', 'domain')
        data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR', self.config.get('ROOT_PATH', '.')))
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

        self.logger.warning("No observation file found for Xinanjiang")
        return False

    def _run_simulation(
        self,
        forcing: Dict[str, np.ndarray],
        params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """Run Xinanjiang simulation. Returns runoff in mm/day."""
        if not self._ensure_simulate_fn():
            raise RuntimeError("Xinanjiang simulation function not available")

        precip = forcing['precip']
        pet = forcing['pet']

        assert self._simulate_fn is not None

        if self.snow_module == 'snow17' and 'temp' in forcing:
            from symfluence.models.xinanjiang.parameters import split_params

            temp = forcing['temp']
            xaj_dict, snow17_dict = split_params(params)

            # Compute day_of_year from time index
            if self._time_index is not None:
                doy = np.array([d.timetuple().tm_yday for d in self._time_index])
            else:
                doy = np.tile(np.arange(1, 366), len(precip) // 365 + 1)[:len(precip)]

            runoff, _ = self._simulate_fn(
                precip, pet,
                params=xaj_dict,
                warmup_days=self.warmup_days,
                use_jax=self.use_jax,
                temp=temp,
                day_of_year=doy,
                snow17_params=snow17_dict,
                latitude=self.latitude,
                si=self.si,
            )
        else:
            runoff, _ = self._simulate_fn(
                precip, pet,
                params=params,
                warmup_days=self.warmup_days,
                use_jax=self.use_jax,
            )

        return runoff

    def _ensure_simulate_fn(self) -> bool:
        """Ensure simulation function is loaded."""
        if self._simulate_fn is not None:
            return True

        try:
            from symfluence.models.xinanjiang.model import simulate
            self._simulate_fn = simulate
            return True
        except ImportError as e:
            self.logger.error(f"Failed to import Xinanjiang model: {e}")
            return False

    def _initialize_model(self) -> bool:
        return self._ensure_simulate_fn()

    def supports_native_gradients(self) -> bool:
        """Check if JAX is available for gradient computation."""
        try:
            from symfluence.models.xinanjiang.model import HAS_JAX
            return HAS_JAX
        except ImportError:
            return False

    def compute_gradient(self, params: Dict[str, float], metric: str = 'KGE') -> Optional[Dict[str, float]]:
        """Compute gradient of loss w.r.t. parameters using JAX autodiff.

        Args:
            params: Current parameter values
            metric: 'KGE' or 'NSE'

        Returns:
            Dictionary of parameter gradients, or None if JAX unavailable
        """
        if not self.supports_native_gradients():
            return None

        try:
            from symfluence.models.xinanjiang.losses import get_kge_gradient_fn, get_nse_gradient_fn
            import jax.numpy as jnp

            assert self._forcing is not None, "Forcing data not loaded"
            precip = jnp.array(self._forcing['precip'])
            pet = jnp.array(self._forcing['pet'])
            obs = jnp.array(self._observations)

            if metric.upper() == 'KGE':
                grad_fn = get_kge_gradient_fn(precip, pet, obs, warmup_days=self.warmup_days)
            else:
                grad_fn = get_nse_gradient_fn(precip, pet, obs, warmup_days=self.warmup_days)

            if grad_fn is None:
                return None

            param_names = list(params.keys())
            params_array = jnp.array([params[n] for n in param_names])
            grads = grad_fn(params_array, param_names)

            return dict(zip(param_names, np.array(grads)))

        except Exception as e:
            self.logger.warning(f"Gradient computation failed: {e}")
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_xinanjiang_parameters_worker(task_data)


def _evaluate_xinanjiang_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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
        worker = XinanjiangWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'Xinanjiang worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
