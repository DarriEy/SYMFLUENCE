"""
TOPMODEL Calibration Worker.

Worker implementation for TOPMODEL optimization.
Runs TOPMODEL in-memory for efficient DDS/evolutionary calibration.
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

from symfluence.optimization.workers.inmemory_worker import InMemoryModelWorker, HAS_JAX
from symfluence.optimization.workers.base_worker import WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.core.constants import ModelDefaults

# Lazy JAX import
if HAS_JAX:
    import jax.numpy as jnp


@OptimizerRegistry.register_worker('TOPMODEL')
class TopmodelWorker(InMemoryModelWorker):
    """Worker for TOPMODEL calibration.

    Supports:
    - Standard evolutionary optimization (evaluate -> apply -> run -> metrics)
    - Efficient in-memory simulation (no file I/O during calibration)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize TOPMODEL worker."""
        super().__init__(config, logger)

        self._simulate_fn = None
        self._use_jax = HAS_JAX

    # =========================================================================
    # InMemoryModelWorker Abstract Method Implementations
    # =========================================================================

    def _get_model_name(self) -> str:
        """Return the model identifier."""
        return 'TOPMODEL'

    def _get_forcing_subdir(self) -> str:
        """Return the forcing subdirectory name."""
        return 'TOPMODEL_input'

    def _get_forcing_variable_map(self) -> Dict[str, str]:
        """Return mapping from standard names to TOPMODEL variable names."""
        return {
            'precip': 'pr',
            'temp': 'temp',
            'pet': 'pet',
        }

    def _load_forcing(self, task=None) -> bool:
        """Load forcing data.

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
        domain_name = self.config.get('DOMAIN_NAME', 'domain')
        var_map = self._get_forcing_variable_map()

        nc_patterns = [
            forcing_dir / f"{domain_name}_topmodel_forcing.nc",
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
                            f"Loaded forcing from {nc_file.name}: "
                            f"{len(self._forcing['precip'])} timesteps"
                        )
                        return True
                except (OSError, RuntimeError, KeyError) as e:
                    self.logger.warning(f"Error loading {nc_file}: {e}")

        self.logger.error(f"No forcing file found in {forcing_dir}")
        return False

    def _load_observations(self, task=None) -> bool:
        """Load observations.

        Returns:
            True if loading successful
        """
        if self._observations is not None:
            return True

        from pathlib import Path

        domain_name = self.config.get('DOMAIN_NAME', 'domain')
        data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR', self.config.get('ROOT_PATH', '.')))
        project_dir = data_dir / f"domain_{domain_name}"

        obs_patterns = [
            project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv",
        ]

        for obs_file in obs_patterns:
            if obs_file.exists():
                try:
                    obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

                    if not isinstance(obs_df.index, pd.DatetimeIndex):
                        obs_df.index = pd.to_datetime(obs_df.index)

                    obs_cms = obs_df.iloc[:, 0]

                    # Resample to daily if sub-daily
                    if len(obs_cms) > 1:
                        time_diff = obs_cms.index[1] - obs_cms.index[0]
                        if time_diff < pd.Timedelta(days=1):
                            obs_cms = obs_cms.resample('D').mean().dropna()

                    # Convert m3/s to mm/day
                    area_km2 = self.get_catchment_area()
                    seconds_per_day = 86400
                    conversion_factor = seconds_per_day / (area_km2 * 1e6 * 0.001)
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

    def _run_simulation(
        self,
        forcing: Dict[str, np.ndarray],
        params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """Run TOPMODEL simulation.

        Args:
            forcing: Dictionary with 'precip', 'temp', 'pet' arrays
            params: Parameter dictionary

        Returns:
            Runoff array in mm/day
        """
        if not self._ensure_simulate_fn():
            raise RuntimeError("TOPMODEL simulation function not available")

        precip = forcing['precip']
        temp = forcing['temp']
        pet = forcing['pet']

        if self._use_jax:
            precip = jnp.array(precip)
            temp = jnp.array(temp)
            pet = jnp.array(pet)

        assert self._simulate_fn is not None
        runoff, _ = self._simulate_fn(
            precip, temp, pet,
            params=params,
            warmup_days=self.warmup_days,
            use_jax=self._use_jax,
        )

        if self._use_jax:
            return np.array(runoff)
        return runoff

    # =========================================================================
    # Model-Specific Methods
    # =========================================================================

    def _ensure_simulate_fn(self) -> bool:
        """Ensure simulation function is loaded."""
        if self._simulate_fn is not None:
            return True

        try:
            from symfluence.models.topmodel.model import simulate, HAS_JAX as MODEL_HAS_JAX
            self._simulate_fn = simulate
            self._use_jax = MODEL_HAS_JAX and HAS_JAX
            return True
        except ImportError as e:
            self.logger.error(f"Failed to import TOPMODEL: {e}")
            return False

    def _initialize_model(self) -> bool:
        """Initialize TOPMODEL components."""
        return self._ensure_simulate_fn()

    # =========================================================================
    # Static Worker Function for Process Pool
    # =========================================================================

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_topmodel_parameters_worker(task_data)


def _evaluate_topmodel_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary containing params, config, etc.

    Returns:
        Result dictionary with score and metrics.
    """
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
        worker = TopmodelWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'TOPMODEL worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
