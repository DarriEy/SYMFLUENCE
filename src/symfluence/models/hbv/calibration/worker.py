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

from symfluence.optimization.workers.inmemory_worker import InMemoryModelWorker, HAS_JAX
from symfluence.optimization.workers.base_worker import WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.core.constants import ModelDefaults

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
            Runoff array in mm/day
        """
        if not self._ensure_simulate_fn():
            raise RuntimeError("HBV simulation function not available")

        from symfluence.models.hbv.model import create_initial_state

        precip = forcing['precip']
        temp = forcing['temp']
        pet = forcing['pet']

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
