"""
cFUSE Calibration Worker with Native Gradient Support.

Provides the CFUSEWorker class for parameter evaluation during optimization.
Supports both numerical and native Enzyme AD gradients via PyTorch autograd.

Refactored to use InMemoryModelWorker base class for common functionality.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from symfluence.optimization.workers.inmemory_worker import InMemoryModelWorker
from symfluence.optimization.workers.base_worker import WorkerTask
from symfluence.optimization.registry import OptimizerRegistry

# Lazy PyTorch imports
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

# Lazy cFUSE imports
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
    HAS_ENZYME = getattr(cfuse_core, 'HAS_ENZYME', False)
except ImportError:
    HAS_CFUSE_CORE = False
    HAS_ENZYME = False
    cfuse_core = None

# PyTorch integration
try:
    from cfuse.torch import DifferentiableFUSEBatch, FUSEModule
    HAS_TORCH_INTEGRATION = True
except ImportError:
    HAS_TORCH_INTEGRATION = False
    DifferentiableFUSEBatch = None
    FUSEModule = None


def _get_model_config(structure: str) -> dict:
    """Get model configuration for a given structure."""
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
    return PRMS_CONFIG.to_dict()


@OptimizerRegistry.register_worker('CFUSE')
class CFUSEWorker(InMemoryModelWorker):
    """Worker for cFUSE model evaluation with native gradient support.

    Key Features:
    - Native gradient computation via Enzyme AD (when available)
    - PyTorch autograd fallback for gradient computation
    - Support for both lumped and distributed (multi-HRU) modes
    - Efficient batch processing for distributed simulations
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize cFUSE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        if not HAS_CFUSE:
            self.logger.warning("cFUSE not installed. Model execution will fail.")

        if not HAS_TORCH:
            self.logger.warning("PyTorch not installed. Gradient computation unavailable.")

        # Model configuration
        self.model_structure = self.config.get('CFUSE_MODEL_STRUCTURE', 'prms')
        self.enable_snow = self.config.get('CFUSE_ENABLE_SNOW', True)
        self.spatial_mode = self.config.get('CFUSE_SPATIAL_MODE', 'lumped')
        self.timestep_days = float(self.config.get('CFUSE_TIMESTEP_DAYS', 1.0))

        # Gradient configuration
        self.use_native_gradients = self.config.get('CFUSE_USE_NATIVE_GRADIENTS', True)
        self.device_name = self.config.get('CFUSE_DEVICE', 'cpu')

        # Set up PyTorch device
        if HAS_TORCH:
            if self.device_name == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = None

        # Model config dict for C++ core
        self.config_dict = _get_model_config(self.model_structure)
        if self.enable_snow:
            self.config_dict['enable_snow'] = True

        # cFUSE-specific attributes
        self._n_hrus = 1
        self._n_states = 10
        self._forcing_tensor = None

        # Get number of states from core
        if HAS_CFUSE_CORE:
            try:
                self._n_states = cfuse_core.get_num_active_states(self.config_dict)
            except Exception as e:
                self.logger.debug(f"Could not get state count: {e}")

    # =========================================================================
    # InMemoryModelWorker Abstract Method Implementations
    # =========================================================================

    def _get_model_name(self) -> str:
        """Return the model identifier."""
        return 'CFUSE'

    def _get_forcing_subdir(self) -> str:
        """Return the forcing subdirectory name."""
        return 'CFUSE_input'

    def _get_forcing_variable_map(self) -> Dict[str, str]:
        """Return mapping from standard names to cFUSE variable names."""
        return {
            'precip': 'precip',
            'temp': 'temp',
            'pet': 'pet',
        }

    def _get_warmup_days_config(self) -> int:
        """Get warmup days from config."""
        return int(self.config.get('CFUSE_WARMUP_DAYS', 365))

    def _run_simulation(
        self,
        forcing: Dict[str, np.ndarray],
        params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """Run cFUSE model simulation.

        Args:
            forcing: Dictionary with 'precip', 'temp', 'pet' arrays
            params: Parameter dictionary
            **kwargs: Additional arguments

        Returns:
            Runoff array in mm/day
        """
        if not HAS_CFUSE_CORE:
            raise RuntimeError("cFUSE core not installed")

        # Convert parameters to array
        params_array = self._params_to_array(params)

        # Get forcing as numpy array
        if self._forcing_tensor is not None and HAS_TORCH:
            forcing_np = self._forcing_tensor.cpu().numpy()
        else:
            precip = forcing['precip']
            pet = forcing['pet']
            temp = forcing['temp']
            if precip.ndim == 1:
                forcing_np = np.stack([precip, pet, temp], axis=-1)[:, np.newaxis, :]
            else:
                forcing_np = np.stack([precip, pet, temp], axis=-1)

        initial_states = self._get_initial_states()

        # Run simulation
        final_states, runoff = cfuse_core.run_fuse_batch(
            initial_states.astype(np.float32),
            forcing_np.astype(np.float32),
            params_array.astype(np.float32),
            self.config_dict,
            float(self.timestep_days)
        )

        # Handle multi-HRU output
        if self._n_hrus == 1:
            return runoff.flatten()
        return runoff

    # =========================================================================
    # Model Initialization
    # =========================================================================

    def _initialize_model(self) -> bool:
        """Initialize cFUSE model components."""
        if not HAS_CFUSE_CORE:
            self.logger.error("cFUSE core not installed")
            return False
        return True

    def initialize(self, task: Optional[WorkerTask] = None) -> bool:
        """Initialize model and load data with cFUSE-specific setup."""
        if self._initialized:
            return True

        # Load forcing first
        if not self._load_forcing(task):
            return False

        # Determine n_hrus from forcing shape
        assert self._forcing is not None
        precip = self._forcing['precip']
        if precip.ndim > 1:
            self._n_hrus = precip.shape[1]
        else:
            self._n_hrus = 1

        # Prepare forcing tensor for PyTorch
        self._prepare_forcing_tensor()

        # Initialize model
        if not self._initialize_model():
            return False

        # Load observations
        if not self._load_observations(task):
            self.logger.warning("No observations loaded - calibration will fail")

        self._initialized = True
        n_timesteps = len(self._forcing['precip'])
        self.logger.info(
            f"cFUSE worker initialized: {n_timesteps} timesteps, "
            f"{self._n_hrus} HRUs, device={self.device_name}"
        )
        return True

    def _prepare_forcing_tensor(self) -> None:
        """Prepare forcing as PyTorch tensor."""
        if self._forcing is None or not HAS_TORCH:
            return

        precip = self._forcing['precip']
        pet = self._forcing['pet']
        temp = self._forcing['temp']

        # Shape: [n_timesteps, n_hrus, 3]
        if precip.ndim == 1:
            forcing_array = np.stack([precip, pet, temp], axis=-1)[:, np.newaxis, :]
        else:
            forcing_array = np.stack([precip, pet, temp], axis=-1)

        self._forcing_tensor = torch.tensor(
            forcing_array, dtype=torch.float32, device=self.device
        )

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array in correct order."""
        if not HAS_CFUSE:
            return np.array(list(params.values()), dtype=np.float32)

        full_params = DEFAULT_PARAMS.copy()
        full_params.update(params)
        return np.array([full_params.get(name, 0.0) for name in PARAM_NAMES], dtype=np.float32)

    def _get_initial_states(self) -> np.ndarray:
        """Get initial state array."""
        states = np.zeros((self._n_hrus, self._n_states), dtype=np.float32)
        states[:, 0] = self.config.get('CFUSE_INITIAL_S1', 50.0)
        if self._n_states > 1:
            states[:, 1] = 20.0
        if self._n_states > 2:
            states[:, 2] = self.config.get('CFUSE_INITIAL_S2', 200.0)
        return states

    # =========================================================================
    # Override Metric Calculation for Multi-HRU
    # =========================================================================

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate metrics from cFUSE output."""
        if self._last_runoff is None:
            return {'kge': self.penalty_score, 'error': 'No simulation results'}

        if self._observations is None:
            return {'kge': self.penalty_score, 'error': 'No observations'}

        try:
            runoff = self._last_runoff
            if self._n_hrus > 1:
                runoff = np.sum(runoff, axis=1)

            # Skip warmup and calculate metrics
            sim = runoff[self.warmup_days:]
            obs = self._observations[self.warmup_days:]

            return self.calculate_streamflow_metrics(sim, obs, skip_warmup=False)

        except Exception as e:
            self.logger.error(f"Error calculating cFUSE metrics: {e}")
            return {'kge': self.penalty_score, 'error': str(e)}

    # =========================================================================
    # Native Gradient Support (PyTorch/Enzyme)
    # =========================================================================

    def supports_native_gradients(self) -> bool:
        """Check if native gradient computation is available."""
        return HAS_TORCH and HAS_CFUSE_CORE

    def supports_enzyme_gradients(self) -> bool:
        """Check if Enzyme AD gradients are available."""
        return HAS_ENZYME

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """Compute gradient using Enzyme AD or PyTorch."""
        if not self.supports_native_gradients():
            return None

        if not self._initialized:
            if not self.initialize():
                return None

        if self._observations is None:
            return None

        try:
            _, grad_dict = self.evaluate_with_gradient(params, metric)
            return grad_dict

        except Exception as e:
            self.logger.error(f"Gradient computation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """Evaluate loss and compute gradient using DifferentiableFUSEBatch."""
        if not self.supports_native_gradients():
            raise NotImplementedError(
                f"Native gradient computation not supported for {self._get_model_name()} worker. "
                "Use supports_native_gradients() to check availability before calling."
            )

        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize cFUSE worker")

        if self._observations is None:
            raise ValueError("No observations available")

        if not HAS_TORCH_INTEGRATION:
            raise NotImplementedError(
                "PyTorch integration not available for cFUSE gradient computation. "
                "Install PyTorch and torch-fuse to enable native gradients."
            )

        try:
            param_names = list(params.keys())

            # Create full parameter tensor with gradients
            full_params = DEFAULT_PARAMS.copy()
            full_params.update(params)

            params_tensor = torch.tensor(
                [full_params.get(name, 0.0) for name in PARAM_NAMES],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True
            )

            # Get forcing tensor
            forcing = self._forcing_tensor

            # Initial states
            initial_states = torch.tensor(
                self._get_initial_states(),
                dtype=torch.float32,
                device=self.device
            )

            # Forward pass using DifferentiableFUSEBatch
            runoff = DifferentiableFUSEBatch.apply(
                params_tensor,
                initial_states,
                forcing,
                self.config_dict,
                self.timestep_days
            )

            # Handle warmup
            if self.warmup_days > 0:
                runoff = runoff[self.warmup_days:]

            # Sum across HRUs if distributed
            if self._n_hrus > 1:
                runoff = runoff.sum(dim=1)
            else:
                runoff = runoff.squeeze()

            # Get observations
            obs = torch.tensor(
                self._observations[self.warmup_days:],
                dtype=torch.float32,
                device=self.device
            )

            # Align lengths
            if len(obs) > len(runoff):
                obs = obs[:len(runoff)]
            elif len(runoff) > len(obs):
                runoff = runoff[:len(obs)]

            # Compute loss
            if metric.lower() == 'nse':
                loss = self._nse_loss(obs, runoff)
            else:
                loss = self._kge_loss(obs, runoff)

            # Backward pass
            loss.backward()

            # Extract gradients for calibrated parameters
            grad_array = params_tensor.grad.cpu().numpy()
            grad_dict = {}
            for name in param_names:
                if name in PARAM_NAMES:
                    idx = PARAM_NAMES.index(name)
                    grad_dict[name] = float(grad_array[idx])
                else:
                    grad_dict[name] = 0.0

            return float(loss.item()), grad_dict

        except Exception as e:
            self.logger.error(f"Value and gradient computation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _kge_loss(self, obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """Compute negative KGE loss (PyTorch differentiable)."""
        valid_mask = ~(torch.isnan(obs) | torch.isnan(sim))
        obs_valid = obs[valid_mask]
        sim_valid = sim[valid_mask]

        if len(obs_valid) < 10:
            return torch.tensor(999.0, device=self.device)

        obs_mean = obs_valid.mean()
        sim_mean = sim_valid.mean()
        obs_std = obs_valid.std()
        sim_std = sim_valid.std()

        cov = ((obs_valid - obs_mean) * (sim_valid - sim_mean)).mean()
        r = cov / (obs_std * sim_std + 1e-10)
        alpha = sim_std / (obs_std + 1e-10)
        beta = sim_mean / (obs_mean + 1e-10)

        kge_val = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return -kge_val

    def _nse_loss(self, obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """Compute negative NSE loss (PyTorch differentiable)."""
        valid_mask = ~(torch.isnan(obs) | torch.isnan(sim))
        obs_valid = obs[valid_mask]
        sim_valid = sim[valid_mask]

        if len(obs_valid) < 10:
            return torch.tensor(999.0, device=self.device)

        ss_res = ((obs_valid - sim_valid) ** 2).sum()
        ss_tot = ((obs_valid - obs_valid.mean()) ** 2).sum()
        nse_val = 1 - ss_res / (ss_tot + 1e-10)

        return -nse_val

    def evaluate_parameters(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> float:
        """Evaluate a parameter set and return the metric value."""
        if not self._initialized:
            if not self.initialize():
                return self.penalty_score

        try:
            params_array = self._params_to_array(params)

            # Get forcing
            if HAS_TORCH and self._forcing_tensor is not None:
                forcing_np = self._forcing_tensor.cpu().numpy()
            else:
                assert self._forcing is not None
                precip = self._forcing['precip']
                pet = self._forcing['pet']
                temp = self._forcing['temp']
                if precip.ndim == 1:
                    forcing_np = np.stack([precip, pet, temp], axis=-1)[:, np.newaxis, :]
                else:
                    forcing_np = np.stack([precip, pet, temp], axis=-1)

            initial_states = self._get_initial_states()

            # Run simulation
            _, runoff = cfuse_core.run_fuse_batch(
                initial_states.astype(np.float32),
                forcing_np.astype(np.float32),
                params_array.astype(np.float32),
                self.config_dict,
                float(self.timestep_days)
            )

            # Handle warmup
            if self.warmup_days > 0 and len(runoff) > self.warmup_days:
                runoff = runoff[self.warmup_days:]

            # Sum across HRUs
            if self._n_hrus > 1:
                runoff = np.sum(runoff, axis=1)
            else:
                runoff = runoff.flatten()

            obs = self._observations
            if obs is None:
                return self.penalty_score

            # Skip warmup for observations
            obs = obs[self.warmup_days:]

            # Align and filter
            min_len = min(len(runoff), len(obs))
            sim = runoff[:min_len]
            obs_arr = obs[:min_len]

            valid_mask = ~(np.isnan(sim) | np.isnan(obs_arr))
            sim = sim[valid_mask]
            obs_arr = obs_arr[valid_mask]

            if len(sim) < 10:
                return self.penalty_score

            from symfluence.evaluation.metrics import kge, nse
            if metric.lower() == 'nse':
                return float(nse(obs_arr, sim, transfo=1))
            return float(kge(obs_arr, sim, transfo=1))

        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return self.penalty_score

    # =========================================================================
    # Static Worker Function
    # =========================================================================

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_cfuse_parameters_worker(task_data)


def _evaluate_cfuse_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution."""
    worker = CFUSEWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
