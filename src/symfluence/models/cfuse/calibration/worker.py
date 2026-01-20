"""
cFUSE Calibration Worker with Native Gradient Support.

Provides the CFUSEWorker class for parameter evaluation during optimization.
Supports both numerical and native Enzyme AD gradients via PyTorch autograd.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.metrics import kge, nse

# Lazy PyTorch and cFUSE imports
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

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

# Import PyTorch integration if available
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
    return VIC_CONFIG.to_dict()  # Default


@OptimizerRegistry.register_worker('CFUSE')
class CFUSEWorker(BaseWorker):
    """
    Worker for cFUSE model evaluation with native gradient support.

    Key Features:
    - Native gradient computation via Enzyme AD (when available)
    - PyTorch autograd fallback for gradient computation
    - Support for both lumped and distributed (multi-HRU) modes
    - Efficient batch processing for distributed simulations
    - Falls back to numerical gradients when native unavailable

    The worker maintains a cached PyTorch model for efficiency.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize cFUSE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Check dependencies
        if not HAS_CFUSE:
            self.logger.warning("cFUSE not installed. Model execution will fail.")

        if not HAS_TORCH:
            self.logger.warning("PyTorch not installed. Gradient computation will be unavailable.")

        # Model configuration
        self.model_structure = self.config.get('CFUSE_MODEL_STRUCTURE', 'vic')
        self.enable_snow = self.config.get('CFUSE_ENABLE_SNOW', True)
        self.warmup_days = int(self.config.get('CFUSE_WARMUP_DAYS', 365))
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

        # Cached model and data
        self._forcing = None
        self._observations = None
        self._time_index = None
        self._n_hrus = 1
        self._n_states = 10  # Default, updated when core is loaded
        self._initialized = False

        # Cache for simulation results
        self._last_params = None
        self._last_runoff = None

        # Get number of states from core if available
        if HAS_CFUSE_CORE:
            try:
                self._n_states = cfuse_core.get_num_active_states(self.config_dict)
            except Exception as e:
                self.logger.debug(f"Could not get state count from cfuse_core: {e}")

    def supports_native_gradients(self) -> bool:
        """
        Check if native gradient computation is available.

        Returns:
            True if PyTorch and cFUSE core are both installed.
        """
        return HAS_TORCH and HAS_CFUSE_CORE

    def supports_enzyme_gradients(self) -> bool:
        """
        Check if Enzyme AD gradients are available.

        Returns:
            True if Enzyme AD is available in the cFUSE core.
        """
        return HAS_ENZYME

    def _initialize_model_and_data(self, task: Optional[WorkerTask] = None) -> bool:
        """
        Initialize cFUSE model and load forcing/observation data.

        Args:
            task: Optional task containing paths

        Returns:
            True if initialization successful.
        """
        if self._initialized:
            return True

        if not HAS_CFUSE_CORE:
            self.logger.error("cFUSE core not installed. Cannot initialize model.")
            return False

        try:
            # Load forcing data
            forcing_dir = self._get_forcing_dir(task)
            domain_name = self.config.get('DOMAIN_NAME', 'domain')

            # Try NetCDF first (distributed or lumped)
            nc_file_distributed = forcing_dir / f"{domain_name}_cfuse_forcing_distributed.nc"
            nc_file = forcing_dir / f"{domain_name}_cfuse_forcing.nc"

            if nc_file_distributed.exists():
                ds = xr.open_dataset(nc_file_distributed)
                self._load_distributed_forcing(ds)
                ds.close()
            elif nc_file.exists():
                ds = xr.open_dataset(nc_file)
                self._load_lumped_forcing(ds)
                ds.close()
            else:
                # Try CSV for lumped
                csv_file = forcing_dir / f"{domain_name}_cfuse_forcing.csv"
                if csv_file.exists():
                    self._load_csv_forcing(csv_file)
                else:
                    self.logger.error(f"No forcing file found in {forcing_dir}")
                    return False

            # Load observations
            obs_file = forcing_dir / f"{domain_name}_observations.csv"
            if obs_file.exists():
                obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
                obs = obs_df.iloc[:, 0].values

                # Handle warmup alignment
                if len(obs) > self.warmup_days:
                    obs = obs[self.warmup_days:]

                if HAS_TORCH:
                    self._observations = torch.tensor(obs, dtype=torch.float32, device=self.device)
                else:
                    self._observations = obs
            else:
                self.logger.warning(f"No observation file found: {obs_file}")
                self._observations = None

            self._initialized = True
            n_timesteps = len(self._forcing['precip'])
            self.logger.info(
                f"cFUSE worker initialized: {n_timesteps} timesteps, "
                f"{self._n_hrus} HRUs, device={self.device_name}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize cFUSE worker: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_lumped_forcing(self, ds: xr.Dataset) -> None:
        """Load lumped forcing from NetCDF dataset."""
        precip = ds['precip'].values.flatten()
        temp = ds['temp'].values.flatten()
        pet = ds['pet'].values.flatten()
        self._time_index = pd.to_datetime(ds.time.values)
        self._n_hrus = 1

        # Store as tensors if PyTorch available
        if HAS_TORCH:
            # Shape: [n_timesteps, n_hrus, 3] for batch interface
            forcing_array = np.stack([precip, pet, temp], axis=-1)  # [time, 3]
            forcing_array = forcing_array[:, np.newaxis, :]  # [time, 1, 3]
            self._forcing = {
                'tensor': torch.tensor(forcing_array, dtype=torch.float32, device=self.device),
                'precip': precip,
                'temp': temp,
                'pet': pet,
            }
        else:
            self._forcing = {
                'precip': precip,
                'temp': temp,
                'pet': pet,
            }

    def _load_distributed_forcing(self, ds: xr.Dataset) -> None:
        """Load distributed forcing from NetCDF dataset."""
        precip = ds['precip'].values  # [time, hru]
        temp = ds['temp'].values
        pet = ds['pet'].values
        self._time_index = pd.to_datetime(ds.time.values)
        self._n_hrus = precip.shape[1] if precip.ndim > 1 else 1

        if HAS_TORCH:
            # Shape: [n_timesteps, n_hrus, 3] for batch interface
            forcing_array = np.stack([precip, pet, temp], axis=-1)
            self._forcing = {
                'tensor': torch.tensor(forcing_array, dtype=torch.float32, device=self.device),
                'precip': precip,
                'temp': temp,
                'pet': pet,
            }
        else:
            self._forcing = {
                'precip': precip,
                'temp': temp,
                'pet': pet,
            }

    def _load_csv_forcing(self, csv_file: Path) -> None:
        """Load lumped forcing from CSV file."""
        df = pd.read_csv(csv_file)
        precip = df['precip'].values
        temp = df['temp'].values
        pet = df['pet'].values
        self._time_index = pd.to_datetime(df['time'])
        self._n_hrus = 1

        if HAS_TORCH:
            forcing_array = np.stack([precip, pet, temp], axis=-1)
            forcing_array = forcing_array[:, np.newaxis, :]
            self._forcing = {
                'tensor': torch.tensor(forcing_array, dtype=torch.float32, device=self.device),
                'precip': precip,
                'temp': temp,
                'pet': pet,
            }
        else:
            self._forcing = {
                'precip': precip,
                'temp': temp,
                'pet': pet,
            }

    def _get_forcing_dir(self, task: Optional[WorkerTask] = None) -> Path:
        """Get path to forcing directory."""
        if task and task.settings_dir:
            parent = task.settings_dir.parent.parent if task.settings_dir.parent else task.settings_dir
            forcing_dir = parent / 'forcing' / 'CFUSE_input'
            if forcing_dir.exists():
                return forcing_dir

        root_path = Path(self.config.get('ROOT_PATH', '.'))
        domain_name = self.config.get('DOMAIN_NAME', 'domain')
        return root_path / domain_name / 'forcing' / 'CFUSE_input'

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array in correct order."""
        if not HAS_CFUSE:
            return np.array(list(params.values()), dtype=np.float32)

        # Use full PARAM_NAMES order, filling with defaults
        full_params = DEFAULT_PARAMS.copy()
        full_params.update(params)
        return np.array([full_params.get(name, 0.0) for name in PARAM_NAMES], dtype=np.float32)

    def _get_initial_states(self) -> np.ndarray:
        """Get initial state array."""
        states = np.zeros((self._n_hrus, self._n_states), dtype=np.float32)
        # Set initial storages
        states[:, 0] = self.config.get('CFUSE_INITIAL_S1', 50.0)
        if self._n_states > 1:
            states[:, 1] = 20.0  # Upper free storage
        if self._n_states > 2:
            states[:, 2] = self.config.get('CFUSE_INITIAL_S2', 200.0)
        return states

    # =========================================================================
    # BaseWorker Abstract Method Implementations
    # =========================================================================

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters for cFUSE simulation.

        For cFUSE, parameters are passed directly to the model, so this
        stores the parameters for run_model.

        Args:
            params: Parameter dictionary
            settings_dir: Settings directory path
            **kwargs: Additional arguments (unused)

        Returns:
            True (always succeeds for cFUSE)
        """
        task = kwargs.get('task')
        if not self._initialized:
            if not self._initialize_model_and_data(task):
                return False

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
        Run cFUSE model simulation.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory
            **kwargs: Must contain 'params' for the simulation

        Returns:
            True if simulation successful.
        """
        if not HAS_CFUSE_CORE:
            self.logger.error("cFUSE core not installed")
            return False

        try:
            params = kwargs.get('params', getattr(self, '_current_params', None))
            if params is None:
                self.logger.error("No parameters provided for simulation")
                return False

            # Convert parameters to array
            params_array = self._params_to_array(params)

            # Get forcing and initial states
            if HAS_TORCH and 'tensor' in self._forcing:
                forcing_np = self._forcing['tensor'].cpu().numpy()
            else:
                precip = self._forcing['precip']
                pet = self._forcing['pet']
                temp = self._forcing['temp']
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

            # Handle warmup
            if self.warmup_days > 0 and len(runoff) > self.warmup_days:
                runoff = runoff[self.warmup_days:]

            # Store result
            if self._n_hrus == 1:
                self._last_runoff = runoff.flatten()
            else:
                self._last_runoff = runoff  # [time, hrus]
            self._last_params = params

            return True

        except Exception as e:
            self.logger.error(f"cFUSE simulation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from cFUSE simulation results.

        Args:
            output_dir: Output directory (unused for cFUSE - results in memory)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary with metrics (KGE, NSE, etc.)
        """
        if self._last_runoff is None:
            self.logger.error("No simulation results available")
            return {'error': 'No simulation results'}

        if self._observations is None:
            self.logger.warning("No observations available for metrics")
            return {'error': 'No observations'}

        try:
            sim = self._last_runoff
            if self._n_hrus > 1:
                # Sum across HRUs for total catchment runoff
                sim = np.sum(sim, axis=1)

            # Get observations
            if HAS_TORCH and isinstance(self._observations, torch.Tensor):
                obs = self._observations.cpu().numpy()
            else:
                obs = self._observations

            # Align lengths
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs = obs[:min_len]

            # Remove NaN values
            valid_mask = ~(np.isnan(sim) | np.isnan(obs))
            sim = sim[valid_mask]
            obs = obs[valid_mask]

            if len(sim) < 10:
                return {'error': 'Insufficient valid data points', 'n_points': len(sim)}

            # Calculate metrics
            kge_val = float(kge(obs, sim, transfo=1))
            nse_val = float(nse(obs, sim, transfo=1))

            # KGE components
            r = np.corrcoef(obs, sim)[0, 1]
            alpha = np.std(sim) / (np.std(obs) + 1e-10)
            beta = np.mean(sim) / (np.mean(obs) + 1e-10)

            metrics = {
                'KGE': kge_val,
                'NSE': nse_val,
                'kge_r': float(r),
                'kge_alpha': float(alpha),
                'kge_beta': float(beta),
                'mean_sim': float(np.mean(sim)),
                'mean_obs': float(np.mean(obs)),
                'n_points': len(sim),
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Metric calculation failed: {e}")
            return {'error': str(e)}

    # =========================================================================
    # Native Gradient Methods
    # =========================================================================

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradient of loss with respect to parameters using Enzyme AD or PyTorch.

        Args:
            params: Dictionary mapping parameter names to values
            metric: Objective metric ('kge' or 'nse')

        Returns:
            Dictionary mapping parameter names to gradient values,
            or None if native gradients not supported.
        """
        if not self.supports_native_gradients():
            return None

        if not self._initialized:
            if not self._initialize_model_and_data():
                return None

        if self._observations is None:
            self.logger.error("No observations available for gradient computation")
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
        """
        Evaluate loss and compute gradient in a single pass.

        This uses cfuse.torch.DifferentiableFUSEBatch for efficient gradient
        computation via Enzyme AD or numerical differentiation.

        Args:
            params: Dictionary mapping parameter names to values
            metric: Objective metric ('kge' or 'nse')

        Returns:
            Tuple of (loss_value, gradient_dict):
            - loss_value: Scalar loss (negative of metric)
            - gradient_dict: Dictionary mapping parameter names to gradients
        """
        if not self.supports_native_gradients():
            raise NotImplementedError(
                "Native gradients not supported. PyTorch or cFUSE core not installed."
            )

        if not self._initialized:
            if not self._initialize_model_and_data():
                raise RuntimeError("Failed to initialize cFUSE worker")

        if self._observations is None:
            raise ValueError("No observations available for gradient computation")

        try:
            # Get parameter names we're calibrating
            param_names = list(params.keys())

            # Create full parameter array with gradients only for calibrated params
            full_params = DEFAULT_PARAMS.copy()
            full_params.update(params)

            # Convert to tensor with gradients
            params_array = torch.tensor(
                [full_params.get(name, 0.0) for name in PARAM_NAMES],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True
            )

            # Get forcing tensor
            forcing = self._forcing['tensor']  # [time, hru, 3]

            # Initial states
            initial_states = torch.tensor(
                self._get_initial_states(),
                dtype=torch.float32,
                device=self.device
            )

            # Forward pass using DifferentiableFUSEBatch
            runoff = DifferentiableFUSEBatch.apply(
                params_array,
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
            obs = self._observations
            if len(obs) > len(runoff):
                obs = obs[:len(runoff)]
            elif len(runoff) > len(obs):
                runoff = runoff[:len(obs)]

            # Compute loss (negative metric)
            if metric.lower() == 'nse':
                loss = self._nse_loss(obs, runoff)
            else:
                loss = self._kge_loss(obs, runoff)

            # Backward pass
            loss.backward()

            # Extract gradients for calibrated parameters
            grad_array = params_array.grad.cpu().numpy()
            grad_dict = {}
            for name in param_names:
                if name in PARAM_NAMES:
                    idx = PARAM_NAMES.index(name)
                    grad_dict[name] = float(grad_array[idx])
                else:
                    grad_dict[name] = 0.0

            loss_value = float(loss.item())

            return loss_value, grad_dict

        except Exception as e:
            self.logger.error(f"Value and gradient computation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _kge_loss(self, obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """Compute negative KGE loss (PyTorch differentiable)."""
        # Remove NaN (use mask)
        valid_mask = ~(torch.isnan(obs) | torch.isnan(sim))
        obs_valid = obs[valid_mask]
        sim_valid = sim[valid_mask]

        if len(obs_valid) < 10:
            return torch.tensor(999.0, device=self.device)

        # KGE components
        obs_mean = obs_valid.mean()
        sim_mean = sim_valid.mean()
        obs_std = obs_valid.std()
        sim_std = sim_valid.std()

        # Correlation
        cov = ((obs_valid - obs_mean) * (sim_valid - sim_mean)).mean()
        r = cov / (obs_std * sim_std + 1e-10)

        # Alpha (variability ratio)
        alpha = sim_std / (obs_std + 1e-10)

        # Beta (bias ratio)
        beta = sim_mean / (obs_mean + 1e-10)

        # KGE
        kge_val = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        return -kge_val  # Negative for minimization

    def _nse_loss(self, obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """Compute negative NSE loss (PyTorch differentiable)."""
        valid_mask = ~(torch.isnan(obs) | torch.isnan(sim))
        obs_valid = obs[valid_mask]
        sim_valid = sim[valid_mask]

        if len(obs_valid) < 10:
            return torch.tensor(999.0, device=self.device)

        # NSE
        ss_res = ((obs_valid - sim_valid) ** 2).sum()
        ss_tot = ((obs_valid - obs_valid.mean()) ** 2).sum()
        nse_val = 1 - ss_res / (ss_tot + 1e-10)

        return -nse_val  # Negative for minimization

    def evaluate_parameters(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> float:
        """
        Evaluate a parameter set and return the metric value.

        Args:
            params: Parameter dictionary
            metric: Evaluation metric ('kge' or 'nse')

        Returns:
            Metric value (higher is better)
        """
        if not self._initialized:
            if not self._initialize_model_and_data():
                return self.penalty_score

        try:
            # Convert parameters to array
            params_array = self._params_to_array(params)

            # Get forcing
            if HAS_TORCH and 'tensor' in self._forcing:
                forcing_np = self._forcing['tensor'].cpu().numpy()
            else:
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

            # Sum across HRUs if distributed
            if self._n_hrus > 1:
                runoff = np.sum(runoff, axis=1)
            else:
                runoff = runoff.flatten()

            # Get observations
            if HAS_TORCH and isinstance(self._observations, torch.Tensor):
                obs = self._observations.cpu().numpy()
            else:
                obs = self._observations

            if obs is None:
                return self.penalty_score

            # Align and filter
            min_len = min(len(runoff), len(obs))
            sim = runoff[:min_len]
            obs_arr = obs[:min_len]

            valid_mask = ~(np.isnan(sim) | np.isnan(obs_arr))
            sim = sim[valid_mask]
            obs_arr = obs_arr[valid_mask]

            if len(sim) < 10:
                return self.penalty_score

            # Calculate metric
            if metric.lower() == 'nse':
                return float(nse(obs_arr, sim, transfo=1))
            return float(kge(obs_arr, sim, transfo=1))

        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return self.penalty_score
