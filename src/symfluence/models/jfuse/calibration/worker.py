"""
jFUSE Calibration Worker with Native Gradient Support.

Provides the JFUSEWorker class for parameter evaluation during optimization.
Supports both finite-difference and native JAX autodiff gradients.

Refactored to use InMemoryModelWorker base class for common functionality.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.optimization.workers.inmemory_worker import InMemoryModelWorker, HAS_JAX
from symfluence.optimization.workers.base_worker import WorkerTask
from symfluence.optimization.registry import OptimizerRegistry

# Lazy JAX imports
if HAS_JAX:
    import jax
    import jax.numpy as jnp

# Lazy jFUSE imports
try:
    import jfuse
    import equinox as eqx
    from jfuse import (
        create_fuse_model, Parameters, PARAM_BOUNDS, kge_loss, nse_loss,
        CoupledModel, create_network_from_topology, load_network,
        FUSEModel, ModelConfig, BaseflowType, UpperLayerArch, LowerLayerArch,
        PercolationType, SurfaceRunoffType, EvaporationType, InterflowType
    )
    from jfuse.fuse.config import SnowType, RoutingType, RainfallErrorType

    # Custom config optimized for gradient-based calibration (ADAM/LBFGS)
    PRMS_GRADIENT_CONFIG = ModelConfig(
        upper_arch=UpperLayerArch.TENSION2_FREE,
        lower_arch=LowerLayerArch.SINGLE_NOEVAP,
        baseflow=BaseflowType.NONLINEAR,
        percolation=PercolationType.FREE_STORAGE,
        surface_runoff=SurfaceRunoffType.UZ_PARETO,
        evaporation=EvaporationType.SEQUENTIAL,
        interflow=InterflowType.LINEAR,
        snow=SnowType.TEMP_INDEX,
        routing=RoutingType.NONE,
        rainfall_error=RainfallErrorType.ADDITIVE,
    )

    # Maximum gradient config - Sacramento-based architecture
    MAX_GRADIENT_CONFIG = ModelConfig(
        upper_arch=UpperLayerArch.TENSION2_FREE,
        lower_arch=LowerLayerArch.TENSION_2RESERV,
        baseflow=BaseflowType.PARALLEL_LINEAR,
        percolation=PercolationType.LOWER_DEMAND,
        surface_runoff=SurfaceRunoffType.UZ_PARETO,
        evaporation=EvaporationType.ROOT_WEIGHT,
        interflow=InterflowType.LINEAR,
        snow=SnowType.TEMP_INDEX,
        routing=RoutingType.NONE,
        rainfall_error=RainfallErrorType.ADDITIVE,
    )

    JFUSE_CONFIGS = {
        'prms': None,
        'prms_gradient': PRMS_GRADIENT_CONFIG,
        'max_gradient': MAX_GRADIENT_CONFIG,
        'topmodel': None,
        'sacramento': None,
        'vic': None,
    }
    HAS_JFUSE = True
except ImportError:
    HAS_JFUSE = False
    jfuse = None
    eqx = None
    create_fuse_model = None
    Parameters = None
    PARAM_BOUNDS = {}
    kge_loss = None
    nse_loss = None
    CoupledModel = None
    create_network_from_topology = None
    load_network = None
    FUSEModel = None
    ModelConfig = None
    PRMS_GRADIENT_CONFIG = None
    MAX_GRADIENT_CONFIG = None
    JFUSE_CONFIGS = {}


@OptimizerRegistry.register_worker('JFUSE')
class JFUSEWorker(InMemoryModelWorker):
    """Worker for jFUSE model evaluation with native gradient support.

    Key Features:
    - Native gradient computation via JAX autodiff (when available)
    - Support for both lumped and distributed modes
    - Efficient value_and_grad for combined loss and gradient computation
    - Falls back to finite differences when JAX unavailable
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize jFUSE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        if not HAS_JFUSE:
            self.logger.warning("jFUSE not installed. Model execution will fail.")

        # Model configuration
        self.model_config_name = self.config.get('JFUSE_MODEL_CONFIG_NAME', 'prms_gradient')
        self.enable_snow = self.config.get('JFUSE_ENABLE_SNOW', True)
        self.spatial_mode = self.config.get('JFUSE_SPATIAL_MODE', 'lumped')

        # Distributed mode configuration
        self.n_hrus = int(self.config.get('JFUSE_N_HRUS', 1))
        self.network_file = self.config.get('JFUSE_NETWORK_FILE', None)
        self.hru_areas_file = self.config.get('JFUSE_HRU_AREAS_FILE', None)

        self._is_distributed = (
            self.spatial_mode == 'distributed' or
            self.n_hrus > 1 or
            self.network_file is not None
        )

        # JAX configuration
        self.jit_compile = self.config.get('JFUSE_JIT_COMPILE', True)
        self.use_gpu = self.config.get('JFUSE_USE_GPU', False)

        if HAS_JAX and not self.use_gpu:
            jax.config.update('jax_platform_name', 'cpu')

        # jFUSE-specific model components
        self._model = None
        self._default_params = None
        self._coupled_model = None
        self._network = None
        self._hru_areas = None
        self._forcing_tuple = None

        # Distributed mode output
        self._last_outlet_q = None

        # Gradient coverage tracking
        self._gradient_coverage_checked = False
        self._param_warning_logged = False

    # =========================================================================
    # InMemoryModelWorker Abstract Method Implementations
    # =========================================================================

    def _get_model_name(self) -> str:
        """Return the model identifier."""
        return 'JFUSE'

    def _get_forcing_subdir(self) -> str:
        """Return the forcing subdirectory name."""
        return 'JFUSE_input'

    def _get_forcing_variable_map(self) -> Dict[str, str]:
        """Return mapping from standard names to jFUSE variable names."""
        return {
            'precip': 'precip',
            'temp': 'temp',
            'pet': 'pet',
        }

    def _get_warmup_days_config(self) -> int:
        """Get warmup days from config."""
        return int(self.config.get('JFUSE_WARMUP_DAYS', 365))

    def _run_simulation(
        self,
        forcing: Dict[str, np.ndarray],
        params: Dict[str, float],
        **kwargs
    ) -> np.ndarray:
        """Run jFUSE model simulation.

        Args:
            forcing: Dictionary with 'precip', 'temp', 'pet' arrays
            params: Parameter dictionary
            **kwargs: Additional arguments

        Returns:
            Runoff array in mm/day
        """
        if not HAS_JFUSE or self._model is None:
            raise RuntimeError("jFUSE model not initialized")

        # Convert params to jFUSE Parameters object
        params_obj = self._dict_to_params(params)

        # Run simulation based on mode
        if self._is_distributed and self._coupled_model is not None:
            outlet_q, runoff = self._coupled_model.simulate(self._forcing_tuple, params_obj)
            outlet_q_arr = np.array(outlet_q) if HAS_JAX else outlet_q
            runoff_arr = np.array(runoff) if HAS_JAX else runoff
            self._last_outlet_q = outlet_q_arr
            return runoff_arr
        else:
            runoff, _ = self._model.simulate(self._forcing_tuple, params_obj)
            runoff_arr = np.array(runoff) if HAS_JAX else runoff
            self._last_outlet_q = None
            return runoff_arr

    # =========================================================================
    # Model Initialization
    # =========================================================================

    def _initialize_model(self) -> bool:
        """Initialize jFUSE model components."""
        if not HAS_JFUSE:
            self.logger.error("jFUSE not installed. Cannot initialize model.")
            return False

        try:
            # Determine number of HRUs from forcing if available
            if self._forcing is not None:
                precip = self._forcing['precip']
                if precip.ndim > 1:
                    actual_n_hrus = precip.shape[1]
                    if self.n_hrus == 1 and actual_n_hrus > 1:
                        self.n_hrus = actual_n_hrus
                        self._is_distributed = True
                        self.logger.info(f"Auto-detected {self.n_hrus} HRUs, using distributed mode")

            # Initialize model based on mode
            if self._is_distributed:
                self._initialize_distributed_model()
            else:
                self._initialize_lumped_model()

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize jFUSE model: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _initialize_lumped_model(self) -> None:
        """Initialize FUSEModel for lumped mode."""
        if self.model_config_name in JFUSE_CONFIGS and JFUSE_CONFIGS[self.model_config_name] is not None:
            custom_config = JFUSE_CONFIGS[self.model_config_name]
            self._model = FUSEModel(custom_config, n_hrus=1)
            self.logger.info(f"Initialized lumped FUSEModel with config: {self.model_config_name}")
        else:
            self._model = create_fuse_model(self.model_config_name, n_hrus=1)
            self.logger.debug(f"Initialized lumped FUSEModel with config: {self.model_config_name}")
        self._default_params = Parameters.default(n_hrus=1)

    def _initialize_distributed_model(self) -> None:
        """Initialize CoupledModel for distributed mode with routing."""
        forcing_dir = self._get_forcing_dir()
        domain_name = self.config.get('DOMAIN_NAME', 'domain')

        # Load or create network
        network_file = self.network_file
        if network_file is None:
            network_file = forcing_dir / f"{domain_name}_network.nc"

        if network_file and Path(network_file).exists():
            self.logger.info(f"Loading network from {network_file}")
            self._network = load_network(str(network_file))
        else:
            self.logger.warning("No network file found. Creating simple sequential network.")
            reach_ids = list(range(1, self.n_hrus + 1))
            downstream_ids = list(range(2, self.n_hrus + 1)) + [-1]
            lengths = [1000.0] * self.n_hrus
            slopes = [0.01] * self.n_hrus
            self._network = create_network_from_topology(
                reach_ids=reach_ids,
                downstream_ids=downstream_ids,
                lengths=lengths,
                slopes=slopes
            )

        # Load HRU areas
        if self.hru_areas_file and Path(self.hru_areas_file).exists():
            areas_df = pd.read_csv(self.hru_areas_file)
            self._hru_areas = jnp.array(areas_df.iloc[:, 0].values)
        else:
            self._hru_areas = jnp.ones(self.n_hrus) * 1e6

        # Create CoupledModel
        self._coupled_model = CoupledModel(
            network=self._network.to_arrays(),
            hru_areas=self._hru_areas,
            n_hrus=self.n_hrus
        )
        self._model = self._coupled_model.fuse_model
        self._default_params = self._coupled_model.default_params()
        self.logger.debug(f"Initialized CoupledModel with {self.n_hrus} HRUs")

    # =========================================================================
    # Override Data Loading to Handle jFUSE-specific Requirements
    # =========================================================================

    def initialize(self, task: Optional[WorkerTask] = None) -> bool:
        """Initialize model and load data with jFUSE-specific setup."""
        if self._initialized:
            return True

        # Load forcing first (base class method)
        if not self._load_forcing(task):
            return False

        # Initialize model (needs forcing shape for distributed mode)
        if not self._initialize_model():
            return False

        # Prepare forcing tuple for jFUSE
        self._prepare_forcing_tuple()

        # Load observations (base class handles unit conversion)
        if not self._load_observations(task):
            self.logger.warning("No observations loaded - calibration will fail")

        self._initialized = True
        n_timesteps = len(self._forcing['precip']) if self._forcing else 0
        mode_str = "distributed" if self._is_distributed else "lumped"
        area_str = f", area={self.get_catchment_area():.1f} km²" if self._catchment_area_km2 else ""
        self.logger.info(
            f"jFUSE worker initialized: {n_timesteps} timesteps, "
            f"{self.n_hrus} HRUs, {mode_str} mode{area_str}"
        )
        return True

    def _prepare_forcing_tuple(self) -> None:
        """Prepare forcing as tuple for jFUSE model."""
        if self._forcing is None or not HAS_JAX:
            return

        precip = self._forcing['precip']
        pet = self._forcing['pet']
        temp = self._forcing['temp']

        # Reshape if needed
        if precip.ndim == 1:
            precip = precip.reshape(-1, 1)
            pet = pet.reshape(-1, 1)
            temp = temp.reshape(-1, 1)

        # Squeeze if lumped mode with shape (n_timesteps, 1)
        if not self._is_distributed and precip.shape[1] == 1:
            precip = precip.squeeze(-1)
            pet = pet.squeeze(-1)
            temp = temp.squeeze(-1)

        # Convert to JAX arrays
        self._forcing_tuple = (
            jnp.array(precip),
            jnp.array(pet),
            jnp.array(temp)
        )

    # =========================================================================
    # Parameter Conversion
    # =========================================================================

    def _dict_to_params(self, param_dict: Dict[str, float]) -> Any:
        """Convert parameter dictionary to jFUSE Parameters object."""
        params = self._default_params

        # Debug logging for first call
        matched = []
        unmatched = []
        for name in param_dict.keys():
            if hasattr(params, name):
                matched.append(name)
            else:
                unmatched.append(name)

        if unmatched and not self._param_warning_logged:
            self._param_warning_logged = True
            self.logger.warning(
                f"jFUSE parameter mismatch - Matched: {matched}, "
                f"Unmatched (will use defaults): {unmatched}"
            )

        if self._is_distributed:
            fuse_params = params.fuse_params
            for name, value in param_dict.items():
                if hasattr(fuse_params, name):
                    arr = jnp.ones(self.n_hrus) * float(value)
                    fuse_params = eqx.tree_at(
                        lambda p, n=name: getattr(p, n), fuse_params, arr  # type: ignore[misc]
                    )
            params = eqx.tree_at(lambda p: p.fuse_params, params, fuse_params)  # type: ignore[misc]
        else:
            for name, value in param_dict.items():
                if hasattr(params, name):
                    params = eqx.tree_at(
                        lambda p, n=name: getattr(p, n), params, jnp.array(float(value))  # type: ignore[misc]
                    )

        return params

    # =========================================================================
    # Override Metric Calculation for Distributed Mode
    # =========================================================================

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate metrics from jFUSE output."""
        if self._last_runoff is None and self._last_outlet_q is None:
            return {'kge': self.penalty_score, 'error': 'No simulation results'}

        if self._observations is None:
            return {'kge': self.penalty_score, 'error': 'No observations'}

        try:
            # For distributed mode, use outlet discharge
            if self._is_distributed and self._last_outlet_q is not None:
                # outlet_q might be in different units - check and convert
                sim = self._last_outlet_q[self.warmup_days:]
            else:
                sim = self._last_runoff[self.warmup_days:]
                if sim.ndim > 1:
                    sim = sim[:, 0] if sim.shape[1] > 0 else sim.flatten()

            obs = self._observations[self.warmup_days:]

            # Calculate metrics (both in mm/day)
            return self.calculate_streamflow_metrics(sim, obs, skip_warmup=False)

        except Exception as e:
            self.logger.error(f"Error calculating jFUSE metrics: {e}")
            return {'kge': self.penalty_score, 'error': str(e)}

    # =========================================================================
    # Native Gradient Support (JAX autodiff)
    # =========================================================================

    def supports_native_gradients(self) -> bool:
        """Check if native gradient computation is available."""
        return HAS_JAX and HAS_JFUSE

    def check_gradient_coverage(
        self,
        param_names: list,
        epsilon: float = 1e-6
    ) -> Dict[str, bool]:
        """Check which parameters have non-zero gradients."""
        if not self._initialized:
            self.initialize()

        if not HAS_JAX or self._model is None or self._observations is None:
            return {name: True for name in param_names}

        gradient_status = {}
        zero_grad_params = []
        working_params = []

        for param_name in param_names:
            if param_name not in PARAM_BOUNDS:
                gradient_status[param_name] = False
                zero_grad_params.append(param_name)
                continue

            try:
                bounds = PARAM_BOUNDS[param_name]
                mid_val = (bounds[0] + bounds[1]) / 2.0

                forcing_tuple = self._forcing_tuple
                obs = jnp.array(self._observations)
                warmup = self.warmup_days
                fuse_model = self._model
                default_params = self._default_params

                def loss_fn(val, pn=param_name):
                    params = default_params
                    params = eqx.tree_at(lambda p, n=pn: getattr(p, n), params, val)
                    runoff, _ = fuse_model.simulate(forcing_tuple, params)
                    sim = runoff[warmup:]
                    obs_aligned = obs[:len(sim)]
                    return kge_loss(sim[:len(obs_aligned)], obs_aligned)

                grad_fn = jax.grad(loss_fn)
                grad_val = float(grad_fn(jnp.array(mid_val)))

                has_gradient = abs(grad_val) > epsilon
                gradient_status[param_name] = has_gradient

                if has_gradient:
                    working_params.append(param_name)
                else:
                    zero_grad_params.append(param_name)

            except Exception as e:
                self.logger.debug(f"Could not check gradient for {param_name}: {e}")
                gradient_status[param_name] = True

        if zero_grad_params:
            self.logger.warning(
                f"GRADIENT WARNING: {len(zero_grad_params)} parameters have zero gradients: {zero_grad_params}"
            )

        return gradient_status

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """Compute gradient using JAX autodiff."""
        if not self.supports_native_gradients():
            return None

        if not self._initialized:
            if not self.initialize():
                return None

        if self._observations is None:
            return None

        try:
            param_names = list(params.keys())
            forcing_tuple = self._forcing_tuple
            obs = jnp.array(self._observations)
            warmup = self.warmup_days
            default_params = self._default_params
            is_distributed = self._is_distributed
            n_hrus = self.n_hrus
            coupled_model = self._coupled_model
            fuse_model = self._model

            def array_to_params(arr):
                p = default_params
                if is_distributed:
                    fuse_p = p.fuse_params
                    for i, name in enumerate(param_names):
                        if hasattr(fuse_p, name):
                            fuse_p = eqx.tree_at(
                                lambda x, n=name: getattr(x, n), fuse_p, jnp.ones(n_hrus) * arr[i]  # type: ignore[misc]
                            )
                    p = eqx.tree_at(lambda x: x.fuse_params, p, fuse_p)  # type: ignore[misc]
                else:
                    for i, name in enumerate(param_names):
                        if hasattr(p, name):
                            p = eqx.tree_at(lambda x, n=name: getattr(x, n), p, arr[i])  # type: ignore[misc]
                return p

            def loss_from_array(param_array):
                params_obj = array_to_params(param_array)
                if is_distributed:
                    outlet_q, _ = coupled_model.simulate(forcing_tuple, params_obj)
                    sim_eval = outlet_q[warmup:]
                else:
                    runoff, _ = fuse_model.simulate(forcing_tuple, params_obj)
                    sim_eval = runoff[warmup:]
                obs_aligned = obs[:len(sim_eval)]
                if metric.lower() == 'nse':
                    return nse_loss(sim_eval[:len(obs_aligned)], obs_aligned)
                return kge_loss(sim_eval[:len(obs_aligned)], obs_aligned)

            param_array = jnp.array([params[name] for name in param_names])
            grad_fn = jax.grad(loss_from_array)
            grad_array = grad_fn(param_array)

            return {name: float(grad_array[i]) for i, name in enumerate(param_names)}

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
        """Evaluate loss and compute gradient in a single pass."""
        if not self.supports_native_gradients():
            raise NotImplementedError(
                f"Native gradient computation not supported for {self._get_model_name()} worker. "
                "Use supports_native_gradients() to check availability before calling."
            )

        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize jFUSE worker")

        if self._observations is None:
            raise ValueError("No observations available")

        # Check gradient coverage once
        if not self._gradient_coverage_checked:
            self._gradient_coverage_checked = True
            self.check_gradient_coverage(list(params.keys()))

        try:
            param_names = list(params.keys())
            forcing_tuple = self._forcing_tuple
            obs = jnp.array(self._observations)
            warmup = self.warmup_days
            default_params = self._default_params
            is_distributed = self._is_distributed
            n_hrus = self.n_hrus
            coupled_model = self._coupled_model
            fuse_model = self._model

            def array_to_params(arr):
                p = default_params
                if is_distributed:
                    fuse_p = p.fuse_params
                    for i, name in enumerate(param_names):
                        if hasattr(fuse_p, name):
                            fuse_p = eqx.tree_at(
                                lambda x, n=name: getattr(x, n), fuse_p, jnp.ones(n_hrus) * arr[i]  # type: ignore[misc]
                            )
                    p = eqx.tree_at(lambda x: x.fuse_params, p, fuse_p)  # type: ignore[misc]
                else:
                    for i, name in enumerate(param_names):
                        if hasattr(p, name):
                            p = eqx.tree_at(lambda x, n=name: getattr(x, n), p, arr[i])  # type: ignore[misc]
                return p

            def loss_from_array(param_array):
                params_obj = array_to_params(param_array)
                if is_distributed:
                    outlet_q, _ = coupled_model.simulate(forcing_tuple, params_obj)
                    sim_eval = outlet_q[warmup:]
                else:
                    runoff, _ = fuse_model.simulate(forcing_tuple, params_obj)
                    sim_eval = runoff[warmup:]
                obs_aligned = obs[:len(sim_eval)]
                if metric.lower() == 'nse':
                    return nse_loss(sim_eval[:len(obs_aligned)], obs_aligned)
                return kge_loss(sim_eval[:len(obs_aligned)], obs_aligned)

            param_array = jnp.array([params[name] for name in param_names])
            val_and_grad_fn = jax.value_and_grad(loss_from_array)
            loss_val, grad_array = val_and_grad_fn(param_array)

            grad_dict = {name: float(grad_array[i]) for i, name in enumerate(param_names)}
            return float(loss_val), grad_dict

        except Exception as e:
            self.logger.error(f"Value and gradient computation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

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
            params_obj = self._dict_to_params(params)

            if self._is_distributed:
                outlet_q, runoff = self._coupled_model.simulate(self._forcing_tuple, params_obj)
                sim = np.array(outlet_q) if HAS_JAX else outlet_q
            else:
                runoff, _ = self._model.simulate(self._forcing_tuple, params_obj)
                sim = np.array(runoff) if HAS_JAX else runoff

            sim = sim[self.warmup_days:]
            obs = np.array(self._observations) if HAS_JAX else self._observations

            if obs is None:
                return self.penalty_score

            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs_arr = obs[:min_len]

            if sim.ndim > 1:
                sim = sim[:, 0] if sim.shape[1] > 0 else sim.flatten()

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
        return _evaluate_jfuse_parameters_worker(task_data)


def _evaluate_jfuse_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution."""
    worker = JFUSEWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
