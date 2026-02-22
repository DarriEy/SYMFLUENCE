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
        create_fuse_model, Parameters, PARAM_BOUNDS,
        CoupledModel, create_network_from_topology, load_network,
        FUSEModel, ModelConfig, BaseflowType, UpperLayerArch, LowerLayerArch,
        PercolationType, SurfaceRunoffType, EvaporationType, InterflowType
    )
    from jfuse.fuse.config import SnowType, RoutingType, RainfallErrorType

    # Define JAX-native loss functions locally (jfuse doesn't export them anymore)
    def kge_loss(sim, obs):
        """KGE loss function for jFUSE (1 - KGE, lower is better).

        JAX-compatible implementation using jax.numpy.
        """
        # Ensure inputs are JAX arrays
        sim = jnp.asarray(sim)
        obs = jnp.asarray(obs)

        # Calculate KGE components
        r = jnp.corrcoef(obs, sim)[0, 1]  # Correlation
        alpha = jnp.std(sim) / jnp.std(obs)  # Variability ratio
        beta = jnp.mean(sim) / jnp.mean(obs)  # Bias ratio

        # KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
        kge = 1.0 - jnp.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)

        # Return loss (1 - KGE, lower is better)
        return 1.0 - kge

    def nse_loss(sim, obs):
        """NSE loss function for jFUSE (1 - NSE, lower is better).

        JAX-compatible implementation using jax.numpy.
        """
        # Ensure inputs are JAX arrays
        sim = jnp.asarray(sim)
        obs = jnp.asarray(obs)

        # NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)
        numerator = jnp.sum((obs - sim)**2)
        denominator = jnp.sum((obs - jnp.mean(obs))**2)
        nse = 1.0 - (numerator / denominator)

        # Return loss (1 - NSE, lower is better)
        return 1.0 - nse

    def multi_gauge_kge_loss(Q_all, gauge_indices, gauge_obs, warmup, aggregation='median'):
        """Multi-gauge KGE loss aggregated across gauges.

        Args:
            Q_all: Simulated discharge for all segments (time x segments)
            gauge_indices: Indices mapping gauges to segments
            gauge_obs: Observed discharge per gauge (time x gauges)
            warmup: Number of warmup timesteps to skip
            aggregation: 'median' or 'mean' for aggregating per-gauge losses
        """
        losses = []
        for i, seg_idx in enumerate(gauge_indices):
            sim_g = Q_all[warmup:, seg_idx]
            obs_g = gauge_obs[warmup:, i] if gauge_obs.ndim > 1 else gauge_obs[warmup:]
            valid = ~jnp.isnan(obs_g)
            if jnp.sum(valid) < 10:
                continue
            losses.append(kge_loss(sim_g[valid], obs_g[valid]))
        if not losses:
            return jnp.array(2.0)
        loss_arr = jnp.stack(losses)
        if aggregation == 'median':
            return jnp.median(loss_arr)
        return jnp.mean(loss_arr)

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
    multi_gauge_kge_loss = None
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

        # Model configuration — SymfluenceConfig.from_file() strips JFUSE_ prefix
        # and stores bare keys in model_extra, so try both flat and bare key names
        self.model_config_name = self._get_jfuse_config('model_config_name', 'prms_gradient')
        self.enable_snow = self._get_jfuse_config('enable_snow', True)
        self.spatial_mode = self._get_jfuse_config('spatial_mode', 'lumped')

        # FUSE decision options (shared format with Fortran FUSE)
        self.decision_options = self._get_jfuse_config('decision_options', None)
        if self.decision_options is None:
            # Also check flat config dict key
            self.decision_options = (self.config or {}).get('JFUSE_DECISION_OPTIONS', None)

        # Distributed mode configuration
        self.n_hrus = int(self._get_jfuse_config('n_hrus', 1))
        self.network_file = self._get_jfuse_config('network_file', None)
        self.hru_areas_file = self._get_jfuse_config('hru_areas_file', None)

        self._is_distributed = (
            self.spatial_mode == 'distributed' or
            self.n_hrus > 1 or
            self.network_file is not None
        )

        # JAX configuration
        self.jit_compile = self._get_jfuse_config('jit_compile', True)
        self.use_gpu = self._get_jfuse_config('use_gpu', False)

        if HAS_JAX and not self.use_gpu:
            jax.config.update('jax_platform_name', 'cpu')

        # Initial state configuration
        init_state_mode = self._get_jfuse_config('initial_state', 'default')
        if init_state_mode is None:
            init_state_mode = (self.config or {}).get('JFUSE_INITIAL_STATE', 'default')
        self._initial_state_mode = str(init_state_mode).lower()

        # jFUSE-specific model components
        self._model = None
        self._default_params = None
        self._initial_state = None  # Set during model initialization
        self._coupled_model = None
        self._network = None
        self._hru_areas = None
        self._forcing_tuple = None
        self._network_arrays = None

        # Distributed mode output
        self._last_outlet_q = None

        # Multi-gauge calibration
        self._gauge_obs = None           # [T, G] JAX array, NaN for missing
        self._gauge_reach_indices = None  # [G] JAX array, indices into Q_all
        self._gauge_names = None          # list of gauge names for logging
        self._n_gauges = 0

        # Gradient coverage tracking
        self._gradient_coverage_checked = False
        self._param_warning_logged = False

        # Transfer function state
        self._use_transfer_functions = False
        self._tf_config = None
        self._tf_attr_matrix = None
        self._tf_default_params = None
        self._tf_param_indices = None
        self._tf_lower_bounds = None
        self._tf_upper_bounds = None

    def _get_jfuse_config(self, bare_key: str, default: Any = None) -> Any:
        """Get JFUSE config value, handling both flat and model_extra storage.

        SymfluenceConfig.from_file() strips the JFUSE_ prefix via ModelRegistry
        transformers and stores bare keys in model_extra as {type: value} dicts.
        Direct dict-based configs use flat keys (e.g., 'JFUSE_SPATIAL_MODE').
        This method tries all paths.
        """
        # Try flat key first (works for dict configs and ensure_typed_config)
        flat_key = f'JFUSE_{bare_key.upper()}'
        val = self._cfg(flat_key)
        if val is not None:
            return val

        if hasattr(self.config, 'model_extra') and self.config.model_extra:
            # Try _extra dict (where unknown flat keys land)
            extra = self.config.model_extra.get('_extra', {})
            if isinstance(extra, dict):
                val = extra.get(flat_key)
                if val is not None:
                    return val

            # Try bare key from model_extra (from_file ModelRegistry path)
            # ModelRegistry transformers create {type: value} dicts
            val = self.config.model_extra.get(bare_key)
            if val is not None:
                if isinstance(val, dict) and len(val) == 1:
                    # Extract actual value from {type: value} dict
                    return next(iter(val.values()))
                return val

        return default

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
        return int(self._get_jfuse_config('warmup_days', 365))

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
            outlet_q, runoff = self._coupled_model.simulate(
                self._forcing_tuple, params_obj,
                initial_state=self._initial_state)
            outlet_q_arr = np.array(outlet_q) if HAS_JAX else outlet_q
            runoff_arr = np.array(runoff) if HAS_JAX else runoff
            self._last_outlet_q = outlet_q_arr
            return runoff_arr
        else:
            runoff, _ = self._model.simulate(
                self._forcing_tuple, params_obj,
                initial_state=self._initial_state)
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
        # Check for FUSE decision options first (shared format with Fortran FUSE)
        if self.decision_options and isinstance(self.decision_options, dict):
            custom_config = self._build_config_from_decisions(self.decision_options)
            self._model = FUSEModel(custom_config, n_hrus=1)
            self.logger.info(f"Initialized lumped FUSEModel from FUSE decisions: {self.decision_options}")
        elif self.model_config_name in JFUSE_CONFIGS and JFUSE_CONFIGS[self.model_config_name] is not None:
            custom_config = JFUSE_CONFIGS[self.model_config_name]
            self._model = FUSEModel(custom_config, n_hrus=1)
            self.logger.info(f"Initialized lumped FUSEModel with config: {self.model_config_name}")
        else:
            self._model = create_fuse_model(self.model_config_name, n_hrus=1)
            self.logger.debug(f"Initialized lumped FUSEModel with config: {self.model_config_name}")
        self._default_params = Parameters.default(n_hrus=1)

        # Set initial state based on config
        if self._initial_state_mode == 'zeros':
            default_state = self._model.default_state()
            zero_state = jax.tree.map(lambda x: jnp.zeros_like(x), default_state)
            self._initial_state = zero_state
            self.logger.info("Using zero initial states for jFUSE (JFUSE_INITIAL_STATE=zeros)")
        else:
            self._initial_state = None  # Uses default_state() internally
            self.logger.info("Using default initial states for jFUSE")

    @staticmethod
    def _build_config_from_decisions(decisions: dict) -> 'ModelConfig':
        """Build a jFUSE ModelConfig from Fortran FUSE decision options."""
        # Import the decision map from the runner module
        from symfluence.models.jfuse.runner import FUSE_DECISION_MAP
        resolved = {}
        for key, value in decisions.items():
            if isinstance(value, list):
                resolved[key] = value[0] if value else None
            else:
                resolved[key] = value

        arch2_val = resolved.get('ARCH2', 'fixedsiz_2')

        config_kwargs = {
            'upper_arch': FUSE_DECISION_MAP['ARCH1'].get(
                resolved.get('ARCH1', 'tension1_1'), UpperLayerArch.TENSION_FREE),
            'lower_arch': FUSE_DECISION_MAP['ARCH2'].get(
                arch2_val, LowerLayerArch.SINGLE_NOEVAP),
            'baseflow': FUSE_DECISION_MAP['ARCH2_BASEFLOW'].get(
                arch2_val, BaseflowType.LINEAR),
            'percolation': FUSE_DECISION_MAP['QPERC'].get(
                resolved.get('QPERC', 'perc_f2sat'), PercolationType.FREE_STORAGE),
            'surface_runoff': FUSE_DECISION_MAP['QSURF'].get(
                resolved.get('QSURF', 'arno_x_vic'), SurfaceRunoffType.UZ_PARETO),
            'evaporation': FUSE_DECISION_MAP['ESOIL'].get(
                resolved.get('ESOIL', 'rootweight'), EvaporationType.ROOT_WEIGHT),
            'interflow': FUSE_DECISION_MAP['QINTF'].get(
                resolved.get('QINTF', 'intflwnone'), InterflowType.NONE),
            'snow': FUSE_DECISION_MAP['SNOWM'].get(
                resolved.get('SNOWM', 'temp_index'), SnowType.TEMP_INDEX),
            'routing': FUSE_DECISION_MAP['Q_TDH'].get(
                resolved.get('Q_TDH', 'rout_gamma'), RoutingType.GAMMA),
            'rainfall_error': FUSE_DECISION_MAP['RFERR'].get(
                resolved.get('RFERR', 'multiplc_e'), RainfallErrorType.MULTIPLICATIVE),
        }

        return ModelConfig(**config_kwargs)

    def _initialize_distributed_model(self) -> None:
        """Initialize CoupledModel for distributed mode with routing."""
        domain_name = self._cfg('DOMAIN_NAME', 'domain')
        data_dir = Path(self._cfg('SYMFLUENCE_DATA_DIR', self._cfg('ROOT_PATH', '.')))

        # Load network from topology file
        network_file = self.network_file
        if network_file is None:
            # Auto-detect from settings dir
            network_file = (data_dir / f"domain_{domain_name}" /
                           'settings' / 'mizuRoute' / 'topology.nc')

        if network_file and Path(network_file).exists():
            self.logger.info(f"Loading network from {network_file}")
            self._network, self._hru_areas = load_network(str(network_file))
            self.logger.info(
                f"Network loaded: {self._network.n_reaches} reaches, "
                f"{len(self._hru_areas)} HRU areas"
            )
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
            self._hru_areas = jnp.ones(self.n_hrus) * 1e6

        # Override HRU areas from file if specified
        if self.hru_areas_file and Path(self.hru_areas_file).exists():
            areas_df = pd.read_csv(self.hru_areas_file)
            self._hru_areas = jnp.array(areas_df.iloc[:, 0].values)

        # Get model config
        if self.decision_options and isinstance(self.decision_options, dict):
            fuse_config = self._build_config_from_decisions(self.decision_options)
        elif self.model_config_name in JFUSE_CONFIGS and JFUSE_CONFIGS[self.model_config_name] is not None:
            fuse_config = JFUSE_CONFIGS[self.model_config_name]
        else:
            fuse_config = None

        # Convert network to arrays if it's a RiverNetwork object
        network_arrays = (self._network.to_arrays()
                         if hasattr(self._network, 'to_arrays')
                         else self._network)
        # Store network arrays for multi-gauge reach ID mapping
        self._network_arrays = network_arrays

        # Create CoupledModel
        self._coupled_model = CoupledModel(
            fuse_config=fuse_config,
            network=network_arrays,
            hru_areas=self._hru_areas,
            n_hrus=len(self._hru_areas),
        )
        self._model = self._coupled_model.fuse_model
        self._default_params = self._coupled_model.default_params()
        self.n_hrus = len(self._hru_areas)
        self.logger.info(
            f"Initialized CoupledModel with {self.n_hrus} HRUs, "
            f"{network_arrays.n_reaches} reaches"
        )

        # Initialize transfer functions if enabled
        self._init_transfer_functions()

    def _init_transfer_functions(self) -> None:
        """Initialize transfer functions for spatially varying parameters.

        Reads JFUSE_USE_TRANSFER_FUNCTIONS and JFUSE_ATTRIBUTES_PATH from config,
        instantiates JaxTransferFunctionConfig, and pre-converts arrays to JAX.
        """
        use_tf = self._get_jfuse_config('use_transfer_functions', False)
        if isinstance(use_tf, str):
            use_tf = use_tf.lower() in ('true', '1', 'yes')
        if not use_tf:
            return

        attributes_path = self._get_jfuse_config('attributes_path')
        # Also check flat config key
        if attributes_path is None:
            attributes_path = self._cfg('JFUSE_ATTRIBUTES_PATH')
        if attributes_path is None:
            self.logger.warning(
                "Transfer functions enabled but JFUSE_ATTRIBUTES_PATH not set, "
                "falling back to uniform parameters"
            )
            return

        from symfluence.models.jfuse.calibration.transfer_functions import (
            JaxTransferFunctionConfig,
        )

        # Parse calibrated params from config
        jfuse_params_str = self._cfg('JFUSE_PARAMS_TO_CALIBRATE')
        if jfuse_params_str and jfuse_params_str != 'default':
            calibrated_params = [p.strip() for p in str(jfuse_params_str).split(',') if p.strip()]
        else:
            calibrated_params = None  # uses default 14

        b_bounds_cfg = self._get_jfuse_config('tf_b_bounds', (-5.0, 5.0))
        if isinstance(b_bounds_cfg, (list, tuple)):
            b_bounds_cfg = tuple(b_bounds_cfg)
        else:
            b_bounds_cfg = (-5.0, 5.0)

        self._tf_config = JaxTransferFunctionConfig(
            attributes_path=str(attributes_path),
            calibrated_params=calibrated_params,
            b_bounds=b_bounds_cfg,
            logger=self.logger,
        )

        # Pre-compute JAX arrays for the loss function closure
        self._tf_attr_matrix = self._tf_config.get_attr_matrix_jax()
        self._tf_default_params = self._tf_config.get_default_full_params_jax()
        self._tf_param_indices = self._tf_config.get_param_indices_jax()
        self._tf_lower_bounds = self._tf_config.get_lower_bounds_jax()
        self._tf_upper_bounds = self._tf_config.get_upper_bounds_jax()

        self._use_transfer_functions = True
        self.logger.info(
            f"Transfer functions initialized: {self._tf_config.n_coefficients} "
            f"coefficients → {self._tf_config.n_grus} GRUs × "
            f"{self._tf_config.n_calibrated_params} params"
        )

    # =========================================================================
    # Override Data Loading to Handle jFUSE-specific Requirements
    # =========================================================================

    def initialize(self, task: Optional[WorkerTask] = None) -> bool:
        """Initialize model and load data with jFUSE-specific setup."""
        if getattr(self, "_initialized", False):
            return True

        # Load forcing — use distributed loader if in distributed mode
        if self._is_distributed:
            if not self._load_distributed_forcing(task):
                self.logger.warning("Distributed forcing failed, falling back to base loader")
                if not self._load_forcing(task):
                    return False
        else:
            if not self._load_forcing(task):
                return False

        # Initialize model (needs forcing shape for distributed mode)
        if not self._initialize_model():
            return False

        # Prepare forcing tuple for jFUSE
        self._prepare_forcing_tuple()

        # Load observations — try multi-gauge first for distributed mode
        multi_gauge = self._cfg('MULTI_GAUGE_CALIBRATION', False)
        if self._is_distributed and multi_gauge:
            if self._load_multi_gauge_observations():
                self.logger.info(
                    f"Multi-gauge calibration enabled with {self._n_gauges} gauges"
                )
            else:
                self.logger.warning(
                    "Multi-gauge loading failed, falling back to single-outlet obs"
                )
                if not self._load_observations(task):
                    self.logger.warning("No observations loaded - calibration will fail")
        else:
            if not self._load_observations(task):
                self.logger.warning("No observations loaded - calibration will fail")

        self._initialized = True
        n_timesteps = len(self._forcing['precip']) if self._forcing else 0
        mode_str = "distributed" if self._is_distributed else "lumped"
        gauge_str = f", {self._n_gauges} gauges" if self._n_gauges > 0 else ""
        area_str = f", area={self.get_catchment_area():.1f} km²" if self._catchment_area_km2 else ""
        self.logger.info(
            f"jFUSE worker initialized: {n_timesteps} timesteps, "
            f"{self.n_hrus} HRUs, {mode_str} mode{gauge_str}{area_str}"
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

    def _load_distributed_forcing(self, task: Optional[WorkerTask] = None) -> bool:
        """Load forcing from FUSE input file preserving spatial structure.

        For distributed mode, the base class _load_forcing() flattens spatial
        dims. This method loads the FUSE input NetCDF directly and keeps
        the 2D (time, gru) structure needed for distributed simulation.

        Returns:
            True if loading successful
        """
        import xarray as xr

        data_dir = Path(self._cfg('SYMFLUENCE_DATA_DIR', self._cfg('ROOT_PATH', '.')))
        domain_name = self._cfg('DOMAIN_NAME', 'domain')

        # Look for FUSE input file
        forcing_source = self._get_jfuse_config('forcing_source', 'FUSE_input')
        fuse_input_dir = data_dir / f"domain_{domain_name}" / 'forcing' / forcing_source

        input_patterns = [
            fuse_input_dir / f"{domain_name}_input.nc",
            fuse_input_dir / f"domain_{domain_name}_input.nc",
        ]

        input_file = None
        for p in input_patterns:
            if p.exists():
                input_file = p
                break

        if input_file is None:
            # Fall back to glob
            nc_files = list(fuse_input_dir.glob("*_input.nc")) if fuse_input_dir.exists() else []
            if nc_files:
                input_file = nc_files[0]

        if input_file is None:
            self.logger.error(f"No FUSE input file found in {fuse_input_dir}")
            return False

        self.logger.info(f"Loading distributed forcing from {input_file}")
        ds = xr.open_dataset(input_file)

        # Read forcing variables - shape (time, latitude=1, longitude=7618)
        try:
            precip = ds['precip'].values
            temp = ds['temp'].values
            pet = ds['pet'].values
        except KeyError as e:
            self.logger.error(f"Missing variable in forcing file: {e}")
            ds.close()
            return False

        # Squeeze singleton latitude dimension if present
        if precip.ndim == 3 and precip.shape[1] == 1:
            precip = precip.squeeze(axis=1)
            temp = temp.squeeze(axis=1)
            pet = pet.squeeze(axis=1)

        # Load mapping file to filter to non-coastal GRUs
        mapping_file = self._get_jfuse_config('mapping_file')
        if mapping_file is None:
            mapping_file = (data_dir / f"domain_{domain_name}" /
                           'settings' / 'mizuRoute' / 'fuse_to_routing_mapping.csv')

        if Path(mapping_file).exists():
            mapping_df = pd.read_csv(mapping_file)
            # Filter to non-coastal GRUs
            non_coastal_df = mapping_df[mapping_df['is_coastal'] == False]  # noqa: E712
            non_coastal_indices = np.asarray(non_coastal_df['fuse_gru_idx'].values, dtype=int)
            n_non_coastal = len(non_coastal_indices)
            precip = precip[:, non_coastal_indices]
            temp = temp[:, non_coastal_indices]
            pet = pet[:, non_coastal_indices]
            self.logger.info(
                f"Filtered to {n_non_coastal} non-coastal GRUs "
                f"(from {ds.sizes.get('longitude', '?')} total)"
            )
        else:
            # Assume first 6600 are non-coastal
            n_non_coastal = int(self._get_jfuse_config('n_non_coastal', 6600))
            if precip.shape[1] > n_non_coastal:
                precip = precip[:, :n_non_coastal]
                temp = temp[:, :n_non_coastal]
                pet = pet[:, :n_non_coastal]
                self.logger.info(
                    f"Using first {n_non_coastal} GRUs as non-coastal (no mapping file)"
                )

        # Store time index
        if 'time' in ds.coords:
            self._time_index = pd.to_datetime(ds.time.values)

        ds.close()

        # Store as forcing dict preserving 2D shape
        self._forcing = {
            'precip': precip,
            'temp': temp,
            'pet': pet,
        }

        self.n_hrus = precip.shape[1]
        self.logger.info(
            f"Distributed forcing loaded: shape {precip.shape} "
            f"({precip.shape[0]} timesteps, {precip.shape[1]} GRUs)"
        )
        return True

    def _load_multi_gauge_observations(self) -> bool:
        """Load multi-gauge observations for distributed calibration.

        Reads gauge-to-segment mapping and LamaH-Ice observation files,
        aligns to simulation period, and builds JAX arrays for the
        multi_gauge_kge_loss function.

        Returns:
            True if sufficient gauges loaded
        """
        gauge_mapping_file = self._cfg('GAUGE_SEGMENT_MAPPING')
        obs_dir = self._cfg('MULTI_GAUGE_OBS_DIR')
        min_gauges = int(self._cfg('MULTI_GAUGE_MIN_GAUGES', 5))
        exclude_ids = self._cfg('MULTI_GAUGE_EXCLUDE_IDS', [])

        # Quality filters
        max_distance = self._cfg('MULTI_GAUGE_MAX_DISTANCE')
        min_obs_cv = self._cfg('MULTI_GAUGE_MIN_OBS_CV')
        min_specific_q = self._cfg('MULTI_GAUGE_MIN_SPECIFIC_Q')
        if max_distance is not None:
            max_distance = float(max_distance)
        if min_obs_cv is not None:
            min_obs_cv = float(min_obs_cv)
        if min_specific_q is not None:
            min_specific_q = float(min_specific_q)

        if not gauge_mapping_file or not Path(gauge_mapping_file).exists():
            self.logger.warning(f"Gauge mapping file not found: {gauge_mapping_file}")
            return False

        if not obs_dir or not Path(obs_dir).exists():
            self.logger.warning(f"Observation directory not found: {obs_dir}")
            return False

        # Read gauge-segment mapping
        gauge_df = pd.read_csv(gauge_mapping_file)
        self.logger.info(f"Loaded gauge mapping with {len(gauge_df)} gauges")

        # Get simulation time index
        if self._time_index is None:
            self.logger.warning("No time index available for gauge alignment")
            return False

        sim_dates = pd.to_datetime(self._time_index).normalize()
        n_timesteps = len(sim_dates)

        # Get network reach IDs in topological order for index mapping
        if not hasattr(self, '_network_arrays') or self._network_arrays is None:
            self.logger.warning("Network arrays not available, cannot map gauge segments")
            return False

        reach_ids = np.array(self._network_arrays.reach_ids)

        # Build reach_id -> topological index map
        reach_id_to_idx = {int(rid): idx for idx, rid in enumerate(reach_ids)}

        gauge_obs_list = []
        gauge_idx_list = []
        gauge_name_list = []

        for _, row in gauge_df.iterrows():
            gauge_id = int(row.get('id', row.get('ID', -1)))
            gauge_name = row.get('name', f'gauge_{gauge_id}')
            nearest_seg = int(row.get('nearest_segment', -1))

            # Skip excluded gauges
            if gauge_id in exclude_ids:
                continue

            # Distance filter
            if max_distance is not None:
                dist = float(row.get('distance_to_segment', 0.0))
                if dist > max_distance:
                    self.logger.info(
                        f"Gauge {gauge_name} (id={gauge_id}): excluded — "
                        f"distance {dist:.4f}° > {max_distance}°"
                    )
                    continue

            # Map segment to topological index
            if nearest_seg not in reach_id_to_idx:
                self.logger.debug(f"Gauge {gauge_name}: segment {nearest_seg} not in network")
                continue
            topo_idx = reach_id_to_idx[nearest_seg]

            # Load observation file (LamaH-Ice format: ID_{id}.csv)
            obs_file = Path(obs_dir) / f"ID_{gauge_id}.csv"
            if not obs_file.exists():
                self.logger.debug(f"Observation file not found: {obs_file}")
                continue

            try:
                obs_df = pd.read_csv(obs_file, sep=';')
                obs_df['date'] = pd.to_datetime(
                    obs_df[['YYYY', 'MM', 'DD']].rename(
                        columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
                    )
                )
                obs_df = obs_df.set_index('date')

                # Get discharge column
                qobs = obs_df['qobs'].copy()
                qobs[qobs < 0] = np.nan  # negative = missing

                # Align to simulation period
                qobs_aligned = qobs.reindex(sim_dates)
                n_valid = qobs_aligned.notna().sum()

                # Require at least 30% overlap
                if n_valid < n_timesteps * 0.3:
                    self.logger.debug(
                        f"Gauge {gauge_name}: insufficient overlap "
                        f"({n_valid}/{n_timesteps})"
                    )
                    continue

                # Observation CV filter (exclude glacier-buffered / near-constant)
                if min_obs_cv is not None:
                    valid_q = qobs_aligned.dropna()
                    if len(valid_q) > 10 and valid_q.mean() > 0:
                        cv = float(valid_q.std() / valid_q.mean())
                        if cv < min_obs_cv:
                            self.logger.info(
                                f"Gauge {gauge_name} (id={gauge_id}): excluded — "
                                f"obs CV={cv:.3f} < {min_obs_cv}"
                            )
                            continue

                # Specific discharge filter (exclude groundwater-loss catchments)
                if min_specific_q is not None:
                    area_km2 = float(row.get('area_calc', 0.0))
                    if area_km2 > 0:
                        valid_q = qobs_aligned.dropna()
                        mean_q_m3s = float(valid_q.mean()) if len(valid_q) > 0 else 0.0
                        # Specific Q in mm/yr: Q [m³/s] * 86400 * 365.25 / (area [km²] * 1e6) * 1000
                        specific_q = mean_q_m3s * 86400 * 365.25 / (area_km2 * 1e6) * 1000
                        if specific_q < min_specific_q:
                            self.logger.info(
                                f"Gauge {gauge_name} (id={gauge_id}): excluded — "
                                f"specific Q={specific_q:.0f} mm/yr < {min_specific_q}"
                            )
                            continue

                gauge_obs_list.append(qobs_aligned.values)
                gauge_idx_list.append(topo_idx)
                gauge_name_list.append(str(gauge_name))

            except Exception as e:
                self.logger.debug(f"Error loading gauge {gauge_name}: {e}")
                continue

        if len(gauge_obs_list) < min_gauges:
            self.logger.warning(
                f"Only {len(gauge_obs_list)} gauges loaded, "
                f"need at least {min_gauges}"
            )
            return False

        # Build JAX arrays
        # gauge_obs: [T, G] with NaN for missing
        gauge_obs_np = np.column_stack(gauge_obs_list)  # [T, G]
        self._gauge_obs = jnp.array(gauge_obs_np)
        self._gauge_reach_indices = jnp.array(gauge_idx_list, dtype=jnp.int32)
        self._gauge_names = gauge_name_list
        self._n_gauges = len(gauge_name_list)

        self.logger.info(
            f"Multi-gauge observations loaded: {self._n_gauges} gauges "
            f"({n_timesteps} timesteps)"
        )
        for i, name in enumerate(gauge_name_list):
            n_valid = int(np.sum(~np.isnan(gauge_obs_np[:, i])))
            self.logger.debug(
                f"  Gauge {name}: segment {gauge_idx_list[i]}, "
                f"{n_valid}/{n_timesteps} valid obs"
            )

        return True

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
                init_state = self._initial_state

                def loss_fn(val, pn=param_name):
                    params = default_params
                    params = eqx.tree_at(lambda p, n=pn: getattr(p, n), params, val)
                    runoff, _ = fuse_model.simulate(
                        forcing_tuple, params, initial_state=init_state)
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

    def _build_array_to_params(self, param_names):
        """Build a reusable array_to_params closure for gradient functions.

        If transfer functions are enabled, delegates to _build_array_to_params_tf().
        """
        if self._use_transfer_functions:
            return self._build_array_to_params_tf(param_names)

        default_params = self._default_params
        is_distributed = self._is_distributed
        n_hrus = self.n_hrus

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

        return array_to_params

    def _build_array_to_params_tf(self, coeff_names):
        """Build array_to_params closure using transfer functions.

        Takes a coefficient array (28,) and produces CoupledParams with
        per-GRU spatially varying FUSE parameters via apply_transfer_functions().

        Args:
            coeff_names: List of coefficient names (e.g. ['S1_max_a', 'S1_max_b', ...]).
                Only used for logging; the actual mapping is in _tf_config.

        Returns:
            Closure: coeff_array (28,) -> CoupledParams
        """
        from symfluence.models.jfuse.calibration.transfer_functions import (
            apply_transfer_functions,
        )
        from jfuse.fuse.state import Parameters

        default_coupled_params = self._default_params
        attr_matrix = self._tf_attr_matrix
        default_full_params = self._tf_default_params
        param_indices = self._tf_param_indices
        lower_bounds = self._tf_lower_bounds
        upper_bounds = self._tf_upper_bounds
        n_grus = self._tf_config.n_grus

        def array_to_params(coeff_array):
            # Apply transfer functions: (28,) -> (n_grus, 30)
            full_params_2d = apply_transfer_functions(
                coeff_array, attr_matrix, default_full_params,
                param_indices, lower_bounds, upper_bounds, n_grus,
            )

            # Convert to jFUSE Parameters (per-HRU)
            fuse_params = Parameters.from_array_validated(
                full_params_2d, n_hrus=n_grus, clip=True
            )

            # Wrap in CoupledParams with default routing params
            p = eqx.tree_at(
                lambda x: x.fuse_params, default_coupled_params, fuse_params
            )
            return p

        return array_to_params

    def _build_loss_fn(self, param_names, metric='kge'):
        """Build loss function for gradient computation.

        Supports multi-gauge (distributed + gauge obs) and single-outlet paths.
        """
        array_to_params = self._build_array_to_params(param_names)
        forcing_tuple = self._forcing_tuple
        warmup = self.warmup_days
        is_distributed = self._is_distributed
        coupled_model = self._coupled_model
        fuse_model = self._model
        initial_state = self._initial_state
        has_multi_gauge = self._gauge_obs is not None and self._gauge_reach_indices is not None
        gauge_obs = self._gauge_obs
        gauge_indices = self._gauge_reach_indices
        aggregation = self._cfg('MULTI_GAUGE_AGGREGATION', 'median')

        if has_multi_gauge:
            obs = gauge_obs
        elif self._observations is not None:
            obs = jnp.array(self._observations)
        else:
            obs = None

        def loss_from_array(param_array):
            params_obj = array_to_params(param_array)
            if is_distributed and has_multi_gauge:
                # Multi-gauge path: simulate_full -> multi_gauge_kge_loss
                _, Q_all, _ = coupled_model.simulate_full(forcing_tuple, params_obj)
                return multi_gauge_kge_loss(
                    Q_all, gauge_indices, obs, warmup, aggregation
                )
            elif is_distributed:
                # Single outlet path
                outlet_q, _ = coupled_model.simulate(
                    forcing_tuple, params_obj, initial_state=initial_state)
                sim_eval = outlet_q[warmup:]
            else:
                # Lumped path
                runoff, _ = fuse_model.simulate(
                    forcing_tuple, params_obj, initial_state=initial_state)
                sim_eval = runoff[warmup:]

            assert obs is not None, "Observations required for single-gauge loss"
            obs_aligned = obs[:len(sim_eval)]
            if metric.lower() == 'nse':
                return nse_loss(sim_eval[:len(obs_aligned)], obs_aligned)
            return kge_loss(sim_eval[:len(obs_aligned)], obs_aligned)

        return loss_from_array

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

        if self._observations is None and self._gauge_obs is None:
            return None

        try:
            param_names = list(params.keys())
            loss_fn = self._build_loss_fn(param_names, metric)

            param_array = jnp.array([params[name] for name in param_names])
            grad_fn = jax.grad(loss_fn)
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

        if self._observations is None and self._gauge_obs is None:
            raise ValueError("No observations available")

        # Check gradient coverage once (skip for multi-gauge and transfer functions)
        if (not self._gradient_coverage_checked
                and self._gauge_obs is None
                and not self._use_transfer_functions):
            self._gradient_coverage_checked = True
            self.check_gradient_coverage(list(params.keys()))

        try:
            param_names = list(params.keys())
            loss_fn = self._build_loss_fn(param_names, metric)

            param_array = jnp.array([params[name] for name in param_names])
            val_and_grad_fn = jax.value_and_grad(loss_fn)
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

            # Multi-gauge evaluation path
            if self._is_distributed and self._gauge_obs is not None:
                _, Q_all, _ = self._coupled_model.simulate_full(
                    self._forcing_tuple, params_obj
                )
                aggregation = self._get_config_value(
                    lambda: None, default='median', dict_key='MULTI_GAUGE_AGGREGATION')
                loss_val = float(multi_gauge_kge_loss(
                    Q_all, self._gauge_reach_indices, self._gauge_obs,
                    self.warmup_days, aggregation
                ))
                # Return 1 - loss (KGE value, higher is better)
                return 1.0 - loss_val

            # Single outlet / lumped evaluation path
            if self._is_distributed:
                outlet_q, runoff = self._coupled_model.simulate(
                    self._forcing_tuple, params_obj,
                    initial_state=self._initial_state)
                sim = np.array(outlet_q) if HAS_JAX else outlet_q
            else:
                runoff, _ = self._model.simulate(
                    self._forcing_tuple, params_obj,
                    initial_state=self._initial_state)
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
