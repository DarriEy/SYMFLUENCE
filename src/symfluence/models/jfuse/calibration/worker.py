"""
jFUSE Calibration Worker with Native Gradient Support.

Provides the JFUSEWorker class for parameter evaluation during optimization.
Supports both finite-difference and native JAX autodiff gradients.
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
from symfluence.core.constants import UnitConversion

# Lazy JAX and jFUSE imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, value_and_grad
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None
    grad = None
    value_and_grad = None

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
    # Uses NONLINEAR baseflow to enable gradients for ks, S2_max, and n
    # Uses UZ_PARETO surface runoff to enable gradients for S1_max
    # Working parameters (14): S1_max, S2_max, ku, ki, ks, n, Ac_max, b, f_rchr,
    #                          T_rain, T_melt, MFMAX, MFMIN, smooth_frac
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

    # Maximum gradient config - Sacramento-based architecture for most parameters
    # This configuration activates the MAXIMUM number of calibratable parameters:
    # Active parameters (21):
    #   Storage: S1_max, S2_max, f_tens (derived), f_rchr, f_base
    #   Evaporation: r1 (ROOT_WEIGHT)
    #   Percolation: ku, alpha, psi (LOWER_DEMAND)
    #   Lower layer: kappa (TENSION_2RESERV)
    #   Interflow: ki (LINEAR)
    #   Baseflow: v_A, v_B (PARALLEL_LINEAR)
    #   Surface runoff: Ac_max, b (UZ_PARETO)
    #   Snow: T_rain, T_melt, MFMAX, MFMIN
    #   Overflow: smooth_frac
    # NOT active (mutually exclusive or unused):
    #   c (needs TOTAL_STORAGE percolation), ks, n, v (different baseflow types)
    #   chi (needs LZ_GAMMA), lam, mu_t, lapse_rate, opg (not used in lumped mode)
    MAX_GRADIENT_CONFIG = ModelConfig(
        upper_arch=UpperLayerArch.TENSION2_FREE,      # f_tens, f_rchr, smooth_frac
        lower_arch=LowerLayerArch.TENSION_2RESERV,    # f_base, kappa, lower evap
        baseflow=BaseflowType.PARALLEL_LINEAR,        # v_A, v_B (forced by TENSION_2RESERV)
        percolation=PercolationType.LOWER_DEMAND,     # ku, alpha, psi
        surface_runoff=SurfaceRunoffType.UZ_PARETO,   # Ac_max, b, S1_max
        evaporation=EvaporationType.ROOT_WEIGHT,      # r1
        interflow=InterflowType.LINEAR,               # ki
        snow=SnowType.TEMP_INDEX,                     # T_rain, T_melt, MFMAX, MFMIN
        routing=RoutingType.NONE,
        rainfall_error=RainfallErrorType.ADDITIVE,
    )

    # Extended config map including custom gradient-optimized configs
    JFUSE_CONFIGS = {
        'prms': None,  # Use jfuse's default
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
class JFUSEWorker(BaseWorker):
    """
    Worker for jFUSE model evaluation with native gradient support.

    Key Features:
    - Native gradient computation via JAX autodiff (when available)
    - Support for both lumped and distributed modes
    - Efficient value_and_grad for combined loss and gradient computation
    - Falls back to finite differences when JAX unavailable

    The worker maintains a cached JAX-compiled model for efficiency.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize jFUSE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Check dependencies
        if not HAS_JFUSE:
            self.logger.warning("jFUSE not installed. Model execution will fail.")

        # Model configuration - default to prms_gradient for full gradient support
        self.model_config_name = self.config.get('JFUSE_MODEL_CONFIG_NAME', 'prms_gradient')
        self.enable_snow = self.config.get('JFUSE_ENABLE_SNOW', True)
        self.warmup_days = int(self.config.get('JFUSE_WARMUP_DAYS', 365))
        self.spatial_mode = self.config.get('JFUSE_SPATIAL_MODE', 'lumped')

        # Distributed mode configuration
        self.n_hrus = int(self.config.get('JFUSE_N_HRUS', 1))
        self.network_file = self.config.get('JFUSE_NETWORK_FILE', None)
        self.hru_areas_file = self.config.get('JFUSE_HRU_AREAS_FILE', None)

        # Determine if using distributed mode
        self._is_distributed = (
            self.spatial_mode == 'distributed' or
            self.n_hrus > 1 or
            self.network_file is not None
        )

        # JAX configuration
        self.jit_compile = self.config.get('JFUSE_JIT_COMPILE', True)
        self.use_gpu = self.config.get('JFUSE_USE_GPU', False)

        # Configure JAX device
        if HAS_JAX and not self.use_gpu:
            jax.config.update('jax_platform_name', 'cpu')

        # Cached model and data
        self._model = None  # FUSEModel or CoupledModel
        self._forcing = None
        self._observations = None
        self._loss_fn = None
        self._grad_fn = None
        self._value_and_grad_fn = None
        self._initialized = False

        # Distributed mode specific
        self._network = None
        self._hru_areas = None
        self._coupled_model = None

        # Catchment area for unit conversion (mm/day to m³/s)
        self._catchment_area_km2 = None

        # Cache for simulation results
        self._last_params = None
        self._last_runoff = None
        self._last_outlet_q = None  # For distributed mode

        # Gradient coverage tracking (for warning about zero-gradient params)
        self._gradient_coverage_checked = False

    def supports_native_gradients(self) -> bool:
        """
        Check if native gradient computation is available.

        Returns:
            True if JAX and jFUSE are both installed.
        """
        return HAS_JAX and HAS_JFUSE

    def _initialize_model_and_data(self, task: Optional[WorkerTask] = None) -> bool:
        """
        Initialize jFUSE model and load forcing/observation data.

        Supports both lumped mode (FUSEModel) and distributed mode (CoupledModel).

        Args:
            task: Optional task containing paths

        Returns:
            True if initialization successful.
        """
        if self._initialized:
            return True

        if not HAS_JFUSE:
            self.logger.error("jFUSE not installed. Cannot initialize model.")
            return False

        try:
            # Load forcing data first to determine n_hrus
            forcing_dir = self._get_forcing_dir(task)
            domain_name = self.config.get('DOMAIN_NAME', 'domain')

            # Load forcing data
            precip, temp, pet, self._time_index = self._load_forcing(forcing_dir, domain_name)

            # Determine number of HRUs from forcing shape
            if precip.ndim == 1:
                actual_n_hrus = 1
                # Reshape to 2D for consistency
                precip = precip.reshape(-1, 1)
                temp = temp.reshape(-1, 1)
                pet = pet.reshape(-1, 1)
            else:
                actual_n_hrus = precip.shape[1]

            # Update n_hrus if auto-detected
            if self.n_hrus == 1 and actual_n_hrus > 1:
                self.n_hrus = actual_n_hrus
                self._is_distributed = True
                self.logger.info(f"Auto-detected {self.n_hrus} HRUs, using distributed mode")

            # Initialize model based on mode
            if self._is_distributed:
                self._initialize_distributed_model(forcing_dir, domain_name)
            else:
                self._initialize_lumped_model()

            # Store forcing as JAX arrays
            if HAS_JAX:
                self._forcing = {
                    'precip': jnp.array(precip),
                    'temp': jnp.array(temp),
                    'pet': jnp.array(pet),
                }
            else:
                self._forcing = {
                    'precip': precip,
                    'temp': temp,
                    'pet': pet,
                }

            # Load observations
            self._load_observations(forcing_dir, domain_name)

            # Load catchment area for unit conversion (mm/day to m³/s)
            # This is critical for comparing simulated runoff to observed streamflow
            self._get_catchment_area()

            # Create loss and gradient functions if JAX available
            if HAS_JAX and self._observations is not None:
                self._create_jax_functions()

            self._initialized = True
            n_timesteps = precip.shape[0]
            mode_str = "distributed" if self._is_distributed else "lumped"
            area_str = f", area={self._catchment_area_km2:.1f} km²" if self._catchment_area_km2 else ""
            self.logger.info(
                f"jFUSE worker initialized: {n_timesteps} timesteps, "
                f"{self.n_hrus} HRUs, {mode_str} mode{area_str}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize jFUSE worker: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _get_forcing_dir(self, task: Optional[WorkerTask] = None) -> Path:
        """Get path to forcing directory."""
        if task and task.settings_dir:
            # Look relative to settings dir
            parent = task.settings_dir.parent.parent if task.settings_dir.parent else task.settings_dir
            forcing_dir = parent / 'forcing' / 'JFUSE_input'
            if forcing_dir.exists():
                return forcing_dir

        # Fall back to config-based path using SYMFLUENCE_DATA_DIR
        data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR', self.config.get('ROOT_PATH', '.')))
        domain_name = self.config.get('DOMAIN_NAME', 'domain')
        return data_dir / f"domain_{domain_name}" / 'forcing' / 'JFUSE_input'

    def _load_forcing(
        self,
        forcing_dir: Path,
        domain_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Load forcing data from NetCDF or CSV files.

        Args:
            forcing_dir: Directory containing forcing files
            domain_name: Domain name for file naming

        Returns:
            Tuple of (precip, temp, pet, time_index) arrays
        """
        # Try NetCDF first
        nc_file = forcing_dir / f"{domain_name}_jfuse_forcing.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            precip = ds['precip'].values
            temp = ds['temp'].values
            pet = ds['pet'].values
            time_index = pd.to_datetime(ds.time.values)
            ds.close()
            return precip, temp, pet, time_index

        # Try CSV
        csv_file = forcing_dir / f"{domain_name}_jfuse_forcing.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            precip = df['precip'].values
            temp = df['temp'].values
            pet = df['pet'].values
            time_index = pd.to_datetime(df['time'])
            return precip, temp, pet, time_index

        raise FileNotFoundError(f"No forcing file found: {nc_file} or {csv_file}")

    def _initialize_lumped_model(self) -> None:
        """Initialize FUSEModel for lumped mode (single HRU)."""
        # Check if using a custom config (e.g., prms_gradient)
        if self.model_config_name in JFUSE_CONFIGS and JFUSE_CONFIGS[self.model_config_name] is not None:
            custom_config = JFUSE_CONFIGS[self.model_config_name]
            self._model = FUSEModel(custom_config, n_hrus=1)
            self.logger.info(f"Initialized lumped FUSEModel with custom config: {self.model_config_name}")
        else:
            # Use jfuse's built-in configs
            self._model = create_fuse_model(self.model_config_name, n_hrus=1)
            self.logger.debug(f"Initialized lumped FUSEModel with config: {self.model_config_name}")
        self._default_params = Parameters.default(n_hrus=1)

    def _initialize_distributed_model(self, forcing_dir: Path, domain_name: str) -> None:
        """
        Initialize CoupledModel for distributed mode with routing.

        Args:
            forcing_dir: Directory containing network files
            domain_name: Domain name for file naming
        """
        # Load or create network
        network_file = self.network_file
        if network_file is None:
            # Try default location
            network_file = forcing_dir / f"{domain_name}_network.nc"

        if network_file and Path(network_file).exists():
            self.logger.info(f"Loading network from {network_file}")
            self._network = load_network(str(network_file))
        else:
            self.logger.warning(
                f"No network file found at {network_file}. "
                "Creating simple sequential network."
            )
            # Create a simple sequential network
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

        # Load or create HRU areas
        if self.hru_areas_file and Path(self.hru_areas_file).exists():
            areas_df = pd.read_csv(self.hru_areas_file)
            self._hru_areas = jnp.array(areas_df.iloc[:, 0].values)
        else:
            # Default: equal areas of 1 km² each
            self._hru_areas = jnp.ones(self.n_hrus) * 1e6  # m²

        # Create CoupledModel
        self._coupled_model = CoupledModel(
            network=self._network.to_arrays(),
            hru_areas=self._hru_areas,
            n_hrus=self.n_hrus
        )
        self._model = self._coupled_model.fuse_model
        self._default_params = self._coupled_model.default_params()

        self.logger.debug(
            f"Initialized CoupledModel with {self.n_hrus} HRUs and routing"
        )

    def _load_observations(self, forcing_dir: Path, domain_name: str) -> None:
        """Load observation data for calibration."""
        obs_file = forcing_dir / f"{domain_name}_observations.csv"
        if obs_file.exists():
            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # Detect timestep and resample to daily if needed (jFUSE runs at daily timestep)
            if len(obs_df) > 1:
                time_diff = obs_df.index[1] - obs_df.index[0]
                if time_diff < pd.Timedelta(days=1):
                    # Hourly or sub-daily data - resample to daily mean
                    self.logger.info(
                        f"Resampling observations from {time_diff} to daily "
                        f"({len(obs_df)} → {len(obs_df.resample('D').mean().dropna())} values)"
                    )
                    obs_df = obs_df.resample('D').mean().dropna()

            obs = obs_df.iloc[:, 0].values

            # Observations are in m³/s, convert to mm/day to match jFUSE internal units
            # Formula: Q(mm/day) = Q(m³/s) × 86.4 / area(km²)
            area_km2 = self._get_catchment_area()
            obs_mm_day = obs * UnitConversion.MM_DAY_TO_CMS / area_km2
            self.logger.info(
                f"Converted observations from m³/s to mm/day "
                f"(area={area_km2:.1f} km², mean obs: {np.mean(obs):.2f} m³/s → {np.mean(obs_mm_day):.2f} mm/day)"
            )

            # Handle warmup alignment (skip warmup_days worth of daily values)
            if len(obs_mm_day) > self.warmup_days:
                obs_mm_day = obs_mm_day[self.warmup_days:]

            if HAS_JAX:
                self._observations = jnp.array(obs_mm_day)
            else:
                self._observations = obs_mm_day

            self.logger.info(f"Loaded {len(obs_mm_day)} daily observations for calibration")
        else:
            self.logger.warning(f"No observation file found: {obs_file}")
            self._observations = None

    def _get_catchment_area(self) -> float:
        """
        Get catchment area in km² for unit conversion (mm/day to m³/s).

        Tries multiple sources in order:
        1. Shapefile in project directory
        2. Config value (CATCHMENT_AREA_KM2 or domain.catchment_area_km2)
        3. Default fallback (1000 km²)

        Returns:
            Catchment area in km²
        """
        # Return cached value if available
        if self._catchment_area_km2 is not None:
            return self._catchment_area_km2

        # Try reading from shapefile
        try:
            import geopandas as gpd
            data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR', self.config.get('ROOT_PATH', '.')))
            domain_name = self.config.get('DOMAIN_NAME', 'domain')
            catchment_dir = data_dir / f"domain_{domain_name}" / 'shapefiles' / 'catchment'

            # Try common shapefile patterns
            for pattern in ['*_HRUs_*.shp', '*_catchment*.shp', '*.shp']:
                shp_files = list(catchment_dir.glob(pattern))
                if shp_files:
                    gdf = gpd.read_file(shp_files[0])
                    area_cols = [c for c in gdf.columns if 'area' in c.lower()]
                    if area_cols:
                        # Area in shapefile is typically in m², convert to km²
                        total_area_m2 = gdf[area_cols[0]].sum()
                        self._catchment_area_km2 = float(total_area_m2) / 1e6
                        self.logger.info(f"Catchment area from shapefile: {self._catchment_area_km2:.2f} km²")
                        return self._catchment_area_km2
        except Exception as e:
            self.logger.debug(f"Could not read catchment area from shapefile: {e}")

        # Try config values
        area_km2 = self.config.get('CATCHMENT_AREA_KM2')
        if area_km2 is None:
            # Try nested config structure
            domain_config = self.config.get('DOMAIN', {})
            if isinstance(domain_config, dict):
                area_km2 = domain_config.get('catchment_area_km2')

        if area_km2 is not None:
            self._catchment_area_km2 = float(area_km2)
            self.logger.info(f"Catchment area from config: {self._catchment_area_km2:.2f} km²")
            return self._catchment_area_km2

        # Default fallback
        self.logger.warning("Could not determine catchment area, using default 1000 km²")
        self._catchment_area_km2 = 1000.0
        return self._catchment_area_km2

    def _convert_runoff_to_cms(self, runoff):
        """
        Convert runoff from mm/day to m³/s.

        Formula: Q(m³/s) = runoff(mm/day) × area(km²) / 86.4

        Args:
            runoff: Runoff array in mm/day (JAX or numpy array)

        Returns:
            Streamflow in m³/s
        """
        area_km2 = self._get_catchment_area()
        return runoff * area_km2 / UnitConversion.MM_DAY_TO_CMS

    def _create_jax_functions(self) -> None:
        """Create JIT-compiled loss and gradient functions."""
        if not HAS_JAX or self._model is None:
            return

        # jFUSE expects forcing as tuple: (precip, pet, temp)
        # For distributed mode, shape is (n_timesteps, n_hrus)
        # For lumped mode, shape is (n_timesteps,) or (n_timesteps, 1)
        precip = self._forcing['precip']
        pet = self._forcing['pet']
        temp = self._forcing['temp']

        # Squeeze if lumped mode with shape (n_timesteps, 1)
        if not self._is_distributed and precip.ndim > 1 and precip.shape[1] == 1:
            precip = precip.squeeze(-1)
            pet = pet.squeeze(-1)
            temp = temp.squeeze(-1)

        self._forcing_tuple = (precip, pet, temp)
        self.logger.info("JAX loss and gradient functions ready")

    def check_gradient_coverage(
        self,
        param_names: list,
        epsilon: float = 1e-6
    ) -> Dict[str, bool]:
        """
        Check which parameters have non-zero gradients for gradient-based optimization.

        Some jFUSE model structures have parameters with zero gradients due to
        model architecture (e.g., storage capacity parameters in PRMS when storage
        never fills). This method identifies which parameters can be effectively
        optimized with gradient-based methods like ADAM.

        Args:
            param_names: List of parameter names to check
            epsilon: Threshold below which gradient is considered zero

        Returns:
            Dict mapping parameter name to True if gradient is non-zero
        """
        if not self._initialized:
            self._initialize_model_and_data()

        if not HAS_JAX or self._model is None or self._observations is None:
            return {name: True for name in param_names}  # Assume all work if can't check

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

                # Create loss function for single parameter
                forcing_tuple = self._forcing_tuple
                obs = self._observations
                warmup = self.warmup_days
                fuse_model = self._model
                default_params = self._default_params

                def loss_fn(val, pn=param_name):
                    """Loss function capturing param_name by value."""
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
                gradient_status[param_name] = True  # Assume works if check fails

        # Log warnings for zero-gradient parameters
        if zero_grad_params:
            self.logger.warning(
                f"⚠️  GRADIENT WARNING: {len(zero_grad_params)} parameters have zero gradients "
                f"and cannot be optimized with ADAM: {zero_grad_params}"
            )
            self.logger.warning(
                f"   Parameters with working gradients: {working_params}"
            )
            self.logger.warning(
                "   Options: 1) Use JFUSE_PARAMS_TO_CALIBRATE to only calibrate working params, "
                "2) Switch to TOPMODEL structure, or 3) Use DDS optimizer instead"
            )

        return gradient_status

    def _dict_to_params(self, param_dict: Dict[str, float]) -> Any:
        """
        Convert a parameter dictionary to a jFUSE Parameters or CoupledParams object.

        For lumped mode, updates Parameters directly.
        For distributed mode, updates fuse_params within CoupledParams.

        Args:
            param_dict: Dictionary mapping parameter names to values

        Returns:
            jFUSE Parameters (lumped) or CoupledParams (distributed) object
        """
        params = self._default_params

        # Debug: log parameter matching
        matched = []
        unmatched = []
        for name in param_dict.keys():
            if hasattr(params, name):
                matched.append(name)
            else:
                unmatched.append(name)

        if unmatched and not hasattr(self, '_param_warning_logged'):
            self._param_warning_logged = True
            self.logger.warning(
                f"jFUSE parameter mismatch - Matched: {matched}, "
                f"Unmatched (will use defaults): {unmatched}"
            )


        if self._is_distributed:
            # For CoupledParams, update the fuse_params attribute
            fuse_params = params.fuse_params
            for name, value in param_dict.items():
                if hasattr(fuse_params, name):
                    # Broadcast scalar to all HRUs
                    # Use default argument to capture name by value (avoids closure issues)
                    arr = jnp.ones(self.n_hrus) * float(value)
                    fuse_params = eqx.tree_at(
                        lambda p, n=name: getattr(p, n), fuse_params, arr
                    )
            # Update CoupledParams with new fuse_params
            params = eqx.tree_at(lambda p: p.fuse_params, params, fuse_params)
        else:
            # For lumped mode, update Parameters directly
            for name, value in param_dict.items():
                if hasattr(params, name):
                    # Use default argument to capture name by value (avoids closure issues)
                    params = eqx.tree_at(
                        lambda p, n=name: getattr(p, n), params, jnp.array(float(value))
                    )

        return params

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
        Apply parameters for jFUSE simulation.

        For jFUSE, parameters are passed directly to the model, so this
        is essentially a no-op that stores the parameters for run_model.

        Args:
            params: Parameter dictionary
            settings_dir: Settings directory path
            **kwargs: Additional arguments (unused)

        Returns:
            True (always succeeds for jFUSE)
        """
        # Initialize model and data if not done
        task = kwargs.get('task')
        if not self._initialized:
            if not self._initialize_model_and_data(task):
                return False

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
        Run jFUSE model simulation.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory
            **kwargs: Must contain 'params' for the simulation

        Returns:
            True if simulation successful.
        """
        if not HAS_JFUSE or self._model is None:
            self.logger.error("jFUSE model not initialized")
            return False

        try:
            # Get parameters
            param_dict = kwargs.get('params', getattr(self, '_current_params', None))
            if param_dict is None:
                self.logger.error("No parameters provided for simulation")
                return False

            # Convert dict to Parameters/CoupledParams object
            params_obj = self._dict_to_params(param_dict)

            # Run simulation based on mode
            if self._is_distributed:
                # CoupledModel.simulate returns (outlet_Q, runoff)
                # outlet_Q is already in m³/s, runoff is mm/day per HRU
                outlet_q, runoff = self._coupled_model.simulate(self._forcing_tuple, params_obj)
                outlet_q_arr = np.array(outlet_q) if HAS_JAX else outlet_q
                runoff_arr = np.array(runoff) if HAS_JAX else runoff
                self._last_outlet_q = outlet_q_arr[self.warmup_days:]
                self._last_runoff = runoff_arr[self.warmup_days:]
            else:
                # FUSEModel.simulate returns (runoff, state)
                runoff, final_state = self._model.simulate(self._forcing_tuple, params_obj)
                runoff_arr = np.array(runoff) if HAS_JAX else runoff
                self._last_runoff = runoff_arr[self.warmup_days:]
                self._last_outlet_q = None

            self._last_params = param_dict

            # Save output files for final evaluation (required for calibration target)
            if output_dir is not None:
                # Get time index after warmup removal
                time_index = self._time_index
                if time_index is not None and len(time_index) > self.warmup_days:
                    time_index = time_index[self.warmup_days:]
                self._save_output_files(output_dir, time_index)

            return True

        except Exception as e:
            self.logger.error(f"jFUSE simulation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _save_output_files(self, output_dir: Path, time_index: pd.DatetimeIndex) -> None:
        """
        Save simulation results to output files for final evaluation.

        This is critical for the calibration target to find and evaluate
        jFUSE outputs during final evaluation (otherwise it may find
        and incorrectly use SUMMA or other model outputs).

        Args:
            output_dir: Directory to save output files
            time_index: Time index for the results (after warmup removal)
        """
        if self._last_runoff is None:
            self.logger.warning("No runoff data to save")
            return

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get domain name from config
            domain_name = self.config.get('DOMAIN_NAME', 'domain')

            # Get catchment area for unit conversion (km² -> m²)
            catchment_area_km2 = self._get_catchment_area()
            catchment_area_m2 = catchment_area_km2 * 1e6

            # Get runoff (mm/day) - handle both lumped and distributed modes
            if self._is_distributed and self._last_outlet_q is not None:
                # For distributed mode, outlet_q is already in m³/s
                streamflow_cms = self._last_outlet_q
                # Convert m³/s back to mm/day for storage
                runoff = streamflow_cms * UnitConversion.MM_DAY_TO_CMS / catchment_area_km2
            else:
                runoff = self._last_runoff
                # Handle 2D array from lumped mode
                if runoff.ndim > 1:
                    runoff = runoff[:, 0] if runoff.shape[1] > 0 else runoff.flatten()
                # Convert mm/day to m³/s: runoff_mm * area_m2 / (1000 mm/m * 86400 s/day)
                streamflow_cms = runoff * catchment_area_m2 / (1000.0 * 86400.0)

            # Ensure time_index length matches runoff
            if time_index is not None and len(time_index) != len(runoff):
                self.logger.warning(
                    f"Time index length ({len(time_index)}) != runoff length ({len(runoff)}). "
                    "Creating synthetic time index."
                )
                time_index = pd.date_range(start='2000-01-01', periods=len(runoff), freq='D')

            if time_index is None:
                time_index = pd.date_range(start='2000-01-01', periods=len(runoff), freq='D')

            # Save CSV
            results_df = pd.DataFrame({
                'datetime': time_index,
                'streamflow_mm_day': runoff,
                'streamflow_cms': streamflow_cms,
            })
            csv_file = output_dir / f"{domain_name}_jfuse_output.csv"
            results_df.to_csv(csv_file, index=False)
            self.logger.debug(f"Saved jFUSE output to: {csv_file}")

            # Save NetCDF with proper variable names for evaluator detection
            ds = xr.Dataset(
                data_vars={
                    'streamflow': (['time'], streamflow_cms),
                    'runoff': (['time'], runoff),
                },
                coords={
                    'time': time_index,
                },
                attrs={
                    'model': 'jFUSE',
                    'spatial_mode': self.spatial_mode,
                    'catchment_area_m2': catchment_area_m2,
                    'catchment_area_km2': catchment_area_km2,
                }
            )
            ds['streamflow'].attrs = {'units': 'm3/s', 'long_name': 'Streamflow'}
            ds['runoff'].attrs = {'units': 'mm/day', 'long_name': 'Runoff depth'}

            nc_file = output_dir / f"{domain_name}_jfuse_output.nc"
            ds.to_netcdf(nc_file)
            self.logger.debug(f"Saved jFUSE NetCDF to: {nc_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save jFUSE output files: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from jFUSE simulation results.

        Args:
            output_dir: Output directory (unused for jFUSE - results in memory)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary with metrics (KGE, NSE, etc.)
        """
        if self._last_runoff is None and self._last_outlet_q is None:
            self.logger.error("No simulation results available")
            return {'error': 'No simulation results'}

        if self._observations is None:
            self.logger.warning("No observations available for metrics")
            return {'error': 'No observations'}

        try:
            # For distributed mode, use outlet discharge (already in m³/s)
            # For lumped mode, use runoff (mm/day) and convert to m³/s for reporting
            area_km2 = self._get_catchment_area()

            if self._is_distributed and self._last_outlet_q is not None:
                sim = self._last_outlet_q
            else:
                sim = self._last_runoff
                # For lumped mode with 2D array, take first HRU
                if sim.ndim > 1:
                    sim = sim[:, 0] if sim.shape[1] > 0 else sim.flatten()
                # Convert mm/day to m³/s for reporting
                # Q(m³/s) = runoff(mm/day) × area(km²) / 86.4
                sim = np.array(sim) * area_km2 / UnitConversion.MM_DAY_TO_CMS

            # Get observations (stored internally in mm/day, convert back to m³/s for reporting)
            if HAS_JAX:
                obs = np.array(self._observations)
            else:
                obs = self._observations
            # Convert observations from mm/day back to m³/s for reporting
            obs = obs * area_km2 / UnitConversion.MM_DAY_TO_CMS

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
        Compute gradient of loss with respect to parameters using JAX autodiff.

        Args:
            params: Dictionary mapping parameter names to values
            metric: Objective metric ('kge' or 'nse')

        Returns:
            Dictionary mapping parameter names to gradient values,
            or None if native gradients not supported.
        """
        if not self.supports_native_gradients():
            return None

        # Initialize if needed
        if not self._initialized:
            if not self._initialize_model_and_data():
                return None

        if self._observations is None:
            self.logger.error("No observations available for gradient computation")
            return None

        try:
            # Get parameter names that we're calibrating
            param_names = list(params.keys())

            forcing_tuple = self._forcing_tuple
            obs = self._observations  # Already converted to mm/day in _load_observations
            warmup = self.warmup_days
            default_params = self._default_params
            is_distributed = self._is_distributed
            n_hrus = self.n_hrus
            coupled_model = self._coupled_model
            fuse_model = self._model

            def array_to_params(arr):
                """Convert optimization array to Parameters/CoupledParams object."""
                p = default_params
                if is_distributed:
                    # Update fuse_params within CoupledParams
                    fuse_p = p.fuse_params
                    for i, name in enumerate(param_names):
                        if hasattr(fuse_p, name):
                            # Use default arg to capture name by value
                            fuse_p = eqx.tree_at(
                                lambda x, n=name: getattr(x, n), fuse_p, jnp.ones(n_hrus) * arr[i]
                            )
                    p = eqx.tree_at(lambda x: x.fuse_params, p, fuse_p)
                else:
                    for i, name in enumerate(param_names):
                        if hasattr(p, name):
                            # Use default arg to capture name by value
                            p = eqx.tree_at(lambda x, n=name: getattr(x, n), p, arr[i])
                return p

            def loss_from_array(param_array):
                """Loss function that takes array input."""
                params_obj = array_to_params(param_array)

                if is_distributed:
                    # CoupledModel returns (outlet_q, runoff)
                    outlet_q, _ = coupled_model.simulate(forcing_tuple, params_obj)
                    sim_eval = outlet_q[warmup:]
                else:
                    # FUSEModel returns (runoff, state) - runoff in mm/day
                    # Observations already converted to mm/day, so compare directly
                    runoff, _ = fuse_model.simulate(forcing_tuple, params_obj)
                    sim_eval = runoff[warmup:]

                obs_aligned = obs[:len(sim_eval)]

                if metric.lower() == 'nse':
                    return nse_loss(sim_eval[:len(obs_aligned)], obs_aligned)
                return kge_loss(sim_eval[:len(obs_aligned)], obs_aligned)

            # Convert params to array
            param_array = jnp.array([params[name] for name in param_names])

            # Compute gradient
            grad_fn = jax.grad(loss_from_array)
            grad_array = grad_fn(param_array)

            # Convert back to dict
            grad_dict = {name: float(grad_array[i]) for i, name in enumerate(param_names)}

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

        This is more efficient than calling evaluate + compute_gradient separately
        as it uses jax.value_and_grad to share computation.

        Args:
            params: Dictionary mapping parameter names to values
            metric: Objective metric ('kge' or 'nse')

        Returns:
            Tuple of (loss_value, gradient_dict):
            - loss_value: Scalar loss (1 - metric, so minimizing loss maximizes metric)
            - gradient_dict: Dictionary mapping parameter names to gradients
        """
        if not self.supports_native_gradients():
            raise NotImplementedError(
                "Native gradients not supported. JAX or jFUSE not installed."
            )

        # Initialize if needed
        if not self._initialized:
            if not self._initialize_model_and_data():
                raise RuntimeError("Failed to initialize jFUSE worker")

        if self._observations is None:
            raise ValueError("No observations available for gradient computation")

        # Check gradient coverage once at start of optimization
        if not self._gradient_coverage_checked:
            self._gradient_coverage_checked = True
            param_names_to_check = list(params.keys())
            self.check_gradient_coverage(param_names_to_check)

        try:
            param_names = list(params.keys())

            forcing_tuple = self._forcing_tuple
            obs = self._observations  # Already converted to mm/day in _load_observations
            warmup = self.warmup_days
            default_params = self._default_params
            is_distributed = self._is_distributed
            n_hrus = self.n_hrus
            coupled_model = self._coupled_model
            fuse_model = self._model

            def array_to_params(arr):
                """Convert optimization array to Parameters/CoupledParams object."""
                p = default_params
                if is_distributed:
                    # Update fuse_params within CoupledParams
                    fuse_p = p.fuse_params
                    for i, name in enumerate(param_names):
                        if hasattr(fuse_p, name):
                            # Use default arg to capture name by value
                            fuse_p = eqx.tree_at(
                                lambda x, n=name: getattr(x, n), fuse_p, jnp.ones(n_hrus) * arr[i]
                            )
                    p = eqx.tree_at(lambda x: x.fuse_params, p, fuse_p)
                else:
                    for i, name in enumerate(param_names):
                        if hasattr(p, name):
                            # Use default arg to capture name by value
                            p = eqx.tree_at(lambda x, n=name: getattr(x, n), p, arr[i])
                return p

            def loss_from_array(param_array):
                """Loss function that takes array input."""
                params_obj = array_to_params(param_array)

                if is_distributed:
                    # CoupledModel returns (outlet_q, runoff)
                    outlet_q, _ = coupled_model.simulate(forcing_tuple, params_obj)
                    sim_eval = outlet_q[warmup:]
                else:
                    # FUSEModel returns (runoff, state) - runoff in mm/day
                    # Observations already converted to mm/day, so compare directly
                    runoff, _ = fuse_model.simulate(forcing_tuple, params_obj)
                    sim_eval = runoff[warmup:]

                obs_aligned = obs[:len(sim_eval)]

                if metric.lower() == 'nse':
                    return nse_loss(sim_eval[:len(obs_aligned)], obs_aligned)
                return kge_loss(sim_eval[:len(obs_aligned)], obs_aligned)

            # Convert params to array
            param_array = jnp.array([params[name] for name in param_names])

            # Compute value and gradient together
            val_and_grad_fn = jax.value_and_grad(loss_from_array)
            loss_val, grad_array = val_and_grad_fn(param_array)

            # Convert to Python types
            loss_float = float(loss_val)
            grad_dict = {name: float(grad_array[i]) for i, name in enumerate(param_names)}

            return loss_float, grad_dict

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
            # Convert dict to Parameters/CoupledParams object
            params_obj = self._dict_to_params(params)

            # Run simulation based on mode
            if self._is_distributed:
                # CoupledModel returns (outlet_q, runoff)
                outlet_q, runoff = self._coupled_model.simulate(self._forcing_tuple, params_obj)
                sim = np.array(outlet_q) if HAS_JAX else outlet_q
            else:
                # FUSEModel returns (runoff, state) - runoff in mm/day
                # Observations already converted to mm/day, so compare directly
                runoff, _ = self._model.simulate(self._forcing_tuple, params_obj)
                sim = np.array(runoff) if HAS_JAX else runoff

            # Skip warmup
            sim = sim[self.warmup_days:]

            # Observations are already in mm/day (converted in _load_observations)
            obs = np.array(self._observations) if (HAS_JAX and self._observations is not None) else self._observations

            if obs is None:
                return self.penalty_score

            # Align and filter
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs_arr = obs[:min_len]

            # Handle multi-dimensional sim (for lumped mode with 2D arrays)
            if sim.ndim > 1:
                sim = sim[:, 0] if sim.shape[1] > 0 else sim.flatten()

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
