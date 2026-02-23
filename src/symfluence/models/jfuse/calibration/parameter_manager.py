"""
jFUSE Parameter Manager.

Provides parameter bounds, transformations, and management for jFUSE calibration.
Uses jFUSE's native PARAM_BOUNDS when available, with fallback defaults.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry

# Try to import jFUSE parameter bounds
try:
    from jfuse import PARAM_BOUNDS as JFUSE_PARAM_BOUNDS
    HAS_JFUSE = True
except ImportError:
    HAS_JFUSE = False
    JFUSE_PARAM_BOUNDS = {}


# Fallback parameter bounds if jFUSE not installed
# Based on typical ranges from literature and jFUSE documentation
FALLBACK_PARAM_BOUNDS = {
    # Storage parameters
    'S1_max': (10.0, 500.0),      # Upper zone storage capacity (mm)
    'S2_max': (50.0, 2000.0),     # Lower zone storage capacity (mm)

    # Drainage/flux parameters
    'ku': (0.01, 0.99),           # Upper layer drainage coefficient
    'ki': (0.001, 0.5),           # Interflow coefficient
    'ks': (0.0001, 0.1),          # Baseflow coefficient
    'kp': (0.001, 0.99),          # Percolation coefficient

    # TOPMODEL parameters
    'n': (0.1, 5.0),              # TOPMODEL exponential parameter
    'v': (0.001, 0.99),           # Linear reservoir parameter

    # Surface/saturation area
    'Ac_max': (0.01, 0.99),       # Maximum saturated area fraction

    # Snow parameters
    'T_melt': (-5.0, 5.0),        # Threshold temperature for snowmelt (°C)
    'melt_rate': (1.0, 10.0),     # Degree-day factor (mm/°C/day)
    'T_snow': (-3.0, 3.0),        # Snow/rain threshold temperature (°C)

    # Evapotranspiration
    'c_pet': (0.5, 1.5),          # PET correction factor
    'soil_pet': (0.1, 1.0),       # Soil moisture threshold for ET

    # Surface runoff
    'sat_threshold': (0.0, 1.0),  # Saturation threshold for surface runoff

    # Routing (if used)
    'K': (0.5, 10.0),             # Muskingum K (travel time)
    'X': (0.0, 0.5),              # Muskingum X (weighting factor)
}

# Default parameter values (mid-range of bounds)
FALLBACK_DEFAULT_PARAMS = {
    'S1_max': 100.0,
    'S2_max': 500.0,
    'ku': 0.5,
    'ki': 0.1,
    'ks': 0.01,
    'kp': 0.1,
    'n': 1.0,
    'v': 0.5,
    'Ac_max': 0.5,
    'T_melt': 0.0,
    'melt_rate': 3.0,
    'T_snow': 0.0,
    'c_pet': 1.0,
    'soil_pet': 0.5,
    'sat_threshold': 0.5,
    'K': 2.0,
    'X': 0.2,
}

# Get actual bounds (jFUSE if available, else fallback)
PARAM_BOUNDS = JFUSE_PARAM_BOUNDS if HAS_JFUSE else FALLBACK_PARAM_BOUNDS

# Get default parameters from jFUSE Parameters class
if HAS_JFUSE:
    try:
        from jfuse import Parameters
        _default_params_obj = Parameters.default(n_hrus=1)
        # Extract scalar values from Parameters object
        DEFAULT_PARAMS = {}
        for name in FALLBACK_DEFAULT_PARAMS.keys():
            if hasattr(_default_params_obj, name):
                DEFAULT_PARAMS[name] = float(getattr(_default_params_obj, name))
            else:
                DEFAULT_PARAMS[name] = FALLBACK_DEFAULT_PARAMS[name]
    except (ImportError, AttributeError):
        DEFAULT_PARAMS = FALLBACK_DEFAULT_PARAMS
else:
    DEFAULT_PARAMS = FALLBACK_DEFAULT_PARAMS


@OptimizerRegistry.register_parameter_manager('JFUSE')
class JFUSEParameterManager(BaseParameterManager):
    """
    Manages jFUSE parameters for calibration.

    Provides:
    - Parameter bounds retrieval (from jFUSE or fallback)
    - Transformation between normalized [0,1] and physical space
    - Default values
    - Parameter validation

    When jFUSE is installed, uses the native PARAM_BOUNDS from the package.
    Otherwise falls back to reasonable defaults from literature.

    When JFUSE_USE_TRANSFER_FUNCTIONS is enabled, switches to coefficient
    mode: calibration parameters become transfer function coefficients
    (e.g. S1_max_a, S1_max_b) instead of raw FUSE parameters.
    """

    def __init__(
        self,
        config: Dict,
        logger: logging.Logger,
        jfuse_settings_dir: Path
    ):
        """
        Initialize parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            jfuse_settings_dir: Path to jFUSE settings directory
        """
        super().__init__(config, logger, jfuse_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Check for transfer function mode
        self._use_transfer_functions = bool(self._get_jfuse_cfg('USE_TRANSFER_FUNCTIONS', default=False))
        self._tf_config = None

        # Parse jFUSE parameters to calibrate from config
        jfuse_params_str = self._get_jfuse_cfg('PARAMS_TO_CALIBRATE')
        # Handle None, empty string, or 'default' as signal to use default parameter list
        if jfuse_params_str is None or jfuse_params_str == '' or jfuse_params_str == 'default':
            # Default 14 parameters with non-zero gradients for prms_gradient config
            jfuse_params_str = 'S1_max,S2_max,ku,ki,ks,n,Ac_max,b,f_rchr,T_rain,T_melt,MFMAX,MFMIN,smooth_frac'

        self.jfuse_params = [p.strip() for p in str(jfuse_params_str).split(',') if p.strip()]

        # Validate parameters against available bounds
        if HAS_JFUSE:
            self._validate_params()

        # Store internal references
        self.all_bounds = PARAM_BOUNDS.copy()
        self.defaults = DEFAULT_PARAMS.copy()
        self.calibration_params = self.jfuse_params

        # Apply custom bounds from config if provided (tuple storage for TF mode)
        custom_bounds = self._get_jfuse_cfg('PARAM_BOUNDS', default={})
        if custom_bounds:
            for param_name, bnd in custom_bounds.items():
                if isinstance(bnd, (list, tuple)) and len(bnd) == 2:
                    self.all_bounds[param_name] = (float(bnd[0]), float(bnd[1]))
                elif isinstance(bnd, dict) and 'min' in bnd and 'max' in bnd:
                    self.all_bounds[param_name] = (float(bnd['min']), float(bnd['max']))

        # Initialize transfer function mode if enabled
        if self._use_transfer_functions:
            self._init_transfer_function_mode(config)

    def _get_jfuse_cfg(self, bare_key: str, default=None):
        """Get JFUSE config value, handling model_extra {type: value} dicts.

        SymfluenceConfig.from_file() strips the JFUSE_ prefix via ModelRegistry
        and stores bare keys in model_extra as {type: value} dicts.
        This tries all access paths.
        """
        flat_key = f'JFUSE_{bare_key.upper()}'

        # Try _get_config_value first (handles dict configs and overrides)
        val = self._get_config_value(lambda: None, default=None, dict_key=flat_key)
        if val is not None:
            return val

        # Try model_extra paths (SymfluenceConfig from_file)
        cfg = self.config
        if hasattr(cfg, 'model_extra') and cfg.model_extra:
            # Try _extra dict
            extra = cfg.model_extra.get('_extra', {})
            if isinstance(extra, dict):
                val = extra.get(flat_key)
                if val is not None:
                    return val

            # Try bare key (ModelRegistry transformer path)
            val = cfg.model_extra.get(bare_key.lower())
            if val is not None:
                # Unwrap {type: value} dict from ModelRegistry
                if isinstance(val, dict) and len(val) == 1:
                    return next(iter(val.values()))
                return val

        return default

    def _validate_params(self) -> None:
        """Validate that calibration parameters exist in bounds."""
        invalid = [p for p in self.jfuse_params if p not in PARAM_BOUNDS]
        if invalid:
            self.logger.warning(
                f"Unknown jFUSE parameters: {invalid}. "
                f"Available parameters: {list(PARAM_BOUNDS.keys())}"
            )

    def _init_transfer_function_mode(self, config: Dict) -> None:
        """Initialize transfer function mode from config."""
        from symfluence.models.jfuse.calibration.transfer_functions import (
            JaxTransferFunctionConfig,
        )

        attributes_path = self._get_config_value(lambda: None, default=None, dict_key='JFUSE_ATTRIBUTES_PATH')
        if not attributes_path:
            self.logger.error(
                "JFUSE_USE_TRANSFER_FUNCTIONS=true but JFUSE_ATTRIBUTES_PATH not set"
            )
            self._use_transfer_functions = False
            return

        b_bounds_cfg = self._get_config_value(lambda: None, default=(-5.0, 5.0), dict_key='JFUSE_TF_B_BOUNDS')
        if isinstance(b_bounds_cfg, list):
            b_bounds_cfg = tuple(b_bounds_cfg)

        self._tf_config = JaxTransferFunctionConfig(
            attributes_path=attributes_path,
            calibrated_params=self.jfuse_params,
            b_bounds=b_bounds_cfg,
            logger=self.logger,
        )

        # Override calibration params and bounds with coefficient versions
        self.calibration_params = self._tf_config.coefficient_names
        for name, bounds in zip(
            self._tf_config.coefficient_names,
            self._tf_config.coefficient_bounds,
        ):
            self.all_bounds[name] = bounds

        # Set defaults for coefficients: a = actual parameter default, b = 0
        # Using actual defaults (not midpoint of bounds) gives a much better
        # starting point — midpoint gives e.g. S1_max=2502 instead of 200.
        for i, pname in enumerate(self.jfuse_params):
            lo, hi = PARAM_BOUNDS[pname]
            default_val = self.defaults.get(pname, (lo + hi) / 2.0)
            self.defaults[f'{pname}_a'] = default_val
            self.defaults[f'{pname}_b'] = 0.0

        self.logger.info(
            f"Transfer function mode enabled: {self._tf_config.n_coefficients} "
            f"coefficients for {self._tf_config.n_calibrated_params} params, "
            f"{self._tf_config.n_grus} GRUs"
        )

    @property
    def use_transfer_functions(self) -> bool:
        """Whether transfer function mode is active."""
        return self._use_transfer_functions

    @property
    def transfer_function_config(self) -> Optional[object]:
        """The JaxTransferFunctionConfig if TF mode is active, else None."""
        return self._tf_config

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return parameter/coefficient names for calibration."""
        if self._use_transfer_functions and self._tf_config is not None:
            return self._tf_config.coefficient_names
        return self.jfuse_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return parameter/coefficient bounds for calibration.

        Config overrides (JFUSE_PARAM_BOUNDS) are applied via
        _apply_config_bounds_override to preserve any transform metadata.
        """
        if self._use_transfer_functions and self._tf_config is not None:
            return {
                name: {'min': bounds[0], 'max': bounds[1]}
                for name, bounds in zip(
                    self._tf_config.coefficient_names,
                    self._tf_config.coefficient_bounds,
                )
            }
        bounds = {
            name: {'min': self.all_bounds[name][0], 'max': self.all_bounds[name][1]}
            for name in self.jfuse_params
            if name in self.all_bounds
        }
        # Apply config overrides with transform preservation
        config_bounds = self._get_config_value(lambda: None, default={}, dict_key='JFUSE_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)
        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        jFUSE doesn't have a parameter file to update.
        Parameters are passed directly to the model during simulation.
        """
        return True

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from config or defaults."""
        initial_params = self._get_config_value(lambda: None, default='default', dict_key='JFUSE_INITIAL_PARAMS')

        if initial_params == 'default':
            self.logger.debug("Using standard jFUSE defaults for initial parameters")
            # In TF mode, return coefficient defaults (S1_max_a, S1_max_b, ...)
            if self._use_transfer_functions and self._tf_config is not None:
                return {p: self.defaults.get(p, 0.0) for p in self.calibration_params}
            return {p: self.defaults.get(p, 0.0) for p in self.jfuse_params}

        # Parse string-based initial params if provided
        if isinstance(initial_params, str) and initial_params != 'default':
            try:
                param_dict = {}
                for pair in initial_params.split(','):
                    if '=' in pair:
                        k, v = pair.split('=')
                        param_dict[k.strip()] = float(v.strip())
                return param_dict
            except Exception as e:
                self.logger.warning(f"Could not parse JFUSE_INITIAL_PARAMS: {e}")
                return {p: self.defaults.get(p, 0.0) for p in self.jfuse_params}

        return {p: self.defaults.get(p, 0.0) for p in self.jfuse_params}

    def get_bounds(self, param_name: str) -> Tuple[float, float]:
        """
        Get bounds for a single parameter.

        Args:
            param_name: Parameter name

        Returns:
            Tuple of (min, max)

        Raises:
            KeyError: If parameter not found
        """
        if param_name not in self.all_bounds:
            raise KeyError(f"Unknown jFUSE parameter: {param_name}")
        return self.all_bounds[param_name]

    def get_calibration_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Get bounds for all calibration parameters.

        Returns:
            Dict mapping param_name -> {'min': float, 'max': float}
        """
        result = {}
        for name in self.calibration_params:
            if name in self.all_bounds:
                bounds = self.all_bounds[name]
                result[name] = {'min': bounds[0], 'max': bounds[1]}
        return result

    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds as arrays for optimization algorithms.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        lower = []
        upper = []
        for p in self.calibration_params:
            if p in self.all_bounds:
                lower.append(self.all_bounds[p][0])
                upper.append(self.all_bounds[p][1])
        return np.array(lower), np.array(upper)

    def get_default(self, param_name: str) -> float:
        """Get default value for a parameter."""
        return self.defaults.get(param_name, 0.0)

    def get_default_vector(self) -> np.ndarray:
        """Get default values as array for calibration parameters."""
        return np.array([self.defaults.get(p, 0.0) for p in self.calibration_params])

    def normalize(self, params: Dict[str, float]) -> np.ndarray:
        """
        Normalize parameters to [0, 1] range.

        Args:
            params: Dictionary of parameter values

        Returns:
            Array of normalized values
        """
        normalized = []
        for name in self.calibration_params:
            value = params.get(name, self.defaults.get(name, 0.0))
            if name in self.all_bounds:
                low, high = self.all_bounds[name]
                norm_val = (value - low) / (high - low + 1e-10)
                normalized.append(np.clip(norm_val, 0, 1))
            else:
                normalized.append(0.5)  # Default to middle if no bounds
        return np.array(normalized)

    def denormalize(self, values: np.ndarray) -> Dict[str, float]:
        """
        Convert normalized [0, 1] values to physical parameter values.

        Args:
            values: Array of normalized values

        Returns:
            Dictionary of parameter values
        """
        params = {}
        for i, name in enumerate(self.calibration_params):
            if name in self.all_bounds:
                low, high = self.all_bounds[name]
                params[name] = low + values[i] * (high - low)
            else:
                params[name] = self.defaults.get(name, 0.0)
        return params

    def array_to_dict(self, values: np.ndarray) -> Dict[str, float]:
        """
        Convert parameter array to dictionary.

        Args:
            values: Array of parameter values (physical space)

        Returns:
            Dictionary mapping param names to values
        """
        return dict(zip(self.calibration_params, values))

    def dict_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """
        Convert parameter dictionary to array.

        Args:
            params: Dictionary of parameter values

        Returns:
            Array of values in calibration parameter order
        """
        return np.array([
            params.get(p, self.defaults.get(p, 0.0))
            for p in self.calibration_params
        ])

    def validate(self, params: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate parameter values are within bounds.

        Args:
            params: Dictionary of parameter values

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        for name, value in params.items():
            if name in self.all_bounds:
                low, high = self.all_bounds[name]
                if value < low:
                    violations.append(f"{name}={value} < min={low}")
                elif value > high:
                    violations.append(f"{name}={value} > max={high}")

        return len(violations) == 0, violations

    def clip_to_bounds(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Clip parameter values to their bounds.

        Args:
            params: Dictionary of parameter values

        Returns:
            Dictionary with clipped values
        """
        clipped = {}
        for name, value in params.items():
            if name in self.all_bounds:
                low, high = self.all_bounds[name]
                clipped[name] = np.clip(value, low, high)
            else:
                clipped[name] = value
        return clipped

    def get_complete_params(self, partial_params: Dict[str, float]) -> Dict[str, float]:
        """
        Complete partial parameter dict with defaults.

        Args:
            partial_params: Dictionary with some parameters

        Returns:
            Complete dictionary with all parameters
        """
        complete = self.defaults.copy()
        complete.update(partial_params)
        return complete


def get_jfuse_calibration_bounds(
    params_to_calibrate: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to get jFUSE calibration bounds.

    Args:
        params_to_calibrate: List of parameters to include.
                           If None, uses common calibration set.

    Returns:
        Dict mapping param_name -> {'min': float, 'max': float}
    """
    if params_to_calibrate is None:
        params_to_calibrate = [
            'S1_max', 'S2_max', 'ku', 'ki', 'ks',
            'n', 'v', 'Ac_max', 'T_melt', 'melt_rate'
        ]

    return {
        name: {'min': PARAM_BOUNDS[name][0], 'max': PARAM_BOUNDS[name][1]}
        for name in params_to_calibrate if name in PARAM_BOUNDS
    }


# Parameter descriptions for documentation/UI
PARAM_DESCRIPTIONS = {
    'S1_max': {
        'name': 'Upper Storage Capacity',
        'description': 'Maximum water storage in upper zone',
        'unit': 'mm',
        'category': 'storage'
    },
    'S2_max': {
        'name': 'Lower Storage Capacity',
        'description': 'Maximum water storage in lower zone',
        'unit': 'mm',
        'category': 'storage'
    },
    'ku': {
        'name': 'Upper Drainage',
        'description': 'Drainage coefficient from upper zone',
        'unit': '1/day',
        'category': 'flux'
    },
    'ki': {
        'name': 'Interflow Rate',
        'description': 'Interflow coefficient',
        'unit': '1/day',
        'category': 'flux'
    },
    'ks': {
        'name': 'Baseflow Rate',
        'description': 'Baseflow recession coefficient',
        'unit': '1/day',
        'category': 'flux'
    },
    'n': {
        'name': 'TOPMODEL Decay',
        'description': 'Exponential decay parameter for TOPMODEL',
        'unit': '-',
        'category': 'topmodel'
    },
    'v': {
        'name': 'Linear Rate',
        'description': 'Linear baseflow rate parameter',
        'unit': '-',
        'category': 'flux'
    },
    'Ac_max': {
        'name': 'Max Saturated Area',
        'description': 'Maximum fraction of saturated contributing area',
        'unit': '-',
        'category': 'surface'
    },
    'T_melt': {
        'name': 'Melt Threshold',
        'description': 'Temperature threshold for snowmelt',
        'unit': '°C',
        'category': 'snow'
    },
    'melt_rate': {
        'name': 'Degree-Day Factor',
        'description': 'Snowmelt rate per degree above threshold',
        'unit': 'mm/°C/day',
        'category': 'snow'
    },
    'T_snow': {
        'name': 'Snow Threshold',
        'description': 'Temperature threshold for snow/rain partitioning',
        'unit': '°C',
        'category': 'snow'
    },
}
