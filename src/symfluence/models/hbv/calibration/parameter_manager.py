"""
HBV Parameter Manager.

Provides parameter bounds, transformations, and management for HBV-96 calibration.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_hbv_bounds
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.models.hbv.model import PARAM_BOUNDS, DEFAULT_PARAMS


@OptimizerRegistry.register_parameter_manager('HBV')
class HBVParameterManager(BaseParameterManager):
    """
    Manages HBV-96 parameters for calibration.

    Provides:
    - Parameter bounds retrieval
    - Transformation between normalized [0,1] and physical space
    - Default values
    - Parameter validation
    """

    def __init__(self, config: Dict, logger: logging.Logger, hbv_settings_dir: Path):
        """
        Initialize parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            hbv_settings_dir: Path to HBV settings directory
        """
        super().__init__(config, logger, hbv_settings_dir)

        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse HBV parameters to calibrate from config
        hbv_params_str = config.get('HBV_PARAMS_TO_CALIBRATE') or config.get('PARAMS_TO_CALIBRATE')

        # Handle None, empty string, or 'default' as signal to use ALL available parameters
        if hbv_params_str is None or hbv_params_str == '' or hbv_params_str == 'default':
            # Use all available HBV parameters (excluding smoothing which is a numerical parameter)
            self.hbv_params = [p for p in PARAM_BOUNDS.keys() if p != 'smoothing']
            logger.debug(f"Using all available HBV parameters for calibration: {self.hbv_params}")
        else:
            self.hbv_params = [p.strip() for p in str(hbv_params_str).split(',') if p.strip()]
            logger.debug(f"Using user-specified HBV parameters for calibration: {self.hbv_params}")

        # Store internal references
        self.all_bounds = PARAM_BOUNDS.copy()
        self.defaults = DEFAULT_PARAMS.copy()
        self.calibration_params = self.hbv_params

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return HBV parameter names from config."""
        return self.hbv_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return HBV parameter bounds from central registry."""
        return get_hbv_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        HBV doesn't have a parameter file to update.
        Parameters are passed directly to the model during simulation.
        We return True as 'applying' parameters happens in the worker/runner.
        """
        return True

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from config or defaults."""
        # Check for explicit initial params in config
        initial_params = self.config_dict.get('HBV_INITIAL_PARAMS', 'default')

        if initial_params == 'default':
            # Use HBV defaults from model module
            self.logger.debug("Using standard HBV defaults for initial parameters")
            return {p: self.defaults[p] for p in self.hbv_params}

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
                self.logger.warning(f"Could not parse HBV_INITIAL_PARAMS: {e}")
                return {p: self.defaults[p] for p in self.hbv_params}

        return {p: self.defaults[p] for p in self.hbv_params}

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
            raise KeyError(f"Unknown HBV parameter: {param_name}")
        return self.all_bounds[param_name]

    def get_calibration_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Get bounds for all calibration parameters.

        Returns:
            Dict mapping param_name -> {'min': float, 'max': float}
        """
        return {
            name: {'min': self.all_bounds[name][0], 'max': self.all_bounds[name][1]}
            for name in self.calibration_params
        }

    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds as arrays for optimization algorithms.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        lower = np.array([self.all_bounds[p][0] for p in self.calibration_params])
        upper = np.array([self.all_bounds[p][1] for p in self.calibration_params])
        return lower, upper

    def get_default(self, param_name: str) -> float:
        """Get default value for a parameter."""
        return self.defaults.get(param_name, 0.0)

    def get_default_vector(self) -> np.ndarray:
        """Get default values as array for calibration parameters."""
        return np.array([self.defaults[p] for p in self.calibration_params])

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
            value = params.get(name, self.defaults[name])
            low, high = self.all_bounds[name]
            norm_val = (value - low) / (high - low + 1e-10)
            normalized.append(np.clip(norm_val, 0, 1))
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
            low, high = self.all_bounds[name]
            params[name] = low + values[i] * (high - low)
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
        return np.array([params.get(p, self.defaults[p]) for p in self.calibration_params])

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


def get_hbv_calibration_bounds(
    params_to_calibrate: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to get HBV calibration bounds.

    Args:
        params_to_calibrate: List of parameters to include.
                           If None, uses ALL available parameters.

    Returns:
        Dict mapping param_name -> {'min': float, 'max': float}
    """
    if params_to_calibrate is None:
        # Use ALL available parameters (excluding smoothing which is numerical)
        params_to_calibrate = [p for p in PARAM_BOUNDS.keys() if p != 'smoothing']

    # Return bounds directly from PARAM_BOUNDS
    return {
        name: {'min': PARAM_BOUNDS[name][0], 'max': PARAM_BOUNDS[name][1]}
        for name in params_to_calibrate if name in PARAM_BOUNDS
    }


# Parameter descriptions for documentation/UI
PARAM_DESCRIPTIONS = {
    'tt': {
        'name': 'Threshold Temperature',
        'description': 'Temperature threshold for rain/snow partitioning',
        'unit': '°C',
        'category': 'snow'
    },
    'cfmax': {
        'name': 'Degree-Day Factor',
        'description': 'Snowmelt rate per degree above threshold',
        'unit': 'mm/°C/day',
        'category': 'snow'
    },
    'sfcf': {
        'name': 'Snowfall Correction Factor',
        'description': 'Multiplier for snowfall gauge undercatch',
        'unit': '-',
        'category': 'snow'
    },
    'cfr': {
        'name': 'Refreezing Coefficient',
        'description': 'Rate of liquid water refreezing in snowpack',
        'unit': '-',
        'category': 'snow'
    },
    'cwh': {
        'name': 'Water Holding Capacity',
        'description': 'Fraction of snow that can hold liquid water',
        'unit': '-',
        'category': 'snow'
    },
    'fc': {
        'name': 'Field Capacity',
        'description': 'Maximum soil moisture storage',
        'unit': 'mm',
        'category': 'soil'
    },
    'lp': {
        'name': 'LP Threshold',
        'description': 'Soil moisture threshold for ET reduction',
        'unit': 'fraction of FC',
        'category': 'soil'
    },
    'beta': {
        'name': 'Soil Shape',
        'description': 'Non-linearity of soil moisture recharge',
        'unit': '-',
        'category': 'soil'
    },
    'k0': {
        'name': 'Surface Flow Recession',
        'description': 'Recession coefficient for fast surface runoff',
        'unit': '1/day',
        'category': 'response'
    },
    'k1': {
        'name': 'Interflow Recession',
        'description': 'Recession coefficient for intermediate flow',
        'unit': '1/day',
        'category': 'response'
    },
    'k2': {
        'name': 'Baseflow Recession',
        'description': 'Recession coefficient for slow groundwater flow',
        'unit': '1/day',
        'category': 'response'
    },
    'uzl': {
        'name': 'Upper Zone Threshold',
        'description': 'Storage threshold for surface runoff generation',
        'unit': 'mm',
        'category': 'response'
    },
    'perc': {
        'name': 'Percolation Rate',
        'description': 'Maximum percolation from upper to lower zone',
        'unit': 'mm/day',
        'category': 'response'
    },
    'maxbas': {
        'name': 'Routing Length',
        'description': 'Base of triangular routing function',
        'unit': 'days',
        'category': 'routing'
    },
    'smoothing': {
        'name': 'Smoothing Factor',
        'description': 'Sharpness of threshold approximations for differentiability',
        'unit': '-',
        'category': 'numerical'
    },
}
