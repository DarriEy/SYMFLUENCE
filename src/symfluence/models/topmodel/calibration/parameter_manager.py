"""
TOPMODEL Parameter Manager.

Provides parameter bounds, transformations, and management for TOPMODEL calibration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from symfluence.models.topmodel.parameters import DEFAULT_PARAMS, PARAM_BOUNDS
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('TOPMODEL')
class TopmodelParameterManager(BaseParameterManager):
    """
    Manages TOPMODEL parameters for calibration.

    Provides:
    - Parameter bounds retrieval
    - Transformation between normalized [0,1] and physical space
    - Default values
    - Parameter validation
    """

    def __init__(self, config: Dict, logger: logging.Logger, topmodel_settings_dir: Path):
        """
        Initialize parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            topmodel_settings_dir: Path to TOPMODEL settings directory
        """
        super().__init__(config, logger, topmodel_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Parse TOPMODEL parameters to calibrate from config
        topmodel_params_str = self._get_config_value(lambda: self.config.model.topmodel.params_to_calibrate, default=None, dict_key='TOPMODEL_PARAMS_TO_CALIBRATE') or self._get_config_value(lambda: None, default=None, dict_key='PARAMS_TO_CALIBRATE')

        # Handle None, empty string, or 'default' as signal to use ALL available parameters
        if topmodel_params_str is None or topmodel_params_str == '' or topmodel_params_str == 'default':
            self.topmodel_params = list(PARAM_BOUNDS.keys())
            logger.debug(f"Using all available TOPMODEL parameters for calibration: {self.topmodel_params}")
        else:
            self.topmodel_params = [p.strip() for p in str(topmodel_params_str).split(',') if p.strip()]
            logger.debug(f"Using user-specified TOPMODEL parameters: {self.topmodel_params}")

        # Store internal references
        self.all_bounds = PARAM_BOUNDS.copy()
        self.defaults = DEFAULT_PARAMS.copy()
        self.calibration_params = self.topmodel_params

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return TOPMODEL parameter names from config."""
        return self.topmodel_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return TOPMODEL parameter bounds from central registry."""
        from symfluence.optimization.core.parameter_bounds_registry import get_topmodel_bounds
        return get_topmodel_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """TOPMODEL runs in-memory; parameters passed directly to model."""
        return True

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from config or defaults."""
        initial_params = self._get_config_value(lambda: None, default='default', dict_key='TOPMODEL_INITIAL_PARAMS')

        if initial_params == 'default':
            self.logger.debug("Using standard TOPMODEL defaults for initial parameters")
            return {p: self.defaults[p] for p in self.topmodel_params}

        if isinstance(initial_params, str) and initial_params != 'default':
            try:
                param_dict = {}
                for pair in initial_params.split(','):
                    if '=' in pair:
                        k, v = pair.split('=')
                        param_dict[k.strip()] = float(v.strip())
                return param_dict
            except Exception as e:
                self.logger.warning(f"Could not parse TOPMODEL_INITIAL_PARAMS: {e}")
                return {p: self.defaults[p] for p in self.topmodel_params}

        return {p: self.defaults[p] for p in self.topmodel_params}

    def get_bounds(self, param_name: str) -> Tuple[float, float]:
        """Get bounds for a single parameter."""
        if param_name not in self.all_bounds:
            raise KeyError(f"Unknown TOPMODEL parameter: {param_name}")
        return self.all_bounds[param_name]

    def get_calibration_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get bounds for all calibration parameters."""
        return {
            name: {'min': self.all_bounds[name][0], 'max': self.all_bounds[name][1]}
            for name in self.calibration_params
        }

    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds as arrays for optimization algorithms."""
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
        """Normalize parameters to [0, 1] range."""
        normalized = []
        for name in self.calibration_params:
            value = params.get(name, self.defaults[name])
            low, high = self.all_bounds[name]
            norm_val = (value - low) / (high - low + 1e-10)
            normalized.append(np.clip(norm_val, 0, 1))
        return np.array(normalized)

    def denormalize(self, values: np.ndarray) -> Dict[str, float]:
        """Convert normalized [0, 1] values to physical parameter values."""
        params = {}
        for i, name in enumerate(self.calibration_params):
            low, high = self.all_bounds[name]
            params[name] = low + values[i] * (high - low)
        return params

    def array_to_dict(self, values: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return dict(zip(self.calibration_params, values))

    def dict_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array."""
        return np.array([params.get(p, self.defaults[p]) for p in self.calibration_params])

    def validate(self, params: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate parameter values are within bounds."""
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
        """Clip parameter values to their bounds."""
        clipped = {}
        for name, value in params.items():
            if name in self.all_bounds:
                low, high = self.all_bounds[name]
                clipped[name] = np.clip(value, low, high)
            else:
                clipped[name] = value
        return clipped

    def get_complete_params(self, partial_params: Dict[str, float]) -> Dict[str, float]:
        """Complete partial parameter dict with defaults."""
        complete = self.defaults.copy()
        complete.update(partial_params)
        return complete


def get_topmodel_calibration_bounds(
    params_to_calibrate: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to get TOPMODEL calibration bounds.

    Args:
        params_to_calibrate: List of parameters. If None, uses all.

    Returns:
        Dict mapping param_name -> {'min': float, 'max': float}
    """
    if params_to_calibrate is None:
        params_to_calibrate = list(PARAM_BOUNDS.keys())

    return {
        name: {'min': PARAM_BOUNDS[name][0], 'max': PARAM_BOUNDS[name][1]}
        for name in params_to_calibrate if name in PARAM_BOUNDS
    }
