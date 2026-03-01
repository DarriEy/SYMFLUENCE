# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
cFUSE Parameter Manager.

Provides parameter bounds, transformations, and management for cFUSE calibration.
Uses cFUSE's native PARAM_BOUNDS when available, with fallback defaults.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry

# Try to import cFUSE parameter bounds
try:
    from cfuse import DEFAULT_PARAMS as CFUSE_DEFAULT_PARAMS
    from cfuse import PARAM_BOUNDS as CFUSE_PARAM_BOUNDS
    from cfuse import PARAM_NAMES as CFUSE_PARAM_NAMES
    HAS_CFUSE = True
except ImportError:
    HAS_CFUSE = False
    CFUSE_PARAM_BOUNDS = {}
    CFUSE_DEFAULT_PARAMS = {}
    CFUSE_PARAM_NAMES = []


# Fallback parameter bounds if cFUSE not installed
# Based on typical ranges from FUSE literature
FALLBACK_PARAM_BOUNDS = {
    # Storage parameters
    'S1_max': (50.0, 5000.0),      # Upper zone storage capacity (mm)
    'S2_max': (100.0, 10000.0),    # Lower zone storage capacity (mm)

    # Fraction parameters
    'f_tens': (0.05, 0.95),        # Fraction tension storage
    'f_rchr': (0.05, 0.95),        # Fraction recharge
    'f_base': (0.05, 0.95),        # Fraction baseflow
    'r1': (0.05, 0.95),            # Fraction parameter

    # Drainage/flux parameters
    'ku': (0.01, 1000.0),          # Upper layer drainage coefficient
    'c': (1.0, 20.0),              # Percolation curve exponent
    'alpha': (1.0, 250.0),         # Percolation scaling
    'psi': (1.0, 5.0),             # Percolation exponent
    'kappa': (0.05, 0.95),         # Percolation fraction
    'ki': (0.01, 1000.0),          # Interflow coefficient
    'ks': (0.001, 10000.0),        # Baseflow coefficient

    # TOPMODEL parameters
    'n': (1.0, 10.0),              # TOPMODEL exponential parameter
    'v': (0.001, 0.25),            # Linear reservoir parameter
    'v_A': (0.001, 0.25),          # Linear reservoir A
    'v_B': (0.001, 0.25),          # Linear reservoir B

    # Surface/saturation area
    'Ac_max': (0.05, 0.95),        # Maximum saturated area fraction
    'b': (0.001, 3.0),             # VIC b parameter
    'lambda': (5.0, 10.0),         # TOPMODEL lambda
    'chi': (2.0, 5.0),             # TOPMODEL chi
    'mu_t': (0.01, 5.0),           # TOPMODEL mu

    # Snow parameters
    'T_rain': (-2.0, 4.0),         # Rain threshold temperature (°C)
    'T_melt': (-2.0, 4.0),         # Melt threshold temperature (°C)
    'melt_rate': (1.0, 10.0),      # Degree-day factor (mm/°C/day)
    'lapse_rate': (-9.8, 0.0),     # Temperature lapse rate (°C/km)
    'opg': (0.0, 1.0),             # Orographic precipitation gradient
    'MFMAX': (1.0, 10.0),          # Max melt factor
    'MFMIN': (0.0, 10.0),          # Min melt factor

    # Routing
    'shape_t': (1.0, 10.0),        # Gamma shape for routing UH
    'smooth_frac': (0.001, 0.1),   # Smoothing fraction for thresholds
}

# Default parameter values (mid-range of bounds)
FALLBACK_DEFAULT_PARAMS = {
    'S1_max': 200.0,
    'S2_max': 2000.0,
    'f_tens': 0.5,
    'f_rchr': 0.5,
    'f_base': 0.5,
    'r1': 0.5,
    'ku': 10.0,
    'c': 4.0,
    'alpha': 100.0,
    'psi': 2.0,
    'kappa': 0.5,
    'ki': 10.0,
    'ks': 100.0,
    'n': 2.0,
    'v': 0.1,
    'v_A': 0.1,
    'v_B': 0.1,
    'Ac_max': 0.5,
    'b': 1.0,
    'lambda': 7.5,
    'chi': 3.5,
    'mu_t': 1.0,
    'T_rain': 1.0,
    'T_melt': 0.0,
    'melt_rate': 3.0,
    'lapse_rate': -6.5,
    'opg': 0.0,
    'MFMAX': 4.0,
    'MFMIN': 1.0,
    'shape_t': 3.0,
    'smooth_frac': 0.01,
}

# Get actual bounds (cFUSE if available, else fallback)
PARAM_BOUNDS = CFUSE_PARAM_BOUNDS if HAS_CFUSE else FALLBACK_PARAM_BOUNDS
DEFAULT_PARAMS = CFUSE_DEFAULT_PARAMS if HAS_CFUSE else FALLBACK_DEFAULT_PARAMS


@OptimizerRegistry.register_parameter_manager('CFUSE')
class CFUSEParameterManager(BaseParameterManager):
    """
    Manages cFUSE parameters for calibration.

    Provides:
    - Parameter bounds retrieval (from cFUSE or fallback)
    - Transformation between normalized [0,1] and physical space
    - Default values
    - Parameter validation

    When cFUSE is installed, uses the native PARAM_BOUNDS from the package.
    Otherwise falls back to reasonable defaults from literature.
    """

    def __init__(
        self,
        config: Dict,
        logger: logging.Logger,
        cfuse_settings_dir: Path
    ):
        """
        Initialize parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            cfuse_settings_dir: Path to cFUSE settings directory
        """
        super().__init__(config, logger, cfuse_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Parse cFUSE parameters to calibrate from config
        cfuse_params_str = self._get_config_value(lambda: self.config.model.cfuse.params_to_calibrate, default=None, dict_key='CFUSE_PARAMS_TO_CALIBRATE')
        # Handle None, empty string, or 'default' as signal to use default parameter list
        if cfuse_params_str is None or cfuse_params_str == '' or cfuse_params_str == 'default':
            # Default 13 parameters for best FUSE decision structure
            cfuse_params_str = 'S1_max,ku,Ac_max,b,kappa,v_A,v_B,r1,smooth_frac,T_rain,T_melt,MFMAX,MFMIN'

        self.cfuse_params = [p.strip() for p in str(cfuse_params_str).split(',') if p.strip()]

        # Validate parameters against available bounds
        if HAS_CFUSE:
            self._validate_params()

        # Store internal references
        self.all_bounds = PARAM_BOUNDS.copy()
        self.defaults = DEFAULT_PARAMS.copy()
        self.calibration_params = self.cfuse_params

        # Apply custom bounds from config if provided
        custom_bounds = self._get_config_value(lambda: None, default={}, dict_key='CFUSE_PARAM_BOUNDS')
        if custom_bounds:
            for param_name, bnd in custom_bounds.items():
                if isinstance(bnd, (list, tuple)) and len(bnd) == 2:
                    self.all_bounds[param_name] = (float(bnd[0]), float(bnd[1]))
                elif isinstance(bnd, dict) and 'min' in bnd and 'max' in bnd:
                    self.all_bounds[param_name] = (float(bnd['min']), float(bnd['max']))

    def _validate_params(self) -> None:
        """Validate that calibration parameters exist in bounds."""
        invalid = [p for p in self.cfuse_params if p not in PARAM_BOUNDS]
        if invalid:
            self.logger.warning(
                f"Unknown cFUSE parameters: {invalid}. "
                f"Available parameters: {list(PARAM_BOUNDS.keys())}"
            )

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return cFUSE parameter names from config."""
        return self.cfuse_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return cFUSE parameter bounds (with any config overrides applied)."""
        return {
            name: {'min': self.all_bounds[name][0], 'max': self.all_bounds[name][1]}
            for name in self.cfuse_params
            if name in self.all_bounds
        }

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        cFUSE doesn't have a parameter file to update.
        Parameters are passed directly to the model during simulation.
        """
        return True

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from config or defaults."""
        initial_params = self._get_config_value(lambda: None, default='default', dict_key='CFUSE_INITIAL_PARAMS')

        if initial_params == 'default':
            self.logger.debug("Using standard cFUSE defaults for initial parameters")
            return {p: self.defaults.get(p, 0.0) for p in self.cfuse_params}

        # Parse string-based initial params if provided
        if isinstance(initial_params, str) and initial_params != 'default':
            try:
                param_dict = {}
                for pair in initial_params.split(','):
                    if '=' in pair:
                        k, v = pair.split('=')
                        param_dict[k.strip()] = float(v.strip())
                return param_dict
            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.warning(f"Could not parse CFUSE_INITIAL_PARAMS: {e}")
                return {p: self.defaults.get(p, 0.0) for p in self.cfuse_params}

        return {p: self.defaults.get(p, 0.0) for p in self.cfuse_params}

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
            raise KeyError(f"Unknown cFUSE parameter: {param_name}")
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


def get_cfuse_calibration_bounds(
    params_to_calibrate: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to get cFUSE calibration bounds.

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
    'T_rain': {
        'name': 'Rain Threshold',
        'description': 'Temperature threshold for snow/rain partitioning',
        'unit': '°C',
        'category': 'snow'
    },
    'b': {
        'name': 'VIC b Parameter',
        'description': 'Variable infiltration capacity shape parameter',
        'unit': '-',
        'category': 'surface'
    },
}
