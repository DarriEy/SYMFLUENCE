"""
SAC-SMA Parameter Manager.

Provides parameter bounds, transformations, and management for SAC-SMA calibration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from symfluence.models.sacsma.parameters import (
    DEFAULT_PARAMS,
    LOG_TRANSFORM_PARAMS,
    PARAM_BOUNDS,
    SACSMA_DEFAULTS,
    SACSMA_PARAM_BOUNDS,
)
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_sacsma_bounds
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('SACSMA')
class SacSmaParameterManager(BaseParameterManager):
    """Manages SAC-SMA + Snow-17 parameters for calibration."""

    def __init__(self, config: Dict, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, dict_key='EXPERIMENT_ID')

        # Snow module determines parameter set
        self.snow_module = str(self._get_config_value(
            lambda: self.config.model.sacsma.snow_module,
            default='snow17'
        ))

        if self.snow_module == 'none':
            available_bounds = SACSMA_PARAM_BOUNDS.copy()
            available_defaults = SACSMA_DEFAULTS.copy()
        else:
            available_bounds = PARAM_BOUNDS.copy()
            available_defaults = DEFAULT_PARAMS.copy()

        # Parse parameters to calibrate
        params_str = self._get_config_value(
            lambda: self.config.model.sacsma.params_to_calibrate,
            default='all'
        )
        if params_str is None or params_str == '' or params_str == 'all':
            self.sacsma_params = list(available_bounds.keys())
            logger.debug(f"Calibrating all {len(self.sacsma_params)} SAC-SMA parameters")
        else:
            self.sacsma_params = [p.strip() for p in str(params_str).split(',') if p.strip()]
            logger.debug(f"Calibrating SAC-SMA parameters: {self.sacsma_params}")

        # Apply config-level bounds overrides
        config_bounds = self._get_config_value(
            lambda: self.config.model.sacsma.param_bounds,
            default=None
        )
        if config_bounds:
            for param_name, override in config_bounds.items():
                if param_name in available_bounds:
                    if isinstance(override, (list, tuple)) and len(override) == 2:
                        available_bounds[param_name] = (float(override[0]), float(override[1]))
                    elif isinstance(override, dict) and 'min' in override and 'max' in override:
                        available_bounds[param_name] = (float(override['min']), float(override['max']))

        self.all_bounds = available_bounds
        self.defaults = available_defaults
        self.calibration_params = self.sacsma_params

    def _get_parameter_names(self) -> List[str]:
        return self.sacsma_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        bounds = get_sacsma_bounds()

        # Apply config-level overrides (same pattern as FUSE)
        config_bounds = self._get_config_value(
            lambda: self.config.model.sacsma.param_bounds,
            dict_key='SACSMA_PARAM_BOUNDS'
        )
        if config_bounds:
            self.logger.info("Using SACSMA_PARAM_BOUNDS from config (overriding registry defaults)")
            self._apply_config_bounds_override(bounds, config_bounds)

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """SAC-SMA runs in-memory; no files to update."""
        return True

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values.

        Uses geometric mean for log-transformed params.
        """
        initial = {}
        for p in self.sacsma_params:
            if p in LOG_TRANSFORM_PARAMS:
                lo, hi = self.all_bounds[p]
                initial[p] = np.sqrt(lo * hi)  # Geometric mean
            else:
                initial[p] = self.defaults[p]
        return initial

    def get_bounds(self, param_name: str) -> Tuple[float, float]:
        if param_name not in self.all_bounds:
            raise KeyError(f"Unknown SAC-SMA parameter: {param_name}")
        return self.all_bounds[param_name]

    def get_calibration_bounds(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {'min': self.all_bounds[name][0], 'max': self.all_bounds[name][1]}
            for name in self.calibration_params
        }

    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.all_bounds[p][0] for p in self.calibration_params])
        upper = np.array([self.all_bounds[p][1] for p in self.calibration_params])
        return lower, upper

    def get_default(self, param_name: str) -> float:
        return self.defaults.get(param_name, 0.0)

    def get_default_vector(self) -> np.ndarray:
        return np.array([self.defaults[p] for p in self.calibration_params])

    def normalize(self, params: Dict[str, float]) -> np.ndarray:
        normalized = []
        for name in self.calibration_params:
            value = params.get(name, self.defaults[name])
            low, high = self.all_bounds[name]
            if name in LOG_TRANSFORM_PARAMS:
                norm_val = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low) + 1e-10)
            else:
                norm_val = (value - low) / (high - low + 1e-10)
            normalized.append(np.clip(norm_val, 0, 1))
        return np.array(normalized)

    def denormalize(self, values: np.ndarray) -> Dict[str, float]:
        params = {}
        for i, name in enumerate(self.calibration_params):
            low, high = self.all_bounds[name]
            if name in LOG_TRANSFORM_PARAMS:
                params[name] = np.exp(np.log(low) + values[i] * (np.log(high) - np.log(low)))
            else:
                params[name] = low + values[i] * (high - low)
        return params

    def array_to_dict(self, values: np.ndarray) -> Dict[str, float]:
        return dict(zip(self.calibration_params, values))

    def dict_to_array(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params.get(p, self.defaults[p]) for p in self.calibration_params])

    def validate(self, params: Dict[str, float]) -> Tuple[bool, List[str]]:
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
        clipped = {}
        for name, value in params.items():
            if name in self.all_bounds:
                low, high = self.all_bounds[name]
                clipped[name] = np.clip(value, low, high)
            else:
                clipped[name] = value
        return clipped

    def get_complete_params(self, partial_params: Dict[str, float]) -> Dict[str, float]:
        complete = self.defaults.copy()
        complete.update(partial_params)
        return complete
