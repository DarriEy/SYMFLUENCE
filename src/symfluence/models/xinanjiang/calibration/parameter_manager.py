# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Xinanjiang Parameter Manager.

Provides parameter bounds, transformations, and management for Xinanjiang calibration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from symfluence.models.xinanjiang.parameters import (
    DEFAULT_PARAMS,
    LOG_TRANSFORM_PARAMS,
    PARAM_BOUNDS,
    enforce_ki_kg_constraint,
)
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_snow17_bounds, get_xinanjiang_bounds
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('XINANJIANG')
class XinanjiangParameterManager(BaseParameterManager):
    """Manages Xinanjiang parameters for calibration."""

    def __init__(self, config: Dict, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)

        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse parameters to calibrate
        params_str = config.get('XINANJIANG_PARAMS_TO_CALIBRATE', 'all')
        if params_str is None or params_str == '' or params_str == 'all':
            self.xinanjiang_params = list(PARAM_BOUNDS.keys())
            logger.debug(f"Calibrating all {len(self.xinanjiang_params)} Xinanjiang parameters")
        else:
            self.xinanjiang_params = [p.strip() for p in str(params_str).split(',') if p.strip()]
            logger.debug(f"Calibrating Xinanjiang parameters: {self.xinanjiang_params}")

        self.all_bounds = PARAM_BOUNDS.copy()
        self.defaults = DEFAULT_PARAMS.copy()

        # Add Snow-17 parameters when snow module enabled
        self.snow_module = str(config.get('XINANJIANG_SNOW_MODULE', 'none'))
        if self.snow_module == 'snow17':
            from symfluence.models.snow17.parameters import SNOW17_DEFAULTS, SNOW17_PARAM_BOUNDS
            self.all_bounds.update(SNOW17_PARAM_BOUNDS)
            self.defaults.update(SNOW17_DEFAULTS)
            self.xinanjiang_params = self.xinanjiang_params + list(SNOW17_PARAM_BOUNDS.keys())
            logger.info(
                f"Snow-17 coupling enabled: {len(self.xinanjiang_params)} total calibration parameters"
            )

        self.calibration_params = self.xinanjiang_params

    def _get_parameter_names(self) -> List[str]:
        return self.xinanjiang_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        bounds = get_xinanjiang_bounds()
        if self.snow_module == 'snow17':
            bounds.update(get_snow17_bounds())
        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Xinanjiang runs in-memory; no files to update."""
        return True

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values.

        Uses geometric mean for log-transformed params.
        Enforces KI+KG constraint.
        """
        initial = {}
        for p in self.xinanjiang_params:
            if p in LOG_TRANSFORM_PARAMS:
                lo, hi = self.all_bounds[p]
                initial[p] = np.sqrt(lo * hi)  # Geometric mean
            else:
                initial[p] = self.defaults[p]

        # Enforce KI+KG constraint
        if 'KI' in initial and 'KG' in initial:
            initial['KI'], initial['KG'] = enforce_ki_kg_constraint(
                initial['KI'], initial['KG']
            )

        return initial

    def get_bounds(self, param_name: str) -> Tuple[float, float]:
        if param_name not in self.all_bounds:
            raise KeyError(f"Unknown Xinanjiang parameter: {param_name}")
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

        # Enforce KI+KG constraint after denormalization
        if 'KI' in params and 'KG' in params:
            params['KI'], params['KG'] = enforce_ki_kg_constraint(
                params['KI'], params['KG']
            )

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

        # Check KI+KG constraint
        ki = params.get('KI', 0.0)
        kg = params.get('KG', 0.0)
        if ki + kg >= 1.0:
            violations.append(f"KI+KG={ki+kg:.3f} >= 1.0")

        return len(violations) == 0, violations

    def clip_to_bounds(self, params: Dict[str, float]) -> Dict[str, float]:
        clipped = {}
        for name, value in params.items():
            if name in self.all_bounds:
                low, high = self.all_bounds[name]
                clipped[name] = np.clip(value, low, high)
            else:
                clipped[name] = value

        # Enforce KI+KG after clipping
        if 'KI' in clipped and 'KG' in clipped:
            clipped['KI'], clipped['KG'] = enforce_ki_kg_constraint(
                clipped['KI'], clipped['KG']
            )

        return clipped

    def get_complete_params(self, partial_params: Dict[str, float]) -> Dict[str, float]:
        complete = self.defaults.copy()
        complete.update(partial_params)
        return complete
