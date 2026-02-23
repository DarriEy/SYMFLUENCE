"""
ML Parameter Manager

Generic parameter manager for ML-based models (LSTM/GNN).
Uses config-defined parameter lists and bounds.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('ML')
class MLParameterManager(BaseParameterManager):
    """
    Parameter manager for ML models using config-defined bounds.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        settings_dir: Path,
        params_key: str,
        bounds_key: str
    ):
        self.params_key = params_key
        self.bounds_key = bounds_key
        super().__init__(config, logger, settings_dir)

    def _get_parameter_names(self) -> List[str]:
        params_raw = self._get_config_value(lambda: None, default='', dict_key=self.params_key)
        params = [p.strip() for p in str(params_raw).split(',') if p.strip()]
        if not params:
            self.logger.warning(f"No parameters configured for {self.params_key}")
        return params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        bounds = self._get_config_value(lambda: None, default=None, dict_key=self.bounds_key) or self._get_config_value(lambda: None, default={}, dict_key='PARAMETER_BOUNDS')
        parsed_bounds: Dict[str, Dict[str, float]] = {}

        for param in self._get_parameter_names():
            if param not in bounds:
                self.logger.warning(f"Missing bounds for ML parameter: {param}")
                continue
            limits = bounds[param]
            if not isinstance(limits, (list, tuple)) or len(limits) < 2:
                self.logger.warning(f"Invalid bounds for {param}: {limits}")
                continue
            parsed_bounds[param] = {'min': float(limits[0]), 'max': float(limits[1])}

        return parsed_bounds

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        # ML models read parameters from config at runtime; no files to update.
        return True

    def get_initial_parameters(self) -> Dict[str, Any]:
        import math

        initial_params: Dict[str, Any] = {}
        for param in self._get_parameter_names():
            cfg_val = self._get_config_value(lambda: None, default=None, dict_key=param)
            if cfg_val is not None:
                initial_params[param] = cfg_val
                continue
            bounds = self.param_bounds.get(param)
            if bounds:
                if bounds.get('transform') == 'log' and bounds['min'] > 0:
                    initial_params[param] = math.sqrt(bounds['min'] * bounds['max'])
                else:
                    initial_params[param] = (bounds['min'] + bounds['max']) / 2.0
        return initial_params
