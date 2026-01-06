#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GR Parameter Manager

Handles GR parameter bounds, normalization, and configuration updates.
Since GR doesn't use parameter files but receives them via config/runner,
this manager simply prepares the parameters for the GRRunner.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_gr_bounds


class GRParameterManager(BaseParameterManager):
    """Handles GR parameter bounds, normalization, and configuration updates."""

    def __init__(self, config: Dict, logger: logging.Logger, gr_settings_dir: Path):
        super().__init__(config, logger, gr_settings_dir)

        # GR-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse GR parameters to calibrate from config
        gr_params_str = config.get('GR_PARAMS_TO_CALIBRATE', 'X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff')
        self.gr_params = [p.strip() for p in gr_params_str.split(',') if p.strip()]

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return GR parameter names from config."""
        return self.gr_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return GR parameter bounds from central registry."""
        return get_gr_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        GR doesn't have a parameter file to update.
        Parameters are passed via GR_EXTERNAL_PARAMS in config.
        We return True as 'applying' parameters happens in the worker/runner.
        """
        return True

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from defaults."""
        return self._get_default_initial_values()

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (midpoint of bounds)."""
        params = {}
        for param_name in self.gr_params:
            bounds = self.param_bounds.get(param_name)
            if bounds:
                params[param_name] = (bounds['min'] + bounds['max']) / 2
            else:
                # Default values for GR parameters if not in bounds registry
                defaults = {
                    'X1': 350.0, 'X2': 0.0, 'X3': 100.0, 'X4': 1.7,
                    'CTG': 0.0, 'Kf': 3.69
                }
                params[param_name] = defaults.get(param_name, 1.0)
        return params
