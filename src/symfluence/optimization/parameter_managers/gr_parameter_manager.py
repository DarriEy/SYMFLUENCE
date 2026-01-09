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
import numpy as np

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

    # airGR calibrated defaults - well-tuned starting points from literature
    AIRGR_CALIBRATED_DEFAULTS = {
        'X1': 257.24,       # Production store capacity (mm)
        'X2': 1.012,        # Groundwater exchange coefficient (mm/day)
        'X3': 88.23,        # Routing store capacity (mm)
        'X4': 2.208,        # Unit hydrograph time constant (days)
        'CTG': 0.0,         # Snow process parameter
        'Kf': 3.69,         # Melt factor (mm/°C/day)
        'Gratio': 0.1,      # Thermal coefficient of snow pack
        'Albedo_diff': 0.1, # Albedo diffusion coefficient
    }

    # Parameters that benefit from log-scale transformation during optimization
    LOG_SCALE_PARAMS = {'X1', 'X3'}

    def _get_default_initial_values(self) -> Dict[str, float]:
        """
        Get default initial parameter values using airGR calibrated defaults.

        Uses well-tuned defaults from the airGR package rather than midpoint
        of bounds, which provides a much better starting point for optimization.
        """
        params = {}
        for param_name in self.gr_params:
            # Prefer airGR calibrated defaults over midpoint
            if param_name in self.AIRGR_CALIBRATED_DEFAULTS:
                params[param_name] = self.AIRGR_CALIBRATED_DEFAULTS[param_name]
            elif param_name in self.param_bounds:
                # Fallback to midpoint for unknown parameters
                bounds = self.param_bounds[param_name]
                params[param_name] = (bounds['min'] + bounds['max']) / 2
            else:
                params[param_name] = 1.0
        return params

    # ========================================================================
    # LOG-SCALE TRANSFORMATION FOR IMPROVED OPTIMIZATION
    # ========================================================================
    # X1 and X3 have wide positive ranges where log-scale sampling is more
    # effective. This ensures uniform exploration across orders of magnitude.

    def normalize_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Normalize parameters to [0, 1] range with log-scale for X1 and X3.

        Log-scale transformation for X1 and X3 ensures DDS perturbations
        explore the parameter space more uniformly across orders of magnitude.
        """
        normalized = np.zeros(len(self.all_param_names))

        for i, param_name in enumerate(self.all_param_names):
            if param_name not in params or param_name not in self.param_bounds:
                self.logger.warning(f"Parameter {param_name} missing, using 0.5")
                normalized[i] = 0.5
                continue

            bounds = self.param_bounds[param_name]
            value = self._extract_scalar_value(params[param_name])

            if param_name in self.LOG_SCALE_PARAMS:
                # Log-scale normalization: normalize in log space
                # Ensures uniform exploration across orders of magnitude
                log_min = np.log(max(bounds['min'], 1e-10))
                log_max = np.log(max(bounds['max'], 1e-10))
                log_value = np.log(max(value, 1e-10))
                if log_max > log_min:
                    normalized[i] = (log_value - log_min) / (log_max - log_min)
                else:
                    normalized[i] = 0.5
            else:
                # Standard linear normalization
                range_size = bounds['max'] - bounds['min']
                if range_size == 0:
                    normalized[i] = 0.5
                else:
                    normalized[i] = (value - bounds['min']) / range_size

        return np.clip(normalized, 0.0, 1.0)

    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, Any]:
        """
        Denormalize parameters from [0, 1] range with log-scale for X1 and X3.

        Log-scale transformation ensures perturbations in normalized space
        translate to proportional changes in the original space.
        """
        params = {}

        for i, param_name in enumerate(self.all_param_names):
            if param_name not in self.param_bounds:
                self.logger.warning(f"No bounds for {param_name}, skipping")
                continue

            bounds = self.param_bounds[param_name]
            norm_val = normalized_array[i]

            if param_name in self.LOG_SCALE_PARAMS:
                # Log-scale denormalization
                log_min = np.log(max(bounds['min'], 1e-10))
                log_max = np.log(max(bounds['max'], 1e-10))
                log_value = log_min + norm_val * (log_max - log_min)
                denorm_value = np.exp(log_value)
            else:
                # Standard linear denormalization
                denorm_value = bounds['min'] + norm_val * (bounds['max'] - bounds['min'])

            # Clip to bounds for safety
            denorm_value = np.clip(denorm_value, bounds['min'], bounds['max'])
            params[param_name] = float(denorm_value)

        return params
