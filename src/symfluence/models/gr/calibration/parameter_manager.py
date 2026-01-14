#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GR Parameter Manager

Handles GR parameter bounds, normalization, and configuration updates.
Since GR doesn't use parameter files but receives them via config/runner,
this manager simply prepares the parameters for the GRRunner.
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_gr_bounds
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('GR')
class GRParameterManager(BaseParameterManager):
    """Handles GR parameter bounds, normalization, and configuration updates."""

    def __init__(self, config: Dict, logger: logging.Logger, gr_settings_dir: Path):
        super().__init__(config, logger, gr_settings_dir)

        # GR-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse GR parameters to calibrate from config
        gr_params_str = config.get('GR_PARAMS_TO_CALIBRATE')
        if gr_params_str is None:
            gr_params_str = 'X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff'

        self.gr_params = [p.strip() for p in str(gr_params_str).split(',') if p.strip()]

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
        """Get initial parameter values from config or defaults."""
        # Check for explicit initial params in config
        initial_params = self.config.get('GR_INITIAL_PARAMS', 'default')

        if initial_params == 'default':
            # Try to load from previous internal calibration if it exists
            params = self._load_params_from_rdata()
            if params:
                self.logger.info(f"Loaded {len(params)} initial parameters from previous GR calibration")
                return {k: v for k, v in params.items() if k in self.gr_params}

            # Fallback to standard airGR defaults (instead of bounds midpoints)
            self.logger.debug("Using standard airGR defaults for initial parameters")
            defaults = {
                'X1': 350.0, 'X2': 0.0, 'X3': 100.0, 'X4': 1.7,
                'CTG': 0.5, 'Kf': 4.0, 'Gratio': 0.1, 'Albedo_diff': 0.1
            }
            return {k: v for k, v in defaults.items() if k in self.gr_params}

        if initial_params and isinstance(initial_params, dict):
            # Filter to only include parameters we are calibrating
            return {k: float(v) for k, v in initial_params.items() if k in self.gr_params}

        return self._get_default_initial_values()

    def _load_params_from_rdata(self) -> Optional[Dict[str, float]]:
        """Attempt to load parameters from GR_calib.Rdata in the simulation directory."""
        try:
            data_dir = self.config.get('SYMFLUENCE_DATA_DIR')
            domain = self.config.get('DOMAIN_NAME')
            exp_id = self.config.get('EXPERIMENT_ID')

            if not all([data_dir, domain, exp_id]):
                return None

            rdata_path = Path(data_dir) / f"domain_{domain}" / "simulations" / exp_id / "GR" / "GR_calib.Rdata"

            if not rdata_path.exists():
                return None

            import rpy2.robjects as robjects
            robjects.r['load'](str(rdata_path))

            if 'OutputsCalib' not in robjects.globalenv:
                return None

            outputs_calib = robjects.globalenv['OutputsCalib']
            param_final = list(outputs_calib.rx2('ParamFinalR'))

            # Map based on parameter count
            # 4: GR4J
            # 6: GR4J + CemaNeige (CTG, Kf)
            # 8: GR4J + CemaNeige + Hysteresis (Gratio, Albedo_diff)
            if len(param_final) == 4:
                param_names = ['X1', 'X2', 'X3', 'X4']
            elif len(param_final) == 6:
                param_names = ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf']
            elif len(param_final) == 8:
                param_names = ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf', 'Gratio', 'Albedo_diff']
            else:
                self.logger.warning(f"Unexpected number of parameters in Rdata: {len(param_final)}")
                # Generic names
                param_names = [f"P{i+1}" for i in range(len(param_final))]

            self.logger.info(f"Loaded {len(param_final)} parameters from Rdata: {param_names}")
            return {name: val for name, val in zip(param_names, param_final)}

        except Exception as e:
            self.logger.debug(f"Could not load initial parameters from Rdata: {e}")
            return None

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
