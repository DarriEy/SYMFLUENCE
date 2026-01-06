#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HYPE Parameter Manager

Handles HYPE parameter bounds, normalization, and par.txt file updates.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_hype_bounds


class HYPEParameterManager(BaseParameterManager):
    """Handles HYPE parameter bounds, normalization, and file updates"""

    def __init__(self, config: Dict, logger: logging.Logger, hype_settings_dir: Path):
        super().__init__(config, logger, hype_settings_dir)

        # HYPE-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse HYPE parameters to calibrate from config
        hype_params_str = config.get('HYPE_PARAMS_TO_CALIBRATE',
                                     'ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rivvel,damp')
        self.hype_params = [p.strip() for p in hype_params_str.split(',') if p.strip()]

        # Path to par.txt file
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.hype_setup_dir = self.project_dir / 'settings' / 'HYPE'
        self.par_file_path = self.hype_setup_dir / 'par.txt'

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return HYPE parameter names from config."""
        return self.hype_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return HYPE parameter bounds from central registry."""
        return get_hype_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update par.txt file with new parameter values."""
        return self.update_par_file(params)

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from par.txt or defaults."""
        try:
            if not self.par_file_path.exists():
                self.logger.warning(f"par.txt not found: {self.par_file_path}")
                return self._get_default_initial_values()

            # Parse existing par.txt
            params = {}
            with open(self.par_file_path, 'r') as f:
                content = f.read()

            for param_name in self.hype_params:
                # Match: param_name followed by value(s), ignoring comments
                pattern = rf'^{param_name}\s+([\d\.\-\+eE]+)'
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    params[param_name] = float(match.group(1))
                else:
                    # Use midpoint of bounds
                    bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2

            # Validate and fix any problematic initial values
            params = self._validate_and_fix_initial_parameters(params)
            return params

        except Exception as e:
            self.logger.error(f"Error reading initial parameters: {e}")
            return self._get_default_initial_values()

    # ========================================================================
    # HYPE-SPECIFIC METHODS
    # ========================================================================

    def update_par_file(self, params: Dict[str, float]) -> bool:
        """
        Update par.txt file with new parameter values.

        Uses regex to find and replace parameter values while preserving
        file structure and comments.
        """
        try:
            if not self.par_file_path.exists():
                self.logger.error(f"par.txt not found: {self.par_file_path}")
                return False

            with open(self.par_file_path, 'r') as f:
                content = f.read()

            updated = 0
            for param_name, value in params.items():
                # Match parameter line and replace first numeric value
                # Pattern: param_name + whitespace + numeric value(s)
                pattern = rf'^({param_name}\s+)([\d\.\-\+eE]+)'

                def replacer(match):
                    nonlocal updated
                    updated += 1
                    return f"{match.group(1)}{value:.6f}"

                content, n = re.subn(pattern, replacer, content, count=1, flags=re.MULTILINE)

                if n == 0:
                    self.logger.warning(f"Parameter {param_name} not found in par.txt")

            # Write updated content
            with open(self.par_file_path, 'w') as f:
                f.write(content)

            self.logger.debug(f"Updated {updated} parameters in par.txt")
            return True

        except Exception as e:
            self.logger.error(f"Error updating par.txt: {e}")
            return False

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (midpoint of bounds)."""
        params = {}
        for param_name in self.hype_params:
            bounds = self.param_bounds[param_name]
            params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def _validate_and_fix_initial_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Validate initial parameters and fix common issues.

        Addresses problems where initial par.txt may have problematic values
        that prevent the optimizer from finding good solutions.
        """
        fixed_params = params.copy()

        # Critical fix: cevp=0.0 disables evapotranspiration entirely
        # This breaks lumped models by preventing water from leaving the system
        if 'cevp' in fixed_params and fixed_params['cevp'] <= 0.01:
            bounds = self.param_bounds.get('cevp', {'min': 0.0, 'max': 1.0})
            # Set to a reasonable initial value (60% of PET)
            fixed_params['cevp'] = 0.6
            self.logger.warning(
                f"Initial cevp value was {params.get('cevp', 'unknown'):.4f} (disables ET). "
                f"Reset to {fixed_params['cevp']:.4f} to enable evapotranspiration."
            )

        return fixed_params
