#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RHESSys Parameter Manager

Handles RHESSys parameter bounds, normalization, and definition file updates.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_rhessys_bounds


class RHESSysParameterManager(BaseParameterManager):
    """Handles RHESSys parameter bounds, normalization, and file updates."""

    # Mapping from parameter names to definition files
    PARAM_FILE_MAP = {
        # basin.def parameters
        'sat_to_gw_coeff': 'basin.def',
        'gw_loss_coeff': 'basin.def',
        'n_routing_power': 'basin.def',
        'psi_air_entry': 'basin.def',
        'pore_size_index': 'basin.def',

        # soil.def (patch defaults) parameters
        'porosity_0': 'soil.def',
        'porosity_decay': 'soil.def',
        'Ksat_0': 'soil.def',
        'Ksat_0_v': 'soil.def',
        'm': 'soil.def',
        'm_z': 'soil.def',
        'soil_depth': 'soil.def',
        'active_zone_z': 'soil.def',
        'snow_melt_Tcoef': 'soil.def',
        'maximum_snow_energy_deficit': 'soil.def',

        # zone.def parameters
        'max_snow_temp': 'zone.def',
        'min_rain_temp': 'zone.def',

        # stratum.def (vegetation) parameters
        'epc.max_lai': 'stratum.def',
        'epc.gl_smax': 'stratum.def',
        'epc.gl_c': 'stratum.def',
        'epc.vpd_open': 'stratum.def',
        'epc.vpd_close': 'stratum.def',
    }

    def __init__(self, config: Dict, logger: logging.Logger, rhessys_settings_dir: Path):
        super().__init__(config, logger, rhessys_settings_dir)

        # RHESSys-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse RHESSys parameters to calibrate from config
        rhessys_params_str = config.get('RHESSYS_PARAMS_TO_CALIBRATE')
        if rhessys_params_str is None:
            rhessys_params_str = 'sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,soil_depth,snow_melt_Tcoef'
            
        self.rhessys_params = [p.strip() for p in str(rhessys_params_str).split(',') if p.strip()]

        # Path to definition files
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.defs_dir = self.project_dir / 'RHESSys_input' / 'defs'

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return RHESSys parameter names from config."""
        return self.rhessys_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return RHESSys parameter bounds from central registry."""
        return get_rhessys_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update RHESSys definition files with new parameter values."""
        return self.update_def_files(params)

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from definition files or defaults."""
        try:
            params = {}
            for param_name in self.rhessys_params:
                value = self._read_param_from_def(param_name)
                if value is not None:
                    params[param_name] = value
                else:
                    # Use midpoint of bounds
                    bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2
            return params

        except Exception as e:
            self.logger.error(f"Error reading initial parameters: {e}")
            return self._get_default_initial_values()

    # ========================================================================
    # RHESSYS-SPECIFIC METHODS
    # ========================================================================

    def _read_param_from_def(self, param_name: str) -> Optional[float]:
        """
        Read a parameter value from its definition file.

        Args:
            param_name: Parameter name

        Returns:
            Parameter value or None if not found
        """
        def_file_name = self.PARAM_FILE_MAP.get(param_name)
        if not def_file_name:
            return None

        def_file = self.defs_dir / def_file_name
        if not def_file.exists():
            return None

        try:
            with open(def_file, 'r') as f:
                content = f.read()

            # RHESSys def file format: value<whitespace>label
            # e.g., "0.000005    sat_to_gw_coeff"
            pattern = rf'^([\d\.\-\+eE]+)\s+{re.escape(param_name)}$'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                return float(match.group(1))

            return None

        except Exception as e:
            self.logger.warning(f"Error reading {param_name} from {def_file}: {e}")
            return None

    def update_def_files(self, params: Dict[str, float]) -> bool:
        """
        Update RHESSys definition files with new parameter values.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if successful
        """
        try:
            # Group parameters by definition file
            params_by_file: Dict[str, Dict[str, float]] = {}
            for param_name, value in params.items():
                def_file_name = self.PARAM_FILE_MAP.get(param_name)
                if def_file_name:
                    if def_file_name not in params_by_file:
                        params_by_file[def_file_name] = {}
                    params_by_file[def_file_name][param_name] = value
                else:
                    self.logger.warning(f"No def file mapping for parameter: {param_name}")

            # Update each definition file
            for def_file_name, file_params in params_by_file.items():
                def_file = self.defs_dir / def_file_name
                if not def_file.exists():
                    self.logger.error(f"Definition file not found: {def_file}")
                    continue

                self._update_single_def_file(def_file, file_params)

            return True

        except Exception as e:
            self.logger.error(f"Error updating definition files: {e}")
            return False

    def _update_single_def_file(self, def_file: Path, params: Dict[str, float]) -> bool:
        """
        Update a single RHESSys definition file.

        Args:
            def_file: Path to definition file
            params: Parameters to update in this file

        Returns:
            True if successful
        """
        try:
            with open(def_file, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                updated = False
                for param_name, value in params.items():
                    # Match: value<whitespace>param_name
                    pattern = rf'^([\d\.\-\+eE]+)(\s+)({re.escape(param_name)})(\s*)$'
                    match = re.match(pattern, line)
                    if match:
                        # Preserve whitespace formatting
                        new_line = f"{value:.6f}{match.group(2)}{match.group(3)}{match.group(4)}"
                        updated_lines.append(new_line)
                        updated = True
                        self.logger.debug(f"Updated {param_name} = {value:.6f} in {def_file.name}")
                        break

                if not updated:
                    updated_lines.append(line)

            with open(def_file, 'w') as f:
                f.writelines(updated_lines)

            return True

        except Exception as e:
            self.logger.error(f"Error updating {def_file}: {e}")
            return False

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (midpoint of bounds)."""
        params = {}
        for param_name in self.rhessys_params:
            bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
            params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def copy_defs_to_worker_dir(self, worker_defs_dir: Path) -> bool:
        """
        Copy definition files to a worker-specific directory for parallel calibration.

        Args:
            worker_defs_dir: Target directory for worker's definition files

        Returns:
            True if successful
        """
        import shutil

        try:
            worker_defs_dir.mkdir(parents=True, exist_ok=True)

            # Copy all .def files
            for def_file in self.defs_dir.glob('*.def'):
                shutil.copy2(def_file, worker_defs_dir / def_file.name)

            return True

        except Exception as e:
            self.logger.error(f"Error copying def files to {worker_defs_dir}: {e}")
            return False
