#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MESH Parameter Manager

Handles MESH parameter bounds, normalization, and .ini file updates.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import re

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_mesh_bounds


class MESHParameterManager(BaseParameterManager):
    """Handles MESH parameter bounds, normalization, and file updates"""

    def __init__(self, config: Dict, logger: logging.Logger, mesh_settings_dir: Path):
        super().__init__(config, logger, mesh_settings_dir)

        # MESH-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse MESH parameters to calibrate from config
        mesh_params_str = config.get('MESH_PARAMS_TO_CALIBRATE',
                                     'ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR')
        self.mesh_params = [p.strip() for p in mesh_params_str.split(',') if p.strip()]

        # Paths to parameter files
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.mesh_forcing_dir = self.project_dir / 'forcing' / 'MESH_input'

        # MESH parameter files (in forcing directory, not settings)
        self.class_params_file = self.mesh_forcing_dir / 'MESH_parameters_CLASS.ini'
        self.hydro_params_file = self.mesh_forcing_dir / 'MESH_parameters_hydrology.ini'
        self.routing_params_file = self.mesh_forcing_dir / 'MESH_parameters.txt'

        # Map parameters to files
        self.param_file_map = {
            'ZSNL': 'CLASS', 'ZPLG': 'CLASS', 'ZPLS': 'CLASS',
            'FRZTH': 'CLASS', 'MANN': 'CLASS',
            'RCHARG': 'hydrology', 'DRAINFRAC': 'hydrology', 'BASEFLW': 'hydrology',
            'DTMINUSR': 'routing',  # In main MESH_parameters.txt
        }

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return MESH parameter names from config."""
        return self.mesh_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return MESH parameter bounds from central registry."""
        return get_mesh_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update MESH parameter .ini files."""
        return self.update_mesh_params(params)

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from .ini files or defaults."""
        try:
            params = {}

            for param_name in self.mesh_params:
                value = self._read_param_from_file(param_name)
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
    # MESH-SPECIFIC METHODS
    # ========================================================================

    def update_mesh_params(self, params: Dict[str, float]) -> bool:
        """
        Update MESH parameter files with new values.

        MESH uses .ini format: KEY value
        """
        try:
            # Group parameters by file
            class_params = {}
            hydro_params = {}
            routing_params = {}

            for param_name, value in params.items():
                file_type = self.param_file_map.get(param_name, 'unknown')
                if file_type == 'CLASS':
                    class_params[param_name] = value
                elif file_type == 'hydrology':
                    hydro_params[param_name] = value
                elif file_type == 'routing':
                    routing_params[param_name] = value

            success = True

            # Update CLASS parameters
            if class_params:
                success = success and self._update_ini_file(
                    self.class_params_file, class_params
                )

            # Update hydrology parameters
            if hydro_params:
                success = success and self._update_ini_file(
                    self.hydro_params_file, hydro_params
                )

            # Update routing parameters (in main MESH_parameters.txt)
            if routing_params:
                success = success and self._update_ini_file(
                    self.routing_params_file, routing_params
                )

            return success

        except Exception as e:
            self.logger.error(f"Error updating MESH parameters: {e}")
            return False

    def _update_ini_file(self, file_path: Path, params: Dict[str, float]) -> bool:
        """Update a .ini format parameter file."""
        try:
            if not file_path.exists():
                self.logger.error(f"Parameter file not found: {file_path}")
                return False

            with open(file_path, 'r') as f:
                content = f.read()

            updated = 0
            for param_name, value in params.items():
                # Match: KEY value (ignore comments starting with !)
                pattern = rf'^({param_name}\s+)([\d\.\-\+eE]+)'

                def replacer(match):
                    nonlocal updated
                    updated += 1
                    return f"{match.group(1)}{value:.6f}"

                content, n = re.subn(pattern, replacer, content, count=1, flags=re.MULTILINE)

                if n == 0:
                    self.logger.warning(f"Parameter {param_name} not found in {file_path.name}")

            with open(file_path, 'w') as f:
                f.write(content)

            self.logger.debug(f"Updated {updated} parameters in {file_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating {file_path.name}: {e}")
            return False

    def _read_param_from_file(self, param_name: str) -> Optional[float]:
        """Read a parameter value from the appropriate file."""
        try:
            file_type = self.param_file_map.get(param_name)

            if file_type == 'CLASS':
                file_path = self.class_params_file
            elif file_type == 'hydrology':
                file_path = self.hydro_params_file
            elif file_type == 'routing':
                file_path = self.routing_params_file
            else:
                return None

            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                content = f.read()

            pattern = rf'^{param_name}\s+([\d\.\-\+eE]+)'
            match = re.search(pattern, content, re.MULTILINE)

            if match:
                return float(match.group(1))

            return None

        except Exception as e:
            self.logger.warning(f"Error reading {param_name}: {e}")
            return None

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (midpoint of bounds)."""
        params = {}
        for param_name in self.mesh_params:
            bounds = self.param_bounds[param_name]
            params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params
