#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MESH Parameter Manager

Handles MESH parameter bounds, normalization, and .ini file updates.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_mesh_bounds
from symfluence.optimization.registry import OptimizerRegistry

logger = logging.getLogger(__name__)

@OptimizerRegistry.register_parameter_manager('MESH')
class MESHParameterManager(BaseParameterManager):
    """Handles MESH parameter bounds, normalization, and file updates"""

    def __init__(self, config: Dict, logger: logging.Logger, mesh_settings_dir: Path):
        super().__init__(config, logger, mesh_settings_dir)

        # MESH-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse MESH parameters to calibrate from config
        mesh_params_str = config.get('MESH_PARAMS_TO_CALIBRATE')
        if mesh_params_str is None:
            # Default to parameters that control runoff generation
            # CLASS params: KSAT (infiltration), DRN (drainage), SDEP (soil depth)
            # Hydrology params: XSLP (slope), MANN_CLASS (roughness)
            mesh_params_str = 'KSAT,DRN,SDEP,XSLP,MANN_CLASS'

        self.mesh_params = [p.strip() for p in str(mesh_params_str).split(',') if p.strip()]

        # Paths to parameter files
        # mesh_settings_dir is the base directory for model files in this run context
        self.mesh_settings_dir = mesh_settings_dir

        # MESH parameter files (usually in forcing directory, but mirrored in settings for workers)
        self.class_params_file = self.mesh_settings_dir / 'MESH_parameters_CLASS.ini'
        self.hydro_params_file = self.mesh_settings_dir / 'MESH_parameters_hydrology.ini'
        self.routing_params_file = self.mesh_settings_dir / 'MESH_parameters.txt'

        # Map parameters to files
        # NOTE: meshflow creates MESH_parameters_hydrology.ini with ZSNL, ZPLS, ZPLG, WF_R2
        # CLASS.ini has fixed-format parameters that control runoff generation
        self.param_file_map = {
            # CLASS.ini parameters (fixed-format, control runoff)
            'KSAT': 'CLASS',      # Saturated hydraulic conductivity (mm/hr) - Line 13, pos 4
            'DRN': 'CLASS',       # Drainage parameter - Line 12, pos 1
            'SDEP': 'CLASS',      # Soil depth (m) - Line 12, pos 2
            'XSLP': 'CLASS',      # Slope - Line 13, pos 1
            'XDRAINH': 'CLASS',   # Horizontal drainage - Line 13, pos 2
            'MANN_CLASS': 'CLASS', # Manning for overland flow - Line 13, pos 3
            # Meshflow-generated parameters (MESH_parameters_hydrology.ini)
            'ZSNL': 'hydrology', 'ZPLG': 'hydrology', 'ZPLS': 'hydrology',
            'WF_R2': 'hydrology',  # Channel roughness coefficient
            'FLZ': 'hydrology',    # Baseflow recession coefficient (critical for winter flow)
            'PWR': 'hydrology',    # Power coefficient for LZS drainage
            # Legacy parameters (may not exist in meshflow-generated files)
            'FRZTH': 'CLASS',
            'MANN': 'hydrology',
            'RCHARG': 'hydrology', 'DRAINFRAC': 'hydrology', 'BASEFLW': 'hydrology',
            'DTMINUSR': 'routing',  # In main MESH_parameters.txt
        }

        # CLASS.ini line mappings (0-indexed line number, 0-indexed position in line)
        # Format: param_name -> (line_index, value_position, num_values_in_line)
        # CLASS.ini structure (counting from 0):
        #   Line 12 (file line 13): DRN/SDEP/FARE/DD (4 values)
        #   Line 13 (file line 14): XSLP/XDRAINH/MANN/KSAT/MID (5 values, MID is text)
        self.class_param_positions = {
            'DRN': (12, 0, 4),       # File line 13 (0-indexed: 12), position 0
            'SDEP': (12, 1, 4),      # File line 13, position 1
            'XSLP': (13, 0, 5),      # File line 14 (0-indexed: 13), position 0
            'XDRAINH': (13, 1, 5),   # File line 14, position 1
            'MANN_CLASS': (13, 2, 5), # File line 14, position 2
            'KSAT': (13, 3, 5),      # File line 14, position 3
        }

    def _get_parameter_names(self) -> List[str]:
        """Return MESH parameter names from config."""
        return self.mesh_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return MESH parameter bounds, preferring config-specified bounds."""
        # Start with registry defaults
        bounds = get_mesh_bounds()

        # Override with config-specified bounds if available
        config_bounds = self.config.get('MESH_PARAM_BOUNDS', {})
        if config_bounds:
            self.logger.debug(f"Using config-specified MESH bounds for: {list(config_bounds.keys())}")
            for param_name, param_bounds in config_bounds.items():
                if isinstance(param_bounds, (list, tuple)) and len(param_bounds) == 2:
                    bounds[param_name] = {'min': float(param_bounds[0]), 'max': float(param_bounds[1])}
                elif isinstance(param_bounds, dict):
                    bounds[param_name] = param_bounds

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update MESH parameter .ini files."""
        self.logger.debug(f"Updating MESH files with params: {params}")
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

            # Update CLASS parameters (fixed-format file)
            if class_params:
                success = success and self._update_class_file(
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

    # Parameters that have multiple values per line (one per GRU/river class)
    ARRAY_PARAMS = {'WF_R2'}

    def _update_class_file(self, file_path: Path, params: Dict[str, float]) -> bool:
        """Update CLASS.ini fixed-format parameter file.

        CLASS.ini has a fixed-format where parameters are at specific line/column positions.
        Line numbers and positions are defined in self.class_param_positions.
        """
        try:
            self.logger.debug(f"Updating CLASS file: {file_path}")
            if not file_path.exists():
                self.logger.error(f"CLASS parameter file not found: {file_path}")
                return False

            with open(file_path, 'r') as f:
                lines = f.readlines()

            updated = 0
            for param_name, value in params.items():
                if param_name not in self.class_param_positions:
                    self.logger.warning(f"CLASS parameter {param_name} position not defined")
                    continue

                line_idx, pos_idx, num_values = self.class_param_positions[param_name]

                if line_idx >= len(lines):
                    self.logger.warning(f"Line {line_idx + 1} not found in CLASS file for {param_name}")
                    continue

                line = lines[line_idx]

                # Parse the line - values are space-separated, with comment at end
                # Example: "   0.030   0.350   0.100   0.050   100 Temp_sub-_gras   13 XSLP/..."
                parts = line.split()

                if pos_idx >= len(parts):
                    self.logger.warning(f"Position {pos_idx} not found in line for {param_name}")
                    continue

                # Format the new value (use appropriate precision)
                if abs(value) < 0.01:
                    new_val_str = f"{value:.3f}"
                elif abs(value) < 1:
                    new_val_str = f"{value:.3f}"
                elif abs(value) < 100:
                    new_val_str = f"{value:.2f}"
                else:
                    new_val_str = f"{value:.1f}"

                # Replace the value at the specific position
                old_val = parts[pos_idx]
                parts[pos_idx] = new_val_str

                # Reconstruct the line with proper spacing
                # CLASS.ini uses fixed-width columns (typically 8 characters per value)
                new_line = ""
                for i, part in enumerate(parts):
                    if i < num_values:
                        # Numeric values - right-align in 8-char field
                        new_line += f"{part:>8}"
                    else:
                        # Rest of line (comments, line number, etc.)
                        new_line += " " + part
                new_line += "\n"

                lines[line_idx] = new_line
                self.logger.debug(f"Updated {param_name}: {old_val} -> {new_val_str}")
                updated += 1

            with open(file_path, 'w') as f:
                f.writelines(lines)

            self.logger.debug(f"Updated {updated} CLASS parameters in {file_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating CLASS file {file_path.name}: {e}")
            return False

    def _update_ini_file(self, file_path: Path, params: Dict[str, float]) -> bool:
        """Update a .ini format parameter file."""
        try:
            self.logger.debug(f"Updating file: {file_path}")
            if not file_path.exists():
                self.logger.debug(f"File not found: {file_path}")
                if file_path.parent.exists():
                    self.logger.debug(f"Directory contents of {file_path.parent}: {os.listdir(file_path.parent)}")
                else:
                    self.logger.debug(f"Parent directory does not exist: {file_path.parent}")
                self.logger.error(f"Parameter file not found: {file_path}")
                return False

            with open(file_path, 'r') as f:
                content = f.read()

            updated = 0
            for param_name, value in params.items():
                if param_name in self.ARRAY_PARAMS:
                    # Handle array parameters (e.g., WF_R2 has one value per river class)
                    # Format: WF_R2  0.30    0.30    0.30  # comment
                    # Replace all numeric values on the line with the calibrated value
                    pattern = rf'^({param_name}\s+)([\d\.\s\-\+eE]+)(#.*)?$'

                    def replace_array_values(m):
                        prefix = m.group(1)
                        values_str = m.group(2)
                        comment = m.group(3) or ''
                        # Count how many values there were
                        num_values = len(re.findall(r'[\d\.\-\+eE]+', values_str))
                        # Create new values string with same count
                        new_values = '    '.join([f"{value:.2f}"] * num_values)
                        return prefix + new_values + '  ' + comment

                    content, n = re.subn(pattern, replace_array_values, content, count=1, flags=re.MULTILINE)
                else:
                    # Match: KEY value (ignore comments starting with !)
                    # Use word boundary \b to avoid matching partial names
                    # and handle KEY=value or KEY value
                    pattern = rf'\b({param_name})\b\s*[\s=]+\s*([\d\.\-\+eE]+)'
                    content, n = re.subn(pattern, lambda m: m.group(1) + " " + f"{value:.6f}", content, count=1, flags=re.IGNORECASE)

                if n > 0:
                    updated += 1
                else:
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
