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
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_mesh_bounds
from symfluence.optimization.registry import OptimizerRegistry

logger = logging.getLogger(__name__)

@OptimizerRegistry.register_parameter_manager('MESH')
class MESHParameterManager(BaseParameterManager):
    """Handles MESH parameter bounds, normalization, and file updates"""

    # Marker comments used by meshflow to identify parameter lines in CLASS.ini
    # Maps: marker substring -> (expected param names at each position, total_value_count)
    CLASS_LINE_MARKERS = {
        'DRN/SDEP': (['DRN', 'SDEP', 'FARE', 'DD'], 4),
        'XSLP/XDRAINH/MANN/KSAT': (['XSLP', 'XDRAINH', 'MANN_CLASS', 'KSAT'], 5),
    }

    def __init__(self, config: Dict, logger: logging.Logger, mesh_settings_dir: Path):
        super().__init__(config, logger, mesh_settings_dir)

        # MESH-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse MESH parameters to calibrate from config
        mesh_params_str = config.get('MESH_PARAMS_TO_CALIBRATE')
        if mesh_params_str is None:
            # Default to parameters that control runoff generation.
            # CLASS params: KSAT (infiltration), DRN (drainage), SDEP (soil depth),
            #               XSLP (slope), XDRAINH (horiz. drainage), MANN_CLASS (roughness)
            # Hydrology params: FLZ/PWR (baseflow recession), ZSNL/ZPLS (ponding),
            #                   RCHARG (groundwater recharge)
            # Note: FRZTH excluded by default because parameter_fixer sets
            # FROZENSOILINFILFLAG=0 in run_options, making the frozen soil
            # infiltration threshold inert.  Add FRZTH to
            # MESH_PARAMS_TO_CALIBRATE only if FROZENSOILINFILFLAG is enabled.
            mesh_params_str = 'KSAT,DRN,SDEP,XSLP,XDRAINH,MANN_CLASS,FLZ,PWR,ZSNL,ZPLS,RCHARG'

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
            # FRZTH is a soil parameter written to hydrology.ini via key-value format.
            # Note: FREZTH in run_options is a separate flag (frozen soil infiltration on/off);
            # FRZTH here is the frozen soil infiltration *threshold* depth (m).
            'FRZTH': 'hydrology',
            'MANN': 'hydrology',
            'RCHARG': 'hydrology', 'DRAINFRAC': 'hydrology', 'BASEFLW': 'hydrology',
            'DTMINUSR': 'routing',  # In main MESH_parameters.txt
        }

        # CLASS.ini line mappings (0-indexed line number, 0-indexed position in line)
        # Format: param_name -> (line_index, value_position, num_values_in_line)
        # These are detected dynamically via _detect_class_param_positions()
        # and fall back to standard meshflow layout if detection fails.
        self.class_param_positions = self._detect_class_param_positions()

    def _get_parameter_names(self) -> List[str]:
        """Return MESH parameter names from config."""
        return self.mesh_params

    def _detect_class_param_positions(self) -> Dict[str, tuple]:
        """Dynamically detect parameter positions in CLASS.ini by scanning for
        marker comments like ``DRN/SDEP`` and ``XSLP/XDRAINH/MANN/KSAT``.

        Falls back to the standard meshflow layout (lines 12-13, 0-indexed)
        when the file is missing or markers are not found.

        Returns:
            Dict mapping parameter name to (line_index, col_position, n_values).
        """
        # Standard meshflow defaults (0-indexed lines)
        defaults = {
            'DRN': (12, 0, 4),
            'SDEP': (12, 1, 4),
            'XSLP': (13, 0, 5),
            'XDRAINH': (13, 1, 5),
            'MANN_CLASS': (13, 2, 5),
            'KSAT': (13, 3, 5),
        }

        class_file = self._resolve_ini_file('CLASS')
        if class_file is None:
            return defaults

        try:
            with open(class_file, 'r') as f:
                lines = f.readlines()
        except OSError:
            return defaults

        detected: Dict[str, tuple] = {}

        for line_idx, raw_line in enumerate(lines):
            for marker, (param_names, n_values) in self.CLASS_LINE_MARKERS.items():
                if marker in raw_line:
                    # The marker is in the *comment* portion of the DATA line that
                    # precedes or is on the same line.  meshflow puts the comment
                    # on the same line as the values (e.g.
                    #   "   1.000   2.500   1.000  50.000  12 DRN/SDEP/FARE/DD")
                    # The parameter DATA is on this same line.
                    data_line_idx = line_idx

                    # Also check the preceding line — some formats place the
                    # comment on a separate header line above the data.
                    parts = raw_line.split()
                    # If all initial tokens are numeric, the data is on this line
                    numeric_count = 0
                    for tok in parts:
                        try:
                            float(tok)
                            numeric_count += 1
                        except ValueError:
                            break
                    if numeric_count < n_values and line_idx + 1 < len(lines):
                        # Marker line is a header/comment; data is on the NEXT line
                        data_line_idx = line_idx + 1

                    for pos, pname in enumerate(param_names):
                        if pname in self.param_file_map:
                            detected[pname] = (data_line_idx, pos, n_values)

        if detected:
            self.logger.debug(
                f"Detected CLASS param positions from markers: "
                f"{', '.join(f'{k}=line{v[0]+1}:pos{v[1]}' for k, v in detected.items())}"
            )
            # Merge with defaults so that any params NOT matched still have
            # fallback positions (useful when a marker is missing)
            merged = dict(defaults)
            merged.update(detected)
            return merged

        self.logger.debug("No CLASS markers found; using default line positions")
        return defaults

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

    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, Any]:
        """
        Denormalize parameters and enforce MESH-specific feasibility constraints.

        Overrides base implementation to apply physical constraints that prevent
        CLASS snow energy balance crashes.
        """
        params = super().denormalize_parameters(normalized_array)
        params = self._validate_parameter_feasibility(params)
        return params

    def _validate_parameter_feasibility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce physical feasibility constraints between MESH parameters.

        These constraints prevent CLASS energy balance crashes caused by
        physically impossible or numerically unstable parameter combinations.

        Constraints enforced:
        1. KSAT * DRN <= 400 mm/hr (cap unrealistic drainage rate)
        2. KSAT * DRN * (1/SDEP) <= 250 (prevent instant soil emptying)
        3. XSLP >= 0.005 (prevent CLASS numerical issues)
        4. SDEP >= 1.5m always (shallow soils + high KSAT cause crashes)
        5. XDRAINH <= 0.5 when KSAT > 100 (cap combined drainage)
        6. FRZTH <= SDEP (frozen threshold can't exceed soil depth)
        7. MANN_CLASS >= 0.01 (prevent zero-friction overland flow instability)
        8. RCHARG + DRAINFRAC <= 1.0 (water balance closure)
        """
        validated = params.copy()

        ksat = validated.get('KSAT')
        drn = validated.get('DRN')
        sdep = validated.get('SDEP')
        xslp = validated.get('XSLP')
        xdrainh = validated.get('XDRAINH')
        frzth = validated.get('FRZTH')
        mann_class = validated.get('MANN_CLASS')
        rcharg = validated.get('RCHARG')
        drainfrac = validated.get('DRAINFRAC')

        # Constraint 4: SDEP >= 1.5m (applied early so downstream constraints use it)
        if sdep is not None and float(sdep) < 1.5:
            self.logger.debug(
                f"Feasibility: raised SDEP {float(sdep):.2f} -> 1.5m "
                f"(minimum for CLASS stability)"
            )
            validated['SDEP'] = 1.5
            sdep = 1.5

        # Constraint 1: KSAT * DRN <= 400
        # Moderately tightened from 500 to reduce crashes without
        # over-constraining DRN (which controls runoff generation)
        if ksat is not None and drn is not None:
            product = float(ksat) * float(drn)
            if product > 400.0:
                scale = 400.0 / product
                validated['DRN'] = float(drn) * scale
                self.logger.debug(
                    f"Feasibility: scaled DRN {float(drn):.3f} -> {validated['DRN']:.3f} "
                    f"(KSAT*DRN={product:.1f} > 400)"
                )
                drn = validated['DRN']

        # Constraint 2: KSAT * DRN * (1/SDEP) <= 250
        # Moderately tightened from 300 to prevent timestep-scale soil
        # depletion while preserving sufficient drainage search space
        if ksat is not None and drn is not None and sdep is not None:
            drain_intensity = float(ksat) * float(drn) / float(sdep)
            if drain_intensity > 250.0:
                scale = 250.0 / drain_intensity
                validated['DRN'] = float(drn) * scale
                self.logger.debug(
                    f"Feasibility: scaled DRN {float(drn):.3f} -> {validated['DRN']:.3f} "
                    f"(KSAT*DRN/SDEP={drain_intensity:.1f} > 250)"
                )

        # Constraint 3: XSLP >= 0.005
        if xslp is not None and float(xslp) < 0.005:
            validated['XSLP'] = 0.005
            self.logger.debug(
                f"Feasibility: raised XSLP {float(xslp):.4f} -> 0.005 "
                f"(minimum for CLASS stability)"
            )

        # Constraint 5: XDRAINH <= 0.5 when KSAT > 100
        if xdrainh is not None and ksat is not None:
            if float(ksat) > 100.0 and float(xdrainh) > 0.5:
                validated['XDRAINH'] = 0.5
                self.logger.debug(
                    f"Feasibility: capped XDRAINH {float(xdrainh):.3f} -> 0.5 "
                    f"(KSAT={float(ksat):.1f} > 100)"
                )

        # Constraint 6: FRZTH <= SDEP
        if frzth is not None and sdep is not None:
            if float(frzth) > float(validated['SDEP']):
                validated['FRZTH'] = float(validated['SDEP'])
                self.logger.debug(
                    f"Feasibility: capped FRZTH {float(frzth):.2f} -> {validated['FRZTH']:.2f} "
                    f"(must be <= SDEP)"
                )

        # Constraint 7: MANN_CLASS >= 0.01
        # Very low Manning's n causes overland flow velocity to blow up,
        # leading to numerical instability in the surface water balance
        if mann_class is not None and float(mann_class) < 0.01:
            validated['MANN_CLASS'] = 0.01
            self.logger.debug(
                f"Feasibility: raised MANN_CLASS {float(mann_class):.4f} -> 0.01 "
                f"(minimum for numerical stability)"
            )

        # Constraint 8: RCHARG + DRAINFRAC <= 1.0
        # These fractions partition soil drainage; their sum cannot exceed 1.0
        if rcharg is not None and drainfrac is not None:
            total = float(rcharg) + float(drainfrac)
            if total > 1.0:
                scale = 1.0 / total
                validated['RCHARG'] = float(rcharg) * scale
                validated['DRAINFRAC'] = float(drainfrac) * scale
                self.logger.debug(
                    f"Feasibility: scaled RCHARG+DRAINFRAC {total:.3f} -> 1.0"
                )

        return validated

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update MESH parameter .ini files."""
        self.logger.debug(f"Updating MESH files with params: {params}")
        return self.update_mesh_params(params)

    def _resolve_ini_file(self, file_type: str) -> Optional[Path]:
        """Resolve .ini file path, falling back to forcing/MESH_input/.

        The optimizer creates this parameter manager with settings/MESH/
        as the base directory, but the actual .ini files created by
        meshflow live in forcing/MESH_input/.  When reading initial
        values we need to find the real files.
        """
        if file_type == 'CLASS':
            filename = 'MESH_parameters_CLASS.ini'
        elif file_type == 'hydrology':
            filename = 'MESH_parameters_hydrology.ini'
        elif file_type == 'routing':
            filename = 'MESH_parameters.txt'
        else:
            return None

        # Primary: settings dir (mesh_settings_dir)
        primary = self.mesh_settings_dir / filename
        if primary.exists():
            return primary

        # Fallback: forcing/MESH_input/ (where meshflow writes the files)
        # mesh_settings_dir is typically {project_dir}/settings/MESH/
        # so project_dir is two levels up
        project_dir = self.mesh_settings_dir.parent.parent
        fallback = project_dir / 'forcing' / 'MESH_input' / filename
        if fallback.exists():
            self.logger.debug(
                f"File {filename} not in {self.mesh_settings_dir}, "
                f"using fallback: {fallback}"
            )
            return fallback

        return None

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from .ini files or defaults.

        Values read from files are checked against parameter bounds.
        If a value falls in the bottom or top 5% of the feasible range,
        it is replaced with the bounds midpoint to give DDS a more
        central starting position in the search space.
        """
        try:
            params = {}

            for param_name in self.mesh_params:
                bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                midpoint = (bounds['min'] + bounds['max']) / 2
                value = self._read_param_from_file(param_name)

                if value is not None:
                    # Check if value is near the boundary (bottom/top 5%)
                    param_range = bounds['max'] - bounds['min']
                    if param_range > 0:
                        normalized = (value - bounds['min']) / param_range
                        if normalized < 0.05 or normalized > 0.95:
                            self.logger.info(
                                f"Initial {param_name}={value:.4g} is near boundary "
                                f"[{bounds['min']:.4g}, {bounds['max']:.4g}] "
                                f"(normalized={normalized:.3f}); "
                                f"using midpoint {midpoint:.4g} for DDS start"
                            )
                            value = midpoint
                    params[param_name] = value
                else:
                    params[param_name] = midpoint

            self.logger.info(
                "Initial parameters: "
                + ", ".join(f"{k}={v:.4g}" for k, v in params.items())
            )

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

            if updated == 0:
                self.logger.error(
                    f"No CLASS parameters were updated in {file_path.name} "
                    f"(requested: {list(params.keys())}). "
                    f"File has {len(lines)} lines; check that line positions are correct."
                )
                return False

            with open(file_path, 'w') as f:
                f.writelines(lines)

            self.logger.debug(f"Updated {updated}/{len(params)} CLASS parameters in {file_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating CLASS file {file_path.name}: {e}")
            return False

    def _update_ini_file(self, file_path: Path, params: Dict[str, float]) -> bool:
        """Update a .ini format parameter file.

        If a scalar parameter is not found in the file, it is appended so that
        subsequent iterations (and MESH itself) can pick it up.
        """
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
            injected = 0
            for param_name, value in params.items():
                if param_name in self.ARRAY_PARAMS:
                    # Handle array parameters (e.g., WF_R2 has one value per river class)
                    # Format: WF_R2  0.30    0.30    0.30  # comment
                    # Replace all numeric values on the line with the calibrated value
                    pattern = rf'^({param_name}\s+)([\d\.\s\-\+eE]+)(#.*)?$'

                    def replace_array_values(m, _value=value):
                        prefix = m.group(1)
                        values_str = m.group(2)
                        comment = m.group(3) or ''
                        # Count how many values there were
                        num_values = len(re.findall(r'[\d\.\-\+eE]+', values_str))
                        # Create new values string with same count
                        new_values = '    '.join([f"{_value:.2f}"] * num_values)
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
                    # Parameter not found — inject it at the end of the file
                    # so MESH can read it and future regex updates will match.
                    if param_name not in self.ARRAY_PARAMS:
                        inject_line = f"{param_name}  {value:.6f}  # injected by SYMFLUENCE calibration\n"
                        if not content.endswith('\n'):
                            content += '\n'
                        content += inject_line
                        injected += 1
                        self.logger.info(
                            f"Injected missing parameter {param_name}={value:.6f} "
                            f"into {file_path.name}"
                        )
                    else:
                        self.logger.warning(
                            f"Array parameter {param_name} not found in "
                            f"{file_path.name}; cannot inject automatically"
                        )

            with open(file_path, 'w') as f:
                f.write(content)

            total = updated + injected
            if total == 0:
                self.logger.error(
                    f"No hydrology parameters were updated or injected in "
                    f"{file_path.name} (requested: {list(params.keys())})"
                )
                return False

            self.logger.debug(
                f"Hydrology file {file_path.name}: "
                f"{updated} updated, {injected} injected, "
                f"{len(params) - total} failed"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error updating {file_path.name}: {e}")
            return False

    def _read_param_from_file(self, param_name: str) -> Optional[float]:
        """Read a parameter value from the appropriate file.

        CLASS parameters use fixed-format positional lines, so they are
        read using ``self.class_param_positions`` instead of a key-value
        regex.  Hydrology and routing parameters use a standard
        ``KEY value`` .ini format.

        Falls back to forcing/MESH_input/ if the file is not found in
        the primary settings directory.
        """
        try:
            file_type = self.param_file_map.get(param_name)
            if file_type is None:
                return None

            file_path = self._resolve_ini_file(file_type)
            if file_path is None:
                return None

            # CLASS parameters are positional (fixed-format lines)
            if file_type == 'CLASS':
                if param_name not in self.class_param_positions:
                    self.logger.warning(
                        f"No positional mapping for CLASS param {param_name}"
                    )
                    return None

                line_idx, pos_idx, _num_values = self.class_param_positions[param_name]

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                if line_idx >= len(lines):
                    return None

                parts = lines[line_idx].split()
                if pos_idx >= len(parts):
                    return None

                return float(parts[pos_idx])

            # Hydrology / routing: key-value .ini format
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
