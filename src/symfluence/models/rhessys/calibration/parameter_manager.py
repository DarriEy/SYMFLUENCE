#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RHESSys Parameter Manager

Handles RHESSys parameter bounds, normalization, and definition file updates.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_rhessys_bounds
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('RHESSys')
class RHESSysParameterManager(BaseParameterManager):
    """Handles RHESSys parameter bounds, normalization, and file updates."""

    # Mapping from parameter names to definition files
    # NOTE: Parameter locations verified from RHESSys .params output files
    PARAM_FILE_MAP = {
        # hillslope.def parameters
        'gw_loss_coeff': 'hillslope.def',   # Groundwater loss/baseflow coefficient (slow)
        'gw_loss_fast_coeff': 'hillslope.def',   # Fast groundwater loss coefficient
        'gw_loss_fast_threshold': 'hillslope.def',  # Threshold storage for fast flow (m)

        # basin.def parameters
        'n_routing_power': 'basin.def',
        'psi_air_entry': 'basin.def',
        'pore_size_index': 'basin.def',

        # soil.def (patch defaults) parameters
        'sat_to_gw_coeff': 'soil.def',  # Saturated zone to GW recharge coefficient
        'porosity_0': 'soil.def',
        'porosity_decay': 'soil.def',
        'Ksat_0': 'soil.def',
        'Ksat_0_v': 'soil.def',
        'm': 'soil.def',
        'm_z': 'soil.def',
        'soil_depth': 'soil.def',
        'active_zone_z': 'soil.def',
        'snow_melt_Tcoef': 'soil.def',
        'snow_water_capacity': 'soil.def',
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
        'theta_mean_std_p1': 'soil.def',
        'theta_mean_std_p2': 'soil.def',
        'precip_lapse_rate': 'worldfile',  # Applied to worldfile, not def files
    }

    def __init__(self, config: Dict, logger: logging.Logger, rhessys_settings_dir: Path):
        super().__init__(config, logger, rhessys_settings_dir)

        # RHESSys-specific setup
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Parse RHESSys parameters to calibrate from config
        # Try typed config first, then legacy dict access
        rhessys_params_str = None
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'rhessys'):
                rhessys_params_str = config.model.rhessys.params_to_calibrate
        except (AttributeError, TypeError):
            pass

        # Fallback to dict-style access
        if rhessys_params_str is None:
            rhessys_params_str = self._get_config_value(lambda: self.config.model.rhessys.params_to_calibrate, default=None, dict_key='RHESSYS_PARAMS_TO_CALIBRATE')

        if rhessys_params_str is None:
            # Final fallback - includes Ksat_0_v which is critical for lateral vs vertical flow
            # gw_loss_fast_coeff and gw_loss_fast_threshold control fast GW drainage
            rhessys_params_str = (
                'sat_to_gw_coeff,gw_loss_coeff,gw_loss_fast_coeff,gw_loss_fast_threshold,'
                'm,Ksat_0,Ksat_0_v,porosity_0,porosity_decay,'
                'soil_depth,m_z,active_zone_z,snow_melt_Tcoef,snow_water_capacity,'
                'max_snow_temp,min_rain_temp,maximum_snow_energy_deficit,'
                'epc.gl_smax,epc.max_lai,theta_mean_std_p1,theta_mean_std_p2,precip_lapse_rate'
            )
            logger.warning(
                "RHESSYS_PARAMS_TO_CALIBRATE missing in config; using fallback list: "
                f"{rhessys_params_str}"
            )

        self.rhessys_params = [p.strip() for p in str(rhessys_params_str).split(',') if p.strip()]

        # Path to definition files
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.defs_dir = self.project_dir / 'settings' / 'RHESSys' / 'defs'

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return RHESSys parameter names from config."""
        return self.rhessys_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return RHESSys parameter bounds, with config overrides.

        Priority:
        1. Config RHESSYS_PARAM_BOUNDS (if specified)
        2. Central registry defaults

        This allows users to customize bounds in their config file while
        maintaining sensible defaults from the registry.
        """
        # Start with registry defaults
        bounds = get_rhessys_bounds()

        # Check for config overrides (preserves transform metadata from registry)
        config_bounds = self._get_config_value(lambda: None, default={}, dict_key='RHESSYS_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)

        # Log final bounds for calibrated parameters
        for param_name in self.rhessys_params:
            if param_name in bounds:
                b = bounds[param_name]
                transform = b.get('transform', 'linear')
                transform_str = " (log-space)" if transform == 'log' else ""
                self.logger.info(f"RHESSys param {param_name}: bounds=[{b['min']:.6f}, {b['max']:.6f}]{transform_str}")

        return bounds

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

        except Exception as e:  # noqa: BLE001 — calibration resilience
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

        # Worldfile params: read from the worldfile instead of def files
        if def_file_name == 'worldfile':
            world_file = self.project_dir / 'settings' / 'RHESSys' / 'worldfiles' / f'{self.domain_name}.world'
            if world_file.exists():
                try:
                    with open(world_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    pattern = rf'([\d\.\-\+eE]+)\s+{re.escape(param_name)}(\s.*|)$'
                    match = re.search(pattern, content, re.MULTILINE)
                    if match:
                        return float(match.group(1))
                except Exception:  # noqa: BLE001 — calibration resilience
                    pass
            return None

        def_file = self.defs_dir / def_file_name
        if not def_file.exists():
            return None

        try:
            with open(def_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # RHESSys def file format: value<whitespace>label
            # e.g., "0.000005    sat_to_gw_coeff"
            pattern = rf'^([\d\.\-\+eE]+)\s+{re.escape(param_name)}(\s.*|)$'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                return float(match.group(1))

            return None

        except Exception as e:  # noqa: BLE001 — calibration resilience
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

        except Exception as e:  # noqa: BLE001 — calibration resilience
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
            with open(def_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                updated = False
                for param_name, value in params.items():
                    # Match: value<whitespace>param_name (allow trailing comments)
                    pattern = rf'^([\d\.\-\+eE]+)(\s+)({re.escape(param_name)})(\s.*|)$'
                    match = re.match(pattern, line)
                    if match:
                        # Preserve whitespace formatting
                        new_line = f"{value:.6f}{match.group(2)}{match.group(3)}{match.group(4)}\n"
                        new_line = new_line.replace('\n\n', '\n')
                        updated_lines.append(new_line)
                        updated = True
                        self.logger.debug(f"Updated {param_name} = {value:.6f} in {def_file.name}")
                        break

                if not updated:
                    updated_lines.append(line)

            with open(def_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
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

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error copying def files to {worker_defs_dir}: {e}")
            return False
