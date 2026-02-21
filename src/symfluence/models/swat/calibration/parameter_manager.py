"""
SWAT Parameter Manager

Handles SWAT parameter bounds, normalization, and text file updates.

SWAT parameters live in various text files (.bsn, .gw, .hru, .sol, .mgt)
within the TxtInOut directory. Parameters are modified using one of three
methods:
    - r__ (relative): new_value = original_value * (1 + change)
    - v__ (value replacement): new_value = change
    - a__ (absolute): new_value = original_value + change
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional


from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry

from ..parameters import PARAM_BOUNDS, PARAM_CHANGE_METHOD, PARAM_FILE_MAP


@OptimizerRegistry.register_parameter_manager('SWAT')
class SWATParameterManager(BaseParameterManager):
    """Handles SWAT parameter bounds, normalization, and file updates.

    SWAT parameters are stored in fixed-format text files. Each parameter
    has a designated file extension and modification method (relative,
    value replacement, or absolute).
    """

    def __init__(self, config: Dict, logger: logging.Logger, swat_settings_dir: Path):
        """
        Initialize SWAT parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            swat_settings_dir: Path to SWAT TxtInOut directory
        """
        super().__init__(config, logger, swat_settings_dir)

        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse SWAT parameters to calibrate from config
        swat_params_str = None
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'swat'):
                swat_params_str = config.model.swat.params_to_calibrate
        except (AttributeError, TypeError):
            pass

        if swat_params_str is None:
            swat_params_str = config.get('SWAT_PARAMS_TO_CALIBRATE')

        if swat_params_str is None:
            swat_params_str = 'CN2,ALPHA_BF,GW_DELAY,GWQMN,GW_REVAP,ESCO,SOL_AWC,SOL_K,SURLAG,SFTMP,SMTMP,SMFMX,SMFMN,TIMP'
            logger.warning(
                f"SWAT_PARAMS_TO_CALIBRATE missing; using fallback: {swat_params_str}"
            )

        self.swat_params = [p.strip() for p in str(swat_params_str).split(',') if p.strip()]

        # Path to TxtInOut directory
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        txtinout_name = 'TxtInOut'
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'swat'):
                txtinout_name = config.model.swat.txtinout_dir or txtinout_name
        except (AttributeError, TypeError):
            pass
        self.txtinout_dir = self.project_dir / 'SWAT_input' / txtinout_name

        # Store original parameter values for relative changes
        self._original_values: Dict[str, Dict[Path, Dict[str, float]]] = {}

    def _get_parameter_names(self) -> List[str]:
        """Return SWAT parameter names from config."""
        return self.swat_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return SWAT parameter bounds.

        Returns the bounds from parameters.py, with optional config overrides.
        """
        swat_bounds = dict(PARAM_BOUNDS)

        # Check for config overrides (preserves transform metadata from registry)
        config_bounds = self.config.get('SWAT_PARAM_BOUNDS', {})
        if config_bounds:
            self._apply_config_bounds_override(swat_bounds, config_bounds)

        # Log bounds for calibrated parameters
        for param_name in self.swat_params:
            if param_name in swat_bounds:
                b = swat_bounds[param_name]
                self.logger.info(f"SWAT param {param_name}: bounds=[{b['min']:.4f}, {b['max']:.4f}]")

        return swat_bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        Update SWAT text files with new parameter values.

        Groups parameters by file extension and applies changes using
        the appropriate method (relative, value replacement, or absolute).

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if all updates succeeded
        """
        try:
            if not self.txtinout_dir.exists():
                self.logger.error(f"SWAT TxtInOut directory not found: {self.txtinout_dir}")
                return False

            # Group parameters by file extension
            params_by_ext: Dict[str, Dict[str, float]] = {}
            for param_name, value in params.items():
                if param_name not in PARAM_FILE_MAP:
                    self.logger.warning(f"Unknown SWAT parameter: {param_name}")
                    continue
                ext = PARAM_FILE_MAP[param_name]
                if ext not in params_by_ext:
                    params_by_ext[ext] = {}
                params_by_ext[ext][param_name] = value

            # Apply changes to each file type
            success = True
            for ext, ext_params in params_by_ext.items():
                if ext == '.bsn':
                    # Basin file is a single file
                    if not self._update_basin_file(ext_params):
                        success = False
                else:
                    # Other file types may have multiple files (one per sub/HRU)
                    if not self._update_extension_files(ext, ext_params):
                        success = False

            return success

        except Exception as e:
            self.logger.error(f"Error updating SWAT parameter files: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_basin_file(self, params: Dict[str, float]) -> bool:
        """
        Update the basin file (.bsn) with new parameter values.

        All basin-level parameters use value replacement (v__).

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if successful
        """
        bsn_files = list(self.txtinout_dir.glob('*.bsn'))
        if not bsn_files:
            self.logger.error("No .bsn file found in TxtInOut")
            return False

        for bsn_file in bsn_files:
            try:
                with open(bsn_file, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    modified = False
                    for param_name, value in params.items():
                        if param_name in PARAM_CHANGE_METHOD and PARAM_FILE_MAP.get(param_name) == '.bsn':
                            # Check if this line contains the parameter
                            if f'| {param_name}' in line or f'|{param_name}' in line:
                                method = PARAM_CHANGE_METHOD[param_name]
                                new_line = self._apply_change_to_line(
                                    line, param_name, value, method
                                )
                                new_lines.append(new_line)
                                modified = True
                                break
                    if not modified:
                        new_lines.append(line)

                with open(bsn_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)

                self.logger.debug(f"Updated basin file: {bsn_file}")

            except Exception as e:
                self.logger.error(f"Error updating {bsn_file}: {e}")
                return False

        return True

    def _update_extension_files(self, ext: str, params: Dict[str, float]) -> bool:
        """
        Update all files with a given extension in TxtInOut.

        Handles .gw, .hru, .mgt, .sol files which may exist per sub-basin/HRU.

        Args:
            ext: File extension (e.g., '.gw', '.hru')
            params: Dictionary of parameter names to values

        Returns:
            True if successful
        """
        target_files = list(self.txtinout_dir.glob(f'*{ext}'))
        if not target_files:
            self.logger.warning(f"No {ext} files found in TxtInOut")
            return True  # Not an error, may just not exist

        success = True
        for target_file in target_files:
            try:
                with open(target_file, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    modified = False
                    for param_name, value in params.items():
                        if PARAM_FILE_MAP.get(param_name) != ext:
                            continue

                        # Check if this line contains the parameter
                        if f'| {param_name}' in line or f'|{param_name}' in line:
                            method = PARAM_CHANGE_METHOD.get(param_name, 'v__')
                            new_line = self._apply_change_to_line(
                                line, param_name, value, method
                            )
                            new_lines.append(new_line)
                            modified = True
                            break

                    if not modified:
                        new_lines.append(line)

                with open(target_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)

            except Exception as e:
                self.logger.error(f"Error updating {target_file}: {e}")
                success = False

        self.logger.debug(f"Updated {len(target_files)} {ext} files")
        return success

    def _apply_change_to_line(
        self,
        line: str,
        param_name: str,
        value: float,
        method: str
    ) -> str:
        """
        Apply a parameter change to a SWAT text file line.

        SWAT text file lines have the format:
            <value>    | PARAM_NAME : description

        Args:
            line: Original file line
            param_name: Parameter name
            value: New value or change amount
            method: Change method ('r__', 'v__', or 'a__')

        Returns:
            Modified line string
        """
        # Extract the current value from the line
        # SWAT format: leading whitespace + value + whitespace + | + param description
        match = re.match(r'^(\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(.*)', line)
        if not match:
            # Try integer format
            match = re.match(r'^(\s*)([-+]?\d+)(.*)', line)

        if not match:
            self.logger.warning(f"Could not parse value from line for {param_name}: {line.strip()}")
            return line

        prefix = match.group(1)
        original_str = match.group(2)
        suffix = match.group(3)

        try:
            original_value = float(original_str)
        except ValueError:
            self.logger.warning(f"Could not parse float from '{original_str}' for {param_name}")
            return line

        # Apply change method
        if method == 'r__':
            # Relative: new = original * (1 + value)
            new_value = original_value * (1.0 + value)
        elif method == 'v__':
            # Value replacement: new = value
            new_value = value
        elif method == 'a__':
            # Absolute: new = original + value
            new_value = original_value + value
        else:
            self.logger.warning(f"Unknown change method '{method}' for {param_name}")
            new_value = value

        # Format the new value to match original width
        # Determine if original was integer or float
        if '.' in original_str:
            # Float format - match decimal places
            decimal_places = len(original_str.split('.')[-1])
            new_value_str = f"{new_value:{len(original_str)}.{decimal_places}f}"
        else:
            # Integer format
            new_value_str = f"{int(round(new_value)):>{len(original_str)}d}"

        new_line = f"{prefix}{new_value_str}{suffix}"
        self.logger.debug(
            f"  {param_name} ({method}): {original_value:.4f} -> {new_value:.4f}"
        )
        return new_line

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from SWAT files or defaults."""
        try:
            if not self.txtinout_dir.exists():
                return self._get_default_initial_values()

            params = {}

            for param_name in self.swat_params:
                ext = PARAM_FILE_MAP.get(param_name)
                method = PARAM_CHANGE_METHOD.get(param_name, 'v__')

                if ext is None:
                    bounds = self.param_bounds.get(param_name, {'min': 0.0, 'max': 1.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2
                    continue

                # For relative parameters, initial change is 0.0 (no change)
                if method == 'r__':
                    params[param_name] = 0.0
                    continue

                # For value/absolute parameters, try to read from file
                value = self._read_param_from_file(param_name, ext)
                if value is not None:
                    params[param_name] = value
                else:
                    from ..parameters import DEFAULT_PARAMS
                    if param_name in DEFAULT_PARAMS:
                        params[param_name] = DEFAULT_PARAMS[param_name]
                    else:
                        bounds = self.param_bounds.get(param_name, {'min': 0.0, 'max': 1.0})
                        params[param_name] = (bounds['min'] + bounds['max']) / 2

            return params

        except Exception as e:
            self.logger.error(f"Error reading initial parameters: {e}")
            return self._get_default_initial_values()

    def _read_param_from_file(self, param_name: str, ext: str) -> Optional[float]:
        """
        Read a parameter value from a SWAT text file.

        Args:
            param_name: Parameter name to search for
            ext: File extension to search in

        Returns:
            Parameter value if found, None otherwise
        """
        if ext == '.bsn':
            target_files = list(self.txtinout_dir.glob('*.bsn'))
        else:
            target_files = list(self.txtinout_dir.glob(f'*{ext}'))

        if not target_files:
            return None

        # Read from the first file
        target_file = target_files[0]
        try:
            with open(target_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if f'| {param_name}' in line or f'|{param_name}' in line:
                        match = re.match(r'^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                        if match:
                            return float(match.group(1))
        except Exception:
            pass

        return None

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values."""
        from ..parameters import DEFAULT_PARAMS
        params = {}
        for param_name in self.swat_params:
            if param_name in DEFAULT_PARAMS:
                params[param_name] = DEFAULT_PARAMS[param_name]
            else:
                bounds = self.param_bounds.get(param_name, {'min': 0.0, 'max': 1.0})
                params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def copy_params_to_worker_dir(self, worker_txtinout_dir: Path) -> bool:
        """
        Copy TxtInOut files to a worker-specific directory for parallel calibration.

        Args:
            worker_txtinout_dir: Target directory for worker's TxtInOut files

        Returns:
            True if successful
        """
        try:
            worker_txtinout_dir.mkdir(parents=True, exist_ok=True)

            if self.txtinout_dir.exists():
                # Copy all SWAT input files
                for f in self.txtinout_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, worker_txtinout_dir / f.name)

            return True

        except Exception as e:
            self.logger.error(f"Error copying TxtInOut to {worker_txtinout_dir}: {e}")
            return False
