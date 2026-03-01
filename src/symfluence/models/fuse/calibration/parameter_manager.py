#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
FUSE Parameter Manager - FIXED for NetCDF Index Issues

This version fixes the "Index exceeds dimension bound" error by ensuring
proper parameter file structure and indexing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import netCDF4 as nc
import xarray as xr

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_fuse_bounds
from symfluence.optimization.registry import OptimizerRegistry

# Mapping of FUSE decisions to the parameters they require for meaningful calibration.
# If a decision is active but its required params are not being calibrated, the model
# structure effectively has uncalibrated degrees of freedom (poisoned pathways).
DECISION_REQUIRED_PARAMS = {
    # TOPMODEL surface runoff needs LOGLAMB (log of topographic index) and TISHAPE
    'tmdl_param': {'LOGLAMB', 'TISHAPE'},
    # Interflow requires IFLWRTE (interflow rate)
    'intflwsome': {'IFLWRTE'},
    # Gamma routing needs TIMEDELAY
    'rout_gamma': {'TIMEDELAY'},
    # Temperature index snow model needs MBASE, MFMAX, MFMIN, PXTEMP
    'temp_index': {'MBASE', 'MFMAX', 'MFMIN', 'PXTEMP'},
    # Percolation from field capacity to saturation needs PERCRTE, PERCEXP
    'perc_f2sat': {'PERCRTE', 'PERCEXP'},
    # Percolation from wilting point to saturation needs PERCRTE, PERCEXP
    'perc_w2sat': {'PERCRTE', 'PERCEXP'},
    # SAC-style lower-layer percolation needs PERCRTE
    'perc_lower': {'PERCRTE'},
}


@OptimizerRegistry.register_parameter_manager('FUSE')
class FUSEParameterManager(BaseParameterManager):
    """Handles FUSE parameter bounds, normalization, and file updates - FIXED VERSION"""

    def __init__(self, config: Dict, logger: logging.Logger, fuse_settings_dir: Path):
        # Initialize base class
        super().__init__(config, logger, fuse_settings_dir)

        # FUSE-specific setup
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default=None,
            dict_key='DOMAIN_NAME'
        )

        # Parse FUSE parameters to calibrate
        fuse_params_str = self._get_config_value(
            lambda: self.config.model.fuse.params_to_calibrate,
            default=None,
            dict_key='SETTINGS_FUSE_PARAMS_TO_CALIBRATE'
        )
        # Handle None, empty string, or 'default' as signal to use default parameter list
        if fuse_params_str is None or fuse_params_str == '' or fuse_params_str == 'default':
            # Provide sensible defaults if not specified
            self.logger.info("Using default FUSE calibration parameters.")
            fuse_params_str = 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE'

        self.fuse_params = [p.strip() for p in fuse_params_str.split(',') if p.strip()]

        # Path to FUSE parameter files
        self.data_dir = Path(self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.',
            dict_key='SYMFLUENCE_DATA_DIR'
        ))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.fuse_sim_dir = self.project_dir / 'simulations' / self.experiment_id / 'FUSE'
        self.fuse_setup_dir = self.project_dir / 'settings' / 'FUSE'
        self.fuse_id = self._get_config_value(lambda: self.config.model.fuse.file_id, default=self.experiment_id, dict_key='FUSE_FILE_ID')

        # Parameter file paths
        self.para_def_path = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_def.nc"
        self.para_sce_path = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_sce.nc"
        self.para_best_path = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_best.nc"

        # CRITICAL: Use para_sce.nc for calibration iterations, but ensure it's properly structured
        self.param_file_path = self.para_def_path

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS FROM BASE CLASS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return parameter/coefficient names depending on regionalization mode."""
        regionalization = self._get_config_value(
            lambda: self.config.model.fuse.parameter_regionalization,
            default='lumped',
            dict_key='PARAMETER_REGIONALIZATION'
        )
        use_tf = self._get_config_value(
            lambda: self.config.model.fuse.use_transfer_functions,
            default=False,
            dict_key='USE_TRANSFER_FUNCTIONS'
        )
        if use_tf:
            regionalization = 'transfer_function'

        if regionalization == 'transfer_function':
            coeff_bounds = self._get_config_value(
                lambda: self.config.model.fuse.transfer_function_coeff_bounds,
                default={},
                dict_key='TRANSFER_FUNCTION_COEFF_BOUNDS'
            )
            if coeff_bounds:
                return list(coeff_bounds.keys())

        return self.fuse_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return FUSE parameter bounds, preferring config values over registry defaults.

        Priority:
        1. TRANSFER_FUNCTION_COEFF_BOUNDS if regionalization enabled
        2. FUSE_PARAM_BOUNDS from config (user-specified)
        3. Registry defaults for any parameters not in config
        """
        # Get registry defaults first
        bounds = get_fuse_bounds()

        # Check for parameter regionalization mode
        regionalization = self._get_config_value(
            lambda: self.config.model.fuse.parameter_regionalization,
            default='lumped',
            dict_key='PARAMETER_REGIONALIZATION'
        )
        use_transfer_functions = self._get_config_value(
            lambda: self.config.model.fuse.use_transfer_functions,
            default=False,
            dict_key='USE_TRANSFER_FUNCTIONS'
        )

        if regionalization == 'transfer_function' or use_transfer_functions:
            # Use transfer function coefficient bounds instead of raw parameter bounds
            coeff_bounds = self._get_config_value(
                lambda: self.config.model.fuse.transfer_function_coeff_bounds,
                default=None,
                dict_key='TRANSFER_FUNCTION_COEFF_BOUNDS'
            )
            if coeff_bounds:
                self.logger.debug(
                    f"Using transfer function coefficient bounds "
                    f"({len(coeff_bounds)} coefficients)"
                )
                bounds = {}  # Replace all bounds with coefficient bounds
                for coeff_name, coeff_range in coeff_bounds.items():
                    if isinstance(coeff_range, (list, tuple)) and len(coeff_range) == 2:
                        bounds[coeff_name] = {
                            'min': float(coeff_range[0]),
                            'max': float(coeff_range[1])
                        }
                    elif isinstance(coeff_range, dict):
                        bounds[coeff_name] = {
                            'min': float(coeff_range.get('min', 0)),
                            'max': float(coeff_range.get('max', 1))
                        }
                return bounds
            else:
                self.logger.warning(
                    "PARAMETER_REGIONALIZATION=transfer_function but "
                    "TRANSFER_FUNCTION_COEFF_BOUNDS not configured"
                )

        # Check for user-specified bounds in config
        config_bounds = self._get_config_value(
            lambda: self.config.model.fuse.param_bounds,
            default=None,
            dict_key='FUSE_PARAM_BOUNDS'
        )
        if config_bounds:
            self.logger.info("Using FUSE_PARAM_BOUNDS from config (overriding registry defaults)")
            self._apply_config_bounds_override(bounds, config_bounds)

        return bounds

    def _get_default_fuse_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get central registry defaults for FUSE parameters."""
        return get_fuse_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        Update FUSE constraints file with new parameter values.

        FUSE's run_def mode regenerates para_def.nc from the constraints file,
        so we must modify the constraints file to change parameter values.

        When using parameter regionalization (transfer_function, zones, distributed),
        the worker handles applying the coefficients via the regionalization system.
        """
        # Check for parameter regionalization mode
        regionalization = self._get_config_value(
            lambda: self.config.model.fuse.parameter_regionalization,
            default='lumped',
            dict_key='PARAMETER_REGIONALIZATION'
        )
        use_transfer_functions = self._get_config_value(
            lambda: self.config.model.fuse.use_transfer_functions,
            default=False,
            dict_key='USE_TRANSFER_FUNCTIONS'
        )

        if regionalization != 'lumped' or use_transfer_functions:
            # In regionalization mode, the worker handles parameter application
            # via _apply_regionalization(). Skip constraints file update here.
            self.logger.debug(
                f"Regionalization mode ({regionalization}): "
                f"skipping constraints file update (handled by worker)"
            )
            return True

        return self._update_constraints_file(params)

    def _update_constraints_file(self, params: Dict[str, float]) -> bool:
        """Update the fuse_zConstraints_snow.txt file with new default values.

        FUSE uses Fortran fixed-width format: (L1,1X,I1,1X,3(F9.3,1X),...)
        The default value column starts at position 4 and is exactly 9 characters.
        """
        try:
            constraints_file = self.fuse_setup_dir / 'fuse_zConstraints_snow.txt'

            if not constraints_file.exists():
                self.logger.error(f"FUSE constraints file not found: {constraints_file}")
                return False

            # Read the constraints file with encoding fallback
            try:
                with open(constraints_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                self.logger.warning(
                    f"UTF-8 decode error reading {constraints_file}, falling back to latin-1"
                )
                with open(constraints_file, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            # Fortran format: (L1,1X,I1,1X,3(F9.3,1X),...)
            # Default value column: position 4-12 (9 chars, F9.3 format)
            DEFAULT_VALUE_START = 4
            DEFAULT_VALUE_WIDTH = 9

            updated_lines = []
            params_updated = set()

            for line in lines:
                # Skip header line (starts with '(') and comment lines
                stripped = line.strip()
                if stripped.startswith('(') or stripped.startswith('*') or stripped.startswith('!'):
                    updated_lines.append(line)
                    continue

                # Check if this line contains any of our parameters
                updated = False
                for param_name, value in params.items():
                    # Match exact parameter name (avoid partial matches)
                    parts = line.split()
                    if len(parts) >= 13 and param_name in parts:
                        # Format value to exactly 9 characters (F9.3 format)
                        new_value = f"{value:9.3f}"

                        # Replace the fixed-width column in the line
                        # Position 4-12 is the default value (9 characters)
                        if len(line) > DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:
                            new_line = (
                                line[:DEFAULT_VALUE_START] +
                                new_value +
                                line[DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:]
                            )
                            updated_lines.append(new_line)
                            params_updated.add(param_name)
                            updated = True
                            break

                if not updated:
                    updated_lines.append(line)

            # Write updated constraints file
            with open(constraints_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            if params_updated:
                self.logger.debug(f"Updated FUSE constraints: {params_updated}")

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating FUSE constraints file: {e}")
            return False

    # Note: get_initial_parameters() is already defined below and matches the signature
    # Note: Parameter bounds are now provided by the central ParameterBoundsRegistry

    def verify_and_fix_parameter_files(self) -> bool:
        """Verify parameter file structure and fix indexing issues"""
        try:
            self.logger.debug("Verifying FUSE parameter file structure...")

            # Check each parameter file
            for file_path in [self.para_def_path, self.para_sce_path, self.para_best_path]:
                if file_path.exists():
                    self.logger.debug(f"Checking {file_path.name}")

                    with xr.open_dataset(file_path) as ds:
                        #self.logger.debug(f"  Dimensions: {dict(ds.dims)}")
                        #self.logger.debug(f"  Parameters available: {list(ds.data_vars.keys())}")

                        # Check if 'par' dimension exists and has correct size
                        if 'par' in ds.dims:
                            par_size = ds.sizes['par']
                            self.logger.debug(f"  Parameter dimension size: {par_size}")

                            if par_size == 0:
                                self.logger.error(f"  ERROR: {file_path.name} has empty parameter dimension!")
                                return False

                            # Verify that we can access parameter set 0
                            try:
                                for param in self.fuse_params:
                                    if param in ds.variables:
                                        test_value = ds[param].isel(par=0).values
                                        self.logger.debug(f"  {param}[0] = {test_value}")
                                    else:
                                        self.logger.warning(f"  Parameter {param} not found in {file_path.name}")

                                self.logger.debug(f"  ✓ {file_path.name} parameter indexing OK")

                            except Exception as e:  # noqa: BLE001 — calibration resilience
                                self.logger.error(f"  ERROR: Cannot access parameter set 0 in {file_path.name}: {str(e)}")
                                # Try to fix the file
                                if self._fix_parameter_file_indexing(file_path):
                                    self.logger.info(f"  ✓ Fixed parameter indexing in {file_path.name}")
                                else:
                                    return False
                        else:
                            self.logger.error(f"  ERROR: {file_path.name} missing 'par' dimension!")
                            return False
                else:
                    self.logger.warning(f"Parameter file {file_path.name} does not exist")

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error verifying parameter files: {str(e)}")
            return False

    def _fix_parameter_file_indexing(self, file_path: Path) -> bool:
        """Fix parameter file indexing issues"""
        try:
            self.logger.info(f"Rebuilding parameter file: {file_path.name}")

            # Create backup
            backup_path = file_path.with_suffix('.nc.backup')
            if file_path.exists() and not backup_path.exists():
                import shutil
                shutil.copy2(file_path, backup_path)

            # Create new file with proper structure
            import netCDF4 as nc
            with nc.Dataset(file_path, 'w', format='NETCDF4') as ds:
                # Create dimensions - CRITICAL: Use size 1, not unlimited
                ds.createDimension('par', 1)

                # Create coordinate variable
                par_var = ds.createVariable('par', 'i4', ('par',))
                par_var[:] = [0]  # Set to 0-based indexing

                # Create parameter variables with default values
                for param_name in self.fuse_params:
                    param_var = ds.createVariable(param_name, 'f8', ('par',))
                    default_val = self._get_default_parameter_value(param_name)
                    param_var[:] = [default_val]

                # Ensure the file is properly closed and synced
                ds.sync()

            self.logger.info(f"Successfully rebuilt {file_path.name}")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to fix {file_path.name}: {str(e)}")
            return False

    def _get_default_parameter_value(self, param_name: str) -> float:
        """Get default value for a parameter"""
        if param_name in self.param_bounds:
            bounds = self.param_bounds[param_name]
            return (bounds['min'] + bounds['max']) / 2.0
        else:
            # Generic default for unknown parameters
            return 1.0

    # Note: all_param_names property and get_parameter_bounds() are inherited from BaseParameterManager

    def update_parameter_file(self, params: Dict[str, float], use_best_file: bool = False) -> bool:
        """Update FUSE parameter NetCDF file with new parameter values - FIXED VERSION"""
        try:
            # FIRST: Verify parameter files are properly structured
            #if not self.verify_and_fix_parameter_files():
            #    self.logger.error("Cannot proceed - parameter files have structural issues")
            #    return False

            # Choose which file to update
            target_file = self.param_file_path

            if not target_file.exists():
                self.logger.error(f"Parameter file does not exist: {target_file}")
                return False

            self.logger.debug(f"Updating parameter file: {target_file}")

            # SAFE NetCDF writing with proper error handling
            try:
                with nc.Dataset(target_file, 'r+') as ds:
                    # Verify the file structure first
                    if 'par' not in ds.dimensions:
                        self.logger.error(f"Missing 'par' dimension in {target_file}")
                        return False

                    par_size = ds.dimensions['par'].size
                    if par_size == 0:
                        self.logger.error(f"Empty 'par' dimension in {target_file}")
                        return False

                    changed = 0
                    for p, v in params.items():
                        if p in ds.variables:
                            try:
                                # Always use index 0 for single parameter set
                                before = float(ds.variables[p][0])
                                ds.variables[p][0] = float(v)
                                after = float(ds.variables[p][0])
                                self.logger.debug(f"[param write] {p}: {before} -> {after}")
                                changed += (abs(after - before) > 1e-10)
                            except Exception as e:  # noqa: BLE001 — calibration resilience
                                self.logger.error(f"Error updating parameter {p}: {str(e)}")
                                return False
                        else:
                            self.logger.error(f"[param write] {p} NOT FOUND in {target_file}")
                            return False

                    # Force sync to disk
                    ds.sync()

                if changed == 0:
                    self.logger.warning("[param write] No values changed - check parameter bounds or file structure")
                else:
                    self.logger.debug(f"Successfully updated {changed} FUSE parameters in {target_file}")


                return True

            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.error(f"NetCDF error updating {target_file}: {str(e)}")
                return False

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating parameter file: {str(e)}")
            return False

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from existing FUSE parameter file or coefficient bounds."""
        try:
            # Check for parameter regionalization mode
            regionalization = self._get_config_value(
                lambda: self.config.model.fuse.parameter_regionalization,
                default='lumped',
                dict_key='PARAMETER_REGIONALIZATION'
            )
            use_transfer_functions = self._get_config_value(
                lambda: self.config.model.fuse.use_transfer_functions,
                default=False,
                dict_key='USE_TRANSFER_FUNCTIONS'
            )

            if regionalization != 'lumped' or use_transfer_functions:
                # In regionalization mode, return initial guesses for coefficients
                # (not raw parameters from file, since we're calibrating coefficients)
                return self._get_default_initial_values()

            if not self.param_file_path.exists():
                self.logger.warning(f"FUSE parameter file not found: {self.param_file_path}")
                return self._get_default_initial_values()

            with xr.open_dataset(self.param_file_path) as ds:
                params = {}
                for param_name in self.fuse_params:
                    if param_name in ds.variables:
                        # Get the parameter value (assuming parameter set 0)
                        params[param_name] = float(ds[param_name].isel(par=0).values)
                    else:
                        self.logger.warning(f"Parameter {param_name} not found in file")
                        # Use default value from bounds
                        bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                        params[param_name] = (bounds['min'] + bounds['max']) / 2

                return params

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading initial parameters: {str(e)}")
            return self._get_default_initial_values()

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (or coefficient values in regionalization mode)."""
        params = {}
        bounds = self.param_bounds

        # In regionalization mode, param_bounds contains coefficient bounds
        # In lumped mode, it contains original parameter bounds
        for param_name, param_bounds_dict in bounds.items():
            # Use middle of bounds as default
            params[param_name] = (param_bounds_dict['min'] + param_bounds_dict['max']) / 2

        return params

    def validate_params_for_decisions(self, decisions_path: Path) -> List[str]:
        """
        Validate that calibrated parameters cover the active model decisions.

        Reads the FUSE decisions file, checks each active decision against
        DECISION_REQUIRED_PARAMS, and warns if required parameters are missing
        from the calibration set.

        Args:
            decisions_path: Path to the fuse_zDecisions_*.txt file

        Returns:
            List of warning messages (empty if all OK)
        """
        warnings_list: List[str] = []

        if not decisions_path.exists():
            self.logger.debug(f"Decisions file not found for validation: {decisions_path}")
            return warnings_list

        # Parse active decisions from the file
        active_decisions = {}
        try:
            with open(decisions_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Lines 2-10 (1-indexed) contain decisions: "value KEY ! comment"
            for line in lines[1:10]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    decision_value = parts[0]
                    decision_key = parts[1]
                    active_decisions[decision_key] = decision_value
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.debug(f"Could not parse decisions file: {e}")
            return warnings_list

        calibrated_params = set(self.fuse_params)

        for decision_key, decision_value in active_decisions.items():
            required = DECISION_REQUIRED_PARAMS.get(decision_value)
            if required:
                missing = required - calibrated_params
                if missing:
                    msg = (
                        f"Decision {decision_key}={decision_value} requires parameters "
                        f"{missing} but they are not being calibrated. "
                        f"This may cause poor model performance."
                    )
                    warnings_list.append(msg)
                    self.logger.warning(f"FUSE decision-param mismatch: {msg}")

            # Warn about no_snowmod for catchments with snow params configured
            if decision_value == 'no_snowmod':
                snow_params_in_calibration = calibrated_params & {
                    'MBASE', 'MFMAX', 'MFMIN', 'PXTEMP', 'LAPSE'
                }
                if snow_params_in_calibration:
                    msg = (
                        f"SNOWM=no_snowmod but snow parameters {snow_params_in_calibration} "
                        f"are being calibrated. This is likely wrong for a catchment with "
                        f"snow processes. Consider using SNOWM=temp_index."
                    )
                    warnings_list.append(msg)
                    self.logger.warning(f"FUSE snow mismatch: {msg}")

        return warnings_list
