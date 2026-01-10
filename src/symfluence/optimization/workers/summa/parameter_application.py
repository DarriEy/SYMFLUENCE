#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Application for SUMMA Workers

This module contains functions for applying optimization parameters
to SUMMA model configuration files.
"""

import os
import re
import time
import random
from pathlib import Path
from typing import Dict

import numpy as np
import netCDF4 as nc
import xarray as xr


def _apply_parameters_worker(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Apply parameters consistently with sequential approach"""
    try:
        config = task_data['config']
        logger.debug(f"Applying parameters: {list(params.keys())} (consistent method)")

        # Parse parameter lists EXACTLY as ParameterManager does
        local_params = [p.strip() for p in (config.get('PARAMS_TO_CALIBRATE') or '').split(',') if p.strip()]
        basin_params = [p.strip() for p in (config.get('BASIN_PARAMS_TO_CALIBRATE') or '').split(',') if p.strip()]
        depth_params = ['total_mult', 'shape_factor'] if config.get('CALIBRATE_DEPTH', False) else []

        # Add support for new multiplier
        if 'total_soil_depth_multiplier' in params:
            if 'total_soil_depth_multiplier' not in depth_params:
                depth_params.append('total_soil_depth_multiplier')

        mizuroute_params = []

        if config.get('CALIBRATE_MIZUROUTE', False):
            mizuroute_params_str = config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', 'velo,diff')
            mizuroute_params = [p.strip() for p in mizuroute_params_str.split(',') if p.strip()]

        # 1. Handle soil depth parameters
        has_depth_params = any(p in params for p in ['total_soil_depth_multiplier', 'total_mult', 'shape_factor'])
        if has_depth_params:
            logger.debug("Updating soil depths (consistent)")
            if not _update_soil_depths_worker(params, task_data, settings_dir, logger, debug_info):
                return False

        # 2. Handle mizuRoute parameters
        if mizuroute_params and any(p in params for p in mizuroute_params):
            logger.debug("Updating mizuRoute parameters (consistent)")
            if not _update_mizuroute_params_worker(params, task_data, logger, debug_info):
                return False

        # 3. Generate trial parameters file (same exclusion logic as ParameterManager)
        hydrological_params = {k: v for k, v in params.items()
                          if k not in depth_params + mizuroute_params}

        if hydrological_params:
            logger.debug(f"Generating trial parameters file with: {list(hydrological_params.keys())} (consistent)")
            if not _generate_trial_params_worker(hydrological_params, settings_dir, logger, debug_info):
                return False

        logger.debug("Parameter application completed successfully (consistent)")
        return True

    except Exception as e:
        error_msg = f"Error applying parameters (consistent): {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _update_soil_depths_worker(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced soil depth update with better error handling"""
    try:
        original_depths_list = task_data.get('original_depths')
        if not original_depths_list:
            logger.debug("No original depths provided, skipping soil depth update")
            return True

        original_depths = np.array(original_depths_list)

        # Support both new and legacy parameter names
        total_mult = params.get('total_soil_depth_multiplier')
        if total_mult is None:
            total_mult = params.get('total_mult', 1.0)

        shape_factor = params.get('shape_factor', 1.0)

        if isinstance(total_mult, np.ndarray): total_mult = total_mult[0]
        if isinstance(shape_factor, np.ndarray): shape_factor = shape_factor[0]

        logger.debug(f"Updating soil depths: total_mult={total_mult:.3f}, shape_factor={shape_factor:.3f}")

        # Calculate new depths
        arr = original_depths.copy()
        n = len(arr)
        idx = np.arange(n)

        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)

        w /= w.mean()
        new_depths = arr * w * total_mult

        # Calculate heights
        heights = np.zeros(len(new_depths) + 1)
        for i in range(len(new_depths)):
            heights[i + 1] = heights[i] + new_depths[i]

        # Update coldState.nc
        coldstate_path = settings_dir / 'coldState.nc'
        if not coldstate_path.exists():
            error_msg = f"coldState.nc not found: {coldstate_path}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        debug_info['files_checked'].append(f"coldState.nc: {coldstate_path}")

        with nc.Dataset(coldstate_path, 'r+') as ds:
            if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                error_msg = "Required depth variables not found in coldState.nc"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return False

            num_hrus = ds.dimensions['hru'].size
            for h in range(num_hrus):
                ds.variables['mLayerDepth'][:, h] = new_depths
                ds.variables['iLayerHeight'][:, h] = heights

        logger.debug("Soil depths updated successfully")
        return True

    except Exception as e:
        error_msg = f"Error updating soil depths: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _update_mizuroute_params_worker(params: Dict, task_data: Dict, logger, debug_info: Dict) -> bool:
    """Enhanced mizuRoute parameter update with better error handling"""
    try:
        config = task_data['config']
        mizuroute_params = [p.strip() for p in config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]

        mizuroute_settings_dir = Path(task_data['mizuroute_settings_dir'])
        param_file = mizuroute_settings_dir / "param.nml.default"

        if not param_file.exists():
            logger.warning(f"mizuRoute param file not found: {param_file}")
            return True

        debug_info['files_checked'].append(f"mizuRoute param file: {param_file}")

        with open(param_file, 'r') as f:
            content = f.read()

        updated_content = content
        for param_name in mizuroute_params:
            if param_name in params:
                param_value = params[param_name]
                pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'

                if param_name in ['tscale']:
                    replacement = rf'\g<1>{int(param_value)}'
                else:
                    replacement = rf'\g<1>{param_value:.6f}'

                updated_content = re.sub(pattern, replacement, updated_content)
                logger.debug(f"Updated {param_name} = {param_value}")

        with open(param_file, 'w') as f:
            f.write(updated_content)

        logger.debug("mizuRoute parameters updated successfully")
        return True

    except Exception as e:
        error_msg = f"Error updating mizuRoute params: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _generate_trial_params_worker(params: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced trial parameters generation with better error handling and file locking"""
    try:
        if not params:
            logger.debug("No hydrological parameters to write")
            return True

        trial_params_path = settings_dir / 'trialParams.nc'
        attr_file_path = settings_dir / 'attributes.nc'

        if not attr_file_path.exists():
            error_msg = f"Attributes file not found: {attr_file_path}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        debug_info['files_checked'].append(f"attributes.nc: {attr_file_path}")

        # Add retry logic with file locking
        max_retries = 5
        base_delay = 0.1

        for attempt in range(max_retries):
            try:
                # Create temporary file first, then move it
                temp_path = trial_params_path.with_suffix(f'.tmp_{os.getpid()}_{random.randint(1000,9999)}')

                logger.debug(f"Attempt {attempt + 1}: Writing trial parameters to {temp_path}")

                # Define parameter levels
                routing_params = ['routingGammaShape', 'routingGammaScale']
                basin_params = ['basin__aquiferBaseflowExp', 'basin__aquiferScaleFactor', 'basin__aquiferHydCond']
                gru_level_params = routing_params + basin_params

                with xr.open_dataset(attr_file_path) as ds:
                    num_hrus = ds.sizes.get('hru', 1)
                    num_grus = ds.sizes.get('gru', 1)
                    hru_ids = ds['hruId'].values if 'hruId' in ds else np.arange(1, num_hrus + 1)
                    gru_ids = ds['gruId'].values if 'gruId' in ds else np.array([1])

                logger.debug(f"Writing parameters for {num_hrus} HRUs, {num_grus} GRUs")

                # Write to temporary file with exclusive access
                with nc.Dataset(temp_path, 'w', format='NETCDF4') as output_ds:
                    # Create dimensions
                    output_ds.createDimension('hru', num_hrus)
                    output_ds.createDimension('gru', num_grus)

                    # Create coordinate variables
                    hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
                    hru_var[:] = hru_ids

                    gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
                    gru_var[:] = gru_ids

                    # Add parameters
                    for param_name, param_values in params.items():
                        param_values_array = np.asarray(param_values)

                        if param_values_array.ndim > 1:
                            param_values_array = param_values_array.flatten()

                        if param_name in gru_level_params:
                            # GRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"

                            if len(param_values_array) >= num_grus:
                                param_var[:] = param_values_array[:num_grus]
                            else:
                                param_var[:] = param_values_array[0]
                        else:
                            # HRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"

                            if len(param_values_array) == num_hrus:
                                param_var[:] = param_values_array
                            elif len(param_values_array) == 1:
                                param_var[:] = param_values_array[0]
                            else:
                                param_var[:] = param_values_array[:num_hrus]

                        logger.debug(f"Added parameter {param_name} with shape {param_var.shape}")

                # Atomically move temporary file to final location
                try:
                    os.chmod(temp_path, 0o664)  # Set appropriate permissions
                    temp_path.rename(trial_params_path)
                    logger.debug(f"Trial parameters file created successfully: {trial_params_path}")
                    debug_info['files_checked'].append(f"trialParams.nc (created): {trial_params_path}")
                    return True
                except Exception as move_error:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise move_error

            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")

                # Clean up temp file if it exists
                if 'temp_path' in locals() and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                else:
                    error_msg = f"Failed to generate trial params after {max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    debug_info['errors'].append(error_msg)
                    return False

        return False

    except Exception as e:
        error_msg = f"Error generating trial params: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False
