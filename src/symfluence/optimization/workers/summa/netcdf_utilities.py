#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NetCDF Utilities for SUMMA Workers

This module contains functions for handling NetCDF time precision issues
and converting SUMMA output formats for mizuRoute compatibility.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

# Module-level logger for standalone function calls
_logger = logging.getLogger(__name__)


def fix_summa_time_precision_inplace(input_file: Path, logger=None) -> None:
    """Fix time precision in-place without temp files"""
    import netCDF4 as nc4

    try:
        # Open in read-write mode (not read-only)
        with nc4.Dataset(input_file, 'r+') as ds:
            if 'time' not in ds.variables:
                return

            # Read time values
            time_var = ds.variables['time']
            time_values = time_var[:]

            # Parse and round timestamps
            units_str = time_var.units
            calendar = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'
            timestamps = pd.to_datetime(
                nc4.num2date(time_values, units=units_str, calendar=calendar)
            )
            rounded_timestamps = timestamps.round('h')
            if rounded_timestamps.tz is not None:
                rounded_timestamps = rounded_timestamps.tz_localize(None)

            # Convert back to hours since reference
            ref_time_calc = pd.Timestamp('1990-01-01')
            rounded_hours = (rounded_timestamps - ref_time_calc).total_seconds() / 3600.0

            # Update in-place
            time_var[:] = rounded_hours
            time_var.units = 'hours since 1990-01-01 00:00:00'
            time_var.calendar = calendar
            time_var.long_name = 'time'

            # File is automatically closed and changes are saved

        if logger:
            logger.info(f"Fixed time precision in-place: {input_file.name}")

    except (OSError, RuntimeError, KeyError, ValueError) as e:
        if logger:
            logger.error(f"Error fixing time precision in-place: {e}")
        # Fall back to original method if in-place modification fails
        fix_summa_time_precision(input_file, None)


def fix_summa_time_precision(input_file, output_file=None, logger: Optional[logging.Logger] = None):
    """
    Round SUMMA time dimension to nearest hour to fix mizuRoute compatibility
    Fixed to handle timezone mismatch issues
    """
    log = logger or _logger
    log.debug(f"Opening {input_file}")

    try:
        # Open without decoding times to avoid conflicts
        ds = xr.open_dataset(input_file, decode_times=False)

        log.debug(f"Original time range: {ds.time.min().values} to {ds.time.max().values}")

        # Convert to datetime, round, then convert back
        time_vals = ds.time.values

        # First convert the time values to actual timestamps
        if 'units' in ds.time.attrs:
            time_units = ds.time.attrs['units']
            log.debug(f"Original time units: {time_units}")

            # Parse the reference time
            if 'since' in time_units:
                ref_time_str = time_units.split('since')[1].strip()
                ref_time = pd.Timestamp(ref_time_str)

                # Get the time unit (hours, days, seconds, etc.)
                unit = time_units.split()[0].lower()

                # Convert to timedelta and add to reference
                if unit.startswith('hour'):
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='h')
                elif unit.startswith('day'):
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='D')
                elif unit.startswith('second'):
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='s')
                elif unit.startswith('minute'):
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='min')
                else:
                    # Default to hours
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='h')
            else:
                # Fallback: assume hourly from a standard reference
                ref_time = pd.Timestamp('1990-01-01')
                timestamps = ref_time + pd.to_timedelta(time_vals, unit='h')
        else:
            # No units attribute, try to interpret as hours since 1990
            ref_time = pd.Timestamp('1990-01-01')
            timestamps = ref_time + pd.to_timedelta(time_vals, unit='h')

        # Round to nearest hour
        rounded_timestamps = timestamps.round('h')
        log.debug(f"Rounded time range: {rounded_timestamps.min()} to {rounded_timestamps.max()}")

        # FIX: Ensure both timestamps are timezone-naive for consistent calculation
        ref_time_calc = pd.Timestamp('1990-01-01')

        # Remove timezone from rounded_timestamps if present
        if rounded_timestamps.tz is not None:
            rounded_timestamps = rounded_timestamps.tz_localize(None)
            log.debug("Removed timezone from rounded timestamps")

        # Convert back to hours since reference time
        rounded_hours = (rounded_timestamps - ref_time_calc).total_seconds() / 3600.0

        # Create new time coordinate with cleared attributes
        new_time = xr.DataArray(
            rounded_hours,
            dims=('time',),
            attrs={}  # Start with empty attributes
        )

        # Preserve original calendar attribute
        orig_calendar = ds.time.attrs.get('calendar', 'standard')

        # Set clean attributes
        new_time.attrs['units'] = 'hours since 1990-01-01 00:00:00'
        new_time.attrs['calendar'] = orig_calendar
        new_time.attrs['long_name'] = 'time'

        # Replace time coordinate
        ds = ds.assign_coords(time=new_time)

        # Clean up encoding to avoid conflicts
        if 'time' in ds.encoding:
            del ds.encoding['time']

        # Load data into memory and close original
        ds.load()
        original_ds = ds
        ds = ds.copy()  # Create a clean copy
        original_ds.close()

        # Determine output path
        output_path = output_file if output_file else input_file
        log.debug(f"Saving to {output_path}")

        # Write to temporary file first, then move to final location
        temp_file = None
        try:
            # Create temporary file in same directory
            temp_dir = os.path.dirname(output_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=temp_dir) as tmp:
                temp_file = tmp.name

            # Save to temporary file with clean encoding
            ds.to_netcdf(temp_file, format='NETCDF4')
            ds.close()

            # Make output file writable if overwriting
            if output_file is None and os.path.exists(input_file):
                if os.name != 'nt':
                    os.chmod(input_file, 0o664)  # nosec B103 - Group-writable for HPC shared access

            # Atomically move to final location
            shutil.move(temp_file, output_path)
            temp_file = None  # Successfully moved

            log.debug("Time precision fix completed successfully")

        except (OSError, RuntimeError, KeyError, ValueError) as e:
            # Clean up temp file if it exists
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError as cleanup_err:
                    log.warning(f"Could not remove temp file {temp_file}: {cleanup_err}")
            raise e

    except (OSError, RuntimeError, KeyError, ValueError) as e:
        log.error(f"Error fixing time precision: {e}")
        raise


def _convert_lumped_to_distributed_worker(task_data: Dict, summa_dir: Path, logger, debug_info: Dict) -> bool:
    """Convert lumped SUMMA output for distributed routing"""
    try:
        # Find SUMMA timestep file
        timestep_files = list(summa_dir.glob("*timestep.nc"))
        if not timestep_files:
            logger.error("No SUMMA timestep files found for conversion")
            return False

        summa_file = timestep_files[0]
        logger.info(f"Converting SUMMA file: {summa_file}")

        # Load topology to get HRU information
        mizuroute_settings_dir = Path(task_data['mizuroute_settings_dir'])
        topology_file = mizuroute_settings_dir / task_data['config'].get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')

        if not topology_file.exists():
            logger.error(f"Topology file not found: {topology_file}")
            return False

        with xr.open_dataset(topology_file) as topo_ds:
            # Get HRU information - use first HRU ID as lumped GRU ID
            hru_ids = topo_ds['hruId'].values
            n_hrus = len(hru_ids)
            lumped_gru_id = 1  # Use ID=1 for consistency

            logger.info(f"Creating single lumped GRU (ID={lumped_gru_id}) for {n_hrus} HRUs in topology")

        # Ensure the original file is writable
        try:
            if os.name != 'nt':
                os.chmod(summa_file, 0o664)  # nosec B103 - Group-writable for HPC shared access
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            logger.warning(f"Could not change file permissions: {str(e)}")

        # Load and convert SUMMA output
        summa_ds = None
        try:
            # Open without decoding times to avoid conversion issues
            summa_ds = xr.open_dataset(summa_file, decode_times=False)

            # Handle the case where config has 'default' as value - use model-specific default
            routing_var_config = task_data['config'].get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
            if routing_var_config in ('default', None, ''):
                routing_var = 'averageRoutedRunoff'  # SUMMA default for routing
            else:
                routing_var = routing_var_config
            available_vars = list(summa_ds.variables.keys())

            # Find the best variable to use
            source_var = None
            if routing_var in summa_ds:
                source_var = routing_var
                logger.info(f"Using configured routing variable: {routing_var}")
            else:
                # Try fallback variables
                fallback_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
                for var in fallback_vars:
                    if var in summa_ds:
                        source_var = var
                        logger.info(f"Routing variable {routing_var} not found, using: {source_var}")
                        break

            if source_var is None:
                logger.error(f"No suitable routing variable found in {available_vars}")
                return False

            # Create mizuRoute forcing dataset
            mizuForcing = xr.Dataset()

            # Copy time coordinate (preserve original format)
            original_time = summa_ds['time']
            mizuForcing['time'] = xr.DataArray(
                original_time.values,
                dims=('time',),
                attrs=dict(original_time.attrs)
            )

            # Clean up time units if needed
            if 'units' in mizuForcing['time'].attrs:
                time_units = mizuForcing['time'].attrs['units']
                if 'T' in time_units:
                    mizuForcing['time'].attrs['units'] = time_units.replace('T', ' ')

            # Create single GRU using lumped GRU ID
            mizuForcing['gru'] = xr.DataArray([lumped_gru_id], dims=('gru',))
            mizuForcing['gruId'] = xr.DataArray([lumped_gru_id], dims=('gru',))

            # Extract runoff data
            var_data = summa_ds[source_var]
            runoff_data = var_data.values

            # Handle different shapes
            if len(runoff_data.shape) == 2:
                if runoff_data.shape[1] > 1:
                    runoff_data = runoff_data.mean(axis=1)
                    logger.info(f"Used mean across {var_data.shape[1]} spatial elements")
                else:
                    runoff_data = runoff_data[:, 0]
            else:
                runoff_data = runoff_data.flatten()

            # Keep as single GRU: (time,) -> (time, 1)
            single_gru_data = runoff_data[:, np.newaxis]

            # Create runoff variable
            mizuForcing[routing_var] = xr.DataArray(
                single_gru_data, dims=('time', 'gru'),
                attrs={'long_name': 'Lumped runoff for distributed routing', 'units': 'm/s'}
            )

            # Copy global attributes
            mizuForcing.attrs.update(summa_ds.attrs)

            # Load data and close original
            mizuForcing.load()
            summa_ds.close()
            summa_ds = None

        except (OSError, RuntimeError, KeyError, ValueError) as e:
            if summa_ds is not None:
                summa_ds.close()
            raise e

        # Write to temporary file first, then atomically move
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.nc',
                dir=summa_dir,
                prefix='temp_mizu_'
            ) as tmp:
                temp_file = Path(tmp.name)

            # Save to temporary file
            mizuForcing.to_netcdf(temp_file, format='NETCDF4')
            mizuForcing.close()

            # Set permissions and move
            if os.name != 'nt':
                os.chmod(temp_file, 0o664)  # nosec B103 - Group-writable for HPC shared access
            shutil.move(str(temp_file), str(summa_file))
            temp_file = None

            logger.info("Successfully converted SUMMA file: single lumped GRU for distributed routing")

            # CRITICAL: Now fix time precision for mizuRoute compatibility
            fix_summa_time_precision(summa_file)
            logger.info("Fixed SUMMA time precision for mizuRoute compatibility")

            return True

        except (OSError, RuntimeError, KeyError, ValueError) as e:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError as cleanup_err:
                    logger.warning(f"Could not remove temp file {temp_file}: {cleanup_err}")
            raise e

    except (OSError, RuntimeError, KeyError, ValueError) as e:
        logger.error(f"Conversion failed: {str(e)}")
        debug_info['errors'].append(f"Lumped-to-distributed conversion error: {str(e)}")
        return False
