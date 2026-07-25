# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shared runoff loading utilities for routing models.

Provides model-specific runoff file resolution, variable detection,
and time precision fixing used by mizuRoute, tRoute, and dRoute.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelRunoffConfig:
    """Configuration for model-specific runoff settings."""

    output_dir_key: str
    output_dir_name: str
    default_var: str
    default_units: str
    default_dt: str
    output_file_pattern: str
    hru_dim: str = 'gru'
    hru_var: str = 'gruId'
    comment_name: str = 'model'


MODEL_CONFIGS: Dict[str, ModelRunoffConfig] = {
    'summa': ModelRunoffConfig(
        output_dir_key='EXPERIMENT_OUTPUT_SUMMA',
        output_dir_name='SUMMA',
        default_var='averageRoutedRunoff',
        default_units='m/s',
        default_dt='3600',
        output_file_pattern='{experiment_id}_timestep.nc',
        hru_dim='hru',
        hru_var='hruId',
        comment_name='SUMMA',
    ),
    'fuse': ModelRunoffConfig(
        output_dir_key='EXPERIMENT_OUTPUT_FUSE',
        output_dir_name='FUSE',
        default_var='q_routed',
        default_units='m/s',
        default_dt='86400',
        output_file_pattern='{domain_name}_{experiment_id}_runs_def.nc',
        hru_dim='gru',
        hru_var='gruId',
        comment_name='FUSE',
    ),
    'gr': ModelRunoffConfig(
        output_dir_key='EXPERIMENT_OUTPUT_GR',
        output_dir_name='GR',
        default_var='q_routed',
        default_units='m/s',
        default_dt='86400',
        output_file_pattern='{domain_name}_{experiment_id}_runs_def.nc',
        hru_dim='gru',
        hru_var='gruId',
        comment_name='GR4J',
    ),
    'hype': ModelRunoffConfig(
        output_dir_key='EXPERIMENT_OUTPUT_HYPE',
        output_dir_name='HYPE',
        default_var='cout',
        default_units='m3/s',
        default_dt='86400',
        output_file_pattern='{experiment_id}_timestep.nc',
        hru_dim='gru',
        hru_var='gruId',
        comment_name='HYPE',
    ),
    'ngen': ModelRunoffConfig(
        output_dir_key='EXPERIMENT_OUTPUT_NGEN',
        output_dir_name='NGEN',
        default_var='runoff',
        default_units='m/s',
        default_dt='3600',
        output_file_pattern='{experiment_id}_runoff.nc',
        hru_dim='hru',
        hru_var='hruId',
        comment_name='NGEN',
    ),
}


def get_model_config(source_model: str) -> ModelRunoffConfig:
    """Get the runoff configuration for a source model."""
    key = source_model.strip().upper()
    # Normalize common variants
    if key in ('GR4J', 'GR5J', 'GR6J'):
        key = 'GR'
    return MODEL_CONFIGS.get(key.lower(), MODEL_CONFIGS['summa'])


def resolve_runoff_file(
    source_model: str,
    project_dir: Path,
    experiment_id: str,
    domain_name: str = '',
    config: Any = None,
) -> Optional[Path]:
    """
    Resolve the runoff output file path for a given source model.

    Handles FUSE's 6-char file ID truncation and model-specific
    output directory overrides from config.
    """
    model_cfg = get_model_config(source_model)
    model_key = source_model.strip().upper()

    # Resolve output directory
    output_dir: Optional[Path] = None
    if config is not None:
        try:
            config_dir_map = {
                'SUMMA': lambda: config.model.summa.experiment_output,
                'FUSE': lambda: config.model.fuse.experiment_output,
                'GR': lambda: config.model.gr.experiment_output,
                'HYPE': lambda: config.model.hype.experiment_output,
                'NGEN': lambda: config.model.ngen.experiment_output,
            }
            getter = config_dir_map.get(model_key)
            if getter:
                configured = getter()
                if configured and configured != 'default':
                    output_dir = Path(configured)
        except (AttributeError, TypeError):
            pass

    if output_dir is None:
        output_dir = project_dir / f"simulations/{experiment_id}" / model_cfg.output_dir_name

    # Resolve filename
    file_id = experiment_id
    if model_key == 'FUSE':
        if config is not None:
            try:
                fuse_file_id = config.model.fuse.file_id
                if fuse_file_id:
                    file_id = fuse_file_id
            except (AttributeError, TypeError):
                pass
        if len(file_id) > 6:
            import hashlib
            file_id = hashlib.md5(file_id.encode(), usedforsecurity=False).hexdigest()[:6]

    filename = model_cfg.output_file_pattern.format(
        experiment_id=file_id if model_key == 'FUSE' else experiment_id,
        domain_name=domain_name,
    )

    runoff_path = output_dir / filename

    if not runoff_path.exists():
        # Fallback: find any .nc file in the directory
        if output_dir.exists():
            nc_files = [f for f in output_dir.glob("*.nc") if '_para_' not in f.name]
            if nc_files:
                logger.info(f"Using fallback output file: {nc_files[0]}")
                return nc_files[0]
        logger.warning(f"Runoff file not found: {runoff_path}")
        return None

    return runoff_path


def detect_runoff_variable(ds: Any, source_model: str = 'SUMMA') -> Optional[str]:
    """
    Detect the runoff variable in a dataset based on source model.

    Tries model-specific variable first, then falls through common names.
    """
    model_cfg = get_model_config(source_model)

    # Model-specific variable first
    candidates = [model_cfg.default_var]

    # Common runoff variable names as fallback
    common_vars = [
        'averageRoutedRunoff', 'scalarTotalRunoff', 'q_routed',
        'runoff', 'q_runoff', 'cout', 'streamflow',
    ]
    for v in common_vars:
        if v not in candidates:
            candidates.append(v)

    for var in candidates:
        if var in ds.data_vars:
            return var

    # Last resort: any variable with 'runoff' in name
    for var in ds.data_vars:
        if 'runoff' in str(var).lower():
            return var

    return None


def fix_time_precision(
    runoff_filepath: Path,
    rounding_freq: str = 'h',
) -> Optional[Path]:
    """
    Fix model output time precision by rounding to configurable frequency.

    Detects time format (SUMMA seconds-since-1990, generic since, datetime64)
    and applies appropriate rounding. Writes result to a temp file then
    atomically replaces the original.

    Args:
        runoff_filepath: Path to the NetCDF runoff file
        rounding_freq: Frequency to round to ('h', 'min', 'none' to disable)

    Returns:
        Path to the (fixed) runoff file, or None on failure.
    """
    if rounding_freq == 'none':
        return runoff_filepath

    import os

    import pandas as pd
    import xarray as xr

    if not runoff_filepath.exists():
        logger.error(f"Runoff file not found: {runoff_filepath}")
        return None

    if runoff_filepath.stat().st_size < 1000:
        logger.error(f"Runoff file too small ({runoff_filepath.stat().st_size} bytes): {runoff_filepath}")
        return None

    try:
        ds = xr.open_dataset(runoff_filepath, decode_times=False)
    except (OSError, RuntimeError, ValueError) as nc_err:
        logger.error(f"Failed to open runoff file: {runoff_filepath} - {nc_err}")
        return None

    time_attrs = ds.time.attrs
    time_values = ds.time.values

    if len(time_values) == 0:
        ds.close()
        logger.error(f"Empty time dimension in: {runoff_filepath}")
        return None

    needs_fix = False
    time_format = None
    time_unit = 's'

    if 'units' in time_attrs:
        units_str = time_attrs['units'].lower()

        if 'since 1990-01-01' in units_str:
            time_format = 'summa_seconds_1990'
            first_time = pd.to_datetime(time_values[0], unit='s', origin='1990-01-01')
            needs_fix = (first_time != first_time.round(rounding_freq))

        elif 'since' in units_str:
            since_match = re.search(r'since\s+([0-9-]+(?:\s+[0-9:]+)?)', units_str)
            if since_match:
                ref_time_str = since_match.group(1).strip()
                time_format = f'generic_since_{ref_time_str}'

                if 'second' in units_str:
                    first_time = pd.to_datetime(time_values[0], unit='s', origin=ref_time_str)
                    time_unit = 's'
                elif 'hour' in units_str:
                    first_time = pd.Timestamp(ref_time_str) + pd.Timedelta(hours=float(time_values[0]))
                    time_unit = 'h'
                elif 'day' in units_str:
                    first_time = pd.to_datetime(time_values[0], unit='D', origin=ref_time_str)
                    time_unit = 'D'
                else:
                    first_time = pd.to_datetime(time_values[0], unit='s', origin=ref_time_str)
                    time_unit = 's'

                needs_fix = (first_time != first_time.round(rounding_freq))
        else:
            time_format = _try_decode_datetime64(runoff_filepath, rounding_freq)
            if time_format == 'datetime64':
                needs_fix = True
            else:
                ds.close()
                return runoff_filepath
    else:
        time_format = _try_decode_datetime64(runoff_filepath, rounding_freq)
        if time_format != 'datetime64':
            ds.close()
            return runoff_filepath
        needs_fix = True

    if not needs_fix:
        ds.close()
        return runoff_filepath

    # Apply fix based on format
    if time_format == 'summa_seconds_1990':
        time_stamps = pd.to_datetime(time_values, unit='s', origin='1990-01-01')
        rounded_stamps = time_stamps.round(rounding_freq)
        reference = pd.Timestamp('1990-01-01')
        rounded_seconds = (rounded_stamps - reference).total_seconds().values

        ds = ds.assign_coords(time=rounded_seconds)
        ds.time.attrs.clear()
        ds.time.attrs['units'] = 'seconds since 1990-01-01 00:00:00'
        ds.time.attrs['calendar'] = 'standard'
        ds.time.attrs['long_name'] = 'time'

    elif time_format and time_format.startswith('generic_since_'):
        ref_time_str = time_format.split('generic_since_')[1]
        reference = pd.Timestamp(ref_time_str)

        if time_unit == 's':
            time_stamps = reference + pd.to_timedelta(time_values, unit='s')
        elif time_unit == 'h':
            time_stamps = reference + pd.to_timedelta(time_values, unit='h')
        elif time_unit == 'D':
            time_stamps = reference + pd.to_timedelta(time_values, unit='D')
        else:
            time_stamps = reference + pd.to_timedelta(time_values, unit='s')

        rounded_stamps = time_stamps.round(rounding_freq)

        if time_unit == 's':
            rounded_values = (rounded_stamps - reference).total_seconds().values
        elif time_unit == 'h':
            rounded_values = (rounded_stamps - reference) / pd.Timedelta(hours=1)
        elif time_unit == 'D':
            rounded_values = (rounded_stamps - reference) / pd.Timedelta(days=1)
        else:
            rounded_values = (rounded_stamps - reference).total_seconds().values

        ds = ds.assign_coords(time=rounded_values)
        ds.time.attrs.clear()
        _unit_names = {'s': 'seconds', 'h': 'hours', 'D': 'days'}
        full_unit_name = _unit_names.get(time_unit, time_unit)
        normalized_ref = reference.strftime('%Y-%m-%d %H:%M:%S')
        ds.time.attrs['units'] = f"{full_unit_name} since {normalized_ref}"
        ds.time.attrs['calendar'] = 'standard'
        ds.time.attrs['long_name'] = 'time'

    elif time_format == 'datetime64':
        ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
        time_stamps = pd.to_datetime(ds_decoded.time.values)
        rounded_stamps = time_stamps.round(rounding_freq)
        ds = ds_decoded.assign_coords(time=rounded_stamps)
        ds_decoded.close()

    # Save atomically
    ds.load()
    temp_filepath = runoff_filepath.with_suffix('.tmp.nc')
    ds.to_netcdf(temp_filepath, format='NETCDF4')
    ds.close()

    if os.name != 'nt':
        os.chmod(temp_filepath, 0o664)  # nosec B103
    temp_filepath.replace(runoff_filepath)

    logger.info(f"Fixed time precision in {runoff_filepath}")
    return runoff_filepath


def _try_decode_datetime64(filepath: Path, rounding_freq: str) -> Optional[str]:
    """Try to open file with decoded times to check if rounding needed."""
    import pandas as pd
    import xarray as xr

    try:
        ds_decoded = xr.open_dataset(filepath, decode_times=True)
        first_time = pd.Timestamp(ds_decoded.time.values[0])
        needs_fix = (first_time != first_time.round(rounding_freq))
        ds_decoded.close()
        if needs_fix:
            return 'datetime64'
        return None
    except (ValueError, TypeError, KeyError):
        return None
