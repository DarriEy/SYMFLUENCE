"""
FUSE Parameter Application

Standalone functions for applying calibration parameters to FUSE files:
- para_def.nc (NetCDF parameter definition file)
- fuse_zConstraints_snow.txt (Fortran fixed-width constraints file)
- Regionalized parameters via transfer functions
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


def update_para_def_nc(
    para_def_path: Path,
    params: Dict[str, float],
    log: Optional[logging.Logger] = None
) -> Set[str]:
    """
    Update FUSE para_def.nc file with new parameter values.

    Args:
        para_def_path: Path to para_def.nc file
        params: Parameter values to apply
        log: Logger instance

    Returns:
        Set of parameter names that were updated
    """
    import netCDF4 as nc

    log = log or logger
    params_updated: set = set()

    try:
        with nc.Dataset(para_def_path, 'r+') as ds:
            if 'par' not in ds.dimensions:
                log.error(f"Missing 'par' dimension in {para_def_path}")
                return params_updated

            par_size = ds.dimensions['par'].size
            if par_size == 0:
                log.error(f"Empty 'par' dimension in {para_def_path}")
                return params_updated

            for param_name, value in params.items():
                if param_name in ds.variables:
                    try:
                        before = float(ds.variables[param_name][0])
                        ds.variables[param_name][0] = float(value)
                        after = float(ds.variables[param_name][0])
                        log.debug(f"  NC: {param_name}: {before:.4f} -> {after:.4f}")
                        params_updated.add(param_name)
                    except (IndexError, ValueError, TypeError) as e:
                        log.warning(f"Error updating {param_name} in NetCDF: {e}")
                else:
                    log.debug(f"  NC: {param_name} not in file (may be structure param)")

            ds.sync()

    except (OSError, IOError) as e:
        log.error(f"I/O error updating {para_def_path}: {e}")
    except (KeyError, ValueError) as e:
        log.error(f"Data error updating {para_def_path}: {e}")

    # Verify write succeeded
    if params_updated:
        _verify_para_def_write(para_def_path, params, params_updated, log)

    return params_updated


def _verify_para_def_write(
    para_def_path: Path,
    params: Dict[str, float],
    params_updated: Set[str],
    log: logging.Logger
) -> None:
    """Verify that para_def.nc write succeeded by reading back a value."""
    import netCDF4 as nc

    try:
        with nc.Dataset(para_def_path, 'r') as ds:
            first_param = next(iter(params_updated))
            if first_param in ds.variables:
                actual_value = float(ds.variables[first_param][0])
                expected_value = params[first_param]
                # Tolerance of 1e-3 to match FUSE's Fortran F9.3 format
                if abs(actual_value - expected_value) > 1e-3:
                    log.warning(
                        f"Parameter write verification: {first_param} expected {expected_value:.6f} "
                        f"but file contains {actual_value:.6f} (diff={abs(actual_value - expected_value):.2e})"
                    )
    except Exception as e:
        log.debug(f"Could not verify para_def.nc write: {e}")


def apply_regionalization(
    para_def_path: Path,
    calibration_params: Dict[str, float],
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None
) -> Set[str]:
    """
    Apply parameter regionalization to generate spatially distributed parameters.

    Supports multiple regionalization methods:
    - lumped: Uniform parameters across all subcatchments
    - transfer_function: Power-law functions based on catchment attributes
    - zones: Shared parameters within predefined zones
    - distributed: Independent parameters per subcatchment

    Args:
        para_def_path: Path to para_def.nc file
        calibration_params: Calibration parameter/coefficient values
        config: Configuration dictionary
        log: Logger instance

    Returns:
        Set of parameter names that were updated
    """
    import netCDF4 as nc
    import pandas as pd
    from symfluence.models.fuse.calibration.parameter_regionalization import (
        RegionalizationFactory
    )

    log = log or logger
    params_updated: set = set()

    try:
        # Get regionalization method
        method = config.get('PARAMETER_REGIONALIZATION', 'lumped')
        if config.get('USE_TRANSFER_FUNCTIONS', False) and method == 'lumped':
            method = 'transfer_function'

        # Get original parameter bounds
        param_bounds = config.get('FUSE_PARAM_BOUNDS', {})
        if not param_bounds:
            log.error("FUSE_PARAM_BOUNDS not configured")
            return params_updated

        param_bounds_tuples = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in param_bounds.items()
        }

        # Load attributes if needed
        attributes = None
        if method == 'transfer_function':
            attributes_path = config.get('TRANSFER_FUNCTION_ATTRIBUTES')
            if attributes_path:
                attributes = pd.read_csv(attributes_path)
            else:
                log.error("TRANSFER_FUNCTION_ATTRIBUTES not configured")
                return params_updated

        # Determine number of subcatchments
        if attributes is not None:
            n_subcatchments = len(attributes)
        else:
            with nc.Dataset(para_def_path, 'r') as ds:
                n_subcatchments = ds.dimensions['par'].size

        # Create regionalization strategy
        regionalization = RegionalizationFactory.create(
            method=method,
            param_bounds=param_bounds_tuples,
            n_subcatchments=n_subcatchments,
            config=config,
            attributes=attributes,
            logger=log
        )

        log.debug(f"Using '{regionalization.name}' parameter regionalization")

        # Convert calibration parameters to distributed values
        param_array, param_names = regionalization.to_distributed(calibration_params)

        log.debug(
            f"Regionalization: {len(param_names)} params x {n_subcatchments} subcatchments"
        )

        # Read existing file structure for potential resize
        _resize_para_def_if_needed(para_def_path, n_subcatchments, log)

        # Write the distributed values
        with nc.Dataset(para_def_path, 'r+') as ds:
            for i, param_name in enumerate(param_names):
                if param_name not in ds.variables:
                    continue

                values = param_array[:, i]
                ds.variables[param_name][:] = values
                log.debug(
                    f"  {param_name}: distributed [{values.min():.3f}, {values.max():.3f}]"
                )
                params_updated.add(param_name)

            # Ensure numerical solver settings are reasonable
            _enforce_numerix_defaults(ds, log)
            ds.sync()

        if params_updated:
            log.debug(
                f"Applied regionalization to {len(params_updated)} parameters: "
                f"{', '.join(sorted(params_updated))}"
            )

    except Exception as e:
        log.error(f"Error applying transfer functions: {e}")
        import traceback
        log.error(traceback.format_exc())

    return params_updated


def _resize_para_def_if_needed(
    para_def_path: Path,
    n_subcatchments: int,
    log: logging.Logger
) -> None:
    """Resize para_def.nc if par dimension doesn't match n_subcatchments."""
    import netCDF4 as nc
    import shutil

    with nc.Dataset(para_def_path, 'r') as ds:
        par_size = ds.dimensions['par'].size
        if par_size == n_subcatchments:
            return

        # Read existing data
        existing_vars = {}
        for vname in ds.variables:
            if vname == 'par':
                continue
            existing_vars[vname] = {
                'values': ds.variables[vname][:].copy(),
                'attrs': {a: ds.variables[vname].getncattr(a)
                          for a in ds.variables[vname].ncattrs()},
            }
        global_attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}

    log.debug(f"Resizing para_def.nc: par={par_size} -> {n_subcatchments}")

    tmp_path = para_def_path.with_suffix('.tmp.nc')
    with nc.Dataset(tmp_path, 'w', format='NETCDF4') as ds_new:
        ds_new.createDimension('par', n_subcatchments)
        ds_new.createVariable('par', 'i4', ('par',))
        ds_new.variables['par'][:] = np.arange(n_subcatchments)
        for attr_name, attr_val in global_attrs.items():
            ds_new.setncattr(attr_name, attr_val)

        for vname, vinfo in existing_vars.items():
            fill_value = vinfo['attrs'].get('_FillValue', None)
            ds_new.createVariable(vname, 'f8', ('par',), fill_value=fill_value)
            ds_new.variables[vname][:] = np.full(
                n_subcatchments, float(vinfo['values'][0])
            )
            for attr_name, attr_val in vinfo['attrs'].items():
                if attr_name == '_FillValue':
                    continue
                ds_new.variables[vname].setncattr(attr_name, attr_val)

    shutil.move(str(tmp_path), str(para_def_path))


def _enforce_numerix_defaults(ds: Any, log: logging.Logger) -> None:
    """Ensure numerical solver settings are reasonable in para_def.nc."""
    numerix_defaults = {
        'SOLUTION': 0.0,        # Explicit Euler (fastest)
        'TIMSTEP_TYP': 0.0,     # Fixed time steps
        'ERRITERFUNC': 1e-4,
        'ERR_ITER_DX': 1e-4,
        'NITER_TOTAL': 5000.0,
        'MIN_TSTEP': 0.001 / 1440.0,
    }
    for nvar, nval in numerix_defaults.items():
        if nvar in ds.variables:
            cur = float(ds.variables[nvar][0])
            if nvar in ('SOLUTION', 'TIMSTEP_TYP'):
                if cur != nval:
                    ds.variables[nvar][:] = nval
                    log.debug(f"  Set {nvar}: {cur:.0f} -> {nval:.0f}")
            elif cur < nval * 0.01:
                ds.variables[nvar][:] = nval
                log.debug(f"  Relaxed {nvar}: {cur:.2e} -> {nval:.2e}")


def update_constraints_file(
    constraints_file: Path,
    params: Dict[str, float],
    log: Optional[logging.Logger] = None
) -> Set[str]:
    """
    Update FUSE constraints file with new parameter default values.

    FUSE uses Fortran fixed-width format: (L1,1X,I1,1X,3(F9.3,1X),...)
    The default value column starts at position 4 and is exactly 9 characters.

    Args:
        constraints_file: Path to constraints file
        params: Parameter values to apply
        log: Logger instance

    Returns:
        Set of parameter names that were updated
    """
    log = log or logger
    params_updated: set = set()

    try:
        lines = _read_constraints_file(constraints_file, log)

        # Fortran format: default value column at position 4-12 (9 chars, F9.3)
        DEFAULT_VALUE_START = 4
        DEFAULT_VALUE_WIDTH = 9

        updated_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('(') or stripped.startswith('*') or stripped.startswith('!'):
                updated_lines.append(line)
                continue

            updated = False
            for param_name, value in params.items():
                parts = line.split()
                if len(parts) >= 14 and param_name in parts:
                    if parts[13] == param_name:
                        new_value = f"{value:9.3f}"
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

        with open(constraints_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

    except (OSError, IOError) as e:
        log.warning(f"I/O error updating constraints file: {e}")
    except (IndexError, ValueError) as e:
        log.warning(f"Format error updating constraints file: {e}")

    return params_updated


def _read_constraints_file(constraints_file: Path, log: logging.Logger) -> list:
    """Read constraints file with encoding fallback."""
    try:
        with open(constraints_file, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        log.warning(
            f"UTF-8 decode error reading {constraints_file}, falling back to latin-1"
        )
        with open(constraints_file, 'r', encoding='latin-1') as f:
            return f.readlines()
