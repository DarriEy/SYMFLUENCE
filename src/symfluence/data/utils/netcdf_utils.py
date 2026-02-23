"""
NetCDF utility functions for SYMFLUENCE.

Provides common operations for NetCDF file handling, including
standardized encoding for compressed output files.
"""

from typing import Any, Dict, Optional

import xarray as xr


def create_netcdf_encoding(
    dataset: xr.Dataset,
    compression: bool = True,
    complevel: int = 4,
    fill_value: float = -9999.0,
    dtype: str = 'float32',
    time_dtype: str = 'float64',
    int_vars: Optional[Dict[str, str]] = None,
    custom_encoding: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Create a standardized encoding dictionary for NetCDF export.

    Consolidates the encoding creation logic from FUSE, SUMMA, and GR runners
    into a single reusable utility.

    Args:
        dataset: xarray Dataset to create encoding for
        compression: Whether to enable zlib compression (default: True)
        complevel: Compression level 1-9 (default: 4)
        fill_value: Fill value for missing data (default: -9999.0)
        dtype: Default dtype for data variables (default: 'float32')
        time_dtype: Dtype for time coordinate (default: 'float64')
        int_vars: Dict mapping variable names to int dtypes (e.g., {'gruId': 'int32'})
        custom_encoding: Dict of custom encoding per variable (overrides defaults)

    Returns:
        Dict[str, Dict]: Encoding dictionary for xr.Dataset.to_netcdf()

    Examples:
        >>> # Basic usage
        >>> encoding = create_netcdf_encoding(ds)
        >>> ds.to_netcdf('output.nc', encoding=encoding)

        >>> # With custom integer variables
        >>> encoding = create_netcdf_encoding(
        ...     ds,
        ...     int_vars={'gru': 'int32', 'gruId': 'int32'}
        ... )

        >>> # With custom encoding for specific variables
        >>> encoding = create_netcdf_encoding(
        ...     ds,
        ...     custom_encoding={'q_routed': {'shuffle': True}}
        ... )
    """
    encoding: Dict[str, Dict[str, Any]] = {}

    # Handle data variables
    for var in dataset.data_vars:
        var_name = str(var)
        var_encoding: Dict[str, Any] = {
            'dtype': dtype,
            '_FillValue': fill_value,
        }

        # Add compression if enabled
        if compression:
            var_encoding.update({
                'zlib': True,
                'complevel': complevel,
            })

        # Check if this is an integer variable
        if int_vars and var_name in int_vars:
            var_encoding['dtype'] = int_vars[var_name]
            var_encoding['_FillValue'] = None  # No fill value for integers

        encoding[var_name] = var_encoding

    # Handle coordinates
    for coord in dataset.coords:
        coord_name = str(coord)
        if coord_name == 'time':
            encoding[coord_name] = {
                'dtype': time_dtype,
                '_FillValue': None,
            }
        elif coord_name in ('gru', 'hru', 'subcatchment', 'param_set'):
            # Integer coordinates
            encoding[coord_name] = {
                'dtype': 'int32',
            }
        else:
            # Other coordinates (lat, lon, etc.)
            encoding[coord_name] = {
                'dtype': 'float64',
            }

    # Apply custom encoding overrides
    if custom_encoding:
        for var_name, var_enc in custom_encoding.items():
            str_var_name = str(var_name)
            if str_var_name in encoding:
                encoding[str_var_name].update(var_enc)
            else:
                encoding[str_var_name] = var_enc

    return encoding


def create_minimal_encoding(
    dataset: xr.Dataset,
    preserve_fill: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Create minimal encoding (no compression, preserve original dtypes).

    Useful for intermediate files or when speed is more important than size.

    Args:
        dataset: xarray Dataset to create encoding for
        preserve_fill: Whether to preserve existing _FillValue attributes

    Returns:
        Dict[str, Dict]: Encoding dictionary for xr.Dataset.to_netcdf()
    """
    encoding: Dict[str, Dict[str, Any]] = {}

    for var in dataset.data_vars:
        var_name = str(var)
        var_encoding: Dict[str, Any] = {}

        if not preserve_fill:
            var_encoding['_FillValue'] = None

        if var_encoding:
            encoding[var_name] = var_encoding

    # Ensure time has no fill value
    if 'time' in dataset.coords:
        encoding['time'] = {'_FillValue': None}

    return encoding
