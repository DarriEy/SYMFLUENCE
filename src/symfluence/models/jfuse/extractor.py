# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
jFUSE Result Extractor.

Handles extraction of simulation results from jFUSE (JAX-based FUSE)
model outputs for integration with the evaluation framework.
"""

from pathlib import Path
from typing import Dict, List, cast

import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class JFUSEResultExtractor(ModelResultExtractor):
    """jFUSE-specific result extraction.

    Handles jFUSE's unique output characteristics:
    - Variable naming: streamflow, runoff
    - File patterns: *_jfuse_output.nc, *_jfuse_output.csv
    - Units: streamflow in m³/s, runoff in mm/day
    - Dimensions: time, optionally hru for distributed mode
    """

    def __init__(self):
        """Initialize the jFUSE result extractor."""
        super().__init__('JFUSE')

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for jFUSE outputs."""
        return {
            'streamflow': [
                '*_jfuse_output.nc',
                '*_jfuse_output.csv',
                '*_jfuse_output_distributed.nc',
                '*_runs_def.nc',
            ],
            'runoff': [
                '*_jfuse_output.nc',
                '*_jfuse_output_distributed.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get jFUSE variable names for different types."""
        variable_mapping = {
            'streamflow': ['streamflow', 'discharge', 'q_routed', 'Q'],
            'runoff': ['runoff', 'total_runoff', 'q'],
            'et': ['et', 'evapotranspiration', 'aet'],
            'snow': ['snow', 'swe', 'snow_water_equivalent'],
            'soil_moisture': ['soil_moisture', 'sm', 'S1', 'S2'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from jFUSE output.

        Args:
            output_file: Path to jFUSE output file (NetCDF or CSV)
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - catchment_area: Catchment area in m² for unit conversion

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found
        """
        output_file = Path(output_file)
        var_names = self.get_variable_names(variable_type)

        if output_file.suffix == '.csv':
            return self._extract_from_csv(output_file, var_names)
        else:
            return self._extract_from_netcdf(output_file, var_names, variable_type, **kwargs)

    def _extract_from_csv(self, output_file: Path, var_names: List[str]) -> pd.Series:
        """Extract variable from CSV output."""
        df = pd.read_csv(output_file, index_col='datetime', parse_dates=True)

        for var_name in var_names:
            # Check for exact match
            if var_name in df.columns:
                return df[var_name]
            # Check for streamflow_cms column (common in jFUSE output)
            if var_name == 'streamflow' and 'streamflow_cms' in df.columns:
                return df['streamflow_cms']

        raise ValueError(
            f"No suitable variable found for extraction in {output_file}. "
            f"Tried: {var_names}. Available: {list(df.columns)}"
        )

    def _extract_from_netcdf(
        self,
        output_file: Path,
        var_names: List[str],
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from NetCDF output."""
        with xr.open_dataset(output_file) as ds:
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]

                    # Handle spatial dimensions (hru, etc.)
                    var = self._handle_spatial_dimensions(var)

                    # Convert units if needed for streamflow
                    result = cast(pd.Series, var.to_pandas())

                    if variable_type == 'streamflow':
                        # jFUSE outputs streamflow in m³/s, no conversion needed
                        # unless it's runoff (mm/day) that needs conversion
                        catchment_area = kwargs.get('catchment_area')
                        if catchment_area is not None and 'runoff' in var_name.lower():
                            # mm/day to m³/s: (mm/day) * (area_m²) / (1000 mm/m) / (86400 s/day)
                            result = result * catchment_area / 1000 / 86400

                    return result

            raise ValueError(
                f"No suitable variable found for '{variable_type}' in {output_file}. "
                f"Tried: {var_names}. Available: {list(ds.data_vars)}"
            )

    def _handle_spatial_dimensions(self, var: xr.DataArray) -> xr.DataArray:
        """Handle jFUSE spatial dimensions.

        jFUSE outputs may have:
        - hru: Hydrologic response unit dimension (select first or sum)
        - param_set: Parameter set dimension (select first)

        Args:
            var: xarray DataArray

        Returns:
            DataArray with spatial dimensions reduced
        """
        # Select first param_set if present
        if 'param_set' in var.dims:
            var = var.isel(param_set=0)

        # Select first hru if present (for lumped equivalent)
        if 'hru' in var.dims:
            var = var.isel(hru=0)

        # Handle any remaining non-time dimensions
        non_time_dims = [dim for dim in var.dims if dim != 'time']
        for dim in non_time_dims:
            var = var.isel({dim: 0})

        return var

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """jFUSE outputs streamflow in m³/s, runoff in mm/day."""
        return variable_type == 'runoff'

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """jFUSE uses selection for spatial aggregation."""
        return 'selection'
