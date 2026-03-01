# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HBV Result Extractor.

Handles extraction of simulation results from HBV-96 model outputs
for integration with the evaluation framework.
"""

from pathlib import Path
from typing import Dict, List, cast

import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class HBVResultExtractor(ModelResultExtractor):
    """HBV-96-specific result extraction.

    Handles HBV's unique output characteristics:
    - Variable naming: streamflow, runoff, snow, sm, suz, slz
    - File patterns: *_hbv_output.nc, *_hbv_output.csv
    - Units: streamflow in m³/s, runoff in mm/day
    - Dimensions: time, optionally hru for distributed mode
    """

    def __init__(self):
        """Initialize the HBV result extractor."""
        super().__init__('HBV')

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for HBV outputs."""
        return {
            'streamflow': [
                '*_hbv_output.nc',
                '*_hbv_output.csv',
                '*_hbv_output_distributed.nc',
                '*_runs_def.nc',
            ],
            'runoff': [
                '*_hbv_output.nc',
                '*_hbv_output_distributed.nc',
            ],
            'snow': [
                '*_hbv_output.nc',
                '*_hbv_states.nc',
            ],
            'soil_moisture': [
                '*_hbv_output.nc',
                '*_hbv_states.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get HBV variable names for different types."""
        variable_mapping = {
            'streamflow': ['streamflow', 'discharge', 'Q', 'q_routed'],
            'runoff': ['runoff', 'total_runoff', 'q'],
            'snow': ['snow', 'swe', 'snow_water', 'snow_water_equivalent'],
            'soil_moisture': ['sm', 'soil_moisture', 'soil_water'],
            'upper_zone': ['suz', 'upper_storage', 'upper_zone_storage'],
            'lower_zone': ['slz', 'lower_storage', 'lower_zone_storage'],
            'et': ['et', 'evapotranspiration', 'aet'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from HBV output.

        Args:
            output_file: Path to HBV output file (NetCDF or CSV)
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
            # Check for streamflow_cms column (common in HBV output)
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
                        # HBV outputs streamflow in m³/s, no conversion needed
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
        """Handle HBV spatial dimensions.

        HBV outputs may have:
        - hru: Hydrologic response unit dimension (select first or sum)
        - node: Network node dimension for distributed mode

        Args:
            var: xarray DataArray

        Returns:
            DataArray with spatial dimensions reduced
        """
        # Select first hru if present (for lumped equivalent)
        if 'hru' in var.dims:
            var = var.isel(hru=0)

        # Select first node if present
        if 'node' in var.dims:
            var = var.isel(node=0)

        # Handle any remaining non-time dimensions
        non_time_dims = [dim for dim in var.dims if dim != 'time']
        for dim in non_time_dims:
            var = var.isel({dim: 0})

        return var

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """HBV outputs streamflow in m³/s, runoff in mm/day."""
        return variable_type == 'runoff'

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """HBV uses selection for spatial aggregation."""
        return 'selection'
