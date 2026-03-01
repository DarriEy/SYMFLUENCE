# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HEC-HMS Result Extractor.

Handles extraction of simulation results from HEC-HMS model outputs
for integration with the evaluation framework.
"""

from pathlib import Path
from typing import Dict, List, cast

import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class HecHmsResultExtractor(ModelResultExtractor):
    """HEC-HMS-specific result extraction.

    Handles HEC-HMS output characteristics:
    - Variable naming: streamflow, runoff
    - File patterns: *_hechms_output.nc, *_hechms_output.csv
    - Units: streamflow in m3/s, runoff in mm/day
    """

    def __init__(self):
        """Initialize the HEC-HMS result extractor."""
        super().__init__('HECHMS')

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for HEC-HMS outputs."""
        return {
            'streamflow': [
                '*_hechms_output.nc',
                '*_hechms_output.csv',
            ],
            'runoff': [
                '*_hechms_output.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get HEC-HMS variable names for different types."""
        variable_mapping = {
            'streamflow': ['streamflow', 'discharge', 'Q'],
            'runoff': ['runoff', 'total_runoff', 'q'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from HEC-HMS output.

        Args:
            output_file: Path to HEC-HMS output file (NetCDF or CSV)
            variable_type: Type of variable to extract
            **kwargs: Additional options (catchment_area for unit conversion)

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
            if var_name in df.columns:
                return df[var_name]
            if var_name == 'streamflow' and 'streamflow_cms' in df.columns:
                return df['streamflow_cms']

        raise ValueError(
            f"No suitable variable found in {output_file}. "
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

                    # Handle non-time dimensions
                    non_time_dims = [dim for dim in var.dims if dim != 'time']
                    for dim in non_time_dims:
                        var = var.isel({dim: 0})

                    result = cast(pd.Series, var.to_pandas())

                    if variable_type == 'streamflow':
                        catchment_area = kwargs.get('catchment_area')
                        if catchment_area is not None and 'runoff' in var_name.lower():
                            result = result * catchment_area / 1000 / 86400

                    return result

            raise ValueError(
                f"No suitable variable found for '{variable_type}' in {output_file}. "
                f"Tried: {var_names}. Available: {list(ds.data_vars)}"
            )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """HEC-HMS outputs streamflow in m3/s, runoff in mm/day."""
        return variable_type == 'runoff'

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """HEC-HMS uses selection for spatial aggregation."""
        return 'selection'
