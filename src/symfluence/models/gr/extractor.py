# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GR Result Extractor.

Handles extraction of simulation results from GR model outputs.
GR models (GR4J/GR5J/GR6J) can run in lumped (CSV) or distributed (NetCDF) modes.
"""

from pathlib import Path
from typing import Dict, List, cast

import pandas as pd

from symfluence.models.base import ModelResultExtractor


class GRResultExtractor(ModelResultExtractor):
    """GR-specific result extraction.

    Handles GR model's output characteristics:
    - Lumped mode: CSV file (GR_results.csv)
    - Distributed mode: NetCDF file (*_runs_def.nc)
    - Routing: Can use mizuRoute for distributed runs
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for GR outputs."""
        return {
            'streamflow': [
                # Lumped mode CSV
                'GR_results.csv',
                # Distributed mode NetCDF
                '*_runs_def.nc',
                '*_runs_best.nc',
                # mizuRoute routing
                'mizuRoute/*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get GR variable names for different types."""
        variable_mapping = {
            'streamflow': ['Qsim', 'Q', 'streamflow', 'discharge'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from GR output.

        Args:
            output_file: Path to GR output file (CSV or NetCDF)
            variable_type: Type of variable to extract
            **kwargs: Additional options

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable cannot be extracted
        """
        if output_file.suffix == '.csv':
            # Lumped mode CSV output
            return self._extract_from_csv(output_file)
        elif output_file.suffix == '.nc':
            # Distributed mode NetCDF or mizuRoute output
            return self._extract_from_netcdf(output_file)

        raise ValueError(f"Unsupported file format: {output_file.suffix}")

    def _extract_from_csv(self, output_file: Path) -> pd.Series:
        """Extract streamflow from GR CSV output.

        Args:
            output_file: Path to GR_results.csv

        Returns:
            Time series of streamflow
        """
        try:
            df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
            # Look for Qsim or similar column
            for col in ['Qsim', 'Q', 'streamflow', 'discharge']:
                if col in df.columns:
                    return df[col]
            # Fallback: use first numeric column
            return df.iloc[:, 0]
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(f"Failed to parse GR CSV output {output_file}: {e}") from e

    def _extract_from_netcdf(self, output_file: Path) -> pd.Series:
        """Extract streamflow from GR NetCDF output.

        Args:
            output_file: Path to GR NetCDF output

        Returns:
            Time series of streamflow
        """
        import numpy as np
        import xarray as xr

        with xr.open_dataset(output_file) as ds:
            # Try mizuRoute variables first
            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if 'seg' in var.dims:
                        outlet_idx = np.argmax(var.mean(dim='time').values)
                        return cast(pd.Series, var.isel(seg=outlet_idx).to_pandas())
                    elif 'reachID' in var.dims:
                        outlet_idx = np.argmax(var.mean(dim='time').values)
                        return cast(pd.Series, var.isel(reachID=outlet_idx).to_pandas())

            # Try GR-specific variables
            for var_name in self.get_variable_names('streamflow'):
                if var_name in ds.variables:
                    var = ds[var_name]
                    # Handle spatial dimensions if present
                    if len(var.shape) > 1:
                        spatial_dims = [d for d in var.dims if d != 'time']
                        if spatial_dims:
                            var = var.isel({spatial_dims[0]: 0})
                    return cast(pd.Series, var.to_pandas())

        raise ValueError(f"No suitable streamflow variable found in {output_file}")

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """GR outputs are typically in mm/day or m³/s depending on mode."""
        return False  # Units handled by evaluator if needed

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """GR can be lumped or distributed."""
        return 'outlet_selection'  # For distributed mode
