# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIC Result Extractor.

Handles extraction of simulation results from VIC model outputs.
VIC outputs are in NetCDF format with standard variable names.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.models.base import ModelResultExtractor


class VICResultExtractor(ModelResultExtractor):
    """VIC-specific result extraction.

    Handles VIC model's output characteristics:
    - File formats: NetCDF (.nc)
    - Variable naming: OUT_RUNOFF, OUT_BASEFLOW, OUT_EVAP, etc.
    - Spatial aggregation: Grid cells to basin
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for VIC outputs."""
        return {
            'streamflow': [
                '*fluxes*.nc',
                'vic_output*.nc',
                '*runoff*.nc',
            ],
            'et': [
                '*fluxes*.nc',
                'vic_output*.nc',
            ],
            'snow': [
                '*fluxes*.nc',
                'vic_output*.nc',
            ],
            'soil_moisture': [
                '*fluxes*.nc',
                'vic_output*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get VIC variable names for different types.

        VIC output variables follow the OUT_* naming convention:
        - OUT_RUNOFF: Surface runoff [mm]
        - OUT_BASEFLOW: Baseflow [mm]
        - OUT_EVAP: Evaporation [mm]
        - OUT_SWE: Snow water equivalent [mm]
        """
        variable_mapping = {
            'streamflow': [
                'OUT_RUNOFF',
                'OUT_BASEFLOW',
                'RUNOFF',
                'BASEFLOW',
            ],
            'runoff': [
                'OUT_RUNOFF',
                'RUNOFF',
            ],
            'baseflow': [
                'OUT_BASEFLOW',
                'BASEFLOW',
            ],
            'et': [
                'OUT_EVAP',
                'OUT_EVAP_CANOP',
                'OUT_TRANSP_VEG',
                'EVAP',
            ],
            'snow': [
                'OUT_SWE',
                'OUT_SNOW_DEPTH',
                'SWE',
            ],
            'soil_moisture': [
                'OUT_SOIL_MOIST',
                'OUT_SOIL_WET',
                'SOIL_MOIST',
            ],
            'precipitation': [
                'OUT_PREC',
                'PREC',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from VIC output.

        Args:
            output_file: Path to VIC output file (NetCDF)
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - catchment_area: Catchment area in m² for unit conversion
                - aggregate: Whether to aggregate grid cells (default: True)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        import xarray as xr

        var_names = self.get_variable_names(variable_type)
        aggregate = kwargs.get('aggregate', True)
        catchment_area = kwargs.get('catchment_area')

        try:
            ds = xr.open_dataset(output_file)

            # Find the variable
            found_var = None
            for var_name in var_names:
                if var_name in ds.data_vars:
                    found_var = var_name
                    break

            if found_var is None:
                available = list(ds.data_vars)
                ds.close()
                raise ValueError(
                    f"No suitable variable found in {output_file}. "
                    f"Tried: {var_names}. Available: {available}"
                )

            data = ds[found_var]

            # Aggregate spatially if needed
            if aggregate and len(data.dims) > 1:
                # Sum over spatial dimensions (typical for runoff/baseflow)
                spatial_dims = [d for d in data.dims if d not in ['time']]
                if variable_type in ['streamflow', 'runoff', 'baseflow']:
                    # For streamflow, we want to sum runoff + baseflow
                    if variable_type == 'streamflow':
                        runoff = data.sum(dim=spatial_dims)
                        # Also try to add baseflow
                        baseflow_vars = self.get_variable_names('baseflow')
                        for bvar in baseflow_vars:
                            if bvar in ds.data_vars and bvar != found_var:
                                baseflow = ds[bvar].sum(dim=spatial_dims)
                                runoff = runoff + baseflow
                                break
                        data = runoff
                    else:
                        data = data.sum(dim=spatial_dims)
                else:
                    # For other variables, take mean
                    data = data.mean(dim=spatial_dims)

            # Convert to pandas Series
            if 'time' in data.dims:
                series = data.to_series()
            else:
                # Single value case
                series = pd.Series([float(data.values)], index=[pd.Timestamp.now()])

            # Unit conversion for streamflow
            if variable_type == 'streamflow' and catchment_area:
                # VIC outputs in mm/timestep
                # Convert to m³/s: mm * area_m² / 1000 / seconds_per_day
                seconds_per_day = 86400
                series = series * catchment_area / 1000 / seconds_per_day

            ds.close()
            return series

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(
                f"Error reading VIC output file {output_file}: {str(e)}"
            ) from e

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract total streamflow (runoff + baseflow) from VIC output.

        Args:
            output_dir: Directory containing VIC outputs
            catchment_area: Catchment area in m² for conversion to m³/s
            **kwargs: Additional options

        Returns:
            Time series of streamflow
        """
        import xarray as xr

        # Find output file
        patterns = self.get_output_file_patterns()['streamflow']
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            raise FileNotFoundError(
                f"No VIC output file found in {output_dir} matching {patterns}"
            )

        ds = xr.open_dataset(output_file)

        # Get runoff and baseflow
        runoff = None
        baseflow = None

        for var in self.get_variable_names('runoff'):
            if var in ds.data_vars:
                runoff = ds[var]
                break

        for var in self.get_variable_names('baseflow'):
            if var in ds.data_vars:
                baseflow = ds[var]
                break

        if runoff is None:
            ds.close()
            raise ValueError("Could not find runoff variable in VIC output")

        # Aggregate spatially
        spatial_dims = [d for d in runoff.dims if d not in ['time']]
        total_runoff = runoff.sum(dim=spatial_dims)

        if baseflow is not None:
            total_baseflow = baseflow.sum(dim=spatial_dims)
            total_flow = total_runoff + total_baseflow
        else:
            total_flow = total_runoff

        # Convert to series
        series = total_flow.to_series()

        # Unit conversion
        if catchment_area:
            # mm -> m³/s
            series = series * catchment_area / 1000 / 86400

        ds.close()
        return series

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion."""
        return variable_type in ['streamflow', 'runoff', 'baseflow']

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['streamflow', 'runoff', 'baseflow', 'et', 'precipitation']:
            return 'sum'  # Sum over area
        else:
            return 'mean'  # Average for states like SWE, soil moisture

    def aggregate_to_basin(
        self,
        output_file: Path,
        variable_type: str,
        cell_areas: Optional[Dict] = None
    ) -> pd.Series:
        """
        Aggregate grid cell outputs to basin level.

        Args:
            output_file: Path to VIC output file
            variable_type: Type of variable
            cell_areas: Optional dict mapping cell indices to areas

        Returns:
            Basin-aggregated time series
        """
        import xarray as xr

        ds = xr.open_dataset(output_file)
        var_names = self.get_variable_names(variable_type)

        found_var = None
        for var in var_names:
            if var in ds.data_vars:
                found_var = var
                break

        if found_var is None:
            ds.close()
            raise ValueError(f"Variable type {variable_type} not found")

        data = ds[found_var]

        # Determine aggregation method
        method = self.get_spatial_aggregation_method(variable_type)
        spatial_dims = [d for d in data.dims if d not in ['time']]

        if method == 'sum':
            if cell_areas:
                # Area-weighted sum
                weights = xr.DataArray(
                    list(cell_areas.values()),
                    dims=['cell']
                )
                aggregated = (data * weights).sum(dim=spatial_dims)
            else:
                aggregated = data.sum(dim=spatial_dims)
        else:
            if cell_areas:
                # Area-weighted mean
                total_area = sum(cell_areas.values())
                weights = xr.DataArray(
                    [a / total_area for a in cell_areas.values()],
                    dims=['cell']
                )
                aggregated = (data * weights).sum(dim=spatial_dims)
            else:
                aggregated = data.mean(dim=spatial_dims)

        series = aggregated.to_series()
        ds.close()

        return series
