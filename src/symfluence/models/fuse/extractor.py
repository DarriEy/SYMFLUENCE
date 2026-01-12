"""
FUSE Result Extractor.

Handles extraction of simulation results from FUSE (Framework for Understanding
Structural Errors) model outputs.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class FUSEResultExtractor(ModelResultExtractor):
    """FUSE-specific result extraction.

    Handles FUSE's unique output characteristics:
    - Variable naming: q_routed (routed streamflow)
    - File patterns: *_runs_best.nc, *_runs_def.nc
    - Units: mm/day (needs conversion to m³/s)
    - Dimensions: param_set, latitude, longitude, time
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for FUSE outputs."""
        return {
            'streamflow': [
                '*_runs_best.nc',
                '*_runs_def.nc',
                '*_output.nc',
                '*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get FUSE variable names for different types."""
        variable_mapping = {
            'streamflow': ['q_routed', 'qrouted', 'streamflow', 'discharge'],
            'et': ['evapotranspiration', 'et'],
            'soil_moisture': ['soil_moisture', 'sm'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from FUSE output.

        Args:
            output_file: Path to FUSE NetCDF output
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - catchment_area: Catchment area in m² for unit conversion

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found
        """
        var_names = self.get_variable_names(variable_type)

        with xr.open_dataset(output_file) as ds:
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]

                    # Handle spatial dimensions (param_set, lat, lon)
                    var = self._handle_spatial_dimensions(var)

                    # Convert units if needed
                    if variable_type == 'streamflow':
                        result = var.to_pandas()
                        # Convert mm/day to m³/s if catchment area provided
                        catchment_area = kwargs.get('catchment_area')
                        if catchment_area is not None:
                            # mm/day to m³/s: (mm/day) * (area_m²) / (1000 mm/m) / (86400 s/day)
                            result = result * catchment_area / 1000 / 86400
                        return result
                    else:
                        return var.to_pandas()

            raise ValueError(
                f"No suitable variable found for '{variable_type}' in {output_file}. "
                f"Tried: {var_names}"
            )

    def _handle_spatial_dimensions(self, var: xr.DataArray) -> xr.DataArray:
        """Handle FUSE spatial dimensions.

        FUSE outputs may have:
        - param_set: Different parameter sets (take first or best)
        - latitude/longitude: Spatial coordinates (select first)

        Args:
            var: xarray DataArray

        Returns:
            DataArray with spatial dimensions reduced
        """
        # Select first param_set if present
        if 'param_set' in var.dims:
            var = var.isel(param_set=0)

        # Select first latitude if present
        if 'latitude' in var.dims:
            var = var.isel(latitude=0)

        # Select first longitude if present
        if 'longitude' in var.dims:
            var = var.isel(longitude=0)

        # Handle any remaining spatial dimensions
        non_time_dims = [dim for dim in var.dims if dim != 'time']
        for dim in non_time_dims:
            var = var.isel({dim: 0})

        return var

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """FUSE outputs mm/day, which needs conversion to m³/s."""
        return variable_type == 'streamflow'

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """FUSE uses parameter sets and spatial selection."""
        return 'selection'  # Select first param_set and spatial point
