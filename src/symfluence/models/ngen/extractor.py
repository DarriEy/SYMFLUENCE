"""
NGEN Result Extractor.

Handles extraction of simulation results from NextGen framework outputs.
NGEN outputs can come from troute (routing) or catchment-level results.
"""

from pathlib import Path
from typing import List, Dict
import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class NGENResultExtractor(ModelResultExtractor):
    """NGEN-specific result extraction.

    Handles NextGen framework's output characteristics:
    - Variable naming: depends on formulation (CFE, NOAH, etc.)
    - File patterns: nexus_data.nc, catchment_data.nc, troute outputs
    - Routing: Often uses troute for streamflow routing
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for NGEN outputs."""
        return {
            'streamflow': [
                '**/nex-troute-out.nc',  # Troute routed outputs
                '**/troute_*.nc',
                '**/nexus_data.nc',      # Nexus outputs
                '**/catchment_data.nc',  # Catchment outputs
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get NGEN variable names for different types."""
        variable_mapping = {
            'streamflow': [
                'streamflow',
                'discharge',
                'q_lateral',
                'water_out',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from NGEN output.

        Args:
            output_file: Path to NGEN NetCDF output
            variable_type: Type of variable to extract
            **kwargs: Additional options

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

                    # Handle spatial dimensions if present
                    if len(var.shape) > 1:
                        # Try to find outlet/main nexus
                        spatial_dims = [d for d in var.dims if d != 'time']
                        if spatial_dims:
                            # Select first spatial unit as fallback
                            # TODO: Could be enhanced to find outlet nexus
                            var = var.isel({spatial_dims[0]: 0})

                    return var.to_pandas()

            raise ValueError(
                f"No suitable variable found for '{variable_type}' in {output_file}. "
                f"Tried: {var_names}"
            )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """NGEN outputs are typically in standard units."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """NGEN can have catchment or nexus level outputs."""
        return 'outlet_selection'
