"""
mizuRoute Result Extractor.

Handles extraction of routed streamflow from mizuRoute model outputs.
Encapsulates mizuRoute-specific logic for reach/segment identification
and routing variable names.
"""

from pathlib import Path
from typing import List, Dict
import pandas as pd
import xarray as xr
import numpy as np

from symfluence.models.base import ModelResultExtractor


class MizuRouteResultExtractor(ModelResultExtractor):
    """mizuRoute-specific result extraction.

    Handles mizuRoute's unique output characteristics:
    - Variable naming: IRFroutedRunoff, KWTroutedRunoff, averageRoutedRunoff
    - Spatial dimension: seg (river segments) or reachID (reaches)
    - Outlet identification: reach/segment with highest mean discharge
    - File patterns: mizuRoute/*.nc, *.h.*.nc
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for mizuRoute outputs."""
        return {
            'streamflow': [
                'mizuRoute/*.nc',
                '*.h.*.nc',
                'mizuRoute/**/*.nc',
                '*_routed.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get mizuRoute variable names for different types."""
        variable_mapping = {
            'streamflow': ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract routed streamflow from mizuRoute output.

        Args:
            output_file: Path to mizuRoute NetCDF output
            variable_type: Type of variable (typically 'streamflow')
            **kwargs: Additional options (currently unused)

        Returns:
            Time series of routed discharge at outlet

        Raises:
            ValueError: If no routed runoff variable found
        """
        if variable_type != 'streamflow':
            raise ValueError(
                f"mizuRoute extractor only supports 'streamflow', got '{variable_type}'"
            )

        var_names = self.get_variable_names(variable_type)

        with xr.open_dataset(output_file) as ds:
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]

                    # Find outlet based on spatial dimension
                    if 'seg' in var.dims:
                        # Older mizuRoute: segment-based
                        segment_means = var.mean(dim='time').values
                        outlet_seg_idx = np.argmax(segment_means)
                        result = var.isel(seg=outlet_seg_idx).to_pandas()
                        return result

                    elif 'reachID' in var.dims:
                        # Newer mizuRoute: reach-based
                        reach_means = var.mean(dim='time').values
                        outlet_reach_idx = np.argmax(reach_means)
                        result = var.isel(reachID=outlet_reach_idx).to_pandas()
                        return result

                    else:
                        # No spatial dimension - use as-is
                        return var.to_pandas()

            raise ValueError(
                f"No suitable routed runoff variable found in {output_file}. "
                f"Tried: {var_names}"
            )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """mizuRoute outputs are already in m³/s, no conversion needed."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """mizuRoute automatically aggregates to reaches/segments."""
        return 'outlet_selection'  # Select outlet reach with max discharge
