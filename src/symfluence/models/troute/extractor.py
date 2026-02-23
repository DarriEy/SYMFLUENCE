"""
TRoute Result Extractor.

Handles extraction of simulation results from t-route model outputs
for integration with the evaluation framework.
"""

from pathlib import Path
from typing import Dict, List, cast

import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class TRouteResultExtractor(ModelResultExtractor):
    """TRoute-specific result extraction.

    Handles TRoute's unique output characteristics:
    - Variable naming: flowveldepth, flow, streamflow
    - File patterns: troute/*.nc, *_output.nc
    - Units: m³/s (cms)
    - Dimensions: time, feature_id/segment
    """

    def __init__(self):
        """Initialize the TRoute result extractor."""
        super().__init__('TROUTE')

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for TRoute outputs."""
        return {
            'streamflow': [
                'troute/*.nc',
                'TROUTE/*.nc',
                '*_troute_output.nc',
                '*_output.nc',
                '*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get TRoute variable names for different types."""
        variable_mapping = {
            'streamflow': [
                'flowveldepth',
                'flow',
                'streamflow',
                'discharge',
                'q',
                'Q',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from TRoute output.

        Args:
            output_file: Path to TRoute NetCDF output file
            variable_type: Type of variable to extract ('streamflow')
            **kwargs: Additional options:
                - feature_id: Specific segment/reach ID to extract
                - select_outlet: If True, auto-select outlet (highest flow)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found
        """
        output_file = Path(output_file)
        var_names = self.get_variable_names(variable_type)

        with xr.open_dataset(output_file) as ds:
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]

                    # Handle spatial dimensions
                    var = self._handle_spatial_dimensions(var, **kwargs)

                    result = cast(pd.Series, var.to_pandas())
                    return result

            raise ValueError(
                f"No suitable variable found for '{variable_type}' in {output_file}. "
                f"Tried: {var_names}. Available: {list(ds.data_vars)}"
            )

    def _handle_spatial_dimensions(
        self,
        var: xr.DataArray,
        **kwargs
    ) -> xr.DataArray:
        """Handle TRoute spatial dimensions.

        TRoute outputs have:
        - feature_id: River segment identifier
        - qlateral: Lateral inflow (if present)

        Args:
            var: xarray DataArray
            **kwargs: Options including feature_id for selection

        Returns:
            DataArray with spatial dimensions reduced
        """
        feature_id = kwargs.get('feature_id')
        select_outlet = kwargs.get('select_outlet', True)

        # Handle feature_id dimension
        if 'feature_id' in var.dims:
            if feature_id is not None:
                # Select specific feature
                try:
                    if 'feature_id' in var.coords:
                        var = var.sel(feature_id=int(feature_id))
                    else:
                        var = var.isel(feature_id=int(feature_id))
                except (KeyError, IndexError):
                    # Fall back to outlet selection
                    var = self._select_outlet_by_flow(var)
            elif select_outlet:
                # Auto-select outlet (highest mean flow)
                var = self._select_outlet_by_flow(var)
            else:
                # Just take first segment
                var = var.isel(feature_id=0)

        # Handle any remaining non-time dimensions
        non_time_dims = [d for d in var.dims if d not in ('time', 'reference_time')]
        for dim in non_time_dims:
            var = var.isel({dim: 0})

        return var

    def _select_outlet_by_flow(self, var: xr.DataArray) -> xr.DataArray:
        """Select the outlet segment based on highest mean flow.

        Args:
            var: DataArray with feature_id dimension

        Returns:
            DataArray with feature_id removed (outlet selected)
        """
        if 'feature_id' not in var.dims:
            return var

        # Calculate mean flow for each segment
        mean_flow = var.mean(dim='time')

        # Find segment with highest mean flow
        outlet_idx = int(mean_flow.argmax())

        return var.isel(feature_id=outlet_idx)

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """TRoute outputs streamflow in m³/s (cms), no conversion needed."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """TRoute uses selection for spatial aggregation."""
        return 'selection'
