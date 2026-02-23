"""
dRoute Result Extractor.

Handles extraction of routed streamflow from dRoute model outputs.
Encapsulates dRoute-specific logic for outlet identification and
routing variable names.
"""

from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from symfluence.models.base import ModelResultExtractor


class DRouteResultExtractor(ModelResultExtractor):
    """
    dRoute-specific result extraction.

    Handles dRoute's unique output characteristics:
    - Variable naming: routedRunoff, outletStreamflow
    - Spatial dimension: seg (river segments)
    - Outlet identification: pre-computed outlet or max discharge segment
    - File patterns: *_droute_output.nc
    """

    def __init__(self):
        """Initialize dRoute result extractor."""
        super().__init__('DROUTE')

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for dRoute outputs."""
        return {
            'streamflow': [
                'dRoute/*_droute_output.nc',
                '*_droute_output.nc',
                'dRoute/**/*.nc',
                '*_routed.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get dRoute variable names for different types."""
        variable_mapping = {
            'streamflow': ['outletStreamflow', 'routedRunoff'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """
        Extract routed streamflow from dRoute output.

        Args:
            output_file: Path to dRoute NetCDF output
            variable_type: Type of variable (typically 'streamflow')
            **kwargs: Additional options:
                - outlet_idx: Specific outlet segment index
                - segment_id: Specific segment ID to extract

        Returns:
            Time series of routed discharge at outlet

        Raises:
            ValueError: If no routed runoff variable found
        """
        if variable_type != 'streamflow':
            raise ValueError(
                f"dRoute extractor only supports 'streamflow', got '{variable_type}'"
            )

        outlet_idx = kwargs.get('outlet_idx')
        segment_id = kwargs.get('segment_id')

        if not HAS_XARRAY:
            return self._extract_without_xarray(output_file, outlet_idx, segment_id)

        var_names = self.get_variable_names(variable_type)

        with xr.open_dataset(output_file) as ds:
            # First try outlet streamflow (1D, pre-extracted)
            if 'outletStreamflow' in ds.variables:
                result = cast(pd.Series, ds['outletStreamflow'].to_pandas())
                return result

            # Otherwise extract from full routed array
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]

                    # Check dimensions
                    if 'seg' in var.dims:
                        if outlet_idx is not None:
                            # Use specified outlet
                            result = cast(pd.Series, var.isel(seg=outlet_idx).to_pandas())
                            return result
                        elif segment_id is not None:
                            # Find segment by ID
                            if 'segId' in ds:
                                seg_ids = ds['segId'].values
                                idx = np.where(seg_ids == segment_id)[0]
                                if len(idx) > 0:
                                    result = cast(pd.Series, var.isel(seg=idx[0]).to_pandas())
                                    return result
                        else:
                            # Default: find outlet by max mean discharge
                            segment_means = var.mean(dim='time').values
                            outlet_seg_idx = np.argmax(segment_means)
                            result = cast(pd.Series, var.isel(seg=outlet_seg_idx).to_pandas())
                            return result

                    else:
                        # No spatial dimension - use as-is
                        return cast(pd.Series, var.to_pandas())

            raise ValueError(
                f"No suitable routed runoff variable found in {output_file}. "
                f"Tried: {var_names}"
            )

    def _extract_without_xarray(
        self,
        output_file: Path,
        outlet_idx=None,
        segment_id=None
    ) -> pd.Series:
        """Fallback extraction using netCDF4 directly."""
        import netCDF4 as nc4

        with nc4.Dataset(output_file, 'r') as ds:
            time = ds.variables['time'][:]
            time_units = ds.variables['time'].units
            time_calendar = getattr(ds.variables['time'], 'calendar', 'standard')

            # Convert time to datetime
            time_dt = nc4.num2date(time, units=time_units, calendar=time_calendar)
            time_index = pd.DatetimeIndex([pd.Timestamp(t) for t in time_dt])

            # Try outlet streamflow first
            if 'outletStreamflow' in ds.variables:
                values = ds.variables['outletStreamflow'][:]
                return pd.Series(values, index=time_index, name='streamflow')

            # Extract from routedRunoff
            if 'routedRunoff' in ds.variables:
                data = ds.variables['routedRunoff'][:]

                if outlet_idx is not None:
                    values = data[:, outlet_idx]
                else:
                    # Find outlet by max mean
                    means = np.nanmean(data, axis=0)
                    outlet_idx = np.argmax(means)
                    values = data[:, outlet_idx]

                return pd.Series(values, index=time_index, name='streamflow')

        raise ValueError(f"No streamflow data found in {output_file}")

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """dRoute outputs are already in m3/s, no conversion needed."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """dRoute aggregates to outlet segment."""
        return 'outlet_selection'


__all__ = ['DRouteResultExtractor']
