"""
FUSE (Framework for Understanding Structural Errors) model postprocessor.

Simplified implementation using StandardModelPostprocessor.
"""

from typing import Dict, Any

import numpy as np
import xarray as xr

from ..base import StandardModelPostprocessor
from ..registry import ModelRegistry


@ModelRegistry.register_postprocessor('FUSE')
class FUSEPostprocessor(StandardModelPostprocessor):
    """
    Postprocessor for FUSE model outputs.

    FUSE outputs NetCDF files with routed streamflow in mm/day.
    This postprocessor extracts the streamflow and converts to cms.
    """

    model_name = "FUSE"
    output_file_pattern = "{domain}_{experiment}_runs_best.nc"
    streamflow_variable = "q_routed"
    streamflow_unit = "mm_per_day"  # Will be converted to cms
    netcdf_selections = {"latitude": 0, "longitude": 0}  # param_set determined dynamically

    def _get_netcdf_selections(self) -> Dict[str, Any]:
        """
        Get NetCDF dimension selections with auto-detected param_set.

        FUSE writes valid data to different param_set indices depending on
        the run mode (e.g., run_def writes to param_set 1, not 0).
        This method auto-detects which param_set has valid (non-NaN) data.
        """
        selections = dict(self.netcdf_selections)

        # Try to auto-detect valid param_set from output file
        try:
            output_file = self._get_output_file()
            if output_file.exists():
                with xr.open_dataset(output_file) as ds:
                    if self.streamflow_variable in ds and 'param_set' in ds[self.streamflow_variable].dims:
                        var = ds[self.streamflow_variable]
                        n_param_sets = var.sizes.get('param_set', 1)
                        valid_param_set = 0
                        for ps in range(n_param_sets):
                            test_slice = var.isel(param_set=ps)
                            if 'latitude' in test_slice.dims:
                                test_slice = test_slice.isel(latitude=0)
                            if 'longitude' in test_slice.dims:
                                test_slice = test_slice.isel(longitude=0)
                            if not np.all(np.isnan(test_slice.values)):
                                valid_param_set = ps
                                break
                        selections['param_set'] = valid_param_set
                        self.logger.debug(f"Auto-detected valid param_set: {valid_param_set}")
                    elif 'param_set' in ds.dims:
                        # Variable doesn't have param_set but dataset does
                        selections['param_set'] = 0
        except Exception as e:
            self.logger.debug(f"Could not auto-detect param_set: {e}, defaulting to 0")
            selections['param_set'] = 0

        return selections
