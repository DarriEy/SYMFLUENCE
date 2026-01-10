"""
FUSE (Framework for Understanding Structural Errors) model postprocessor.

Simplified implementation using StandardModelPostprocessor.
"""

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
    netcdf_selections = {"param_set": 0, "latitude": 0, "longitude": 0}
