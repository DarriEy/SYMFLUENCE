"""
SUMMA postprocessor module.

Simplified implementation using RoutedModelPostprocessor for mizuRoute outputs.
"""

from ..base import RoutedModelPostprocessor
from ..registry import ModelRegistry


@ModelRegistry.register_postprocessor('SUMMA')
class SUMMAPostprocessor(RoutedModelPostprocessor):
    """
    Postprocessor for SUMMA model outputs via MizuRoute routing.

    SUMMA uses mizuRoute for streamflow routing, outputting hourly
    routed runoff (IRFroutedRunoff) in cms. This postprocessor
    extracts the outlet reach and resamples to daily.
    """

    model_name = "SUMMA"
    # RoutedModelPostprocessor defaults handle:
    # - routing_file_pattern = "{experiment}.h.{start_date}-03600.nc"
    # - routing_variable = "IRFroutedRunoff"
    # - resample_frequency = "D" (hourly to daily)
    # - streamflow_unit = "cms" (no conversion needed)
