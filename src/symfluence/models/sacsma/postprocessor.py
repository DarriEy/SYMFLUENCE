"""
SAC-SMA Model Postprocessor.

Uses StandardModelPostprocessor for minimal boilerplate.
"""

from symfluence.models.base.standard_postprocessor import StandardModelPostprocessor
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_postprocessor('SACSMA')
class SacSmaPostprocessor(StandardModelPostprocessor):
    """Postprocessor for SAC-SMA + Snow-17 model output."""

    model_name = "SACSMA"
    output_file_pattern = "{domain}_sacsma_output.nc"
    streamflow_variable = "streamflow"
    streamflow_unit = "cms"

    text_file_separator = ","
    text_file_skiprows = 0
    text_file_date_column = "datetime"
    text_file_flow_column = "streamflow_mm_day"

    resample_frequency = None

    def _get_output_file(self):
        """Get output file path, checking both NetCDF and CSV."""
        output_dir = self._get_output_dir()

        nc_file = output_dir / self._format_pattern(self.output_file_pattern)
        if nc_file.exists():
            return nc_file

        csv_pattern = "{domain}_sacsma_output.csv"
        csv_file = output_dir / self._format_pattern(csv_pattern)
        if csv_file.exists():
            return csv_file

        return nc_file
