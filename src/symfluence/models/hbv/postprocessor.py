"""
HBV Model Postprocessor.

Extracts and processes HBV-96 model output for analysis and visualization.
Uses StandardModelPostprocessor for minimal boilerplate.
"""

from symfluence.models.base.standard_postprocessor import StandardModelPostprocessor
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_postprocessor('HBV')
class HBVPostprocessor(StandardModelPostprocessor):
    """
    Postprocessor for HBV-96 model output.

    Handles streamflow extraction from both lumped (CSV/NetCDF) and
    distributed (NetCDF for mizuRoute) output formats.

    Attributes:
        model_name: Model identifier for file patterns
        output_file_pattern: Pattern for locating output files
        streamflow_variable: Variable name in NetCDF output
        streamflow_unit: Unit of streamflow in output ('mm_per_day')
    """

    # Model identification
    model_name = "HBV"

    # Output file configuration
    output_file_pattern = "{domain}_hbv_output.nc"

    # NetCDF variable configuration
    streamflow_variable = "streamflow"
    streamflow_unit = "cms"  # Output is already in m³/s from runner

    # Text file configuration (for CSV fallback)
    text_file_separator = ","
    text_file_skiprows = 0
    text_file_date_column = "datetime"
    text_file_flow_column = "streamflow_mm_day"

    # No resampling needed (HBV outputs daily)
    resample_frequency = None

    def _get_output_file(self):
        """
        Get output file path, checking both NetCDF and CSV.

        Returns NetCDF if available, otherwise CSV.
        """
        output_dir = self._get_output_dir()

        # Try NetCDF first
        nc_file = output_dir / self._format_pattern(self.output_file_pattern)
        if nc_file.exists():
            return nc_file

        # Fall back to CSV
        csv_pattern = "{domain}_hbv_output.csv"
        csv_file = output_dir / self._format_pattern(csv_pattern)
        if csv_file.exists():
            return csv_file

        # Return NetCDF path (will show proper error message)
        return nc_file


@ModelRegistry.register_postprocessor('HBV_routed')
class HBVRoutedPostprocessor(StandardModelPostprocessor):
    """
    Postprocessor for routed HBV output (via mizuRoute).

    Handles extraction from mizuRoute routing output files.
    """

    model_name = "HBV_routed"

    # Use routing output
    use_routing_output = True
    routing_variable = "IRFroutedRunoff"
    routing_file_pattern = "{experiment}.h.{start_date}-03600.nc"

    # Routing output is hourly, resample to daily
    resample_frequency = "D"

    # Routing output is already in cms
    streamflow_unit = "cms"
