# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HEC-HMS Model Postprocessor.

Extracts and processes HEC-HMS model output for analysis and visualization.
Uses StandardModelPostprocessor for minimal boilerplate.
"""

from symfluence.models.base.standard_postprocessor import StandardModelPostprocessor
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_postprocessor('HECHMS')
class HecHmsPostprocessor(StandardModelPostprocessor):
    """
    Postprocessor for HEC-HMS model output.

    Handles streamflow extraction from lumped CSV/NetCDF output formats.
    """

    # Model identification
    model_name = "HECHMS"

    # Output file configuration
    output_file_pattern = "{domain}_hechms_output.nc"

    # NetCDF variable configuration
    streamflow_variable = "streamflow"
    streamflow_unit = "cms"

    # Text file configuration (for CSV fallback)
    text_file_separator = ","
    text_file_skiprows = 0
    text_file_date_column = "datetime"
    text_file_flow_column = "streamflow_mm_day"

    # No resampling needed (HEC-HMS outputs daily)
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
        csv_pattern = "{domain}_hechms_output.csv"
        csv_file = output_dir / self._format_pattern(csv_pattern)
        if csv_file.exists():
            return csv_file

        return nc_file
