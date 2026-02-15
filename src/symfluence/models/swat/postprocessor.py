"""
SWAT model postprocessor.

Handles extraction and processing of SWAT model simulation results.
Uses StandardModelPostprocessor for reduced boilerplate.

SWAT outputs streamflow in output.rch as fixed-width text with columns
including RCH, GIS, MON, AREAkm2, FLOW_OUTcms, etc.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..registry import ModelRegistry
from ..base import StandardModelPostprocessor


@ModelRegistry.register_postprocessor('SWAT')
class SWATPostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for the SWAT model.

    SWAT outputs streamflow in output.rch as fixed-width text. The key
    column is FLOW_OUTcms which contains discharge in m3/s. The output
    must be parsed from the fixed-width format and filtered by reach number.

    Attributes:
        model_name: "SWAT"
        output_file_pattern: "output.rch"
        streamflow_variable: "FLOW_OUTcms"
        streamflow_unit: "cms"
    """

    model_name = "SWAT"

    output_file_pattern = "output.rch"

    streamflow_variable = "FLOW_OUTcms"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "SWAT"

    def _setup_model_specific_paths(self) -> None:
        """Set up SWAT-specific paths."""
        txtinout_name = 'TxtInOut'
        try:
            txtinout_name = self.config.model.swat.txtinout_dir or txtinout_name
        except (AttributeError, TypeError):
            pass
        self.swat_output_dir = self.project_dir / 'SWAT_input' / txtinout_name
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """SWAT outputs to TxtInOut directory."""
        return self.swat_output_dir

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from SWAT output.rch.

        Parses the fixed-width output.rch file, extracts FLOW_OUTcms
        for the outlet reach, and saves as a CSV time series.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from SWAT output.rch")

        output_dir = self._get_output_dir()
        output_rch = output_dir / 'output.rch'

        if not output_rch.exists():
            self.logger.error(f"SWAT output.rch not found in {output_dir}")
            return None

        try:
            # Parse output.rch
            streamflow = self._parse_output_rch(output_rch)

            if streamflow is None or len(streamflow) == 0:
                self.logger.error("No streamflow data extracted from output.rch")
                return None

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='SWAT_discharge_cms'
            )

        except Exception as e:
            import traceback
            self.logger.error(f"Error extracting SWAT streamflow: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

    def _parse_output_rch(
        self,
        output_rch: Path,
        reach_id: int = 1
    ) -> Optional[pd.Series]:
        """
        Parse SWAT output.rch fixed-width text file.

        The output.rch format has a 9-line header followed by data rows with:
        - Column 1-5: REACH (RCH)
        - Column 6-10: GIS code
        - Column 11-16: MON (day/month/year code)
        - Column 17-26: AREAkm2
        - Column 27-36: FLOW_INcms
        - Column 37-46: FLOW_OUTcms
        - ... additional columns

        Args:
            output_rch: Path to output.rch file
            reach_id: Reach number to extract (default: 1, the outlet)

        Returns:
            pandas Series of streamflow (m3/s) indexed by date, or None
        """
        try:
            with open(output_rch, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Skip header lines (typically 9 lines)
            header_end = 0
            for i, line in enumerate(lines):
                if 'REACH' in line.upper() or 'RCH' in line.upper():
                    header_end = i + 1
                    break
            if header_end == 0:
                header_end = 9  # Default SWAT header length

            data_lines = lines[header_end:]

            # Parse data
            records = []
            for line in data_lines:
                line = line.rstrip('\n')
                if len(line) < 46:
                    continue

                try:
                    # SWAT output.rch fixed-width format
                    # Columns vary by SWAT version, use flexible parsing
                    parts = line.split()
                    if len(parts) < 6:
                        continue

                    rch = int(parts[0])
                    # GIS = parts[1]
                    mon = int(parts[2])
                    area_km2 = float(parts[3])
                    flow_out = float(parts[5])

                    if rch == reach_id:
                        records.append({
                            'rch': rch,
                            'mon': mon,
                            'area_km2': area_km2,
                            'flow_out_cms': flow_out,
                        })
                except (ValueError, IndexError):
                    continue

            if not records:
                self.logger.warning(f"No records found for reach {reach_id}")
                return None

            # Build time series
            # MON encoding: for daily output, MON = julian day within each year
            # For monthly output, MON = month number
            # We need simulation start date from config
            try:
                start_str = self.config.domain.time_start
                start_date = pd.to_datetime(start_str)
            except (AttributeError, TypeError):
                start_date = pd.to_datetime('2000-01-01')

            # Determine output frequency
            n_records = len(records)
            flow_values = [r['flow_out_cms'] for r in records]

            # Assume daily output and build date index
            dates = pd.date_range(start=start_date, periods=n_records, freq='D')

            streamflow = pd.Series(flow_values, index=dates, name='SWAT_discharge_cms')
            streamflow = streamflow.clip(lower=0.0)

            self.logger.info(
                f"Extracted {len(streamflow)} streamflow records from output.rch "
                f"(reach {reach_id})"
            )
            return streamflow

        except Exception as e:
            self.logger.error(f"Error parsing output.rch: {e}")
            return None
