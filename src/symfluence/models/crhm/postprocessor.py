"""
CRHM model postprocessor.

Handles extraction and processing of CRHM model simulation results.
Uses StandardModelPostprocessor for reduced boilerplate.
"""

from pathlib import Path
from typing import Optional

from ..registry import ModelRegistry
from ..base import StandardModelPostprocessor


@ModelRegistry.register_postprocessor('CRHM')
class CRHMPostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for the CRHM model.

    CRHM outputs streamflow and other variables in CSV format with
    columns for date, flow (m3/s), SWE (mm), soil moisture, etc.
    The flow column typically represents basin outlet discharge.

    Attributes:
        model_name: "CRHM"
        output_file_pattern: "*.csv"
        streamflow_variable: "flow"
        streamflow_unit: "cms"
    """

    model_name = "CRHM"

    output_file_pattern = "*.csv"

    streamflow_variable = "flow"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "CRHM"

    def _setup_model_specific_paths(self) -> None:
        """Set up CRHM-specific paths."""
        self.crhm_output_dir = self.project_dir / 'simulations' / self.experiment_id / 'CRHM'
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """CRHM outputs to standard simulation directory."""
        return self.project_dir / 'simulations' / self.experiment_id / 'CRHM'

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from CRHM CSV outputs.

        CRHM outputs flow directly in m3/s in CSV format.
        Parses the CSV to extract the flow column.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from CRHM outputs")

        output_dir = self._get_output_dir()

        # Find output file
        output_file = None
        for pattern in ['*output*.csv', '*.csv']:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            # Check settings directory
            settings_dir = self.project_dir / 'CRHM_input' / 'settings'
            for pattern in ['*output*.csv', '*.csv']:
                matches = list(settings_dir.glob(pattern))
                if matches:
                    output_file = matches[0]
                    break

        if output_file is None:
            self.logger.error(f"CRHM output not found in {output_dir}")
            return None

        try:
            import pandas as pd

            df = pd.read_csv(output_file, parse_dates=True, index_col=0)

            # Find flow column
            flow_col = None
            for col in ['flow', 'Flow', 'discharge', 'Discharge', 'Q', 'flow_cms']:
                if col in df.columns:
                    flow_col = col
                    break

            if flow_col is None:
                # Try to find any column with 'flow' in the name
                flow_cols = [c for c in df.columns if 'flow' in c.lower()]
                if flow_cols:
                    flow_col = flow_cols[0]

            if flow_col is None:
                self.logger.error("No flow variable found in CRHM output")
                return None

            streamflow = df[flow_col]

            # CRHM outputs flow in m3/s, no conversion needed
            # If units are mm/day, convert
            if streamflow.mean() < 0.001 and streamflow.max() > 0:
                # Likely in mm/day, convert
                streamflow = self.convert_mm_per_day_to_cms(streamflow)

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='CRHM_discharge_cms'
            )

        except Exception as e:
            import traceback
            self.logger.error(f"Error extracting CRHM streamflow: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
