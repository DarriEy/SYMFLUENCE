"""
MIKE-SHE model postprocessor.

Handles extraction and processing of MIKE-SHE model simulation results.
Uses StandardModelPostprocessor for reduced boilerplate.

MIKE-SHE outputs time series as .dfs0 files or CSV exports.
This postprocessor reads the CSV export format for streamflow extraction.
"""

from pathlib import Path
from typing import Optional

from ..registry import ModelRegistry
from ..base import StandardModelPostprocessor


@ModelRegistry.register_postprocessor('MIKESHE')
class MIKESHEPostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for the MIKE-SHE model.

    MIKE-SHE outputs total discharge as the sum of overland flow,
    drain flow, and baseflow components. Results are typically in
    .dfs0 format or CSV export with units of m3/s.

    Attributes:
        model_name: "MIKESHE"
        output_file_pattern: "*.csv" (CSV export) or "*.dfs0"
        streamflow_variable: "discharge"
        streamflow_unit: "m3_per_s"
    """

    model_name = "MIKESHE"

    output_file_pattern = "*.csv"

    streamflow_variable = "discharge"
    streamflow_unit = "m3_per_s"

    def _get_model_name(self) -> str:
        return "MIKESHE"

    def _setup_model_specific_paths(self) -> None:
        """Set up MIKE-SHE-specific paths."""
        self.mikeshe_output_dir = (
            self.project_dir / 'simulations' / self.experiment_id / 'MIKESHE'
        )
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """MIKE-SHE outputs to standard simulation directory."""
        return self.project_dir / 'simulations' / self.experiment_id / 'MIKESHE'

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from MIKE-SHE outputs.

        Reads CSV export files containing discharge time series.
        MIKE-SHE discharge is typically already in m3/s.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from MIKE-SHE outputs")

        output_dir = self._get_output_dir()

        # Find output file
        output_file = None
        for pattern in ['*discharge*.csv', '*flow*.csv', '*.csv', '*.dfs0']:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            self.logger.error(f"MIKE-SHE output not found in {output_dir}")
            return None

        try:
            import pandas as pd

            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0])

                # Identify the datetime column
                datetime_col = df.columns[0]
                df.set_index(datetime_col, inplace=True)

                # Find discharge column
                discharge_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if any(
                        kw in col_lower
                        for kw in ['discharge', 'flow', 'runoff', 'q_total']
                    ):
                        discharge_col = col
                        break

                if discharge_col is None:
                    # Use first numeric column
                    numeric_cols = df.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        discharge_col = numeric_cols[0]
                    else:
                        self.logger.error(
                            "No numeric discharge column found in MIKE-SHE output"
                        )
                        return None

                streamflow = df[discharge_col]
                streamflow.name = 'MIKESHE_discharge_cms'

            else:
                # dfs0 format - attempt basic text parsing
                self.logger.warning(
                    "dfs0 format requires DHI Python tools; "
                    "attempting text-based parsing"
                )
                df = pd.read_csv(output_file, sep=r'\s+', parse_dates=[0])
                datetime_col = df.columns[0]
                df.set_index(datetime_col, inplace=True)
                streamflow = df.iloc[:, 0]
                streamflow.name = 'MIKESHE_discharge_cms'

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='MIKESHE_discharge_cms'
            )

        except Exception as e:
            import traceback
            self.logger.error(f"Error extracting MIKE-SHE streamflow: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
