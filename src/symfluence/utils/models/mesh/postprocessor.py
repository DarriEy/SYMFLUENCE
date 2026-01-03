"""
MESH model postprocessor.

Handles extraction and processing of MESH model simulation results.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ..registry import ModelRegistry
from ..base import BaseModelPostProcessor


@ModelRegistry.register_postprocessor('MESH')
class MESHPostProcessor(BaseModelPostProcessor):
    """
    Postprocessor for the MESH model.

    Handles extraction and processing of MESH model simulation results.
    """

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "MESH"

    def _setup_model_specific_paths(self) -> None:
        """Set up MESH-specific paths."""
        self.mesh_setup_dir = self.project_dir / "settings" / "MESH"
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_mesh_path = self.project_dir / 'forcing' / 'MESH_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config_dict.get('DOMAIN_DISCRETIZATION')}.shp"

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from MESH outputs.

        MESH outputs streamflow to MESH_output_streamflow.csv with columns:
        DAY, YEAR, QOMEAS1, QOSIM1, QOMEAS2, QOSIM2, ...

        Returns:
            Optional[Path]: Path to processed streamflow file, or None if extraction fails
        """
        self.logger.info("Extracting streamflow from MESH outputs")

        # MESH outputs to the forcing directory where it runs
        mesh_output_file = self.forcing_mesh_path / 'MESH_output_streamflow.csv'

        if not mesh_output_file.exists():
            # Try alternative timestep output
            mesh_output_file = self.forcing_mesh_path / 'MESH_output_streamflow_ts.csv'
            if not mesh_output_file.exists():
                self.logger.error(f"MESH streamflow output not found at {mesh_output_file}")
                return None

        try:
            # Read the CSV file
            df = pd.read_csv(mesh_output_file, skipinitialspace=True)

            # Convert DAY and YEAR to datetime
            # DAY is Julian day (day of year), YEAR is the year
            def julian_to_datetime(row):
                return datetime(int(row['YEAR']), 1, 1) + timedelta(days=int(row['DAY']) - 1)

            df['datetime'] = df.apply(julian_to_datetime, axis=1)

            # Extract simulated streamflow (QOSIM1 is the first simulated column)
            # MESH format: DAY, YEAR, QOMEAS1, QOSIM1, QOMEAS2, QOSIM2, ...
            streamflow_columns = [col for col in df.columns if col.startswith('QOSIM')]

            if not streamflow_columns:
                self.logger.error("No simulated streamflow columns (QOSIM*) found in MESH output")
                return None

            # Use the first simulated column (basin outlet)
            simulated_col = streamflow_columns[0]
            self.logger.info(f"Extracting streamflow from column: {simulated_col}")

            # Create output DataFrame with datetime and streamflow
            output_df = pd.DataFrame({
                'datetime': df['datetime'],
                'streamflow_m3s': df[simulated_col]
            })

            # Save to output directory
            output_dir = self.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / 'streamflow.csv'

            output_df.to_csv(output_file, index=False)
            self.logger.info(f"Streamflow extracted to {output_file}")

            return output_file

        except Exception as e:
            self.logger.error(f"Error extracting MESH streamflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
