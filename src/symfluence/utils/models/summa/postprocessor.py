"""
SUMMA postprocessor module.

Handles extraction and processing of SUMMA model simulation results,
typically via MizuRoute routing outputs.
"""

# Standard library imports
from pathlib import Path
from typing import Dict, Any, Optional

# Third-party imports
import pandas as pd  # type: ignore
import xarray as xr  # type: ignore

# SYMFLUENCE imports
from ..base import BaseModelPostProcessor


class SUMMAPostprocessor(BaseModelPostProcessor):
    """
    Postprocessor for SUMMA model outputs via MizuRoute routing.
    Handles extraction and processing of simulation results.
    """

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "SUMMA"

    def _setup_model_specific_paths(self) -> None:
        """Set up SUMMA-specific paths for MizuRoute outputs."""
        self.mizuroute_dir = self.project_dir / 'simulations' / self.experiment_id / 'mizuRoute'

    def extract_streamflow(self) -> Optional[Path]:
        """Extract streamflow from MizuRoute outputs for spatial mode."""
        self.logger.info("Extracting SUMMA/MizuRoute streamflow results")
        try:
            # Get simulation output path
            if self.config.get('SIMULATIONS_PATH') == 'default':
                # Parse the start time and extract the date portion
                start_date = self.config.get('EXPERIMENT_TIME_START').split()[0]  # Gets '2011-01-01' from '2011-01-01 01:00'
                sim_file_path = self.mizuroute_dir / f"{self.experiment_id}.h.{start_date}-03600.nc"
            else:
                sim_file_path = Path(self.config.get('SIMULATIONS_PATH'))

            if not sim_file_path.exists():
                self.logger.error(f"SUMMA/MizuRoute output file not found at: {sim_file_path}")
                return None

            # Get simulation reach ID
            sim_reach_ID = int(self.config.get('SIM_REACH_ID'))

            # Read simulation data
            ds = xr.open_dataset(sim_file_path, engine='netcdf4')

            # Extract data for the specific reach
            segment_index = ds['reachID'].values == sim_reach_ID
            sim_df = ds.sel(seg=segment_index)
            q_sim = sim_df['IRFroutedRunoff'].to_dataframe().reset_index()
            q_sim.set_index('time', inplace=True)
            q_sim.index = q_sim.index.round(freq='h')

            # Convert from hourly to daily average
            q_sim_daily = q_sim['IRFroutedRunoff'].resample('D').mean()

            # Use inherited helper to save results
            return self.save_streamflow_to_results(q_sim_daily)

        except Exception as e:
            self.logger.error(f"Error extracting SUMMA streamflow: {str(e)}")
            raise
