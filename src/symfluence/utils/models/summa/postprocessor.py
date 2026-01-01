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


class SUMMAPostprocessor:
    """
    Postprocessor for SUMMA model outputs via MizuRoute routing.
    Handles extraction and processing of simulation results.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_streamflow(self) -> Optional[Path]:
        """Extract streamflow from MizuRoute outputs for spatial mode."""
        self.logger.info("Extracting SUMMA/MizuRoute streamflow results")
        try:
            self.logger.info("Extracting SUMMA/MizuRoute streamflow results")

            # Get simulation output path
            if self.config.get('SIMULATIONS_PATH') == 'default':
                # Parse the start time and extract the date portion
                start_date = self.config.get('EXPERIMENT_TIME_START').split()[0]  # Gets '2011-01-01' from '2011-01-01 01:00'
                sim_file_path = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'mizuRoute' / f"{self.config.get('EXPERIMENT_ID')}.h.{start_date}-03600.nc"
            else:
                sim_file_path = Path(self.config.get('SIMULATIONS_PATH'))

            if not sim_file_path.exists():
                self.logger.error(f"SUMMA/MizuRoute output file not found at: {sim_file_path}")
                return None

            # Get simulation reach ID
            sim_reach_ID = self.config.get('SIM_REACH_ID')

            # Read simulation data
            ds = xr.open_dataset(sim_file_path, engine='netcdf4')

            # Extract data for the specific reach
            segment_index = ds['reachID'].values == int(sim_reach_ID)
            sim_df = ds.sel(seg=segment_index)
            q_sim = sim_df['IRFroutedRunoff'].to_dataframe().reset_index()
            q_sim.set_index('time', inplace=True)
            q_sim.index = q_sim.index.round(freq='h')

            # Convert from hourly to daily average
            q_sim_daily = q_sim['IRFroutedRunoff'].resample('D').mean()

            # Read existing results file if it exists
            output_file = self.results_dir / f"{self.config.get('EXPERIMENT_ID')}_results.csv"
            if output_file.exists():
                results_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
            else:
                results_df = pd.DataFrame(index=q_sim_daily.index)

            # Add SUMMA results
            results_df['SUMMA_discharge_cms'] = q_sim_daily

            # Save updated results
            results_df.to_csv(output_file)

            return output_file

        except Exception as e:
            self.logger.error(f"Error extracting SUMMA streamflow: {str(e)}")
            raise
