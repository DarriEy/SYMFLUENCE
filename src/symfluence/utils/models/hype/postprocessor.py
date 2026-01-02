"""
HYPE model postprocessor.

Handles output extraction, processing, and analysis for HYPE model outputs.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from ..registry import ModelRegistry
from ..base import BaseModelPostProcessor


@ModelRegistry.register_postprocessor('HYPE')
class HYPEPostProcessor(BaseModelPostProcessor):
    """
    Postprocessor for HYPE model outputs within SYMFLUENCE.
    Handles output extraction, processing, and analysis.
    Inherits common functionality from BaseModelPostProcessor.

    Attributes:
        config (Dict[str, Any]): Configuration settings (inherited)
        logger (logging.Logger): Logger instance (inherited)
        project_dir (Path): Project directory path (inherited)
        domain_name (str): Name of the modeling domain (inherited)
        sim_dir (Path): HYPE simulation output directory
        results_dir (Path): Results directory (inherited)
    """

    def _get_model_name(self) -> str:
        """Return model name for HYPE."""
        return "HYPE"

    def extract_results(self) -> Dict[str, Path]:
        """
        Extract and process all HYPE results.

        Returns:
            Dict[str, Path]: Paths to processed result files
        """
        self.logger.info("Extracting HYPE results")

        try:
            # Process streamflow
            self.extract_streamflow()
            self.logger.info("Streamflow extracted successfully")

            self.plot_streamflow_comparison()
            self.logger.info("Streamflow comparison plot created successfully")

        except Exception as e:
            self.logger.error(f"Error extracting HYPE results: {str(e)}")
            raise

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from HYPE output and save to CSV.
        Reads timeCOUT.txt file (HYPE-specific format) and extracts outlet discharge.

        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Processing HYPE streamflow results for outlet")

            # Read HYPE timeCOUT.txt output (HYPE-specific format)
            cout_path = self.sim_dir / "timeCOUT.txt"
            self.logger.info(f"Reading HYPE output from: {cout_path}")

            cout = pd.read_csv(cout_path, sep='\t', skiprows=lambda x: x == 0, parse_dates=['DATE'])
            cout.set_index('DATE', inplace=True)

            # Extract outlet discharge
            outlet_id = str(self.config.get('SIM_REACH_ID'))
            self.logger.info(f"Processing outlet ID: {outlet_id}")

            if outlet_id not in cout.columns:
                self.logger.error(f"Outlet ID {outlet_id} not found in columns: {cout.columns.tolist()}")
                raise KeyError(f"Outlet ID {outlet_id} not found in HYPE output")

            # Extract streamflow series for outlet
            q_sim = cout[outlet_id]

            # Use inherited save method
            return self.save_streamflow_to_results(
                q_sim,
                model_column_name='HYPE_discharge_cms'
            )

        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            self.logger.exception("Full traceback:")
            return None

    def plot_streamflow_comparison(self) -> Optional[Path]:
        """
        Create a comparison plot of simulated vs observed streamflow.

        Returns:
            Optional[Path]: Path to the saved plot if successful, None otherwise
        """
        try:
            self.logger.info("Creating streamflow comparison plot")

            # Read simulated streamflow
            sim_path = self.results_dir / f"{self.config.get('EXPERIMENT_ID')}_streamflow.csv"
            self.logger.info(f"Reading simulated streamflow from: {sim_path}")

            # Add explicit time parsing
            sim_flow = pd.read_csv(sim_path)
            self.logger.info("Original sim_flow columns: " + str(sim_flow.columns.tolist()))

            # Convert the first column to datetime index
            time_col = sim_flow.columns[0]  # Get the name of the first column
            self.logger.info(f"Converting time column: {time_col}")
            sim_flow[time_col] = pd.to_datetime(sim_flow[time_col])
            sim_flow.set_index(time_col, inplace=True)

            self.logger.info(f"Simulated flow DataFrame shape: {sim_flow.shape}")
            self.logger.info(f"Simulated flow columns: {sim_flow.columns.tolist()}")
            self.logger.info(f"Simulated flow index type: {type(sim_flow.index)}")

            # Read observed streamflow
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
            self.logger.info(f"Reading observed streamflow from: {obs_path}")

            # Add explicit datetime parsing for observed data
            obs_flow = pd.read_csv(obs_path)
            obs_flow['datetime'] = pd.to_datetime(obs_flow['datetime'])
            obs_flow.set_index('datetime', inplace=True)

            self.logger.info(f"Observed flow DataFrame shape: {obs_flow.shape}")
            self.logger.info(f"Observed flow columns: {obs_flow.columns.tolist()}")
            self.logger.info(f"Observed flow index type: {type(obs_flow.index)}")

            # Get outlet ID
            outlet_id = str(self.config.get('SIM_REACH_ID'))
            self.logger.info(f"Processing outlet ID: {outlet_id}")

            sim_col = 'HYPE_discharge_cms'
            self.logger.info(f"Looking for simulation column: {sim_col}")

            if sim_col not in sim_flow.columns:
                self.logger.error(f"Column {sim_col} not found in simulated flow columns: {sim_flow.columns.tolist()}")
                raise KeyError(f"Column {sim_col} not found in simulated flow data")

            if 'discharge_cms' not in obs_flow.columns:
                self.logger.error(f"Column 'discharge_cms' not found in observed flow columns: {obs_flow.columns.tolist()}")
                raise KeyError("Column 'discharge_cms' not found in observed flow data")

            # Create figure
            plt.figure(figsize=(12, 6))
            plt.plot(sim_flow.index, sim_flow[sim_col], label='Simulated', color='blue', alpha=0.7)
            plt.plot(obs_flow.index, obs_flow['discharge_cms'], label='Observed', color='red', alpha=0.7)

            plt.title(f'Streamflow Comparison - {self.domain_name}\nOutlet ID: {outlet_id}')
            plt.xlabel('Date')
            plt.ylabel('Discharge (m³/s)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Ensure the plots directory exists
            plot_dir = self.project_dir / "plots" / "results"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Save plot
            plot_path = plot_dir / f"{self.config.get('EXPERIMENT_ID')}_HYPE_streamflow_comparison.png"
            self.logger.info(f"Saving plot to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            return plot_path

        except Exception as e:
            self.logger.error(f"Error creating streamflow comparison plot: {str(e)}")
            self.logger.exception("Full traceback:")
            return None
