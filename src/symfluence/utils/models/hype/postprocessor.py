"""
HYPE model postprocessor.

Handles output extraction, processing, and analysis for HYPE model outputs.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd  # type: ignore

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
        reporting_manager (Any): Reporting manager instance (inherited)
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
            outlet_id = str(self.config_dict.get('SIM_REACH_ID'))
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


