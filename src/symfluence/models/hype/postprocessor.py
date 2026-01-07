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
        Extract simulated streamflow from HYPE or mizuRoute output and save to CSV.

        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Processing HYPE streamflow results")

            # NOTE: For HYPE, we always use direct output (timeCOUT.txt) rather than
            # mizuRoute. This is because HYPE's timeCOUT.txt already contains correctly
            # routed/accumulated discharge at each subbasin outlet. mizuRoute expects
            # local runoff per HRU (not accumulated discharge), so using it with HYPE
            # output gives incorrect results.
            #
            # Read directly from HYPE output (timeCOUT.txt)
            cout_path = self.sim_dir / "timeCOUT.txt"
            if not cout_path.exists():
                self.logger.error(f"HYPE output file not found: {cout_path}")
                return None

            self.logger.info(f"Reading HYPE output from: {cout_path}")
            cout = pd.read_csv(cout_path, sep='\t', skiprows=lambda x: x == 0, parse_dates=['DATE'])
            cout.set_index('DATE', inplace=True)

            # Extract outlet discharge
            outlet_id = str(self.config_dict.get('SIM_REACH_ID'))
            self.logger.info(f"Processing outlet ID: {outlet_id}")

            if outlet_id not in cout.columns:
                # Auto-select outlet: column with highest mean flow (downstream outlet)
                import numpy as np
                col_means = cout.mean()
                outlet_col = col_means.idxmax()
                self.logger.warning(
                    f"Outlet ID '{outlet_id}' not found in columns. "
                    f"Auto-selecting outlet column '{outlet_col}' (highest mean flow: {col_means.max():.2f} cms)"
                )
                q_sim = cout[outlet_col]
            else:
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


