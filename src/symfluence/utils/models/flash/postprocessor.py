"""
FLASH model postprocessor.

Handles extraction, processing, and saving of FLASH simulation results.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import xarray as xr  # type: ignore

from ..registry import ModelRegistry
from ..base import BaseModelPostProcessor


@ModelRegistry.register_postprocessor('FLASH')
class FLASHPostProcessor(BaseModelPostProcessor):
    """
    Postprocessor for FLASH model outputs.
    Handles extraction, processing, and saving of simulation results.

    Attributes:
        config (Dict[str, Any]): Configuration settings for FLASH
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "FLASH"

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from FLASH output and save to CSV.

        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Extracting FLASH streamflow results")

            # Define paths
            sim_path = self.sim_dir / f'{self.experiment_id}_FLASH_output.nc'

            # Read simulation results
            ds = xr.open_dataset(sim_path)

            # Extract streamflow
            q_sim = ds['predicted_streamflow'].to_pandas()

            # Use inherited helper to save results
            return self.save_streamflow_to_results(q_sim)

        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            raise

    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Helper method to get file paths from config or defaults."""
        if self.config.get(file_type) == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(file_type))
