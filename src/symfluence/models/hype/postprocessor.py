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

            # Check if routing was used (mizuRoute)
            experiment_id = self.config_dict.get('EXPERIMENT_ID', 'run_1')
            mizuroute_dir = self.project_dir / "simulations" / experiment_id / "mizuRoute"
            use_routed_output = False
            sim_file_path = None

            if mizuroute_dir.exists():
                # Look for mizuRoute output - usually experiment_id.h.*.nc or similar
                experiment_id = self.config_dict.get('EXPERIMENT_ID', 'run_1')
                mizu_files = list(mizuroute_dir.glob(f"{experiment_id}.h.*.nc"))
                if not mizu_files:
                    mizu_files = list(mizuroute_dir.glob("*.h.*.nc"))
                
                if mizu_files:
                    sim_file_path = mizu_files[0]
                    use_routed_output = True
                    self.logger.info(f"Using mizuRoute output for streamflow extraction: {sim_file_path}")

            if use_routed_output:
                import xarray as xr
                import numpy as np
                with xr.open_dataset(sim_file_path) as ds:
                    # Find routing variable
                    routing_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
                    routing_var = None
                    for v in routing_vars:
                        if v in ds.variables:
                            routing_var = v
                            break
                    
                    if routing_var is None:
                        self.logger.error(f"No routing variable found in {sim_file_path}")
                        return None
                    
                    var = ds[routing_var]
                    
                    # Select outlet (highest mean runoff)
                    if 'seg' in var.dims:
                        seg_means = var.mean(dim='time').values
                        outlet_idx = np.argmax(seg_means)
                        simulated = var.isel(seg=outlet_idx)
                    elif 'reachID' in var.dims:
                        reach_means = var.mean(dim='time').values
                        outlet_idx = np.argmax(reach_means)
                        simulated = var.isel(reachID=outlet_idx)
                    else:
                        simulated = var.isel({var.dims[1]: 0})
                    
                    q_sim = simulated.to_series()
                    # Resample to daily if needed
                    q_sim = q_sim.resample('D').mean()
            else:
                # Fallback to direct HYPE output (timeCOUT.txt)
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
                    # Fallback to first column after index
                    self.logger.warning(f"Outlet ID {outlet_id} not found in columns. Using first column.")
                    q_sim = cout.iloc[:, 0]
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


