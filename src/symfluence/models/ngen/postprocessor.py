"""
NGen Model Postprocessor.

Processes simulation outputs from the NOAA NextGen Framework (ngen).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelPostProcessor


@ModelRegistry.register_postprocessor('NGEN')
class NgenPostprocessor(BaseModelPostProcessor):
    """
    Postprocessor for NextGen Framework outputs.
    Handles extraction and analysis of simulation results.
    Inherits common functionality from BaseModelPostProcessor.

    Attributes:
        config (Dict[str, Any]): Configuration settings (inherited)
        logger (Any): Logger instance (inherited)
        project_dir (Path): Project directory path (inherited)
        domain_name (str): Name of the modeling domain (inherited)
        results_dir (Path): Results directory (inherited)
        reporting_manager (Any): Reporting manager instance (inherited)
    """

    def _get_model_name(self) -> str:
        """Return model name for NGEN."""
        return "NGEN"
    
    def extract_streamflow(self, experiment_id: str = None) -> Optional[Path]:
        """
        Extract streamflow from ngen nexus outputs.

        Note: NGEN postprocessor accepts an optional experiment_id parameter,
        which differs from the base class signature. This is necessary to support
        NGEN's multi-experiment workflow.

        Args:
            experiment_id: Experiment identifier (default: from config)

        Returns:
            Path to extracted streamflow CSV file
        """
        self.logger.info("Extracting streamflow from ngen outputs")
        
        if experiment_id is None:
            experiment_id = self.config_dict.get('EXPERIMENT_ID', 'run_1')
        
        # Get output directory
        output_dir = self.project_dir / 'simulations' / experiment_id / 'ngen'
        
        # Find nexus output files
        nexus_files = list(output_dir.glob('nex-*_output.csv'))
        
        if not nexus_files:
            self.logger.error(f"No nexus output files found in {output_dir}")
            return None
            
        # Filter by CALIBRATION_NEXUS_ID if configured
        target_nexus = self.config_dict.get('CALIBRATION_NEXUS_ID')
        if target_nexus:
            # Normalize ID
            target_files = [f for f in nexus_files if f.stem == f"{target_nexus}_output" or f.stem == target_nexus]
            
            if target_files:
                self.logger.info(f"Post-processing restricted to target nexus: {target_nexus}")
                nexus_files = target_files
            else:
                self.logger.warning(f"Configured CALIBRATION_NEXUS_ID '{target_nexus}' not found in output files. Processing all files.")

        self.logger.info(f"Found {len(nexus_files)} nexus output file(s)")
        
        # Read and process each nexus file
        all_streamflow = []
        for nexus_file in nexus_files:
            nexus_id = nexus_file.stem.replace('_output', '')
            
            try:
                # Read nexus output
                # Check if file has header or is headerless (common in NGEN)
                # First try reading first few lines to sniff
                df = pd.read_csv(nexus_file)
                
                # Check for standard NGEN headerless format (index, time, flow)
                is_headerless = False
                if len(df.columns) == 3:
                    # Check if first row looks like it should be part of data (e.g. date in col 1)
                    # or if the current column names are garbage (e.g. '0', '2002...', '20.82')
                    try:
                        # Try parsing the FIRST row's second column as date
                        pd.to_datetime(df.columns[1])
                        # If that worked, the header is actually data.
                        is_headerless = True
                    except (ValueError, TypeError):
                        # Might be a proper header
                        pass

                if is_headerless:
                    # Reload with header=None
                    df = pd.read_csv(nexus_file, header=None, names=['index', 'time', 'flow'])
                    flow_col = 'flow'
                else:
                    # Check for flow column (common names)
                    flow_col = None
                    for col_name in ['flow', 'Flow', 'Q_OUT', 'streamflow', 'discharge']:
                        if col_name in df.columns:
                            flow_col = col_name
                            break
                
                if flow_col is None:
                    self.logger.warning(f"No flow column found in {nexus_file}. Columns: {df.columns.tolist()}")
                    continue
                
                # Extract time and flow
                if 'time' in df.columns:
                    time = pd.to_datetime(df['time'])
                elif 'Time' in df.columns:
                    time = pd.to_datetime(df['Time'], unit='ns')
                else:
                    self.logger.warning(f"No time column found in {nexus_file}")
                    continue
                
                # Create streamflow dataframe
                streamflow_df = pd.DataFrame({
                    'datetime': time,
                    'streamflow_cms': df[flow_col],
                    'nexus_id': nexus_id
                })
                
                all_streamflow.append(streamflow_df)
                
            except Exception as e:
                self.logger.error(f"Error processing {nexus_file}: {e}")
                continue
        
        if not all_streamflow:
            self.logger.error("No streamflow data could be extracted")
            return None
        
        # Combine all nexus outputs
        combined_streamflow = pd.concat(all_streamflow, ignore_index=True)
        
        # Prepare for standard saving: index by datetime, extract streamflow column
        # Assuming we want to save the first nexus or sum? 
        # For standardization, let's assume we are interested in one main outlet or we sum them up?
        # The base `save_streamflow_to_results` expects a Series.
        # If there are multiple nexuses, this might be tricky. 
        # However, typically we look at the outlet. 
        # Let's aggregate by time (summing if multiple outlets? or taking mean?).
        # For now, let's assume one main outlet or aggregate sum.
        aggregated_flow = combined_streamflow.groupby('datetime')['streamflow_cms'].sum()
        
        # Save using standard method
        return self.save_streamflow_to_results(
            aggregated_flow,
            model_column_name=f"NGEN_{experiment_id}_discharge_cms"
        )
    
    
    def _calculate_nse(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency."""
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]
        
        if len(obs) == 0:
            return np.nan
        
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1 - (numerator / denominator)
    
    def _calculate_kge(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Kling-Gupta Efficiency."""
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]
        
        if len(obs) == 0:
            return np.nan
        
        # Calculate components
        r = np.corrcoef(obs, sim)[0, 1]  # Correlation
        alpha = np.std(sim) / np.std(obs)  # Variability ratio
        beta = np.mean(sim) / np.mean(obs)  # Bias ratio
        
        # Calculate KGE
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        return kge
