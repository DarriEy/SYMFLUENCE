"""
NGen Model Postprocessor.

Processes simulation outputs from the NOAA NextGen Framework (ngen).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.utils.models.registry import ModelRegistry
from symfluence.utils.models.base import BaseModelPostProcessor


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
        
        self.logger.info(f"Found {len(nexus_files)} nexus output file(s)")
        
        # Read and process each nexus file
        all_streamflow = []
        for nexus_file in nexus_files:
            nexus_id = nexus_file.stem.replace('_output', '')
            
            try:
                # Read nexus output
                df = pd.read_csv(nexus_file)
                
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
                if 'Time' in df.columns:
                    time = pd.to_datetime(df['Time'], unit='ns')
                elif 'time' in df.columns:
                    time = pd.to_datetime(df['time'])
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
        
        # Save to results directory
        output_file = self.results_dir / f"ngen_streamflow_{experiment_id}.csv"
        combined_streamflow.to_csv(output_file, index=False)
        
        self.logger.info(f"Extracted streamflow saved to: {output_file}")
        self.logger.info(f"Total timesteps: {len(combined_streamflow)}")
        
        return output_file
    
    def plot_streamflow(self, experiment_id: str = None, observed_file: Path = None) -> Optional[Path]:
        """
        Create streamflow plots comparing simulated and observed (if available).
        
        Args:
            experiment_id: Experiment identifier
            observed_file: Path to observed streamflow CSV file
            
        Returns:
            Path to plot file
        """
        if not self.reporting_manager:
            self.logger.info("Reporting manager not available, skipping visualization")
            return None

        self.logger.info("Creating streamflow plots")
        
        if experiment_id is None:
            experiment_id = self.config_dict.get('EXPERIMENT_ID', 'run_1')
        
        # Get streamflow file
        streamflow_file = self.results_dir / f"ngen_streamflow_{experiment_id}.csv"
        
        if not streamflow_file.exists():
            self.logger.info("Streamflow file not found, extracting first...")
            streamflow_file = self.extract_streamflow(experiment_id)
            if streamflow_file is None:
                return None
        
        # Read simulated streamflow
        sim_df = pd.read_csv(streamflow_file)
        sim_df['datetime'] = pd.to_datetime(sim_df['datetime'])
        
        # Read observed if available
        obs_df = None
        if observed_file and Path(observed_file).exists():
            obs_df = pd.read_csv(observed_file)
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'])
            
        self.reporting_manager.visualize_ngen_results(
            sim_df, obs_df, experiment_id, self.results_dir
        )
        
        return self.results_dir / f"ngen_streamflow_plot_{experiment_id}.png"
    
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
