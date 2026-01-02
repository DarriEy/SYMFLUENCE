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
            experiment_id = self.config.get('EXPERIMENT_ID', 'run_1')
        
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
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        self.logger.info("Creating streamflow plots")
        
        if experiment_id is None:
            experiment_id = self.config.get('EXPERIMENT_ID', 'run_1')
        
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
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Full time series
        ax1 = axes[0]
        ax1.plot(sim_df['datetime'], sim_df['streamflow_cms'], 
                label='NGEN Simulated', color='blue', linewidth=0.8)
        
        # Add observed if available
        if observed_file and Path(observed_file).exists():
            obs_df = pd.read_csv(observed_file)
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'])
            ax1.plot(obs_df['datetime'], obs_df['streamflow_cms'], 
                    label='Observed', color='red', linewidth=0.8, alpha=0.7)
            
            # Calculate metrics
            merged = pd.merge(sim_df, obs_df, on='datetime', suffixes=('_sim', '_obs'))
            nse = self._calculate_nse(merged['streamflow_cms_obs'], merged['streamflow_cms_sim'])
            kge = self._calculate_kge(merged['streamflow_cms_obs'], merged['streamflow_cms_sim'])
            
            ax1.text(0.02, 0.98, f'NSE: {nse:.3f}\nKGE: {kge:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_ylabel('Streamflow (cms)', fontsize=12)
        ax1.set_title(f'NGEN Streamflow - {experiment_id}', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Plot 2: Flow duration curve
        ax2 = axes[1]
        sorted_flow = np.sort(sim_df['streamflow_cms'].values)[::-1]
        exceedance = np.arange(1, len(sorted_flow) + 1) / len(sorted_flow) * 100
        ax2.semilogy(exceedance, sorted_flow, label='NGEN Simulated', color='blue', linewidth=1.5)
        
        if observed_file and Path(observed_file).exists():
            sorted_obs = np.sort(obs_df['streamflow_cms'].values)[::-1]
            exceedance_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs) * 100
            ax2.semilogy(exceedance_obs, sorted_obs, label='Observed', 
                        color='red', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Exceedance Probability (%)', fontsize=12)
        ax2.set_ylabel('Streamflow (cms)', fontsize=12)
        ax2.set_title('Flow Duration Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"ngen_streamflow_plot_{experiment_id}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Streamflow plot saved to: {plot_file}")
        
        return plot_file
    
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
