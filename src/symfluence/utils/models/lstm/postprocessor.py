"""
LSTM Model Postprocessor.

Handles result saving, visualization, and metric calculation for the LSTM model.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import xarray as xr

from symfluence.utils.common.metrics import (
    get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp
)

class LSTMPostprocessor:
    """
    Handles postprocessing for the LSTM model.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        logger (Any): Logger instance.
        project_dir (Path): Project directory path.
    """

    def __init__(self, config: Dict[str, Any], logger: Any, project_dir: Path, reporting_manager: Any = None):
        self.config = config
        self.config_dict = config
        self.logger = logger
        self.project_dir = project_dir
        self.reporting_manager = reporting_manager

    def save_results(self, results: pd.DataFrame, use_snow: bool):
        """
        Save LSTM model results to disk as NetCDF.
        
        Args:
            results (pd.DataFrame): Simulation results dataframe.
            use_snow (bool): Whether snow was simulated.
        """
        self.logger.info("Saving LSTM model results")

        # Prepare the output directory
        output_dir = self.project_dir / 'simulations' / self.config_dict.get('EXPERIMENT_ID') / 'LSTM'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset dictionary with streamflow
        data_vars = {
            "predicted_streamflow": (["time"], results['predicted_streamflow'].values)
        }

        # Add SWE if enabled in config
        if use_snow and 'predicted_SWE' in results.columns:
            data_vars["scalarSWE"] = (["time"], results['predicted_SWE'].values)

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": results.index
            }
        )

        # Add attributes
        ds.predicted_streamflow.attrs['units'] = 'm3 s-1'
        ds.predicted_streamflow.attrs['long_name'] = 'Routed streamflow'

        if use_snow and 'predicted_SWE' in results.columns:
            ds.scalarSWE.attrs['units'] = 'mm'
            ds.scalarSWE.attrs['long_name'] = 'Snow Water Equivalent'

        # Save as NetCDF
        output_file = output_dir / f"{self.config_dict.get('EXPERIMENT_ID')}_LSTM_output.nc"
        ds.to_netcdf(output_file)

        self.logger.info(f"LSTM results saved to {output_file}")

    def calculate_metrics(self, obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            obs (pd.Series): Observed data.
            sim (pd.Series): Simulated data.
            
        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        # Ensure obs and sim have the same length and are aligned
        aligned_data = pd.concat([obs, sim], axis=1, keys=['obs', 'sim']).dropna()
        obs_vals = aligned_data['obs'].values
        sim_vals = aligned_data['sim'].values

        if len(obs_vals) == 0:
            self.logger.warning("No overlapping data for metric calculation")
            return {}

        return {
            'RMSE': get_RMSE(obs_vals, sim_vals, transfo=1),
            'KGE': get_KGE(obs_vals, sim_vals, transfo=1),
            'KGEp': get_KGEp(obs_vals, sim_vals, transfo=1),
            'NSE': get_NSE(obs_vals, sim_vals, transfo=1),
            'MAE': get_MAE(obs_vals, sim_vals, transfo=1),
            'KGEnp': get_KGEnp(obs_vals, sim_vals, transfo=1)
        }

    def visualize_results(
        self,
        results_df: pd.DataFrame,
        obs_streamflow: pd.DataFrame,
        obs_snow: pd.DataFrame,
        use_snow: bool
    ):
        """
        Visualize simulation results and save plot.
        
        Args:
            results_df (pd.DataFrame): Simulation results.
            obs_streamflow (pd.DataFrame): Observed streamflow.
            obs_snow (pd.DataFrame): Observed snow (can be empty).
            use_snow (bool): Whether snow metrics/plots are required.
        """
        if self.reporting_manager:
            output_dir = self.project_dir / 'plots' / 'results'
            experiment_id = self.config_dict.get('EXPERIMENT_ID')
            self.reporting_manager.visualize_lstm_results(
                results_df, obs_streamflow, obs_snow, use_snow, output_dir, experiment_id
            )
        else:
            self.logger.info("Reporting manager not available, skipping visualization")
