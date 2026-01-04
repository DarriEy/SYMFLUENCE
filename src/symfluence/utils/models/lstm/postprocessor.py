"""
LSTM Model Postprocessor.

Handles result saving, visualization, and metric calculation for the LSTM model.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import xarray as xr

from symfluence.utils.models.registry import ModelRegistry
from symfluence.utils.models.base import BaseModelPostProcessor
from symfluence.utils.evaluation.metrics import (
    kge, kge_prime, nse, mae, rmse, kge_np
)

@ModelRegistry.register_postprocessor('LSTM')
class LSTMPostprocessor(BaseModelPostProcessor):
    """
    Handles postprocessing for the LSTM model.
    Inherits from BaseModelPostProcessor to use standardized results saving and plotting.
    """

    def _get_model_name(self) -> str:
        return "LSTM"

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from LSTM output.
        
        For LSTM, results are typically passed directly to save_results by the runner.
        This method is implemented to satisfy the abstract base class and can be used
        if we need to reload from the NetCDF file.
        """
        try:
            output_file = self.sim_dir / f"{self.experiment_id}_LSTM_output.nc"
            if not output_file.exists():
                return None
            
            ds = xr.open_dataset(output_file)
            if 'predicted_streamflow' in ds:
                return self.save_streamflow_to_results(ds['predicted_streamflow'].to_series())
            return None
        except Exception as e:
            self.logger.error(f"Error extracting LSTM streamflow: {str(e)}")
            return None

    def save_results(self, results: pd.DataFrame, use_snow: bool):
        """
        Save LSTM model results to disk as NetCDF and CSV (for standardization).
        
        Args:
            results (pd.DataFrame): Simulation results dataframe.
            use_snow (bool): Whether snow was simulated.
        """
        self.logger.info("Saving LSTM model results")

        # Prepare the output directory
        self.sim_dir.mkdir(parents=True, exist_ok=True)

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

        # Save as NetCDF (Original LSTM format)
        output_file = self.sim_dir / f"{self.experiment_id}_LSTM_output.nc"
        ds.to_netcdf(output_file)
        self.logger.info(f"LSTM results saved to {output_file}")

        # Save Streamflow to Standardized CSV (Triggers Plotting)
        # We save specifically the streamflow series
        self.save_streamflow_to_results(
            results['predicted_streamflow'],
            model_column_name=f"{self.model_name}_discharge_cms"
        )

    def calculate_metrics(self, obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics.
        """
        # Ensure obs and sim have the same length and are aligned
        aligned_data = pd.concat([obs, sim], axis=1, keys=['obs', 'sim']).dropna()
        obs_vals = aligned_data['obs'].values
        sim_vals = aligned_data['sim'].values

        if len(obs_vals) == 0:
            self.logger.warning("No overlapping data for metric calculation")
            return {}

        return {
            'RMSE': rmse(obs_vals, sim_vals, transfo=1),
            'KGE': kge(obs_vals, sim_vals, transfo=1),
            'KGEp': kge_prime(obs_vals, sim_vals, transfo=1),
            'NSE': nse(obs_vals, sim_vals, transfo=1),
            'MAE': mae(obs_vals, sim_vals, transfo=1),
            'KGEnp': kge_np(obs_vals, sim_vals, transfo=1)
        }
