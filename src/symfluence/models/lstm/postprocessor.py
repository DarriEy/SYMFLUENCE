"""
LSTM Model Postprocessor.

Handles result saving, visualization, and metric calculation for the LSTM model.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import xarray as xr

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelPostProcessor
from symfluence.evaluation.metrics import (
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

    def save_results(self, results: pd.DataFrame, use_snow: bool, hru_ids: Optional[List[int]] = None):
        """
        Save LSTM model results to disk as NetCDF and CSV (for standardization).
        
        Args:
            results (pd.DataFrame): Simulation results dataframe.
            use_snow (bool): Whether snow was simulated.
            hru_ids (List[int]): Optional list of HRU IDs for distributed mode.
        """
        self.logger.info("Saving LSTM model results")

        # Prepare the output directory
        self.sim_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.sim_dir / f"{self.experiment_id}_LSTM_output.nc"
        
        # Check if distributed
        is_distributed = hru_ids is not None and len(hru_ids) > 1

        if not is_distributed:
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
            
            # Save Streamflow to Standardized CSV (Triggers Plotting)
            self.save_streamflow_to_results(
                results['predicted_streamflow'],
                model_column_name=f"{self.model_name}_discharge_cms"
            )
        else:
            # Distributed mode: Unflatten results to (time, gru)
            # results index is [time, hruId]
            pivot_df = results.reset_index().pivot(index='time', columns='hruId', values='predicted_streamflow')
            
            # Reorder columns to match hru_ids
            pivot_df = pivot_df[hru_ids]
            
            n_hrus = len(hru_ids)
            n_timesteps = len(pivot_df)
            
            # Create Dataset compatible with mizuRoute/dRoute
            ds = xr.Dataset(
                coords={
                    'time': pivot_df.index,
                    'gru': np.arange(n_hrus)
                }
            )
            
            ds['gruId'] = (['gru'], np.array(hru_ids, dtype='int32'))
            
            # Use same var name as configured for routing if possible
            routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
            if routing_var == 'default':
                routing_var = 'averageRoutedRunoff'
                
            ds[routing_var] = (['time', 'gru'], pivot_df.values)
            ds[routing_var].attrs['units'] = 'm3 s-1' # Default LSTM output is CMS
            ds[routing_var].attrs['long_name'] = 'LSTM runoff'
            
            # Add SWE if enabled
            if use_snow and 'predicted_SWE' in results.columns:
                swe_pivot = results.reset_index().pivot(index='time', columns='hruId', values='predicted_SWE')
                swe_pivot = swe_pivot[hru_ids]
                ds['scalarSWE'] = (['time', 'gru'], swe_pivot.values)
                ds.scalarSWE.attrs['units'] = 'mm'

            # Also save outlet prediction if available (e.g. sum of HRUs or if model predicts at outlet)
            # For now, just save the average as a fallback standardized result
            self.save_streamflow_to_results(
                pivot_df.sum(axis=1), # Simple sum as placeholder
                model_column_name=f"{self.model_name}_sum_discharge_cms"
            )

        # Common attributes
        ds.attrs['model'] = 'LSTM'
        ds.attrs['experiment_id'] = self.experiment_id
        
        # Save as NetCDF
        ds.to_netcdf(output_file)
        self.logger.info(f"LSTM results saved to {output_file}")
        
        # Also produce the standard format file needed by some evaluators
        standard_file = self.sim_dir / f"{self.experiment_id}_timestep.nc"
        if not standard_file.exists():
            import shutil
            shutil.copy2(output_file, standard_file)
            
        return output_file

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
