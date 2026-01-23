"""
TRoute Model Optimizer.

Implements the BaseModelOptimizer for the T-Route routing model.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import netCDF4 as nc4
import numpy as np
import xarray as xr

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.models.troute.runner import TRouteRunner
from symfluence.models.troute.preprocessor import TRoutePreProcessor


class TRouteModelOptimizer(BaseModelOptimizer):
    """
    Optimizer for T-Route routing model.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, output_dir: Path, reporting_manager: Optional[Any] = None):
        super().__init__(config, logger, output_dir, reporting_manager)
        self.model_name = "TROUTE"

        # Initialize model components
        self.runner = TRouteRunner(config, logger)
        self.preprocessor = TRoutePreProcessor(config, logger)

    def _update_model_parameters(self, params: Dict[str, float]) -> None:
        """
        Update T-Route configuration/parameters.

        Updates parameters in the T-Route topology file (NetCDF).
        Supports updating Manning's n ('n', 'mannings_n').
        """
        topology_name = self.config_dict.get('SETTINGS_TROUTE_TOPOLOGY', 'troute_topology.nc')
        topology_path = self.preprocessor.setup_dir / topology_name

        if not topology_path.exists():
            self.logger.warning(f"T-Route topology file not found: {topology_path}")
            return

        try:
            with nc4.Dataset(topology_path, 'r+') as ncid:
                updated = False

                # Map common parameter names to NetCDF variables
                param_map = {
                    'n': 'n',
                    'mannings_n': 'n',
                    'roughness': 'n'
                }

                for param, value in params.items():
                    nc_var_name = param_map.get(param.lower())

                    if nc_var_name and nc_var_name in ncid.variables:
                        # Update the variable
                        # Assuming scalar multiplier or global replacement for now
                        # Check if we should treat as multiplier or absolute value
                        # Defaulting to global replacement for scalar optimization

                        var = ncid.variables[nc_var_name]
                        # Apply uniform value across all links
                        var[:] = np.full(var.shape, value)
                        self.logger.debug(f"Updated {nc_var_name} to {value} in {topology_name}")
                        updated = True
                    else:
                        self.logger.warning(f"Parameter {param} (mapped to {nc_var_name}) not found in {topology_name}")

                if not updated:
                    self.logger.warning("No T-Route parameters were updated.")

        except Exception as e:
            self.logger.error(f"Failed to update T-Route parameters: {e}")
            raise

    def _run_model(self) -> bool:
        """Execute the T-Route model."""
        return self.runner.run_troute()

    def _get_simulation_results(self) -> Dict[str, Any]:
        """
        Retrieve simulation results.

        Returns:
            Dictionary of simulation outputs
        """
        self._get_config_value(lambda: self.config.domain.experiment_id, dict_key='EXPERIMENT_ID')
        output_dir = self.runner.get_experiment_output_dir()

        # T-Route output is typically in a channel output file
        # Pattern often matches: chrto_out_*.nc

        files = list(output_dir.glob("*.nc"))
        if not files:
            self.logger.error(f"No T-Route output files found in {output_dir}")
            return {}

        # Sort by modification time to get latest, or name
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        output_file = files[0]

        try:
            ds = xr.open_dataset(output_file)
            results = {}

            # T-Route streamflow variable is usually 'streamflow' or 'velocity'
            # Check variables
            flow_var = 'streamflow'
            if flow_var not in ds and 'velocity' in ds:
                flow_var = 'velocity' # Fallback if flow not found (unlikely for routing)

            if flow_var in ds:
                # Return dataframe
                # Dimensions are typically time x feature_id
                df = ds[flow_var].to_pandas()
                results['streamflow'] = df

            ds.close()
            return results

        except Exception as e:
            self.logger.error(f"Error reading T-Route results: {e}")
            return {}
