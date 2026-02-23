"""
MizuRoute Model Optimizer.

Implements the BaseModelOptimizer for the MizuRoute routing model.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import xarray as xr

from symfluence.models.mizuroute.preprocessor import MizuRoutePreProcessor
from symfluence.models.mizuroute.runner import MizuRouteRunner
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer


class MizuRouteModelOptimizer(BaseModelOptimizer):
    """
    Optimizer for MizuRoute routing model.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, output_dir: Path, reporting_manager: Optional[Any] = None):
        super().__init__(config, logger, output_dir, reporting_manager)
        self.model_name = "MIZUROUTE"

        # Initialize model components
        self.runner = MizuRouteRunner(config, logger)
        self.preprocessor = MizuRoutePreProcessor(config, logger)

    def _update_model_parameters(self, params: Dict[str, float]) -> None:
        """
        Update MizuRoute configuration/parameters.

        Updates parameters in the MizuRoute namelist file (param.nml).
        """
        settings_path = self.preprocessor.setup_dir
        param_file_name = self._get_config_value(lambda: self.config.model.mizuroute.parameters, default='param.nml.default', dict_key='SETTINGS_MIZU_PARAMETERS')
        param_file = settings_path / param_file_name

        if not param_file.exists():
            self.logger.warning(f"MizuRoute parameter file not found: {param_file}")
            return

        try:
            with open(param_file, 'r', encoding='utf-8') as f:
                content = f.read()

            for param, value in params.items():
                # Regex to find parameter assignments in namelist (e.g., param = 1.0 or param=1.0)
                # Matches: param_name = value, case insensitive
                pattern = re.compile(fr"(\s*{param}\s*=\s*)([\d\.\-eE]+)(.*)", re.IGNORECASE)

                if pattern.search(content):
                    content = pattern.sub(fr"\g<1>{value}\g<3>", content)
                    self.logger.debug(f"Updated {param} to {value} in {param_file_name}")
                else:
                    self.logger.warning(f"Parameter {param} not found in {param_file_name}")

            with open(param_file, 'w', encoding='utf-8') as f:
                f.write(content)

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Failed to update MizuRoute parameters: {e}")
            raise

    def _run_model(self) -> bool:
        """Execute the MizuRoute model."""
        return self.runner.run_mizuroute()

    def _get_simulation_results(self) -> Dict[str, Any]:
        """
        Retrieve simulation results.

        Returns:
            Dictionary of simulation outputs (e.g., streamflow series)
        """
        experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, dict_key='EXPERIMENT_ID')
        # MizuRoute output directory
        output_dir = self.runner.get_experiment_output_dir()

        # MizuRoute output file pattern (check runner or defaults)
        # Typically <case_name>.q_routed.nc or similar
        # Based on config_manager/runner, it seems to be defined by output settings
        # Default often matches experiment_id

        output_file = output_dir / f"{experiment_id}.h.nc" # Hourly output common default
        if not output_file.exists():
             output_file = output_dir / f"{experiment_id}.d.nc" # Daily fallback

        if not output_file.exists():
             # Try globbing
             files = list(output_dir.glob(f"{experiment_id}*.nc"))
             if files:
                 output_file = files[0]

        if not output_file.exists():
            self.logger.error(f"MizuRoute output file not found in {output_dir}")
            return {}

        try:
            ds = xr.open_dataset(output_file)

            results = {}

            # Common output variables
            # IRF routing: KWTroutedRunoff
            # Grid/Network routing: flow

            # Use configuration to find variable name if possible, else heuristics
            routing_var = self._get_config_value(lambda: self.config.model.mizuroute.output_var, default='KWTroutedRunoff', dict_key='SETTINGS_MIZU_OUTPUT_VAR')

            if routing_var not in ds:
                # Fallbacks
                for var in ['flow', 'droutedRunoff', 'routedRunoff']:
                    if var in ds:
                        routing_var = var
                        break

            if routing_var in ds:
                # Return time series for all segments/HRUs or aggregate
                # For optimization, we typically need a specific gauge
                # Use self.config_dict.get('GAUGE_ID') or similar if implemented
                # For now, return the full dataset as pandas DataFrame (time x segment)

                # Check for segment/hru dimension
                spatial_dim = None
                for dim in ['seg', 'reach', 'hru', 'gru']:
                    if dim in ds.dims:
                        spatial_dim = dim
                        break

                if spatial_dim:
                    # Return mean flow for now if specific gauge logic isn't here
                    # Ideally, OptimizationManager handles gauge extraction from this dict

                    # Convert to pandas dataframe
                    df = ds[routing_var].to_pandas()
                    results['streamflow'] = df
                else:
                    results['streamflow'] = ds[routing_var].to_series()

            ds.close()
            return results

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading MizuRoute results: {e}")
            return {}
