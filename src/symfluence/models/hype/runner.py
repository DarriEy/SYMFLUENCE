"""
HYPE model runner.

Handles HYPE model execution and run-time management.
Refactored to use the Unified Model Execution Framework.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import subprocess
import shutil

from ..registry import ModelRegistry
from ..base import BaseModelRunner
from ..execution import ModelExecutor, SpatialOrchestrator
from symfluence.core.exceptions import ModelExecutionError


@ModelRegistry.register_runner('HYPE', method_name='run_hype')
class HYPERunner(BaseModelRunner, ModelExecutor, SpatialOrchestrator):
    """
    Runner class for the HYPE model within SYMFLUENCE.
    Handles model execution and run-time management.

    Uses the Unified Model Execution Framework for subprocess execution.

    Attributes:
        config (Dict[str, Any]): Configuration settings
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the modeling domain
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up HYPE-specific paths."""
        self.setup_dir = self.project_dir / "settings" / "HYPE"

        # HYPE executable path (check multiple standard locations)
        try:
            self.hype_exe = self.get_model_executable(
                install_path_key='HYPE_INSTALL_PATH',
                default_install_subpath='installs/hype/bin',
                exe_name_key='HYPE_EXE',
                default_exe_name='hype',
                typed_exe_accessor=lambda: self.typed_config.model.hype.exe if (self.typed_config and self.typed_config.model.hype) else None,
                must_exist=True
            )
        except FileNotFoundError:
            # Try alternate location without 'bin'
            self.hype_exe = self.get_model_executable(
                install_path_key='HYPE_INSTALL_PATH',
                default_install_subpath='installs/hype',
                exe_name_key='HYPE_EXE',
                default_exe_name='hype',
                typed_exe_accessor=lambda: self.typed_config.model.hype.exe if (self.typed_config and self.typed_config.model.hype) else None,
                must_exist=True
            )

    def _get_model_name(self) -> str:
        """Return model name for HYPE."""
        return "HYPE"

    def _get_output_dir(self) -> Path:
        """HYPE uses custom output path resolution."""
        if self.config:
            experiment_id = self.config.domain.experiment_id
        else:
            experiment_id = self.config_dict.get('EXPERIMENT_ID')
        return self.get_config_path('EXPERIMENT_OUTPUT_HYPE', f"simulations/{experiment_id}/HYPE")

    def run_hype(self) -> Optional[Path]:
        """
        Run the HYPE model simulation.

        Returns:
            Optional[Path]: Path to output directory if successful, None otherwise
        """
        self.logger.debug("Starting HYPE model run")

        try:
            # Ensure setup directory exists
            if not self.setup_dir.exists():
                self.logger.error(f"HYPE setup directory not found: {self.setup_dir}")
                return None

            # Create run command
            cmd = self._create_run_command()

            # Set up logging
            log_dir = self.get_log_path()
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f'hype_run_{current_time}.log'

            # Execute HYPE
            self.logger.debug(f"Executing command: {' '.join(map(str, cmd))}")

            result = self.execute_model_subprocess(
                cmd,
                log_file,
                cwd=self.setup_dir,
                check=False,  # Don't raise on non-zero exit, we'll handle it
                success_message="HYPE simulation completed successfully"
            )

            # Check execution success
            if result.returncode == 0 and self._verify_outputs():
                # Handle routing if needed
                routing_model = self.config_dict.get('ROUTING_MODEL', 'none')
                if routing_model == 'mizuRoute':
                    self.logger.info("Starting mizuRoute routing for HYPE")
                    if self._convert_hype_to_mizuroute_format():
                        success = self._run_distributed_routing()
                        if not success:
                            self.logger.error("HYPE routing failed")
                            raise ModelExecutionError("HYPE routing failed")
                    else:
                        self.logger.error("Failed to convert HYPE output to mizuRoute format")
                        raise ModelExecutionError("Failed to convert HYPE output to mizuRoute format")

                return self.output_dir
            else:
                error_msg = f"HYPE simulation failed with return code {result.returncode}"
                self.logger.error(error_msg)
                self.logger.error(f"Command: {' '.join(map(str, cmd))}")
                raise ModelExecutionError(error_msg)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"HYPE execution failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error running HYPE: {str(e)}")
            raise

    def _run_distributed_routing(self) -> bool:
        """Run mizuRoute routing for HYPE output."""
        # Use SpatialOrchestrator method
        spatial_config = self.get_spatial_config('HYPE')
        result = self._run_mizuroute(spatial_config, model_name='hype')
        return result is not None

    def _convert_hype_to_mizuroute_format(self) -> bool:
        """Convert HYPE subbasin output to mizuRoute-compatible format."""
        try:
            import xarray as xr
            import numpy as np
            import pandas as pd

            # Use the same logic as HYPEWorker
            experiment_id = self.config_dict.get('EXPERIMENT_ID', 'run_1')
            sim_file = self.output_dir / 'timeCOUT.txt'
            
            if not sim_file.exists():
                self.logger.error(f"HYPE output file not found: {sim_file}")
                return False

            # Read simulation
            sim_df = pd.read_csv(sim_file, sep='\s+', skiprows=1)
            
            # Extract time index
            time_col = 'DATE' if 'DATE' in sim_df.columns else 'time'
            if time_col not in sim_df.columns:
                self.logger.error(f"No time column found in HYPE output: {sim_file}")
                return False
                
            times = pd.to_datetime(sim_df[time_col], format='%Y-%m-%d')
            sim_df = sim_df.drop(columns=[time_col])
            
            subids = [int(c) for c in sim_df.columns]
            n_gru = len(subids)
            
            routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
            if routing_var in ('default', None, ''):
                routing_var = 'q_routed'

            # Read GeoData.txt for areas
            geodata_file = self.setup_dir / 'GeoData.txt'
            if not geodata_file.exists():
                self.logger.error(f"GeoData.txt not found in {self.setup_dir}")
                return False
                
            geodata = pd.read_csv(geodata_file, sep='\t')
            area_map = dict(zip(geodata['subid'], geodata['area']))
            
            q_values = sim_df.values
            v_values = np.zeros_like(q_values, dtype=float)
            for i, subid in enumerate(subids):
                area = area_map.get(subid)
                if area and area > 0:
                    v_values[:, i] = q_values[:, i] / area
                else:
                    v_values[:, i] = 0.0

            # Create NetCDF
            ds_routing = xr.Dataset({
                routing_var: (('time', 'gru'), v_values)
            }, coords={
                'time': times.values,
                'gru': np.arange(n_gru)
            })

            ds_routing['gruId'] = ('gru', np.array(subids, dtype=int))
            ds_routing[routing_var].attrs['units'] = 'm/s'
            
            # Save to the filename mizuRoute expects for standard runs
            # MizuRouteRunner looks for {experiment_id}_timestep.nc in its fix_time_precision
            # But the control file generated by SpatialOrchestrator uses model-specific patterns
            # Standard pattern for SpatialOrchestrator: {experiment_id}_hype_runoff.nc
            expected_file = self.output_dir / f"{experiment_id}_hype_runoff.nc"
            
            # Also create the {experiment_id}_timestep.nc for backward compatibility
            # and to satisfy MizuRouteRunner.fix_time_precision if it runs later
            compat_file = self.output_dir / f"{experiment_id}_timestep.nc"
            
            ds_routing.to_netcdf(expected_file)
            shutil.copy2(expected_file, compat_file)
            
            ds_routing.close()
            return True

        except Exception as e:
            self.logger.error(f"Error converting HYPE output for mizuRoute: {e}")
            return False

    def _create_run_command(self) -> List[str]:
        """Create HYPE execution command."""
        return [
            str(self.hype_exe),
            str(self.setup_dir.absolute()).rstrip('/') + '/'
        ]

    def _verify_outputs(self) -> bool:
        """Verify HYPE output files exist."""
        required_outputs = [
            'timeCOUT.txt',  # Computed discharge
            'timeEVAP.txt',  # Evaporation
            'timeSNOW.txt'   # Snow water equivalent
        ]

        return self.verify_model_outputs(required_outputs)
