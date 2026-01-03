"""
TRoute Model Runner.

Manages the execution of the t-route routing model.
Refactored to use the Unified Model Execution Framework.
"""

import subprocess
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional, List

from symfluence.utils.models.registry import ModelRegistry
from symfluence.utils.models.base import BaseModelRunner
from symfluence.utils.models.execution import ModelExecutor


@ModelRegistry.register_runner('TROUTE', method_name='run_troute')
class TRouteRunner(BaseModelRunner, ModelExecutor):
    """
    A standalone runner for the t-route model.

    Uses the Unified Model Execution Framework for subprocess execution.
    """
    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _get_model_name(self) -> str:
        """Return model name for TRoute."""
        return "TRoute"

    def _should_create_output_dir(self) -> bool:
        """TRoute creates directories on-demand."""
        return False

    def run_troute(self):
        """
        Prepares runoff data and executes t-route as a Python module.
        """
        self.logger.info("--- Starting t-route Run ---")

        # 1. Prepare runoff file by renaming the variable
        self._prepare_runoff_file()

        # 2. Set up paths for execution
        settings_path = self.project_dir / 'settings' / 'troute'
        config_file = self.config_dict.get('SETTINGS_TROUTE_CONFIG_FILE', 'troute_config.yml')
        config_filepath = settings_path / config_file
        troute_out_path = self.get_experiment_output_dir()
        log_path = self.get_log_path()
        log_file_path = log_path / "troute_run.log"

        # 3. Construct and run the command
        command = f"python -m nwm_routing {config_filepath}"
        self.logger.info(f'Executing t-route command: {command}')

        try:
            self.execute_model_subprocess(
                command,
                log_file_path,
                shell=True,
                success_message=f"t-route run completed successfully. Log file available at: {log_file_path}"
            )
            self.logger.info("--- t-route Run Finished ---")
            return troute_out_path
        except subprocess.CalledProcessError:
            self.logger.error(f"t-route run failed. See full log for details: {log_file_path}")
            if log_file_path.exists():
                with open(log_file_path, 'r') as f:
                    self.logger.error("--- t-route Log Output ---\n" + f.read())
            raise

    def _prepare_runoff_file(self):
        """
        Loads the hydrological model output and renames the runoff variable
        to 'q_lateral' as required by t-route.
        """
        self.logger.info("Preparing runoff file for t-route...")

        source_model = self.config_dict.get('TROUTE_FROM_MODEL', 'SUMMA').upper()
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        runoff_filepath = self.project_dir / f"simulations/{experiment_id}/{source_model}/{experiment_id}_timestep.nc"

        # Verify runoff file exists
        self.verify_required_files(runoff_filepath, context="t-route runoff preparation")

        # Fetch the original runoff variable name from config
        original_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
        
        self.logger.debug(f"Checking for variable '{original_var}' in {runoff_filepath}")

        ds = xr.open_dataset(runoff_filepath)
        if original_var in ds.data_vars:
            self.logger.info(f"Found '{original_var}', renaming to 'q_lateral'.")
            ds = ds.rename({original_var: 'q_lateral'})
            ds.to_netcdf(runoff_filepath, 'w', format='NETCDF4')
            self.logger.info("Runoff variable successfully renamed.")
        elif 'q_lateral' in ds.data_vars:
            self.logger.info("Runoff variable is already named 'q_lateral'. No action needed.")
        else:
            ds.close()
            self.logger.error(f"Expected runoff variable '{original_var}' not found in {runoff_filepath}.")
            raise ValueError(f"Runoff variable not found in {runoff_filepath}")
        ds.close()
