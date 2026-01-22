"""
HYPE model runner.

Handles HYPE model execution and run-time management.
Refactored to use the Unified Model Execution Framework.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from ..registry import ModelRegistry
from ..base import BaseModelRunner
from ..execution import ModelExecutor
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler


@ModelRegistry.register_runner('HYPE', method_name='run_hype')
class HYPERunner(BaseModelRunner, ModelExecutor):
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
        """
        Initialize the HYPE model runner.

        Sets up HYPE-specific paths including settings directory and
        executable location using the Unified Model Execution Framework.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                HYPE installation path and simulation settings.
            logger: Logger instance for status messages and debugging.
            reporting_manager: Optional reporting manager for experiment tracking.
        """
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up HYPE-specific paths."""
        self.setup_dir = self.project_dir / "settings" / "HYPE"

        # HYPE executable path (installation dir + exe name)
        self.hype_exe = self.get_model_executable(
            install_path_key='HYPE_INSTALL_PATH',
            default_install_subpath='installs/hype/bin',
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

        with symfluence_error_handler(
            "HYPE model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Create run command
            cmd = self._create_run_command()
            self.logger.debug(f"HYPE command: {cmd}")

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
                success_message="HYPE simulation completed successfully",
                success_log_level=logging.DEBUG
            )

            # Check execution success
            if result.returncode == 0 and self._verify_outputs():
                return self.output_dir
            else:
                self.logger.error("HYPE simulation failed")
                return None

    def _create_run_command(self) -> List[str]:
        """Create HYPE execution command."""
        return [
            str(self.hype_exe),
            str(self.setup_dir).rstrip('/') + '/'
        ]

    def _verify_outputs(self) -> bool:
        """Verify HYPE output files exist."""
        required_outputs = [
            'timeCOUT.txt',  # Computed discharge
            'timeEVAP.txt',  # Evaporation
            'timeSNOW.txt'   # Snow water equivalent
        ]
        return self.verify_model_outputs(required_outputs)
