"""
HYPE model runner.

Handles HYPE model execution and run-time management.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import subprocess

from ..registry import ModelRegistry
from ..base import BaseModelRunner


@ModelRegistry.register_runner('HYPE', method_name='run_hype')
class HYPERunner(BaseModelRunner):
    """
    Runner class for the HYPE model within SYMFLUENCE.
    Handles model execution and run-time management.

    Attributes:
        config (Dict[str, Any]): Configuration settings
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the modeling domain
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize HYPE runner."""
        # Call base class
        super().__init__(config, logger)

    def _setup_model_specific_paths(self) -> None:
        """Set up HYPE-specific paths."""
        self.setup_dir = self.project_dir / "settings" / "HYPE"

        # HYPE-specific: Get installation path
        self.hype_dir = self.get_install_path('HYPE_INSTALL_PATH', 'installs/hype')

    def _get_model_name(self) -> str:
        """Return model name for HYPE."""
        return "HYPE"

    def _get_output_dir(self) -> Path:
        """HYPE uses custom output path resolution."""
        if self.typed_config:
            experiment_id = self.typed_config.domain.experiment_id
        else:
            experiment_id = self.config.get('EXPERIMENT_ID')
        return self.get_config_path('EXPERIMENT_OUTPUT_HYPE', f"simulations/{experiment_id}/HYPE")

    def run_hype(self) -> Optional[Path]:
        """
        Run the HYPE model simulation.

        Returns:
            Optional[Path]: Path to output directory if successful, None otherwise
        """
        self.logger.info("Starting HYPE model run")

        try:
            # Create run command
            cmd = self._create_run_command()

            # Set up logging
            log_dir = self.get_log_path()
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f'hype_run_{current_time}.log'

            # Execute HYPE
            self.logger.info(f"Executing command: {' '.join(map(str, cmd))}")

            result = self.execute_model_subprocess(
                cmd,
                log_file,
                cwd=self.setup_dir,
                check=False,  # Don't raise on non-zero exit, we'll handle it
                success_message="HYPE simulation completed successfully"
            )

            # Check execution success
            if result.returncode == 0 and self._verify_outputs():
                return self.output_dir
            else:
                self.logger.error("HYPE simulation failed")
                self._analyze_log_file(log_file)
                return None

        except subprocess.CalledProcessError as e:
            self.logger.error(f"HYPE execution failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error running HYPE: {str(e)}")
            raise

    def _create_run_command(self) -> List[str]:
        """Create HYPE execution command."""
        if self.typed_config and self.typed_config.model.hype:
            hype_exe_name = self.typed_config.model.hype.exe or 'hype'
        else:
            hype_exe_name = self.config.get('HYPE_EXE', 'hype')

        hype_exe = self.hype_dir / hype_exe_name

        cmd = [
            str(hype_exe),
            str(self.setup_dir) + '/'  # HYPE requires trailing slash
        ]
        print(cmd)
        return cmd

    def _verify_outputs(self) -> bool:
        """Verify HYPE output files exist."""
        required_outputs = [
            'timeCOUT.txt',  # Computed discharge
            'timeEVAP.txt',  # Evaporation
            'timeSNOW.txt'   # Snow water equivalent
        ]

        return self.verify_model_outputs(required_outputs)
