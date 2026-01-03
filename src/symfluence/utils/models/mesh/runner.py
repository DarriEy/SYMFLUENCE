"""
MESH model runner.

Handles MESH model execution, state management, and output processing.
Refactored to use the Unified Model Execution Framework.
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..base import BaseModelRunner
from ..execution import ModelExecutor
from ..registry import ModelRegistry


@ModelRegistry.register_runner('MESH', method_name='run_mesh')
class MESHRunner(BaseModelRunner, ModelExecutor):
    """
    Runner class for the MESH model.
    Handles model execution, state management, and output processing.

    Uses the Unified Model Execution Framework for subprocess execution.

    Attributes:
        config (Dict[str, Any]): Configuration settings for MESH model
        logger (Any): Logger object for recording run information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up MESH-specific paths."""
        # MESH executable path (installation dir + exe name)
        self.mesh_exe = self.get_model_executable(
            install_path_key='MESH_INSTALL_PATH',
            default_install_subpath='installs/MESH-DEV',
            exe_name_key='MESH_EXE',
            default_exe_name='sa_mesh',
            typed_exe_accessor=lambda: self.config.model.mesh.exe if (self.typed_config and self.config.model.mesh) else None
        )

        # Catchment paths (now has PathResolverMixin via BaseModelRunner)
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            discretization = self.config_dict.get('DOMAIN_DISCRETIZATION')
            self.catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"

        # MESH-specific paths
        self.mesh_setup_dir = self.project_dir / "settings" / "MESH"
        self.forcing_mesh_path = self.project_dir / 'forcing' / 'MESH_input'

    def _get_model_name(self) -> str:
        """Return model name for MESH."""
        return "MESH"

    def _get_output_dir(self) -> Path:
        """MESH output directory."""
        return self.get_experiment_output_dir()

    def run_mesh(self) -> Optional[Path]:
        """
        Run the MESH model.

        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise
        """
        self.logger.info("Starting MESH model run")

        try:
            # Create run command
            cmd = self._create_run_command()

            # Set up logging
            log_dir = self.get_log_path()
            from datetime import datetime
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f'mesh_run_{current_time}.log'

            # Execute MESH (it must run in the forcing directory)
            self.logger.info(f"Executing command: {' '.join(map(str, cmd))}")

            result = self.execute_model_subprocess(
                cmd,
                log_file,
                cwd=self.forcing_mesh_path,
                check=False,  # Don't raise on non-zero exit, we'll handle it
                success_message="MESH simulation completed successfully"
            )

            # Clean up copied executable
            mesh_exe_in_forcing = self.forcing_mesh_path / self.mesh_exe.name
            if mesh_exe_in_forcing.exists() and mesh_exe_in_forcing.is_file():
                mesh_exe_in_forcing.unlink()

            # Check execution success
            if result.returncode == 0 and self._verify_outputs():
                return self.output_dir
            else:
                self.logger.error("MESH simulation failed")
                return None

        except subprocess.CalledProcessError as e:
            self.logger.error(f"MESH execution failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error running MESH: {str(e)}")
            raise

    def _create_run_command(self) -> List[str]:
        """Create MESH execution command."""
        # Copy mesh executable to forcing path
        mesh_exe_dest = self.forcing_mesh_path / self.mesh_exe.name
        shutil.copy2(self.mesh_exe, mesh_exe_dest)
        # Make sure it's executable
        mesh_exe_dest.chmod(0o755)

        cmd = [
            f'./{self.mesh_exe.name}'  # Use relative path since we run in that directory
        ]
        return cmd

    def _verify_outputs(self) -> bool:
        """Verify MESH output files exist."""
        required_outputs = [
            'MESH_output_streamflow.csv',
        ]

        # Check in forcing directory where MESH runs
        for output_file in required_outputs:
            output_path = self.forcing_mesh_path / output_file
            if not output_path.exists():
                self.logger.warning(f"Required output file not found: {output_file}")
                return False

        return True
