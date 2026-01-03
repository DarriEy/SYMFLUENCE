"""
MESH model runner.

Handles MESH model execution, state management, and output processing.
Refactored to use the Unified Model Execution Framework.
"""

import os
import shutil
import subprocess
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

    def run_MESH(self) -> Optional[Path]:
        """
        Run the MESH model.

        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise
        """
        # Store current directory
        original_dir = os.getcwd()

        # Change to forcing directory for execution
        os.chdir(self.forcing_mesh_path)

        cmd = self._create_run_command()
        subprocess.run(cmd, check=True)

        # Change back to original directory
        os.chdir(original_dir)
        shutil.rmtree(self.forcing_mesh_path / self.mesh_exe.name)
        return

    def _create_run_command(self) -> List[str]:
        """Create MESH execution command."""
        # Copy mesh executable to forcing path
        mesh_exe_dest = self.forcing_mesh_path / self.mesh_exe.name
        shutil.copy2(self.mesh_exe, mesh_exe_dest)

        cmd = [
            str(self.mesh_exe)
        ]
        print(cmd)
        return cmd
