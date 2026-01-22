"""
MESH model runner.

Handles MESH model execution, state management, and output processing.
Refactored to use the Unified Model Execution Framework.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..base import BaseModelRunner
from ..execution import ModelExecutor
from ..registry import ModelRegistry
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler


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
        """
        Initialize the MESH model runner.

        Sets up MESH-specific paths including executable location, forcing
        directory, and catchment shapefile paths.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                MESH installation path, domain settings, and execution parameters.
            logger: Logger instance for status messages and debugging.
            reporting_manager: Optional reporting manager for experiment tracking.
        """
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up MESH-specific paths."""
        self.mesh_exe = self.get_model_executable(
            install_path_key='MESH_INSTALL_PATH',
            default_install_subpath='installs/mesh/bin',
            exe_name_key='MESH_EXE',
            default_exe_name='mesh.exe',
            typed_exe_accessor=lambda: self.typed_config.model.mesh.exe if (self.typed_config and self.typed_config.model.mesh) else None
        )

        # Catchment paths (use backward-compatible path resolution)
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

        # MESH-specific paths
        self.mesh_setup_dir = self.project_dir / "settings" / "MESH"
        self.forcing_dir = self.project_dir / 'forcing' / 'MESH_input'

        # Initialize forcing_mesh_path to forcing_dir (can be overridden for parallel execution)
        self.forcing_mesh_path = self.forcing_dir

    def _get_model_name(self) -> str:
        """Return model name for MESH."""
        return "MESH"

    def _get_output_dir(self) -> Path:
        """MESH output directory."""
        return self.get_experiment_output_dir()

    def set_process_directories(self, forcing_dir: Path, output_dir: Path) -> None:
        """
        Set process-specific directories for parallel execution.

        Args:
            forcing_dir: Process-specific forcing directory
            output_dir: Process-specific output directory
        """
        self.forcing_mesh_path = forcing_dir
        self.output_dir = output_dir
        self.logger.debug(f"Set MESH paths: forcing={forcing_dir}, output={output_dir}")

    def run_mesh(self) -> Optional[Path]:
        """
        Run the MESH model simulation.

        Executes MESH in the forcing directory, verifies outputs, and cleans
        up temporary files on success. MESH requires execution from its input
        directory due to relative path assumptions in the model.

        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise.

        Note:
            MESH executable is temporarily copied to the forcing directory for
            execution and removed after successful completion.
        """
        self.logger.info("Starting MESH model run")

        with symfluence_error_handler(
            "MESH model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Create run command
            cmd = self._create_run_command()

            # Set up logging
            log_dir = self.get_log_path()
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f'mesh_run_{current_time}.log'

            # Execute MESH (it must run in the forcing directory)
            self.logger.info(f"Executing command: {' '.join(map(str, cmd))}")

            # Prepare environment
            run_env = os.environ.copy()
            if sys.platform == 'darwin':
                # Ensure homebrew paths are included
                brew_lib = "/opt/homebrew/lib"
                if brew_lib not in run_env.get("DYLD_LIBRARY_PATH", ""):
                    run_env['DYLD_LIBRARY_PATH'] = f"{brew_lib}:{run_env.get('DYLD_LIBRARY_PATH', '')}"

            result = self.execute_model_subprocess(
                cmd,
                log_file,
                cwd=self.forcing_mesh_path,
                env=run_env,
                check=False,  # Don't raise on non-zero exit, we'll handle it
                success_message="MESH simulation completed successfully"
            )

            # Check execution success
            if result.returncode == 0 and self._verify_outputs():
                # Clean up copied executable only on success
                mesh_exe_in_forcing = self.forcing_mesh_path / self.mesh_exe.name
                if mesh_exe_in_forcing.exists() and mesh_exe_in_forcing.is_file():
                    mesh_exe_in_forcing.unlink()
                return self.output_dir
            else:
                self.logger.error(f"MESH simulation failed with code {result.returncode}")
                # Log the end of the log file for easier debugging
                if log_file.exists():
                     with open(log_file, 'r', errors='replace') as f:  # Handle non-UTF-8 characters
                         lines = f.readlines()
                         last_lines = lines[-20:]
                         self.logger.error("Last 20 lines of model log:")
                         for line in last_lines:
                             self.logger.error(f"  {line.strip()}")
                return None

    def _create_run_command(self) -> List[str]:
        """
        Create MESH execution command.

        Copies the MESH executable to the forcing directory (required by MESH),
        ensures it has execute permissions, and creates the results subdirectory
        that MESH expects for output.

        Returns:
            List[str]: Command arguments for subprocess execution.
        """
        # Copy mesh executable to forcing path
        mesh_exe_dest = self.forcing_mesh_path / self.mesh_exe.name
        shutil.copy2(self.mesh_exe, mesh_exe_dest)
        # Make sure it's executable
        mesh_exe_dest.chmod(0o755)

        # Create results directory that MESH expects
        results_dir = self.forcing_mesh_path / 'results'
        self.logger.info(f"Creating MESH results directory: {results_dir}")
        results_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Results directory exists: {results_dir.exists()}")

        cmd = [
            f'./{self.mesh_exe.name}'
        ]
        return cmd

    def _verify_outputs(self) -> bool:
        """
        Verify MESH output files exist.

        Checks for required output files in both the output directory and
        forcing directory (MESH writes outputs to its working directory).

        Returns:
            bool: True if all required outputs found, False otherwise.
        """
        required_outputs = [
            'MESH_output_streamflow.csv',
        ]

        # Check in output directory (may be process-specific during parallel calibration)
        # or fall back to forcing directory (default MESH behavior)
        check_dirs = [self.output_dir, self.forcing_mesh_path]

        for output_file in required_outputs:
            found = False
            for check_dir in check_dirs:
                output_path = check_dir / output_file
                if output_path.exists():
                    found = True
                    break

            if not found:
                self.logger.warning(f"Required output file not found: {output_file}")
                return False

        return True

    def _copy_outputs(self) -> None:
        """
        Copy MESH outputs from forcing directory to simulation directory.

        MESH writes outputs to its working directory (forcing_mesh_path).
        This method copies key output files to the standard simulation
        output directory for consistency with other models.

        Copied files:
            - MESH_output_streamflow.csv: Simulated streamflow timeseries
            - MESH_output_echo_print.txt: Model run summary
            - MESH_output_echo_results.txt: Detailed results log
        """
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        outputs_to_copy = [
            'MESH_output_streamflow.csv',
            'MESH_output_echo_print.txt',
            'MESH_output_echo_results.txt'
        ]

        for out_file in outputs_to_copy:
            src = self.forcing_mesh_path / out_file
            if src.exists():
                shutil.copy2(src, self.output_dir / out_file)
