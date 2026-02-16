"""
CRHM Model Runner

Executes the CRHM (Cold Regions Hydrological Model) using prepared input files.
CRHM is driven by a .prj project file and reads forcing from a .obs file.
"""
import logging
import subprocess
import os
from pathlib import Path
from typing import Optional

from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner("CRHM")
class CRHMRunner(BaseModelRunner):
    """
    Runs the CRHM model.

    Handles:
    - Executable path resolution
    - Project file setup
    - Model execution with crhm binary
    - Output verification
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the CRHM runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Setup paths
        self.crhm_input_dir = self.project_dir / "CRHM_input"
        self.settings_dir = self.crhm_input_dir / "settings"

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "CRHM"

    def _get_crhm_executable(self) -> Path:
        """
        Get the CRHM executable path.

        Uses the standardized get_model_executable method.

        Returns:
            Path: Path to CRHM executable.

        Raises:
            FileNotFoundError: If executable not found.
        """
        return self.get_model_executable(
            install_path_key='CRHM_INSTALL_PATH',
            default_install_subpath='installs/crhm',
            default_exe_name='crhm',
            typed_exe_accessor=lambda: (
                self.config.model.crhm.exe
                if self.config.model and self.config.model.crhm
                else None
            ),
            candidates=['bin', ''],
            must_exist=True
        )

    def _get_project_file(self) -> Path:
        """Get path to the CRHM project file."""
        project_file_name = self._get_config_value(
            lambda: self.config.model.crhm.project_file,
            default='model.prj'
        )
        return self.settings_dir / project_file_name

    def _get_timeout(self) -> int:
        """Get execution timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.crhm.timeout,
            default=3600
        )

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute the CRHM model.

        Args:
            **kwargs: Additional arguments (unused)

        Returns:
            Path to output directory on success

        Raises:
            ModelExecutionError: If model execution fails
        """
        logger.info(f"Running CRHM for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "CRHM model execution",
            logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = (
                self.project_dir / "simulations" /
                self.config.domain.experiment_id / "CRHM"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            crhm_exe = self._get_crhm_executable()
            logger.info(f"Using CRHM executable: {crhm_exe}")

            # Get project file
            project_file = self._get_project_file()
            if not project_file.exists():
                raise ModelExecutionError(
                    f"CRHM project file not found: {project_file}"
                )

            # Build command: crhm [options] <project_file>
            # The project file is a positional argument. The -f flag is for
            # output format (STD/OBS), NOT for specifying the project file.
            # Use --obs_file_directory to point CRHM at the obs files when
            # the working directory differs from the project file location.
            # -o <path>  : write output to a file
            # -p <int>   : progress reporting interval (% steps)
            obs_dir = str(project_file.parent) + os.sep
            output_file = self.output_dir / "crhm_output.txt"
            cmd = [
                str(crhm_exe),
                '--obs_file_directory', obs_dir,
                '-o', str(output_file),
                '-p', '30',
                str(project_file),
            ]

            logger.info(f"Executing command: {' '.join(cmd)}")

            # Set environment
            env = os.environ.copy()

            # Run the model
            timeout = self._get_timeout()
            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Log output
            if result.stdout:
                logger.debug(f"CRHM stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"CRHM stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"CRHM execution returned code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}")
                raise ModelExecutionError(
                    f"CRHM execution failed with return code {result.returncode}"
                )

            logger.info("CRHM execution completed successfully")

            # Verify output was produced
            self._verify_output()

            return self.output_dir

    def _verify_output(self) -> None:
        """
        Verify that CRHM produced valid output files.

        Raises:
            RuntimeError: If expected output files are missing or empty
        """
        # Look for CSV output files
        output_files = list(self.output_dir.glob("*.csv"))

        if not output_files:
            # Also check for output in the settings directory
            output_files = list(self.settings_dir.glob("*.csv"))

        if not output_files:
            # Check for any text output files
            output_files = list(self.output_dir.glob("*.txt"))

        if not output_files:
            raise RuntimeError(
                f"CRHM did not produce expected output files "
                f"in {self.output_dir}"
            )

        # Verify files have content
        for output_file in output_files:
            if output_file.stat().st_size == 0:
                raise RuntimeError(f"CRHM output file is empty: {output_file}")

        logger.info(f"Verified CRHM output: {len(output_files)} output file(s) produced")

    def run_crhm(self, **kwargs) -> Optional[Path]:
        """
        Alternative entry point for CRHM execution.

        Args:
            **kwargs: Additional arguments passed to run()

        Returns:
            Path to output directory
        """
        return self.run(**kwargs)
