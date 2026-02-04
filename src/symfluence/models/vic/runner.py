"""
VIC Model Runner

Executes the VIC model using prepared input files.
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


@ModelRegistry.register_runner("VIC")
class VICRunner(BaseModelRunner):
    """
    Runs the VIC model.

    Handles:
    - Executable path resolution
    - Global parameter file setup
    - Model execution with image driver
    - Output verification
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the VIC runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Setup paths
        self.vic_input_dir = self.project_dir / "VIC_input"
        self.settings_dir = self.vic_input_dir / "settings"
        self.forcing_dir = self.vic_input_dir / "forcing"
        self.params_dir = self.vic_input_dir / "parameters"

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "VIC"

    def _get_vic_executable(self) -> Path:
        """
        Get the VIC executable path.

        Uses the standardized get_model_executable method.

        Returns:
            Path: Path to VIC executable.

        Raises:
            FileNotFoundError: If executable not found.
        """
        # Get driver type from config
        driver = self._get_config_value(
            lambda: self.config.model.vic.driver,
            default='image'
        )

        # Default executable name based on driver
        default_exe = 'vic_image.exe' if driver == 'image' else 'vic_classic.exe'

        return self.get_model_executable(
            install_path_key='VIC_INSTALL_PATH',
            default_install_subpath='installs/vic',
            default_exe_name=default_exe,
            typed_exe_accessor=lambda: (
                self.config.model.vic.exe
                if self.config.model and self.config.model.vic
                else None
            ),
            candidates=['bin', ''],
            must_exist=True
        )

    def _get_global_param_file(self) -> Path:
        """Get path to the VIC global parameter file."""
        global_file_name = self._get_config_value(
            lambda: self.config.model.vic.global_param_file,
            default='vic_global.txt'
        )
        return self.settings_dir / global_file_name

    def _get_timeout(self) -> int:
        """Get execution timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.vic.timeout,
            default=7200
        )

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute the VIC model.

        Args:
            **kwargs: Additional arguments (unused)

        Returns:
            Path to output directory on success

        Raises:
            ModelExecutionError: If model execution fails
        """
        logger.info(f"Running VIC for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "VIC model execution",
            logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = (
                self.project_dir / "simulations" /
                self.config.domain.experiment_id / "VIC"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            vic_exe = self._get_vic_executable()
            logger.info(f"Using VIC executable: {vic_exe}")

            # Get global parameter file
            global_param_file = self._get_global_param_file()
            if not global_param_file.exists():
                raise ModelExecutionError(
                    f"VIC global parameter file not found: {global_param_file}"
                )

            # Build command
            cmd = [str(vic_exe), '-g', str(global_param_file)]

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
                logger.debug(f"VIC stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"VIC stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"VIC execution returned code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}")
                raise ModelExecutionError(
                    f"VIC execution failed with return code {result.returncode}"
                )

            logger.info("VIC execution completed successfully")

            # Verify output was produced
            self._verify_output()

            return self.output_dir

    def _verify_output(self) -> None:
        """
        Verify that VIC produced valid output files.

        Raises:
            RuntimeError: If expected output files are missing or empty
        """
        # Get output prefix from config
        output_prefix = self._get_config_value(
            lambda: self.config.model.vic.output_prefix,
            default='vic_output'
        )

        # Look for output files
        output_files = list(self.output_dir.glob(f"{output_prefix}*.nc"))

        if not output_files:
            # Also check in VIC_output subdirectory
            vic_output_dir = self.output_dir / "VIC_output"
            if vic_output_dir.exists():
                output_files = list(vic_output_dir.glob("*.nc"))

        if not output_files:
            raise RuntimeError(
                f"VIC did not produce expected output files with prefix '{output_prefix}' "
                f"in {self.output_dir}"
            )

        # Verify files have content
        for output_file in output_files:
            if output_file.stat().st_size == 0:
                raise RuntimeError(f"VIC output file is empty: {output_file}")

        logger.info(f"Verified VIC output: {len(output_files)} NetCDF file(s) produced")

    def run_vic(self, **kwargs) -> Optional[Path]:
        """
        Alternative entry point for VIC execution.

        Args:
            **kwargs: Additional arguments passed to run()

        Returns:
            Path to output directory
        """
        return self.run(**kwargs)
