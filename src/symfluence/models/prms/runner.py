"""
PRMS Model Runner

Executes the PRMS model using prepared input files.
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


@ModelRegistry.register_runner("PRMS")
class PRMSRunner(BaseModelRunner):
    """
    Runs the PRMS model.

    Handles:
    - Executable path resolution (prms)
    - Control file setup
    - Model execution (prms -C control.dat)
    - Output verification (statvar output files)
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the PRMS runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Setup paths
        self.prms_input_dir = self.project_dir / "PRMS_input"
        self.settings_dir = self.prms_input_dir / "settings"

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "PRMS"

    def _get_prms_executable(self) -> Path:
        """
        Get the PRMS executable path.

        Returns:
            Path: Path to PRMS executable.

        Raises:
            FileNotFoundError: If executable not found.
        """
        return self.get_model_executable(
            install_path_key='PRMS_INSTALL_PATH',
            default_install_subpath='installs/prms',
            default_exe_name='prms',
            typed_exe_accessor=lambda: (
                self.config.model.prms.exe
                if self.config.model and self.config.model.prms
                else None
            ),
            candidates=['bin', ''],
            must_exist=True
        )

    def _get_control_file(self) -> Path:
        """Get path to the PRMS control file."""
        control_name = self._get_config_value(
            lambda: self.config.model.prms.control_file,
            default='control.dat'
        )
        return self.settings_dir / control_name

    def _get_timeout(self) -> int:
        """Get execution timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.prms.timeout,
            default=3600
        )

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute the PRMS model.

        Args:
            **kwargs: Additional arguments (unused)

        Returns:
            Path to output directory on success

        Raises:
            ModelExecutionError: If model execution fails
        """
        logger.info(f"Running PRMS for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "PRMS model execution",
            logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = (
                self.project_dir / "simulations" /
                self.config.domain.experiment_id / "PRMS"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            prms_exe = self._get_prms_executable()
            logger.info(f"Using PRMS executable: {prms_exe}")

            # Get control file
            control_file = self._get_control_file()
            if not control_file.exists():
                raise ModelExecutionError(
                    f"PRMS control file not found: {control_file}"
                )

            # Build command: prms -C control.dat
            cmd = [str(prms_exe), '-C', str(control_file)]

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
                logger.debug(f"PRMS stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"PRMS stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"PRMS execution returned code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}")
                raise ModelExecutionError(
                    f"PRMS execution failed with return code {result.returncode}"
                )

            logger.info("PRMS execution completed successfully")

            # Verify output was produced
            self._verify_output()

            return self.output_dir

    def _verify_output(self) -> None:
        """
        Verify that PRMS produced valid output files.

        Expects statvar output files (CSV or NetCDF).

        Raises:
            RuntimeError: If expected output files are missing or empty
        """
        # Look for statvar files
        output_files = list(self.output_dir.glob("statvar*"))

        # Also check in settings directory (PRMS writes output relative to control file)
        if not output_files:
            output_files = list(self.settings_dir.glob("statvar*"))

        # Also look for any .csv or .nc output
        if not output_files:
            output_files = (
                list(self.output_dir.glob("*.csv")) +
                list(self.output_dir.glob("*.nc"))
            )

        if not output_files:
            raise RuntimeError(
                f"PRMS did not produce expected statvar output files "
                f"in {self.output_dir} or {self.settings_dir}"
            )

        # Verify files have content
        for output_file in output_files:
            if output_file.stat().st_size == 0:
                raise RuntimeError(f"PRMS output file is empty: {output_file}")

        logger.info(f"Verified PRMS output: {len(output_files)} output file(s) produced")
