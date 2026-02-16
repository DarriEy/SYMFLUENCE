"""
SWAT Model Runner

Executes the SWAT model using prepared TxtInOut input files.
SWAT runs from within its TxtInOut directory and produces output.rch
as its primary output file.
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


@ModelRegistry.register_runner("SWAT")
class SWATRunner(BaseModelRunner):
    """
    Runs the SWAT model.

    Handles:
    - Executable path resolution
    - TxtInOut directory setup
    - Model execution (swat_rel.exe runs from TxtInOut)
    - Output verification (output.rch)
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the SWAT runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Setup paths
        txtinout_name = self._get_config_value(
            lambda: self.config.model.swat.txtinout_dir,
            default='TxtInOut'
        )
        self.txtinout_dir = self.project_dir / "SWAT_input" / txtinout_name

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "SWAT"

    def _get_swat_executable(self) -> Path:
        """
        Get the SWAT executable path.

        Uses the standardized get_model_executable method.

        Returns:
            Path: Path to SWAT executable.

        Raises:
            FileNotFoundError: If executable not found.
        """
        return self.get_model_executable(
            install_path_key='SWAT_INSTALL_PATH',
            default_install_subpath='installs/swat',
            default_exe_name='swat_rel.exe',
            typed_exe_accessor=lambda: (
                self.config.model.swat.exe
                if self.config.model and self.config.model.swat
                else None
            ),
            candidates=['bin', ''],
            must_exist=True
        )

    def _get_timeout(self) -> int:
        """Get execution timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.swat.timeout,
            default=3600
        )

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute the SWAT model.

        SWAT is run from within its TxtInOut directory. The executable
        reads file.cio and all associated input files, then produces
        output.rch and other output files in the same directory.

        Args:
            **kwargs: Additional arguments (unused)

        Returns:
            Path to TxtInOut directory on success

        Raises:
            ModelExecutionError: If model execution fails
        """
        logger.info(f"Running SWAT for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "SWAT model execution",
            logger,
            error_type=ModelExecutionError
        ):
            # Verify TxtInOut directory exists
            if not self.txtinout_dir.exists():
                raise ModelExecutionError(
                    f"SWAT TxtInOut directory not found: {self.txtinout_dir}"
                )

            # Verify file.cio exists (master control file)
            file_cio = self.txtinout_dir / 'file.cio'
            if not file_cio.exists():
                raise ModelExecutionError(
                    f"SWAT master control file not found: {file_cio}"
                )

            # Get executable
            swat_exe = self._get_swat_executable()
            logger.info(f"Using SWAT executable: {swat_exe}")

            # Build command - SWAT runs from the TxtInOut directory
            cmd = [str(swat_exe)]

            logger.info(f"Executing SWAT in: {self.txtinout_dir}")

            # Set environment
            env = os.environ.copy()

            # Run the model from within TxtInOut
            timeout = self._get_timeout()
            result = subprocess.run(
                cmd,
                cwd=str(self.txtinout_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Log output
            if result.stdout:
                logger.debug(f"SWAT stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"SWAT stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"SWAT execution returned code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}")
                raise ModelExecutionError(
                    f"SWAT execution failed with return code {result.returncode}"
                )

            logger.info("SWAT execution completed successfully")

            # Verify output was produced
            self._verify_output()

            return self.txtinout_dir

    def _verify_output(self) -> None:
        """
        Verify that SWAT produced valid output files.

        Raises:
            RuntimeError: If expected output files are missing or empty
        """
        output_rch = self.txtinout_dir / 'output.rch'

        if not output_rch.exists():
            raise RuntimeError(
                f"SWAT did not produce output.rch in {self.txtinout_dir}"
            )

        if output_rch.stat().st_size == 0:
            raise RuntimeError(f"SWAT output.rch is empty: {output_rch}")

        logger.info(f"Verified SWAT output: output.rch ({output_rch.stat().st_size} bytes)")

    def run_swat(self, **kwargs) -> Optional[Path]:
        """
        Alternative entry point for SWAT execution.

        Args:
            **kwargs: Additional arguments passed to run()

        Returns:
            Path to TxtInOut directory
        """
        return self.run(**kwargs)
