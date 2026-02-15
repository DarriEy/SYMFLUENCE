"""
mHM Model Runner

Executes the mHM (mesoscale Hydrological Model) using prepared input files.
mHM is run from within the settings directory where the namelists reside.
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


@ModelRegistry.register_runner("MHM")
class MHMRunner(BaseModelRunner):
    """
    Runs the mHM model.

    Handles:
    - Executable path resolution
    - Namelist file validation
    - Model execution from the settings directory
    - Output verification
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the mHM runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Setup paths
        self.mhm_input_dir = self.project_dir / "MHM_input"
        self.settings_dir = self.mhm_input_dir / "settings"
        self.forcing_dir = self.mhm_input_dir / "forcing"

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "MHM"

    def _get_mhm_executable(self) -> Path:
        """
        Get the mHM executable path.

        Uses the standardized get_model_executable method.

        Returns:
            Path: Path to mHM executable.

        Raises:
            FileNotFoundError: If executable not found.
        """
        return self.get_model_executable(
            install_path_key='MHM_INSTALL_PATH',
            default_install_subpath='installs/mhm',
            default_exe_name='mhm',
            typed_exe_accessor=lambda: (
                self.config.model.mhm.exe
                if self.config.model and self.config.model.mhm
                else None
            ),
            candidates=['bin', ''],
            must_exist=True
        )

    def _get_timeout(self) -> int:
        """Get execution timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.mhm.timeout,
            default=3600
        )

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute the mHM model.

        mHM is run from within the settings directory where it reads
        mhm.nml and mrm.nml namelists.

        Args:
            **kwargs: Additional arguments (unused)

        Returns:
            Path to output directory on success

        Raises:
            ModelExecutionError: If model execution fails
        """
        logger.info(f"Running mHM for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "mHM model execution",
            logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = (
                self.project_dir / "simulations" /
                self.config.domain.experiment_id / "MHM"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            mhm_exe = self._get_mhm_executable()
            logger.info(f"Using mHM executable: {mhm_exe}")

            # Verify namelist files exist in settings directory
            namelist_file = self._get_config_value(
                lambda: self.config.model.mhm.namelist_file,
                default='mhm.nml'
            )
            namelist_path = self.settings_dir / namelist_file
            if not namelist_path.exists():
                raise ModelExecutionError(
                    f"mHM namelist file not found: {namelist_path}"
                )

            # Build command - mHM is executed from the settings directory
            cmd = [str(mhm_exe)]

            logger.info(f"Executing command: {' '.join(cmd)} (cwd: {self.settings_dir})")

            # Set environment
            env = os.environ.copy()

            # Run the model from the settings directory
            timeout = self._get_timeout()
            result = subprocess.run(
                cmd,
                cwd=str(self.settings_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Log output
            if result.stdout:
                logger.debug(f"mHM stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"mHM stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"mHM execution returned code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}")
                raise ModelExecutionError(
                    f"mHM execution failed with return code {result.returncode}"
                )

            logger.info("mHM execution completed successfully")

            # Verify output was produced
            self._verify_output()

            return self.output_dir

    def _verify_output(self) -> None:
        """
        Verify that mHM produced valid output files.

        mHM produces discharge_*.nc and mHM_Fluxes_States_*.nc files.

        Raises:
            RuntimeError: If expected output files are missing or empty
        """
        # Look for discharge and fluxes/states output files
        output_files = []

        # Search in output directory and settings directory (mHM may write to either)
        for search_dir in [self.output_dir, self.settings_dir, self.settings_dir / 'output']:
            if search_dir.exists():
                output_files.extend(search_dir.glob("discharge_*.nc"))
                output_files.extend(search_dir.glob("mHM_Fluxes_States_*.nc"))

        if not output_files:
            raise RuntimeError(
                f"mHM did not produce expected output files (discharge_*.nc or "
                f"mHM_Fluxes_States_*.nc) in {self.output_dir} or {self.settings_dir}"
            )

        # Verify files have content
        for output_file in output_files:
            if output_file.stat().st_size == 0:
                raise RuntimeError(f"mHM output file is empty: {output_file}")

        logger.info(f"Verified mHM output: {len(output_files)} NetCDF file(s) produced")

    def run_mhm(self, **kwargs) -> Optional[Path]:
        """
        Alternative entry point for mHM execution.

        Args:
            **kwargs: Additional arguments passed to run()

        Returns:
            Path to output directory
        """
        return self.run(**kwargs)
