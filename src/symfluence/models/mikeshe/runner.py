"""
MIKE-SHE Model Runner

Executes the MIKE-SHE model (MikeSheEngine.exe) using prepared input files.
Supports optional WINE wrapper for running on Unix platforms.
"""
import logging
import subprocess
import os
import platform
from pathlib import Path
from typing import Optional

from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner("MIKESHE")
class MIKESHERunner(BaseModelRunner):
    """
    Runs the MIKE-SHE model.

    Handles:
    - Executable path resolution (MikeSheEngine.exe)
    - Optional WINE wrapper for Unix platforms
    - .she setup file configuration
    - Model execution and output verification
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the MIKE-SHE runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Setup paths
        self.mikeshe_input_dir = self.project_dir / "MIKESHE_input"
        self.settings_dir = self.mikeshe_input_dir / "settings"

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "MIKESHE"

    def _get_mikeshe_executable(self) -> Path:
        """
        Get the MIKE-SHE executable path.

        Uses the standardized get_model_executable method.

        Returns:
            Path: Path to MikeSheEngine.exe.

        Raises:
            FileNotFoundError: If executable not found.
        """
        return self.get_model_executable(
            install_path_key='MIKESHE_INSTALL_PATH',
            default_install_subpath='installs/mikeshe',
            default_exe_name='MikeSheEngine.exe',
            typed_exe_accessor=lambda: (
                self.config.model.mikeshe.exe
                if self.config.model and self.config.model.mikeshe
                else None
            ),
            candidates=['bin', ''],
            must_exist=True
        )

    def _get_setup_file(self) -> Path:
        """Get path to the MIKE-SHE .she setup file."""
        setup_file_name = self._get_config_value(
            lambda: self.config.model.mikeshe.setup_file,
            default='model.she'
        )
        return self.settings_dir / setup_file_name

    def _get_timeout(self) -> int:
        """Get execution timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.mikeshe.timeout,
            default=7200
        )

    def _use_wine(self) -> bool:
        """Check if WINE wrapper should be used (non-Windows platforms)."""
        use_wine = self._get_config_value(
            lambda: self.config.model.mikeshe.use_wine,
            default=False
        )
        # Auto-enable wine on non-Windows if configured
        if use_wine and platform.system() != 'Windows':
            return True
        return False

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute the MIKE-SHE model.

        Args:
            **kwargs: Additional arguments (unused)

        Returns:
            Path to output directory on success

        Raises:
            ModelExecutionError: If model execution fails
        """
        logger.info(f"Running MIKE-SHE for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "MIKE-SHE model execution",
            logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = (
                self.project_dir / "simulations" /
                self.config.domain.experiment_id / "MIKESHE"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            mikeshe_exe = self._get_mikeshe_executable()
            logger.info(f"Using MIKE-SHE executable: {mikeshe_exe}")

            # Get setup file
            setup_file = self._get_setup_file()
            if not setup_file.exists():
                raise ModelExecutionError(
                    f"MIKE-SHE setup file not found: {setup_file}"
                )

            # Build command
            cmd = []
            if self._use_wine():
                cmd.append('wine')
                logger.info("Using WINE wrapper for MIKE-SHE execution")
            cmd.extend([str(mikeshe_exe), str(setup_file)])

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
                logger.debug(f"MIKE-SHE stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"MIKE-SHE stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"MIKE-SHE execution returned code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}")
                raise ModelExecutionError(
                    f"MIKE-SHE execution failed with return code {result.returncode}"
                )

            logger.info("MIKE-SHE execution completed successfully")

            # Verify output was produced
            self._verify_output()

            return self.output_dir

    def _verify_output(self) -> None:
        """
        Verify that MIKE-SHE produced valid output files.

        MIKE-SHE produces .dfs0 time series files or CSV exports.

        Raises:
            RuntimeError: If expected output files are missing or empty
        """
        # Look for output files (.dfs0 or .csv)
        output_files = list(self.output_dir.glob("*.dfs0"))
        if not output_files:
            output_files = list(self.output_dir.glob("*.csv"))

        if not output_files:
            # Check subdirectories
            for subdir in self.output_dir.iterdir():
                if subdir.is_dir():
                    output_files.extend(subdir.glob("*.dfs0"))
                    output_files.extend(subdir.glob("*.csv"))

        if not output_files:
            raise RuntimeError(
                f"MIKE-SHE did not produce expected output files "
                f"(.dfs0 or .csv) in {self.output_dir}"
            )

        # Verify files have content
        for output_file in output_files:
            if output_file.stat().st_size == 0:
                raise RuntimeError(f"MIKE-SHE output file is empty: {output_file}")

        logger.info(
            f"Verified MIKE-SHE output: {len(output_files)} file(s) produced"
        )

    def run_mikeshe(self, **kwargs) -> Optional[Path]:
        """
        Alternative entry point for MIKE-SHE execution.

        Args:
            **kwargs: Additional arguments passed to run()

        Returns:
            Path to output directory
        """
        return self.run(**kwargs)
