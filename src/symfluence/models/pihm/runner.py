"""
PIHM Model Runner

Executes PIHM (MM-PIHM) from a prepared simulation directory.
PIHM reads <project>.para from the working directory to discover
all model input files.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner("PIHM", method_name="run_pihm")
class PIHMRunner(BaseModelRunner):
    """
    Runs PIHM via direct invocation.

    Handles:
    - Executable path resolution
    - Input file copying to simulation directory
    - Model execution (pihm <project_name>)
    - Output verification (.rivflx, .gwhead, .surf files)
    """

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)
        self.settings_dir = self.project_dir / "settings" / "PIHM"

    def _get_model_name(self) -> str:
        return "PIHM"

    def _get_pihm_executable(self) -> Path:
        """Get the PIHM executable path."""
        return self.get_model_executable(
            install_path_key='PIHM_INSTALL_PATH',
            default_install_subpath='installs/pihm',
            default_exe_name='pihm',
            typed_exe_accessor=lambda: (
                self.config.model.pihm.exe
                if self.config.model and self.config.model.pihm
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_timeout(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.pihm.timeout,
            default=3600,
        )

    def run_pihm(self, sim_dir: Optional[Path] = None, **kwargs) -> Optional[Path]:
        """
        Execute PIHM.

        Args:
            sim_dir: Optional override for simulation directory.

        Returns:
            Path to output directory on success.

        Raises:
            ModelExecutionError: If execution fails.
        """
        logger.info(f"Running PIHM for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "PIHM model execution",
            logger,
            error_type=ModelExecutionError,
        ):
            if sim_dir is None:
                self.output_dir = (
                    self.project_dir / "simulations"
                    / self.config.domain.experiment_id / "PIHM"
                )
            else:
                self.output_dir = sim_dir

            self.output_dir.mkdir(parents=True, exist_ok=True)

            pihm_exe = self._get_pihm_executable()
            logger.info(f"Using PIHM executable: {pihm_exe}")

            self._setup_sim_directory(self.output_dir)

            project_name = "pihm_lumped"
            cmd = [str(pihm_exe), project_name]
            logger.info(f"Executing PIHM from: {self.output_dir}")

            env = os.environ.copy()
            timeout = self._get_timeout()

            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.stdout:
                logger.debug(f"PIHM stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"PIHM stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"PIHM execution returned code {result.returncode}")
                logger.error(
                    f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}"
                )
                raise ModelExecutionError(
                    f"PIHM execution failed with return code {result.returncode}"
                )

            logger.info("PIHM execution completed successfully")
            self._verify_output()

            return self.output_dir

    def _setup_sim_directory(self, sim_dir: Path) -> None:
        """Copy all PIHM input files to simulation directory."""
        if not self.settings_dir.exists():
            raise ModelExecutionError(
                f"PIHM settings directory not found: {self.settings_dir}. "
                "Run preprocessing first."
            )

        for src in self.settings_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, sim_dir / src.name)
                logger.debug(f"Copied {src.name} to simulation directory")

    def _verify_output(self) -> None:
        """Verify PIHM produced valid output files."""
        rivflx_files = list(self.output_dir.glob("*.rivflx*"))
        gwhead_files = list(self.output_dir.glob("*.gwhead*"))

        if not rivflx_files and not gwhead_files:
            raise RuntimeError(
                f"PIHM did not produce expected output in {self.output_dir}"
            )

        logger.info(
            f"Verified PIHM output: {len(rivflx_files)} rivflx file(s), "
            f"{len(gwhead_files)} gwhead file(s)"
        )

    def run(self, **kwargs) -> Optional[Path]:
        """Alternative entry point for PIHM execution."""
        return self.run_pihm(**kwargs)
