"""
HydroGeoSphere Model Runner

Executes HGS via two-step process:
1. grok <prefix> — preprocesses input into binary format
2. hgs <prefix> — runs the simulation
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


@ModelRegistry.register_runner("HYDROGEOSPHERE")
class HGSRunner(BaseModelRunner):
    """
    Runs HydroGeoSphere via grok + hgs invocation.

    Handles:
    - Executable path resolution
    - Input file copying to simulation directory
    - Two-step execution (grok preprocessing + hgs solver)
    - Output verification (.hen, hydrograph files)
    """


    MODEL_NAME = "HYDROGEOSPHERE"
    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)
        self.settings_dir = self.project_dir / "settings" / "HYDROGEOSPHERE"

    def _get_hgs_executable(self) -> Path:
        """Get the HGS solver executable path."""
        return self.get_model_executable(
            install_path_key='HGS_INSTALL_PATH',
            default_install_subpath='installs/hydrogeosphere',
            default_exe_name='hgs',
            typed_exe_accessor=lambda: (
                self.config.model.hydrogeosphere.exe
                if self.config.model and self.config.model.hydrogeosphere
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_grok_executable(self) -> Path:
        """Get the grok preprocessor executable path."""
        return self.get_model_executable(
            install_path_key='HGS_INSTALL_PATH',
            default_install_subpath='installs/hydrogeosphere',
            default_exe_name='grok',
            typed_exe_accessor=lambda: (
                self.config.model.hydrogeosphere.grok_exe
                if self.config.model and self.config.model.hydrogeosphere
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_timeout(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.hydrogeosphere.timeout,
            default=7200,
        )

    def run_hydrogeosphere(self, sim_dir: Optional[Path] = None, **kwargs) -> Optional[Path]:
        """
        Execute HydroGeoSphere (grok + hgs).

        Args:
            sim_dir: Optional override for simulation directory.

        Returns:
            Path to output directory on success.
        """
        logger.info(f"Running HydroGeoSphere for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "HydroGeoSphere model execution",
            logger,
            error_type=ModelExecutionError,
        ):
            if sim_dir is None:
                self.output_dir = (
                    self.project_dir / "simulations"
                    / self.config.domain.experiment_id / "HYDROGEOSPHERE"
                )
            else:
                self.output_dir = sim_dir

            self.output_dir.mkdir(parents=True, exist_ok=True)

            grok_exe = self._get_grok_executable()
            hgs_exe = self._get_hgs_executable()
            logger.info(f"Using grok: {grok_exe}, hgs: {hgs_exe}")

            self._setup_sim_directory(self.output_dir)

            # Read prefix from batch.pfx
            prefix = self._get_prefix()

            env = os.environ.copy()
            timeout = self._get_timeout()

            # Step 1: Run grok preprocessor
            logger.info(f"Running grok preprocessor from: {self.output_dir}")
            grok_result = subprocess.run(
                [str(grok_exe), prefix],
                cwd=str(self.output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout // 4,
            )

            if grok_result.returncode != 0:
                logger.error(f"grok failed (rc={grok_result.returncode})")
                if grok_result.stderr:
                    logger.error(f"grok stderr: {grok_result.stderr[-2000:]}")
                raise ModelExecutionError(
                    f"HGS grok preprocessing failed with return code {grok_result.returncode}"
                )

            # Step 2: Run HGS solver
            logger.info(f"Running HGS solver from: {self.output_dir}")
            hgs_result = subprocess.run(
                [str(hgs_exe), prefix],
                cwd=str(self.output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if hgs_result.stdout:
                logger.debug(f"HGS stdout: {hgs_result.stdout[-2000:]}")
            if hgs_result.stderr:
                logger.debug(f"HGS stderr: {hgs_result.stderr[-2000:]}")

            if hgs_result.returncode != 0:
                logger.error(f"HGS solver returned code {hgs_result.returncode}")
                raise ModelExecutionError(
                    f"HGS solver failed with return code {hgs_result.returncode}"
                )

            logger.info("HGS execution completed successfully")
            self._verify_output()

            return self.output_dir

    def _get_prefix(self) -> str:
        """Read problem prefix from batch.pfx."""
        pfx_file = self.output_dir / "batch.pfx"
        if pfx_file.exists():
            return pfx_file.read_text().strip()
        return "hgs_lumped"

    def _setup_sim_directory(self, sim_dir: Path) -> None:
        """Copy all HGS input files to simulation directory."""
        if not self.settings_dir.exists():
            raise ModelExecutionError(
                f"HGS settings directory not found: {self.settings_dir}. "
                "Run preprocessing first."
            )

        for src in self.settings_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, sim_dir / src.name)
                logger.debug(f"Copied {src.name} to simulation directory")

    def _verify_output(self) -> None:
        """Verify HGS produced valid output files."""
        hydrograph_files = list(self.output_dir.glob("*hydrograph*"))
        head_files = list(self.output_dir.glob("*head*"))

        if not hydrograph_files and not head_files:
            raise RuntimeError(
                f"HGS did not produce expected output in {self.output_dir}"
            )

        logger.info(
            f"Verified HGS output: {len(hydrograph_files)} hydrograph file(s), "
            f"{len(head_files)} head file(s)"
        )

    def run(self, **kwargs) -> Optional[Path]:
        """Alternative entry point for HGS execution."""
        return self.run_hydrogeosphere(**kwargs)
