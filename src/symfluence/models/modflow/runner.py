"""
MODFLOW 6 Model Runner

Executes MODFLOW 6 (mf6) from a prepared simulation directory.
MODFLOW 6 reads mfsim.nam from the current working directory to
discover all model packages and input files.
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


@ModelRegistry.register_runner("MODFLOW", method_name="run_modflow")
class MODFLOWRunner(BaseModelRunner):
    """
    Runs MODFLOW 6 via direct mf6 invocation.

    Handles:
    - Executable path resolution (download or source build)
    - Input file copying to simulation directory
    - Model execution (mf6 reads mfsim.nam from cwd)
    - Output verification (*.hds and *.bud files)
    """

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.settings_dir = self.project_dir / "settings" / "MODFLOW"

    def _get_model_name(self) -> str:
        return "MODFLOW"

    def _get_mf6_executable(self) -> Path:
        """Get the MODFLOW 6 (mf6) executable path."""
        return self.get_model_executable(
            install_path_key='MODFLOW_INSTALL_PATH',
            default_install_subpath='installs/modflow',
            default_exe_name='mf6',
            typed_exe_accessor=lambda: (
                self.config.model.modflow.exe
                if self.config.model and self.config.model.modflow
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_timeout(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.modflow.timeout,
            default=3600,
        )

    def run_modflow(self, sim_dir: Optional[Path] = None, **kwargs) -> Optional[Path]:
        """
        Execute MODFLOW 6.

        Args:
            sim_dir: Optional override for simulation directory. If None,
                     uses standard output path.

        Returns:
            Path to output directory on success.

        Raises:
            ModelExecutionError: If execution fails.
        """
        logger.info(f"Running MODFLOW 6 for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "MODFLOW 6 model execution",
            logger,
            error_type=ModelExecutionError,
        ):
            # Setup output directory
            if sim_dir is None:
                self.output_dir = (
                    self.project_dir / "simulations"
                    / self.config.domain.experiment_id / "MODFLOW"
                )
            else:
                self.output_dir = sim_dir

            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            mf6_exe = self._get_mf6_executable()
            logger.info(f"Using MODFLOW 6 executable: {mf6_exe}")

            # Copy input files to simulation directory
            self._setup_sim_directory(self.output_dir)

            # Execute: MODFLOW 6 reads mfsim.nam from cwd
            cmd = [str(mf6_exe)]
            logger.info(f"Executing MODFLOW 6 from: {self.output_dir}")

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
                logger.debug(f"MODFLOW stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"MODFLOW stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"MODFLOW execution returned code {result.returncode}")
                logger.error(
                    f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}"
                )
                # Also check mfsim.lst for error details
                lst_file = self.output_dir / "mfsim.lst"
                if lst_file.exists():
                    lst_content = lst_file.read_text()
                    # Find error lines
                    error_lines = [
                        l for l in lst_content.splitlines()
                        if 'error' in l.lower() or 'failed' in l.lower()
                    ]
                    if error_lines:
                        logger.error(f"MODFLOW listing errors: {error_lines[-5:]}")

                raise ModelExecutionError(
                    f"MODFLOW 6 execution failed with return code {result.returncode}"
                )

            logger.info("MODFLOW 6 execution completed successfully")
            self._verify_output()

            return self.output_dir

    def _setup_sim_directory(self, sim_dir: Path) -> None:
        """Copy all MODFLOW input files to simulation directory."""
        if not self.settings_dir.exists():
            raise ModelExecutionError(
                f"MODFLOW settings directory not found: {self.settings_dir}. "
                "Run preprocessing first."
            )

        # Copy all MODFLOW input files
        modflow_files = [
            'mfsim.nam', 'gwf.nam', 'gwf.tdis', 'gwf.dis',
            'gwf.ic', 'gwf.npf', 'gwf.sto', 'gwf.rch',
            'gwf.drn', 'gwf.oc', 'gwf.ims', 'recharge.ts',
        ]

        for name in modflow_files:
            src = self.settings_dir / name
            if src.exists():
                shutil.copy2(src, sim_dir / name)
                logger.debug(f"Copied {name} to simulation directory")
            else:
                logger.warning(f"MODFLOW input file not found: {src}")

    def _verify_output(self) -> None:
        """Verify MODFLOW produced valid output files."""
        # Check for head and budget files
        hds_files = list(self.output_dir.glob("*.hds"))
        bud_files = list(self.output_dir.glob("*.bud"))

        if not hds_files:
            raise RuntimeError(
                f"MODFLOW did not produce expected *.hds output in {self.output_dir}"
            )

        if not bud_files:
            logger.warning("MODFLOW did not produce *.bud budget files")

        for f in hds_files:
            if f.stat().st_size == 0:
                raise RuntimeError(f"MODFLOW head output file is empty: {f}")

        logger.info(
            f"Verified MODFLOW output: {len(hds_files)} head file(s), "
            f"{len(bud_files)} budget file(s)"
        )

    def run(self, **kwargs) -> Optional[Path]:
        """Alternative entry point for MODFLOW execution."""
        return self.run_modflow(**kwargs)
