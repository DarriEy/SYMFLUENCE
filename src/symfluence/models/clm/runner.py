"""
CLM Model Runner

Executes CLM5 (cesm.exe) with prepared namelists and input files.
At runtime, CIME is bypassed — cesm.exe is invoked directly from
the run directory where namelists are placed.
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


@ModelRegistry.register_runner("CLM")
class CLMRunner(BaseModelRunner):
    """
    Runs CLM5 via direct cesm.exe invocation.

    Handles:
    - Executable path resolution
    - Namelist copying to run directory
    - Model execution
    - Output verification (*.clm2.h0.*.nc)
    """

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.clm_input_dir = self.project_dir / "CLM_input"
        self.settings_dir = self.clm_input_dir / "settings"
        self.forcing_dir = self.clm_input_dir / "forcing"
        self.params_dir = self.clm_input_dir / "parameters"

    def _get_model_name(self) -> str:
        return "CLM"

    def _get_clm_executable(self) -> Path:
        """Get the CLM (cesm.exe) executable path."""
        return self.get_model_executable(
            install_path_key='CLM_INSTALL_PATH',
            default_install_subpath='installs/clm',
            default_exe_name='cesm.exe',
            typed_exe_accessor=lambda: (
                self.config.model.clm.exe
                if self.config.model and self.config.model.clm
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_timeout(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.clm.timeout,
            default=3600,
        )

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute CLM5.

        Returns:
            Path to output directory on success.

        Raises:
            ModelExecutionError: If execution fails.
        """
        logger.info(f"Running CLM for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "CLM model execution",
            logger,
            error_type=ModelExecutionError,
        ):
            # Setup output directory
            self.output_dir = (
                self.project_dir / "simulations"
                / self.config.domain.experiment_id / "CLM"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            clm_exe = self._get_clm_executable()
            logger.info(f"Using CLM executable: {clm_exe}")

            # Copy namelists to run directory (CLM reads from cwd)
            self._setup_run_directory(self.output_dir)

            # Execute
            cmd = [str(clm_exe)]
            logger.info(f"Executing CLM from: {self.output_dir}")

            env = os.environ.copy()
            # Disable macOS nano malloc zone — causes false heap corruption
            # detection in ESMF's nlohmann::json metadata handling
            env['MallocNanoZone'] = '0'
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
                logger.debug(f"CLM stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"CLM stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"CLM execution returned code {result.returncode}")
                logger.error(
                    f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}"
                )
                raise ModelExecutionError(
                    f"CLM execution failed with return code {result.returncode}"
                )

            logger.info("CLM execution completed successfully")
            self._verify_output()

            return self.output_dir

    def _setup_run_directory(self, run_dir: Path) -> None:
        """Copy all NUOPC runtime files to run directory."""
        # All files that cesm.exe needs in its cwd
        runtime_files = [
            'nuopc.runconfig', 'nuopc.runseq', 'fd.yaml',
            'datm_in', 'datm.streams.xml', 'lnd_in',
            'drv_in', 'drv_flds_in', 'CASEROOT',
            'user_nl_clm',
        ]
        # Create timing dirs that cesm.exe expects
        (run_dir / 'timing' / 'checkpoints').mkdir(parents=True, exist_ok=True)

        for name in runtime_files:
            src = self.settings_dir / name
            if src.exists():
                shutil.copy2(src, run_dir / name)
                logger.debug(f"Copied {name} to run directory")
            else:
                logger.warning(f"Runtime file not found: {src}")

    def _verify_output(self) -> None:
        """Verify CLM produced valid history output files."""
        output_files = list(self.output_dir.glob("*.clm2.h0.*.nc"))

        if not output_files:
            # Check subdirectories
            for subdir in ['run', 'hist', 'results']:
                sub = self.output_dir / subdir
                if sub.exists():
                    output_files = list(sub.glob("*.clm2.h0.*.nc"))
                    if output_files:
                        break

        if not output_files:
            raise RuntimeError(
                f"CLM did not produce expected *.clm2.h0.*.nc output in {self.output_dir}"
            )

        for f in output_files:
            if f.stat().st_size == 0:
                raise RuntimeError(f"CLM output file is empty: {f}")

        logger.info(f"Verified CLM output: {len(output_files)} history file(s)")

    def run_clm(self, **kwargs) -> Optional[Path]:
        """Alternative entry point for CLM execution."""
        return self.run(**kwargs)
