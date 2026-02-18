"""
GSFLOW Model Runner.

Executes the GSFLOW binary which internally couples PRMS and MODFLOW-NWT.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from symfluence.models.base import BaseModelRunner

logger = logging.getLogger(__name__)


class GSFLOWRunner(BaseModelRunner):
    """Runner for the GSFLOW coupled model."""

    def _get_model_name(self) -> str:
        """Return the model name."""
        return 'GSFLOW'

    def run(self, **kwargs) -> Optional[Path]:
        """Run GSFLOW model.

        Executes the single gsflow binary with a combined control file.
        GSFLOW internally manages PRMS-MODFLOW-NWT coupling.
        """
        try:
            settings_dir = self.project_dir / 'GSFLOW_input' / 'settings'
            output_dir = self.project_dir / 'GSFLOW_output'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            gsflow_exe = self._get_executable()
            if not gsflow_exe.exists():
                logger.error(f"GSFLOW executable not found: {gsflow_exe}")
                return None

            # Get control file
            control_file = self._get_config_value(
                lambda: self.config.model.gsflow.control_file,
                default='control.dat',
                dict_key='GSFLOW_CONTROL_FILE'
            )
            control_path = settings_dir / control_file
            if not control_path.exists():
                logger.error(f"GSFLOW control file not found: {control_path}")
                return None

            # Build command
            cmd = [str(gsflow_exe), str(control_path)]

            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            timeout = self._get_config_value(
                lambda: self.config.model.gsflow.timeout,
                default=7200,
                dict_key='GSFLOW_TIMEOUT'
            )

            stdout_file = output_dir / 'gsflow_stdout.log'
            stderr_file = output_dir / 'gsflow_stderr.log'

            logger.info(f"Running GSFLOW: {' '.join(cmd)}")

            with open(stdout_file, 'w') as stdout_f, \
                 open(stderr_file, 'w') as stderr_f:
                result = subprocess.run(
                    cmd,
                    cwd=str(settings_dir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    timeout=timeout
                )

            if result.returncode != 0:
                logger.error(f"GSFLOW failed with return code {result.returncode}")
                return None

            # Verify output
            if self._verify_output(settings_dir, output_dir):
                return output_dir
            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"GSFLOW timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error running GSFLOW: {e}")
            return None

    def _get_executable(self) -> Path:
        """Get GSFLOW executable path."""
        install_path = self._get_config_value(
            lambda: self.config.model.gsflow.install_path,
            default='default',
            dict_key='GSFLOW_INSTALL_PATH'
        )
        exe_name = self._get_config_value(
            lambda: self.config.model.gsflow.exe,
            default='gsflow',
            dict_key='GSFLOW_EXE'
        )

        if install_path == 'default':
            return self.data_dir / "installs" / "gsflow" / "bin" / exe_name

        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        return install_path

    def _verify_output(self, settings_dir: Path, output_dir: Path) -> bool:
        """Verify GSFLOW produced output files."""
        output_files = (
            list(settings_dir.glob('statvar*')) +
            list(output_dir.glob('statvar*')) +
            list(settings_dir.glob('*.csv')) +
            list(output_dir.glob('*.csv'))
        )

        if not output_files:
            logger.error("No GSFLOW output files produced")
            return False

        logger.info(f"GSFLOW output verified: {len(output_files)} files")
        return True
