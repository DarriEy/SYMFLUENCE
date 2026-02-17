"""
PIHM Worker

Worker implementation for PIHM model optimization.
Handles parameter application, PIHM execution, and metric calculation.
"""

import logging
import os
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.core.constants import ModelDefaults


@OptimizerRegistry.register_worker('PIHM')
class PIHMWorker(BaseWorker):
    """
    Worker for PIHM model calibration.

    Handles parameter application to PIHM input files,
    PIHM execution, and streamflow metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, logger)

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs,
    ) -> bool:
        """Apply parameters to PIHM input files."""
        try:
            self.logger.debug(f"Applying PIHM parameters to {settings_dir}")

            from .parameter_manager import PIHMParameterManager

            config = kwargs.get('config', self.config) or {}
            pm = PIHMParameterManager(config, self.logger, settings_dir)
            return pm.update_model_files(params)

        except Exception as e:
            self.logger.error(f"Error applying PIHM parameters: {e}")
            return False

    def run_model(
        self,
        config: Dict,
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """Execute PIHM for calibration."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Clean stale output
            for f in output_dir.glob('*.rivflx*'):
                f.unlink()
            for f in output_dir.glob('*.gwhead*'):
                f.unlink()

            # Copy input files
            settings_dir = Path(settings_dir)
            for src in settings_dir.iterdir():
                if src.is_file():
                    shutil.copy2(src, output_dir / src.name)

            # Get executable
            install_path = config.get('PIHM_INSTALL_PATH', 'default')
            if install_path == 'default':
                code_dir = Path(config.get('SYMFLUENCE_CODE_DIR', '.'))
                install_path = str(code_dir.parent / (code_dir.name + '_data') / 'installs' / 'pihm')

            exe_name = config.get('PIHM_EXE', 'pihm')
            pihm_exe = Path(install_path) / 'bin' / exe_name
            if not pihm_exe.exists():
                pihm_exe = Path(install_path) / exe_name
            if not pihm_exe.exists():
                self.logger.error(f"PIHM executable not found: {pihm_exe}")
                return False

            project_name = "pihm_lumped"
            timeout = int(config.get('PIHM_TIMEOUT', 3600))

            env = os.environ.copy()
            result = subprocess.run(
                [str(pihm_exe), project_name],
                cwd=str(output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                self.logger.error(
                    f"PIHM failed (rc={result.returncode}): "
                    f"{result.stderr[-300:] if result.stderr else 'no stderr'}"
                )
                return False

            # Verify output
            rivflx_files = list(output_dir.glob('*.rivflx*'))
            if not rivflx_files:
                self.logger.error("No PIHM river flux output files produced")
                return False

            self.logger.debug(f"PIHM run complete: {len(rivflx_files)} output file(s)")
            return True

        except Exception as e:
            self.logger.error(f"PIHM execution error: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict,
        **kwargs,
    ) -> Dict:
        """Calculate streamflow metrics from PIHM output."""
        try:
            output_dir = Path(output_dir)

            from symfluence.models.pihm.extractor import PIHMResultExtractor
            extractor = PIHMResultExtractor()

            start_date = str(config.get('EXPERIMENT_TIME_START', '2000-01-01'))

            river_flux = extractor.extract_variable(
                output_dir, 'river_flux', start_date=start_date,
            )

            if river_flux.empty:
                return {'kge': self.penalty_score, 'error': 'No PIHM output'}

            sim_series = river_flux

            # Load observations
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'

            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            obs_series = pd.Series(obs_values, index=obs_index)

            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series
            )

            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            return results

        except Exception as e:
            self.logger.error(f"Error calculating PIHM metrics: {e}")
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_pihm_parameters_worker(task_data)


def _evaluate_pihm_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution."""
    import traceback

    try:
        worker = PIHMWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'PIHM worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1),
        }
