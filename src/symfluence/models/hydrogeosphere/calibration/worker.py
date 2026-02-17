"""
HydroGeoSphere Worker

Worker implementation for HGS model optimization.
Handles parameter application, grok+hgs execution, and metric calculation.
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


@OptimizerRegistry.register_worker('HYDROGEOSPHERE')
class HGSWorker(BaseWorker):
    """
    Worker for HGS model calibration.

    Handles parameter application to HGS input files,
    two-step grok+hgs execution, and streamflow metric calculation.
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
        """Apply parameters to HGS input files."""
        try:
            self.logger.debug(f"Applying HGS parameters to {settings_dir}")

            from .parameter_manager import HGSParameterManager

            config = kwargs.get('config', self.config) or {}
            pm = HGSParameterManager(config, self.logger, settings_dir)
            return pm.update_model_files(params)

        except Exception as e:
            self.logger.error(f"Error applying HGS parameters: {e}")
            return False

    def run_model(
        self,
        config: Dict,
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """Execute HGS (grok + hgs) for calibration."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Clean stale output
            for f in output_dir.glob('*hydrograph*'):
                f.unlink()
            for f in output_dir.glob('*head*'):
                f.unlink()

            # Copy input files
            settings_dir = Path(settings_dir)
            for src in settings_dir.iterdir():
                if src.is_file():
                    shutil.copy2(src, output_dir / src.name)

            # Get executables
            install_path = config.get('HGS_INSTALL_PATH', 'default')
            if install_path == 'default':
                code_dir = Path(config.get('SYMFLUENCE_CODE_DIR', '.'))
                install_path = str(code_dir.parent / (code_dir.name + '_data') / 'installs' / 'hydrogeosphere')

            hgs_exe_name = config.get('HGS_EXE', 'hgs')
            grok_exe_name = config.get('HGS_GROK_EXE', 'grok')

            hgs_exe = Path(install_path) / 'bin' / hgs_exe_name
            grok_exe = Path(install_path) / 'bin' / grok_exe_name

            if not hgs_exe.exists():
                hgs_exe = Path(install_path) / hgs_exe_name
            if not grok_exe.exists():
                grok_exe = Path(install_path) / grok_exe_name

            if not hgs_exe.exists():
                self.logger.error(f"HGS executable not found: {hgs_exe}")
                return False
            if not grok_exe.exists():
                self.logger.error(f"grok executable not found: {grok_exe}")
                return False

            # Read prefix
            pfx_file = output_dir / 'batch.pfx'
            prefix = pfx_file.read_text().strip() if pfx_file.exists() else 'hgs_lumped'

            timeout = int(config.get('HGS_TIMEOUT', 7200))
            env = os.environ.copy()

            # Step 1: Run grok
            grok_result = subprocess.run(
                [str(grok_exe), prefix],
                cwd=str(output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout // 4,
            )

            if grok_result.returncode != 0:
                self.logger.error(
                    f"grok failed (rc={grok_result.returncode}): "
                    f"{grok_result.stderr[-300:] if grok_result.stderr else 'no stderr'}"
                )
                return False

            # Step 2: Run HGS
            hgs_result = subprocess.run(
                [str(hgs_exe), prefix],
                cwd=str(output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if hgs_result.returncode != 0:
                self.logger.error(
                    f"HGS failed (rc={hgs_result.returncode}): "
                    f"{hgs_result.stderr[-300:] if hgs_result.stderr else 'no stderr'}"
                )
                return False

            # Verify output
            hydrograph_files = list(output_dir.glob('*hydrograph*'))
            if not hydrograph_files:
                self.logger.error("No HGS hydrograph output files produced")
                return False

            self.logger.debug(f"HGS run complete: {len(hydrograph_files)} output file(s)")
            return True

        except Exception as e:
            self.logger.error(f"HGS execution error: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict,
        **kwargs,
    ) -> Dict:
        """Calculate streamflow metrics from HGS output."""
        try:
            output_dir = Path(output_dir)

            from symfluence.models.hydrogeosphere.extractor import HGSResultExtractor
            extractor = HGSResultExtractor()

            start_date = str(config.get('EXPERIMENT_TIME_START', '2000-01-01'))

            hydrograph = extractor.extract_variable(
                output_dir, 'hydrograph', start_date=start_date,
            )

            if hydrograph.empty:
                return {'kge': self.penalty_score, 'error': 'No HGS output'}

            sim_series = hydrograph

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
            self.logger.error(f"Error calculating HGS metrics: {e}")
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_hgs_parameters_worker(task_data)


def _evaluate_hgs_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution."""
    import traceback

    try:
        worker = HGSWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'HGS worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1),
        }
