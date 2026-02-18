"""
WATFLOOD Worker.

Worker implementation for WATFLOOD model optimization.
"""

import logging
import os
import shutil
import subprocess
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.core.constants import ModelDefaults


@OptimizerRegistry.register_worker('WATFLOOD')
class WATFLOODWorker(BaseWorker):
    """Worker for WATFLOOD model calibration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        """Apply parameters to WATFLOOD .par file."""
        import re
        try:
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_dir = data_dir / f'domain_{domain_name}' / 'WATFLOOD_input' / 'settings'

            if original_dir.exists() and original_dir.resolve() != settings_dir.resolve():
                settings_dir.mkdir(parents=True, exist_ok=True)
                for f in original_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, settings_dir / f.name)

            par_file = config.get('WATFLOOD_PAR_FILE', 'params.par')
            par_path = settings_dir / par_file
            if not par_path.exists():
                self.logger.error(f"WATFLOOD .par file not found: {par_path}")
                return False

            content = par_path.read_text(encoding='utf-8')
            for param_name, value in params.items():
                pattern = re.compile(
                    rf'^(\s*{re.escape(param_name)}\s+)([\d.eE+\-]+)',
                    re.MULTILINE | re.IGNORECASE
                )
                content = pattern.sub(lambda m: f"{m.group(1)}{value:.6f}", content)

            par_path.write_text(content, encoding='utf-8')
            return True
        except Exception as e:
            self.logger.error(f"Error applying WATFLOOD parameters: {e}")
            return False

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        """Run WATFLOOD model from working directory."""
        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            sim_dir.mkdir(parents=True, exist_ok=True)

            install_path = config.get('WATFLOOD_INSTALL_PATH', 'default')
            exe_name = config.get('WATFLOOD_EXE', 'watflood')
            if install_path == 'default':
                watflood_exe = data_dir / "installs" / "watflood" / "bin" / exe_name
            else:
                p = Path(install_path)
                watflood_exe = p / exe_name if p.is_dir() else p

            if not watflood_exe.exists():
                self.logger.error(f"WATFLOOD executable not found: {watflood_exe}")
                return False

            # Copy exe to working dir (MESH pattern)
            work_exe = settings_dir / watflood_exe.name
            if not work_exe.exists():
                shutil.copy2(watflood_exe, work_exe)
                work_exe.chmod(0o755)

            cmd = [str(work_exe)]
            env = os.environ.copy()
            env['MallocStackLogging'] = '0'
            timeout = config.get('WATFLOOD_TIMEOUT', 3600)

            try:
                with open(sim_dir / 'watflood_stdout.log', 'w') as out, \
                     open(sim_dir / 'watflood_stderr.log', 'w') as err:
                    result = subprocess.run(
                        cmd, cwd=str(settings_dir), env=env,
                        stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"WATFLOOD timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"WATFLOOD failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Collect outputs
            for pattern in ['*.tb0', '*.csv']:
                for f in settings_dir.glob(pattern):
                    if f.is_file():
                        shutil.copy2(f, sim_dir / f.name)

            output_files = list(sim_dir.glob('*.tb0')) + list(sim_dir.glob('*.csv'))
            if not output_files:
                self._last_error = "No WATFLOOD output files produced"
                self.logger.error(self._last_error)
                return False
            return True
        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running WATFLOOD: {e}")
            return False

    def calculate_metrics(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate metrics from WATFLOOD output."""
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))

            # Find output files
            tb0_files = list(sim_dir.glob('spl*.tb0')) + list(sim_dir.glob('resin*.tb0'))
            csv_files = list(sim_dir.glob('*.csv'))
            output_files = tb0_files + csv_files

            if not output_files:
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_series = self._extract_streamflow(output_files[0])
            if sim_series is None or len(sim_series) == 0:
                return {'kge': self.penalty_score, 'error': 'No streamflow data'}

            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'

            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            obs_series = pd.Series(obs_values, index=obs_index)
            cal_period_str = config.get('CALIBRATION_PERIOD', '')
            cal_period_tuple = None
            if cal_period_str and ',' in str(cal_period_str):
                parts = str(cal_period_str).split(',')
                cal_period_tuple = (parts[0].strip(), parts[1].strip())

            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series, calibration_period=cal_period_tuple
            )
            return self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
        except Exception as e:
            self.logger.error(f"Error calculating WATFLOOD metrics: {e}")
            return {'kge': self.penalty_score, 'error': str(e)}

    def _extract_streamflow(self, output_file: Path) -> Optional[pd.Series]:
        """Extract streamflow from WATFLOOD output."""
        try:
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
                for col in df.columns:
                    if any(v in col.lower() for v in ['qo', 'qsim', 'flow']):
                        return df[col]
            elif output_file.suffix == '.tb0':
                lines = output_file.read_text(encoding='utf-8').strip().split('\n')
                data_start = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith((':','#')) and stripped[0].isdigit():
                        data_start = i
                        break
                dates, values = [], []
                for line in lines[data_start:]:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                            value = float(parts[4])
                            dates.append(pd.Timestamp(year=year, month=month, day=day))
                            values.append(value)
                        except (ValueError, IndexError):
                            continue
                if dates:
                    return pd.Series(values, index=dates, name='WATFLOOD_discharge_cms')
            return None
        except Exception as e:
            self.logger.error(f"Error extracting WATFLOOD streamflow: {e}")
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_watflood_parameters_worker(task_data)


def _evaluate_watflood_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for process pool execution."""
    import os
    import signal
    import random
    import time
    import traceback

    def signal_handler(signum, frame):
        sys.exit(1)
    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    os.environ.update({
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MallocStackLogging': '0',
    })
    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = WATFLOODWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'WATFLOOD worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
