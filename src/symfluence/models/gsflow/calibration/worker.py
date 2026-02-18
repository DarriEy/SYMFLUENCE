"""
GSFLOW Worker.

Worker implementation for GSFLOW model optimization.
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


@OptimizerRegistry.register_worker('GSFLOW')
class GSFLOWWorker(BaseWorker):
    """Worker for GSFLOW model calibration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        """Apply parameters to GSFLOW PRMS params.dat and MODFLOW UPW package."""
        try:
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings_dir = data_dir / f'domain_{domain_name}' / 'GSFLOW_input' / 'settings'

            # Copy entire settings tree (including modflow/ subdir) to sim dir
            if original_settings_dir.exists() and original_settings_dir.resolve() != settings_dir.resolve():
                if settings_dir.exists():
                    shutil.rmtree(settings_dir)
                shutil.copytree(original_settings_dir, settings_dir)

            # Ensure output directories exist inside the sim settings dir
            # (control file uses relative paths, so output/ must be under settings_dir)
            for sub in ('output/modflow', 'output/prms'):
                (settings_dir / sub).mkdir(parents=True, exist_ok=True)

            # Update PRMS params
            param_file_name = config.get('GSFLOW_PARAMETER_FILE', 'params.dat')
            param_file = settings_dir / param_file_name
            if param_file.exists():
                prms_params = {k: v for k, v in params.items() if k not in ('K', 'SY')}
                if prms_params:
                    self._update_parameter_file(param_file, prms_params)

            # Update MODFLOW UPW params (K, SY) in modflow/ subdir
            modflow_params = {k: v for k, v in params.items() if k in ('K', 'SY')}
            if modflow_params:
                modflow_dir = settings_dir / 'modflow'
                if not modflow_dir.exists():
                    modflow_dir = settings_dir  # fallback
                self._update_upw_parameters(modflow_dir, modflow_params)

            return True
        except Exception as e:
            self.logger.error(f"Error applying GSFLOW parameters: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _update_upw_parameters(self, search_dir: Path, params: Dict[str, float]) -> bool:
        """Update K and SY in UPW package file.

        UPW layout (after header + layer flags):
          CONSTANT <HK>      # line index 7
          CONSTANT <VKA>     # line index 8
          CONSTANT <SS>      # line index 9
          CONSTANT <SY>      # line index 10
        """
        upw_files = list(search_dir.glob('*.upw'))
        if not upw_files:
            self.logger.warning(f"No UPW package file found in {search_dir}")
            return True

        try:
            upw_file = upw_files[0]
            lines = upw_file.read_text(encoding='utf-8').split('\n')

            # Find the CONSTANT lines after the layer flag lines
            const_indices = [i for i, l in enumerate(lines)
                             if l.strip().upper().startswith('CONSTANT')]

            # Expected: HK=0, VKA=1, SS=2, SY=3 (in order of CONSTANT lines)
            if len(const_indices) >= 4:
                if 'K' in params:
                    lines[const_indices[0]] = f"  CONSTANT  {params['K']:.6e}"
                    lines[const_indices[1]] = f"  CONSTANT  {params['K']:.6e}"  # VKA = HK
                if 'SY' in params:
                    lines[const_indices[3]] = f"  CONSTANT  {params['SY']:.6e}"

            upw_file.write_text('\n'.join(lines), encoding='utf-8')
            self.logger.info(f"Updated MODFLOW UPW: {params}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating UPW: {e}")
            return False

    def _update_parameter_file(self, param_file: Path, params: Dict[str, float]) -> bool:
        """Update PRMS ####-delimited parameter file."""
        try:
            content = param_file.read_text(encoding='utf-8')
            blocks = content.split('####\n')
            updated_blocks = []
            for block in blocks:
                updated_block = block
                for param_name, value in params.items():
                    lines = block.strip().split('\n')
                    for line in lines:
                        if line.strip() == param_name:
                            updated_block = self._replace_block_values(block, param_name, value)
                            break
                updated_blocks.append(updated_block)
            param_file.write_text('####\n'.join(updated_blocks), encoding='utf-8')
            return True
        except Exception as e:
            self.logger.error(f"Error updating parameter file: {e}")
            return False

    def _replace_block_values(self, block: str, param_name: str, value: float) -> str:
        stripped = block.strip()
        lines = stripped.split('\n')
        param_idx = None
        for i, line in enumerate(lines):
            if line.strip() == param_name:
                param_idx = i
                break
        if param_idx is None:
            return block
        try:
            dim_size = int(lines[param_idx + 3].strip())
        except (IndexError, ValueError):
            dim_size = 1
        value_start = param_idx + 5
        formatted = f"{value:.6f}"
        new_lines = lines[:value_start]
        for _ in range(dim_size):
            new_lines.append(formatted)
        remaining = value_start + dim_size
        if remaining < len(lines):
            new_lines.extend(lines[remaining:])
        result = '\n'.join(new_lines)
        if block.endswith('\n'):
            result += '\n'
        return result

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        """Run GSFLOW model."""
        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            sim_dir.mkdir(parents=True, exist_ok=True)

            install_path = config.get('GSFLOW_INSTALL_PATH', 'default')
            exe_name = config.get('GSFLOW_EXE', 'gsflow')
            if install_path == 'default':
                gsflow_exe = data_dir / "installs" / "gsflow" / "bin" / exe_name
            else:
                p = Path(install_path)
                gsflow_exe = p / exe_name if p.is_dir() else p

            if not gsflow_exe.exists():
                self.logger.error(f"GSFLOW executable not found: {gsflow_exe}")
                return False

            control_file = config.get('GSFLOW_CONTROL_FILE', 'control.dat')
            control_path = settings_dir / control_file
            if not control_path.exists():
                self.logger.error(f"GSFLOW control file not found: {control_path}")
                return False

            cmd = [str(gsflow_exe), str(control_path)]
            env = os.environ.copy()
            env['MallocStackLogging'] = '0'
            timeout = config.get('GSFLOW_TIMEOUT', 7200)

            try:
                with open(sim_dir / 'gsflow_stdout.log', 'w') as out, \
                     open(sim_dir / 'gsflow_stderr.log', 'w') as err:
                    result = subprocess.run(
                        cmd, cwd=str(settings_dir), env=env,
                        stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"GSFLOW timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"GSFLOW failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Check for output (relative paths: statvar.dat in settings_dir)
            output_files = (
                list(settings_dir.glob('statvar*')) +
                list(sim_dir.glob('statvar*')) +
                list(settings_dir.glob('output/*.csv'))
            )
            if not output_files:
                self._last_error = "No GSFLOW output files produced"
                self.logger.error(self._last_error)
                return False
            return True
        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running GSFLOW: {e}")
            return False

    def calculate_metrics(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate metrics from GSFLOW output."""
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            settings_dir = kwargs.get('settings_dir', None)

            output_files = list(sim_dir.glob('statvar*'))
            if not output_files and settings_dir:
                output_files = list(Path(settings_dir).glob('statvar*'))
            if not output_files:
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_series = self._extract_streamflow_from_statvar(output_files[0])
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
            self.logger.error(f"Error calculating GSFLOW metrics: {e}")
            return {'kge': self.penalty_score, 'error': str(e)}

    def _extract_streamflow_from_statvar(self, statvar_file: Path) -> Optional[pd.Series]:
        """Extract streamflow from GSFLOW statvar file."""
        try:
            lines = statvar_file.read_text(encoding='utf-8').strip().split('\n')
            dates, values = [], []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 8:
                    try:
                        year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                        # basin_cfs is the first statvar (index 7 after
                        # id year month day hour min sec)
                        streamflow_cfs = float(parts[7])
                        dates.append(pd.Timestamp(year=year, month=month, day=day))
                        values.append(streamflow_cfs * 0.0283168)  # cfs → cms
                    except (ValueError, IndexError):
                        continue
            if dates:
                return pd.Series(values, index=dates, name='GSFLOW_discharge_cms')
            return None
        except Exception as e:
            self.logger.error(f"Error extracting GSFLOW streamflow: {e}")
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_gsflow_parameters_worker(task_data)


def _evaluate_gsflow_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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
        worker = GSFLOWWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'GSFLOW worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
