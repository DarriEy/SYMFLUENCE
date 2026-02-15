"""
mHM Worker

Worker implementation for mHM model optimization.
"""

import logging
import os
import re
import subprocess
import sys
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.core.constants import ModelDefaults


@OptimizerRegistry.register_worker('MHM')
class MHMWorker(BaseWorker):
    """
    Worker for mHM model calibration.

    Handles parameter application to Fortran namelists, mHM execution,
    and metric calculation from discharge output.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize mHM worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    # Shared utilities
    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to mHM namelist files.

        Uses regex-based parsing to update parameter values in the .nml files.

        Args:
            params: Parameter values to apply
            settings_dir: mHM settings directory containing .nml files
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying mHM parameters to {settings_dir}")

            # Always copy fresh namelists from the original MHM_input location
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings_dir = data_dir / f'domain_{domain_name}' / 'MHM_input' / 'settings'

            if original_settings_dir.exists():
                settings_dir.mkdir(parents=True, exist_ok=True)
                for f in original_settings_dir.glob('*.nml'):
                    shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(f"Copied mHM namelists from {original_settings_dir} to {settings_dir}")
            elif not settings_dir.exists():
                self.logger.error(f"mHM settings directory not found: {settings_dir} "
                                  f"(original also missing: {original_settings_dir})")
                return False

            # Find parameter namelist file -- mHM stores calibration
            # parameters in mhm_parameter.nml (with lower/upper/value/flag/scaling
            # columns).  Fall back to mhm.nml for legacy setups.
            param_namelist = settings_dir / 'mhm_parameter.nml'
            if not param_namelist.exists():
                param_namelist = settings_dir / 'mhm.nml'
            if not param_namelist.exists():
                nml_files = list(settings_dir.glob('*.nml'))
                if nml_files:
                    param_namelist = nml_files[0]
                else:
                    self.logger.error(f"mHM namelist file not found in {settings_dir}")
                    return False

            # Update parameter namelist with new parameter values
            return self._update_namelist_file(param_namelist, params)

        except Exception as e:
            self.logger.error(f"Error applying mHM parameters: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_namelist_file(self, namelist_file: Path, params: Dict[str, float]) -> bool:
        """
        Update mHM Fortran namelist file with new parameter values.

        Supports two formats:

        1. **mhm_parameter.nml** (5-element tuple)::

               paramName = lower, upper, value, FLAG, SCALING

           In this case only the *value* (3rd element) is replaced.

        2. **Legacy / mhm.nml** (single value)::

               paramName = value

        Args:
            namelist_file: Path to namelist (.nml) file
            params: Parameters to update

        Returns:
            True if successful
        """
        try:
            content = namelist_file.read_text(encoding='utf-8')

            # Detect whether this is a 5-column parameter file by checking
            # for the characteristic "lower, upper, value, flag, scaling" pattern
            is_param_nml = 'mhm_parameter' in namelist_file.name

            for param_name, value in params.items():
                formatted_value = self._format_namelist_value(value)

                if is_param_nml:
                    # 5-column format: paramName = lower, upper, VALUE, flag, scaling
                    # Capture: (prefix + lower, upper,) (value) (, flag, scaling)
                    pattern = re.compile(
                        r'(\s*' + re.escape(param_name)
                        + r'\s*=\s*'           # paramName =
                        + r'[^,]+,'             # lower,
                        + r'\s*[^,]+,'          # upper,
                        + r'\s*)'               # whitespace before value
                        + r'([^,]+)'            # VALUE  <-- group to replace
                        + r'(,\s*\d+,\s*\d+)',  # , flag, scaling
                        re.IGNORECASE
                    )
                    match = pattern.search(content)
                    if match:
                        content = pattern.sub(
                            r'\g<1>' + formatted_value + r'\g<3>',
                            content
                        )
                        self.logger.debug(
                            f"Updated {param_name} value = {formatted_value} "
                            f"(5-column format)"
                        )
                    else:
                        self.logger.warning(
                            f"Parameter {param_name} not found in {namelist_file}"
                        )
                else:
                    # Legacy single-value format: paramName = value
                    pattern = re.compile(
                        r'(\s*' + re.escape(param_name) + r'\s*=\s*)([^\s,!/]+)',
                        re.IGNORECASE
                    )
                    match = pattern.search(content)
                    if match:
                        content = pattern.sub(
                            r'\g<1>' + formatted_value,
                            content
                        )
                        self.logger.debug(f"Updated {param_name} = {formatted_value}")
                    else:
                        self.logger.warning(
                            f"Parameter {param_name} not found in {namelist_file}"
                        )

            namelist_file.write_text(content, encoding='utf-8')
            return True

        except Exception as e:
            self.logger.error(f"Error updating mHM namelist: {e}")
            return False

    def _format_namelist_value(self, value: float) -> str:
        """Format a parameter value for Fortran namelist syntax."""
        abs_val = abs(value)
        if abs_val == 0.0:
            return '0.0'
        elif abs_val < 0.001 or abs_val >= 1e6:
            return f'{value:.6e}'
        elif abs_val == int(abs_val) and abs_val < 1e6:
            return f'{value:.1f}'
        else:
            return f'{value:.6f}'

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run mHM model.

        mHM is executed from within the settings directory where it reads
        the mhm.nml and mrm.nml namelist files.

        Args:
            config: Configuration dictionary
            settings_dir: mHM settings directory (with .nml files)
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            # Use sim_dir for output if provided
            mhm_output_dir = Path(kwargs.get('sim_dir', output_dir))
            mhm_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up stale output files
            self._cleanup_stale_output(mhm_output_dir)

            # Get executable
            mhm_exe = self._get_mhm_executable(config, data_dir)
            if not mhm_exe.exists():
                self.logger.error(f"mHM executable not found: {mhm_exe}")
                return False

            # Ensure we have namelist files in settings_dir
            if not (settings_dir / 'mhm.nml').exists():
                self.logger.error(f"mhm.nml not found in {settings_dir}")
                return False

            # Build command - mHM runs from within the settings directory
            cmd = [str(mhm_exe)]

            # Set environment
            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            # Run with timeout
            timeout = config.get('MHM_TIMEOUT', 300)

            stdout_file = mhm_output_dir / 'mhm_stdout.log'
            stderr_file = mhm_output_dir / 'mhm_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                     open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = subprocess.run(
                        cmd,
                        cwd=str(settings_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"mHM timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"mHM failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output - check both output_dir and settings output subdirectory
            output_files = list(mhm_output_dir.glob('discharge_*.nc'))
            if not output_files:
                # mHM may write to output subdirectory under settings
                for subdir in [settings_dir / 'output', settings_dir]:
                    output_files = list(subdir.glob('discharge_*.nc'))
                    if output_files:
                        break

            if not output_files:
                self._last_error = "No discharge output files produced"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running mHM: {e}")
            return False

    def _get_mhm_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get mHM executable path."""
        install_path = config.get('MHM_INSTALL_PATH', 'default')
        exe_name = config.get('MHM_EXE', 'mhm')

        if install_path == 'default':
            return data_dir / "installs" / "mhm" / "bin" / exe_name

        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        return install_path

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale mHM output files."""
        for pattern in ['discharge_*.nc', 'mHM_Fluxes_States_*.nc', '*.log']:
            for file_path in output_dir.glob(pattern):
                try:
                    file_path.unlink()
                except (OSError, IOError):
                    pass

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from mHM output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            settings_dir = kwargs.get('settings_dir', None)

            # Find discharge output file - search multiple locations
            output_files = list(sim_dir.glob('discharge_*.nc'))

            if not output_files and settings_dir:
                for subdir in [Path(settings_dir) / 'output', Path(settings_dir)]:
                    output_files = list(subdir.glob('discharge_*.nc'))
                    if output_files:
                        break

            if not output_files:
                output_files = list(sim_dir.glob('*.nc'))

            if not output_files:
                self.logger.error(f"No mHM discharge files found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_file = output_files[0]

            # Extract discharge
            ds = xr.open_dataset(sim_file)

            # Find discharge variable
            discharge = None
            for var in ['Qsim', 'Q', 'discharge', 'Qrouted']:
                if var in ds:
                    discharge = ds[var]
                    break

            if discharge is None:
                ds.close()
                return {'kge': self.penalty_score, 'error': 'No discharge variable'}

            # Handle spatial dimensions (take first gauge)
            spatial_dims = [d for d in discharge.dims if d not in ['time']]
            if spatial_dims:
                discharge = discharge.isel({spatial_dims[0]: 0})

            # mHM discharge is already in m3/s
            times = pd.to_datetime(ds['time'].values)
            streamflow_m3s = discharge.values.flatten()
            ds.close()

            sim_series = pd.Series(streamflow_m3s, index=times)

            # Skip warmup period
            warmup_days = int(config.get('MHM_WARMUP_DAYS', 365))
            if warmup_days > 0 and len(sim_series) > warmup_days:
                sim_series = sim_series.iloc[warmup_days:]
                self.logger.debug(f"Skipped {warmup_days} warmup days for metric calculation")

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

            # Align and calculate metrics
            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series
            )

            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            return results

        except Exception as e:
            self.logger.error(f"Error calculating mHM metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_mhm_parameters_worker(task_data)


def _evaluate_mhm_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    import os
    import signal
    import random
    import time
    import traceback

    # Set up signal handler
    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    # Force single-threaded execution
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MallocStackLogging': '0',
    })

    # Small random delay
    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = MHMWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'mHM worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
