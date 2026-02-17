"""
WRF-Hydro Worker

Worker implementation for WRF-Hydro model optimization.
"""

import logging
import os
import re
import subprocess
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.core.constants import ModelDefaults


@OptimizerRegistry.register_worker('WRFHYDRO')
class WRFHydroWorker(BaseWorker):
    """
    Worker for WRF-Hydro model calibration.

    Handles parameter application to Fortran namelists, WRF-Hydro execution,
    and metric calculation from CHRTOUT output.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to WRF-Hydro namelist files.

        Copies fresh namelists from the original WRFHydro_input location,
        then updates values in hydro.namelist and namelist.hrldas.

        Args:
            params: Parameter values to apply
            settings_dir: WRF-Hydro settings directory
            **kwargs: Additional arguments (config)

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying WRF-Hydro parameters to {settings_dir}")

            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings_dir = data_dir / f'domain_{domain_name}' / 'WRFHydro_input' / 'settings'

            if original_settings_dir.exists() and original_settings_dir.resolve() != settings_dir.resolve():
                settings_dir.mkdir(parents=True, exist_ok=True)
                for pattern in ['*.namelist', 'namelist.*', '*.nc']:
                    for f in original_settings_dir.glob(pattern):
                        shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(
                    f"Copied WRF-Hydro files from {original_settings_dir} to {settings_dir}"
                )
            elif not settings_dir.exists():
                self.logger.error(
                    f"WRF-Hydro settings directory not found: {settings_dir} "
                    f"(original also missing: {original_settings_dir})"
                )
                return False

            # Separate params by target file
            from ..parameters import WRFHYDRO_PARAM_TARGETS

            hydro_params = {}
            hrldas_params = {}

            for param_name, value in params.items():
                target_info = WRFHYDRO_PARAM_TARGETS.get(param_name, {})
                target = target_info.get('target', 'hydro_namelist')
                if target == 'hrldas_namelist':
                    hrldas_params[param_name] = value
                else:
                    hydro_params[param_name] = value

            success = True

            # Update hydro.namelist
            hydro_file = config.get('WRFHYDRO_HYDRO_NAMELIST', 'hydro.namelist')
            hydro_path = settings_dir / hydro_file
            if hydro_params and hydro_path.exists():
                if not self._update_namelist_file(hydro_path, hydro_params):
                    success = False

            # Update namelist.hrldas
            hrldas_file = config.get('WRFHYDRO_NAMELIST_FILE', 'namelist.hrldas')
            hrldas_path = settings_dir / hrldas_file
            if hrldas_params and hrldas_path.exists():
                if not self._update_namelist_file(hrldas_path, hrldas_params):
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Error applying WRF-Hydro parameters: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_namelist_file(self, namelist_file: Path, params: Dict[str, float]) -> bool:
        """
        Update a Fortran namelist file with new parameter values.

        Args:
            namelist_file: Path to namelist file
            params: Parameters to update

        Returns:
            True if successful
        """
        try:
            content = namelist_file.read_text(encoding='utf-8')

            for param_name, value in params.items():
                formatted = self._format_namelist_value(value)

                pattern = re.compile(
                    r'(\s*' + re.escape(param_name) + r'\s*=\s*)([^\s,!/\n]+)',
                    re.IGNORECASE
                )
                match = pattern.search(content)

                if match:
                    content = pattern.sub(r'\g<1>' + formatted, content)
                    self.logger.debug(f"Updated {param_name} = {formatted}")
                else:
                    # Insert before closing '/'
                    insert_line = f" {param_name} = {formatted}\n"
                    slash_pattern = re.compile(r'^(\s*/\s*)$', re.MULTILINE)
                    slash_match = slash_pattern.search(content)
                    if slash_match:
                        content = (content[:slash_match.start()]
                                   + insert_line
                                   + content[slash_match.start():])
                        self.logger.debug(f"Inserted {param_name} = {formatted}")
                    else:
                        self.logger.warning(
                            f"Could not place {param_name} in {namelist_file}"
                        )

            namelist_file.write_text(content, encoding='utf-8')
            return True

        except Exception as e:
            self.logger.error(f"Error updating WRF-Hydro namelist: {e}")
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
        Run WRF-Hydro model.

        WRF-Hydro is executed from within the output directory where it
        reads namelist files (copied to cwd before execution).

        Args:
            config: Configuration dictionary
            settings_dir: WRF-Hydro settings directory (with namelists)
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        import shutil

        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            wrfhydro_output_dir = Path(kwargs.get('sim_dir', output_dir))
            wrfhydro_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean stale output
            self._cleanup_stale_output(wrfhydro_output_dir)

            # Get executable
            wrfhydro_exe = self._get_wrfhydro_executable(config, data_dir)
            if not wrfhydro_exe.exists():
                self.logger.error(f"WRF-Hydro executable not found: {wrfhydro_exe}")
                return False

            # Copy namelists and domain files to working directory
            for pattern in ['*.namelist', 'namelist.*', '*.nc']:
                for f in settings_dir.glob(pattern):
                    shutil.copy2(f, wrfhydro_output_dir / f.name)

            # Also copy routing files if they exist
            domain_name = config.get('DOMAIN_NAME', '')
            routing_dir = data_dir / f'domain_{domain_name}' / 'WRFHydro_input' / 'routing'
            if routing_dir.exists():
                for f in routing_dir.glob('*.nc'):
                    shutil.copy2(f, wrfhydro_output_dir / f.name)

            cmd = [str(wrfhydro_exe)]

            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            timeout = config.get('WRFHYDRO_TIMEOUT', 7200)

            stdout_file = wrfhydro_output_dir / 'wrfhydro_stdout.log'
            stderr_file = wrfhydro_output_dir / 'wrfhydro_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                     open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = subprocess.run(
                        cmd,
                        cwd=str(wrfhydro_output_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"WRF-Hydro timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"WRF-Hydro failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output
            output_files = (
                list(wrfhydro_output_dir.glob('*CHRTOUT*')) +
                list(wrfhydro_output_dir.glob('*LDASOUT*'))
            )

            if not output_files:
                self._last_error = "No CHRTOUT or LDASOUT output files produced"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running WRF-Hydro: {e}")
            return False

    def _get_wrfhydro_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get WRF-Hydro executable path."""
        install_path = config.get('WRFHYDRO_INSTALL_PATH', 'default')
        exe_name = config.get('WRFHYDRO_EXE', 'wrf_hydro.exe')

        if install_path == 'default':
            return data_dir / "installs" / "wrfhydro" / "bin" / exe_name

        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        return install_path

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale WRF-Hydro output files."""
        for pattern in ['*CHRTOUT*', '*LDASOUT*', '*CHANOBS*', '*.log']:
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
        Calculate metrics from WRF-Hydro output.

        Extracts streamflow from CHRTOUT NetCDF files, loads observations,
        aligns time series, and computes KGE/NSE.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            import xarray as xr

            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            settings_dir = kwargs.get('settings_dir', None)

            # Find CHRTOUT output files
            output_files = sorted(sim_dir.glob('*CHRTOUT*'))

            if not output_files and settings_dir:
                output_files = sorted(Path(settings_dir).glob('*CHRTOUT*'))

            if not output_files:
                domain_name = config.get('DOMAIN_NAME', '')
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                wrfhydro_output = data_dir / f'domain_{domain_name}' / 'WRFHydro_input'
                if wrfhydro_output.exists():
                    output_files = sorted(wrfhydro_output.glob('*CHRTOUT*'))

            if not output_files:
                self.logger.error(f"No WRF-Hydro CHRTOUT files found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'No output files'}

            # Extract streamflow from CHRTOUT files
            dates = []
            flows = []
            for chrt_file in output_files:
                try:
                    ds = xr.open_dataset(chrt_file)
                    # Find streamflow variable
                    flow_var = None
                    for var in ['streamflow', 'q_lateral', 'qSfcLatRunoff', 'qBucket']:
                        if var in ds:
                            flow_var = var
                            break

                    if flow_var is None:
                        ds.close()
                        continue

                    flow_data = ds[flow_var]

                    # Handle spatial dims (take outlet / sum)
                    spatial_dims = [d for d in flow_data.dims if d not in ['time', 'reference_time']]
                    if spatial_dims:
                        flow_data = flow_data.isel({spatial_dims[0]: -1})

                    if 'time' in ds:
                        time_val = pd.to_datetime(ds['time'].values)
                        if hasattr(time_val, '__len__'):
                            for t, v in zip(time_val, flow_data.values.flatten()):
                                dates.append(t)
                                flows.append(float(v))
                        else:
                            dates.append(time_val)
                            flows.append(float(flow_data.values.flatten()[0]))
                    else:
                        # Extract time from filename (YYYYMMDDHHMM.CHRTOUT_DOMAIN1)
                        fname = chrt_file.name
                        try:
                            time_str = fname.split('.')[0]
                            date = pd.Timestamp(time_str[:8])
                            dates.append(date)
                            flows.append(float(flow_data.values.flatten()[0]))
                        except (ValueError, IndexError):
                            pass

                    ds.close()

                except Exception as e:
                    self.logger.debug(f"Error reading {chrt_file}: {e}")
                    continue

            if not dates:
                return {'kge': self.penalty_score, 'error': 'No streamflow data extracted'}

            # WRF-Hydro streamflow is already in m3/s
            sim_series = pd.Series(flows, index=dates, name='WRFHYDRO_discharge_cms')
            sim_series = sim_series.resample('D').mean()  # Daily average

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

            # Parse calibration period
            cal_period_str = config.get('CALIBRATION_PERIOD', '')
            cal_period_tuple = None
            if cal_period_str and ',' in str(cal_period_str):
                parts = str(cal_period_str).split(',')
                cal_period_tuple = (parts[0].strip(), parts[1].strip())

            # Align and calculate metrics
            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series, calibration_period=cal_period_tuple
            )

            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            return results

        except Exception as e:
            self.logger.error(f"Error calculating WRF-Hydro metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_wrfhydro_parameters_worker(task_data)


def _evaluate_wrfhydro_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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

    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MallocStackLogging': '0',
    })

    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = WRFHydroWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'WRF-Hydro worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
