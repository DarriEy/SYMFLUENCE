"""
SWAT Worker

Worker implementation for SWAT model optimization.
"""

import logging
import os
import subprocess
import sys
import re
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.core.constants import ModelDefaults

from ..parameters import PARAM_CHANGE_METHOD, PARAM_FILE_MAP


@OptimizerRegistry.register_worker('SWAT')
class SWATWorker(BaseWorker):
    """
    Worker for SWAT model calibration.

    Handles parameter application to SWAT text files, SWAT execution,
    and metric calculation from output.rch.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SWAT worker.

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
        Apply parameters to SWAT text files in a worker TxtInOut directory.

        Copies fresh input files from the original TxtInOut location,
        then applies parameter changes using the r__, v__, or a__ methods.

        Args:
            params: Parameter values to apply
            settings_dir: Worker-specific TxtInOut directory
            **kwargs: Additional arguments (config)

        Returns:
            True if successful
        """
        try:
            self.logger.debug(f"Applying SWAT parameters to {settings_dir}")

            # Copy fresh files from settings + forcing into worker dir
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'
            swat_settings = project_dir / 'settings' / 'SWAT'
            swat_forcing = project_dir / 'data' / 'forcing' / 'SWAT_input'
            # Legacy fallback
            if not swat_settings.exists():
                swat_forcing_legacy = project_dir / 'forcing' / 'SWAT_input'
                if swat_forcing_legacy.exists():
                    swat_forcing = swat_forcing_legacy

            has_source = swat_settings.exists() or swat_forcing.exists()
            if has_source and (not settings_dir.exists() or settings_dir.resolve() != swat_settings.resolve()):
                settings_dir.mkdir(parents=True, exist_ok=True)
                for src_dir in (swat_settings, swat_forcing):
                    if src_dir.exists():
                        for f in src_dir.iterdir():
                            if f.is_file():
                                shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(f"Assembled SWAT TxtInOut in {settings_dir}")
            elif not settings_dir.exists():
                self.logger.error(
                    f"SWAT settings not found: {swat_settings} "
                    f"and forcing not found: {swat_forcing}"
                )
                return False

            # Apply parameter changes
            return self._update_swat_files(settings_dir, params)

        except Exception as e:
            self.logger.error(f"Error applying SWAT parameters: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_swat_files(
        self,
        txtinout_dir: Path,
        params: Dict[str, float]
    ) -> bool:
        """
        Update SWAT text files with parameter values.

        Groups parameters by file extension and applies changes.

        Args:
            txtinout_dir: Path to TxtInOut directory
            params: Parameters to apply

        Returns:
            True if successful
        """
        try:
            # Group parameters by file extension
            params_by_ext: Dict[str, Dict[str, float]] = {}
            for param_name, value in params.items():
                if param_name not in PARAM_FILE_MAP:
                    continue
                ext = PARAM_FILE_MAP[param_name]
                if ext not in params_by_ext:
                    params_by_ext[ext] = {}
                params_by_ext[ext][param_name] = value

            # Apply to each file type
            for ext, ext_params in params_by_ext.items():
                target_files = list(txtinout_dir.glob(f'*{ext}'))
                for target_file in target_files:
                    self._update_single_file(target_file, ext_params)

            return True

        except Exception as e:
            self.logger.error(f"Error updating SWAT files: {e}")
            return False

    def _update_single_file(
        self,
        file_path: Path,
        params: Dict[str, float]
    ) -> None:
        """
        Update a single SWAT text file with parameter changes.

        Args:
            file_path: Path to the SWAT text file
            params: Parameters to update in this file
        """
        ext = file_path.suffix
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            modified = False
            for param_name, value in params.items():
                if PARAM_FILE_MAP.get(param_name) != ext:
                    continue

                if f'| {param_name}' in line or f'|{param_name}' in line:
                    method = PARAM_CHANGE_METHOD.get(param_name, 'v__')
                    new_line = self._apply_change_to_line(line, param_name, value, method)
                    new_lines.append(new_line)
                    modified = True
                    break

            if not modified:
                new_lines.append(line)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    def _apply_change_to_line(
        self,
        line: str,
        param_name: str,
        value: float,
        method: str
    ) -> str:
        """
        Apply a parameter change to a SWAT text file line.

        Args:
            line: Original file line
            param_name: Parameter name
            value: New value or change amount
            method: Change method ('r__', 'v__', or 'a__')

        Returns:
            Modified line string
        """
        # Preserve trailing newline so SWAT file format is not corrupted
        line_stripped = line.rstrip('\n\r')
        trailing = line[len(line_stripped):]

        match = re.match(
            r'^(\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(.*)', line_stripped
        )
        if not match:
            match = re.match(r'^(\s*)([-+]?\d+)(.*)', line_stripped)

        if not match:
            return line

        prefix = match.group(1)
        original_str = match.group(2)
        suffix = match.group(3)

        try:
            original_value = float(original_str)
        except ValueError:
            return line

        # Apply change method
        if method == 'r__':
            new_value = original_value * (1.0 + value)
        elif method == 'v__':
            new_value = value
        elif method == 'a__':
            new_value = original_value + value
        else:
            new_value = value

        # Format the new value
        if '.' in original_str:
            decimal_places = len(original_str.split('.')[-1])
            new_value_str = f"{new_value:{len(original_str)}.{decimal_places}f}"
        else:
            new_value_str = f"{int(round(new_value)):>{len(original_str)}d}"

        result = f"{prefix}{new_value_str}{suffix}{trailing}"
        # Safety: ensure line always ends with newline to prevent line merging
        if not result.endswith('\n'):
            result += '\n'
        return result

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run SWAT model.

        SWAT is executed from within the TxtInOut directory. The executable
        reads file.cio and produces output.rch.

        Args:
            config: Configuration dictionary
            settings_dir: TxtInOut directory (SWAT runs from here)
            output_dir: Output directory (unused, SWAT writes to TxtInOut)
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            # Use sim_dir as working directory if provided
            work_dir = Path(kwargs.get('sim_dir', settings_dir))

            # Copy files from settings_dir to work_dir if they differ.
            # apply_parameters() modifies files in settings_dir each
            # iteration, so we must ALWAYS overwrite to propagate parameter changes.
            if work_dir.resolve() != settings_dir.resolve() and settings_dir.exists():
                work_dir.mkdir(parents=True, exist_ok=True)
                for f in settings_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, work_dir / f.name)
                # Also copy forcing files if not already present in settings_dir
                domain_name = config.get('DOMAIN_NAME', '')
                data_dir_path = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                project_dir = data_dir_path / f'domain_{domain_name}'
                swat_forcing = project_dir / 'data' / 'forcing' / 'SWAT_input'
                if not swat_forcing.exists():
                    swat_forcing = project_dir / 'forcing' / 'SWAT_input'
                if swat_forcing.exists():
                    for f in swat_forcing.iterdir():
                        if f.is_file():
                            dest = work_dir / f.name
                            if not dest.exists():
                                shutil.copy2(f, dest)

            # Clean up stale output
            self._cleanup_stale_output(work_dir)

            # Get executable
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            swat_exe = self._get_swat_executable(config, data_dir)

            if not swat_exe.exists():
                self.logger.error(f"SWAT executable not found: {swat_exe}")
                return False

            # Verify file.cio exists
            file_cio = work_dir / 'file.cio'
            if not file_cio.exists():
                self.logger.error(f"file.cio not found in {work_dir}")
                return False

            # Build command - SWAT runs from TxtInOut
            cmd = [str(swat_exe)]

            # Fsync the working directory so all copied files are flushed
            # before the Fortran executable reads them (prevents SIGBUS on arm64)
            fd = os.open(str(work_dir), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

            # Set single-threaded environment for Fortran runtime
            env = os.environ.copy()
            env.update({
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
            })

            # Run with timeout
            timeout = config.get('SWAT_TIMEOUT', 300)

            stdout_file = work_dir / 'swat_stdout.log'
            stderr_file = work_dir / 'swat_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                     open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = subprocess.run(
                        cmd,
                        cwd=str(work_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"SWAT timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"SWAT failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output
            output_rch = work_dir / 'output.rch'
            if not output_rch.exists() or output_rch.stat().st_size == 0:
                self._last_error = "output.rch not produced or empty"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running SWAT: {e}")
            return False

    def _get_swat_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get SWAT executable path."""
        install_path = config.get('SWAT_INSTALL_PATH', 'default')
        exe_name = config.get('SWAT_EXE', 'swat_rel.exe')

        if install_path == 'default':
            return data_dir / "installs" / "swat" / "bin" / exe_name

        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        return install_path

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale SWAT output files."""
        for pattern in ['output.*', '*.log']:
            for file_path in output_dir.glob(pattern):
                if file_path.name.startswith('output.'):
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
        Calculate metrics from SWAT output.rch.

        Parses the fixed-width output.rch file, extracts FLOW_OUTcms
        for the outlet reach, and computes KGE/NSE against observations.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            output_rch = sim_dir / 'output.rch'

            if not output_rch.exists():
                self.logger.error(f"output.rch not found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'No output.rch'}

            # Parse output.rch for outlet reach streamflow
            flow_data = self._parse_output_rch(output_rch, reach_id=1)

            if flow_data is None or len(flow_data) == 0:
                return {'kge': self.penalty_score, 'error': 'No flow data parsed'}

            # Get simulation start date
            try:
                start_str = (
                    config.get('EXPERIMENT_TIME_START')
                    or config.get('TIME_START')
                )
                if start_str is None and hasattr(config, 'domain'):
                    start_str = config.domain.time_start
                start_date = pd.to_datetime(start_str).normalize() if start_str else pd.to_datetime('2000-01-01')
            except (AttributeError, TypeError):
                start_date = pd.to_datetime('2000-01-01')

            # Build time series
            # SWAT output.rch already starts AFTER the NYSKIP warmup period
            # (file.cio NYSKIP is set to warmup_years by the preprocessor).
            # We must NOT skip warmup again; just offset the start date.
            warmup_years = int(config.get('SWAT_WARMUP_YEARS', 2))
            output_start_date = start_date + pd.DateOffset(years=warmup_years)
            dates = pd.date_range(start=output_start_date, periods=len(flow_data), freq='D')
            sim_series = pd.Series(flow_data, index=dates)
            self.logger.debug(
                f"SWAT output dates: {sim_series.index[0]} to {sim_series.index[-1]} "
                f"({len(sim_series)} days, warmup={warmup_years}yr already skipped by NYSKIP)"
            )

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
            self.logger.error(f"Error calculating SWAT metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    def _parse_output_rch(
        self,
        output_rch: Path,
        reach_id: int = 1
    ) -> Optional[np.ndarray]:
        """
        Parse SWAT output.rch and extract FLOW_OUTcms for a given reach.

        Args:
            output_rch: Path to output.rch
            reach_id: Reach number to extract

        Returns:
            numpy array of flow values, or None
        """
        try:
            with open(output_rch, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Find header end
            header_end = 0
            for i, line in enumerate(lines):
                if 'RCH' in line.upper() and ('FLOW' in line.upper() or 'MON' in line.upper()):
                    header_end = i + 1
                    break
            if header_end == 0:
                header_end = 9

            # Parse data - handle both formats:
            #   "1  0  1  0.00  0.00  0.00 ..."   (reach number at col 0, flow at col 5)
            #   "REACH  1  0  1  0.00  0.00  0.00 ..."  (REACH prefix, reach at col 1, flow at col 6)
            flow_values = []
            for line in lines[header_end:]:
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    # Detect REACH prefix
                    if parts[0].upper() == 'REACH':
                        rch = int(parts[1])
                        flow_col = 6  # FLOW_OUTcms after REACH prefix
                    else:
                        rch = int(parts[0])
                        flow_col = 5  # FLOW_OUTcms without prefix
                    if rch == reach_id and len(parts) > flow_col:
                        flow_out = float(parts[flow_col])
                        flow_values.append(max(0.0, flow_out))
                except (ValueError, IndexError):
                    continue

            if not flow_values:
                return None

            return np.array(flow_values)

        except Exception as e:
            self.logger.error(f"Error parsing output.rch: {e}")
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_swat_parameters_worker(task_data)


def _evaluate_swat_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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
    })

    # Small random delay
    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = SWATWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'SWAT worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
