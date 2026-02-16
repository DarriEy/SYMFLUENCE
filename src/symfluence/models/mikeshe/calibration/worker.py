"""
MIKE-SHE Worker

Worker implementation for MIKE-SHE model optimization.
"""

import logging
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET  # nosec B405 - parsing trusted MIKE-SHE setup files
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.core.constants import ModelDefaults


@OptimizerRegistry.register_worker('MIKESHE')
class MIKESHEWorker(BaseWorker):
    """
    Worker for MIKE-SHE model calibration.

    Handles parameter application (XML parsing), model execution
    (MikeSheEngine.exe with optional WINE wrapper), and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize MIKE-SHE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    # Shared utilities
    _streamflow_metrics = StreamflowMetrics()

    # XML path mappings for parameter updates
    PARAM_XML_PATHS = {
        'manning_m': './/OverlandFlow/ManningM',
        'detention_storage': './/OverlandFlow/DetentionStorage',
        'Ks_uz': './/UnsaturatedFlow/HydraulicConductivity',
        'theta_sat': './/UnsaturatedFlow/SaturatedMoistureContent',
        'theta_fc': './/UnsaturatedFlow/FieldCapacity',
        'theta_wp': './/UnsaturatedFlow/WiltingPoint',
        'Ks_sz_h': './/SaturatedFlow/HorizontalConductivity',
        'specific_yield': './/SaturatedFlow/SpecificYield',
        'ddf': './/SnowMelt/DegreeDayFactor',
        'snow_threshold': './/SnowMelt/ThresholdTemperature',
        'max_canopy_storage': './/Vegetation/MaxCanopyStorage',
    }

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to MIKE-SHE .she setup file via XML parsing.

        Args:
            params: Parameter values to apply
            settings_dir: MIKE-SHE settings directory containing .she file
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying MIKE-SHE parameters to {settings_dir}")

            # Always copy fresh from the original MIKESHE_input location
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings_dir = (
                data_dir / f'domain_{domain_name}' / 'MIKESHE_input' / 'settings'
            )

            if original_settings_dir.exists():
                settings_dir.mkdir(parents=True, exist_ok=True)
                for f in original_settings_dir.glob('*.she'):
                    shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(
                    f"Copied MIKE-SHE setup from {original_settings_dir} "
                    f"to {settings_dir}"
                )
            elif not settings_dir.exists():
                self.logger.error(
                    f"MIKE-SHE settings directory not found: {settings_dir} "
                    f"(original also missing: {original_settings_dir})"
                )
                return False

            # Find .she setup file
            setup_file = None
            she_files = list(settings_dir.glob('*.she'))
            if she_files:
                setup_file = she_files[0]
            else:
                self.logger.error(f"No .she file found in {settings_dir}")
                return False

            # Validate parameter combinations
            validation_error = self._validate_params(params)
            if validation_error:
                self.logger.debug(
                    f"Parameter validation failed: {validation_error}"
                )
                return False

            # Update .she XML file
            return self._update_she_file(setup_file, params)

        except Exception as e:
            self.logger.error(f"Error applying MIKE-SHE parameters: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_she_file(
        self,
        setup_file: Path,
        params: Dict[str, float]
    ) -> bool:
        """
        Update MIKE-SHE .she XML file with new parameter values.

        Args:
            setup_file: Path to .she setup file
            params: Parameters to update

        Returns:
            True if successful
        """
        try:
            tree = ET.parse(setup_file)  # nosec B314
            root = tree.getroot()

            # Detect namespace
            namespace = ''
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0] + '}'

            for param_name, value in params.items():
                xpath = self.PARAM_XML_PATHS.get(param_name)
                if xpath is None:
                    continue

                # Try without namespace first
                element = root.find(xpath)
                if element is None and namespace:
                    ns_xpath = xpath
                    for tag in xpath.replace('.//', '').split('/'):
                        ns_xpath = ns_xpath.replace(
                            tag, f'{namespace}{tag}', 1
                        )
                    element = root.find(ns_xpath)

                if element is not None:
                    element.text = f'{value:.6g}'
                    self.logger.debug(f"Updated {param_name} = {value:.6g}")
                else:
                    self.logger.warning(
                        f"XML element not found for {param_name} at {xpath}"
                    )

            # Save atomically
            temp_file = setup_file.with_suffix('.she.tmp')
            tree.write(temp_file, encoding='utf-8', xml_declaration=True)
            temp_file.replace(setup_file)

            return True

        except Exception as e:
            self.logger.error(f"Error updating MIKE-SHE .she file: {e}")
            return False

    def _validate_params(self, params: Dict[str, float]) -> Optional[str]:
        """
        Validate parameter combinations before applying to MIKE-SHE.

        Returns None if valid, or an error string if invalid.
        """
        # theta_wp < theta_fc < theta_sat
        theta_sat = params.get('theta_sat')
        theta_fc = params.get('theta_fc')
        theta_wp = params.get('theta_wp')

        if theta_sat is not None and theta_fc is not None:
            if theta_fc >= theta_sat:
                return (
                    f"theta_fc ({theta_fc:.4f}) >= "
                    f"theta_sat ({theta_sat:.4f})"
                )

        if theta_fc is not None and theta_wp is not None:
            if theta_wp >= theta_fc:
                return (
                    f"theta_wp ({theta_wp:.4f}) >= "
                    f"theta_fc ({theta_fc:.4f})"
                )

        # Manning's M must be positive
        manning_m = params.get('manning_m')
        if manning_m is not None and manning_m <= 0:
            return f"manning_m ({manning_m:.4f}) must be positive"

        # Hydraulic conductivities must be positive
        for ks_key in ['Ks_uz', 'Ks_sz_h']:
            ks_val = params.get(ks_key)
            if ks_val is not None and ks_val <= 0:
                return f"{ks_key} ({ks_val:.6g}) must be positive"

        return None

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run MIKE-SHE model.

        Args:
            config: Configuration dictionary
            settings_dir: MIKE-SHE settings directory
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            # Use sim_dir for output if provided
            mikeshe_output_dir = Path(kwargs.get('sim_dir', output_dir))
            mikeshe_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up stale output files
            self._cleanup_stale_output(mikeshe_output_dir)

            # Get executable
            mikeshe_exe = self._get_mikeshe_executable(config, data_dir)
            if not mikeshe_exe.exists():
                self.logger.error(
                    f"MIKE-SHE executable not found: {mikeshe_exe}"
                )
                return False

            # Find .she setup file
            setup_file = None
            she_files = list(settings_dir.glob('*.she'))
            if she_files:
                setup_file = she_files[0]
            else:
                self.logger.error(
                    f"No .she setup file found in {settings_dir}"
                )
                return False

            # Build command - optionally prepend wine on non-Windows
            cmd = []
            use_wine = config.get('MIKESHE_USE_WINE', False)
            if use_wine and platform.system() != 'Windows':
                cmd.append('wine')
            cmd.extend([str(mikeshe_exe), str(setup_file)])

            # Set environment
            env = os.environ.copy()

            # Run with timeout
            timeout = config.get('MIKESHE_TIMEOUT', 7200)

            stdout_file = mikeshe_output_dir / 'mikeshe_stdout.log'
            stderr_file = mikeshe_output_dir / 'mikeshe_stderr.log'

            try:
                with (
                    open(stdout_file, 'w', encoding='utf-8') as stdout_f,
                    open(stderr_file, 'w', encoding='utf-8') as stderr_f
                ):
                    result = subprocess.run(
                        cmd,
                        cwd=str(mikeshe_output_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    f"MIKE-SHE timed out after {timeout}s"
                )
                return False

            if result.returncode != 0:
                self._last_error = (
                    f"MIKE-SHE failed with return code {result.returncode}"
                )
                self.logger.error(self._last_error)
                return False

            # Verify output
            output_files = (
                list(mikeshe_output_dir.glob('*.csv'))
                + list(mikeshe_output_dir.glob('*.dfs0'))
            )
            if not output_files:
                self._last_error = "No output files produced"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running MIKE-SHE: {e}")
            return False

    def _get_mikeshe_executable(
        self,
        config: Dict[str, Any],
        data_dir: Path
    ) -> Path:
        """Get MIKE-SHE executable path."""
        install_path = config.get('MIKESHE_INSTALL_PATH', 'default')
        exe_name = config.get('MIKESHE_EXE', 'MikeSheEngine.exe')

        if install_path == 'default':
            return data_dir / "installs" / "mikeshe" / "bin" / exe_name

        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        return install_path

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale MIKE-SHE output files."""
        for pattern in ['*.csv', '*.dfs0', '*.log']:
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
        Calculate metrics from MIKE-SHE output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))

            # Find output file (CSV preferred, then dfs0)
            output_files = list(sim_dir.glob('*discharge*.csv'))
            if not output_files:
                output_files = list(sim_dir.glob('*flow*.csv'))
            if not output_files:
                output_files = list(sim_dir.glob('*.csv'))
            if not output_files:
                output_files = list(sim_dir.glob('*.dfs0'))

            if not output_files:
                self.logger.error(
                    f"No MIKE-SHE output files found in {sim_dir}"
                )
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_file = output_files[0]

            # Read CSV output
            df = pd.read_csv(sim_file, parse_dates=[0])
            datetime_col = df.columns[0]
            df.set_index(datetime_col, inplace=True)

            # Find discharge column
            discharge_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(
                    kw in col_lower
                    for kw in ['discharge', 'flow', 'runoff', 'q_total']
                ):
                    discharge_col = col
                    break

            if discharge_col is None:
                numeric_cols = df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    discharge_col = numeric_cols[0]
                else:
                    return {
                        'kge': self.penalty_score,
                        'error': 'No discharge column'
                    }

            # MIKE-SHE output is typically in m3/s
            sim_series = df[discharge_col]
            sim_series.index = pd.to_datetime(sim_series.index)

            # Skip warmup period
            warmup_days = int(config.get('MIKESHE_WARMUP_DAYS', 365))
            if warmup_days > 0 and len(sim_series) > warmup_days:
                sim_series = sim_series.iloc[warmup_days:]
                self.logger.debug(
                    f"Skipped {warmup_days} warmup days for metric calculation"
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
            self.logger.error(f"Error calculating MIKE-SHE metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_mikeshe_parameters_worker(task_data)


def _evaluate_mikeshe_parameters_worker(
    task_data: Dict[str, Any]
) -> Dict[str, Any]:
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
        worker = MIKESHEWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': (
                f'MIKE-SHE worker exception: {str(e)}\n'
                f'{traceback.format_exc()}'
            ),
            'proc_id': task_data.get('proc_id', -1)
        }
