"""
SUMMA Worker

Worker implementation for SUMMA model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""

import os
import gc
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry


logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('SUMMA')
class SUMMAWorker(BaseWorker):
    """
    Worker for SUMMA model calibration.

    Handles parameter application, SUMMA execution, optional mizuRoute routing,
    and metric calculation for streamflow calibration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SUMMA worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)
        self._routing_needed = None

    def needs_routing(self, config: Dict[str, Any]) -> bool:
        """
        Determine if routing (mizuRoute) is needed.

        Args:
            config: Configuration dictionary

        Returns:
            True if routing is needed
        """
        calibration_var = config.get('CALIBRATION_VARIABLE', 'streamflow')

        if calibration_var != 'streamflow':
            return False

        domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')

        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True

        return False

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to SUMMA configuration files.

        Args:
            params: Parameter values to apply
            settings_dir: SUMMA settings directory
            **kwargs: Additional arguments (task_data for legacy compatibility)

        Returns:
            True if successful
        """
        try:
            # Import existing function
            from .summa_parallel_workers import _apply_parameters_worker

            # Build task_data for legacy function
            task_data = kwargs.get('task_data', {})
            if not task_data:
                task_data = {
                    'config': kwargs.get('config', self.config),
                    'params': params,
                }

            # Create minimal logger for internal use
            internal_logger = logging.getLogger(f'summa_worker_apply')
            internal_logger.setLevel(logging.WARNING)

            debug_info = {'stage': 'apply_parameters', 'files_checked': [], 'errors': []}

            success = _apply_parameters_worker(
                params, task_data, settings_dir, internal_logger, debug_info
            )

            return success

        except ImportError:
            # Fallback: Apply parameters directly
            return self._apply_parameters_direct(params, settings_dir, kwargs.get('config', {}))

        except Exception as e:
            self.logger.error(f"Error applying SUMMA parameters: {e}")
            return False

    def _apply_parameters_direct(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        config: Dict[str, Any]
    ) -> bool:
        """
        Apply parameters directly to SUMMA trial parameter file.

        Args:
            params: Parameter values
            settings_dir: Settings directory
            config: Configuration dictionary

        Returns:
            True if successful
        """
        try:
            import netCDF4 as nc

            # Find trial parameter file
            trial_param_file = settings_dir / 'trialParams.nc'
            if not trial_param_file.exists():
                # Try alternate name
                experiment_id = config.get('EXPERIMENT_ID', 'default')
                trial_param_file = settings_dir / f'{experiment_id}_trialParams.nc'

            if not trial_param_file.exists():
                self.logger.error(f"Trial parameter file not found in {settings_dir}")
                return False

            # Update parameters
            with nc.Dataset(trial_param_file, 'r+') as ds:
                for param_name, value in params.items():
                    if param_name in ds.variables:
                        ds.variables[param_name][:] = value
                    else:
                        self.logger.warning(f"Parameter {param_name} not found in file")

            return True

        except Exception as e:
            self.logger.error(f"Error in direct parameter application: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run SUMMA model (and mizuRoute if needed).

        Args:
            config: Configuration dictionary
            settings_dir: SUMMA settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if model ran successfully
        """
        try:
            # Import existing functions
            from .summa_parallel_workers import _run_summa_worker, _run_mizuroute_worker

            # Resolve paths
            summa_exe = Path(config.get('SUMMA_INSTALL_PATH', 'default'))
            if str(summa_exe) == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                summa_exe = data_dir / 'installs' / 'summa' / 'bin' / 'summa.exe'

            file_manager = settings_dir / 'fileManager.txt'
            if not file_manager.exists():
                # Try alternate names
                experiment_id = config.get('EXPERIMENT_ID', 'default')
                file_manager = settings_dir / f'{experiment_id}_fileManager.txt'

            summa_dir = output_dir
            sim_dir = kwargs.get('sim_dir', output_dir)

            # Create minimal logger
            internal_logger = logging.getLogger('summa_worker_run')
            internal_logger.setLevel(logging.WARNING)

            debug_info = {'stage': 'model_run', 'commands_run': [], 'errors': []}

            # Run SUMMA
            success = _run_summa_worker(
                summa_exe, file_manager, summa_dir, internal_logger, debug_info
            )

            if not success:
                return False

            # Run routing if needed
            if self.needs_routing(config):
                mizuroute_dir = sim_dir / 'mizuRoute' if sim_dir else output_dir / 'mizuRoute'
                mizuroute_dir.mkdir(parents=True, exist_ok=True)

                task_data = {
                    'config': config,
                    'summa_dir': str(summa_dir),
                    'mizuroute_dir': str(mizuroute_dir),
                }

                routing_success = _run_mizuroute_worker(
                    config, task_data, internal_logger, debug_info
                )

                if not routing_success:
                    self.logger.warning("Routing failed, but SUMMA succeeded")
                    # Continue - routing failure may be acceptable for some workflows

            return True

        except ImportError:
            # Fallback: Run SUMMA directly
            return self._run_summa_direct(config, settings_dir, output_dir)

        except Exception as e:
            self.logger.error(f"Error running SUMMA: {e}")
            return False

    def _run_summa_direct(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path
    ) -> bool:
        """
        Run SUMMA directly using subprocess.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory

        Returns:
            True if successful
        """
        try:
            import subprocess

            summa_exe = Path(config.get('SUMMA_INSTALL_PATH', 'default'))
            if str(summa_exe) == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                summa_exe = data_dir / 'installs' / 'summa' / 'bin' / 'summa.exe'

            file_manager = settings_dir / 'fileManager.txt'

            if not summa_exe.exists() or not file_manager.exists():
                return False

            # Run SUMMA
            cmd = [str(summa_exe), '-m', str(file_manager)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.get('SUMMA_TIMEOUT', 600)
            )

            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"Error in direct SUMMA execution: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from SUMMA output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            # Import existing function
            from .summa_parallel_workers import _calculate_metrics_inline_worker

            # Resolve mizuroute_dir if needed
            mizuroute_dir = kwargs.get('mizuroute_dir')
            if not mizuroute_dir and self.needs_routing(config):
                sim_dir = kwargs.get('sim_dir', output_dir)
                mizuroute_dir = sim_dir / 'mizuRoute'

            metrics = _calculate_metrics_inline_worker(
                output_dir, mizuroute_dir, config, internal_logger
            )

            # Ensure return is a dict
            if isinstance(metrics, (int, float)):
                metric_name = config.get('CALIBRATION_METRIC', 'KGE').lower()
                return {metric_name: float(metrics)}

            return metrics or {'kge': self.penalty_score}

        except ImportError:
            # Fallback: Calculate metrics directly
            return self._calculate_metrics_direct(output_dir, config)

        except Exception as e:
            self.logger.error(f"Error calculating SUMMA metrics: {e}")
            return {'kge': self.penalty_score}

    def _calculate_metrics_direct(
        self,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate metrics directly from output files.

        Args:
            output_dir: Output directory
            config: Configuration dictionary

        Returns:
            Dictionary of metrics
        """
        try:
            import xarray as xr
            import pandas as pd
            from symfluence.utils.common.metrics import get_KGE, get_NSE

            # Find output file
            output_files = list(output_dir.glob('*_day.nc')) + list(output_dir.glob('*_output.nc'))
            if not output_files:
                return {'kge': self.penalty_score, 'error': 'No output files found'}

            # Read simulation
            with xr.open_dataset(output_files[0]) as ds:
                if 'scalarTotalRunoff' in ds:
                    sim = ds['scalarTotalRunoff'].values.flatten()
                elif 'averageRoutedRunoff' in ds:
                    sim = ds['averageRoutedRunoff'].values.flatten()
                else:
                    return {'kge': self.penalty_score, 'error': 'No runoff variable found'}

            # Load observations
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            obs_file = (data_dir / f'domain_{domain_name}' / 'observations' /
                       'streamflow' / 'preprocessed' / f'{domain_name}_streamflow_processed.csv')

            if not obs_file.exists():
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            obs = obs_df['discharge_cms'].values

            # Align lengths
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs = obs[:min_len]

            # Calculate metrics
            kge = get_KGE(obs, sim, transfo=1)
            nse = get_NSE(obs, sim, transfo=1)

            return {
                'kge': float(kge),
                'nse': float(nse),
            }

        except Exception as e:
            self.logger.error(f"Error in direct metrics calculation: {e}")
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Maintains backward compatibility with existing parallel processing.

        Args:
            task_data: Task dictionary in legacy format

        Returns:
            Result dictionary in legacy format
        """
        try:
            # Use existing safe wrapper for full functionality
            from .summa_parallel_workers import _evaluate_parameters_worker_safe
            return _evaluate_parameters_worker_safe(task_data)

        except ImportError:
            # Fallback: Use new worker infrastructure
            worker = SUMMAWorker()
            task = WorkerTask.from_legacy_dict(task_data)
            result = worker.evaluate(task)
            return result.to_legacy_dict()
