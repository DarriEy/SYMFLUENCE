"""
MESH Worker

Worker implementation for MESH model optimization.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry
from .utilities.streamflow_metrics import StreamflowMetrics


@OptimizerRegistry.register_worker('MESH')
class MESHWorker(BaseWorker):
    """
    Worker for MESH model calibration.

    Handles parameter application, MESH execution, and metric calculation.
    """

    # Shared streamflow metrics utility
    _streamflow_metrics = StreamflowMetrics()

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize MESH worker."""
        super().__init__(config, logger)

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to MESH configuration files.

        Args:
            params: Parameter values to apply
            settings_dir: MESH settings directory (unused - params in forcing)
            **kwargs: Additional arguments (may include proc_forcing_dir)

        Returns:
            True if successful
        """
        try:
            config = kwargs.get('config', self.config)

            # MESH parameters are in forcing directory, not settings
            # Check if process-specific forcing directory is provided
            proc_forcing_dir = kwargs.get('proc_forcing_dir')

            # Use MESHParameterManager to update parameter files
            from ..parameter_managers import MESHParameterManager

            # If we have process-specific forcing, we need to adjust the manager's paths
            if proc_forcing_dir:
                param_manager = MESHParameterManager(config, self.logger, settings_dir)
                # Override forcing directory to process-specific location
                param_manager.mesh_forcing_dir = Path(proc_forcing_dir)
                param_manager.class_params_file = param_manager.mesh_forcing_dir / 'MESH_parameters_CLASS.ini'
                param_manager.hydro_params_file = param_manager.mesh_forcing_dir / 'MESH_parameters_hydrology.ini'
                param_manager.routing_params_file = param_manager.mesh_forcing_dir / 'MESH_parameters.txt'
            else:
                param_manager = MESHParameterManager(config, self.logger, settings_dir)

            success = param_manager.update_model_files(params)

            return success

        except Exception as e:
            self.logger.error(f"Error applying MESH parameters: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run MESH model.

        Args:
            config: Configuration dictionary
            settings_dir: MESH settings directory (may be process-specific)
            output_dir: Output directory (process-specific during parallel)
            **kwargs: Additional arguments (may include proc_forcing_dir)

        Returns:
            True if model ran successfully
        """
        try:
            # Initialize MESH runner
            from symfluence.models.mesh.runner import MESHRunner
            runner = MESHRunner(config, self.logger)

            # Check if process-specific directories are provided
            proc_forcing_dir = kwargs.get('proc_forcing_dir')

            if proc_forcing_dir:
                # Parallel execution: use process-specific forcing directory
                forcing_dir = Path(proc_forcing_dir)
                runner.set_process_directories(forcing_dir, output_dir)
                self.logger.debug(f"Using process-specific forcing: {forcing_dir}")
            else:
                # Single process: use default project forcing directory
                domain_name = config.get('DOMAIN_NAME')
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
                project_dir = data_dir / f"domain_{domain_name}"
                runner.forcing_mesh_path = project_dir / 'forcing' / 'MESH_input'
                runner.output_dir = output_dir

            # Run MESH
            result_path = runner.run_mesh()

            return result_path is not None

        except Exception as e:
            self.logger.error(f"Error running MESH: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from MESH output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            # MESH output file for streamflow
            # Check common MESH output file names
            sim_file_candidates = [
                output_dir / 'MESH_output_streamflow.csv',
                output_dir / 'streamflow.csv',
            ]

            sim_file = None
            for candidate in sim_file_candidates:
                if candidate.exists():
                    sim_file = candidate
                    break

            if sim_file is None:
                self.logger.error(f"MESH output not found in {output_dir}")
                return {'kge': self.penalty_score, 'error': 'MESH output not found'}

            # Read simulation
            sim_df = pd.read_csv(sim_file, parse_dates=['time'])
            sim_df = sim_df.set_index('time')

            # Get streamflow column (may vary by MESH version)
            flow_col = None
            for col in ['streamflow', 'discharge', 'flow', 'QOSIM']:
                if col in sim_df.columns:
                    flow_col = col
                    break

            if flow_col is None:
                self.logger.error(f"Streamflow column not found in {sim_file}")
                return {'kge': self.penalty_score, 'error': 'Streamflow column not found'}

            sim = sim_df[flow_col].values

            # Load observations using shared utility
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'

            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq=None
            )
            if obs_values is None:
                self.logger.error("Observations not found")
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_series = pd.Series(obs_values, index=obs_index)
            sim_series = sim_df[flow_col]

            # Align and calculate using shared utility
            try:
                obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(sim_series, obs_series)
                return self._streamflow_metrics.calculate_metrics(obs_aligned, sim_aligned, metrics=['kge', 'nse'])
            except ValueError as e:
                self.logger.error(f"No common dates: {e}")
                return {'kge': self.penalty_score, 'error': 'No common dates'}

        except Exception as e:
            self.logger.error(f"Error calculating MESH metrics: {e}")
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_mesh_parameters_worker(task_data)


def _evaluate_mesh_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary with params, config, and process-specific paths

    Returns:
        Result dictionary
    """
    worker = MESHWorker()

    # Extract process-specific forcing directory if present
    proc_forcing_dir = task_data.get('proc_forcing_dir')

    # Create task with additional kwargs
    task = WorkerTask.from_legacy_dict(task_data)

    # Add proc_forcing_dir to kwargs if present
    if proc_forcing_dir:
        if not hasattr(task, 'kwargs') or task.kwargs is None:
            task.kwargs = {}
        task.kwargs['proc_forcing_dir'] = proc_forcing_dir

    result = worker.evaluate(task)
    return result.to_legacy_dict()
