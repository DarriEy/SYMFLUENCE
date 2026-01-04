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
from symfluence.evaluation.metrics import kge, nse


@OptimizerRegistry.register_worker('MESH')
class MESHWorker(BaseWorker):
    """
    Worker for MESH model calibration.

    Handles parameter application, MESH execution, and metric calculation.
    """

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
            settings_dir: MESH settings directory
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        try:
            config = kwargs.get('config', self.config)

            # Use MESHParameterManager to update parameter files
            from ..parameter_managers import MESHParameterManager

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
            settings_dir: MESH settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if model ran successfully
        """
        try:
            # Initialize MESH runner
            runner = MESHRunner(config, self.logger)

            # Override paths for the worker
            # MESH reads from forcing directory, not settings
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
            project_dir = data_dir / f"domain_{domain_name}"

            runner.mesh_forcing_dir = project_dir / 'forcing' / 'MESH_input'
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

            # Load observations
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            obs_file = (data_dir / f'domain_{domain_name}' / 'observations' /
                       'streamflow' / 'preprocessed' / f'{domain_name}_streamflow_processed.csv')

            if not obs_file.exists():
                self.logger.error(f"Observations not found: {obs_file}")
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # Align simulation and observations
            common_idx = sim_df.index.intersection(obs_df.index)
            if len(common_idx) == 0:
                self.logger.error("No common dates between simulation and observations")
                return {'kge': self.penalty_score, 'error': 'No common dates'}

            obs_aligned = df_obs.loc[common_index].values
            sim_aligned = df_sim.loc[common_index].values

            kge_val = kge(obs_aligned, sim_aligned, transfo=1)
            nse_val = nse(obs_aligned, sim_aligned, transfo=1)

            return {'kge': float(kge_val), 'nse': float(nse_val)}

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
        worker = MESHWorker()
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
