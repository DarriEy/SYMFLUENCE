"""
HYPE Worker

Worker implementation for HYPE model optimization.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry
from symfluence.evaluation.metrics import kge, nse


@OptimizerRegistry.register_worker('HYPE')
class HYPEWorker(BaseWorker):
    """
    Worker for HYPE model calibration.

    Handles parameter application, HYPE execution, and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize HYPE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to HYPE configuration files.

        Args:
            params: Parameter values to apply
            settings_dir: HYPE settings directory
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        try:
            config = kwargs.get('config', self.config)
            
            # Use HYPEPreProcessor to regenerate configs with new params
            # We only need to regenerate the par.txt file, but calling 
            # preprocess_models with params is the cleanest way.
            preprocessor = HYPEPreProcessor(config, self.logger, params=params)
            
            # Set model-specific paths to point to the worker's settings dir
            preprocessor.output_path = str(settings_dir) + '/'
            preprocessor.hype_setup_dir = str(settings_dir) + '/'
            
            # Only regenerate the model configs (GeoData, par.txt, info.txt)
            # Forcing doesn't change during calibration
            preprocessor._create_model_configs()
            
            return True

        except Exception as e:
            self.logger.error(f"Error applying HYPE parameters: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run HYPE model.

        Args:
            config: Configuration dictionary
            settings_dir: HYPE settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if model ran successfully
        """
        try:
            # Initialize HYPE runner
            runner = HYPERunner(config, self.logger)
            
            # Override paths for the worker
            runner.setup_dir = settings_dir
            runner.output_dir = output_dir
            
            # Run HYPE
            result_path = runner.run_hype()
            
            return result_path is not None

        except Exception as e:
            self.logger.error(f"Error running HYPE: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from HYPE output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            # HYPE output file for computed discharge
            sim_file = output_dir / 'timeCOUT.txt'
            if not sim_file.exists():
                return {'kge': self.penalty_score, 'error': 'timeCOUT.txt not found'}

            # Read simulation (HYPE output is tab-separated with 'time' column)
            sim_df = pd.read_csv(sim_file, sep='\t', index_label='time')
            
            # For lumped, it's the first column after time
            # HYPE usually has a header like 'time' and then subids
            if 'time' in sim_df.columns:
                sim_df = sim_df.set_index('time')
            
            # Get the first subid column (usually only one for lumped)
            sim = sim_df.iloc[:, 0].values

            # Load observations
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            obs_file = (data_dir / f'domain_{domain_name}' / 'observations' /
                       'streamflow' / 'preprocessed' / f'{domain_name}_streamflow_processed.csv')

            if not obs_file.exists():
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            
            # Align simulation and observations
            # HYPE dates are daily, observations are usually daily but might need resampling
            sim_dates = pd.to_datetime(sim_df.index)
            sim_series = pd.Series(sim, index=sim_dates)
            
            # Align by index
            common_idx = sim_series.index.intersection(obs_df.index)
            if len(common_idx) == 0:
                return {'kge': self.penalty_score, 'error': 'No common dates between sim and obs'}
                
            obs_aligned = df_obs.loc[common_index].values
            sim_aligned = df_sim.loc[common_index].values

            kge_val = kge(obs_aligned, sim_aligned, transfo=1)
            nse_val = nse(obs_aligned, sim_aligned, transfo=1)

            return {'kge': float(kge_val), 'nse': float(nse_val)}

        except Exception as e:
            self.logger.error(f"Error calculating HYPE metrics: {e}")
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
        worker = HYPEWorker()
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
