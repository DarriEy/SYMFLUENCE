"""
LSTM Optimization Worker

Handles individual model evaluations for LSTM calibration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from symfluence.models.lstm import LSTMRunner


class LSTMWorker(BaseWorker):
    """
    Worker class for evaluating LSTM model during optimization.
    """

    def evaluate(self, task: WorkerTask) -> WorkerResult:
        """
        Evaluate a single parameter set.
        
        Args:
            task: Task containing parameters and paths
            
        Returns:
            WorkerResult with score and metrics
        """
        try:
            # 1. Update config with parameters
            # LSTM parameters might be learning rate, hidden size, etc.
            # or multipliers for physical inputs
            eval_config = task.config.copy()
            for key, val in task.params.items():
                eval_config[key] = val
            
            # Override directories for isolation
            eval_config['EXPERIMENT_ID'] = f"{task.config.get('EXPERIMENT_ID')}_eval_{task.proc_id}"
            
            # 2. Run model
            runner = LSTMRunner(eval_config, self.logger)
            runner.run_lstm()
            
            # 3. Calculate score
            # Optimization results are in output_dir (or sim_dir)
            from symfluence.optimization.calibration_targets import StreamflowTarget
            target = StreamflowTarget(eval_config, Path(eval_config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{eval_config.get('DOMAIN_NAME')}", self.logger)
            
            # Determine output directory (if routing was used, score comes from routed output)
            sim_dir = runner.output_dir
            if runner.requires_routing():
                # Check for mizuRoute or dRoute subdirs
                routing_model = eval_config.get('ROUTING_MODEL', 'none').lower()
                if 'droute' in routing_model:
                    sim_dir = sim_dir.parent / 'dRoute'
                else:
                    sim_dir = sim_dir.parent / 'mizuRoute'

            metrics = target.calculate_metrics(sim_dir)
            
            # Return primary metric as score
            metric_name = eval_config.get('OPTIMIZATION_METRIC', 'KGE')
            score = metrics.get(metric_name, -999.0)
            
            return WorkerResult(
                score=score,
                metrics=metrics,
                params=task.params
            )

        except Exception as e:
            self.logger.error(f"Error in LSTM evaluation: {e}")
            import traceback
            return WorkerResult(
                score=-999.0,
                error=traceback.format_exc(),
                params=task.params
            )

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, mode: str = 'run_def') -> bool:
        """Direct model execution hook."""
        runner = LSTMRunner(config, self.logger)
        runner.run_lstm()
        return True

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, config: Optional[Dict[str, Any]] = None) -> bool:
        """Apply parameters to model configuration."""
        # For LSTM, we typically just pass parameters through config
        return True
