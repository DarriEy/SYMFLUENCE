"""
GNN Optimization Worker

Handles individual model evaluations for GNN calibration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask, WorkerResult
from symfluence.models.gnn import GNNRunner


class GNNWorker(BaseWorker):
    """
    Worker class for evaluating GNN model during optimization.
    """

    def evaluate(self, task: WorkerTask) -> WorkerResult:
        try:
            eval_config = task.config.copy()
            for key, val in task.params.items():
                eval_config[key] = val

            eval_config['EXPERIMENT_ID'] = f"{task.config.get('EXPERIMENT_ID')}_eval_{task.proc_id}"

            runner = GNNRunner(eval_config, self.logger)
            runner.run_gnn()

            from symfluence.optimization.calibration_targets import StreamflowTarget
            target = StreamflowTarget(
                eval_config,
                Path(eval_config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{eval_config.get('DOMAIN_NAME')}",
                self.logger
            )

            sim_dir = runner.output_dir
            metrics = target.calculate_metrics(sim_dir)
            metric_name = eval_config.get('OPTIMIZATION_METRIC', 'KGE')
            score = metrics.get(metric_name, self.penalty_score)

            return WorkerResult(score=score, metrics=metrics, params=task.params)

        except Exception as e:
            self.logger.error(f"Error in GNN evaluation: {e}")
            import traceback
            return WorkerResult(
                score=self.penalty_score,
                error=traceback.format_exc(),
                params=task.params
            )

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, mode: str = 'run_def') -> bool:
        runner = GNNRunner(config, self.logger)
        runner.run_gnn()
        return True

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, config: Optional[Dict[str, Any]] = None) -> bool:
        return True
