"""
TOPMODEL Optimizer.

TOPMODEL-specific optimizer inheriting from BaseModelOptimizer.
Supports DDS and other iterative optimization algorithms.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import TopmodelWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('TOPMODEL')
class TopmodelModelOptimizer(BaseModelOptimizer):
    """
    TOPMODEL-specific optimizer using the unified BaseModelOptimizer framework.

    Supports:
    - Standard iterative optimization (DDS, PSO, SCE-UA, DE)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.experiment_id = config.get('EXPERIMENT_ID')
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.topmodel_setup_dir = self.project_dir / 'settings' / 'TOPMODEL'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("TopmodelModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'TOPMODEL'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to TOPMODEL configuration (placeholder for in-memory model)."""
        return self.topmodel_setup_dir / 'topmodel_config.txt'

    def _create_parameter_manager(self):
        """Create TOPMODEL parameter manager."""
        from symfluence.models.topmodel.calibration.parameter_manager import TopmodelParameterManager
        return TopmodelParameterManager(
            self.config,
            self.logger,
            self.topmodel_setup_dir
        )

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run TOPMODEL for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.topmodel_setup_dir)

        return self.worker.run_model(
            self.config,
            self.topmodel_setup_dir,
            output_dir,
            save_output=True
        )

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation with consistent warmup handling for TOPMODEL."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            if not self.worker._initialized:
                if not self.worker.initialize():
                    self.logger.error("Failed to initialize TOPMODEL worker for final evaluation")
                    return None

            if not self.worker.apply_parameters(best_params, self.topmodel_setup_dir):
                self.logger.error("Failed to apply best parameters for final evaluation")
                return None

            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            runoff = self.worker._run_simulation(
                self.worker._forcing,
                best_params
            )

            self.worker.save_output_files(
                runoff[self.worker.warmup_days:],
                final_output_dir,
                self.worker._time_index[self.worker.warmup_days:] if self.worker._time_index is not None else None
            )

            # Calculate metrics using worker's in-memory data
            from symfluence.optimization.metrics.metric_calculator import MetricCalculator
            metric_name = self.config.get('OPTIMIZATION_METRIC', 'KGE')
            calculator = MetricCalculator()

            obs = self.worker._observations
            sim = runoff

            if obs is not None and len(obs) == len(sim):
                warmup = self.worker.warmup_days
                obs_eval = obs[warmup:]
                sim_eval = sim[warmup:]

                valid = ~np.isnan(obs_eval) & ~np.isnan(sim_eval)
                if np.sum(valid) > 30:
                    score = calculator.calculate(metric_name, obs_eval[valid], sim_eval[valid])
                    self.logger.info(f"Final evaluation {metric_name}: {score:.4f}")
                    return {
                        'metric': metric_name,
                        'score': score,
                        'params': best_params,
                        'output_dir': str(final_output_dir),
                    }

            self.logger.warning("Could not calculate final metrics")
            return {'params': best_params, 'output_dir': str(final_output_dir)}

        except Exception as e:
            self.logger.error(f"Final evaluation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
