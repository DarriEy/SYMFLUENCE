"""
GSFLOW Model Optimizer.

GSFLOW-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import GSFLOWWorker  # noqa: F401


@OptimizerRegistry.register_optimizer('GSFLOW')
class GSFLOWModelOptimizer(BaseModelOptimizer):
    """GSFLOW-specific optimizer using the unified BaseModelOptimizer framework."""

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
        self.gsflow_setup_dir = self.project_dir / 'settings' / 'GSFLOW'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)
        self.logger.debug("GSFLOWModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'GSFLOW'

    def _get_final_file_manager_path(self) -> Path:
        control_file = self._get_config_value(
            lambda: self.config.model.gsflow.control_file,
            default='gsflow.control',
            dict_key='GSFLOW_CONTROL_FILE'
        )
        return self.gsflow_setup_dir / control_file

    def _create_parameter_manager(self):
        from .parameter_manager import GSFLOWParameterManager
        return GSFLOWParameterManager(
            self.config, self.logger, self.gsflow_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """GSFLOW has internal SFR routing."""
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        best_result = self.get_best_result()
        best_params = best_result.get('params')
        if not best_params:
            self.logger.warning("No best parameters found")
            return False

        self.worker.apply_parameters(best_params, self.gsflow_setup_dir)
        success = self.worker.run_model(
            self.config, self.gsflow_setup_dir, output_dir
        )

        if success:
            output_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ['statvar*', '*.csv']:
                for f in self.gsflow_setup_dir.glob(pattern):
                    if f.is_file():
                        shutil.copy2(f, output_dir / f.name)

        return success

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        self.logger.info("=" * 60)
        self.logger.info("RUNNING GSFLOW FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            if not self._run_model_for_final_evaluation(final_output_dir):
                return None

            metrics = self.worker.calculate_metrics(
                final_output_dir, self.config, sim_dir=final_output_dir
            )

            if not metrics or metrics.get('kge', -999) <= -999:
                return None

            return {
                'final_metrics': metrics,
                'calibration_metrics': {"KGE_Calib": metrics.get('kge', -999)},
                'evaluation_metrics': {"KGE_Eval": metrics.get('kge', -999)},
                'success': True,
                'best_params': best_params
            }
        except Exception as e:
            self.logger.error(f"Error in final evaluation: {e}")
            return None
