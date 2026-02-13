"""
Xinanjiang Model Optimizer.

Xinanjiang-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import XinanjiangWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('XINANJIANG')
class XinanjiangModelOptimizer(BaseModelOptimizer):
    """Xinanjiang-specific optimizer using the unified BaseModelOptimizer framework."""

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
        self.xinanjiang_setup_dir = self.project_dir / 'settings' / 'XINANJIANG'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)
        self.logger.debug("XinanjiangModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'XINANJIANG'

    def _get_final_file_manager_path(self) -> Path:
        return self.xinanjiang_setup_dir / 'xinanjiang_config.txt'

    def _create_parameter_manager(self):
        from .parameter_manager import XinanjiangParameterManager
        return XinanjiangParameterManager(
            self.config, self.logger, self.xinanjiang_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        return False  # Xinanjiang is lumped only

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        best_result = self.get_best_result()
        best_params = best_result.get('params')
        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.xinanjiang_setup_dir)
        return self.worker.run_model(
            self.config, self.xinanjiang_setup_dir, output_dir, save_output=True
        )
