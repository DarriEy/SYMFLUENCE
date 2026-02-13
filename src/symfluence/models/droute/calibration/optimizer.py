"""
dRoute Model Optimizer.

dRoute-specific optimizer inheriting from BaseModelOptimizer.
Delegates execution to DRouteWorker and exposes gradient support
via the worker's AD capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import DRouteWorker  # noqa: F401 - trigger worker registration


@OptimizerRegistry.register_optimizer('DROUTE')
class DRouteModelOptimizer(BaseModelOptimizer):
    """
    dRoute-specific optimizer using the unified BaseModelOptimizer framework.

    Supports both evolutionary (DDS, PSO, SCE, DE) and gradient-based (ADAM, LBFGS)
    algorithms. Gradient-based methods require dRoute compiled with CoDiPack/Enzyme.

    Example:
        optimizer = DRouteModelOptimizer(config, logger)
        results_path = optimizer.run_dds()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.droute_settings_dir = self.project_dir / 'settings' / 'dRoute'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager)

    def _get_model_name(self) -> str:
        return 'DROUTE'

    def _create_parameter_manager(self):
        """Create dRoute parameter manager."""
        from .parameter_manager import DRouteParameterManager
        return DRouteParameterManager(
            self.config_dict, self.logger, self.droute_settings_dir
        )

    def _check_routing_needed(self) -> bool:
        """dRoute IS the router — no additional routing needed."""
        return False

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """Apply best parameters to worker for final evaluation."""
        if hasattr(self, '_worker') and self._worker is not None:
            self._worker._current_params = best_params
        return True

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run dRoute with best parameters for final evaluation."""
        if hasattr(self, '_worker') and self._worker is not None:
            return self._worker.run_model(
                self.config_dict,
                self.droute_settings_dir,
                self.project_dir / 'simulations' / self.config_dict.get('EXPERIMENT_ID') / 'dRoute',
                save_output=True
            )
        return False

    def _get_final_file_manager_path(self) -> Path:
        """Return path for final evaluation file manager."""
        return self.droute_settings_dir / 'droute_config.yaml'

    def _setup_parallel_dirs(self) -> None:
        """Set up parallel directories for dRoute calibration."""
        n_processors = int(self.config_dict.get('NUMBER_OF_PROCESSORS', 1))
        for i in range(n_processors):
            proc_dir = self.project_dir / 'simulations' / 'calibration' / f'proc_{i}' / 'dRoute'
            proc_dir.mkdir(parents=True, exist_ok=True)
