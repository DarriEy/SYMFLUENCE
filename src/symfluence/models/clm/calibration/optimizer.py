"""
CLM Model Optimizer

CLM-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import CLMWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('CLM')
class CLMModelOptimizer(BaseModelOptimizer):
    """
    CLM-specific optimizer using the unified BaseModelOptimizer framework.

    Supports all standard optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None,
    ):
        self.experiment_id = config.get('EXPERIMENT_ID')
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.clm_setup_dir = self.project_dir / 'CLM_input' / 'settings'

        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )

        self.logger.debug("CLMModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'CLM'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to CLM user_nl_clm namelist."""
        return self.clm_setup_dir / 'user_nl_clm'

    def _create_parameter_manager(self):
        """Create CLM parameter manager."""
        from .parameter_manager import CLMParameterManager
        return CLMParameterManager(
            self.config,
            self.logger,
            self.clm_setup_dir,
        )

    def _check_routing_needed(self) -> bool:
        """CLM does not use external routing."""
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run CLM for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.clm_setup_dir)

        return self.worker.run_model(
            self.config,
            self.clm_setup_dir,
            output_dir,
        )
