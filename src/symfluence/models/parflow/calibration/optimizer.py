"""
ParFlow Model Optimizer

ParFlow-specific optimizer inheriting from BaseModelOptimizer.
Calibrates van Genuchten parameters, saturated hydraulic conductivity,
Manning's roughness, and domain geometry via DDS or other algorithms.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import ParFlowWorker  # noqa: F401 - triggers worker registration


@OptimizerRegistry.register_optimizer('PARFLOW')
class ParFlowModelOptimizer(BaseModelOptimizer):
    """
    ParFlow-specific optimizer using the unified BaseModelOptimizer framework.

    Calibrates integrated variably-saturated subsurface + overland flow parameters
    by iteratively updating the .pfidb database file and re-running ParFlow.
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

        self.parflow_setup_dir = self.project_dir / 'settings' / 'PARFLOW'

        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )

        self.logger.debug("ParFlowModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'PARFLOW'

    def _get_final_file_manager_path(self) -> Path:
        """ParFlow uses .pfidb files, not a file manager."""
        pfidb_files = list(self.parflow_setup_dir.glob('*.pfidb'))
        if pfidb_files:
            return pfidb_files[0]
        return self.parflow_setup_dir / 'parflow_run.pfidb'

    def _create_parameter_manager(self):
        """Create ParFlow parameter manager."""
        from .parameter_manager import ParFlowParameterManager
        return ParFlowParameterManager(
            self.config,
            self.logger,
            self.parflow_setup_dir,
        )

    def _check_routing_needed(self) -> bool:
        """ParFlow handles its own overland flow routing; no external routing needed."""
        return False

    def _setup_parallel_dirs(self) -> None:
        """Setup ParFlow-specific parallel directories."""
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            default='optimization',
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()

        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir, 'PARFLOW', self.experiment_id
        )

        # Copy ParFlow settings (.pfidb, runname.txt) to each parallel directory
        if self.parflow_setup_dir.exists():
            self.copy_base_settings(
                self.parflow_setup_dir, self.parallel_dirs, 'PARFLOW'
            )

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run ParFlow for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        return self.worker.run_model(
            self.config,
            self.parflow_setup_dir,
            output_dir,
            params=best_params,
        )

    def _update_file_manager_for_final_run(self) -> None:
        """ParFlow doesn't use a file manager."""
        pass

    def _restore_file_manager_for_optimization(self) -> None:
        """ParFlow doesn't use a file manager."""
        pass


# Backward compatibility alias
ParFlowOptimizer = ParFlowModelOptimizer
