"""
GNN Model Optimizer

GNN-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from .worker import GNNWorker
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('GNN')
class GNNModelOptimizer(BaseModelOptimizer):
    """
    GNN-specific optimizer using the unified BaseModelOptimizer framework.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.config = config
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)
        self.logger.debug("GNNModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'GNN'

    def _create_parameter_manager(self):
        from symfluence.optimization.parameter_managers.ml_parameter_manager import MLParameterManager
        gnn_settings_dir = self.project_dir / 'settings' / 'GNN'
        gnn_settings_dir.mkdir(parents=True, exist_ok=True)

        return MLParameterManager(
            self.config,
            self.logger,
            gnn_settings_dir,
            params_key='GNN_PARAMS_TO_CALIBRATE',
            bounds_key='GNN_PARAMETER_BOUNDS'
        )

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        from symfluence.models.gnn import GNNRunner
        runner = GNNRunner(self.config, self.logger)
        runner.run_gnn()
        return True

    def _get_final_file_manager_path(self) -> Path:
        return self.project_dir / 'settings' / 'GNN' / 'dummy_fm.txt'

    def _setup_parallel_dirs(self) -> None:
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'optimization').lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'GNN',
            self.experiment_id
        )
