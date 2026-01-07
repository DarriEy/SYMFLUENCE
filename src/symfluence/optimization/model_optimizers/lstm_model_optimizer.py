"""
LSTM Model Optimizer

LSTM-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with LSTM.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..optimizers.base_model_optimizer import BaseModelOptimizer
from ..workers.lstm_worker import LSTMWorker
from ..registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('LSTM')
class LSTMModelOptimizer(BaseModelOptimizer):
    """
    LSTM-specific optimizer using the unified BaseModelOptimizer framework.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.config = config
        # Distributed mode detected from DOMAIN_DEFINITION_METHOD
        self._routing_needed = config.get('DOMAIN_DEFINITION_METHOD', 'lumped') == 'delineate'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.info(f"LSTMModelOptimizer initialized (routing needed: {self._routing_needed})")

    def _get_model_name(self) -> str:
        return 'LSTM'

    def _create_parameter_manager(self):
        """
        LSTM uses standard ParameterManager.
        Parameters could be LSTM hyperparameters or multipliers.
        """
        from ..core.parameter_manager import ParameterManager
        lstm_settings_dir = self.project_dir / 'settings' / 'LSTM'
        if not lstm_settings_dir.exists():
            lstm_settings_dir.mkdir(parents=True, exist_ok=True)
            
        return ParameterManager(
            self.config,
            self.logger,
            lstm_settings_dir
        )

    def _create_calibration_target(self):
        from ..calibration_targets import StreamflowTarget
        return StreamflowTarget(self.config, self.project_dir, self.logger)

    def _create_worker(self) -> LSTMWorker:
        return LSTMWorker(self.config, self.logger)

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run LSTM for final evaluation."""
        from symfluence.models.lstm import LSTMRunner
        runner = LSTMRunner(self.config, self.logger)
        runner.run_lstm()
        return True

    def _get_final_file_manager_path(self) -> Path:
        # LSTM doesn't use a file manager text file like SUMMA
        return self.project_dir / 'settings' / 'LSTM' / 'dummy_fm.txt'

    def _setup_parallel_dirs(self) -> None:
        """Setup parallel directories for LSTM."""
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'optimization').lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'LSTM',
            self.experiment_id
        )
