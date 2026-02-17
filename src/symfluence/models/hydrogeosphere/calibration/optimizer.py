"""
HydroGeoSphere Model Optimizer

Optimizer for HGS calibration. Sets up parallel directories,
manages parameter perturbation, and delegates execution to HGSWorker.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.file_utils import copy_file
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import HGSWorker  # noqa: F401 - triggers worker registration


@OptimizerRegistry.register_optimizer('HYDROGEOSPHERE')
class HGSModelOptimizer(BaseModelOptimizer):
    """Optimizer for HGS calibration."""

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None,
    ):
        self.config = config
        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )

    def _get_model_name(self) -> str:
        return 'HYDROGEOSPHERE'

    def _get_final_file_manager_path(self) -> Path:
        return self.project_dir / 'settings' / 'HYDROGEOSPHERE'

    def _create_parameter_manager(self):
        from .parameter_manager import HGSParameterManager
        settings_dir = self.project_dir / 'settings' / 'HYDROGEOSPHERE'
        return HGSParameterManager(
            self.config, self.logger, settings_dir,
        )

    def _check_routing_needed(self) -> bool:
        return False

    def _setup_parallel_dirs(self) -> None:
        source_hgs = self.project_dir / 'settings' / 'HYDROGEOSPHERE'
        if not source_hgs.exists() or not any(source_hgs.iterdir()):
            self.logger.info("HGS settings not found, generating from config...")
            from symfluence.models.hydrogeosphere.preprocessor import HGSPreProcessor
            preprocessor = HGSPreProcessor(self.config, self.logger)
            if isinstance(self.config, dict) and not preprocessor.config_dict:
                preprocessor.config_dict = self.config
            preprocessor.project_dir = self.project_dir
            preprocessor.run_preprocessing()

        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            default='optimization',
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()

        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir, 'HYDROGEOSPHERE', self.experiment_id,
        )

        if source_hgs.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                dest = dirs['settings_dir']
                dest.mkdir(parents=True, exist_ok=True)
                for item in source_hgs.iterdir():
                    if item.is_file():
                        copy_file(item, dest / item.name)

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        settings_dir = self.project_dir / 'settings' / 'HYDROGEOSPHERE'
        return self.worker.run_model(
            self.config, settings_dir, output_dir,
        )

    def _update_file_manager_for_final_run(self) -> None:
        pass

    def _restore_file_manager_for_optimization(self) -> None:
        pass
