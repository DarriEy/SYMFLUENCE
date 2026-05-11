# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Noah-MP Model Optimizer."""
import logging
import re
from pathlib import Path

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry

from .parameter_manager import NoahMPParameterManager  # noqa: F401
from .worker import NoahMPWorker  # noqa: F401


@OptimizerRegistry.register_optimizer('NOAHMP')
class NoahMPModelOptimizer(BaseModelOptimizer):
    def __init__(self, config, logger: logging.Logger, optimization_settings_dir=None, reporting_manager=None):
        exp_id = config.get('EXPERIMENT_ID') if isinstance(config, dict) else config.domain.experiment_id
        data_dir_str = config.get('SYMFLUENCE_DATA_DIR') if isinstance(config, dict) else config.system.data_dir
        domain_name_str = config.get('DOMAIN_NAME') if isinstance(config, dict) else config.domain.name
        self.data_dir = Path(data_dir_str)
        self.domain_name = domain_name_str
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.noahmp_sim_dir = self.project_dir / 'simulations' / exp_id / 'NOAHMP'
        self.noahmp_setup_dir = self.project_dir / 'settings' / 'NOAHMP'
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

    def _get_model_name(self) -> str:
        return 'NOAHMP'

    def _create_parameter_manager(self):
        return NoahMPParameterManager(self.config, self.logger, self.noahmp_setup_dir)

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        nml = self.noahmp_setup_dir / 'namelist.input'
        if nml.exists():
            text = nml.read_text()
            text = re.sub(r'(output_filename\s*=\s*)"[^"]*"', rf'\1"{output_dir / "output.nc"}"', text)
            nml.write_text(text)
        return self.worker.run_model(self.config, self.noahmp_setup_dir, output_dir)

    def _get_final_file_manager_path(self) -> Path:
        return self.noahmp_setup_dir / 'namelist.input'

    def _setup_parallel_dirs(self) -> None:
        algorithm = self._get_config_value(lambda: self.config.optimization.algorithm, default='optimization', dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM').lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'
        self.parallel_dirs = self.setup_parallel_processing(base_dir, 'NOAHMP', self.experiment_id)
        if self.noahmp_setup_dir.exists():
            self.copy_base_settings(self.noahmp_setup_dir, self.parallel_dirs, 'NOAHMP')
