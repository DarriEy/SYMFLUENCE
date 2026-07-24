# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM calibration optimizer."""
from __future__ import annotations

import re
from pathlib import Path

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.registries import R


@R.optimizers.add('CWATM')
class CWatMModelOptimizer(BaseModelOptimizer):
    """Optimizer for CWatM model calibration."""

    def __init__(self, config, logger, optimization_settings_dir=None, reporting_manager=None):
        self.cwatm_setup_dir = self.project_dir / 'settings' / 'CWATM' if hasattr(self, 'project_dir') else None
        super().__init__(config, logger, optimization_settings_dir, reporting_manager)
        if self.cwatm_setup_dir is None:
            data_dir = self._get_config_value(
                lambda: self.config.system.data_dir,
                default='.', dict_key='SYMFLUENCE_DATA_DIR',
            )
            domain = self._get_config_value(
                lambda: self.config.domain.name,
                default='', dict_key='DOMAIN_NAME',
            )
            self.cwatm_setup_dir = Path(data_dir) / f'domain_{domain}' / 'settings' / 'CWATM'

    def _get_model_name(self) -> str:
        return 'CWATM'

    def _get_final_file_manager_path(self) -> Path:
        return self.cwatm_setup_dir / 'settings.ini'

    def _create_parameter_manager(self):
        from .parameter_manager import CWatMParameterManager
        return CWatMParameterManager(self.config, self.logger, self.cwatm_setup_dir)

    def _check_routing_needed(self) -> bool:
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        return self.worker.run_model(self.config_dict, self.cwatm_setup_dir, output_dir)

    def _update_file_manager_for_final_run(self) -> None:
        """Update CWatM INI for full experiment period."""
        ini_path = self._get_final_file_manager_path()
        if not ini_path.exists():
            return

        import pandas as pd
        sim_start = self._get_config_value(
            lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START',
        )
        sim_end = self._get_config_value(
            lambda: self.config.domain.time_end, dict_key='EXPERIMENT_TIME_END',
        )
        if not sim_start or not sim_end:
            return

        start = pd.to_datetime(sim_start).strftime('%d/%m/%Y')
        end = pd.to_datetime(sim_end).strftime('%d/%m/%Y')

        content = ini_path.read_text()
        content = re.sub(r'(?m)^StepStart\s*=.*$', f'StepStart = {start}', content)
        content = re.sub(r'(?m)^StepEnd\s*=.*$', f'StepEnd = {end}', content)
        ini_path.write_text(content)

    def _update_file_manager_output_path(self, output_dir: Path) -> None:
        """Update CWatM INI output directory."""
        ini_path = self._get_final_file_manager_path()
        if not ini_path.exists():
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        content = ini_path.read_text()
        content = re.sub(r'(?m)^OUT_Dir\s*=.*$', f'OUT_Dir = {output_dir}', content)
        content = re.sub(r'(?m)^PathOut\s*=.*$', f'PathOut = {output_dir}', content)
        ini_path.write_text(content)
