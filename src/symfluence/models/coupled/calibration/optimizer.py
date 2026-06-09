# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Generic coupled-model optimizer (registered ``COUPLED``).

Calibrates an arbitrary set of coupled standalone models (land + optional snow/groundwater/routing)
jointly with the standard BaseModelOptimizer algorithms (DDS/PSO/...), over the union of the
participating models' parameters. The coupled parameter manager and worker delegate parameter
subsets to each standalone model and run the land->...->routing coupling through dCoupler (with a
sequential delegation fallback). This generalizes the SUMMA+MODFLOW ``COUPLED_GW`` optimizer to any
coupled model chain; activation is handled by optimization_manager when a coupled, calibratable
setup is detected.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.registries import R
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer

from .parameter_manager import CoupledModelParameterManager  # noqa: F401 (registration)
from .worker import CoupledModelWorker  # noqa: F401 (registration)


@R.optimizers.add('COUPLED')
class CoupledModelOptimizer(BaseModelOptimizer):
    """Joint calibration of a coupled standalone-model chain via dCoupler + the standard algorithms."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger,
                 optimization_settings_dir: Optional[Path] = None,
                 reporting_manager: Optional[Any] = None):
        self.config = config
        self.land_model_name = str(
            (config.get('HYDROLOGICAL_MODEL', 'SUMMA') if isinstance(config, dict)
             else getattr(getattr(config, 'model', None), 'hydrological_model', None) or 'SUMMA')
        ).split(',')[0].upper()
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

    def _get_model_name(self) -> str:
        return 'COUPLED'

    def _create_parameter_manager(self):
        return CoupledModelParameterManager(self.config, self.logger, self.project_dir / 'settings')

    def _get_final_file_manager_path(self) -> Path:
        # Final evaluation drives the land model; for SUMMA that's fileManager.txt.
        if self.land_model_name == 'SUMMA':
            fm = self._get_config_value(lambda: self.config.model.summa.filemanager,
                                        default='fileManager.txt', dict_key='SETTINGS_SUMMA_FILEMANAGER')
            if not fm or fm == 'default':
                fm = 'fileManager.txt'
            return self.project_dir / 'settings' / 'SUMMA' / fm
        return self.project_dir / 'settings' / self.land_model_name

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        return self.worker.run_model(self.config, self.project_dir / 'settings', output_dir)

    def _update_file_manager_output_path(self, output_dir: Path) -> None:
        """Point the land model's output into ``output_dir/<land>`` (mirrors COUPLED_GW).

        The coupled worker runs each model into ``output_dir/<model>`` and the routing model reads
        the land output from there, so SUMMA must write to ``output_dir/SUMMA`` (not ``output_dir/``).
        Also restores settingsPath to the project land settings (where the trial params were written).
        """
        if self.land_model_name != 'SUMMA':
            return super()._update_file_manager_output_path(output_dir)
        fm = self._get_final_file_manager_path()
        if not fm.exists() or not fm.is_file():
            return
        try:
            land_output = str(output_dir / self.land_model_name)
            if not land_output.endswith('/'):
                land_output += '/'
            land_settings = str(self.project_dir / 'settings' / self.land_model_name)
            if not land_settings.endswith('/'):
                land_settings += '/'
            with open(fm, encoding='utf-8') as f:
                lines = f.readlines()
            out = []
            for line in lines:
                if 'outputPath' in line and not line.strip().startswith('!'):
                    out.append(f"outputPath '{land_output}' \n")
                elif 'settingsPath' in line and not line.strip().startswith('!'):
                    out.append(f"settingsPath '{land_settings}' \n")
                else:
                    out.append(line)
            with open(fm, 'w', encoding='utf-8') as f:
                f.writelines(out)
        except (FileNotFoundError, IOError, ValueError) as e:
            self.logger.error(f"Failed to update file manager output path: {e}")
