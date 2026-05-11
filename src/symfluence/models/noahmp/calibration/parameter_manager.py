# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Noah-MP Parameter Manager."""
import logging
from pathlib import Path
from typing import List

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_noahmp_bounds
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('NOAHMP')
class NoahMPParameterManager(BaseParameterManager):
    NAMELIST_PARAMS = {'rain_snow_thresh', 'ZREF', 'refkdt'}
    SOILPARM_PARAMS = {'bexp', 'smcmax', 'smcref', 'psisat', 'dksat', 'smcwlt'}

    def __init__(self, config, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, default='.', dict_key='SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.noahmp_settings_dir = self.project_dir / 'settings' / 'NOAHMP'
        ps = self._get_config_value(lambda: self.config.model.noahmp.params_to_calibrate if self.config.model and self.config.model.noahmp else None, default=None, dict_key='NOAHMP_PARAMS_TO_CALIBRATE')
        if not ps or ps == 'default':
            ps = 'refkdt,dksat,bexp,smcmax,slope,noah_czil'
        self.noahmp_params = [p.strip() for p in ps.split(',') if p.strip()]

    def _get_parameter_names(self) -> List[str]:
        return self.noahmp_params

    def _load_parameter_bounds(self):
        return get_noahmp_bounds()

    def update_model_files(self, params) -> bool:
        import re
        nml = self.noahmp_settings_dir / 'namelist.input'
        if not nml.exists(): return False
        nml_keys = {'rain_snow_thresh': 'rain_snow_thresh', 'ZREF': 'ZREF', 'refkdt': 'refkdt'}
        nml_p = {k: v for k, v in params.items() if k in self.NAMELIST_PARAMS}
        if nml_p:
            text = nml.read_text()
            for pn, val in nml_p.items():
                k = nml_keys.get(pn)
                if k: text = re.sub(rf'(\s*{k}\s*=\s*)[\d.eE+\-]+', rf'\g<1>{val:.6f}', text)
            nml.write_text(text)
        soil_p = {k: v for k, v in params.items() if k in self.SOILPARM_PARAMS}
        if soil_p:
            from symfluence.models.noahmp.calibration.worker import NoahMPWorker
            w = NoahMPWorker.__new__(NoahMPWorker); w.logger = self.logger
            w._update_soilparm_tbl(self.noahmp_settings_dir, nml, soil_p)
        return True

    def get_initial_parameters(self):
        return {p: (self.param_bounds[p]['min'] + self.param_bounds[p]['max']) / 2 for p in self.noahmp_params if p in self.param_bounds}
