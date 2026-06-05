# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM calibration parameter manager."""

import configparser
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.core.registries import R
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager


@R.parameter_managers.add('CWATM')
class CWatMParameterManager(BaseParameterManager):
    """Manages calibratable parameters for CWatM.

    CWatM calibration parameters live in the ``[CALIBRATION]`` section
    of the settings INI file — no external parameter files to update.
    """

    DEFAULT_BOUNDS: Dict[str, Dict[str, float]] = {
        'SnowMeltCoef': {'min': 0.001, 'max': 0.01},
        'crop_correct': {'min': 0.8, 'max': 1.8},
        'soildepth_factor': {'min': 0.8, 'max': 1.5},
        'preferentialFlowConstant': {'min': 0.5, 'max': 8.0},
        'arnoBeta_add': {'min': 0.01, 'max': 1.0},
        'factor_interflow': {'min': 0.33, 'max': 3.0},
        'recessionCoeff_factor': {'min': 0.1, 'max': 10.0},
        'runoffConc_factor': {'min': 0.1, 'max': 10.0},
        'manningsN': {'min': 0.1, 'max': 10.0},
        'normalStorageLimit': {'min': 0.15, 'max': 0.85},
        'lakeAFactor': {'min': 0.33, 'max': 3.0},
        'lakeEvaFactor': {'min': 0.8, 'max': 2.0},
    }

    PREPROCESSOR_DEFAULTS: Dict[str, float] = {
        'SnowMeltCoef': 0.0027,
        'crop_correct': 1.0,
        'soildepth_factor': 1.0,
        'preferentialFlowConstant': 4.5,
        'arnoBeta_add': 0.1,
        'factor_interflow': 2.0,
        'recessionCoeff_factor': 1.0,
        'runoffConc_factor': 1.0,
        'manningsN': 1.0,
        'normalStorageLimit': 0.5,
        'lakeAFactor': 1.0,
        'lakeEvaFactor': 1.0,
    }

    def _get_parameter_names(self) -> List[str]:
        config_params = self._get_config_value(
            lambda: self.config.model.cwatm.params_to_calibrate if hasattr(self.config.model.cwatm, 'params_to_calibrate') else None,
            default=None,
        )
        if config_params:
            return [p.strip() for p in config_params.split(',')]
        return [
            'SnowMeltCoef', 'crop_correct', 'soildepth_factor',
            'arnoBeta_add', 'recessionCoeff_factor', 'manningsN',
        ]

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        return {p: self.DEFAULT_BOUNDS[p] for p in self._get_parameter_names()
                if p in self.DEFAULT_BOUNDS}

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        return {p: self.PREPROCESSOR_DEFAULTS.get(
            p, (self.DEFAULT_BOUNDS[p]['min'] + self.DEFAULT_BOUNDS[p]['max']) / 2
        ) for p in self._get_parameter_names() if p in self.DEFAULT_BOUNDS}

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        for p, v in params.items():
            if p in self.DEFAULT_BOUNDS:
                params[p] = max(self.DEFAULT_BOUNDS[p]['min'],
                                min(v, self.DEFAULT_BOUNDS[p]['max']))
        return True

    def update_model_files(self, params: Dict[str, float], settings_dir: Optional[Path] = None, **kwargs) -> bool:  # type: ignore[override]
        """Update calibration parameters in the settings INI."""
        if settings_dir is None:
            settings_dir = self.settings_dir if hasattr(self, 'settings_dir') else Path('.')

        ini_path = settings_dir / 'settings.ini'
        if not ini_path.exists():
            self.logger.error(f"Settings file not found: {ini_path}")
            return False

        ini = configparser.ConfigParser()
        ini.optionxform = str
        ini.read(ini_path)

        if 'CALIBRATION' not in ini:
            self.logger.error("No [CALIBRATION] section in settings.ini")
            return False

        for param_name, value in params.items():
            ini['CALIBRATION'][param_name] = str(value)

        with open(ini_path, 'w') as f:
            ini.write(f)

        return True
