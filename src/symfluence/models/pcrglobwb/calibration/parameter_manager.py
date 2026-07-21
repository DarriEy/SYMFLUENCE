# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB calibration parameter manager."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager


@R.parameter_managers.add('PCRGLOBWB')
class PCRGLOBWBParameterManager(BaseParameterManager):
    """Manages calibratable parameters for PCR-GLOBWB.

    Parameters are stored in generated NetCDF files under
    ``settings/PCRGLOBWB/parameters/``.  Each parameter maps to a
    specific variable in a specific file.
    """

    # Parameters that map to individual .map files in the parameters/ directory
    MAP_FILE_PARAMS: Set[str] = {
        'KSat1', 'KSat2', 'satVolWC1', 'satVolWC2', 'resVolWC1', 'resVolWC2',
        'recessionCoeff', 'kSatAquifer', 'specificYield',
    }

    # Physically reasonable bounds (Saxton & Rawls 2006, literature)
    DEFAULT_BOUNDS: Dict[str, Dict[str, float]] = {
        'KSat1': {'min': 0.01, 'max': 10.0},       # m/day
        'KSat2': {'min': 0.005, 'max': 5.0},        # m/day
        'satVolWC1': {'min': 0.3, 'max': 0.6},      # porosity
        'satVolWC2': {'min': 0.3, 'max': 0.6},
        'resVolWC1': {'min': 0.01, 'max': 0.2},     # wilting point
        'resVolWC2': {'min': 0.01, 'max': 0.2},
        'degreeDayFactor': {'min': 0.001, 'max': 0.008},  # m/°C/day
        'freezingT': {'min': -3.0, 'max': 3.0},     # °C
        'snowWaterHoldingCap': {'min': 0.0, 'max': 0.3},
        'refreezingCoeff': {'min': 0.01, 'max': 0.2},
        'recessionCoeff': {'min': 1e-5, 'max': 0.05},    # day⁻¹
        'kSatAquifer': {'min': 0.001, 'max': 1.0},       # m/day
        'specificYield': {'min': 0.01, 'max': 0.3},
        'manningsN': {'min': 0.01, 'max': 0.1},
        'ROUTE_ALPHA': {'min': 0.0, 'max': 0.95},
        'ROUTE_BETA': {'min': 0.9, 'max': 0.9999},
        'ROUTE_SPLIT': {'min': 0.1, 'max': 0.9},
        'ROUTE_BASEFLOW': {'min': 0.0, 'max': 15.0},    # m³/s
    }

    PREPROCESSOR_DEFAULTS: Dict[str, float] = {
        'KSat1': 0.917, 'KSat2': 0.459,
        'satVolWC1': 0.457, 'satVolWC2': 0.457,
        'resVolWC1': 0.095, 'resVolWC2': 0.095,
        'degreeDayFactor': 0.0025, 'freezingT': 0.0,
        'snowWaterHoldingCap': 0.1, 'refreezingCoeff': 0.05,
        'recessionCoeff': 0.001, 'kSatAquifer': 0.092,
        'specificYield': 0.15, 'manningsN': 0.04,
        'ROUTE_ALPHA': 0.5, 'ROUTE_BETA': 0.98,
        'ROUTE_SPLIT': 0.5, 'ROUTE_BASEFLOW': 0.0,
    }

    ROUTING_PARAMS: Set[str] = {'ROUTE_ALPHA', 'ROUTE_BETA', 'ROUTE_SPLIT', 'ROUTE_BASEFLOW'}
    INI_PARAMS: Set[str] = {'degreeDayFactor', 'freezingT', 'snowWaterHoldingCap',
                            'refreezingCoeff', 'manningsN'}

    def _get_parameter_names(self) -> List[str]:
        config_params = self._get_config_value(
            lambda: self.config.model.pcrglobwb.params_to_calibrate if hasattr(self.config.model.pcrglobwb, 'params_to_calibrate') else None,
            default=None,
        )
        if config_params:
            return [p.strip() for p in config_params.split(',')]
        return [
            'KSat1', 'KSat2', 'recessionCoeff',
            'degreeDayFactor', 'freezingT',
            'manningsN',
            'ROUTE_ALPHA', 'ROUTE_BETA', 'ROUTE_SPLIT', 'ROUTE_BASEFLOW',
        ]

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        return {p: self.DEFAULT_BOUNDS[p] for p in self._get_parameter_names()
                if p in self.DEFAULT_BOUNDS}

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        return {p: self.PREPROCESSOR_DEFAULTS.get(p, (self.DEFAULT_BOUNDS[p]['min'] + self.DEFAULT_BOUNDS[p]['max']) / 2)
                for p in self._get_parameter_names() if p in self.DEFAULT_BOUNDS}

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        # Enforce resVolWC < satVolWC (wilting point < porosity)
        for layer in ['1', '2']:
            res_key, sat_key = f'resVolWC{layer}', f'satVolWC{layer}'
            if res_key in params and sat_key in params:
                if params[res_key] >= params[sat_key]:
                    params[res_key] = params[sat_key] * 0.15
        # Clip to bounds
        for p, v in params.items():
            if p in self.DEFAULT_BOUNDS:
                params[p] = max(self.DEFAULT_BOUNDS[p]['min'],
                                min(v, self.DEFAULT_BOUNDS[p]['max']))
        return True

    def update_model_files(self, params: Dict[str, float], settings_dir: Optional[Path] = None, **kwargs) -> bool:  # type: ignore[override]
        """Update .map parameter files and INI with calibrated values."""
        if settings_dir is None:
            settings_dir = self.settings_dir if hasattr(self, 'settings_dir') else Path('.')

        params_dir = settings_dir / 'parameters'
        if not params_dir.exists():
            self.logger.error(f"Parameters directory not found: {params_dir}")
            return False

        ini_updates: Dict[str, float] = {}

        for param_name, value in params.items():
            if param_name in self.ROUTING_PARAMS:
                continue
            if param_name in self.INI_PARAMS:
                ini_updates[param_name] = value
                continue
            if param_name in self.MAP_FILE_PARAMS:
                map_path = params_dir / f'{param_name}.map'
                if not map_path.exists():
                    self.logger.warning(f"Map file not found: {map_path}")
                    continue
                # Overwrite .map with uniform value via PCRaster
                self._update_map_file(map_path, value)

        if ini_updates:
            self._update_ini_params(settings_dir, ini_updates)

        return True

    def _update_map_file(self, map_path: Path, value: float) -> None:
        """Overwrite a PCRaster .map file with a uniform scalar value."""
        import subprocess
        import sys

        script = (
            f"import pcraster as pcr; "
            f"pcr.setclone('{map_path}'); "
            f"pcr.report(pcr.spatial(pcr.scalar({value})), '{map_path}')"
        )

        pyver = f"{sys.version_info.major}{sys.version_info.minor}"
        for env_name in [f"pcraster{pyver}", "pcraster"]:
            try:
                result = run_subprocess(
                    ["conda", "run", "-n", env_name, "python", "-c", script],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0:
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        self.logger.warning(f"Could not update {map_path}")

    def _update_ini_params(self, settings_dir: Path, params: Dict[str, float]) -> None:
        """Update snow/routing parameters in the INI file."""
        import configparser
        ini_path = settings_dir / 'setup.ini'
        if not ini_path.exists():
            return

        ini = configparser.ConfigParser()
        ini.optionxform = str
        ini.read(ini_path)

        snow_params = {'degreeDayFactor', 'freezingT', 'snowWaterHoldingCap', 'refreezingCoeff'}
        for section in ['forestOptions', 'grasslandOptions', 'irrPaddyOptions', 'irrNonPaddyOptions']:
            if section in ini:
                for pname, pval in params.items():
                    if pname in snow_params:
                        ini[section][pname] = str(pval)

        if 'manningsN' in params and 'routingOptions' in ini:
            ini['routingOptions']['manningsN'] = str(params['manningsN'])

        with open(ini_path, 'w') as f:
            ini.write(f)
