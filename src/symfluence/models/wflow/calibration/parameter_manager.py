"""Wflow Parameter Manager."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('WFLOW')
class WflowParameterManager(BaseParameterManager):
    """Handles Wflow parameter bounds, normalization, and file updates."""

    PARAM_VAR_MAP = {
        'KsatVer': 'KsatVer', 'f': 'f', 'SoilThickness': 'SoilThickness',
        'InfiltCapPath': 'InfiltCapPath', 'RootingDepth': 'RootingDepth',
        'KsatHorFrac': 'KsatHorFrac', 'n_river': 'N_River',
        'PathFrac': 'PathFrac', 'thetaS': 'thetaS', 'thetaR': 'thetaR',
        'Cfmax': 'Cfmax', 'TT': 'TT', 'TTI': 'TTI', 'TTM': 'TTM',
        'WHC': 'WHC',
        'CanopyGapFraction': 'CanopyGapFraction', 'Cmax': 'Cmax',
    }

    DEFAULT_BOUNDS: Dict[str, Dict[str, Any]] = {
        'KsatVer': {'min': 10.0, 'max': 3000.0, 'transform': 'log'},
        'f': {'min': 0.5, 'max': 10.0},
        'SoilThickness': {'min': 100.0, 'max': 8000.0},
        'InfiltCapPath': {'min': 10.0, 'max': 500.0},
        'RootingDepth': {'min': 10.0, 'max': 2000.0},
        'KsatHorFrac': {'min': 0.0, 'max': 100.0},
        'n_river': {'min': 0.01, 'max': 0.1},
        'PathFrac': {'min': 0.0, 'max': 1.0},
        'thetaS': {'min': 0.3, 'max': 0.8},
        'thetaR': {'min': 0.01, 'max': 0.15},
        'Cfmax': {'min': 0.5, 'max': 10.0},
        'TT': {'min': -3.0, 'max': 3.0},
        'TTI': {'min': 0.1, 'max': 10.0},
        'TTM': {'min': -2.0, 'max': 2.0},
        'WHC': {'min': 0.0, 'max': 0.4},
        'CanopyGapFraction': {'min': 0.1, 'max': 0.95},
        'Cmax': {'min': 0.5, 'max': 5.0},
    }

    def __init__(self, config: Dict, logger: logging.Logger, wflow_settings_dir: Path):
        super().__init__(config, logger, wflow_settings_dir)
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        wflow_params_str = self._get_config_value(
            lambda: self.config.model.wflow.params_to_calibrate,
            default=None, dict_key='WFLOW_PARAMS_TO_CALIBRATE'
        )
        if wflow_params_str is None:
            wflow_params_str = ','.join(self.PARAM_VAR_MAP.keys())
        self.params_to_calibrate = [p.strip() for p in wflow_params_str.split(',') if p.strip()]

    def _get_parameter_names(self) -> List[str]:
        return list(self.params_to_calibrate)

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        return {p: self.DEFAULT_BOUNDS[p] for p in self.params_to_calibrate if p in self.DEFAULT_BOUNDS}

    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        initial = {}
        for p, b in self._load_parameter_bounds().items():
            if b.get('transform') == 'log' and b['min'] > 0 and b['max'] > 0:
                initial[p] = float(np.sqrt(b['min'] * b['max']))
            else:
                initial[p] = (b['min'] + b['max']) / 2.0
        return initial

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        if 'thetaR' in params and 'thetaS' in params:
            if params['thetaR'] >= params['thetaS']:
                params['thetaR'] = params['thetaS'] * 0.15
        bounds = self._load_parameter_bounds()
        for param, value in params.items():
            if param in bounds:
                params[param] = float(np.clip(value, bounds[param]['min'], bounds[param]['max']))
        return True

    def update_model_files(self, params: Dict[str, float], settings_dir: Optional[Path] = None, **kwargs) -> bool:
        try:
            self.validate_parameters(params)
            target_dir = settings_dir or self.settings_dir
            staticmaps_name = self._get_config_value(
                lambda: self.config.model.wflow.staticmaps_file,
                default='wflow_staticmaps.nc', dict_key='WFLOW_STATICMAPS_FILE'
            )
            path = target_dir / staticmaps_name
            if not path.exists():
                return False
            ds = xr.open_dataset(path)
            ds_new = ds.load()
            ds.close()
            for param_name, value in params.items():
                nc_var = self.PARAM_VAR_MAP.get(param_name)
                if nc_var and nc_var in ds_new:
                    ds_new[nc_var].values[:] = value
            # Write to temp file then rename (avoids macOS HDF5 file lock)
            tmp_path = path.parent / f'{path.stem}_tmp{path.suffix}'
            ds_new.to_netcdf(tmp_path)
            import shutil
            shutil.move(str(tmp_path), str(path))
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to update Wflow parameters: {e}")
            return False
