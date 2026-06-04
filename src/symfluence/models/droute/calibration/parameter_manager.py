# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""dRoute parameter manager (Saint-Venant routing + inline/subgrid lakes).

Calibrates the dRoute routing parameters -- per-reach Manning's n and the inline/subgrid lake
rating coefficients -- via parameter regionalization (transfer functions). The optimisation
variables exposed to the optimiser are the regionalization COEFFICIENTS (e.g. ``manning_n_a``,
``manning_n_b``) rather than one value per reach, so a full-watershed routing calibration stays
low-dimensional. ``update_model_files`` expands the coefficients to per-reach physical values and
writes them to a routing-parameter file that the dRoute worker applies to the network.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.registries import R
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager

from .droute_regionalization import create_droute_regionalization

# Physical bounds for the dRoute routing parameters (min, max, transform). Manning's n and the
# lake rating params mirror the standalone dRoute calibration experiments.
DROUTE_PARAM_BOUNDS: Dict[str, Dict[str, Any]] = {
    'manning_n':       {'min': 0.015, 'max': 0.12,  'transform': 'linear'},
    'lake_q_ref':      {'min': 0.05,  'max': 500.0, 'transform': 'log'},
    'lake_exp':        {'min': 1.0,   'max': 3.0,   'transform': 'linear'},
    'lake_q_min':      {'min': 0.0,   'max': 50.0,  'transform': 'linear'},
    'lake_spill_coef': {'min': 0.1,   'max': 3.0,   'transform': 'linear'},
    'subgrid_q_ref':   {'min': 0.05,  'max': 1000.0, 'transform': 'log'},
    'subgrid_exp':     {'min': 1.0,   'max': 3.0,   'transform': 'linear'},
}
DEFAULT_DROUTE_PARAMS = ['manning_n', 'lake_q_ref', 'lake_q_min', 'lake_exp',
                         'subgrid_q_ref', 'subgrid_exp']


@R.parameter_managers.add('DROUTE')
class DRouteParameterManager(BaseParameterManager):
    """Parameter manager for dRoute (SV + lakes) routing calibration via transfer functions.

    Configuration keys:
        DROUTE_PARAMS_TO_CALIBRATE     : comma-separated routing params (default: Manning + lakes)
        DROUTE_REGIONALIZATION         : 'lumped' | 'transfer_function' | 'zones' | 'distributed'
        SETTINGS_DROUTE_PATH           : dRoute settings dir (holds droute_lakes.yaml)
        RIVER_NETWORK_SHAPEFILE        : path to the domain river-network shapefile

    The exposed calibration parameters are the regionalization coefficients; update_model_files
    writes the expanded per-reach values to ``<settings_dir>/droute_routing_params.json``.
    """

    def __init__(self, config: Any, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)
        self._region = None          # lazy ParameterRegionalization
        self._n_reaches: int = 0
        self._phys_params: List[str] = []   # physical routing params being calibrated

    # ---- configuration helpers --------------------------------------------------------------
    def _get(self, dotted, default, key):
        return self._get_config_value(lambda: dotted, default=default, dict_key=key)

    def _params_to_calibrate(self) -> List[str]:
        raw = self._get(None, None, 'DROUTE_PARAMS_TO_CALIBRATE')
        if not raw or str(raw).lower() in ('default', 'all'):
            return list(DEFAULT_DROUTE_PARAMS)
        return [p.strip() for p in str(raw).split(',') if p.strip()]

    def _regionalization_method(self) -> str:
        return str(self._get(None, 'transfer_function', 'DROUTE_REGIONALIZATION'))

    def _river_network_path(self) -> Path:
        p = self._get(None, None, 'RIVER_NETWORK_SHAPEFILE')
        if p:
            return Path(p)
        # discover under the domain shapefiles dir
        import glob
        cand = glob.glob(str(self.settings_dir.parent.parent / 'shapefiles' / 'river_network' / '*.shp'))
        if not cand:
            raise ConfigurationError("RIVER_NETWORK_SHAPEFILE not set and none found under shapefiles/river_network")
        return Path(cand[0])

    def _load_lakes(self) -> Dict[str, Dict]:
        import yaml
        f = Path(self.settings_dir) / 'droute_lakes.yaml'
        if not f.exists():
            return {'inline': {}, 'subgrid': {}}
        with open(f, encoding='utf-8') as fh:
            raw = yaml.safe_load(fh) or {}
        return {'inline': raw.get('inline_lakes', {}) or {}, 'subgrid': raw.get('subgrid_lakes', {}) or {}}

    def _n_reaches_from_network(self) -> int:
        import geopandas as gpd
        return len(gpd.read_file(self._river_network_path()))

    def _build_regionalization(self):
        if self._region is not None:
            return
        self._phys_params = self._params_to_calibrate()
        # only keep params with known bounds
        self._phys_params = [p for p in self._phys_params if p in DROUTE_PARAM_BOUNDS]
        param_bounds = {p: (DROUTE_PARAM_BOUNDS[p]['min'], DROUTE_PARAM_BOUNDS[p]['max'])
                        for p in self._phys_params}
        self._n_reaches = self._n_reaches_from_network()
        lakes = self._load_lakes()
        self._region = create_droute_regionalization(
            method=self._regionalization_method(),
            param_bounds=param_bounds,
            n_reaches=self._n_reaches,
            river_network_path=self._river_network_path(),
            lakes=lakes,
            logger=self.logger,
        )
        self.logger.info(
            f"dRoute regionalization '{self._regionalization_method()}': {len(self._phys_params)} "
            f"physical params over {self._n_reaches} reaches -> "
            f"{len(self._region.get_calibration_parameters())} calibration coefficients")

    # ---- BaseParameterManager contract ------------------------------------------------------
    def _get_parameter_names(self) -> List[str]:
        self._build_regionalization()
        return list(self._region.get_calibration_parameters().keys())

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        self._build_regionalization()
        return {name: {'min': lo, 'max': hi, 'transform': 'linear'}
                for name, (lo, hi) in self._region.get_calibration_parameters().items()}

    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        # midpoint of each coefficient range (gives spatially-uniform params at the bound midpoints)
        bounds = self.param_bounds
        return {name: 0.5 * (b['min'] + b['max']) for name, b in bounds.items()}

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        """Expand calibration coefficients to per-reach physical params and write them out."""
        self._build_regionalization()
        try:
            arr, pnames = self._region.to_distributed({k: float(v) for k, v in params.items()})
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            self.logger.error(f"dRoute regionalization expansion failed: {e}")
            return False
        # clamp each physical parameter to its bounds
        for j, p in enumerate(pnames):
            b = DROUTE_PARAM_BOUNDS.get(p)
            if b:
                arr[:, j] = np.clip(arr[:, j], b['min'], b['max'])
        per_reach = {p: arr[:, j].tolist() for j, p in enumerate(pnames)}
        out = Path(self.settings_dir) / 'droute_routing_params.json'
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump({'n_reaches': int(self._n_reaches), 'params': per_reach}, fh)
        self.logger.debug(f"Wrote per-reach routing params -> {out}")
        return True
