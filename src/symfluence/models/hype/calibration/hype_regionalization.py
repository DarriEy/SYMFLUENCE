# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE-specific adapter for parameter regionalization.

HYPE parameters in par.txt are keyed by either land-use class (LU) or
soil class, not by catchment/HRU.  The regionalization framework treats each
class as a spatial "unit":

- Land-use dependent params (ttmp, cmlt, cevp, srrcs, ...) have one value
  per LU type.
- Soil-class dependent params (wcwp, wcfc, wcep, rrcs1, rrcs2) have one
  value per soil type.
- Global params (lp, epotdist, rcgrw, rivvel, damp) are scalars.

The adapter composes separate ParameterRegionalization instances
(one per parameter group) and presents a unified coefficient interface.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.calibration.regionalization.strategies import (
    LumpedRegionalization,
    ParameterRegionalization,
    RegionalizationFactory,
)

HYPE_LU_PARAM_CONFIG: Dict[str, Dict[str, Any]] = {
    'ttmp':  {'attribute': 'mean_elevation', 'calibrate_b': True, 'b_sign': 'negative'},  # colder threshold at high elev (CV=1.02)
    'cmlt':  {'attribute': 'temp_C',         'calibrate_b': True, 'b_sign': 'negative'},  # colder areas → higher melt factor (CV=0.61)
    'cevp':  {'attribute': 'aridity',        'calibrate_b': True, 'b_sign': 'positive'},  # drier climate → more ET (CV=0.21)
    'srrcs': {'attribute': 'precip_mm_yr',   'calibrate_b': True, 'b_sign': 'positive'},  # more precip → faster runoff (CV=0.21)
}

HYPE_SOIL_PARAM_CONFIG: Dict[str, Dict[str, Any]] = {
    'wcwp':  {'attribute': 'silt_fraction',  'calibrate_b': True, 'b_sign': 'positive'},  # more silt → higher wilting point (CV=0.53)
    'wcfc':  {'attribute': 'silt_fraction',  'calibrate_b': True, 'b_sign': 'positive'},  # more silt → higher field capacity (CV=0.53)
    'wcep':  {'attribute': 'bulk_density',   'calibrate_b': True, 'b_sign': 'negative'},  # denser soil → lower eff. porosity (CV=0.35)
    'rrcs1': {'attribute': 'sand_fraction',  'calibrate_b': True, 'b_sign': 'positive'},  # more sand → faster drainage (CV=0.32)
    'rrcs2': {'attribute': 'bedrock_depth',  'calibrate_b': True, 'b_sign': 'negative'},  # deeper bedrock → slower deep drainage (CV=0.12)
}

HYPE_GLOBAL_PARAMS: List[str] = [
    'lp', 'epotdist', 'rcgrw', 'rivvel', 'damp',
    'ttpi', 'cmrefr',
]

_IGBP_ATTRIBUTES: Dict[int, Dict[str, float]] = {
    1:  {'lai': 5.0, 'vegetation_height': 20.0, 'root_depth': 2.0, 'glacier_fraction': 0.0},
    2:  {'lai': 6.0, 'vegetation_height': 25.0, 'root_depth': 2.5, 'glacier_fraction': 0.0},
    3:  {'lai': 3.5, 'vegetation_height': 18.0, 'root_depth': 1.8, 'glacier_fraction': 0.0},
    4:  {'lai': 4.5, 'vegetation_height': 20.0, 'root_depth': 2.0, 'glacier_fraction': 0.0},
    5:  {'lai': 4.0, 'vegetation_height': 18.0, 'root_depth': 2.0, 'glacier_fraction': 0.0},
    6:  {'lai': 2.0, 'vegetation_height': 3.0,  'root_depth': 1.0, 'glacier_fraction': 0.0},
    7:  {'lai': 1.5, 'vegetation_height': 1.5,  'root_depth': 0.8, 'glacier_fraction': 0.0},
    8:  {'lai': 3.0, 'vegetation_height': 10.0, 'root_depth': 1.5, 'glacier_fraction': 0.0},
    9:  {'lai': 2.0, 'vegetation_height': 5.0,  'root_depth': 1.0, 'glacier_fraction': 0.0},
    10: {'lai': 2.5, 'vegetation_height': 0.5,  'root_depth': 0.8, 'glacier_fraction': 0.0},
    11: {'lai': 3.0, 'vegetation_height': 2.0,  'root_depth': 0.5, 'glacier_fraction': 0.0},
    12: {'lai': 3.5, 'vegetation_height': 1.0,  'root_depth': 0.8, 'glacier_fraction': 0.0},
    13: {'lai': 1.0, 'vegetation_height': 10.0, 'root_depth': 0.3, 'glacier_fraction': 0.0},
    14: {'lai': 3.0, 'vegetation_height': 5.0,  'root_depth': 1.0, 'glacier_fraction': 0.0},
    15: {'lai': 0.1, 'vegetation_height': 0.1,  'root_depth': 0.1, 'glacier_fraction': 1.0},
    16: {'lai': 0.5, 'vegetation_height': 0.2,  'root_depth': 0.3, 'glacier_fraction': 0.0},
    17: {'lai': 0.0, 'vegetation_height': 0.0,  'root_depth': 0.0, 'glacier_fraction': 0.0},
}


def _compute_lu_climate_attributes(
    geodata_path: Path,
    slc_to_lu: Dict,
    climate_stats_path: Optional[Path],
    logger: logging.Logger,
) -> Dict[int, Dict[str, float]]:
    """Compute area-weighted mean attributes per LU class from GeoData + climate stats."""
    attrs_by_lu: Dict[int, Dict[str, float]] = {}
    try:
        gd = pd.read_csv(geodata_path, sep='\t')
        slc_cols = [c for c in gd.columns if c.startswith('SLC_')]
        if not slc_cols:
            return attrs_by_lu

        climate_cols = ['elev_mean']
        if climate_stats_path and Path(climate_stats_path).exists():
            cs = pd.read_csv(climate_stats_path)
            if len(cs) == len(gd):
                for col in ['precip_mm_yr', 'aridity', 'snow_frac', 'temp_C']:
                    if col in cs.columns:
                        gd[col] = cs[col].values
                        climate_cols.append(col)

        weighted_sums: Dict[int, Dict[str, float]] = {}
        total_weights: Dict[int, float] = {}

        for _, row in gd.iterrows():
            for sc in slc_cols:
                frac = row[sc]
                if frac <= 0:
                    continue
                slc_id = int(sc.split('_')[1])
                lu_id = slc_to_lu.get(slc_id)
                if lu_id is None:
                    continue
                lu_key = int(lu_id)
                if lu_key not in weighted_sums:
                    weighted_sums[lu_key] = {c: 0.0 for c in climate_cols}
                    total_weights[lu_key] = 0.0
                for col in climate_cols:
                    val = row.get(col)
                    if val is not None and not np.isnan(val):
                        weighted_sums[lu_key][col] += frac * val
                total_weights[lu_key] += frac

        for lu_id in weighted_sums:
            w = total_weights[lu_id]
            if w > 0:
                attrs_by_lu[lu_id] = {
                    col: weighted_sums[lu_id][col] / w for col in climate_cols
                }

        logger.info(
            f"Computed per-LU climate attributes for {len(attrs_by_lu)} classes"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not compute LU climate attributes: {e}")

    return attrs_by_lu


def load_lu_attributes(
    geoclass_path: Path,
    logger: Optional[logging.Logger] = None,
    geodata_path: Optional[Path] = None,
    climate_stats_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    logger = logger or logging.getLogger(__name__)
    geoclass_df = pd.read_csv(geoclass_path, sep='\t', skiprows=1, header=None)
    lu_ids = np.sort(geoclass_df.iloc[:, 1].unique())
    slc_to_lu = dict(zip(geoclass_df.iloc[:, 0], geoclass_df.iloc[:, 1]))

    lu_climate: Dict[int, Dict[str, float]] = {}
    if geodata_path is not None and Path(geodata_path).exists():
        lu_climate = _compute_lu_climate_attributes(
            geodata_path, slc_to_lu, climate_stats_path, logger
        )

    rows = []
    for lu_id in lu_ids:
        attrs = _IGBP_ATTRIBUTES.get(int(lu_id), _IGBP_ATTRIBUTES[16])
        row = {'lu_id': int(lu_id)}
        row.update(attrs)
        climate = lu_climate.get(int(lu_id), {})
        row['mean_elevation'] = climate.get('elev_mean', 0.0)
        row['precip_mm_yr'] = climate.get('precip_mm_yr', 1000.0)
        row['aridity'] = climate.get('aridity', 1.0)
        row['snow_frac'] = climate.get('snow_frac', 0.3)
        row['temp_C'] = climate.get('temp_C', 2.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info(f"Loaded LU attributes: {len(df)} classes, IDs={list(lu_ids)}")
    return df, lu_ids


# USDA soil texture class properties (mean clay/sand fractions)
_USDA_TEXTURE: Dict[int, Dict[str, float]] = {
    0:  {'clay_fraction': 0.0,  'sand_fraction': 0.0,  'bulk_density': 1.0,  'bedrock_depth': 10.0},
    1:  {'clay_fraction': 0.03, 'sand_fraction': 0.92, 'bulk_density': 1.55, 'bedrock_depth': 10.0},
    2:  {'clay_fraction': 0.06, 'sand_fraction': 0.82, 'bulk_density': 1.50, 'bedrock_depth': 10.0},
    3:  {'clay_fraction': 0.10, 'sand_fraction': 0.65, 'bulk_density': 1.45, 'bedrock_depth': 10.0},
    4:  {'clay_fraction': 0.13, 'sand_fraction': 0.20, 'bulk_density': 1.35, 'bedrock_depth': 10.0},
    5:  {'clay_fraction': 0.07, 'sand_fraction': 0.50, 'bulk_density': 1.40, 'bedrock_depth': 10.0},
    6:  {'clay_fraction': 0.18, 'sand_fraction': 0.43, 'bulk_density': 1.40, 'bedrock_depth': 10.0},
    7:  {'clay_fraction': 0.27, 'sand_fraction': 0.58, 'bulk_density': 1.35, 'bedrock_depth': 10.0},
    8:  {'clay_fraction': 0.34, 'sand_fraction': 0.32, 'bulk_density': 1.30, 'bedrock_depth': 10.0},
    9:  {'clay_fraction': 0.40, 'sand_fraction': 0.52, 'bulk_density': 1.25, 'bedrock_depth': 10.0},
    10: {'clay_fraction': 0.15, 'sand_fraction': 0.08, 'bulk_density': 1.30, 'bedrock_depth': 10.0},
    11: {'clay_fraction': 0.47, 'sand_fraction': 0.07, 'bulk_density': 1.20, 'bedrock_depth': 10.0},
    12: {'clay_fraction': 0.55, 'sand_fraction': 0.20, 'bulk_density': 1.15, 'bedrock_depth': 10.0},
}


def load_soil_attributes(
    geoclass_path: Path,
    soil_attributes_csv: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    logger = logger or logging.getLogger(__name__)
    geoclass_df = pd.read_csv(geoclass_path, sep='\t', skiprows=1, header=None)
    soil_ids = np.sort(geoclass_df.iloc[:, 2].unique())
    n_soil = len(soil_ids)

    if soil_attributes_csv is not None and Path(soil_attributes_csv).exists():
        df = pd.read_csv(soil_attributes_csv)
        if len(df) >= n_soil:
            df = df[df['soil_id'].isin(soil_ids.astype(int))]
            logger.info(f"Loaded soil attributes from CSV: {len(df)} classes, columns={list(df.columns)}")
            return df, soil_ids
        logger.warning(
            f"Soil CSV has {len(df)} rows but GeoClass has {n_soil} — using USDA lookup"
        )

    rows = []
    for soil_id in soil_ids:
        props = _USDA_TEXTURE.get(int(soil_id), {'clay_fraction': 0.2, 'sand_fraction': 0.4, 'bulk_density': 1.3, 'bedrock_depth': 10.0})
        rows.append({'soil_id': int(soil_id), **props})
    df = pd.DataFrame(rows)
    logger.info(f"Loaded USDA texture properties for {n_soil} soil classes")
    return df, soil_ids


class HypeRegionalization:
    """Composite regionalization for HYPE's three parameter groups."""

    def __init__(
        self,
        lu_strategy: Optional[ParameterRegionalization],
        soil_strategy: Optional[ParameterRegionalization],
        global_strategy: Optional[ParameterRegionalization],
        lu_ids: np.ndarray,
        soil_ids: np.ndarray,
        logger: Optional[logging.Logger] = None,
    ):
        self.lu_strategy = lu_strategy
        self.soil_strategy = soil_strategy
        self.global_strategy = global_strategy
        self.lu_ids = lu_ids
        self.soil_ids = soil_ids
        self.logger = logger or logging.getLogger(__name__)

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        if self.lu_strategy:
            bounds.update(self.lu_strategy.get_calibration_parameters())
        if self.soil_strategy:
            bounds.update(self.soil_strategy.get_calibration_parameters())
        if self.global_strategy:
            bounds.update(self.global_strategy.get_calibration_parameters())
        return bounds

    def get_coefficient_transforms(self) -> Dict[str, str]:
        transforms: Dict[str, str] = {}
        for strat in (self.lu_strategy, self.soil_strategy, self.global_strategy):
            if strat:
                transforms.update(strat.get_coefficient_transforms())
        return transforms

    def expand_to_par_values(
        self, calibration_params: Dict[str, float],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        if self.lu_strategy:
            lu_keys = set(self.lu_strategy.get_calibration_parameters())
            lu_params = {k: v for k, v in calibration_params.items() if k in lu_keys}
            arr, names = self.lu_strategy.to_distributed(lu_params)
            for i, name in enumerate(names):
                max_lu = int(self.lu_ids.max())
                values = np.zeros(max_lu)
                for j, lu_id in enumerate(self.lu_ids):
                    values[int(lu_id) - 1] = arr[j, i]
                result[name] = values.tolist()

        if self.soil_strategy:
            soil_keys = set(self.soil_strategy.get_calibration_parameters())
            soil_params = {k: v for k, v in calibration_params.items() if k in soil_keys}
            arr, names = self.soil_strategy.to_distributed(soil_params)
            for i, name in enumerate(names):
                max_soil = int(self.soil_ids.max())
                values = np.zeros(max_soil)
                for j, soil_id in enumerate(self.soil_ids):
                    values[int(soil_id) - 1] = arr[j, i]
                result[name] = values.tolist()

        if self.global_strategy:
            global_keys = set(self.global_strategy.get_calibration_parameters())
            global_params = {k: v for k, v in calibration_params.items() if k in global_keys}
            arr, names = self.global_strategy.to_distributed(global_params)
            for i, name in enumerate(names):
                result[name] = float(arr[0, i])

        return result


def create_hype_regionalization(
    method: str,
    param_bounds: Dict[str, Tuple[float, float]],
    geoclass_path: Path,
    geodata_path: Optional[Path] = None,
    climate_stats_path: Optional[Path] = None,
    soil_attributes_csv: Optional[Path] = None,
    lu_param_config: Optional[Dict[str, Dict]] = None,
    soil_param_config: Optional[Dict[str, Dict]] = None,
    global_params: Optional[List[str]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> HypeRegionalization:
    logger = logger or logging.getLogger(__name__)
    lu_config = lu_param_config or HYPE_LU_PARAM_CONFIG
    soil_config = soil_param_config or HYPE_SOIL_PARAM_CONFIG
    global_names = global_params or HYPE_GLOBAL_PARAMS
    config: Dict[str, Any] = dict(extra_config or {})

    lu_attrs_df, lu_ids = load_lu_attributes(
        geoclass_path, logger, geodata_path=geodata_path,
        climate_stats_path=climate_stats_path,
    )
    soil_attrs_df, soil_ids = load_soil_attributes(geoclass_path, soil_attributes_csv, logger)

    lu_bounds = {k: v for k, v in param_bounds.items() if k in lu_config}
    soil_bounds = {k: v for k, v in param_bounds.items() if k in soil_config}
    global_bounds = {k: v for k, v in param_bounds.items() if k in global_names}

    lu_strategy: Optional[ParameterRegionalization] = None
    soil_strategy: Optional[ParameterRegionalization] = None
    global_strategy: Optional[ParameterRegionalization] = None

    if method == 'lumped':
        if lu_bounds:
            lu_strategy = LumpedRegionalization(lu_bounds, len(lu_ids), logger)
        if soil_bounds:
            soil_strategy = LumpedRegionalization(soil_bounds, len(soil_ids), logger)
        if global_bounds:
            global_strategy = LumpedRegionalization(global_bounds, 1, logger)
    elif method == 'transfer_function':
        if lu_bounds:
            lu_tf_config = dict(config)
            lu_tf_config['TRANSFER_FUNCTION_PARAM_CONFIG'] = lu_config
            lu_strategy = RegionalizationFactory.create(
                method='transfer_function', param_bounds=lu_bounds,
                n_units=len(lu_ids), config=lu_tf_config,
                attributes=lu_attrs_df, logger=logger,
            )
        if soil_bounds:
            soil_tf_config = dict(config)
            soil_tf_config['TRANSFER_FUNCTION_PARAM_CONFIG'] = soil_config
            soil_strategy = RegionalizationFactory.create(
                method='transfer_function', param_bounds=soil_bounds,
                n_units=len(soil_ids), config=soil_tf_config,
                attributes=soil_attrs_df, logger=logger,
            )
        if global_bounds:
            global_strategy = LumpedRegionalization(global_bounds, 1, logger)
    else:
        raise ValueError(f"Unsupported HYPE regionalization method: {method}")

    logger.info(
        f"HYPE regionalization ({method}): "
        f"{len(lu_bounds)} LU params x {len(lu_ids)} classes, "
        f"{len(soil_bounds)} soil params x {len(soil_ids)} classes, "
        f"{len(global_bounds)} global params"
    )

    return HypeRegionalization(
        lu_strategy=lu_strategy, soil_strategy=soil_strategy,
        global_strategy=global_strategy, lu_ids=lu_ids,
        soil_ids=soil_ids, logger=logger,
    )
