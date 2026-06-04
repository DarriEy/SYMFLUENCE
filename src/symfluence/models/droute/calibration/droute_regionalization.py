# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""dRoute-specific adapter for parameter regionalization (transfer functions).

Wraps the model-agnostic regionalization framework
(:mod:`symfluence.optimization.regionalization.strategies`) with dRoute defaults and
per-REACH attribute loading from the domain river-network shapefile (and the dRoute lake
configuration). This is the routing counterpart of ``summa_regionalization`` (which regionalizes
per-HRU land-surface parameters): here the units are river reaches and the parameters are the
Saint-Venant routing roughness and the inline/subgrid lake rating coefficients.

A transfer function maps a physical reach attribute to a spatially-varying parameter:
    param_reach = a + b * attribute_norm        (MPR-style linear form)
so the calibration optimises a handful of coefficients (a, b) per parameter rather than one value
per reach, which keeps a full-watershed routing calibration low-dimensional and identifiable.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.optimization.regionalization.strategies import (
    ParameterRegionalization,
    RegionalizationFactory,
)

# Default mapping of dRoute routing parameters to reach attributes.
#   'attribute'    : the reach attribute that drives the spatial variation
#   'calibrate_b'  : True  -> the slope coefficient b is calibrated (param varies with the attr)
#                    False -> only the intercept a is calibrated (spatially uniform parameter)
#   'fallback'     : attribute to use if the primary one is unavailable/constant
# Physical rationale:
#   - Manning's n decreases with channel size, so it varies with (log) upstream drainage area;
#   - a reservoir/lake reference release scales with lake surface area / storage;
#   - rating exponents and the spill coefficient are kept spatially uniform by default (intercept
#     only) because there is little physical basis for an attribute relationship.
DROUTE_DEFAULT_PARAM_CONFIG: Dict[str, Dict[str, Any]] = {
    # Channel roughness — larger (higher-order, larger-drainage) reaches are smoother
    'manning_n':        {'attribute': 'log_drainage_area', 'calibrate_b': True, 'fallback': 'strm_order'},
    # Inline lake / reservoir rating — reference release scales with lake size
    'lake_q_ref':       {'attribute': 'log_lake_area',     'calibrate_b': True, 'fallback': 'log_storage_max'},
    'lake_q_min':       {'attribute': 'log_lake_area',     'calibrate_b': False},
    'lake_exp':         {'attribute': 'log_lake_area',     'calibrate_b': False},
    'lake_spill_coef':  {'attribute': 'log_lake_area',     'calibrate_b': False},
    # Subgrid (off-network) lake store rating — scales with the aggregate store size
    'subgrid_q_ref':    {'attribute': 'log_subgrid_storage_max', 'calibrate_b': True},
    'subgrid_exp':      {'attribute': 'log_subgrid_storage_max', 'calibrate_b': False},
}

# Attributes log-transformed before [0,1] normalization (heavy-tailed, span many orders).
DROUTE_LOG_TRANSFORM_ATTRS = ['drainage_area_km2', 'lake_area', 'storage_max', 'subgrid_storage_max']


def load_reach_attributes(
    river_network_path: Path,
    lakes: Optional[Dict[str, Dict]] = None,
    id_field: str = 'LINKNO',
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Load per-reach attributes for transfer-function regionalization.

    The returned DataFrame is indexed by REACH ORDER (0-based, matching the river-network
    shapefile row order, which is also the order dRoute indexes reaches) and contains the columns
    referenced by :data:`DROUTE_DEFAULT_PARAM_CONFIG` (plus their normalized/log variants, which
    the regionalization framework derives).

    Args:
        river_network_path: Path to the domain river-network shapefile (TauDEM-style fields
            ``LINKNO``, ``Slope``, ``Length``, ``strmOrder``, ``DSContArea``).
        lakes: Optional ``{'inline': {seg: {...}}, 'subgrid': {seg: {...}}}`` lake config (as parsed
            from ``droute_lakes.yaml``) to attach lake_area / storage_max attributes by segment id.
        id_field: Reach id field in the shapefile (default ``LINKNO``).
        logger: Logger instance.

    Returns:
        DataFrame with one row per reach and attribute columns.
    """
    logger = logger or logging.getLogger(__name__)
    import geopandas as gpd  # lazy heavy import

    rn = gpd.read_file(river_network_path)
    seg_ids = rn[id_field].astype(int).values
    n = len(rn)

    slope = rn['Slope'].astype(float).values if 'Slope' in rn else np.full(n, 0.005)
    length_m = rn['Length'].astype(float).values if 'Length' in rn else np.full(n, 1000.0)
    strm_order = rn['strmOrder'].astype(float).values if 'strmOrder' in rn else np.ones(n)
    # DSContArea is contributing area [m^2] at the reach outlet (TauDEM); fall back to magnitude.
    if 'DSContArea' in rn:
        drainage_km2 = np.clip(rn['DSContArea'].astype(float).values, 1.0, None) / 1e6
    elif 'Magnitude' in rn:
        drainage_km2 = np.clip(rn['Magnitude'].astype(float).values, 1.0, None)
    else:
        drainage_km2 = np.full(n, 1.0)

    df = pd.DataFrame({
        'seg_id': seg_ids,
        'slope': np.clip(slope, 1e-5, None),
        'length_km': length_m / 1000.0,
        'strm_order': strm_order,
        'drainage_area_km2': drainage_km2,
        'log_drainage_area': np.log10(drainage_km2),
    })

    # --- Lake attributes (per reach, 0 where the reach is not a lake) ----------------------------
    id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
    lake_area = np.zeros(n)
    storage_max = np.zeros(n)
    subgrid_smax = np.zeros(n)
    if lakes:
        for seg, rec in (lakes.get('inline', {}) or {}).items():
            i = id_to_idx.get(int(seg))
            if i is not None:
                lake_area[i] = float(rec.get('lake_area', 0.0) or 0.0)
                storage_max[i] = float(rec.get('storage_max', 0.0) or 0.0)
        for seg, rec in (lakes.get('subgrid', {}) or {}).items():
            i = id_to_idx.get(int(seg))
            if i is not None:
                subgrid_smax[i] = float(rec.get('subgrid_storage_max', 0.0) or 0.0)
    # log variants (guard zeros so non-lake reaches don't dominate the normalization)
    df['lake_area'] = lake_area
    df['log_lake_area'] = np.log10(np.clip(lake_area, 1.0, None))
    df['storage_max'] = storage_max
    df['log_storage_max'] = np.log10(np.clip(storage_max, 1.0, None))
    df['subgrid_storage_max'] = subgrid_smax
    df['log_subgrid_storage_max'] = np.log10(np.clip(subgrid_smax, 1.0, None))

    logger.info(f"Loaded reach attributes: {n} reaches, columns={list(df.columns)}")
    return df


def create_droute_regionalization(
    method: str,
    param_bounds: Dict[str, Tuple[float, float]],
    n_reaches: int,
    river_network_path: Path,
    lakes: Optional[Dict[str, Dict]] = None,
    param_config: Optional[Dict[str, Dict]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> ParameterRegionalization:
    """Create a regionalization strategy configured for dRoute routing parameters.

    Convenience wrapper around :class:`RegionalizationFactory` that loads reach attributes and
    applies dRoute defaults. ``method`` is ``'lumped'`` (one value per parameter, spatially
    uniform), ``'transfer_function'`` (MPR-style ``a + b*attr``), ``'zones'``, or ``'distributed'``.

    Args:
        method: regionalization method.
        param_bounds: ``{param_name: (min, max)}`` for the routing parameters to regionalize.
        n_reaches: number of river reaches.
        river_network_path: path to the river-network shapefile.
        lakes: optional parsed ``droute_lakes.yaml`` (inline/subgrid) for lake attributes.
        param_config: per-parameter config override; defaults to :data:`DROUTE_DEFAULT_PARAM_CONFIG`
            restricted to the parameters present in ``param_bounds``.
        extra_config: additional factory options (b_bounds, transfer_function_type, ...).
        logger: logger instance.

    Returns:
        A :class:`ParameterRegionalization` ready for use by the dRoute parameter manager.
    """
    logger = logger or logging.getLogger(__name__)
    config: Dict[str, Any] = dict(extra_config or {})

    attributes = None
    if method.lower().replace('-', '_') == 'transfer_function':
        attributes = load_reach_attributes(river_network_path, lakes, logger=logger)
        config.setdefault('TRANSFER_FUNCTION_LOG_ATTRS', DROUTE_LOG_TRANSFORM_ATTRS)

    if 'TRANSFER_FUNCTION_PARAM_CONFIG' not in config:
        cfg = param_config or DROUTE_DEFAULT_PARAM_CONFIG
        # only keep entries for parameters actually being calibrated
        config['TRANSFER_FUNCTION_PARAM_CONFIG'] = {p: cfg[p] for p in param_bounds if p in cfg}

    return RegionalizationFactory.create(
        method=method,
        param_bounds=param_bounds,
        n_units=n_reaches,
        config=config,
        attributes=attributes,
        logger=logger,
    )
