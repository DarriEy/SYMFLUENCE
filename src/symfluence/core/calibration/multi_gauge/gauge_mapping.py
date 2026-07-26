# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared gauge-to-segment/subbasin mapping via spatial join.

This module extracts the common spatial-join logic previously duplicated in
``models.fuse.calibration.multi_gauge_metrics.ensure_gauge_segment_mapping``
and ``models.hype.calibration.multi_gauge_metrics.ensure_hype_gauge_mapping``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_gauge_mapping(
    project_dir: Path,
    lamah_root: Path,
    domain_name: str,
    output_subdir: str,
    output_filename: str,
    logger: logging.Logger,
    prefer_coastal_basins: bool = False,
) -> Optional[Path]:
    """Generate a gauge-to-segment (or subbasin) mapping CSV if it does not exist.

    Spatial-joins LaMAH-Ice gauge points against the domain's river-basin
    polygons and writes the result to the canonical project path:
    ``<project_dir>/settings/<output_subdir>/<output_filename>``.

    Args:
        project_dir: Root project directory (``domain_<name>``).
        lamah_root: Root of the LaMAH-Ice dataset (parent of ``D_gauges/``).
        domain_name: Domain name used in shapefile naming.
        output_subdir: Subdirectory under ``settings/`` (e.g. ``'mizuRoute'``,
            ``'HYPE'``).
        output_filename: Name of the output CSV (e.g.
            ``'gauge_segment_mapping.csv'``).
        logger: Logger instance for status/warning messages.
        prefer_coastal_basins: If *True*, try the
            ``_with_coastal.shp`` variant of the river-basins shapefile first
            and fall back to ``_semidistributed.shp``.

    Returns:
        The canonical CSV path on success, or *None* if required inputs are
        missing.
    """
    canonical = project_dir / 'settings' / output_subdir / output_filename
    if canonical.exists():
        return canonical

    # --- locate input shapefiles ---
    gauges_shp = lamah_root / 'D_gauges' / '3_shapefiles' / 'gauges.shp'

    basins_shp = (
        project_dir / 'shapefiles' / 'river_basins'
        / f"{domain_name}_riverBasins_semidistributed.shp"
    )
    if prefer_coastal_basins:
        coastal_shp = (
            project_dir / 'shapefiles' / 'river_basins'
            / f"{domain_name}_riverBasins_with_coastal.shp"
        )
        if coastal_shp.exists():
            basins_shp = coastal_shp

    if not gauges_shp.exists() or not basins_shp.exists():
        logger.warning(
            "Cannot generate gauge mapping: missing "
            f"gauges shapefile ({gauges_shp.exists()=}) or "
            f"river basins shapefile ({basins_shp.exists()=})"
        )
        return None

    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas unavailable; cannot auto-generate gauge mapping")
        return None

    logger.info(f"Auto-generating gauge mapping at {canonical}")

    # --- read & project ---
    gauges = gpd.read_file(gauges_shp).to_crs('EPSG:4326')
    basins = gpd.read_file(basins_shp)
    if basins.crs is None:
        basins = basins.set_crs('EPSG:4326')
    else:
        basins = basins.to_crs('EPSG:4326')

    # --- spatial join ---
    seg_col = 'GRU_ID' if 'GRU_ID' in basins.columns else 'subid'
    joined = gpd.sjoin_nearest(
        gauges[['id', 'name', 'geometry']],
        basins[[seg_col, 'geometry']],
        how='left',
        distance_col='distance_to_segment',
    )
    joined = joined.rename(columns={seg_col: 'nearest_segment'})

    out = pd.DataFrame({
        'id': joined['id'].astype(int),
        'name': joined['name'],
        'nearest_segment': joined['nearest_segment'].astype(int),
        'distance_to_segment': joined['distance_to_segment'].astype(float),
    })

    canonical.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(canonical, index=False)
    logger.info(f"Wrote {len(out)} gauge mapping rows to {canonical}")
    return canonical
