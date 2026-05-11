# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Multi-gauge calibration metrics for HYPE.

Reuses the model-agnostic observation loading, quality filtering, and
KGE aggregation from the FUSE MultiGaugeMetrics base class, overriding
only the simulated-flow extraction to read HYPE's timeCOUT.txt format
instead of mizuRoute NetCDF.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.models.fuse.calibration.multi_gauge_metrics import (
    MultiGaugeMetrics,
)


class HYPEMultiGaugeMetrics(MultiGaugeMetrics):
    """Multi-gauge metrics for HYPE using timeCOUT.txt output.

    HYPE writes per-subbasin daily discharge to timeCOUT.txt. The gauge→
    subbasin mapping (``nearest_segment`` column) maps directly to HYPE
    subbasin IDs (the column headers in timeCOUT.txt).
    """

    _cout_cache: Optional[pd.DataFrame] = None

    def _load_cout(self, cout_path: Path) -> Optional[pd.DataFrame]:
        """Load and cache HYPE timeCOUT.txt."""
        if self._cout_cache is not None:
            return self._cout_cache
        try:
            df = pd.read_csv(cout_path, sep='\t', skiprows=1)
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
                df = df.set_index('DATE')
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            self._cout_cache = df
            return df
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error reading timeCOUT.txt: {e}")
            return None

    def _extract_simulated_at_segment(
        self,
        output_path: Path,
        segment_id: int,
        topology_path: Optional[Path] = None,
    ) -> Optional[pd.Series]:
        """Extract simulated discharge for a HYPE subbasin from timeCOUT.txt.

        Args:
            output_path: Path to the directory containing timeCOUT.txt,
                         or to timeCOUT.txt itself.
            segment_id: HYPE subbasin ID (column header in timeCOUT.txt).
            topology_path: Unused (kept for API compatibility).

        Returns:
            Daily discharge Series (m³/s) or None.
        """
        cout_path = output_path if output_path.name == 'timeCOUT.txt' else output_path / 'timeCOUT.txt'
        df = self._load_cout(cout_path)
        if df is None:
            return None

        col = str(segment_id)
        if col not in df.columns:
            self.logger.warning(f"Subbasin {segment_id} not in timeCOUT.txt columns")
            return None

        return df[col].dropna()

    def calculate_multi_gauge_metrics(
        self,
        output_path: Path,
        gauge_ids: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        topology_path: Optional[Path] = None,
        min_gauges: int = 5,
        aggregation: str = 'mean',
        weights: Optional[Dict[int, float]] = None,
        filter_config: Optional[Dict[str, Any]] = None,
        min_overlap_days: int = 10,
        kge_floor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Delegate to the parent but clear the cout cache per evaluation."""
        self._cout_cache = None
        return super().calculate_multi_gauge_metrics(
            mizuroute_output_path=output_path,
            gauge_ids=gauge_ids,
            start_date=start_date,
            end_date=end_date,
            topology_path=topology_path,
            min_gauges=min_gauges,
            aggregation=aggregation,
            weights=weights,
            filter_config=filter_config,
            min_overlap_days=min_overlap_days,
            kge_floor=kge_floor,
        )


def ensure_hype_gauge_mapping(
    project_dir: Path,
    lamah_root: Path,
    domain_name: str,
    logger: logging.Logger,
) -> Optional[Path]:
    """Generate gauge→subbasin mapping for HYPE at the canonical path.

    Mirrors :func:`ensure_gauge_segment_mapping` but writes to
    ``settings/HYPE/gauge_subbasin_mapping.csv``.  The underlying
    spatial join is identical — HYPE subbasin IDs come from the
    ``GRU_ID`` column of the river-basins shapefile.
    """
    canonical = project_dir / 'settings' / 'HYPE' / 'gauge_subbasin_mapping.csv'
    if canonical.exists():
        return canonical

    gauges_shp = lamah_root / 'D_gauges' / '3_shapefiles' / 'gauges.shp'
    basins_shp = (
        project_dir / 'shapefiles' / 'river_basins'
        / f"{domain_name}_riverBasins_semidistributed.shp"
    )
    coastal_shp = (
        project_dir / 'shapefiles' / 'river_basins'
        / f"{domain_name}_riverBasins_with_coastal.shp"
    )
    # Prefer the version that includes coastal basins
    if coastal_shp.exists():
        basins_shp = coastal_shp

    if not gauges_shp.exists() or not basins_shp.exists():
        logger.warning(
            "Cannot generate HYPE gauge mapping: missing "
            f"gauges shapefile ({gauges_shp.exists()=}) or "
            f"river basins shapefile ({basins_shp.exists()=})"
        )
        return None

    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas unavailable; cannot auto-generate gauge mapping")
        return None

    logger.info(f"Auto-generating HYPE gauge→subbasin mapping at {canonical}")
    gauges = gpd.read_file(gauges_shp).to_crs('EPSG:4326')
    basins = gpd.read_file(basins_shp)
    if basins.crs is None:
        basins = basins.set_crs('EPSG:4326')
    else:
        basins = basins.to_crs('EPSG:4326')

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
    logger.info(f"Wrote {len(out)} gauge→subbasin rows to {canonical}")
    return canonical
