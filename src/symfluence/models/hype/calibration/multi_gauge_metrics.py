# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Multi-gauge calibration metrics for HYPE.

Reuses the model-agnostic observation loading, quality filtering, and
KGE aggregation from the shared MultiGaugeMetrics base class, overriding
only the simulated-flow extraction to read HYPE's timeCOUT.txt format
instead of mizuRoute NetCDF.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.optimization.multi_gauge.metrics import (
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
    """Generate gauge-to-subbasin mapping for HYPE at the canonical path.

    Mirrors :func:`ensure_gauge_segment_mapping` but writes to
    ``settings/HYPE/gauge_subbasin_mapping.csv``.  The underlying
    spatial join is identical -- HYPE subbasin IDs come from the
    ``GRU_ID`` column of the river-basins shapefile.

    Delegates to :func:`symfluence.optimization.multi_gauge.gauge_mapping.ensure_gauge_mapping`.
    """
    from symfluence.optimization.multi_gauge.gauge_mapping import ensure_gauge_mapping

    return ensure_gauge_mapping(
        project_dir,
        lamah_root,
        domain_name,
        output_subdir='HYPE',
        output_filename='gauge_subbasin_mapping.csv',
        logger=logger,
        prefer_coastal_basins=True,
    )
