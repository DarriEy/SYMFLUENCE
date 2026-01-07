#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NextGen (ngen) Calibration Targets

Provides calibration target classes for ngen model outputs.
Currently supports streamflow calibration with plans for snow, ET, etc.

Note: This module has been refactored to use the centralized evaluators in 
symfluence.evaluation.evaluators.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from symfluence.evaluation import metrics
from symfluence.evaluation.evaluators import ModelEvaluator, StreamflowEvaluator


class NgenStreamflowTarget(StreamflowEvaluator):
    """NextGen-specific streamflow evaluator that handles nexus-style outputs."""
    
    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        self.station_id = config.get('STATION_ID', None)

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Find ngen nexus output files (recursive search)."""
        files = list(sim_dir.glob('nex-*_output.csv'))
        if not files:
            # Try recursive search if not found in top level
            files = list(sim_dir.glob('**/nex-*_output.csv'))
        return files

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow from multiple ngen nexus output files."""
        all_streamflow = []
        
        for nexus_file in sim_files:
            try:
                # ngen output format: index, datetime, flow
                df = pd.read_csv(nexus_file, header=None, names=['index', 'datetime', 'flow'])
                if df.empty:
                    continue
                
                index = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(None)
                s = pd.Series(
                    df['flow'].values,
                    index=index,
                    name=nexus_file.stem
                )
                all_streamflow.append(s)
            except Exception as e:
                self.logger.error(f"Error reading {nexus_file}: {e}")
                continue
        
        if not all_streamflow:
            return pd.Series(dtype=float)
            
        # Sum all nexus outputs for total catchment outflow
        combined = pd.concat(all_streamflow, axis=1).sum(axis=1)
        return combined.sort_index()

    def calculate_metrics(self, sim: Any = None, experiment_id: str = None, 
                         output_dir: Optional[Path] = None, **kwargs) -> Dict[str, float]:
        """
        Standardized metrics calculation for NextGen.
        
        Args:
            sim: Optional Path to simulation directory or pre-loaded pd.Series.
                 If None, determined from experiment_id/output_dir.
            experiment_id: Experiment identifier (legacy, uses config if None)
            output_dir: Optional output directory (for parallel mode)
        """
        if sim is None:
            # Determine simulation directory
            if output_dir is not None:
                sim = Path(output_dir)
            else:
                exp_id = experiment_id or self.config.get('EXPERIMENT_ID')
                sim = self.project_dir / 'simulations' / exp_id / 'NGEN'
            
        # Use base class method with our specialized data extraction
        return super().calculate_metrics(sim=sim, **kwargs)

    def _get_catchment_area(self) -> float:
        """Detailed catchment area calculation for NextGen."""
        from pathlib import Path
        import geopandas as gpd
        import numpy as np

        cfg_area = (self.config.get("catchment", {}) or {}).get("area_km2")
        if cfg_area:
            return float(cfg_area)

        domain_dir = self.project_dir
        shp_dir = domain_dir / "shapefiles" / "catchment"
        if not shp_dir.exists():
            return 100.0

        candidates = sorted(shp_dir.glob("*HRUs_GRUs.shp")) + sorted(shp_dir.glob("*.shp"))
        shp_path = next((p for p in candidates if p.exists()), None)
        if not shp_path:
            return 100.0

        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is None or gdf.crs.is_geographic:
                gdf = gdf.to_crs("EPSG:5070")

            area_km2 = float(gdf.geometry.area.sum() / 1e6)
            return area_km2
        except Exception as e:
            self.logger.warning(f"Error calculating catchment area from {shp_path}: {e}")
            return 100.0
