#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GR Streamflow Evaluator
"""

import logging
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Optional, Dict, Any

from symfluence.evaluation.registry import EvaluationRegistry
from .streamflow import StreamflowEvaluator
from symfluence.core.constants import UnitConversion

@EvaluationRegistry.register('GR_STREAMFLOW')
class GRStreamflowEvaluator(StreamflowEvaluator):
    """Streamflow evaluator for GR models"""
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get GR output files (CSV for lumped, NetCDF for distributed)"""
        # Lumped mode output
        lumped_file = sim_dir / 'GR_results.csv'
        if lumped_file.exists():
            return [lumped_file]
            
        # Distributed mode output
        experiment_id = self.config.get('EXPERIMENT_ID')
        dist_file = sim_dir / f"{self.domain_name}_{experiment_id}_runs_def.nc"
        if dist_file.exists():
            return [dist_file]
            
        # Fallback to generic NetCDF search if routing was used
        return super().get_simulation_files(sim_dir)
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow data from GR output files"""
        sim_file = sim_files[0]
        
        if sim_file.suffix == '.csv':
            return self._extract_lumped_gr_streamflow(sim_file)
        else:
            # Check if it's mizuRoute output or GR distributed output
            if self._is_mizuroute_output(sim_file):
                return self._extract_mizuroute_streamflow(sim_file)
            else:
                return self._extract_distributed_gr_streamflow(sim_file)

    def _extract_lumped_gr_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from GR lumped CSV output"""
        df_sim = pd.read_csv(sim_file, index_col='datetime', parse_dates=True)
        # GR4J output is in mm/day. Convert to cms.
        area_m2 = self._get_catchment_area()
        area_km2 = area_m2 / 1e6
        simulated_streamflow = df_sim['q_sim'] * area_km2 / UnitConversion.MM_DAY_TO_CMS
        return simulated_streamflow

    def _extract_distributed_gr_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from GR distributed NetCDF output"""
        with xr.open_dataset(sim_file) as ds:
            # q_routed is in mm/day, aggregated across HRUs
            if 'q_routed' in ds.variables:
                sim_data = ds['q_routed'].sum(dim='gru').to_pandas()
                area_m2 = self._get_catchment_area()
                area_km2 = area_m2 / 1e6
                simulated_streamflow = sim_data * area_km2 / UnitConversion.MM_DAY_TO_CMS
                return simulated_streamflow
            else:
                raise ValueError(f"No suitable streamflow variable found in {sim_file}")

    def _get_catchment_area(self) -> float:
        """Get catchment area in m2, prioritized for GR"""
        # GR specific catchment area logic if needed, otherwise use base
        # GR usually uses the GRU_area from HRU shapefile
        return super()._get_catchment_area()
