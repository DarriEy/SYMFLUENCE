#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GR Worker

Worker implementation for GR model optimization.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import xarray as xr
import numpy as np

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry
from symfluence.models.gr.runner import GRRunner
from symfluence.evaluation.metrics import kge, nse, rmse, mae
from symfluence.core.constants import UnitConversion

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('GR')
class GRWorker(BaseWorker):
    """
    Worker for GR model calibration.
    """

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        For GR, parameters are passed directly to the runner.
        We don't need to modify any files here.
        """
        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run GR model via GRRunner.
        """
        try:
            # Get parameters from kwargs (passed by _evaluate_once)
            # WorkerTask passes params as a direct argument to run_model in newer BaseWorker
            # but let's be safe and check both kwargs and params if it were passed explicitly
            params = kwargs.get('params')
            
            # If not in kwargs, it might be in task (if we're calling it from evaluate)
            # Actually, BaseWorker._evaluate_once calls run_model(task.config, task.settings_dir, task.output_dir, **task.additional_data)
            # So params is NOT passed by default in BaseWorker unless it's in additional_data.
            # We need to ensure params are passed.
            if params:
                self.logger.info(f"Worker received params: {params}")
            else:
                self.logger.warning("Worker run_model received NO params!")
            
            # Create a runner instance
            # We use the config provided in the task and pass settings_dir for isolation
            runner = GRRunner(config, self.logger, settings_dir=settings_dir)
            
            # Override output directory to the one provided for this worker
            runner.output_dir = output_dir
            runner.output_path = output_dir
            
            # Execute GR
            success_path = runner.run_gr(params=params)
            
            return success_path is not None
        except Exception as e:
            self.logger.error(f"Error running GR model in worker: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from GR output.
        """
        try:
            domain_name = config.get('DOMAIN_NAME')
            spatial_mode = config.get('GR_SPATIAL_MODE', 'lumped')
            
            # Read observed streamflow
            obs_values, common_index = self._get_observed_streamflow(config)
            if obs_values is None:
                return {'kge': self.penalty_score}

            # Read simulated streamflow based on spatial mode
            if spatial_mode == 'lumped':
                sim_file = output_dir / 'GR_results.csv'
                if not sim_file.exists():
                    self.logger.error(f"GR lumped output not found: {sim_file}")
                    return {'kge': self.penalty_score}
                
                df_sim = pd.read_csv(sim_file, index_col='datetime', parse_dates=True)
                # GR4J output is in mm/day. Convert to cms.
                area_km2 = self._get_catchment_area(config)
                self.logger.info(f"DEBUG: GRWorker using catchment area: {area_km2:.2f} km2")
                simulated_streamflow = df_sim['q_sim'] * area_km2 / UnitConversion.MM_DAY_TO_CMS
            else:
                # Distributed mode produces NetCDF
                experiment_id = config.get('EXPERIMENT_ID')
                sim_file = output_dir / f"{domain_name}_{experiment_id}_runs_def.nc"
                if not sim_file.exists():
                    self.logger.error(f"GR distributed output not found: {sim_file}")
                    return {'kge': self.penalty_score}
                
                with xr.open_dataset(sim_file) as ds:
                    # q_routed is in mm/day, aggregated across HRUs in the runner for lumped-equivalent
                    # or needs aggregation here if distributed. 
                    # GRRunner currently saves mizuRoute-compatible format.
                    if 'q_routed' in ds.variables:
                        sim_data = ds['q_routed'].sum(dim='gru').to_pandas()
                        area_km2 = self._get_catchment_area(config)
                        simulated_streamflow = sim_data * area_km2 / UnitConversion.MM_DAY_TO_CMS
                    else:
                        self.logger.error(f"No q_routed in GR output. Vars: {list(ds.variables)}")
                        return {'kge': self.penalty_score}

            # Align and calculate
            sim_aligned = simulated_streamflow.reindex(common_index).dropna()
            obs_aligned = pd.Series(obs_values, index=common_index).reindex(sim_aligned.index)
            
            if len(sim_aligned) == 0:
                self.logger.error("No overlapping data for GR metrics")
                return {'kge': self.penalty_score}

            metrics = {
                'kge': float(kge(obs_aligned.values, sim_aligned.values)),
                'nse': float(nse(obs_aligned.values, sim_aligned.values)),
                'rmse': float(rmse(obs_aligned.values, sim_aligned.values)),
                'mae': float(mae(obs_aligned.values, sim_aligned.values)),
            }
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating GR metrics: {e}")
            return {'kge': self.penalty_score}

    def _get_observed_streamflow(self, config: Dict[str, Any]):
        """Helper to get aligned observed streamflow."""
        try:
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"
            
            obs_file = config.get('OBSERVATIONS_PATH', 'default')
            if obs_file == 'default':
                obs_file = project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv"
            else:
                obs_file = Path(obs_file)

            if not obs_file.exists():
                self.logger.error(f"Obs file not found: {obs_file}")
                return None, None

            df_obs = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            observed = df_obs['discharge_cms'].resample('D').mean()
            return observed.values, observed.index
        except Exception as e:
            self.logger.error(f"Error reading observations: {e}")
            return None, None

    def _get_catchment_area(self, config: Dict[str, Any]) -> float:
        """Get catchment area in km2."""
        try:
            import geopandas as gpd
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"
            
            catchment_path = Path(config.get('CATCHMENT_PATH', project_dir / 'shapefiles' / 'catchment'))
            discretization = config.get('DOMAIN_DISCRETIZATION', 'elevation')
            catchment_name = config.get('CATCHMENT_SHP_NAME', f"{domain_name}_HRUs_{discretization}.shp")
            
            catchment_file = catchment_path / catchment_name
            if not catchment_file.exists():
                return 1000.0 # Default fallback
                
            gdf = gpd.read_file(catchment_file)
            if 'GRU_area' in gdf.columns:
                return gdf['GRU_area'].sum() / 1e6
            
            if gdf.crs and not gdf.crs.is_geographic:
                area = gdf.geometry.area.sum()
            else:
                area = gdf.to_crs(gdf.estimate_utm_crs()).geometry.area.sum()
            return area / 1e6
        except Exception:
            return 1000.0

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for parallel execution."""
        return _evaluate_gr_parameters_worker(task_data)


def _evaluate_gr_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI execution.
    Naming matches convention for dynamic resolution in BaseModelOptimizer.
    """
    worker = GRWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
