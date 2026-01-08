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
            import os
            
            # Get parameters from kwargs (passed by _evaluate_once)
            params = kwargs.get('params')
            self.logger.debug(f"GRWorker.run_model called with params: {params}")
            self.logger.debug(f"Current working directory: {os.getcwd()}")
            
            # Validate inputs
            if not config:
                self.logger.error("Config is None or empty")
                return False

            # Use a dictionary for local modifications to avoid SymfluenceConfig immutability/subscriptability issues
            if hasattr(config, 'to_dict'):
                local_config = config.to_dict(flatten=True)
            else:
                local_config = config.copy()

            # Update config with isolated paths for MizuRoute if provided (from ParallelExecutionMixin)
            mizu_dir = kwargs.get('mizuroute_dir')
            if mizu_dir:
                local_config['EXPERIMENT_OUTPUT_MIZUROUTE'] = mizu_dir
                self.logger.debug(f"Updated EXPERIMENT_OUTPUT_MIZUROUTE to {mizu_dir}")
                # Ensure mizuRoute directory exists
                Path(mizu_dir).mkdir(parents=True, exist_ok=True)
                
            mizu_settings_dir = kwargs.get('mizuroute_settings_dir')
            if mizu_settings_dir:
                local_config['SETTINGS_MIZU_PATH'] = mizu_settings_dir
                self.logger.debug(f"Updated SETTINGS_MIZU_PATH to {mizu_settings_dir}")
            
            # Ensure output directory exists and is writable
            output_path = Path(output_dir)

            # Also update GR output path for mizuRoute control file creation
            local_config['EXPERIMENT_OUTPUT_GR'] = str(output_path)
            self.logger.debug(f"Updated EXPERIMENT_OUTPUT_GR to {output_path}")
            
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured output_dir exists: {output_path} (abs: {output_path.resolve()})")
                # Test writeability
                test_file = output_path / '.test_write'
                test_file.write_text('test')
                test_file.unlink()
            except Exception as e:
                self.logger.error(f"Cannot write to output_dir {output_path}: {e}")
                return False
            
            # Ensure settings directory exists
            settings_path = Path(settings_dir)
            try:
                settings_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured settings_dir exists: {settings_path} (abs: {settings_path.resolve()})")
            except Exception as e:
                self.logger.error(f"Cannot create settings_dir {settings_path}: {e}")
                return False
            
            # Create a runner instance
            self.logger.debug(f"Creating GRRunner with settings_dir={settings_path}, output_dir={output_path}")
            runner = GRRunner(local_config, self.logger, settings_dir=settings_path)
            
            # Override output directory to the one provided for this worker
            runner.output_dir = output_path
            runner.output_path = output_path
            self.logger.debug(f"GRRunner created, output_path={runner.output_path}")
            
            # Execute GR
            self.logger.debug(f"Calling runner.run_gr with params={params}")
            success_path = runner.run_gr(params=params)
            self.logger.debug(f"runner.run_gr returned: {success_path}")
            
            if success_path is None:
                self.logger.error("runner.run_gr returned None")
            return success_path is not None
        except Exception as e:
            self.logger.error(f"Error running GR model in worker: {type(e).__name__}: {e}")
            import traceback
            tb_str = traceback.format_exc()
            self.logger.error(tb_str)
            # Store the error message on the instance for potential retrieval
            self._last_error = tb_str
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
            output_dir = Path(output_dir)
            domain_name = config.get('DOMAIN_NAME')
            spatial_mode = config.get('GR_SPATIAL_MODE', 'lumped')
            
            # Handle 'auto' spatial mode
            if spatial_mode in (None, 'auto', 'default'):
                domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                spatial_mode = 'distributed' if domain_method == 'delineate' else 'lumped'

            self.logger.debug(f"calculate_metrics: output_dir={output_dir}, spatial_mode={spatial_mode}, domain={domain_name}")
            
            # Validate output directory
            if not output_dir.exists():
                self.logger.error(f"Output directory does not exist: {output_dir}")
                return {'kge': self.penalty_score}
            
            # Read observed streamflow
            obs_values, common_index = self._get_observed_streamflow(config)
            if obs_values is None:
                self.logger.error("Could not get observed streamflow")
                return {'kge': self.penalty_score}

            # Read simulated streamflow based on spatial mode
            if spatial_mode == 'lumped':
                sim_file = output_dir / 'GR_results.csv'
                if not sim_file.exists():
                    self.logger.error(f"GR lumped output not found: {sim_file}")
                    return {'kge': self.penalty_score}
                
                df_sim = pd.read_csv(sim_file, index_col='datetime', parse_dates=True)
                # GR4J output is in mm/day. Convert to cms (matching observation units).
                area_km2 = self._get_catchment_area(config)
                # Conversion: Q(cms) = Q(mm/day) * Area(km²) * 1000 / 86400 = Area(km²) / 86.4
                simulated_streamflow = df_sim['q_sim'] * area_km2 / UnitConversion.MM_DAY_TO_CMS
            else:
                # Distributed mode produces NetCDF
                experiment_id = config.get('EXPERIMENT_ID')
                
                # Check for routed output first if routing was enabled
                mizuroute_dir_path = kwargs.get('mizuroute_dir')
                if mizuroute_dir_path:
                    mizuroute_dir = Path(mizuroute_dir_path)
                    if mizuroute_dir.exists():
                        # Find any .nc file in mizuRoute directory
                        mizu_files = list(mizuroute_dir.glob("*.nc"))
                        if mizu_files:
                            sim_file_routed = mizu_files[0]
                            self.logger.debug(f"Using routed output: {sim_file_routed}")
                            with xr.open_dataset(sim_file_routed) as ds:
                                # mizuRoute output usually has 'IRFroutedRunoff' or 'averageRoutedRunoff'
                                routing_var = None
                                for v in ['IRFroutedRunoff', 'averageRoutedRunoff', 'q_routed']:
                                    if v in ds.variables:
                                        routing_var = v
                                        break
                                
                                if not routing_var:
                                    self.logger.error(f"No discharge variable in routed output. Vars: {list(ds.variables)}")
                                    return {'kge': self.penalty_score}
                                    
                                # mizuRoute output uses 'seg' dimension for reach segments
                                if 'seg' in ds[routing_var].dims:
                                    simulated_streamflow = ds[routing_var].isel(seg=-1).to_pandas()
                                elif 'gru' in ds[routing_var].dims:
                                    simulated_streamflow = ds[routing_var].isel(gru=-1).to_pandas()
                                else:
                                    self.logger.error(f"Unknown spatial dimension in routed output: {ds[routing_var].dims}")
                                    return {'kge': self.penalty_score}
                                
                                # Check units and convert if needed
                                units = ds[routing_var].attrs.get('units', '').lower()
                                if 'm3' in units or 'cms' in units:
                                    pass # already in cms
                                elif 'm/s' in units or 'm s-1' in units:
                                    # Convert m/s to m3/s: m/s * area_m2
                                    area_km2 = self._get_catchment_area(config)
                                    simulated_streamflow = simulated_streamflow * area_km2 * 1e6
                                else:
                                    area_km2 = self._get_catchment_area(config)
                                    simulated_streamflow = simulated_streamflow * area_km2 / UnitConversion.MM_DAY_TO_CMS
                            
                            # Aligner
                            sim_aligned = simulated_streamflow.reindex(common_index).dropna()
                            obs_aligned = pd.Series(obs_values, index=common_index).reindex(sim_aligned.index)
                            
                            if len(sim_aligned) > 0:
                                metrics = {
                                    'kge': float(kge(obs_aligned.values, sim_aligned.values)),
                                    'nse': float(nse(obs_aligned.values, sim_aligned.values)),
                                    'rmse': float(rmse(obs_aligned.values, sim_aligned.values)),
                                    'mae': float(mae(obs_aligned.values, sim_aligned.values)),
                                }
                                return metrics

                # Fallback to GR output if routing not used or not found
                sim_file = output_dir / f"{domain_name}_{experiment_id}_runs_def.nc"
                if not sim_file.exists():
                    self.logger.error(f"GR distributed output not found: {sim_file}")
                    return {'kge': self.penalty_score}
                
                with xr.open_dataset(sim_file) as ds:
                    # Determine which variable to use (respect config)
                    routing_var = config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
                    if routing_var in ('default', None, ''):
                        routing_var = 'q_routed'
                        
                    if routing_var in ds.variables:
                        var = ds[routing_var]
                        # Consistent with evaluator: use mean across GRUs for depth-based runoff
                        # instead of just the last GRU
                        sim_data = var.mean(dim='gru').to_pandas()
                        
                        units = var.attrs.get('units', '').lower()
                        area_km2 = self._get_catchment_area(config)
                        
                        if 'm/s' in units or 'm s-1' in units:
                            simulated_streamflow = sim_data * area_km2 * 1e6
                        else:
                            simulated_streamflow = sim_data * area_km2 / UnitConversion.MM_DAY_TO_CMS
                    else:
                        self.logger.error(f"Variable '{routing_var}' not found in GR output. Vars: {list(ds.variables)}")
                        return {'kge': self.penalty_score}

            # Align and calculate for non-routed cases
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
            if obs_file == 'default' or not obs_file:
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
            
            # Robust path resolution handling 'default' values
            c_path = config.get('CATCHMENT_PATH', 'default')
            if c_path == 'default' or not c_path:
                catchment_path = project_dir / 'shapefiles' / 'catchment'
            else:
                catchment_path = Path(c_path)
                
            discretization = config.get('DOMAIN_DISCRETIZATION', 'elevation')
            c_name = config.get('CATCHMENT_SHP_NAME', 'default')
            if c_name == 'default' or not c_name:
                catchment_name = f"{domain_name}_HRUs_{discretization}.shp"
            else:
                catchment_name = c_name
            
            catchment_file = catchment_path / catchment_name
            if not catchment_file.exists():
                self.logger.warning(f"Catchment file not found: {catchment_file}. Using default area 1000 km2.")
                return 1000.0 # Default fallback
                
            gdf = gpd.read_file(catchment_file)
            if 'GRU_area' in gdf.columns:
                area_km2 = gdf['GRU_area'].sum() / 1e6
                self.logger.debug(f"Catchment area from GRU_area: {area_km2:.2f} km2")
                return area_km2
            
            if gdf.crs and not gdf.crs.is_geographic:
                area = gdf.geometry.area.sum()
            else:
                area = gdf.to_crs(gdf.estimate_utm_crs()).geometry.area.sum()
            
            area_km2 = area / 1e6
            self.logger.debug(f"Catchment area calculated from geometry: {area_km2:.2f} km2")
            return area_km2
        except Exception as e:
            self.logger.warning(f"Error calculating area: {e}. Using default 1000 km2.")
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
