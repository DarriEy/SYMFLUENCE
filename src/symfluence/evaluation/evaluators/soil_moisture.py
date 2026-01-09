#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Soil Moisture Evaluator
"""

import logging
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional

from symfluence.evaluation.registry import EvaluationRegistry
from .base import ModelEvaluator

@EvaluationRegistry.register('SOIL_MOISTURE')
class SoilMoistureEvaluator(ModelEvaluator):
    """Soil moisture evaluator"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        self.optimization_target = config.get('OPTIMIZATION_TARGET', 'streamflow')
        if self.optimization_target not in ['sm_point', 'sm_smap', 'sm_esa', 'sm_ismn']:
             if any(x in config.get('EVALUATION_VARIABLE', '') for x in ['sm_', 'soil']):
                self.optimization_target = config.get('EVALUATION_VARIABLE')
        
        self.variable_name = self.optimization_target
        
        if self.optimization_target == 'sm_point':
            self.target_depth = config.get('SM_TARGET_DEPTH', 'auto')
            self.depth_tolerance = config.get('SM_DEPTH_TOLERANCE', 0.05)
        elif self.optimization_target == 'sm_smap':
            self.smap_layer = config.get('SMAP_LAYER', 'surface_sm')
            self.temporal_aggregation = config.get('SM_TEMPORAL_AGGREGATION', 'daily_mean')
        elif self.optimization_target == 'sm_ismn':
            self.target_depth = config.get('ISMN_TARGET_DEPTH_M', config.get('SM_TARGET_DEPTH', 'auto'))
            self.temporal_aggregation = config.get('ISMN_TEMPORAL_AGGREGATION', 'daily_mean')
        elif self.optimization_target == 'sm_esa':
            self.temporal_aggregation = config.get('SM_TEMPORAL_AGGREGATION', 'daily_mean')
        
        self.use_quality_control = config.get('SM_USE_QUALITY_CONTROL', True)
        self.min_valid_pixels = config.get('SM_MIN_VALID_PIXELS', 10)
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        sim_file = sim_files[0]
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'sm_point':
                    return self._extract_point_soil_moisture(ds)
                elif self.optimization_target == 'sm_smap':
                    return self._extract_smap_soil_moisture(ds)
                elif self.optimization_target == 'sm_ismn':
                    return self._extract_point_soil_moisture(ds)
                elif self.optimization_target == 'sm_esa':
                    return self._extract_esa_soil_moisture(ds)
                else:
                    return self._extract_point_soil_moisture(ds)
        except Exception as e:
            self.logger.error(f"Error extracting soil moisture data from {sim_file}: {str(e)}")
            raise
    
    def _extract_point_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        if 'mLayerVolFracLiq' not in ds.variables:
            raise ValueError("mLayerVolFracLiq variable not found")
        soil_moisture_var = ds['mLayerVolFracLiq']
        layer_depths = ds['mLayerDepth']
        
        # Collapse spatial dimensions but preserve layer dimension
        sim_xr = soil_moisture_var
        depths_xr = layer_depths
        
        for dim in ['hru', 'gru']:
            if dim in sim_xr.dims:
                if sim_xr.sizes[dim] == 1:
                    sim_xr = sim_xr.isel({dim: 0})
                else:
                    sim_xr = sim_xr.mean(dim=dim)
            if dim in depths_xr.dims:
                if depths_xr.sizes[dim] == 1:
                    depths_xr = depths_xr.isel({dim: 0})
                else:
                    depths_xr = depths_xr.mean(dim=dim)
        
        target_layer_idx = self._find_target_layer(depths_xr)
        layer_dim = [dim for dim in sim_xr.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
        
        # Select target layer and ensure no other spatial dims remain
        sim_xr = sim_xr.isel({layer_dim: target_layer_idx})
        non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
        if non_time_dims:
            sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})
            
        sim_data = sim_xr.to_pandas()
        return sim_data
    
    def _find_target_layer(self, layer_depths: xr.DataArray) -> int:
        try:
            if self.target_depth == 'auto':
                return 0
            try:
                target_depth_m = float(self.target_depth)
            except (ValueError, TypeError):
                return 0
            
            if 'time' in layer_depths.dims:
                depths_sample = layer_depths.isel(time=0).values
            else:
                depths_sample = layer_depths.values
            
            cumulative_depths = np.cumsum(depths_sample) - depths_sample / 2
            depth_differences = np.abs(cumulative_depths - target_depth_m)
            best_layer_idx = np.argmin(depth_differences)
            return int(best_layer_idx)
        except Exception:
            return 0
    
    def _extract_smap_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        soil_moisture_var = ds['mLayerVolFracLiq']
        layer_depths = ds['mLayerDepth'] if 'mLayerDepth' in ds.variables else None

        # Collapse spatial dimensions
        sim_xr = soil_moisture_var
        for dim in ['hru', 'gru']:
            if dim in sim_xr.dims:
                if sim_xr.sizes[dim] == 1:
                    sim_xr = sim_xr.isel({dim: 0})
                else:
                    sim_xr = sim_xr.mean(dim=dim)

        layer_dims = [dim for dim in sim_xr.dims if 'mid' in dim.lower() or 'layer' in dim.lower()]
        if not layer_dims:
            raise ValueError("Layer dimension not found in simulated soil moisture output")
        layer_dim = layer_dims[0]

        if self.smap_layer == 'surface_sm':
            surface_depth = float(self.config.get('SMAP_SURFACE_DEPTH_M', 0.05))
            if layer_depths is None:
                sim_xr = sim_xr.isel({layer_dim: 0})
            else:
                sim_xr = self._depth_weighted_mean(sim_xr, layer_depths, surface_depth, layer_dim)
        elif self.smap_layer == 'rootzone_sm':
            rootzone_depth = float(self.config.get('SMAP_ROOTZONE_DEPTH_M', 1.0))
            if layer_depths is None:
                sim_xr = sim_xr.isel({layer_dim: slice(0, 3)}).mean(dim=layer_dim)
            else:
                sim_xr = self._depth_weighted_mean(sim_xr, layer_depths, rootzone_depth, layer_dim)
        else:
            raise ValueError(f"Unknown SMAP layer: {self.smap_layer}")
            
        # Ensure no other spatial dims remain
        non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
        if non_time_dims:
            sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})
            
        sim_data = sim_xr.to_pandas()
        return sim_data

    def _depth_weighted_mean(
        self,
        sim_xr: xr.DataArray,
        layer_depths: xr.DataArray,
        target_depth_m: float,
        layer_dim: str,
    ) -> xr.DataArray:
        depth_xr = layer_depths
        for dim in ['hru', 'gru']:
            if dim in depth_xr.dims:
                if depth_xr.sizes[dim] == 1:
                    depth_xr = depth_xr.isel({dim: 0})
                else:
                    depth_xr = depth_xr.mean(dim=dim)
        if 'time' in depth_xr.dims:
            depth_xr = depth_xr.isel(time=0)
        other_dims = [dim for dim in depth_xr.dims if dim != layer_dim]
        if other_dims:
            depth_xr = depth_xr.isel({dim: 0 for dim in other_dims})

        depth_vals = np.asarray(depth_xr.values).astype(float).ravel()
        n_layers = sim_xr.sizes[layer_dim]
        if depth_vals.size < n_layers:
            depth_vals = np.pad(depth_vals, (0, n_layers - depth_vals.size), constant_values=0.0)
        depth_vals = depth_vals[:n_layers]

        weights = np.zeros(n_layers, dtype=float)
        remaining = target_depth_m
        for i, thickness in enumerate(depth_vals):
            if remaining <= 0:
                break
            thickness = float(thickness) if np.isfinite(thickness) else 0.0
            if thickness <= 0:
                continue
            use = min(thickness, remaining)
            weights[i] = use
            remaining -= use

        if weights.sum() <= 0:
            return sim_xr.isel({layer_dim: 0})

        weights /= weights.sum()
        weight_da = xr.DataArray(weights, dims=[layer_dim], coords={layer_dim: sim_xr[layer_dim]})
        return (sim_xr * weight_da).sum(dim=layer_dim)
    
    def _extract_esa_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        soil_moisture_var = ds['mLayerVolFracLiq']
        
        # Collapse spatial dimensions
        sim_xr = soil_moisture_var
        for dim in ['hru', 'gru']:
            if dim in sim_xr.dims:
                if sim_xr.sizes[dim] == 1:
                    sim_xr = sim_xr.isel({dim: 0})
                else:
                    sim_xr = sim_xr.mean(dim=dim)
        
        layer_dim = [dim for dim in sim_xr.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
        sim_xr = sim_xr.isel({layer_dim: 0})
        
        # Ensure no other spatial dims remain
        non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
        if non_time_dims:
            sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})
            
        sim_data = sim_xr.to_pandas()
        return sim_data
    
    def get_observed_data_path(self) -> Path:
        if self.optimization_target == 'sm_point':
            return self.project_dir / "observations" / "soil_moisture" / "point" / "processed" / f"{self.domain_name}_sm_processed.csv"
        elif self.optimization_target == 'sm_smap':
            return self.project_dir / "observations" / "soil_moisture" / "smap" / "processed" / f"{self.domain_name}_smap_processed.csv"
        elif self.optimization_target == 'sm_ismn':
            return self.project_dir / "observations" / "soil_moisture" / "ismn" / "processed" / f"{self.domain_name}_ismn_processed.csv"
        elif self.optimization_target == 'sm_esa':
            return self.project_dir / "observations" / "soil_moisture" / "esa_sm" / "processed" / f"{self.domain_name}_esa_processed.csv"
        else:
            # Fallback path if target not perfectly set
             return self.project_dir / "observations" / "soil_moisture" / "processed" / f"{self.domain_name}_sm_processed.csv"

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'sm_point':
            if self.target_depth == 'auto':
                depth_columns = [col for col in columns if col.startswith('sm_')]
                if depth_columns:
                    depths = []
                    for col in depth_columns:
                        try:
                            depth_str = col.split('_')[1]
                            depths.append((float(depth_str), col))
                        except:
                            continue
                    if depths:
                        depths.sort()
                        self.target_depth = str(depths[0][0])
                        return depths[0][1]
            else:
                target_depth_str = str(self.target_depth)
                for col in columns:
                    if col.startswith('sm_') and target_depth_str in col:
                        return col
        elif self.optimization_target == 'sm_smap':
            if self.smap_layer in columns:
                return self.smap_layer
            for col in columns:
                if 'surface_sm' in col.lower() or 'rootzone_sm' in col.lower():
                    return col
        elif self.optimization_target == 'sm_ismn':
            target_depth_str = str(self.target_depth)
            for col in columns:
                if col.startswith('sm_') and target_depth_str in col:
                    return col
            for col in columns:
                if col.startswith('sm_'):
                    return col
        elif self.optimization_target == 'sm_esa':
            for col in columns:
                if any(term in col.lower() for term in ['esa', 'soil_moisture', 'sm']):
                    return col
        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                return None
            
            obs_df = pd.read_csv(obs_path)
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                return None
            
            if self.optimization_target == 'sm_esa':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], format='%d/%m/%Y', errors='coerce')
            else:
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            obs_series = pd.to_numeric(obs_df[data_col], errors='coerce')
            
            if self.optimization_target == 'sm_smap' and self.use_quality_control:
                if 'valid_px' in obs_df.columns:
                    valid_pixels = pd.to_numeric(obs_df['valid_px'], errors='coerce')
                    quality_mask = valid_pixels >= self.min_valid_pixels
                    obs_series = obs_series[quality_mask]
            
            obs_series = obs_series.dropna()
            
            if hasattr(self, 'temporal_aggregation') and self.temporal_aggregation == 'daily_mean':
                obs_series = obs_series.resample('D').mean().dropna()

            return obs_series
        except Exception as e:
            self.logger.error(f"Error loading observed soil moisture data: {str(e)}")
            return None

    def needs_routing(self) -> bool:
        return False
