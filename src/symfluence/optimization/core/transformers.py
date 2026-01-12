"""
Parameter transformation adapters for model-specific calibration.

Provides transformers that apply calibration parameters to model-specific
formats (e.g., soil depth multipliers for SUMMA NetCDF files).
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ParameterTransformer:
    """Base class for parameter transformers."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def apply(self, params: Dict[str, Any], settings_dir: Path) -> bool:
        raise NotImplementedError

class SoilDepthTransformer(ParameterTransformer):
    """Handles transformation of soil depth parameters."""
    
    SPECIAL_PARAMS = ['total_soil_depth_multiplier', 'total_mult', 'shape_factor']

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.original_depths = None

    def apply(self, params: Dict[str, Any], settings_dir: Path) -> bool:
        # Check if any special parameters are present
        if not any(p in params for p in self.SPECIAL_PARAMS):
            return True
            
        try:
            coldstate_path = settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            if not coldstate_path.exists():
                self.logger.error(f"coldState.nc not found at {coldstate_path}")
                return False

            # Load original depths if not already loaded
            if self.original_depths is None:
                self.original_depths = self._get_original_depths(coldstate_path)
            
            if self.original_depths is None:
                return False

            # Extract multipliers
            total_mult = params.get('total_soil_depth_multiplier')
            if total_mult is None:
                total_mult = params.get('total_mult', 1.0)
            
            shape_factor = params.get('shape_factor', 1.0)
            
            if isinstance(total_mult, np.ndarray): total_mult = total_mult[0]
            if isinstance(shape_factor, np.ndarray): shape_factor = shape_factor[0]

            # Calculate new depths
            new_depths = self._calculate_new_depths(self.original_depths, total_mult, shape_factor)
            
            # Calculate layer heights (cumulative sum)
            heights = np.zeros(len(new_depths) + 1)
            for i in range(len(new_depths)):
                heights[i + 1] = heights[i] + new_depths[i]

            # Update NetCDF
            with nc.Dataset(coldstate_path, 'r+') as ds:
                if 'mLayerDepth' in ds.variables and 'iLayerHeight' in ds.variables:
                    num_hrus = ds.dimensions['hru'].size
                    for h in range(num_hrus):
                        ds.variables['mLayerDepth'][:, h] = new_depths
                        ds.variables['iLayerHeight'][:, h] = heights
                else:
                    self.logger.error("Required variables not found in coldState.nc")
                    return False
            
            return True

        except Exception as e:
            self.logger.error(f"Error in SoilDepthTransformer: {str(e)}")
            return False

    def _get_original_depths(self, path: Path) -> Optional[np.ndarray]:
        try:
            with nc.Dataset(path, 'r') as ds:
                return ds.variables['mLayerDepth'][:, 0].copy()
        except:
            return None

    def _calculate_new_depths(self, original: np.ndarray, total_mult: float, shape_factor: float) -> np.ndarray:
        n = len(original)
        idx = np.arange(n)
        
        # Calculate shape weights (exponential stretching)
        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)
        
        # Normalize weights so they average to 1.0
        w /= w.mean()
        
        # Apply multipliers
        return original * w * total_mult

class TransformationManager:
    """Orchestrates all parameter transformations."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.transformers = [
            SoilDepthTransformer(config, logger)
        ]

    def transform(self, params: Dict[str, Any], settings_dir: Path) -> bool:
        for transformer in self.transformers:
            if not transformer.apply(params, settings_dir):
                return False
        return True
