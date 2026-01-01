import os
import sys
import time
import subprocess
from shutil import rmtree, copyfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime
import rasterio # type: ignore
from scipy import ndimage
import csv
import itertools
import matplotlib.pyplot as plt # type: ignore
import xarray as xr # type: ignore
from typing import Dict, List, Tuple, Any
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import PETCalculatorMixin

sys.path.append(str(Path(__file__).resolve().parent.parent))
from symfluence.utils.evaluation.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE # type: ignore
from symfluence.utils.data.utilities.variable_utils import VariableHandler # type: ignore


@ModelRegistry.register_postprocessor('FUSE')
class FUSEPostprocessor:
    """
    Postprocessor for FUSE (Framework for Understanding Structural Errors) model outputs.
    Handles extraction, processing, and saving of simulation results.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from FUSE output and save to CSV.
        Converts units from mm/day to m3/s (cms) using catchment area.
        
        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Extracting FUSE streamflow results")
            
            # Define paths
            sim_path = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FUSE' / f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}_runs_best.nc"
            
            # Read simulation results
            ds = xr.open_dataset(sim_path)
            
            # Extract streamflow (selecting first parameter set and first grid cell)
            q_sim = ds['q_routed'].isel(
                param_set=0,
                latitude=0,
                longitude=0
            )
            
            # Convert to pandas Series
            q_sim = q_sim.to_pandas()
            
            # Get catchment area from river basins shapefile
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            # Calculate total area in km2
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            self.logger.info(f"Total catchment area: {area_km2:.2f} km2")
            
            # Convert units from mm/day to m3/s (cms)
            # Q(cms) = Q(mm/day) * Area(km2) / 86.4
            q_sim_cms = q_sim * area_km2 / 86.4
            
            # Create DataFrame with FUSE-prefixed column name
            results_df = pd.DataFrame({
                'FUSE_discharge_cms': q_sim_cms
            }, index=q_sim.index)
            
            # Add metadata as attributes
            results_df.attrs = {
                'model': 'FUSE',
                'domain': self.domain_name,
                'experiment_id': self.config.get('EXPERIMENT_ID'),
                'catchment_area_km2': area_km2,
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'units': 'm3/s'
            }
            
            # Save to CSV
            output_file = self.results_dir / f"{self.config.get('EXPERIMENT_ID')}_results.csv"
            results_df.to_csv(output_file)
            
            self.logger.info(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            raise
            
    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Helper method to get file paths from config or defaults."""
        if self.config.get(file_type) == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(file_type))

