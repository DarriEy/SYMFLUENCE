"""
Base Dataset Handler for SYMFLUENCE
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xarray as xr
import geopandas as gpd


class BaseDatasetHandler(ABC):
    """
    Abstract base class for dataset-specific handlers.
    """
    
    def __init__(self, config: Dict, logger, project_dir: Path, **kwargs):
        """
        Initialize the dataset handler.
        """
        self.config = config
        self.logger = logger
        self.project_dir = project_dir
        self.domain_name = config['DOMAIN_NAME']
        # Store extra kwargs like forcing_timestep_seconds if provided
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def get_variable_mapping(self) -> Dict[str, str]: pass
    
    @abstractmethod
    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset: pass
    
    @abstractmethod
    def get_coordinate_names(self) -> Tuple[str, str]: pass
    
    @abstractmethod
    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path, 
                        dem_path: Path, elevation_calculator) -> Path: pass
    
    @abstractmethod
    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None: pass
    
    @abstractmethod
    def needs_merging(self) -> bool: pass
    
    def get_file_pattern(self) -> str:
        return f"domain_{self.domain_name}_*.nc"
    
    def get_merged_file_pattern(self, year: int, month: int) -> str:
        dataset_name = self.__class__.__name__.replace('Handler', '').upper()
        return f"{dataset_name}_monthly_{year}{month:02d}.nc"
    
    def setup_time_encoding(self, ds: xr.Dataset) -> xr.Dataset:
        ds['time'].encoding['units'] = 'hours since 1900-01-01'
        ds['time'].encoding['calendar'] = 'gregorian'
        return ds
    
    def add_metadata(self, ds: xr.Dataset, description: str) -> xr.Dataset:
        import time
        ds.attrs.update({'History': f'Created {time.ctime(time.time())}', 'Reason': description})
        return ds
    
    def clean_variable_attributes(self, ds: xr.Dataset, missing_value: float = -999) -> xr.Dataset:
        for var in ds.data_vars:
            if 'missing_value' in ds[var].attrs: del ds[var].attrs['missing_value']
            if '_FillValue' in ds[var].attrs: del ds[var].attrs['_FillValue']
            ds[var].attrs['missing_value'] = missing_value
        return ds