"""
Base Dataset Handler for SYMFLUENCE

This module provides:
- BaseDatasetHandler: Abstract base class for dataset-specific handlers
- StandardVariableAttributes: Standard CF-compliant variable attribute definitions
- apply_standard_variable_attributes: Helper to apply standard attributes to datasets
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xarray as xr
import geopandas as gpd


# Standard variable attribute definitions (CF-compliant)
# These can be overridden by individual handlers if needed
STANDARD_VARIABLE_ATTRIBUTES: Dict[str, Dict[str, str]] = {
    'airpres': {
        'units': 'Pa',
        'long_name': 'air pressure',
        'standard_name': 'air_pressure',
    },
    'airtemp': {
        'units': 'K',
        'long_name': 'air temperature',
        'standard_name': 'air_temperature',
    },
    'pptrate': {
        'units': 'kg m-2 s-1',
        'long_name': 'precipitation rate',
        'standard_name': 'precipitation_flux',
    },
    'windspd': {
        'units': 'm s-1',
        'long_name': 'wind speed',
        'standard_name': 'wind_speed',
    },
    'LWRadAtm': {
        'units': 'W m-2',
        'long_name': 'downward longwave radiation at the surface',
        'standard_name': 'surface_downwelling_longwave_flux_in_air',
    },
    'SWRadAtm': {
        'units': 'W m-2',
        'long_name': 'downward shortwave radiation at the surface',
        'standard_name': 'surface_downwelling_shortwave_flux_in_air',
    },
    'spechum': {
        'units': 'kg kg-1',
        'long_name': 'specific humidity',
        'standard_name': 'specific_humidity',
    },
    'relhum': {
        'units': '%',
        'long_name': 'relative humidity',
        'standard_name': 'relative_humidity',
    },
    'windspd_u': {
        'units': 'm s-1',
        'long_name': 'eastward wind component',
        'standard_name': 'eastward_wind',
    },
    'windspd_v': {
        'units': 'm s-1',
        'long_name': 'northward wind component',
        'standard_name': 'northward_wind',
    },
}


def apply_standard_variable_attributes(
    ds: xr.Dataset,
    variables: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, str]]] = None
) -> xr.Dataset:
    """
    Apply standard CF-compliant attributes to dataset variables.

    This function centralizes the attribute-setting logic that was previously
    duplicated across all dataset handlers.

    Args:
        ds: xarray Dataset to modify
        variables: List of variable names to process. If None, processes all
                  variables that have standard definitions.
        overrides: Optional dict of {var_name: {attr: value}} to override defaults

    Returns:
        Modified dataset with standardized attributes

    Example:
        >>> ds = apply_standard_variable_attributes(ds)
        >>> ds = apply_standard_variable_attributes(ds, variables=['airtemp', 'pptrate'])
        >>> ds = apply_standard_variable_attributes(ds, overrides={'pptrate': {'units': 'mm/s'}})
    """
    # Merge overrides with defaults
    attrs_to_apply = STANDARD_VARIABLE_ATTRIBUTES.copy()
    if overrides:
        for var, var_overrides in overrides.items():
            if var in attrs_to_apply:
                attrs_to_apply[var] = {**attrs_to_apply[var], **var_overrides}
            else:
                attrs_to_apply[var] = var_overrides

    # Determine which variables to process
    if variables is None:
        variables = list(attrs_to_apply.keys())

    # Apply attributes to each variable present in the dataset
    for var_name in variables:
        if var_name in ds.data_vars and var_name in attrs_to_apply:
            ds[var_name].attrs.update(attrs_to_apply[var_name])

    return ds


class BaseDatasetHandler(ABC):
    """
    Abstract base class for dataset-specific forcing data handlers.

    Provides a standardized interface and common functionality for processing different
    meteorological datasets (ERA5, RDRS, CARRA, CONUS404, etc.) into SUMMA-compatible
    forcing files.

    Common Functionality:
        - Variable name standardization (dataset-specific → SUMMA standard)
        - CF-compliant attribute management
        - Time encoding standardization
        - NetCDF metadata handling
        - Missing value conventions
        - Coordinate name resolution

    Required Subclass Methods (Abstract):
        - get_variable_mapping(): Dataset-specific to standard variable names
        - process_dataset(): Apply dataset-specific transformations
        - get_coordinate_names(): Return (lat_name, lon_name) tuple
        - create_shapefile(): Generate forcing station shapefile
        - merge_forcings(): Merge/process raw files to standardized format
        - needs_merging(): Whether dataset requires file merging

    Optional Override Methods:
        - get_file_pattern(): Pattern for raw forcing files
        - get_merged_file_pattern(): Pattern for merged output files
        - setup_time_encoding(): Custom time encoding
        - add_metadata(): Custom metadata
        - clean_variable_attributes(): Custom attribute cleaning

    Attributes:
        config (Dict): Configuration dictionary
        logger: Logger instance
        project_dir (Path): Project root directory
        domain_name (str): Domain identifier

    Example Subclass:
        >>> @DatasetRegistry.register('my_dataset')
        >>> class MyDatasetHandler(BaseDatasetHandler):
        ...     def get_variable_mapping(self):
        ...         return {'t2m': 'airtemp', 'tp': 'pptrate'}
        ...     def process_dataset(self, ds):
        ...         # Apply conversions
        ...         return ds
        ...     # ... implement other abstract methods

    See Also:
        - ERA5Handler: ERA5 reanalysis data
        - RDRSHandler: Canadian Regional Deterministic Reanalysis System
        - CARRAHandler: Arctic regional reanalysis
    """

    def __init__(self, config: Dict, logger, project_dir: Path, **kwargs):
        """
        Initialize the dataset handler.

        Args:
            config: Configuration dictionary with domain and forcing settings
            logger: Logger instance for progress and error reporting
            project_dir: Path to project root directory
            **kwargs: Additional handler-specific parameters
                     (e.g., forcing_timestep_seconds)
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
        """
        Set up standard time encoding for NetCDF output.

        Applies CF-compliant time encoding using hours since 1900-01-01
        with Gregorian calendar.

        Args:
            ds: Dataset with time coordinate

        Returns:
            Dataset with time encoding configured

        Note:
            - Uses 'hours since 1900-01-01' for broad compatibility
            - Gregorian calendar is CF-compliant standard
            - This encoding is recognized by most hydrological models
        """
        ds['time'].encoding['units'] = 'hours since 1900-01-01'
        ds['time'].encoding['calendar'] = 'gregorian'
        return ds

    def add_metadata(self, ds: xr.Dataset, description: str) -> xr.Dataset:
        """
        Add standard metadata attributes to dataset.

        Args:
            ds: Dataset to annotate
            description: Description of processing/purpose

        Returns:
            Dataset with added metadata attributes

        Note:
            Adds 'History' with creation timestamp and 'Reason' with description.
        """
        import time
        ds.attrs.update({'History': f'Created {time.ctime(time.time())}', 'Reason': description})
        return ds

    def clean_variable_attributes(self, ds: xr.Dataset, missing_value: float = -999.0) -> xr.Dataset:
        """
        Standardize missing value handling across all variables.

        Removes missing value attributes from variable attrs dict and sets them
        in the encoding instead, following NetCDF best practices.

        Args:
            ds: Dataset to clean
            missing_value: Value to use for missing data (default: -999.0)

        Returns:
            Dataset with cleaned variable attributes

        Note:
            - Removes 'missing_value' and '_FillValue' from attrs
            - Sets both in encoding for each variable
            - Prevents attribute/encoding conflicts
            - Ensures consistent missing value handling across variables
        """
        for var in ds.data_vars:
            # Remove from attributes to avoid conflicts
            if 'missing_value' in ds[var].attrs: del ds[var].attrs['missing_value']
            if '_FillValue' in ds[var].attrs: del ds[var].attrs['_FillValue']

            # Set in encoding for consistent NetCDF output
            ds[var].encoding['missing_value'] = missing_value
            ds[var].encoding['_FillValue'] = missing_value
        return ds

    def apply_standard_attributes(
        self,
        ds: xr.Dataset,
        overrides: Optional[Dict[str, Dict[str, str]]] = None
    ) -> xr.Dataset:
        """
        Apply standard variable attributes to the dataset.

        Convenience method that wraps apply_standard_variable_attributes.
        Subclasses can override this to customize attribute handling.

        Args:
            ds: Dataset to modify
            overrides: Optional attribute overrides per variable

        Returns:
            Modified dataset
        """
        return apply_standard_variable_attributes(ds, overrides=overrides)