"""
NGen Model Preprocessor.

Handles spatial preprocessing and configuration generation for the NOAA NextGen Framework.
Uses shared utilities for time window management and forcing data processing.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, List
from shutil import copyfile
import netCDF4 as nc4

from symfluence.utils.models.registry import ModelRegistry
from symfluence.utils.models.base import BaseModelPreProcessor
from symfluence.utils.models.mixins import ObservationLoaderMixin
from symfluence.utils.models.utilities import TimeWindowManager, ForcingDataProcessor
from symfluence.utils.models.ngen.config_generator import NgenConfigGenerator
from symfluence.utils.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)


@ModelRegistry.register_preprocessor('NGEN')
class NgenPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):
    """
    Preprocessor for NextGen Framework.

    Handles conversion of SYMFLUENCE data to ngen-compatible formats including:
    - Catchment geometry (geopackage)
    - Nexus points (GeoJSON)
    - Forcing data (NetCDF)
    - Model configurations (CFE, PET, NOAH-OWP)
    - Realization configuration (JSON)

    Inherits observation loading from ObservationLoaderMixin.
    """

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "NGEN"

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the NextGen preprocessor.

        Args:
            config: Configuration dictionary
            logger: Logger object
        """
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

        # NGen-specific configuration (Phase 3: typed config)
        if self.config:
            self.hru_id_col = self.config.paths.catchment_hruid
        else:
            self.hru_id_col = config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')

        self._ngen_lib_paths = self._resolve_ngen_lib_paths()
        self._available_modules = {
            name: path.exists() for name, path in self._ngen_lib_paths.items()
        }
        for name, available in self._available_modules.items():
            if not available:
                self.logger.warning(f"NGEN module library missing for {name}: {self._ngen_lib_paths[name]}")

        self._include_sloth = self._available_modules.get("SLOTH", False)
        self._include_pet = self._available_modules.get("PET", False)
        self._include_noah = self._available_modules.get("NOAH", False)
        self._include_cfe = self._available_modules.get("CFE", True)

    def _resolve_ngen_lib_paths(self) -> Dict[str, Path]:
        lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
        if self.config and self.config.model.ngen:
            install_path = self.config.model.ngen.install_path
        else:
            install_path = self.config_dict.get('NGEN_INSTALL_PATH', 'default')
        if install_path == 'default':
            ngen_root = self.data_dir.parent / 'installs' / 'ngen' / 'build'
        else:
            ngen_root = Path(install_path)

        return {
            "SLOTH": ngen_root / "extern" / "sloth" / "cmake_build" / f"libslothmodel{lib_ext}",
            "PET": ngen_root / "extern" / "evapotranspiration" / "evapotranspiration" / "cmake_build" / f"libpetbmi{lib_ext}",
            "NOAH": ngen_root / "extern" / "noah-owp-modular" / "cmake_build" / f"libsurfacebmi{lib_ext}",
            "CFE": ngen_root / "extern" / "cfe" / "cmake_build" / f"libcfebmi{lib_ext}",
        }
    
    def _copy_noah_parameter_tables(self):
        """
        Copy Noah-OWP parameter tables from package data to domain settings.

        Copies GENPARM.TBL, MPTABLE.TBL, and SOILPARM.TBL from:
            symfluence/data/base_settings/NOAH/parameters/
        To:
            domain_dir/settings/ngen/NOAH/parameters/
        """
        if not self._include_noah:
            self.logger.info("Skipping Noah-OWP parameter tables (NOAH module unavailable)")
            return

        self.logger.info("Copying Noah-OWP parameter tables")

        # Get source directory from package data
        from symfluence.utils.resources import get_base_settings_dir

        try:
            noah_base_dir = get_base_settings_dir('NOAH')
            source_param_dir = noah_base_dir / 'parameters'
        except FileNotFoundError:
            self.logger.warning("NOAH base settings not found in package; skipping parameter table copy")
            return

        # Destination directory
        dest_param_dir = self.setup_dir / 'NOAH' / 'parameters'
        
        # Parameter table files to copy
        param_files = ['GENPARM.TBL', 'MPTABLE.TBL', 'SOILPARM.TBL']
        
        for param_file in param_files:
            source_file = source_param_dir / param_file
            dest_file = dest_param_dir / param_file
            
            if source_file.exists():
                copyfile(source_file, dest_file)
                self.logger.info(f"Copied {param_file} to {dest_param_dir}")
            else:
                self.logger.warning(f"Parameter file not found: {source_file}")

    
    def run_preprocessing(self):
        """
        Execute complete ngen preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails.
        """
        self.logger.info("Starting NextGen preprocessing")
        return self.run_preprocessing_template()

    def create_directories(self, additional_dirs=None):
        """Override to add NGen-specific directories."""
        ngen_dirs = [
            self.setup_dir / "CFE",
            self.setup_dir / "PET",
            self.setup_dir / "NOAH",
            self.setup_dir / "NOAH" / "parameters",
            self.forcing_dir / "csv" # Ensure CSV forcing directory is created
        ]
        if additional_dirs:
            ngen_dirs.extend(additional_dirs)
        super().create_directories(additional_dirs=ngen_dirs)

    def copy_base_settings(self):
        """Override to copy Noah-OWP parameter tables."""
        self._copy_noah_parameter_tables()

    def _prepare_forcing(self) -> None:
        """NGEN-specific forcing data preparation (template hook)."""
        self._forcing_file = self.prepare_forcing_data()

    def _create_model_configs(self) -> None:
        """NGEN-specific configuration file creation (template hook)."""
        # Create spatial data files (moved from _pre_setup to ensure directories exist)
        self._nexus_file = self.create_nexus_geojson()
        self._catchment_file = self.create_catchment_geopackage()

        self.generate_model_configs()
        self.generate_realization_config(
            self._catchment_file,
            self._nexus_file,
            self._forcing_file
        )
        
    def create_nexus_geojson(self) -> Path:
        """
        Create nexus GeoJSON from river network topology.

        Nexus points represent junctions and outlets in the stream network.
        Each catchment flows to a nexus point.

        Returns:
            Path to nexus GeoJSON file
        """
        self.logger.info("Creating nexus GeoJSON")

        # Load river network
        river_network_file = self.get_river_network_path()
        if not river_network_file.exists():
            self.logger.warning(f"River network not found: {river_network_file}")
            # For lumped catchments, create a single outlet nexus
            return self._create_simple_nexus()
        
        river_gdf = gpd.read_file(river_network_file)
        
        # Get segment ID columns
        seg_id_col = self.config_dict.get('RIVER_NETWORK_SHP_SEGID', 'LINKNO')
        downstream_col = self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID', 'DSLINKNO')
        
        # Create nexus points at segment endpoints
        nexus_features = []
        
        for idx, row in river_gdf.iterrows():
            seg_id = row[seg_id_col]
            downstream_id = row[downstream_col]
            
            # Get endpoint of segment
            geom = row.geometry
            if geom.geom_type == 'LineString':
                endpoint = geom.coords[-1]  # Last point
            else:
                # For Point geometries (lumped case)
                endpoint = (geom.x, geom.y)
            
            # Create nexus ID
            nexus_id = f"nex-{int(seg_id)}"
            
            if downstream_id == 0 or pd.isna(downstream_id):
                nexus_type = "poi"
                toid = ""                                  # terminal outlet
            else:
                nexus_type = "nexus"
                toid = f"wb-{int(downstream_id)}"          # nexus -> downstream catchment


            
            feature = {
                "type": "Feature",
                "id": nexus_id,
                "properties": {
                    "toid": toid,
                    "hl_id": None,
                    "hl_uri": "NA",
                    "type": nexus_type
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": list(endpoint)
                }
            }
            nexus_features.append(feature)
        
        # Create GeoJSON
        nexus_geojson = {
            "type": "FeatureCollection",
            "name": "nexus",
            "xy_coordinate_resolution": 1e-06,
            "features": nexus_features
        }
        
        # Save to file
        nexus_file = self.setup_dir / "nexus.geojson"
        with open(nexus_file, 'w') as f:
            json.dump(nexus_geojson, f, indent=2)

        self.logger.info(f"Created nexus file with {len(nexus_features)} nexus points: {nexus_file}")
        return nexus_file

    def _create_simple_nexus(self) -> Path:
        """Create a simple single-nexus for lumped catchments."""
        self.logger.info("Creating simple outlet nexus for lumped catchment")

        # Load catchment to get centroid
        catchment_file = self.get_catchment_path()
        catchment_gdf = gpd.read_file(catchment_file)
        
        # Get catchment centroid in projected CRS, then convert to WGS84
        catchment_utm = catchment_gdf.to_crs(catchment_gdf.estimate_utm_crs())
        centroid_utm = catchment_utm.geometry.centroid
        centroid = centroid_utm.to_crs("EPSG:4326").iloc[0]
        
        # Get catchment ID
        catchment_id = str(catchment_gdf[self.hru_id_col].iloc[0])
        
        nexus_geojson = {
            "type": "FeatureCollection",
            "name": "nexus",
            "xy_coordinate_resolution": 1e-06,
            "features": [{
                "type": "Feature",
                "id": f"nex-{catchment_id}",
                "properties": {
                    "toid": "",  # Terminal outlet - empty toid breaks the cycle
                    "hl_id": None,
                    "hl_uri": "NA",
                    "type": "poi"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [centroid.x, centroid.y]
                }
            }]
        }
        
        nexus_file = self.setup_dir / "nexus.geojson"
        with open(nexus_file, 'w') as f:
            json.dump(nexus_geojson, f, indent=2)

        self.logger.info(f"Created simple nexus file: {nexus_file}")
        return nexus_file

    def create_catchment_geopackage(self) -> Path:
        """
        Create ngen-compatible geopackage from SYMFLUENCE catchment shapefile.

        The geopackage must contain a 'divides' layer with required attributes:
        - divide_id: Catchment identifier
        - toid: ID of downstream catchment (or nexus)
        - areasqkm: Catchment area
        - geometry: Polygon geometry

        Returns:
            Path to catchment geopackage
        """
        self.logger.info("Creating catchment geopackage")

        # Load catchment shapefile
        catchment_file = self.get_catchment_path()
        catchment_gdf = gpd.read_file(catchment_file)
        
        # Create divides layer
        divides_gdf = catchment_gdf.copy()
        
        # Map to ngen schema
        divides_gdf['divide_id'] = divides_gdf[self.hru_id_col].apply(lambda x: f'cat-{x}')
        divides_gdf['id'] = divides_gdf[self.hru_id_col].apply(lambda x: f'wb-{x}')  # Waterbody ID
        
        # Determine downstream connections
        # For lumped catchment, connect to corresponding nexus
        divides_gdf['toid'] = divides_gdf[self.hru_id_col].apply(lambda x: f'nex-{x}')
        
        # Add type
        divides_gdf['type'] = 'network'  # Changed from 'land' to 'network'
        
        # Calculate area in km²
        if 'areasqkm' not in divides_gdf.columns:
            # Convert to equal-area projection for area calculation
            utm_crs = divides_gdf.estimate_utm_crs()
            divides_utm = divides_gdf.to_crs(utm_crs)
            divides_gdf['areasqkm'] = divides_utm.geometry.area / 1e6
        
        # Select required columns
        required_cols = ['divide_id', 'toid', 'type', 'id', 'areasqkm', 'geometry']
        optional_cols = ['ds_id', 'lengthkm', 'tot_drainage_areasqkm', 'has_flowline']
        
        # Add optional columns with defaults if missing
        for col in optional_cols:
            if col not in divides_gdf.columns:
                if col == 'ds_id':
                    divides_gdf[col] = 0.0
                elif col == 'lengthkm':
                    divides_gdf[col] = 0.0
                elif col == 'tot_drainage_areasqkm':
                    divides_gdf[col] = divides_gdf['areasqkm']
                elif col == 'has_flowline':
                    divides_gdf[col] = False
        
        output_cols = required_cols + [c for c in optional_cols if c in divides_gdf.columns]
        divides_gdf = divides_gdf[output_cols]
        
        # Ensure proper CRS (NAD83 Conus Albers - EPSG:5070)
        if divides_gdf.crs != "EPSG:5070":
            divides_gdf = divides_gdf.to_crs("EPSG:5070")
        
        # Remove the column since the index will carry 'id'
        divides_gdf = divides_gdf.drop(columns=['id'])
        
        # Set index to the divide_id for proper feature identification
        divides_gdf.index = divides_gdf['divide_id']
        divides_gdf.index.name = 'id'
        
        # Save as geopackage
        gpkg_file = self.setup_dir / f"{self.domain_name}_catchments.gpkg"
        divides_gdf.to_file(gpkg_file, layer='divides', driver='GPKG')

        geojson_file = self.setup_dir / f"{self.domain_name}_catchments.geojson"
        divides_gdf.to_file(geojson_file, driver='GeoJSON')

        self.logger.info(f"Created catchment geopackage with {len(divides_gdf)} catchments: {gpkg_file}")
        self.logger.info(f"Created catchment GeoJSON with {len(divides_gdf)} catchments: {geojson_file}")
        return gpkg_file

    def prepare_forcing_data(self) -> Path:
        """
        Convert SYMFLUENCE basin-averaged ERA5 forcing to ngen format.

        Uses shared utilities for loading and time management:
        - TimeWindowManager for simulation period handling
        - ForcingDataProcessor for loading and subsetting

        Processes:
        1. Load all monthly forcing files
        2. Merge across time
        3. Map variable names (ERA5 → ngen)
        4. Reorganize dimensions (hru, time) → (catchment-id, time)
        5. Add catchment IDs

        Returns:
            Path to ngen forcing NetCDF file
        """
        self.logger.info("Preparing forcing data for ngen")

        # Load catchment IDs - must match divide_id format in geopackage (cat-X)
        catchment_file = self.get_catchment_path()
        catchment_gdf = gpd.read_file(catchment_file)
        catchment_ids = [f"cat-{x}" for x in catchment_gdf[self.hru_id_col].astype(str).tolist()]
        n_catchments = len(catchment_ids)

        self.logger.info(f"Processing forcing for {n_catchments} catchments")

        # Use shared ForcingDataProcessor for loading
        fdp = ForcingDataProcessor(self.config, self.logger)
        forcing_data = fdp.load_forcing_data(self.forcing_basin_path)

        # Use shared TimeWindowManager for time handling
        twm = TimeWindowManager(self.config, self.logger)
        try:
            start_time, end_time = twm.get_simulation_times(
                forcing_path=self.forcing_basin_path,
                default_start_offset_days=0,
                default_end_offset_days=0
            )
        except ValueError:
            # Fallback to config values if TimeWindowManager fails
            sim_start = self.config_dict.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
            sim_end = self.config_dict.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')
            if sim_start == 'default':
                sim_start = '2000-01-01 00:00:00'
            if sim_end == 'default':
                sim_end = '2000-12-31 23:00:00'
            start_time = twm.parse_time_string(sim_start)
            end_time = twm.parse_time_string(sim_end)

        datasets: List[xr.Dataset] = []
        try:
            # Convert forcing time to datetime if needed
            time_values = pd.to_datetime(forcing_data.time.values)
            forcing_data['time'] = time_values

            # Use shared utility for subsetting
            forcing_data = fdp.subset_to_time_window(forcing_data, start_time, end_time)

            self.logger.info(f"Forcing time range: {forcing_data.time.values[0]} to {forcing_data.time.values[-1]}")

            datasets.append(forcing_data)

            # Create ngen-formatted dataset
            ngen_ds = self._create_ngen_forcing_dataset(forcing_data, catchment_ids)
            datasets.append(ngen_ds)

            # Save to file with NETCDF4 format (supports native string type)
            output_file = self.forcing_dir / "forcing.nc"
            ngen_ds.to_netcdf(output_file, format='NETCDF4')

            # Write per-catchment CSV forcings for CsvPerFeature provider
            self._write_csv_forcing_files(forcing_data, catchment_ids)

            self.logger.info(f"Created ngen forcing file: {output_file}")
            return output_file
        finally:
            for ds in datasets:
                try:
                    ds.close()
                except Exception:
                    pass

    def _write_csv_forcing_files(
        self,
        forcing_data: xr.Dataset,
        catchment_ids: List[str]
    ) -> Path:
        """
        Write per-catchment CSV forcing files compatible with CsvPerFeature.

        Returns:
            Path to CSV forcing directory.
        """
        csv_dir = self.forcing_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        time_values = pd.to_datetime(forcing_data.time.values)

        def _select_series(data: xr.DataArray, idx: int) -> np.ndarray:
            arr = data.values
            if arr.ndim == 1:
                return arr
            if arr.shape[0] == len(time_values):
                return arr[:, idx]
            return arr[idx, :]

        var_mapping = {
            'pptrate': 'precip_rate',
            'airtemp': 'TMP_2maboveground',
            'spechum': 'SPFH_2maboveground',
            'airpres': 'PRES_surface',
            'SWRadAtm': 'DSWRF_surface',
            'LWRadAtm': 'DLWRF_surface'
        }

        wind_data = forcing_data.get('windspd')

        for idx, catchment_id in enumerate(catchment_ids):
            columns = {
                'time': time_values,
            }

            for era5_var, ngen_var in var_mapping.items():
                if era5_var in forcing_data:
                    columns[ngen_var] = _select_series(forcing_data[era5_var], idx)

            if wind_data is not None:
                wind_series = _select_series(wind_data, idx)
                columns['UGRD_10maboveground'] = wind_series * 0.707
                columns['VGRD_10maboveground'] = wind_series * 0.707

            if 'precip_rate' in columns:
                columns['APCP_surface'] = columns['precip_rate']

            df = pd.DataFrame(columns)
            csv_path = csv_dir / f"{catchment_id}_forcing.csv"
            df.to_csv(csv_path, index=False)

        return csv_dir
        
    def _create_ngen_forcing_dataset(self, forcing_data: xr.Dataset, catchment_ids: List[str]) -> xr.Dataset:
        """
        Create ngen-formatted forcing dataset with proper variable mapping.
        
        Args:
            forcing_data: Source forcing dataset (ERA5)
            catchment_ids: List of catchment identifiers
            
        Returns:
            ngen-formatted xarray Dataset
        """
        n_catchments = len(catchment_ids)
        n_times = len(forcing_data.time)
        
        # Convert time to nanoseconds since epoch (ngen format - matching working example)
        time_values = forcing_data.time.values
        
        # Ensure we have datetime64 objects
        if not np.issubdtype(time_values.dtype, np.datetime64):
            # Try to decode using xarray's built-in time decoding
            if 'units' in forcing_data.time.attrs:
                time_values = xr.decode_cf(forcing_data).time.values
            else:
                # Fallback: assume hours since 1900-01-01 (common ERA5 format)
                time_values = pd.to_datetime(time_values, unit='h', origin='1900-01-01').values
        
        # Verify we have datetime64 now
        if not np.issubdtype(time_values.dtype, np.datetime64):
            raise ValueError(f"Could not convert time to datetime64, got dtype: {time_values.dtype}")
        
        # Convert to nanoseconds since 1970-01-01 (Unix epoch)
        time_ns = ((time_values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'ns')).astype(np.int64)
        
        # Sanity check - values should be positive and reasonable (between 1970 and 2100)
        min_ns = 0  # 1970-01-01
        max_ns = 4102444800 * 1e9  # 2100-01-01 in nanoseconds
        if np.any(time_ns < min_ns) or np.any(time_ns > max_ns):
            raise ValueError(f"Time values out of reasonable range. Got min={time_ns.min()}, max={time_ns.max()}. "
                        f"Expected between {min_ns} and {max_ns}")
        
        # Match ngen NetCDF forcing conventions (no coordinate values for dims)
        ngen_ds = xr.Dataset()

        # Add catchment IDs as native NetCDF4 string type
        ngen_ds['ids'] = xr.DataArray(
            np.array(catchment_ids, dtype=object),
            dims=['catchment-id'],
            attrs={'long_name': 'catchment identifiers'}
        )
        
        # Add Time variable (capital T) with nanoseconds for each catchment-time pair
        # Replicate time values for each catchment
        time_data = np.tile(time_ns, (n_catchments, 1)).astype(np.float64)
        ngen_ds['Time'] = xr.DataArray(
            time_data,
            dims=['catchment-id', 'time'],
            attrs={'units': 'ns'}
        )
        
        # Map and add forcing variables
        # ERA5 → ngen variable mapping
        var_mapping = {
            'pptrate': 'precip_rate',
            'airtemp': 'TMP_2maboveground',
            'spechum': 'SPFH_2maboveground',
            'airpres': 'PRES_surface',
            'SWRadAtm': 'DSWRF_surface',
            'LWRadAtm': 'DLWRF_surface'
        }
        
        # Add variables as float32 to match ngen requirements
        for era5_var, ngen_var in var_mapping.items():
            if era5_var in forcing_data:
                # Get data and transpose to (catchment, time)
                data = forcing_data[era5_var].values.T  # (time, hru) → (hru, time)
                
                # Replicate for multiple catchments if needed
                if data.shape[0] == 1 and n_catchments > 1:
                    data = np.tile(data, (n_catchments, 1))
                
                ngen_ds[ngen_var] = xr.DataArray(
                    data.astype(np.float32),  # Ensure float32
                    dims=['catchment-id', 'time']
                )
        
        # Handle wind components
        # ERA5 provides windspd; ngen needs UGRD and VGRD
        # Approximate: assume wind from west, so UGRD = windspd, VGRD = 0
        if 'windspd' in forcing_data:
            windspd_data = forcing_data['windspd'].values.T
            
            if windspd_data.shape[0] == 1 and n_catchments > 1:
                windspd_data = np.tile(windspd_data, (n_catchments, 1))
            
            # Approximate split (could be improved with actual U/V components)
            ngen_ds['UGRD_10maboveground'] = xr.DataArray(
                (windspd_data * 0.707).astype(np.float32),  # Ensure float32
                dims=['catchment-id', 'time']
            )
            ngen_ds['VGRD_10maboveground'] = xr.DataArray(
                (windspd_data * 0.707).astype(np.float32),  # Ensure float32
                dims=['catchment-id', 'time']
            )

        # Optional APCP_surface to align with ngen sample forcings
        if 'precip_rate' in ngen_ds:
            ngen_ds['APCP_surface'] = ngen_ds['precip_rate']
        
        return ngen_ds
    
    def generate_model_configs(self):
        """
        Generate model-specific configuration files for each catchment.

        Uses NgenConfigGenerator for modular config creation.
        Creates: CFE, PET, and NOAH-OWP configs.
        """
        # Load catchment data
        catchment_file = self.get_catchment_path()
        catchment_gdf = gpd.read_file(catchment_file)
        self.catchment_crs = catchment_gdf.crs

        # Use config generator
        config_gen = NgenConfigGenerator(self.config_dict, self.logger, self.setup_dir, self.catchment_crs)
        config_gen.set_module_availability(
            cfe=self._include_cfe,
            pet=self._include_pet,
            noah=self._include_noah,
            sloth=self._include_sloth
        )
        config_gen.generate_all_configs(catchment_gdf, self.hru_id_col)
    
    def _generate_cfe_config(self, catchment_id: str, catchment_row: gpd.GeoSeries):
        """Generate CFE model configuration file."""
        
        # Get catchment-specific parameters (or use defaults)
        # In a full implementation, these would come from soil/vegetation data
        config_text = f"""forcing_file=BMI
surface_partitioning_scheme=Schaake
soil_params.depth=2.0[m]
soil_params.b=5.0[]
soil_params.satdk=5.0e-06[m s-1]
soil_params.satpsi=0.141[m]
soil_params.slop=0.03[m/m]
soil_params.smcmax=0.439[m/m]
soil_params.wltsmc=0.047[m/m]
soil_params.expon=1.0[]
soil_params.expon_secondary=1.0[]
refkdt=1.0
max_gw_storage=0.0129[m]
Cgw=1.8e-05[m h-1]
expon=7.0[]
gw_storage=0.35[m/m]
alpha_fc=0.33
soil_storage=0.35[m/m]
K_nash=0.03[]
K_lf=0.01[]
nash_storage=0.0,0.0
num_timesteps=1
verbosity=1
DEBUG=0
giuh_ordinates=0.65,0.35
"""
        
        config_file = self.setup_dir / "CFE" / f"cat-{catchment_id}_bmi_config_cfe_pass.txt"
        with open(config_file, 'w') as f:
            f.write(config_text)
    
    def _generate_pet_config(self, catchment_id: str, catchment_row: gpd.GeoSeries):
        """Generate PET model configuration file."""
        
        # Get catchment centroid for lat/lon
        centroid = catchment_row.geometry.centroid
        
        # Convert to WGS84 if needed
        if self.catchment_crs != "EPSG:4326":
            geom_wgs84 = gpd.GeoSeries([catchment_row.geometry], crs=self.catchment_crs)
            geom_wgs84 = geom_wgs84.to_crs("EPSG:4326")
            centroid = geom_wgs84.iloc[0].centroid
        
        config_text = f"""forcing_file=BMI
wind_speed_measurement_height_m=10.0
humidity_measurement_height_m=2.0
vegetation_height_m=0.12
zero_plane_displacement_height_m=0.0003
momentum_transfer_roughness_length_m=0.0
heat_transfer_roughness_length_m=0.0
surface_longwave_emissivity=1.0
surface_shortwave_albedo=0.23
latitude_degrees={centroid.y}
longitude_degrees={centroid.x}
site_elevation_m=100.0
time_step_size_s=3600
num_timesteps=1
"""
        
        config_file = self.setup_dir / "PET" / f"cat-{catchment_id}_pet_config.txt"
        with open(config_file, 'w') as f:
            f.write(config_text)
    
    def _generate_noah_config(self, catchment_id: str, catchment_row: gpd.GeoSeries):
        """
        Generate NOAH-OWP model configuration file (.input file).
        
        Creates a Fortran namelist file with all required sections:
        - timing: simulation period and file paths
        - parameters: paths to parameter tables
        - location: catchment lat/lon and terrain
        - forcing: forcing data settings
        - model_options: Noah-OWP model configuration
        - structure: soil/snow/vegetation structure
        - initial_values: initial soil moisture, snow, water table
        """
        # Get catchment centroid for lat/lon
        centroid = catchment_row.geometry.centroid
        
        # Convert to WGS84 if needed
        if self.catchment_crs != "EPSG:4326":
            geom_wgs84 = gpd.GeoSeries([catchment_row.geometry], crs=self.catchment_crs)
            geom_wgs84 = geom_wgs84.to_crs("EPSG:4326")
            centroid = geom_wgs84.iloc[0].centroid
        
        # Get simulation timing from config
        start_time = self.config_dict.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        end_time = self.config_dict.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')
        
        # Convert to Noah-OWP format (YYYYMMDDhhmm)
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        start_str = start_dt.strftime('%Y%m%d%H%M')
        end_str = end_dt.strftime('%Y%m%d%H%M')
        
        # Absolute path to parameters directory
        # Noah-OWP requires absolute path since ngen runs from its build directory
        param_dir = str((self.setup_dir / "NOAH" / "parameters").resolve()) + "/"
        
        # Create Noah-OWP configuration file
        config_text = f"""&timing
  dt                 = 3600.0
  startdate          = "{start_str}"
  enddate            = "{end_str}"
  forcing_filename   = "BMI"
  output_filename    = "out_cat-{catchment_id}.csv"
/

&parameters
  parameter_dir      = "{param_dir}"
  general_table      = "GENPARM.TBL"
  soil_table         = "SOILPARM.TBL"
  noahowp_table      = "MPTABLE.TBL"
  soil_class_name    = "STAS"
  veg_class_name     = "MODIFIED_IGBP_MODIS_NOAH"
/

&location
  lat                = {centroid.y}
  lon                = {centroid.x}
  terrain_slope      = 0.0
  azimuth            = 0.0
/

&forcing
  ZREF               = 10.0
  rain_snow_thresh   = 1.0
/

&model_options
  precip_phase_option               = 1
  snow_albedo_option                = 1
  dynamic_veg_option                = 4
  runoff_option                     = 3
  drainage_option                   = 8
  frozen_soil_option                = 1
  dynamic_vic_option                = 1
  radiative_transfer_option         = 3
  sfc_drag_coeff_option             = 1
  canopy_stom_resist_option         = 1
  crop_model_option                 = 0
  snowsoil_temp_time_option         = 3
  soil_temp_boundary_option         = 2
  supercooled_water_option          = 1
  stomatal_resistance_option        = 1
  evap_srfc_resistance_option       = 4
  subsurface_option                 = 2
/

&structure
 isltyp           = 3
 nsoil            = 4
 nsnow            = 3
 nveg             = 20
 vegtyp           = 10
 croptype         = 0
 sfctyp           = 1
 soilcolor        = 4
/

&initial_values
 dzsnso    =  0.0,  0.0,  0.0,  0.1,  0.3,  0.6,  1.0
 sice      =  0.0,  0.0,  0.0,  0.0
 sh2o      =  0.3,  0.3,  0.3,  0.3
 zwt       =  -2.0
/
"""
        
        config_file = self.setup_dir / "NOAH" / f"cat-{catchment_id}.input"
        with open(config_file, 'w') as f:
            f.write(config_text)

    
    def generate_realization_config(self, catchment_file: Path, nexus_file: Path, forcing_file: Path):
        """
        Generate ngen realization configuration JSON.

        Uses NgenConfigGenerator for modular config creation.
        """
        # Use config generator
        config_gen = NgenConfigGenerator(
            self.config_dict, self.logger, self.setup_dir,
            getattr(self, 'catchment_crs', None)
        )
        config_gen.set_module_availability(
            cfe=self._include_cfe,
            pet=self._include_pet,
            noah=self._include_noah,
            sloth=self._include_sloth
        )
        config_gen.generate_realization_config(forcing_file, self.project_dir)
