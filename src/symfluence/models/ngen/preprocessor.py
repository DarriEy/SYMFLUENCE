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

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.mixins import ObservationLoaderMixin
from symfluence.models.utilities import TimeWindowManager, ForcingDataProcessor
from symfluence.models.ngen.config_generator import NgenConfigGenerator
from symfluence.core.exceptions import (
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
        # Ensure we check for existence using the correctly resolved paths
        self._available_modules = {}
        for name, path in self._ngen_lib_paths.items():
            exists = path.exists()
            self._available_modules[name] = exists
            self.logger.info(f"Module {name} path: {path} (exists={exists})")
            if not exists:
                self.logger.warning(f"NGEN module library missing for {name}: {path}")

        # Determine which modules to include based on config and available libraries
        # Allow user to override which modules are enabled via config
        ngen_config = self.config_dict.get('NGEN', {})

        # SLOTH provides ice fraction and soil moisture variables for CFE
        self._include_sloth = ngen_config.get('ENABLE_SLOTH', self._available_modules.get("SLOTH", True))

        # PET provides evapotranspiration (but has known issues - can be disabled)
        # Note: PET requires wind speed variables (UGRD, VGRD) in forcing
        self._include_pet = ngen_config.get('ENABLE_PET', self._available_modules.get("PET", True))

        # NOAH-OWP provides alternative ET physics (more robust than PET)
        self._include_noah = ngen_config.get('ENABLE_NOAH', self._available_modules.get("NOAH", False))

        # CFE is the core runoff generation module
        self._include_cfe = ngen_config.get('ENABLE_CFE', self._available_modules.get("CFE", True))

        # Log module configuration
        self.logger.info(f"NGEN module configuration:")
        self.logger.info(f"  SLOTH: {'ENABLED' if self._include_sloth else 'DISABLED'}")
        self.logger.info(f"  PET: {'ENABLED' if self._include_pet else 'DISABLED'}")
        self.logger.info(f"  NOAH-OWP: {'ENABLED' if self._include_noah else 'DISABLED'}")
        self.logger.info(f"  CFE: {'ENABLED' if self._include_cfe else 'DISABLED'}")

    def _resolve_ngen_lib_paths(self) -> Dict[str, Path]:
        lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
        if self.config and self.config.model.ngen:
            install_path = self.config.model.ngen.install_path
        else:
            install_path = self.config_dict.get('NGEN_INSTALL_PATH', 'default')
        
        if install_path == 'default':
            ngen_base = self.data_dir.parent / 'installs' / 'ngen'
        else:
            p = Path(install_path)
            if p.name == 'cmake_build':
                ngen_base = p.parent
            else:
                ngen_base = p
        
        self.logger.info(f"Resolved NGEN_BASE to: {ngen_base}")

        # Check both ngen_base/extern and ngen_base/cmake_build/extern
        paths = {}
        module_subpaths = {
            "SLOTH": ("extern/sloth/cmake_build", f"libslothmodel{lib_ext}"),
            "PET": ("extern/evapotranspiration/evapotranspiration/cmake_build", f"libpetbmi{lib_ext}"),
            "NOAH": ("extern/noah-owp-modular/cmake_build", f"libsurfacebmi{lib_ext}"),
            "CFE": ("extern/cfe/cmake_build", f"libcfebmi{lib_ext}"),
        }

        for name, (subpath, libname) in module_subpaths.items():
            # Try direct extern
            p1 = ngen_base / subpath / libname
            # Try extern under cmake_build
            p2 = ngen_base / "cmake_build" / subpath / libname
            
            if p1.exists():
                paths[name] = p1
            elif p2.exists():
                paths[name] = p2
            else:
                # Fallback to p1 for consistent missing path reporting
                paths[name] = p1

        return paths
    
    def _copy_noah_parameter_tables(self):
        """
        Copy Noah-OWP parameter tables from package data to domain settings.
        """
        if not self._include_noah:
            return

        self.logger.info("Copying Noah-OWP parameter tables")
        from symfluence.resources import get_base_settings_dir

        try:
            noah_base_dir = get_base_settings_dir('NOAH')
            source_param_dir = noah_base_dir / 'parameters'
        except FileNotFoundError:
            self.logger.warning("NOAH base settings not found in package; skipping parameter table copy")
            return

        dest_param_dir = self.setup_dir / 'NOAH' / 'parameters'
        param_files = ['GENPARM.TBL', 'MPTABLE.TBL', 'SOILPARM.TBL']
        
        for param_file in param_files:
            source_file = source_param_dir / param_file
            dest_file = dest_param_dir / param_file
            if source_file.exists():
                copyfile(source_file, dest_file)

    def run_preprocessing(self):
        """Execute complete ngen preprocessing workflow."""
        self.logger.info("Starting NextGen preprocessing")
        return self.run_preprocessing_template()

    def create_directories(self, additional_dirs=None):
        """Override to add NGen-specific directories."""
        ngen_dirs = [
            self.setup_dir / "CFE",
            self.setup_dir / "PET",
            self.setup_dir / "NOAH",
            self.setup_dir / "NOAH" / "parameters",
            self.forcing_dir / "csv"
        ]
        if additional_dirs:
            ngen_dirs.extend(additional_dirs)
        super().create_directories(additional_dirs=ngen_dirs)

    def copy_base_settings(self):
        """Override to copy Noah-OWP parameter tables."""
        self._copy_noah_parameter_tables()

    def _prepare_forcing(self) -> None:
        """NGEN-specific forcing data preparation."""
        self._forcing_file = self.prepare_forcing_data()

    def _create_model_configs(self) -> None:
        """NGEN-specific configuration file creation."""
        self._nexus_file = self.create_nexus_geojson()
        self._catchment_file = self.create_catchment_geopackage()

        self.generate_model_configs()
        self.generate_realization_config(
            self._catchment_file,
            self._nexus_file,
            self._forcing_file
        )
        
    def create_nexus_geojson(self) -> Path:
        """Create nexus GeoJSON from river network topology."""
        self.logger.info("Creating nexus GeoJSON")
        river_network_file = self.get_river_network_path()
        if not river_network_file.exists():
            return self._create_simple_nexus()
        
        river_gdf = gpd.read_file(river_network_file)
        seg_id_col = self.config_dict.get('RIVER_NETWORK_SHP_SEGID', 'LINKNO')
        downstream_col = self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID', 'DSLINKNO')
        
        nexus_features = []
        for idx, row in river_gdf.iterrows():
            seg_id = row[seg_id_col]
            downstream_id = row[downstream_col]
            geom = row.geometry
            endpoint = geom.coords[-1] if geom.geom_type == 'LineString' else (geom.x, geom.y)
            nexus_id = f"nex-{int(seg_id)}"
            nexus_type = "poi" if (downstream_id == 0 or pd.isna(downstream_id)) else "nexus"
            toid = "" if nexus_type == "poi" else f"wb-{int(downstream_id)}"
            
            nexus_features.append({
                "type": "Feature", "id": nexus_id,
                "properties": {"toid": toid, "hl_id": None, "hl_uri": "NA", "type": nexus_type},
                "geometry": {"type": "Point", "coordinates": list(endpoint)}
            })
        
        nexus_file = self.setup_dir / "nexus.geojson"
        with open(nexus_file, 'w') as f:
            json.dump({"type": "FeatureCollection", "name": "nexus", "xy_coordinate_resolution": 1e-06, "features": nexus_features}, f, indent=2)
        return nexus_file

    def _create_simple_nexus(self) -> Path:
        """Create a simple single-nexus for lumped catchments."""
        catchment_file = self.get_catchment_path()
        catchment_gdf = gpd.read_file(catchment_file)
        catchment_utm = catchment_gdf.to_crs(catchment_gdf.estimate_utm_crs())
        centroid = catchment_utm.geometry.centroid.to_crs("EPSG:4326").iloc[0]
        catchment_id = str(catchment_gdf[self.hru_id_col].iloc[0])
        
        nexus_file = self.setup_dir / "nexus.geojson"
        with open(nexus_file, 'w') as f:
            json.dump({"type": "FeatureCollection", "name": "nexus", "xy_coordinate_resolution": 1e-06, "features": [{
                "type": "Feature", "id": f"nex-{catchment_id}",
                "properties": {"toid": "", "hl_id": None, "hl_uri": "NA", "type": "poi"},
                "geometry": {"type": "Point", "coordinates": [centroid.x, centroid.y]}
            }]}, f, indent=2)
        return nexus_file

    def create_catchment_geopackage(self) -> Path:
        """Create ngen-compatible geopackage and geojson."""
        from shapely.geometry import mapping

        catchment_file = self.get_catchment_path()
        catchment_gdf = gpd.read_file(catchment_file)
        divides_gdf = catchment_gdf.copy()
        divides_gdf['divide_id'] = divides_gdf[self.hru_id_col].apply(lambda x: f'cat-{x}')
        divides_gdf['toid'] = divides_gdf[self.hru_id_col].apply(lambda x: f'nex-{x}')
        divides_gdf['type'] = 'network'
        utm_crs = divides_gdf.estimate_utm_crs()
        divides_gdf['areasqkm'] = divides_gdf.to_crs(utm_crs).geometry.area / 1e6

        for col in ['ds_id', 'lengthkm', 'tot_drainage_areasqkm', 'has_flowline']:
            if col not in divides_gdf.columns:
                divides_gdf[col] = 0.0 if col != 'has_flowline' else False

        # Add 'id' column for NGEN compatibility
        divides_gdf['id'] = divides_gdf['divide_id']

        divides_gdf = divides_gdf[['id', 'divide_id', 'toid', 'type', 'areasqkm', 'geometry', 'ds_id', 'lengthkm', 'tot_drainage_areasqkm', 'has_flowline']]

        # For GPKG, use EPSG:5070 (Albers Equal Area)
        gpkg_gdf = divides_gdf.copy()
        if gpkg_gdf.crs != "EPSG:5070":
            gpkg_gdf = gpkg_gdf.to_crs("EPSG:5070")
        # Reset index to avoid duplicate 'id' column error when writing
        # The 'id' column is already present in the data for NGEN compatibility
        gpkg_gdf = gpkg_gdf.reset_index(drop=True)

        gpkg_file = self.setup_dir / f"{self.domain_name}_catchments.gpkg"
        gpkg_gdf.to_file(gpkg_file, layer='divides', driver='GPKG')

        # For GeoJSON, use EPSG:4326 (WGS84) and manually set feature-level id
        # NGEN requires feature-level id field which geopandas doesn't set automatically
        geojson_gdf = divides_gdf.to_crs("EPSG:4326")

        features = []
        for _, row in geojson_gdf.iterrows():
            # Build properties dict, handling NaN values
            props = {}
            for k, v in row.drop('geometry').to_dict().items():
                if isinstance(v, float) and np.isnan(v):
                    props[k] = None
                else:
                    props[k] = v

            feat = {
                'type': 'Feature',
                'id': row['divide_id'],  # Feature-level id required by NGEN
                'properties': props,
                'geometry': mapping(row.geometry)
            }
            features.append(feat)

        geojson_data = {
            'type': 'FeatureCollection',
            'name': f'{self.domain_name}_catchments',
            'features': features
        }

        geojson_file = self.setup_dir / f"{self.domain_name}_catchments.geojson"
        with open(geojson_file, 'w') as f:
            json.dump(geojson_data, f, indent=2)

        self.logger.info(f"Created NGEN catchment files: {gpkg_file.name}, {geojson_file.name}")
        return gpkg_file

    def prepare_forcing_data(self) -> Path:
        """Convert forcing data to ngen format (NetCDF and CSV)."""
        catchment_gdf = gpd.read_file(self.get_catchment_path())
        catchment_ids = [f"cat-{x}" for x in catchment_gdf[self.hru_id_col].astype(str).tolist()]
        fdp = ForcingDataProcessor(self.config, self.logger)
        forcing_data = fdp.load_forcing_data(self.forcing_basin_path)
        forcing_data = forcing_data.sortby('time')
        twm = TimeWindowManager(self.config, self.logger)
        try:
            start_time, end_time = twm.get_simulation_times(forcing_path=self.forcing_basin_path)
        except:
            start_time = pd.to_datetime(self.config_dict.get('EXPERIMENT_TIME_START'))
            end_time = pd.to_datetime(self.config_dict.get('EXPERIMENT_TIME_END'))

        time_values = pd.to_datetime(forcing_data.time.values)
        inferred_step_seconds = None
        if len(time_values) > 1:
            time_deltas = np.diff(time_values).astype('timedelta64[s]').astype(int)
            inferred_step_seconds = int(np.median(time_deltas))

        if inferred_step_seconds and inferred_step_seconds != 3600:
            self.logger.info(
                f"Resampling forcing data from {inferred_step_seconds} seconds to hourly for NGEN"
            )
            forcing_data = forcing_data.resample(time='1h').interpolate('linear')
            self.config_dict['FORCING_TIME_STEP_SIZE'] = 3600

        # NGEN requires forcing data beyond the configured end_time to complete the simulation
        # Extend end_time by 4 forcing timesteps to provide necessary lookahead data
        # (empirically determined - NGEN can access up to 3 timesteps beyond configured end_time)
        if inferred_step_seconds and inferred_step_seconds != 3600:
            forcing_timestep_seconds = 3600
        else:
            forcing_timestep_seconds = self.config_dict.get('FORCING_TIME_STEP_SIZE', 3600)
        buffer_timesteps = 4  # Add extra buffer to be safe
        extended_end_time = end_time + pd.Timedelta(seconds=forcing_timestep_seconds * buffer_timesteps)
        self.logger.info(f"Extending forcing data from {end_time} to {extended_end_time} (NGEN requires {buffer_timesteps} timestep buffer)")

        forcing_data = fdp.subset_to_time_window(forcing_data, start_time, extended_end_time)

        # Pad forcing data if source doesn't extend to full buffer
        actual_end = pd.to_datetime(forcing_data.time.values[-1])
        if actual_end < extended_end_time:
            self.logger.warning(f"Source forcing only available to {actual_end}, padding to {extended_end_time}")
            # Create additional timesteps by repeating last timestep values
            last_slice = forcing_data.isel(time=-1)
            padding_times = pd.date_range(
                start=actual_end + pd.Timedelta(seconds=forcing_timestep_seconds),
                end=extended_end_time,
                freq=f'{forcing_timestep_seconds}s'
            )
            padding_data = []
            for t in padding_times:
                padded = last_slice.copy()
                padded['time'] = t
                padding_data.append(padded)
            if padding_data:
                padding_ds = xr.concat(padding_data, dim='time')
                forcing_data = xr.concat([forcing_data, padding_ds], dim='time')
                self.logger.info(f"Added {len(padding_times)} padding timesteps to forcing data")

        ngen_ds = self._create_ngen_forcing_dataset(forcing_data, catchment_ids)
        output_file = self.forcing_dir / "forcing.nc"
        ngen_ds.to_netcdf(output_file, format='NETCDF4')
        self._write_csv_forcing_files(forcing_data, catchment_ids)
        return output_file

    def _write_csv_forcing_files(self, forcing_data: xr.Dataset, catchment_ids: List[str]) -> Path:
        # Ensure forcing_dir exists before creating csv subdirectory
        from pathlib import Path
        self.forcing_dir = Path(self.forcing_dir)  # Ensure it's a Path object
        self.forcing_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = self.forcing_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"CSV directory created: {csv_dir}, exists={csv_dir.exists()}, is_dir={csv_dir.is_dir()}")
        time_values = pd.to_datetime(forcing_data.time.values)

        # Variable mapping from ERA5/internal names to AORC/GRIB standard names
        # Use AORC standard naming (DLWRF_surface, DSWRF_surface) that NGEN recognizes
        var_mapping = {
            'pptrate': 'precip_rate',
            'airtemp': 'TMP_2maboveground',
            'spechum': 'SPFH_2maboveground',
            'airpres': 'PRES_surface',
            'SWRadAtm': 'DSWRF_surface',
            'LWRadAtm': 'DLWRF_surface',
            'windspeed_u': 'UGRD_10maboveground',
            'windspeed_v': 'VGRD_10maboveground',
        }

        for idx, cat_id in enumerate(catchment_ids):
            cols = {'time': time_values}
            for e_v, n_v in var_mapping.items():
                if e_v in forcing_data:
                    arr = forcing_data[e_v].values
                    cols[n_v] = arr[:, idx] if arr.ndim > 1 else arr

            df = pd.DataFrame(cols)
            df['APCP_surface'] = df['precip_rate'] if 'precip_rate' in df else 0

            # Add wind speed variables with default values if not present (required for PET and NOAH)
            if 'UGRD_10maboveground' not in df.columns:
                df['UGRD_10maboveground'] = 1.0  # Default 1 m/s
            if 'VGRD_10maboveground' not in df.columns:
                df['VGRD_10maboveground'] = 1.0  # Default 1 m/s

            df.to_csv(csv_dir / f"{cat_id}_forcing.csv", index=False)
        return csv_dir

    def _create_ngen_forcing_dataset(self, forcing_data: xr.Dataset, catchment_ids: List[str]) -> xr.Dataset:
        n_cats = len(catchment_ids)
        time_values = pd.to_datetime(forcing_data.time.values)
        time_ns = ((time_values.values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'ns')).astype(np.int64)
        ngen_ds = xr.Dataset()
        ngen_ds['ids'] = xr.DataArray(np.array(catchment_ids, dtype=object), dims=['catchment-id'])
        ngen_ds['Time'] = xr.DataArray(np.tile(time_ns, (n_cats, 1)).astype(np.float64), dims=['catchment-id', 'time'], attrs={'units': 'ns'})
        var_mapping = {'pptrate': 'precip_rate', 'airtemp': 'TMP_2maboveground', 'spechum': 'SPFH_2maboveground', 'airpres': 'PRES_surface', 'SWRadAtm': 'DSWRF_surface', 'LWRadAtm': 'DLWRF_surface'}
        for e_v, n_v in var_mapping.items():
            if e_v in forcing_data:
                data = forcing_data[e_v].values.T
                if data.shape[0] == 1 and n_cats > 1: data = np.tile(data, (n_cats, 1))
                ngen_ds[n_v] = xr.DataArray(data.astype(np.float32), dims=['catchment-id', 'time'])
        return ngen_ds

    def generate_model_configs(self):
        catchment_gdf = gpd.read_file(self.get_catchment_path())
        config_gen = NgenConfigGenerator(self.config_dict, self.logger, self.setup_dir, catchment_gdf.crs)
        config_gen.set_module_availability(cfe=self._include_cfe, pet=self._include_pet, noah=self._include_noah, sloth=self._include_sloth)
        config_gen.generate_all_configs(catchment_gdf, self.hru_id_col)

    def generate_realization_config(self, catchment_file: Path, nexus_file: Path, forcing_file: Path):
        config_gen = NgenConfigGenerator(self.config_dict, self.logger, self.setup_dir, getattr(self, 'catchment_crs', None))
        config_gen.set_module_availability(cfe=self._include_cfe, pet=self._include_pet, noah=self._include_noah, sloth=self._include_sloth)
        config_gen.generate_realization_config(forcing_file, self.project_dir, lib_paths=self._ngen_lib_paths)
