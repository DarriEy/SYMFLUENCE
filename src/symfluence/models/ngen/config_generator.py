"""
NGEN Model Configuration Generator

Handles generation of model-specific configuration files for NextGen Framework:
- CFE (Conceptual Functional Equivalent) configs
- PET (Potential Evapotranspiration) configs
- NOAH-OWP (Noah-Owens-Pries) configs

Extracted from NgenPreProcessor to improve modularity and testability.
"""

import sys
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, Optional


class NgenConfigGenerator:
    """
    Generator for NGEN model configuration files.

    Handles creation of:
    - CFE BMI configuration files (.txt)
    - PET configuration files (.txt)
    - NOAH-OWP input files (.input)
    - Realization configuration (JSON)

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        setup_dir: Path to NGEN settings directory
        catchment_crs: CRS of the catchment geometries
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        setup_dir: Path,
        catchment_crs: Any = None
    ):
        """
        Initialize the config generator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            setup_dir: Path to NGEN settings directory
            catchment_crs: Coordinate reference system for catchments
        """
        self.config = config
        self.logger = logger
        self.setup_dir = Path(setup_dir)
        self.catchment_crs = catchment_crs

        # Module availability (set by preprocessor)
        self._include_cfe = True
        self._include_pet = True
        self._include_noah = True
        self._include_sloth = False

    def set_module_availability(
        self,
        cfe: bool = True,
        pet: bool = True,
        noah: bool = True,
        sloth: bool = False
    ):
        """Set which NGEN modules are available."""
        self._include_cfe = cfe
        self._include_pet = pet
        self._include_noah = noah
        self._include_sloth = sloth

    def generate_all_configs(
        self,
        catchment_gdf: gpd.GeoDataFrame,
        hru_id_col: str
    ) -> None:
        """
        Generate all model configuration files for each catchment.

        Args:
            catchment_gdf: GeoDataFrame with catchment geometries
            hru_id_col: Column name for catchment IDs
        """
        self.logger.info("Generating model configuration files")
        self.catchment_crs = catchment_gdf.crs

        for idx, catchment in catchment_gdf.iterrows():
            cat_id = str(catchment[hru_id_col])

            if self._include_cfe:
                self.generate_cfe_config(cat_id, catchment)
            if self._include_pet:
                self.generate_pet_config(cat_id, catchment)
            if self._include_noah:
                self.generate_noah_config(cat_id, catchment)

        self.logger.info(f"Generated configs for {len(catchment_gdf)} catchments")

    def generate_cfe_config(
        self,
        catchment_id: str,
        catchment_row: Optional[gpd.GeoSeries] = None,
        **overrides
    ) -> Path:
        """
        Generate CFE model configuration file.

        Args:
            catchment_id: Catchment identifier
            catchment_row: Optional GeoSeries with catchment data
            **overrides: Parameter overrides (e.g., soil_b=6.0)

        Returns:
            Path to generated config file
        """
        # Default parameters (can be overridden)
        params = {
            'depth': 2.0,
            'soil_b': 5.0,
            'satdk': 5.0e-06,
            'satpsi': 0.141,
            'slop': 0.03,
            'smcmax': 0.439,
            'wltsmc': 0.047,
            'expon': 1.0,
            'expon_secondary': 1.0,
            'refkdt': 1.0,
            'max_gw_storage': 0.0129,
            'cgw': 1.8e-05,
            'gw_expon': 7.0,
            'gw_storage': 0.35,
            'alpha_fc': 0.33,
            'soil_storage': 0.35,
            'k_nash': 0.03,
            'k_lf': 0.01,
        }

        # Apply overrides
        params.update(overrides)

        # Calculate num_timesteps
        start_time = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        end_time = self.config.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')
        if start_time == 'default': start_time = '2000-01-01 00:00:00'
        if end_time == 'default': end_time = '2000-12-31 23:00:00'
        
        try:
            duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
            num_steps = int(duration.total_seconds() / 3600)
        except (ValueError, TypeError):
            num_steps = 1

        config_text = f"""forcing_file=BMI
soil_params.depth={params['depth']}[m]
soil_params.b={params['soil_b']}[]
soil_params.satdk={params['satdk']:.2e}[m s-1]
soil_params.satpsi={params['satpsi']}[m]
soil_params.slop={params['slop']}[m/m]
soil_params.smcmax={params['smcmax']}[m/m]
soil_params.wltsmc={params['wltsmc']}[m/m]
soil_params.expon={params['expon']}[]
soil_params.expon_secondary={params['expon_secondary']}[]
max_gw_storage={params['max_gw_storage']}[m]
Cgw={params['cgw']:.2e}[m h-1]
expon={params['gw_expon']}[]
gw_storage={params['gw_storage']}[m/m]
alpha_fc={params['alpha_fc']}[]
soil_storage={params['soil_storage']}[m/m]
K_nash={params['k_nash']}[]
K_lf={params['k_lf']}[]
nash_storage=0.0,0.0
giuh_ordinates=0.06,0.51,0.28,0.12,0.03
num_timesteps={num_steps}
verbosity=1
surface_runoff_scheme=GIUH
surface_water_partitioning_scheme=Schaake
"""

        config_file = self.setup_dir / "CFE" / f"cat-{catchment_id}_bmi_config_cfe_pass.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_text)

        return config_file

    def generate_pet_config(
        self,
        catchment_id: str,
        catchment_row: gpd.GeoSeries,
        **overrides
    ) -> Path:
        """
        Generate PET model configuration file.

        Args:
            catchment_id: Catchment identifier
            catchment_row: GeoSeries with catchment geometry
            **overrides: Parameter overrides

        Returns:
            Path to generated config file
        """
        # Get catchment centroid
        centroid = self._get_wgs84_centroid(catchment_row)

        # Extract elevation from catchment attributes if available
        catchment_elevation = None
        for elev_attr in ['elevation_m', 'elevation', 'elev_mean', 'mean_elev', 'elev_m']:
            if elev_attr in catchment_row.index and pd.notna(catchment_row[elev_attr]):
                catchment_elevation = float(catchment_row[elev_attr])
                break

        # Get config-level PET settings
        ngen_config = self.config.get('NGEN', {})
        pet_config = ngen_config.get('PET', {})

        # Use config elevation if specified, otherwise catchment attribute, otherwise default
        elevation = pet_config.get('elevation_m', catchment_elevation if catchment_elevation else 100.0)

        # Set vegetation parameters based on elevation (forest vs grass)
        # Mountain watersheds (>1000m) typically have forest/mixed vegetation
        if elevation > 1000:
            default_veg_height = 15.0  # Forest
            default_zero_plane = 10.0
            default_momentum_roughness = 1.5
        else:
            default_veg_height = 0.12  # Grass
            default_zero_plane = 0.0003
            default_momentum_roughness = 0.0

        # Default parameters with intelligent defaults
        # Use forcing timestep for PET instead of hardcoded hourly
        forcing_timestep = self.config.get('FORCING_TIME_STEP_SIZE', 3600)
        try:
            forcing_timestep = int(forcing_timestep)
        except (ValueError, TypeError):
            forcing_timestep = 3600

        params = {
            'wind_height': pet_config.get('wind_height_m', 10.0),
            'humidity_height': pet_config.get('humidity_height_m', 2.0),
            'veg_height': pet_config.get('vegetation_height_m', default_veg_height),
            'zero_plane_displacement': pet_config.get('zero_plane_displacement_m', default_zero_plane),
            'momentum_roughness': pet_config.get('momentum_roughness_length_m', default_momentum_roughness),
            'heat_roughness': pet_config.get('heat_roughness_length_m', 0.0),
            'emissivity': pet_config.get('emissivity', 1.0),
            'albedo': pet_config.get('albedo', 0.23),
            'elevation': elevation,
            'timestep': forcing_timestep,
        }
        params.update(overrides)

        # Log PET configuration parameters
        self.logger.info(f"Generating PET config for {catchment_id}:")
        self.logger.info(f"  Elevation: {params['elevation']:.1f} m" +
                        (" (from catchment attribute)" if catchment_elevation else " (default)"))
        self.logger.info(f"  Vegetation height: {params['veg_height']:.2f} m")
        self.logger.info(f"  Momentum roughness: {params['momentum_roughness']:.2f} m")

        # Calculate num_timesteps
        start_time = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        end_time = self.config.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')
        if start_time == 'default': start_time = '2000-01-01 00:00:00'
        if end_time == 'default': end_time = '2000-12-31 23:00:00'
        
        try:
            duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
            num_steps = int(duration.total_seconds() / 3600)
        except (ValueError, TypeError):
            num_steps = 1

        config_text = f"""verbose=0
pet_method=5
forcing_file=BMI
run_unit_tests=0
yes_aorc=1
yes_wrf=0
wind_speed_measurement_height_m={params['wind_height']}
humidity_measurement_height_m={params['humidity_height']}
vegetation_height_m={params['veg_height']}
zero_plane_displacement_height_m={params['zero_plane_displacement']}
momentum_transfer_roughness_length={params['momentum_roughness']}
heat_transfer_roughness_length_m={params['heat_roughness']}
surface_longwave_emissivity={params['emissivity']}
surface_shortwave_albedo={params['albedo']}
cloud_base_height_known=FALSE
latitude_degrees={centroid.y}
longitude_degrees={centroid.x}
site_elevation_m={params['elevation']}
time_step_size_s={params['timestep']}
num_timesteps={num_steps}
shortwave_radiation_provided=0
"""

        config_file = self.setup_dir / "PET" / f"cat-{catchment_id}_pet_config.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_text)

        return config_file

    def generate_noah_config(
        self,
        catchment_id: str,
        catchment_row: gpd.GeoSeries,
        **overrides
    ) -> Path:
        """
        Generate NOAH-OWP model configuration file (.input file).

        Creates a Fortran namelist file with all required sections.

        Args:
            catchment_id: Catchment identifier
            catchment_row: GeoSeries with catchment geometry
            **overrides: Parameter overrides

        Returns:
            Path to generated config file
        """
        # Get catchment centroid
        centroid = self._get_wgs84_centroid(catchment_row)

        # Get simulation timing from config
        start_time = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        end_time = self.config.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')

        # Handle 'default' strings
        if start_time == 'default':
            start_time = '2000-01-01 00:00:00'
        if end_time == 'default':
            end_time = '2000-12-31 23:00:00'

        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        start_str = start_dt.strftime('%Y%m%d%H%M')
        end_str = end_dt.strftime('%Y%m%d%H%M')

        # Absolute path to parameters directory
        param_dir = str((self.setup_dir / "NOAH" / "parameters").resolve()) + "/"

        # Default NOAH options
        options = {
            'dt': 3600.0,
            'terrain_slope': 0.0,
            'azimuth': 0.0,
            'zref': 10.0,
            'rain_snow_thresh': 1.0,
            'soil_type': 3,
            'veg_type': 10,
        }
        options.update(overrides)

        config_text = f"""&timing
  dt                 = {options['dt']}
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
  terrain_slope      = {options['terrain_slope']}
  azimuth            = {options['azimuth']}
/

&forcing
  ZREF               = {options['zref']}
  rain_snow_thresh   = {options['rain_snow_thresh']}
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
 isltyp           = {options['soil_type']}
 nsoil            = 4
 nsnow            = 3
 nveg             = 20
 vegtyp           = {options['veg_type']}
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
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_text)

        return config_file

    def generate_realization_config(
        self,
        forcing_file: Path,
        project_dir: Path,
        lib_paths: Optional[Dict[str, Path]] = None
    ) -> Path:
        """
        Generate ngen realization configuration JSON.

        Args:
            forcing_file: Path to forcing NetCDF file
            project_dir: Project directory for output paths
            lib_paths: Optional dict of module library paths

        Returns:
            Path to generated realization config
        """
        self.logger.info("Generating realization configuration")

        forcing_abs_path = str(forcing_file.resolve())
        cfe_config_base = str((self.setup_dir / "CFE").resolve())
        pet_config_base = str((self.setup_dir / "PET").resolve())
        noah_config_base = str((self.setup_dir / "NOAH").resolve())

        lib_ext = ".dylib" if sys.platform == "darwin" else ".so"

        forcing_provider = self.config.get("NGEN_FORCING_PROVIDER")
        if not forcing_provider:
            forcing_provider = "CsvPerFeature" if sys.platform == "darwin" else "NetCDF"

        # Handle simulation times
        sim_start = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        sim_end = self.config.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')

        if sim_start == 'default':
            sim_start = '2000-01-01 00:00:00'
        if sim_end == 'default':
            sim_end = '2000-12-31 23:00:00'

        sim_start = pd.to_datetime(sim_start).strftime('%Y-%m-%d %H:%M:%S')
        sim_end = pd.to_datetime(sim_end).strftime('%Y-%m-%d %H:%M:%S')

        # Get forcing timestep from config (default to 3600 if not specified)
        forcing_timestep = self.config.get('FORCING_TIME_STEP_SIZE', 3600)
        try:
            forcing_timestep = int(forcing_timestep)
            self.logger.info(f"Using forcing timestep: {forcing_timestep} seconds ({forcing_timestep/3600:.1f} hours)")
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid FORCING_TIME_STEP_SIZE: {forcing_timestep}, using default 3600")
            forcing_timestep = 3600

        # Build forcing config
        if forcing_provider == "CsvPerFeature":
            forcing_config = {
                "path": str((forcing_file.parent / "csv").resolve()),
                "provider": "CsvPerFeature",
                "file_pattern": ".*{{id}}_forcing.*\\.csv"
            }
        else:
            forcing_config = {
                "path": forcing_abs_path,
                "provider": forcing_provider
            }

        # Build module configurations
        modules = self._build_module_configs(
            cfe_config_base, pet_config_base, noah_config_base, lib_ext,
            lib_paths=lib_paths
        )

        experiment_id = self.config.get('EXPERIMENT_ID', 'default_run')
        output_root = str((project_dir / "simulations" / experiment_id / "NGEN").resolve())

        config = {
            "global": {
                "formulations": [{
                    "name": "bmi_multi",
                    "params": {
                        "model_type_name": "bmi_multi_noahowp_cfe",
                        "init_config": "",
                        "allow_exceed_end_time": True,
                        "main_output_variable": "Q_OUT",
                        "modules": modules,
                    }
                }],
                "forcing": forcing_config
            },
            "time": {
                "start_time": sim_start,
                "end_time": sim_end,
                "output_interval": forcing_timestep
            },
            "output_root": output_root
        }

        config_file = self.setup_dir / "realization_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Created realization config: {config_file}")
        return config_file

    def _build_module_configs(
        self,
        cfe_base: str,
        pet_base: str,
        noah_base: str,
        lib_ext: str,
        lib_paths: Optional[Dict[str, Path]] = None
    ) -> list:
        """Build the list of module configurations for realization."""
        modules = []
        lib_paths = lib_paths or {}

        if self._include_sloth:
            lib_file = str(lib_paths.get("SLOTH", f"./extern/sloth/cmake_build/libslothmodel{lib_ext}"))
            modules.append({
                "name": "bmi_c++",
                "params": {
                    "model_type_name": "bmi_c++_sloth",
                    "library_file": lib_file,
                    "init_config": "/dev/null",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "z",
                    "uses_forcing_file": False,
                    "model_params": {
                        "sloth_ice_fraction_schaake(1,double,m,node)": 0.0,
                        "sloth_ice_fraction_xinanjiang(1,double,1,node)": 0.0,
                        "sloth_smp(1,double,1,node)": 0.0
                    }
                }
            })

        if self._include_pet:
            lib_file = str(lib_paths.get("PET", f"./extern/evapotranspiration/evapotranspiration/cmake_build/libpetbmi{lib_ext}"))
            modules.append({
                "name": "bmi_c",
                "params": {
                    "model_type_name": "bmi_c_pet",
                    "library_file": lib_file,
                    "forcing_file": "",
                    "init_config": f"{pet_base}/{{{{id}}}}_pet_config.txt",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "water_potential_evaporation_flux",
                    "registration_function": "register_bmi_pet",
                    "uses_forcing_file": False
                }
            })

        if self._include_noah:
            lib_file = str(lib_paths.get("NOAH", f"./extern/noah-owp-modular/cmake_build/libsurfacebmi{lib_ext}"))
            modules.append({
                "name": "bmi_fortran",
                "params": {
                    "model_type_name": "bmi_fortran_noahowp",
                    "library_file": lib_file,
                    "forcing_file": "",
                    "init_config": f"{noah_base}/{{{{id}}}}.input",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "QINSUR",
                    "variables_names_map": {
                        "PRCPNONC": "atmosphere_water__liquid_equivalent_precipitation_rate",
                        "Q2": "atmosphere_air_water~vapor__relative_saturation",
                        "SFCTMP": "land_surface_air__temperature",
                        "UU": "land_surface_wind__x_component_of_velocity",
                        "VV": "land_surface_wind__y_component_of_velocity",
                        "LWDN": "land_surface_radiation~incoming~longwave__energy_flux",
                        "SOLDN": "land_surface_radiation~incoming~shortwave__energy_flux",
                        "SFCPRS": "land_surface_air__pressure"
                    }
                }
            })

        if self._include_cfe:
            lib_file = str(lib_paths.get("CFE", f"./extern/cfe/cmake_build/libcfebmi{lib_ext}"))

            # Build variables_names_map for CFE
            # CFE gets forcing variables from the forcing provider and SLOTH/PET outputs from other modules
            variables_map = {
                "atmosphere_water__liquid_equivalent_precipitation_rate": "precip_rate",
            }

            # Add SLOTH variables if SLOTH is enabled
            if self._include_sloth:
                variables_map["ice_fraction_schaake"] = "sloth_ice_fraction_schaake"
                variables_map["ice_fraction_xinanjiang"] = "sloth_ice_fraction_xinanjiang"
                variables_map["soil_moisture_profile"] = "sloth_smp"

            # Add PET variable if PET is enabled
            if self._include_pet:
                variables_map["water_potential_evaporation_flux"] = "water_potential_evaporation_flux"

            modules.append({
                "name": "bmi_c",
                "params": {
                    "model_type_name": "bmi_c_cfe",
                    "library_file": lib_file,
                    "forcing_file": "",
                    "init_config": f"{cfe_base}/{{{{id}}}}_bmi_config_cfe_pass.txt",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "Q_OUT",
                    "registration_function": "register_bmi_cfe",
                    "variables_names_map": variables_map,
                    "uses_forcing_file": False,
                    "output_variable_units": "m"
                }
            })

        return modules

    def _get_wgs84_centroid(self, catchment_row: gpd.GeoSeries):
        """Get the centroid of a catchment in WGS84 coordinates."""
        centroid = catchment_row.geometry.centroid

        if self.catchment_crs and str(self.catchment_crs) != "EPSG:4326":
            geom_wgs84 = gpd.GeoSeries([catchment_row.geometry], crs=self.catchment_crs)
            geom_wgs84 = geom_wgs84.to_crs("EPSG:4326")
            centroid = geom_wgs84.iloc[0].centroid

        return centroid
