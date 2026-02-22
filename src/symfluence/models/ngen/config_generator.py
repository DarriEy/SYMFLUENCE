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

from symfluence.core.mixins import ConfigMixin


class NgenConfigGenerator(ConfigMixin):
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
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.setup_dir = Path(setup_dir)
        self.catchment_crs = catchment_crs

        # Module availability (set by preprocessor)
        self._include_cfe = True
        self._include_pet = True
        self._include_noah = True
        self._include_sloth = False
        self._noah_et_fallback = 'EVAPOTRANS'  # Default when PET disabled but NOAH enabled

    def set_module_availability(
        self,
        cfe: bool = True,
        pet: bool = True,
        noah: bool = True,
        sloth: bool = False,
        noah_et_fallback: str = 'EVAPOTRANS'
    ):
        """Set which NGEN modules are available.

        Args:
            cfe: Enable CFE runoff module
            pet: Enable PET evapotranspiration module
            noah: Enable NOAH-OWP land surface module
            sloth: Enable SLOTH ice fraction module
            noah_et_fallback: When PET disabled but NOAH enabled, which NOAH output
                to use as CFE's ET input. Options:
                - 'EVAPOTRANS': Total evapotranspiration rate [m/s] (default) -
                    correct units for CFE's water_potential_evaporation_flux
                - 'ETRAN': Transpiration [mm] - WRONG UNITS (accumulated depth,
                    dimensionally incompatible with CFE's m/s requirement)
                - 'QSEVA': Evaporation rate [mm/s] - rate but wrong dimension
                Note: All are ACTUAL ET (soil-moisture limited), not potential ET.
                For proper potential ET, enable the PET module.
        """
        self._include_cfe = cfe
        self._include_pet = pet
        self._include_noah = noah
        self._include_sloth = sloth
        self._noah_et_fallback = noah_et_fallback

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
            'max_gw_storage': 0.2,
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

        # Calculate catchment area in km² (required for CFE to convert depth to m³/s)
        catchment_area_km2 = 1.0  # Default fallback
        if catchment_row is not None:
            # Try to get area from pre-computed column first
            # Columns explicitly in km²: areasqkm, area_km2
            # Columns that may be in m²: HRU_area, GRU_area, AREA
            km2_cols = ['areasqkm', 'area_km2']  # Known to be in km²
            m2_cols = ['HRU_area', 'GRU_area', 'AREA']  # Often in m², need conversion check

            for area_col in km2_cols + m2_cols:
                if area_col in catchment_row.index:
                    area_val = catchment_row[area_col]
                    if area_val is not None and pd.notna(area_val) and float(area_val) > 0:
                        area_val = float(area_val)
                        # Check if value is likely in m² (no catchment is > 1 million km²)
                        # If value > 1e6, assume it's in m² and convert to km²
                        if area_col in m2_cols and area_val > 1e6:
                            catchment_area_km2 = area_val / 1e6
                            self.logger.debug(f"Converted {area_col} from m² to km²: {catchment_area_km2:.2f} km²")
                        else:
                            catchment_area_km2 = area_val
                            self.logger.debug(f"Using area from {area_col}: {catchment_area_km2:.2f} km²")
                        break
            else:
                # Calculate from geometry if no area column found
                if hasattr(catchment_row, 'geometry') and catchment_row.geometry is not None:
                    geom = catchment_row.geometry
                    # Project to equal-area CRS for accurate area calculation
                    if self.catchment_crs is not None and self.catchment_crs.is_geographic:
                        # Use UTM zone based on centroid for local accuracy
                        import pyproj
                        centroid = geom.centroid
                        utm_zone = int((centroid.x + 180) / 6) + 1
                        hemisphere = 'north' if centroid.y >= 0 else 'south'
                        utm_crs = pyproj.CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
                        transformer = pyproj.Transformer.from_crs(
                            self.catchment_crs, utm_crs, always_xy=True
                        )
                        from shapely.ops import transform
                        geom_projected = transform(transformer.transform, geom)
                        catchment_area_km2 = geom_projected.area / 1e6
                    else:
                        # Assume already in projected CRS with meters
                        catchment_area_km2 = geom.area / 1e6
                    self.logger.debug(f"Calculated area from geometry: {catchment_area_km2:.2f} km²")

        if catchment_area_km2 <= 0:
            self.logger.warning(
                f"Invalid catchment area ({catchment_area_km2}) for cat-{catchment_id}, "
                f"using default 1.0 km². CFE Q_OUT will be in depth units (m), not m³/s!"
            )
            catchment_area_km2 = 1.0

        # Calculate num_timesteps using configured forcing timestep (not hardcoded hourly)
        start_time = self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01 00:00:00', dict_key='EXPERIMENT_TIME_START')
        end_time = self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31 23:00:00', dict_key='EXPERIMENT_TIME_END')
        if start_time == 'default': start_time = '2000-01-01 00:00:00'
        if end_time == 'default': end_time = '2000-12-31 23:00:00'

        # Get forcing timestep from config (default 3600s = 1 hour)
        forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
        try:
            forcing_timestep = int(forcing_timestep)
        except (ValueError, TypeError):
            forcing_timestep = 3600

        try:
            duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
            # Use configured timestep, add 1 for inclusive end bound
            num_steps = int(duration.total_seconds() / forcing_timestep) + 1
        except (ValueError, TypeError):
            num_steps = 1

        config_text = f"""forcing_file=BMI
catchment_area_km2={catchment_area_km2}[km2]
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
giuh_ordinates={self._generate_giuh_ordinates(catchment_area_km2)}
num_timesteps={num_steps}
verbosity=1
surface_runoff_scheme=GIUH
surface_water_partitioning_scheme=Schaake
"""

        config_file = self.setup_dir / "CFE" / f"cat-{catchment_id}_bmi_config_cfe_pass.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
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
        ngen_config = self.config_dict.get('NGEN', {})
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
        forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
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

        # Calculate num_timesteps using configured forcing timestep (already in params['timestep'])
        start_time = self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01 00:00:00', dict_key='EXPERIMENT_TIME_START')
        end_time = self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31 23:00:00', dict_key='EXPERIMENT_TIME_END')
        if start_time == 'default': start_time = '2000-01-01 00:00:00'
        if end_time == 'default': end_time = '2000-12-31 23:00:00'

        try:
            duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
            # Use configured timestep (forcing_timestep), add 1 for inclusive end bound
            num_steps = int(duration.total_seconds() / forcing_timestep) + 1
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
shortwave_radiation_provided=1
"""

        config_file = self.setup_dir / "PET" / f"cat-{catchment_id}_pet_config.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
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
        start_time = self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01 00:00:00', dict_key='EXPERIMENT_TIME_START')
        end_time = self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31 23:00:00', dict_key='EXPERIMENT_TIME_END')

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

        # Extract terrain slope from catchment attributes if available
        terrain_slope = 0.01  # Default: gentle slope (more physical than 0.0)
        if catchment_row is not None:
            for slope_attr in ['terrain_slope', 'slope_mean', 'mean_slope', 'SLOPE', 'slope']:
                if slope_attr in catchment_row.index and pd.notna(catchment_row[slope_attr]):
                    terrain_slope = float(catchment_row[slope_attr])
                    self.logger.debug(f"Using terrain slope from {slope_attr}: {terrain_slope}")
                    break

        # Default NOAH options
        options = {
            'dt': 3600.0,
            'terrain_slope': terrain_slope,
            'azimuth': 0.0,
            'zref': 10.0,
            'rain_snow_thresh': 1.0,
            'soil_type': 3,
            'veg_type': 10,
            # Soil moisture parameters for physically-consistent initialization
            'smcmax': 0.5,    # Saturation moisture content (porosity)
            'smcwlt': 0.1,    # Wilting point moisture content
        }
        options.update(overrides)

        # Calculate initial soil moisture as a reasonable fraction of saturation
        # This ensures initial conditions are physically consistent with calibration bounds
        # Default: 60% of saturation (between field capacity and saturation)
        initial_moisture_frac = self._get_config_value(
            lambda: self.config.model.ngen.initial_moisture_fraction,
            default=0.6,
            dict_key='NOAH_INITIAL_MOISTURE_FRACTION'
        )
        initial_sh2o = options['smcmax'] * initial_moisture_frac
        # Ensure we're at least at wilting point
        initial_sh2o = max(initial_sh2o, options['smcwlt'])

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
 sh2o      =  {initial_sh2o:.3f},  {initial_sh2o:.3f},  {initial_sh2o:.3f},  {initial_sh2o:.3f}
 zwt       =  -2.0
/
"""

        config_file = self.setup_dir / "NOAH" / f"cat-{catchment_id}.input"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
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

        forcing_provider = self.config_dict.get('NGEN_FORCING_PROVIDER')
        if not forcing_provider:
            # CsvPerFeature: works but has SIGSEGV bug on macOS ARM64 (~19% crash rate,
            # mitigated by retry logic in runner.py).
            # NetCDF: avoids SIGSEGV but requires AORC-format NetCDF with epoch_start
            # attribute and CSDMS variable names. Our forcing pipeline doesn't yet
            # produce this format, so default to CsvPerFeature for now.
            forcing_provider = "CsvPerFeature"

        # Handle simulation times
        sim_start = self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01 00:00:00', dict_key='EXPERIMENT_TIME_START')
        sim_end = self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31 23:00:00', dict_key='EXPERIMENT_TIME_END')

        if sim_start == 'default':
            sim_start = '2000-01-01 00:00:00'
        if sim_end == 'default':
            sim_end = '2000-12-31 23:00:00'

        sim_start = pd.to_datetime(sim_start).strftime('%Y-%m-%d %H:%M:%S')
        sim_end = pd.to_datetime(sim_end).strftime('%Y-%m-%d %H:%M:%S')

        # Get forcing timestep from config (default to 3600 if not specified)
        forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
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

        experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='default_run', dict_key='EXPERIMENT_ID')
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
        with open(config_file, 'w', encoding='utf-8') as f:
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
                        "sloth_smp(1,double,1,node)": 0.0,
                        "sloth_atmosphere_air_water~vapor__relative_saturation(1,double,1,node)": 0.5
                    }
                }
            })

        if self._include_pet:
            lib_file = str(lib_paths.get("PET", f"./extern/evapotranspiration/evapotranspiration/cmake_build/libpetbmi{lib_ext}"))
            pet_params: dict[str, Any] = {
                "model_type_name": "bmi_c_pet",
                "library_file": lib_file,
                "forcing_file": "",
                "init_config": f"{pet_base}/{{{{id}}}}_pet_config.txt",
                "allow_exceed_end_time": True,
                "main_output_variable": "water_potential_evaporation_flux",
                "registration_function": "register_bmi_pet",
                "uses_forcing_file": False
            }
            # PET requires atmosphere_air_water~vapor__relative_saturation as a BMI input.
            # When SLOTH is enabled, it provides this as a dummy value; PET with yes_aorc=1
            # computes actual humidity internally from specific humidity in forcing.
            if self._include_sloth:
                pet_params["variables_names_map"] = {
                    "atmosphere_air_water~vapor__relative_saturation":
                        "sloth_atmosphere_air_water~vapor__relative_saturation"
                }
            pet_config = {
                "name": "bmi_c",
                "params": pet_params
            }
            modules.append(pet_config)

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
                        "Q2": "atmosphere_air_water~vapor__specific_humidity",
                        "SFCTMP": "land_surface_air__temperature",
                        "UU": "land_surface_wind__x_component_of_velocity",
                        "VV": "land_surface_wind__y_component_of_velocity",
                        "LWDN": "land_surface_radiation~incoming~longwave__energy_flux",
                        "SOLDN": "land_surface_radiation~incoming~shortwave__energy_flux",
                        "SFCPRS": "land_surface_air__pressure"
                    },
                    # Expose NOAH outputs for coupling with CFE
                    # Variable names must match NOAH-OWP BMI output_items
                    # (see bmi_noahowp.f90 noahowp_output_var_names)
                    "output_variables": [
                        "QINSUR",       # Net water input to soil surface [m/s]
                        "EVAPOTRANS",   # Total evapotranspiration rate [m/s]
                        "ETRAN",        # Transpiration [mm] (accumulated per timestep)
                        "ECAN",         # Canopy interception evaporation [mm]
                        "QSEVA"         # Direct soil evaporation rate [mm/s]
                    ]
                }
            })

        if self._include_cfe:
            lib_file = str(lib_paths.get("CFE", f"./extern/cfe/cmake_build/libcfebmi{lib_ext}"))

            # Build variables_names_map for CFE
            # QINSUR-based coupling when NOAH is enabled:
            # - CFE receives NOAH's QINSUR (net water input to soil surface, post-snow
            #   and post-interception) as its precipitation input
            # - PET provides potential ET for CFE's internal soil moisture accounting
            # When NOAH is NOT enabled (PET-only, standard coupling):
            # - CFE receives raw forcing precipitation directly
            # - PET provides potential ET
            if self._include_noah:
                variables_map = {
                    # CFE receives NOAH's net surface water input (post-snow, post-interception)
                    "atmosphere_water__liquid_equivalent_precipitation_rate": "QINSUR",
                }
            else:
                variables_map = {
                    # Standard coupling: CFE receives forcing precipitation directly
                    "atmosphere_water__liquid_equivalent_precipitation_rate": "atmosphere_water__liquid_equivalent_precipitation_rate",
                }

            # Add SLOTH variables if SLOTH is enabled (provides ice fraction for partitioning)
            if self._include_sloth:
                variables_map["ice_fraction_schaake"] = "sloth_ice_fraction_schaake"
                variables_map["ice_fraction_xinanjiang"] = "sloth_ice_fraction_xinanjiang"
                variables_map["soil_moisture_profile"] = "sloth_smp"

            # Add evapotranspiration source: PET provides potential ET for CFE's
            # soil moisture depletion, regardless of whether NOAH is enabled
            if self._include_pet:
                variables_map["water_potential_evaporation_flux"] = "water_potential_evaporation_flux"
            elif self._include_noah:
                # Fallback: if PET is not enabled but NOAH is, use configured NOAH output
                # EVAPOTRANS (m/s) is the only NOAH output with correct units for CFE.
                # ETRAN/ECAN are in mm (accumulated depth) — dimensionally incompatible.
                # NOTE: This is ACTUAL ET (soil-moisture limited), not potential ET.
                # For proper potential ET, enable PET module.
                et_var = getattr(self, '_noah_et_fallback', 'EVAPOTRANS')
                variables_map["water_potential_evaporation_flux"] = et_var

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
                    "output_variable_units": "m3/s"
                }
            })

        return modules

    @staticmethod
    def _generate_giuh_ordinates(catchment_area_km2: float) -> str:
        """Generate GIUH ordinates appropriate for basin size.

        Uses a gamma-distribution-shaped unit hydrograph with response time
        scaled to catchment area. Small basins get short, peaked responses;
        large basins get longer, flatter responses.

        Args:
            catchment_area_km2: Catchment area in km²

        Returns:
            Comma-separated string of ordinates summing to 1.0
        """
        import math

        # Estimate response time in hours (empirical: sqrt(area) / 2, min 2h)
        response_time = max(math.sqrt(catchment_area_km2) / 2.0, 2.0)

        # Number of ordinates: cover ~2x response time, min 5
        n_ordinates = max(int(response_time * 2), 5)
        # Cap at 48 ordinates (2 days at hourly timestep)
        n_ordinates = min(n_ordinates, 48)

        # Gamma distribution shape: k=2 gives realistic geomorphological response
        k = 2.0
        theta = response_time / k  # scale parameter

        # Generate gamma PDF values at each hour
        ordinates = []
        for i in range(1, n_ordinates + 1):
            t = float(i)
            # Gamma PDF: t^(k-1) * exp(-t/theta) / (theta^k * Gamma(k))
            val = (t ** (k - 1)) * math.exp(-t / theta) / (theta ** k * math.gamma(k))
            ordinates.append(val)

        # Normalize to sum to 1.0
        total = sum(ordinates)
        if total > 0:
            ordinates = [o / total for o in ordinates]

        # Format with 4 decimal places
        return ','.join(f'{o:.4f}' for o in ordinates)

    def _get_wgs84_centroid(self, catchment_row: gpd.GeoSeries):
        """Get the centroid of a catchment in WGS84 coordinates."""
        centroid = catchment_row.geometry.centroid

        if self.catchment_crs and str(self.catchment_crs) != "EPSG:4326":
            geom_wgs84 = gpd.GeoSeries([catchment_row.geometry], crs=self.catchment_crs)
            geom_wgs84 = geom_wgs84.to_crs("EPSG:4326")
            centroid = geom_wgs84.iloc[0].centroid

        return centroid
