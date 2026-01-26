"""
IGNACIO Fire Model Configuration for SYMFLUENCE

Pydantic configuration model for IGNACIO fire simulation parameters.
Maps SYMFLUENCE configuration to IGNACIO's YAML config format.
"""

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator

# Import FROZEN_CONFIG from the base config module
try:
    from symfluence.core.config.models.base import FROZEN_CONFIG
except ImportError:
    # Fallback for standalone usage
    FROZEN_CONFIG = {"frozen": True, "extra": "forbid"}


class IGNACIOConfig(BaseModel):
    """
    IGNACIO fire spread model configuration.

    Maps SYMFLUENCE configuration parameters to IGNACIO's config format.
    IGNACIO uses YAML configuration files; this class helps generate them
    from SYMFLUENCE's unified configuration system.

    Attributes:
        project_name: Name for the simulation project
        output_dir: Directory for IGNACIO outputs
        dem_path: Path to DEM raster
        fuel_path: Path to fuel type raster
        ignition_shapefile: Path to ignition point shapefile
        ignition_date: Ignition start date/time
        station_path: Path to weather station CSV
        max_duration: Maximum simulation duration in minutes
        dt: Simulation timestep in minutes
        compare_with_wmfire: Whether to compare results with WMFire
    """

    model_config = FROZEN_CONFIG

    # Project settings
    project_name: str = Field(
        default='ignacio_run',
        alias='IGNACIO_PROJECT_NAME',
        description='Name for the IGNACIO simulation project'
    )
    output_dir: str = Field(
        default='default',
        alias='IGNACIO_OUTPUT_DIR',
        description='Output directory for IGNACIO results'
    )
    random_seed: int = Field(
        default=42,
        alias='IGNACIO_RANDOM_SEED',
        ge=0,
        description='Random seed for reproducibility'
    )

    # CRS settings
    working_crs: str = Field(
        default='EPSG:4326',
        alias='IGNACIO_WORKING_CRS',
        description='Working coordinate reference system'
    )
    output_crs: Optional[str] = Field(
        default=None,
        alias='IGNACIO_OUTPUT_CRS',
        description='Output CRS (defaults to working_crs)'
    )

    # Terrain inputs
    dem_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_DEM_PATH',
        description='Path to DEM raster file'
    )
    slope_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_SLOPE_PATH',
        description='Path to pre-computed slope raster (optional)'
    )
    aspect_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_ASPECT_PATH',
        description='Path to pre-computed aspect raster (optional)'
    )

    # Fuel inputs
    fuel_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_FUEL_PATH',
        description='Path to fuel type raster'
    )
    fuel_source_type: Literal['raster', 'vector', 'constant'] = Field(
        default='raster',
        alias='IGNACIO_FUEL_SOURCE_TYPE',
        description='Type of fuel data source'
    )
    default_fuel_type: str = Field(
        default='C-2',
        alias='IGNACIO_DEFAULT_FUEL',
        description='Default FBP fuel type code'
    )
    non_fuel_codes: List[int] = Field(
        default_factory=lambda: [0, 100, 101, 102, -9999],
        alias='IGNACIO_NON_FUEL_CODES',
        description='Raster codes to treat as non-fuel'
    )

    # Ignition configuration
    ignition_shapefile: Optional[str] = Field(
        default=None,
        alias='IGNACIO_IGNITION_SHAPEFILE',
        description='Path to ignition point shapefile'
    )
    ignition_date: Optional[str] = Field(
        default=None,
        alias='IGNACIO_IGNITION_DATE',
        description='Ignition date/time as YYYY-MM-DD HH:MM:SS'
    )
    ignition_cause: str = Field(
        default='Lightning',
        alias='IGNACIO_IGNITION_CAUSE',
        description='Cause of ignition'
    )
    n_iterations: int = Field(
        default=1,
        alias='IGNACIO_N_ITERATIONS',
        ge=1,
        description='Number of simulation iterations'
    )

    # Weather configuration
    station_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_STATION_PATH',
        description='Path to weather station CSV file'
    )
    calculate_fwi: bool = Field(
        default=True,
        alias='IGNACIO_CALCULATE_FWI',
        description='Calculate FWI from weather data'
    )
    fwi_latitude: Optional[float] = Field(
        default=None,
        alias='IGNACIO_FWI_LATITUDE',
        ge=-90,
        le=90,
        description='Latitude for FWI day length adjustment'
    )

    # FBP defaults (used when weather data unavailable)
    default_ffmc: float = Field(
        default=88.0,
        alias='IGNACIO_DEFAULT_FFMC',
        ge=0,
        le=101,
        description='Default Fine Fuel Moisture Code'
    )
    default_dmc: float = Field(
        default=30.0,
        alias='IGNACIO_DEFAULT_DMC',
        ge=0,
        description='Default Duff Moisture Code'
    )
    default_dc: float = Field(
        default=150.0,
        alias='IGNACIO_DEFAULT_DC',
        ge=0,
        description='Default Drought Code'
    )
    default_isi: float = Field(
        default=5.0,
        alias='IGNACIO_DEFAULT_ISI',
        ge=0,
        description='Default Initial Spread Index'
    )
    default_bui: float = Field(
        default=50.0,
        alias='IGNACIO_DEFAULT_BUI',
        ge=0,
        description='Default Buildup Index'
    )
    fmc: float = Field(
        default=100.0,
        alias='IGNACIO_FMC',
        ge=0,
        description='Foliar Moisture Content (%)'
    )
    curing: float = Field(
        default=85.0,
        alias='IGNACIO_CURING',
        ge=0,
        le=100,
        description='Grass curing percentage'
    )

    # Simulation parameters
    dt: float = Field(
        default=1.0,
        alias='IGNACIO_DT',
        ge=0.1,
        le=60.0,
        description='Simulation timestep in minutes'
    )
    max_duration: int = Field(
        default=480,
        alias='IGNACIO_MAX_DURATION',
        ge=1,
        le=43200,
        description='Maximum simulation duration in minutes'
    )
    n_vertices: int = Field(
        default=300,
        alias='IGNACIO_N_VERTICES',
        ge=50,
        le=1000,
        description='Number of vertices for fire perimeter'
    )
    initial_radius: float = Field(
        default=10.0,
        alias='IGNACIO_INITIAL_RADIUS',
        ge=1.0,
        le=1000.0,
        description='Initial fire radius in meters'
    )
    min_ros: float = Field(
        default=0.01,
        alias='IGNACIO_MIN_ROS',
        ge=0,
        description='Minimum rate of spread threshold'
    )
    time_varying_weather: bool = Field(
        default=True,
        alias='IGNACIO_TIME_VARYING_WEATHER',
        description='Use time-varying weather during simulation'
    )

    # Output configuration
    save_perimeters: bool = Field(
        default=True,
        alias='IGNACIO_SAVE_PERIMETERS',
        description='Save fire perimeter shapefiles'
    )
    save_ros_grids: bool = Field(
        default=True,
        alias='IGNACIO_SAVE_ROS_GRIDS',
        description='Save rate of spread grids'
    )
    perimeter_format: Literal['shapefile', 'geojson', 'gpkg'] = Field(
        default='shapefile',
        alias='IGNACIO_PERIMETER_FORMAT',
        description='Output format for perimeter files'
    )
    generate_plots: bool = Field(
        default=True,
        alias='IGNACIO_GENERATE_PLOTS',
        description='Generate visualization plots'
    )

    # Comparison with WMFire
    compare_with_wmfire: bool = Field(
        default=False,
        alias='IGNACIO_COMPARE_WMFIRE',
        description='Compare fire perimeters with WMFire results'
    )
    observed_perimeter_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_OBSERVED_PERIMETER',
        description='Path to observed fire perimeter for validation'
    )

    @field_validator('ignition_date')
    @classmethod
    def validate_ignition_date(cls, v: Optional[str]) -> Optional[str]:
        """Validate ignition date format."""
        if v is not None:
            from datetime import datetime
            # Try multiple formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    datetime.strptime(v, fmt)
                    return v
                except ValueError:
                    continue
            raise ValueError(
                f"Invalid ignition date format: {v}. "
                f"Expected YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"
            )
        return v

    def to_ignacio_config(self) -> Dict[str, Any]:
        """
        Convert to IGNACIO YAML config format.

        Returns:
            Dictionary that can be written as IGNACIO's YAML config.
        """
        config = {
            'project': {
                'name': self.project_name,
                'output_dir': self.output_dir,
                'random_seed': self.random_seed,
            },
            'crs': {
                'working_crs': self.working_crs,
                'output_crs': self.output_crs or self.working_crs,
            },
            'terrain': {
                'dem_path': self.dem_path,
                'slope_path': self.slope_path,
                'aspect_path': self.aspect_path,
            },
            'fuel': {
                'source_type': self.fuel_source_type,
                'path': self.fuel_path,
                'non_fuel_codes': self.non_fuel_codes,
            },
            'ignition': {
                'source_type': 'shapefile',
                'point_path': self.ignition_shapefile,
                'cause': self.ignition_cause,
                'n_iterations': self.n_iterations,
            },
            'weather': {
                'station_path': self.station_path,
                'calculate_fwi': self.calculate_fwi,
                'fwi_latitude': self.fwi_latitude,
            },
            'fbp': {
                'defaults': {
                    'ffmc': self.default_ffmc,
                    'dmc': self.default_dmc,
                    'dc': self.default_dc,
                    'isi': self.default_isi,
                    'bui': self.default_bui,
                },
                'fmc': self.fmc,
                'curing': self.curing,
            },
            'simulation': {
                'dt': self.dt,
                'max_duration': self.max_duration,
                'n_vertices': self.n_vertices,
                'initial_radius': self.initial_radius,
                'min_ros': self.min_ros,
                'time_varying_weather': self.time_varying_weather,
                'start_datetime': self.ignition_date,
            },
            'output': {
                'save_perimeters': self.save_perimeters,
                'save_ros_grids': self.save_ros_grids,
                'perimeter_format': self.perimeter_format,
                'generate_plots': self.generate_plots,
            },
        }
        return config
