"""Machine learning and fire-model configuration classes."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG


class LSTMConfig(BaseModel):
    """LSTM neural network emulator configuration"""
    model_config = FROZEN_CONFIG

    load: bool = Field(default=False, alias='LSTM_LOAD')
    hidden_size: int = Field(default=128, alias='LSTM_HIDDEN_SIZE', ge=8, le=2048)
    num_layers: int = Field(default=3, alias='LSTM_NUM_LAYERS', ge=1, le=10)
    epochs: int = Field(default=300, alias='LSTM_EPOCHS', ge=1, le=10000)
    batch_size: int = Field(default=64, alias='LSTM_BATCH_SIZE', ge=1, le=4096)
    learning_rate: float = Field(default=0.001, alias='LSTM_LEARNING_RATE', gt=0, le=1.0)
    learning_patience: int = Field(default=30, alias='LSTM_LEARNING_PATIENCE', ge=1)
    lookback: int = Field(default=700, alias='LSTM_LOOKBACK', ge=1)
    dropout: float = Field(default=0.2, alias='LSTM_DROPOUT', ge=0, le=0.9)
    l2_regularization: float = Field(default=1e-6, alias='LSTM_L2_REGULARIZATION', ge=0)
    use_attention: bool = Field(default=True, alias='LSTM_USE_ATTENTION')
    use_snow: bool = Field(default=False, alias='LSTM_USE_SNOW')
    train_through_routing: bool = Field(default=False, alias='LSTM_TRAIN_THROUGH_ROUTING')


class WMFireConfig(BaseModel):
    """WMFire wildfire spread module configuration for RHESSys.

    WMFire is a fire spread model that couples with RHESSys to simulate
    wildfire spread based on fuel loads, moisture, wind, and topography.

    Reference:
        Kennedy, M.C., McKenzie, D., Tague, C., Dugger, A.L. 2017.
        Balancing uncertainty and complexity to incorporate fire spread in
        an eco-hydrological model. International Journal of Wildland Fire.
    """
    model_config = FROZEN_CONFIG

    # Grid resolution and timestep
    grid_resolution: int = Field(
        default=30,
        alias='WMFIRE_GRID_RESOLUTION',
        ge=10,
        le=200,
        description='Fire grid cell resolution in meters (30, 60, or 90 recommended)'
    )
    timestep_hours: int = Field(
        default=24,
        alias='WMFIRE_TIMESTEP_HOURS',
        ge=1,
        le=24,
        description='Fire spread timestep in hours (1-24)'
    )

    # Fuel and moisture configuration
    ndays_average: float = Field(
        default=30.0,
        alias='WMFIRE_NDAYS_AVERAGE',
        ge=1.0,
        le=365.0,
        description='Fuel moisture averaging window in days'
    )
    fuel_source: Literal['static', 'rhessys_litter'] = Field(
        default='static',
        alias='WMFIRE_FUEL_SOURCE',
        description='Source of fuel load data: static values or RHESSys litter pools'
    )
    moisture_source: Literal['static', 'rhessys_soil'] = Field(
        default='static',
        alias='WMFIRE_MOISTURE_SOURCE',
        description='Source of moisture data: static values or RHESSys soil moisture'
    )
    carbon_to_fuel_ratio: float = Field(
        default=2.0,
        alias='WMFIRE_CARBON_TO_FUEL_RATIO',
        ge=1.0,
        le=5.0,
        description='Conversion factor from kg carbon to kg fuel'
    )

    # Ignition configuration
    ignition_shapefile: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_SHAPEFILE',
        description='Path to ignition point shapefile (overrides ignition_point if set)'
    )
    ignition_point: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_POINT',
        description='Ignition point as "lat/lon" (e.g., "51.2096/-115.7539")'
    )
    ignition_date: Optional[str] = Field(
        default=None,
        alias='WMFIRE_IGNITION_DATE',
        description='Ignition date as "YYYY-MM-DD" for fire simulation start'
    )
    ignition_name: Optional[str] = Field(
        default='ignition',
        alias='WMFIRE_IGNITION_NAME',
        description='Name for the ignition point (used in output shapefile)'
    )

    # Fire perimeter validation
    perimeter_shapefile: Optional[str] = Field(
        default=None,
        alias='WMFIRE_PERIMETER_SHAPEFILE',
        description='Path to observed fire perimeter shapefile for validation'
    )
    perimeter_dir: Optional[str] = Field(
        default=None,
        alias='WMFIRE_PERIMETER_DIR',
        description='Directory containing fire perimeter shapefiles for comparison'
    )

    # Output options
    write_geotiff: bool = Field(
        default=True,
        alias='WMFIRE_WRITE_GEOTIFF',
        description='Write georeferenced GeoTIFF outputs for visualization'
    )

    # Optional coefficient overrides (None = use defaults from fire.def)
    load_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_LOAD_K1',
        description='Fuel load coefficient k1 (default 3.9)'
    )
    load_k2: Optional[float] = Field(
        default=None,
        alias='WMFIRE_LOAD_K2',
        description='Fuel load coefficient k2 (default 0.07)'
    )
    moisture_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MOISTURE_K1',
        description='Moisture coefficient k1 (default 3.8)'
    )
    moisture_k2: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MOISTURE_K2',
        description='Moisture coefficient k2 (default 0.27)'
    )
    ign_def_mod: Optional[float] = Field(
        default=None,
        alias='WMFIRE_IGN_DEF_MOD',
        description='Ignition probability modifier (default 1.0, increase for more fires)'
    )
    mean_ign: Optional[float] = Field(
        default=None,
        alias='WMFIRE_MEAN_IGN',
        description='Mean ignition events per timestep (default 1.0)'
    )
    windmax: Optional[float] = Field(
        default=None,
        alias='WMFIRE_WINDMAX',
        description='Maximum wind speed multiplier (default 1.0)'
    )
    slope_k1: Optional[float] = Field(
        default=None,
        alias='WMFIRE_SLOPE_K1',
        description='Slope effect coefficient k1 (default 0.91)'
    )

    @field_validator('grid_resolution')
    @classmethod
    def validate_resolution(cls, v):
        """Validate grid resolution is reasonable."""
        recommended = [30, 60, 90]
        if v not in recommended:
            import warnings
            warnings.warn(
                f"Grid resolution {v}m is not standard. "
                f"Recommended values: {recommended}"
            )
        return v

    @field_validator('ignition_point')
    @classmethod
    def validate_ignition_point(cls, v):
        """Validate ignition point format."""
        if v is not None:
            parts = v.split('/')
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid ignition_point format: {v}. "
                    f"Expected 'lat/lon' (e.g., '51.2096/-115.7539')"
                )
            try:
                lat, lon = float(parts[0]), float(parts[1])
                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude {lat} out of range [-90, 90]")
                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude {lon} out of range [-180, 180]")
            except ValueError as e:
                raise ValueError(f"Invalid ignition_point coordinates: {e}")
        return v



class GNNConfig(BaseModel):
    """GNN (Graph Neural Network) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    load: bool = Field(default=False, alias='GNN_LOAD')
    hidden_size: int = Field(default=128, alias='GNN_HIDDEN_SIZE', ge=8, le=2048)
    num_layers: int = Field(default=3, alias='GNN_NUM_LAYERS', ge=1, le=10)
    epochs: int = Field(default=300, alias='GNN_EPOCHS', ge=1, le=10000)
    batch_size: int = Field(default=64, alias='GNN_BATCH_SIZE', ge=1, le=4096)
    learning_rate: float = Field(default=0.001, alias='GNN_LEARNING_RATE', gt=0, le=1.0)
    learning_patience: int = Field(default=30, alias='GNN_LEARNING_PATIENCE', ge=1)
    dropout: float = Field(default=0.2, alias='GNN_DROPOUT', ge=0, le=0.9)
    l2_regularization: float = Field(default=1e-6, alias='GNN_L2_REGULARIZATION', ge=0)
    params_to_calibrate: str = Field(
        default='precip_mult,temp_offset,routing_velocity',
        alias='GNN_PARAMS_TO_CALIBRATE'
    )
    parameter_bounds: Optional[Dict[str, List[float]]] = Field(default=None, alias='GNN_PARAMETER_BOUNDS')


class IGNACIOConfig(BaseModel):
    """IGNACIO fire spread model configuration.

    IGNACIO implements the Canadian Forest Fire Behavior Prediction (FBP) System
    with Richards' elliptical wave propagation for fire spread modeling.

    This is a standalone fire model that can be run independently or compared
    with WMFire results for validation.

    Reference:
        IGNACIO: https://github.com/KatherineHopeReece/Fire-Engine-Framework
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

    # Terrain inputs
    dem_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_DEM_PATH',
        description='Path to DEM raster file'
    )

    # Fuel inputs
    fuel_path: Optional[str] = Field(
        default=None,
        alias='IGNACIO_FUEL_PATH',
        description='Path to fuel type raster'
    )
    default_fuel_type: str = Field(
        default='C-2',
        alias='IGNACIO_DEFAULT_FUEL',
        description='Default FBP fuel type code'
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

    # Output configuration
    save_perimeters: bool = Field(
        default=True,
        alias='IGNACIO_SAVE_PERIMETERS',
        description='Save fire perimeter shapefiles'
    )

    # Comparison with WMFire
    compare_with_wmfire: bool = Field(
        default=False,
        alias='IGNACIO_COMPARE_WMFIRE',
        description='Compare fire perimeters with WMFire results'
    )

    @field_validator('ignition_date')
    @classmethod
    def validate_ignition_date(cls, v: Optional[str]) -> Optional[str]:
        """Validate ignition date format."""
        if v is not None:
            from datetime import datetime
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




__all__ = [
    'LSTMConfig',
    'WMFireConfig',
    'GNNConfig',
    'IGNACIOConfig',
]
