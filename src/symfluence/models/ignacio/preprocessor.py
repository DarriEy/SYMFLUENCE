"""
IGNACIO PreProcessor for SYMFLUENCE

Prepares input data for IGNACIO fire spread simulations including:
- Terrain data (DEM, slope, aspect)
- Fuel type rasters
- Weather station data
- Ignition point configuration
- IGNACIO YAML configuration file generation
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from symfluence.models.registry import ModelRegistry
from symfluence.models.base.base_preprocessor import BaseModelPreProcessor

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor('IGNACIO')
class IGNACIOPreProcessor(BaseModelPreProcessor):
    """
    Preprocessor for IGNACIO fire spread model.

    Handles preparation of input data and configuration for IGNACIO simulations.
    This includes generating the YAML configuration file that IGNACIO expects.
    """

    def __init__(self, config, logger_instance=None, reporting_manager=None):
        """
        Initialize the IGNACIO preprocessor.

        Args:
            config: SymfluenceConfig object with domain and model settings
            logger_instance: Optional logger for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger_instance or logger)

        # Setup IGNACIO-specific paths
        self.ignacio_input_dir = self.project_dir / "IGNACIO_input"
        self.ignacio_config_path = self.ignacio_input_dir / "ignacio_config.yaml"

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "IGNACIO"

    def run_preprocessing(self, **kwargs) -> bool:
        """
        Execute IGNACIO preprocessing.

        Creates the IGNACIO input directory and generates the configuration
        YAML file from SYMFLUENCE configuration.

        Returns:
            True if preprocessing completed successfully
        """
        self.logger.info("Running IGNACIO preprocessing...")

        try:
            # Create input directories
            self.ignacio_input_dir.mkdir(parents=True, exist_ok=True)

            # Get IGNACIO config from SYMFLUENCE config
            ignacio_config = self._get_ignacio_config()

            # Prepare terrain data
            terrain_paths = self._prepare_terrain_data(ignacio_config)

            # Prepare fuel data
            fuel_path = self._prepare_fuel_data(ignacio_config)

            # Prepare weather data
            weather_path = self._prepare_weather_data(ignacio_config)

            # Prepare ignition data
            ignition_path = self._prepare_ignition_data(ignacio_config)

            # Generate IGNACIO YAML config
            self._generate_ignacio_config(
                ignacio_config,
                terrain_paths,
                fuel_path,
                weather_path,
                ignition_path
            )

            self.logger.info(f"IGNACIO preprocessing complete. Config: {self.ignacio_config_path}")
            return True

        except Exception as e:
            self.logger.error(f"IGNACIO preprocessing failed: {e}")
            return False

    def _get_ignacio_config(self) -> Dict[str, Any]:
        """Extract IGNACIO configuration from SYMFLUENCE config."""
        config_dict = {}

        # Try to get from typed config
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'ignacio'):
            ignacio = self.config.model.ignacio
            if ignacio is not None:
                # Convert Pydantic model to dict
                config_dict = ignacio.model_dump() if hasattr(ignacio, 'model_dump') else dict(ignacio)

        # Also check for IGNACIO_ prefixed keys in config_dict
        for key, value in self.config_dict.items():
            if key.startswith('IGNACIO_'):
                # Convert to lowercase without prefix for internal use
                internal_key = key[8:].lower()
                if internal_key not in config_dict:
                    config_dict[internal_key] = value

        # Set defaults
        config_dict.setdefault('project_name', self.config.domain.name)
        config_dict.setdefault('output_dir', str(self.project_dir / 'simulations' /
                                                   self.config.domain.experiment_id / 'IGNACIO'))

        return config_dict

    def _prepare_terrain_data(self, ignacio_config: Dict) -> Dict[str, Optional[Path]]:
        """
        Prepare terrain data (DEM, slope, aspect).

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Dictionary with paths to terrain files
        """
        terrain_paths: dict[str, Path | None] = {
            'dem_path': None,
            'slope_path': None,
            'aspect_path': None,
        }

        # Check if DEM path is specified
        dem_path = ignacio_config.get('dem_path')
        if dem_path and dem_path != 'default':
            dem_path = Path(dem_path)
            if dem_path.exists():
                terrain_paths['dem_path'] = dem_path
                self.logger.info(f"Using DEM: {dem_path}")

        # Try default DEM location
        if terrain_paths['dem_path'] is None:
            default_dem = self.project_dir / 'attributes' / 'dem' / f"{self.config.domain.name}_dem.tif"
            if default_dem.exists():
                terrain_paths['dem_path'] = default_dem
                self.logger.info(f"Using default DEM: {default_dem}")
            else:
                # Try alternative location
                alt_dem = self.project_dir / 'shapefiles' / 'dem' / 'dem.tif'
                if alt_dem.exists():
                    terrain_paths['dem_path'] = alt_dem

        # Check for pre-computed slope/aspect
        slope_path = ignacio_config.get('slope_path')
        if slope_path and Path(slope_path).exists():
            terrain_paths['slope_path'] = Path(slope_path)

        aspect_path = ignacio_config.get('aspect_path')
        if aspect_path and Path(aspect_path).exists():
            terrain_paths['aspect_path'] = Path(aspect_path)

        return terrain_paths

    def _prepare_fuel_data(self, ignacio_config: Dict) -> Optional[Path]:
        """
        Prepare fuel type raster data.

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Path to fuel raster or None
        """
        fuel_path = ignacio_config.get('fuel_path')
        if fuel_path and fuel_path != 'default':
            fuel_path = Path(fuel_path)
            if fuel_path.exists():
                self.logger.info(f"Using fuel raster: {fuel_path}")
                return fuel_path

        # Try default locations
        for default_path in [
            self.project_dir / 'attributes' / 'fuels' / 'fuels.tif',
            self.project_dir / 'shapefiles' / 'fuels' / 'fuels.tif',
        ]:
            if default_path.exists():
                self.logger.info(f"Using default fuel raster: {default_path}")
                return default_path

        self.logger.warning("No fuel raster found. IGNACIO will use default fuel type.")
        return None

    def _prepare_weather_data(self, ignacio_config: Dict) -> Optional[Path]:
        """
        Prepare weather station data for FWI calculation.

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Path to weather CSV or None
        """
        station_path = ignacio_config.get('station_path')
        if station_path and station_path != 'default':
            station_path = Path(station_path)
            if station_path.exists():
                self.logger.info(f"Using weather station data: {station_path}")
                return station_path

        # Check if we can generate from forcing data
        forcing_dir = self.project_dir / 'forcing' / 'processed'
        if forcing_dir.exists():
            self.logger.info("Weather data will be derived from forcing data during simulation")

        return None

    def _prepare_ignition_data(self, ignacio_config: Dict) -> Optional[Path]:
        """
        Prepare ignition point shapefile.

        Args:
            ignacio_config: IGNACIO configuration dictionary

        Returns:
            Path to ignition shapefile or None
        """
        # Check IGNACIO config
        ignition_path = ignacio_config.get('ignition_shapefile')
        if ignition_path and ignition_path != 'default':
            ignition_path = Path(ignition_path)
            if ignition_path.exists():
                self.logger.info(f"Using ignition shapefile: {ignition_path}")
                return ignition_path

        # Check WMFire config for shared ignition
        wmfire_ignition = self.config_dict.get('WMFIRE_IGNITION_SHAPEFILE')
        if wmfire_ignition:
            wmfire_path = Path(wmfire_ignition)
            if wmfire_path.exists():
                self.logger.info(f"Using WMFire ignition shapefile: {wmfire_path}")
                return wmfire_path

        # Check default location
        ignition_dir = self.project_dir / 'shapefiles' / 'ignitions'
        if ignition_dir.exists():
            shapefiles = list(ignition_dir.glob('*.shp'))
            if shapefiles:
                self.logger.info(f"Using default ignition shapefile: {shapefiles[0]}")
                return shapefiles[0]

        self.logger.warning("No ignition shapefile found")
        return None

    def _generate_ignacio_config(
        self,
        ignacio_config: Dict,
        terrain_paths: Dict[str, Optional[Path]],
        fuel_path: Optional[Path],
        weather_path: Optional[Path],
        ignition_path: Optional[Path]
    ) -> None:
        """
        Generate IGNACIO YAML configuration file.

        Args:
            ignacio_config: IGNACIO configuration dictionary
            terrain_paths: Dictionary with terrain file paths
            fuel_path: Path to fuel raster
            weather_path: Path to weather CSV
            ignition_path: Path to ignition shapefile
        """
        # Build IGNACIO config structure
        config = {
            'project': {
                'name': ignacio_config.get('project_name', self.config.domain.name),
                'description': f"IGNACIO simulation for {self.config.domain.name}",
                'output_dir': ignacio_config.get('output_dir', './output'),
                'random_seed': ignacio_config.get('random_seed', 42),
            },
            'crs': {
                'working_crs': ignacio_config.get('working_crs', 'EPSG:4326'),
                'output_crs': ignacio_config.get('output_crs', 'EPSG:4326'),
            },
            'terrain': {
                'dem_path': str(terrain_paths['dem_path']) if terrain_paths['dem_path'] else None,
                'slope_path': str(terrain_paths['slope_path']) if terrain_paths['slope_path'] else None,
                'aspect_path': str(terrain_paths['aspect_path']) if terrain_paths['aspect_path'] else None,
            },
            'fuel': {
                'source_type': 'raster' if fuel_path else 'constant',
                'path': str(fuel_path) if fuel_path else None,
                'non_fuel_codes': ignacio_config.get('non_fuel_codes', [0, 100, 101, 102, -9999]),
            },
            'ignition': {
                'source_type': 'shapefile' if ignition_path else 'point',
                'point_path': str(ignition_path) if ignition_path else None,
                'cause': ignacio_config.get('ignition_cause', 'Lightning'),
                'n_iterations': ignacio_config.get('n_iterations', 1),
            },
            'weather': {
                'station_path': str(weather_path) if weather_path else None,
                'calculate_fwi': ignacio_config.get('calculate_fwi', True),
            },
            'fbp': {
                'defaults': {
                    'ffmc': ignacio_config.get('default_ffmc', 88.0),
                    'dmc': ignacio_config.get('default_dmc', 30.0),
                    'dc': ignacio_config.get('default_dc', 150.0),
                    'isi': ignacio_config.get('default_isi', 5.0),
                    'bui': ignacio_config.get('default_bui', 50.0),
                },
                'fmc': ignacio_config.get('fmc', 100.0),
                'curing': ignacio_config.get('curing', 85.0),
            },
            'simulation': {
                'dt': ignacio_config.get('dt', 1.0),
                'max_duration': ignacio_config.get('max_duration', 480),
                'n_vertices': ignacio_config.get('n_vertices', 300),
                'initial_radius': ignacio_config.get('initial_radius', 10.0),
                'min_ros': ignacio_config.get('min_ros', 0.01),
                'time_varying_weather': ignacio_config.get('time_varying_weather', True),
                'start_datetime': ignacio_config.get('ignition_date'),
            },
            'output': {
                'save_perimeters': ignacio_config.get('save_perimeters', True),
                'save_ros_grids': ignacio_config.get('save_ros_grids', True),
                'perimeter_format': ignacio_config.get('perimeter_format', 'shapefile'),
                'generate_plots': ignacio_config.get('generate_plots', True),
                'log_level': 'INFO',
            },
        }

        # Write YAML config
        with open(self.ignacio_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"IGNACIO config written: {self.ignacio_config_path}")
