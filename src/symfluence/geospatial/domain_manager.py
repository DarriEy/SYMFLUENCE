"""
Domain management facade for SYMFLUENCE geospatial operations.

Coordinates domain definition, delineation, and discretization workflows
with integrated visualization and artifact tracking.
"""

from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, Union, TYPE_CHECKING

from symfluence.geospatial.discretization import DomainDiscretizationRunner, DiscretizationArtifacts # type: ignore
from symfluence.geospatial.delineation import DomainDelineator, create_point_domain_shapefile, DelineationArtifacts # type: ignore

from symfluence.core.mixins import ConfigurableMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class DomainManager(ConfigurableMixin):
    """Manages all domain-related operations including definition, discretization, and visualization."""
    
    def __init__(self, config: 'SymfluenceConfig', logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the Domain Manager.

        Args:
            config: SymfluenceConfig instance
            logger: Logger instance
            reporting_manager: ReportingManager instance

        Raises:
            TypeError: If config is not a SymfluenceConfig instance
        """
        # Import here to avoid circular imports at module level
        from symfluence.core.config.models import SymfluenceConfig

        if not isinstance(config, SymfluenceConfig):
            raise TypeError(
                f"config must be SymfluenceConfig, got {type(config).__name__}. "
                "Use SymfluenceConfig.from_file() to load configuration."
            )

        # Set config via the ConfigMixin property
        self._config = config
        self.logger = logger
        self.reporting_manager = reporting_manager

        # Initialize domain workflows
        self.domain_delineator = DomainDelineator(config, self.logger, self.reporting_manager)
        self.domain_discretizer = None  # Initialized when needed
        self.delineation_artifacts: Optional[DelineationArtifacts] = None
        self.discretization_artifacts: Optional[DiscretizationArtifacts] = None
        
        # Create point domain shapefile if method is 'point'
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method
        )
        if domain_method == 'point':
            self.create_point_domain_shapefile()

    def create_point_domain_shapefile(self) -> Optional[Path]:
        """
        Create a square basin shapefile from bounding box coordinates for point modelling.

        This method creates a rectangular polygon from the BOUNDING_BOX_COORDS and saves it
        as a shapefile for point-based modelling approaches.

        Returns:
            Path to the created shapefile or None if failed
        """
        return create_point_domain_shapefile(self.config, self.logger)
    
    def define_domain(
        self,
    ) -> Tuple[Optional[Union[Path, Tuple[Path, Path]]], DelineationArtifacts]:
        """
        Define the domain using the configured method.
        
        Returns:
            Tuple of the domain result and delineation artifacts
        """
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method
        )
        self.logger.debug(f"Domain definition workflow starting with: {domain_method}")
        
        result, artifacts = self.domain_delineator.define_domain()
        self.delineation_artifacts = artifacts
        
        if result:
            self.logger.info(f"Domain definition completed using method: {domain_method}")
        
        self.logger.debug(f"Domain definition workflow finished")

        return result, artifacts
    

    def discretize_domain(
        self,
    ) -> Tuple[Optional[Union[Path, dict]], DiscretizationArtifacts]:
        """
        Discretize the domain into HRUs or GRUs.
        
        Returns:
            Tuple of HRU shapefile(s) and discretization artifacts
        """
        try:
            discretization_method = self._get_config_value(
                lambda: self.config.domain.discretization
            )
            self.logger.debug(f"Discretizing domain using method: {discretization_method}")

            # Initialize discretizer if not already done
            if self.domain_discretizer is None:
                self.domain_discretizer = DomainDiscretizationRunner(self.config, self.logger)
            
            # Perform discretization
            hru_shapefile, artifacts = self.domain_discretizer.discretize_domain()
            self.discretization_artifacts = artifacts
                        
            # Visualize the discretized domain
            self.visualize_discretized_domain()
            
            return hru_shapefile, artifacts
            
        except Exception as e:
            self.logger.error(f"Error during domain discretization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def visualize_domain(self) -> Optional[Path]:
        """
        Create visualization of the domain.
        
        Returns:
            Path to the created plot or None if failed
        """
        if self.reporting_manager:
            return self.reporting_manager.visualize_domain()
        return None
    
    def visualize_discretized_domain(self) -> Optional[Path]:
        """
        Create visualization of the discretized domain.
        
        Returns:
            Path to the created plot or None if failed
        """
        if self.reporting_manager:
            discretization_method = self._get_config_value(
                lambda: self.config.domain.discretization
            )
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method
            )
            if domain_method != 'point':
                return self.reporting_manager.visualize_discretized_domain(discretization_method)
            else:
                self.logger.info('Point scale model, not creating visualisation')
                return None
        return None
    
    def get_domain_info(self) -> Dict[str, Any]:
        """
        Get information about the current domain configuration.
        
        Returns:
            Dictionary containing domain information
        """
        info = {
            'domain_name': self.domain_name,
            'domain_method': self._get_config_value(
                lambda: self.config.domain.definition_method
            ),
            'spatial_mode': self._get_config_value(
                lambda: self.config.domain.definition_method
            ),
            'discretization_method': self._get_config_value(
                lambda: self.config.domain.discretization
            ),
            'pour_point_coords': self._get_config_value(
                lambda: self.config.domain.pour_point_coords
            ),
            'bounding_box': self._get_config_value(
                lambda: self.config.domain.bounding_box_coords
            ),
            'project_dir': str(self.project_dir),
        }
        
        # Add shapefile paths if they exist
        river_basins_path = self.project_dir / "shapefiles" / "river_basins"
        catchment_path = self.project_dir / "shapefiles" / "catchment"
        
        if river_basins_path.exists():
            info['river_basins_path'] = str(river_basins_path)
        
        if catchment_path.exists():
            info['catchment_path'] = str(catchment_path)
        
        return info
    
    def validate_domain_configuration(self) -> bool:
        """
        Validate the domain configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_settings = [
            ('DOMAIN_NAME', lambda: self.config.domain.name),
            ('DOMAIN_DEFINITION_METHOD', lambda: self.config.domain.definition_method),
            ('DOMAIN_DISCRETIZATION', lambda: self.config.domain.discretization),
            ('BOUNDING_BOX_COORDS', lambda: self.config.domain.bounding_box_coords)
        ]

        # Check required settings
        for setting_name, typed_accessor in required_settings:
            val = self._get_config_value(typed_accessor)
            if not val:
                self.logger.error(f"Required domain setting missing: {setting_name}")
                return False

        # Validate domain definition method
        valid_methods = ['subset', 'lumped', 'delineate', 'point']  # Added 'point' to valid methods
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method
        )
        if domain_method not in valid_methods:
            self.logger.error(f"Invalid domain definition method: {domain_method}. Must be one of {valid_methods}")
            return False
        
        # Validate bounding box format
        bbox = self._get_config_value(
            lambda: self.config.domain.bounding_box_coords,
            ''
        )
        bbox_parts = str(bbox).split('/')
        if len(bbox_parts) != 4:
            self.logger.error(f"Invalid bounding box format: {bbox}. Expected format: lat_max/lon_min/lat_min/lon_max")
            return False
        
        try:
            # Check if values are valid floats
            lat_max, lon_min, lat_min, lon_max = map(float, bbox_parts)
            
            # Basic validation of coordinates
            if lat_max <= lat_min:
                self.logger.error(f"Invalid bounding box: lat_max ({lat_max}) must be greater than lat_min ({lat_min})")
                return False
            if lon_max <= lon_min:
                self.logger.error(f"Invalid bounding box: lon_max ({lon_max}) must be greater than lon_min ({lon_min})")
                return False
                
        except ValueError:
            self.logger.error(f"Invalid bounding box values: {bbox}. All values must be numeric.")
            return False
        
        self.logger.info("Domain configuration validation passed")
        return True
