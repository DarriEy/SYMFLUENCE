"""
Base class for geofabric delineators.

Provides shared infrastructure for all delineation modules including:
- Configuration management
- Path resolution with default fallbacks
- Directory creation
- Common initialization patterns
- Logger integration

Following the BaseModelPreProcessor pattern from model refactoring.

Refactored from geofabric_utils.py (2026-01-01)
"""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


from symfluence.utils.common.path_resolver import PathResolverMixin


class BaseGeofabricDelineator(ABC, PathResolverMixin):
    """
    Abstract base class for all geofabric delineators.

    Provides common initialization, path management, and utility methods
    that are shared across different geofabric delineation strategies.

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        data_dir: Root data directory
        domain_name: Name of the domain
        project_dir: Project-specific directory
        mpi_processes: Number of MPI processes for TauDEM
        max_retries: Maximum number of command retries
        retry_delay: Delay between retries (seconds)
        min_gru_size: Minimum GRU size (km²)
        taudem_dir: Path to TauDEM binary directory
        dem_path: Path to DEM file
        interim_dir: Directory for interim TauDEM files
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize base delineator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        # Core configuration
        self.config = config
        self.logger = logger

        # Base paths
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Common configuration
        self.mpi_processes = self.config.get('MPI_PROCESSES', 1)
        self.max_retries = self.config.get('MAX_RETRIES', 3)
        self.retry_delay = self.config.get('RETRY_DELAY', 5)
        self.min_gru_size = self.config.get('MIN_GRU_SIZE', 0)

        # TauDEM configuration
        self.taudem_dir = self._get_taudem_dir()
        self._set_taudem_path()

        # DEM path
        self.dem_path = self._get_dem_path()

        # Interim directory (subclasses can override)
        self.interim_dir = self.project_dir / "taudem-interim-files"

    def _get_taudem_dir(self) -> str:
        """
        Get TauDEM installation directory.

        Returns:
            Path to TauDEM bin directory
        """
        taudem_dir = self.config.get('TAUDEM_DIR')
        if taudem_dir == "default":
            return str(self.data_dir / 'installs' / 'TauDEM' / 'bin')
        return taudem_dir

    def _set_taudem_path(self):
        """Add TauDEM directory to system PATH."""
        os.environ['PATH'] = f"{os.environ['PATH']}:{self.taudem_dir}"

    def _get_dem_path(self) -> Path:
        """
        Get DEM file path with default handling.

        Returns:
            Path to DEM file
        """
        dem_path = self.config.get('DEM_PATH')
        dem_name = self.config.get('DEM_NAME')

        if dem_name == "default":
            dem_name = f"domain_{self.domain_name}_elv.tif"

        if dem_path == 'default':
            return self.project_dir / 'attributes' / 'elevation' / 'dem' / dem_name

        return Path(dem_path) / dem_name

    def _get_pour_point_path(self) -> Path:
        """
        Get pour point shapefile path.

        Returns:
            Path to pour point shapefile
        """
        pour_point_path = self.config.get('POUR_POINT_SHP_PATH')
        if pour_point_path == 'default':
            pour_point_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pour_point_path = Path(pour_point_path)

        pour_point_name = self.config.get('POUR_POINT_SHP_NAME', "default")
        if pour_point_name == "default":
            pour_point_path = pour_point_path / f"{self.domain_name}_pourPoint.shp"
        else:
            pour_point_path = pour_point_path / pour_point_name

        return pour_point_path

    def _validate_inputs(self):
        """
        Validate required input files exist.

        Raises:
            FileNotFoundError: If DEM file doesn't exist
        """
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")

    def create_directories(self):
        """Create necessary directories for delineation."""
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created interim directory: {self.interim_dir}")

    def cleanup(self):
        """
        Clean up interim files after processing.

        Only removes files if CLEANUP_INTERMEDIATE_FILES is True.
        """
        if self.config.get('CLEANUP_INTERMEDIATE_FILES', True):
            if hasattr(self, 'interim_dir') and self.interim_dir.exists():
                shutil.rmtree(self.interim_dir.parent, ignore_errors=True)
                self.logger.info(f"Cleaned up interim files: {self.interim_dir.parent}")

    @abstractmethod
    def _get_delineation_method_name(self) -> str:
        """
        Return the delineation method name for output files.

        This is used to construct output filenames like:
        - {domain_name}_riverBasins_{method}.shp
        - {domain_name}_riverNetwork_{method}.shp

        Returns:
            Method name string (e.g., 'delineate', 'lumped', 'subset_MERIT')
        """
        pass
