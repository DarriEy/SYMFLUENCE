"""
SWAT Model Preprocessor

Handles preparation of SWAT model inputs including:
- TxtInOut directory structure
- Forcing files (.pcp and .tmp) from ERA5 NetCDF data
- Basin file (.bsn) with default snow/surface parameters
- file.cio master control file

The heavy lifting is delegated to sub-module generators:
- SWATForcingGenerator: .pcp and .tmp forcing files
- SWATBasinGenerator: .bsn, .wgn, .pnd, .wus, .chm files
- SWATSubbasinGenerator: .sub, .hru, .gw, .mgt, .sol files
- SWATRoutingGenerator: file.cio, fig.fig, .rte, .swq, database stubs
"""
import logging
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("SWAT")
class SWATPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a SWAT model run.

    SWAT requires a TxtInOut directory containing:
    - file.cio: Master control file
    - .pcp files: Precipitation data
    - .tmp files: Temperature data (min/max)
    - .bsn: Basin-level parameters
    - .sub: Sub-basin files
    - .hru: HRU files
    - .gw: Groundwater files
    - .mgt: Management files
    - .sol: Soil files
    """


    MODEL_NAME = "SWAT"
    def __init__(self, config, logger):
        """
        Initialize the SWAT preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Standard paths (from base class):
        #   self.setup_dir   = project_dir / settings / SWAT
        #   self.forcing_dir = project_dir / data / forcing / SWAT_input
        # Settings files (file.cio, .bsn, .sub, etc.) go to setup_dir.
        # Forcing files (.pcp, .tmp) go to forcing_dir.
        # The runner assembles both into a TxtInOut for execution.
        self.txtinout_dir = self.setup_dir

        # Lazy-init backing fields for sub-module generators
        self._forcing_generator = None
        self._basin_generator = None
        self._subbasin_generator = None
        self._routing_generator = None

    # ------------------------------------------------------------------
    # Lazy-init properties for sub-module generators
    # ------------------------------------------------------------------

    @property
    def forcing_generator(self):
        """Lazy-init SWATForcingGenerator."""
        if self._forcing_generator is None:
            from .forcing_generator import SWATForcingGenerator
            self._forcing_generator = SWATForcingGenerator(self)
        return self._forcing_generator

    @property
    def basin_generator(self):
        """Lazy-init SWATBasinGenerator."""
        if self._basin_generator is None:
            from .basin_generator import SWATBasinGenerator
            self._basin_generator = SWATBasinGenerator(self)
        return self._basin_generator

    @property
    def subbasin_generator(self):
        """Lazy-init SWATSubbasinGenerator."""
        if self._subbasin_generator is None:
            from .subbasin_generator import SWATSubbasinGenerator
            self._subbasin_generator = SWATSubbasinGenerator(self)
        return self._subbasin_generator

    @property
    def routing_generator(self):
        """Lazy-init SWATRoutingGenerator."""
        if self._routing_generator is None:
            from .routing_generator import SWATRoutingGenerator
            self._routing_generator = SWATRoutingGenerator(self)
        return self._routing_generator

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_preprocessing(self) -> bool:
        """
        Run the complete SWAT preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting SWAT preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Get simulation dates
            start_date, end_date = self._get_simulation_dates()

            # Generate forcing files from ERA5
            self.forcing_generator.generate_forcing_files(start_date, end_date)

            # Generate basin file
            self.basin_generator.generate_basin_file()

            # Generate sub-basin, HRU, groundwater, management, and soil files
            self.subbasin_generator.generate_subbasin_files()

            # Generate watershed routing file (fig.fig) and reach files
            self.routing_generator.generate_fig_file()
            self.routing_generator.generate_route_files()

            # Generate minimal database stub files
            self.routing_generator.generate_database_stubs()

            # Generate file.cio (must be last -- references all other files)
            self.routing_generator.generate_file_cio(start_date, end_date)

            logger.info("SWAT preprocessing complete.")
            return True

        except Exception as e:
            logger.error(f"SWAT preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Helper methods (stay on orchestrator, used by sub-modules via self.pp)
    # ------------------------------------------------------------------

    def _create_directory_structure(self) -> None:
        """Create SWAT directory structure (settings + forcing)."""
        self.setup_dir.mkdir(parents=True, exist_ok=True)
        self.forcing_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created SWAT settings directory at {self.setup_dir}")
        logger.info(f"Created SWAT forcing directory at {self.forcing_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from configuration."""
        start_str = self._get_config_value(lambda: self.config.domain.time_start)
        end_str = self._get_config_value(lambda: self.config.domain.time_end)

        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def _get_catchment_properties(self) -> Dict:
        """
        Get catchment properties from shapefile.

        Returns:
            Dict with centroid lat/lon, area, and elevation
        """
        try:
            import geopandas as gpd
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)

                # Get centroid
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y

                # Project to UTM for accurate area
                utm_zone = int((lon + 180) / 6) + 1
                hemisphere = 'north' if lat >= 0 else 'south'
                utm_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
                gdf_proj = gdf.to_crs(utm_crs)
                area_m2 = gdf_proj.geometry.area.sum()

                elev = float(gdf.get('elev_mean', [1000])[0]) if 'elev_mean' in gdf.columns else 1000.0

                return {
                    'lat': lat,
                    'lon': lon,
                    'area_m2': area_m2,
                    'area_km2': area_m2 / 1e6,
                    'elev': elev
                }
        except Exception as e:
            logger.warning(f"Could not read catchment properties: {e}")

        return {
            'lat': 51.0,
            'lon': -115.0,
            'area_m2': 1e8,
            'area_km2': 100.0,
            'elev': 1000.0
        }

    def _load_forcing_data(self):
        """Load basin-averaged forcing data from ERA5 NetCDF files."""
        import xarray as xr

        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

        if not forcing_files:
            merged_path = self.project_forcing_dir / 'merged_path'
            if merged_path.exists():
                forcing_files = list(merged_path.glob("*.nc"))

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in {self.forcing_basin_path}")

        logger.info(f"Loading forcing from {len(forcing_files)} files")

        try:
            ds = xr.open_mfdataset(forcing_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')
        except ValueError:
            try:
                ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override')
            except Exception:
                datasets = [xr.open_dataset(f) for f in forcing_files]
                ds = xr.merge(datasets)

        ds = self.subset_to_simulation_time(ds, "Forcing")
        return ds

    def _extract_variable(self, ds, candidates, default_val=0.0):
        """Extract a variable from dataset by trying multiple candidate names."""
        for candidate in candidates:
            if candidate in ds:
                data = ds[candidate].values
                # Average over spatial dims if present
                while data.ndim > 1:
                    data = np.nanmean(data, axis=-1)
                return data, candidate
        return None, None

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
