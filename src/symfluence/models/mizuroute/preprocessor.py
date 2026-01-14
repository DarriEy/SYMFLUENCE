"""
MizuRoute Model Preprocessor.

Handles spatial preprocessing and configuration generation for the mizuRoute routing model.
"""

import os
import pandas as pd
import netCDF4 as nc4
import geopandas as gpd
import numpy as np
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from typing import Any, Dict, List, Optional
import easymore
import xarray as xr

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelPreProcessor
from symfluence.geospatial.geometry_utils import GeospatialUtilsMixin
from symfluence.models.mizuroute.mixins import MizuRouteConfigMixin
from symfluence.models.mizuroute.control_writer import ControlFileWriter

def _create_easymore_instance():
    """Create an EASYMORE instance handling different module structures."""
    if hasattr(easymore, "Easymore"):
        return easymore.Easymore()
    if hasattr(easymore, "easymore"):
        return easymore.easymore()
    raise AttributeError("easymore module does not expose an Easymore class")


@ModelRegistry.register_preprocessor('MIZUROUTE')
class MizuRoutePreProcessor(BaseModelPreProcessor, GeospatialUtilsMixin, MizuRouteConfigMixin):
    """
    Spatial preprocessor and configuration generator for the mizuRoute river routing model.

    This preprocessor handles all spatial setup tasks required to run mizuRoute, including
    network topology file creation, remapping file generation, and control file writing.
    It supports multiple domain discretization strategies (lumped, semi-distributed,
    distributed, grid-based) and integrates with various hydrological models as runoff
    sources (SUMMA, FUSE, GR, NextGen, HYPE).

    Supported Domain Types:
        Lumped:
            - Single HRU draining to river network
            - Optional distributed routing via delineated subcatchments
            - Area-weighted remapping for lumped-to-distributed conversion

        Semi-distributed:
            - Multiple HRUs per GRU routing at GRU level
            - Reads SUMMA attributes file to determine HRU/GRU structure
            - GRU-aggregated runoff routing

        Distributed:
            - Elevation bands or attribute-based discretization
            - Routing at finest spatial resolution
            - Optional remapping between catchment scales

        Grid-based:
            - Regular grid cells with D8 flow direction
            - Each cell is both HRU and routing segment
            - Cycle detection and fixing via graph algorithms

    Supported Source Models:
        - SUMMA: Physics-based snow hydrology (HRU or GRU runoff)
        - FUSE: Framework for Understanding Structural Errors
        - GR: Parsimonious hydrological models (GR4J, GR5J, GR6J)
        - NextGen (NGEN): NOAA modular BMI framework
        - HYPE: Semi-distributed hydrological model

    Processing Workflow:
        1. **Initialization**: Set up directories, handle custom paths for parallel runs
        2. **Base Settings**: Copy template parameter and control files
        3. **Network Topology**: Create NetCDF topology file from river network shapefiles
           - Handle headwater basins (synthetic network generation)
           - Detect and fix routing cycles using DFS graph algorithms
           - Support lumped-to-distributed routing via delineated subcatchments
        4. **Remapping** (optional): Create NetCDF remapping file
           - Area-weighted remapping for lumped-to-distributed conversion
           - Equal-weight remapping for uniform distribution
           - Spatial intersection-based remapping for multi-scale modeling
        5. **Control File**: Generate model-specific control file
           - SUMMA control: HRU vs GRU runoff handling
           - FUSE control: Basin-scale runoff routing
           - GR control: Daily timestep alignment, midnight forcing

    Key Methods (36 total):
        Main Workflow:
            run_preprocessing(): Orchestrates all preprocessing steps
            copy_base_settings(): Copy template files to setup directory

        Topology Creation:
            create_network_topology_file(): Main topology file creation (208 lines)
            _create_grid_topology_file(): Grid-based distributed topology
            _check_if_headwater_basin(): Detect headwater basins with no river network
            _create_synthetic_river_network(): Generate single-segment network for headwaters
            _fix_routing_cycles(): Graph algorithm to detect and fix cycles (167 lines)
            _find_closest_segment_to_pour_point(): Locate segment nearest to basin outlet

        Remapping:
            create_area_weighted_remap_file(): Area-based weights from delineated catchments
            create_equal_weight_remap_file(): Uniform weights for all segments
            remap_summa_catchments_to_routing(): Spatial intersection remapping

        Control File Generation:
            create_control_file(): SUMMA-specific control file
            create_fuse_control_file(): FUSE-specific control file
            create_gr_control_file(): GR-specific control file
            _get_control_writer(): Get configured ControlFileWriter instance
            _get_mizu_config(): Extract mizuRoute configuration values

        NetCDF Helpers:
            _set_topology_attributes(): Set file metadata
            _create_topology_dimensions(): Create seg/hru dimensions
            _create_topology_variables(): Create and fill segment/HRU variables
            _create_and_fill_nc_var(): Generic NetCDF variable creation
            _create_remap_file(): Create remapping NetCDF file
            _create_remap_variables(): Fill remapping variables
            _process_remap_variables(): Process spatial intersection results

        Legacy Control File Writers (deprecated):
            _write_control_file_header(): Write control file header
            _write_control_file_directories(): Write directory paths
            _write_control_file_simulation_controls(): Write simulation times
            _write_control_file_topology(): Write topology configuration
            _write_control_file_remapping(): Write remapping configuration
            _write_control_file_parameters(): Write parameter file reference
            _write_control_file_miscellaneous(): Write miscellaneous settings

    Configuration Dependencies:
        Required:
            - DOMAIN_NAME: Basin identifier
            - DOMAIN_DISCRETIZATION: Domain definition method (lumped/TBL/distribute)
            - RIVER_NETWORK_SHP_PATH: Path to river network shapefile
            - RIVER_NETWORK_SHP_NAME: River network shapefile name
            - RIVER_BASINS_PATH: Path to river basin shapefile
            - RIVER_BASINS_NAME: River basin shapefile name
            - EXPERIMENT_ID: Experiment identifier
            - EXPERIMENT_OUTPUT_MIZUROUTE: mizuRoute output directory

        Optional:
            - SETTINGS_MIZU_PATH: Custom setup directory (for parallel runs)
            - SETTINGS_MIZU_TOPOLOGY: Topology file name (default: mizuRoute_topology.nc)
            - SETTINGS_MIZU_REMAP: Remapping file name (default: remap_file.nc)
            - SETTINGS_MIZU_PARAMETERS: Parameter file name
            - SETTINGS_MIZU_NEEDS_REMAP: Enable remapping (T/F)
            - SETTINGS_MIZU_MAKE_OUTLET: Comma-separated segment IDs to force as outlets
            - SETTINGS_MIZU_WITHIN_BASIN: Hillslope routing option (0/1)
            - ROUTING_DELINEATION: Routing delineation method (river_network/basin)
            - GRID_CELL_SIZE: Grid cell size in meters (for distribute mode)
            - MODEL_MIZUROUTE_FROM_MODEL: Source model name (SUMMA/FUSE/GR)

        Shapefile Column Names:
            River Network:
                - RIVER_NETWORK_SHP_SEGID: Segment ID column
                - RIVER_NETWORK_SHP_DOWNSEGID: Downstream segment ID column
                - RIVER_NETWORK_SHP_LENGTH: Segment length column (m)
                - RIVER_NETWORK_SHP_SLOPE: Segment slope column (-)

            River Basins:
                - RIVER_BASIN_SHP_RM_GRUID: GRU ID column
                - RIVER_BASIN_SHP_RM_HRUID: HRU ID column
                - RIVER_BASIN_SHP_RM_AREA: Basin area column (m²)
                - RIVER_BASIN_SHP_RM_HRU2SEG: HRU-to-segment mapping column

    Output Files:
        Network Topology (NetCDF):
            Dimensions: seg, hru
            Segment Variables:
                - segId: Unique segment IDs
                - downSegId: Downstream segment IDs (0 = outlet)
                - slope: Segment slopes
                - length: Segment lengths (m)
            HRU Variables:
                - hruId: Unique HRU IDs
                - hruToSegId: HRU-to-segment drainage mapping
                - area: HRU areas (m²)

        Remapping File (NetCDF, optional):
            Dimensions: hru, data
            Variables:
                - RN_hruId: River network HRU IDs
                - nOverlaps: Number of overlapping source HRUs per routing HRU
                - HM_hruId: Source model HRU/GRU IDs
                - weight: Areal weights for remapping

        Control File (text):
            Sections:
                - Simulation controls (start/end times, routing options)
                - Directory paths (input/output/ancillary)
                - Topology file configuration
                - Remapping configuration (if enabled)
                - Parameter file reference
                - Miscellaneous settings (hillslope routing, output frequency)

    Special Handling:
        Headwater Basins:
            - Detects basins with None/null river network data
            - Creates synthetic single-segment network
            - Uses first HRU ID as segment ID, outlet downstream ID = 0

        Lumped-to-Distributed Routing:
            - Delineates subcatchments within lumped domain
            - Creates area-weighted remapping from single SUMMA GRU to N routing HRUs
            - Enables distributed routing for lumped hydrological models

        Routing Cycles:
            - Detects cycles using iterative DFS graph traversal
            - Breaks cycles by forcing lowest-elevation segment to outlet (downSegId = 0)
            - Logs number of cycles detected and fixed

        GRU-level Runoff:
            - Detects SUMMA simulations with multiple HRUs per GRU
            - Reads SUMMA attributes.nc to determine structure
            - Aggregates HRU areas to GRU level for topology

        Grid-based Distributed:
            - Reads D8 flow direction from grid shapefile
            - Each grid cell becomes both HRU and segment
            - Segment length = grid cell size
            - Fixes cycles in D8 topology

    Integration Patterns:
        SUMMA Integration:
            - Reads attributes.nc to detect HRU/GRU structure
            - Handles both hru/hruId and gru/gruId output formats
            - Sets summa_uses_gru_runoff flag for control file

        FUSE Integration:
            - Basin-scale runoff routing
            - Control file references FUSE output files

        GR Integration:
            - Daily timestep alignment (midnight forcing)
            - R/rpy2 interface output handling
            - Forces simulation times to 00:00 alignment

    Error Handling:
        - Validates shapefile existence before processing
        - Handles missing pour point shapefiles (fallback to outlet segment)
        - Fills missing/null length and slope values with defaults
        - Detects and logs warnings for outlet segment mismatches
        - Raises FileNotFoundError for critical missing files

    Example:
        >>> config = {
        ...     'DOMAIN_NAME': 'bow_river',
        ...     'DOMAIN_DISCRETIZATION': 'lumped',
        ...     'RIVER_NETWORK_SHP_PATH': './shapefiles/river_network',
        ...     'RIVER_NETWORK_SHP_NAME': 'bow_river_riverNetwork_lumped.shp',
        ...     'RIVER_BASINS_PATH': './shapefiles/river_basins',
        ...     'RIVER_BASINS_NAME': 'bow_river_riverBasins_lumped.shp',
        ...     'EXPERIMENT_ID': 'bow_calibration',
        ...     'SETTINGS_MIZU_TOPOLOGY': 'mizuRoute_topology.nc',
        ...     'SETTINGS_MIZU_NEEDS_REMAP': False,
        ...     'MODEL_MIZUROUTE_FROM_MODEL': 'SUMMA'
        ... }
        >>> preprocessor = MizuRoutePreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
        # Creates:
        # - ./settings/mizuRoute/mizuRoute_topology.nc (network topology)
        # - ./settings/mizuRoute/mizuRoute.control (control file)
        # - ./settings/mizuRoute/*.param (parameter files)

    Notes:
        - Topology file must be created before control file generation
        - Remapping is optional and only needed when source and routing HRUs differ
        - Control file references are model-specific (SUMMA uses different variable names than FUSE/GR)
        - Grid-based distributed mode requires D8 flow direction in shapefile
        - Parallel runs can specify custom setup directory via SETTINGS_MIZU_PATH
        - Cycle detection uses O(V+E) iterative DFS to avoid recursion depth issues
        - Minimum segment length enforced (1m) to prevent numerical instabilities
        - Minimum slope enforced (0.001) for routing calculations

    See Also:
        - models.mizuroute.control_writer.ControlFileWriter: Control file generation
        - models.mizuroute.mixins.MizuRouteConfigMixin: Configuration accessors
        - geospatial.geometry_utils.GeospatialUtilsMixin: Spatial utilities
        - models.base.BaseModelPreProcessor: Base preprocessor interface
    """
    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "mizuRoute"

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the mizuRoute preprocessor.

        Sets up directory paths for routing configuration, including optional
        custom settings path for isolated parallel runs during calibration.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                mizuRoute settings, topology paths, and routing parameters.
            logger: Logger instance for status messages and debugging.
        """
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

        self.logger.debug(f"MizuRoutePreProcessor initialized. Default setup_dir: {self.setup_dir}")

        # Override setup_dir if SETTINGS_MIZU_PATH is provided (for isolated parallel runs)
        mizu_settings_path = self.mizu_settings_path
        if mizu_settings_path and mizu_settings_path != 'default':
            self.setup_dir: Path = Path(mizu_settings_path)
            self.logger.debug(f"MizuRoutePreProcessor using custom setup_dir from SETTINGS_MIZU_PATH: {self.setup_dir}")

        # Ensure setup directory exists
        if not self.setup_dir.exists():
            self.logger.info(f"Creating mizuRoute setup directory: {self.setup_dir}")
            self.setup_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.logger.debug(f"mizuRoute setup directory already exists: {self.setup_dir}")


    def run_preprocessing(self):
        """
        Run the complete mizuRoute preprocessing workflow.

        Executes all steps needed to prepare mizuRoute for routing:
        1. Copy base settings files from templates
        2. Create network topology file from river/catchment shapefiles
        3. Create remapping file if source model uses different spatial units
        4. Generate appropriate control file based on source model (SUMMA/FUSE/GR)

        The workflow adapts based on configuration, supporting both lumped-to-distributed
        remapping and distributed model coupling.
        """
        self.logger.debug("Starting mizuRoute spatial preprocessing")
        self.copy_base_settings()
        self.create_network_topology_file()

        # Get config values using typed config
        needs_remap = self._get_config_value(
            lambda: self.config.model.mizuroute.needs_remap if self.config.model and self.config.model.mizuroute else None,
            False
        )
        from_model = self._get_config_value(
            lambda: self.config.model.mizuroute.from_model if self.config.model and self.config.model.mizuroute else None
        )
        fuse_routing = self._get_config_value(
            lambda: self.config.model.fuse.routing_integration if self.config.model and self.config.model.fuse else None
        )
        gr_routing = self._get_config_value(
            lambda: self.config.model.gr.routing_integration if self.config.model and self.config.model.gr else None
        )

        # Check if lumped-to-distributed remapping is needed (set during topology creation)
        if getattr(self, 'needs_remap_lumped_distributed', False):
            self.logger.info("Creating area-weighted remap file for lumped-to-distributed routing")
            self.create_area_weighted_remap_file()
            needs_remap = True  # Override to enable remapping in control file

        self.logger.info(f"Should we remap?: {needs_remap}")
        if needs_remap and not getattr(self, 'needs_remap_lumped_distributed', False):
            self.remap_summa_catchments_to_routing()

        # Choose control writer based on source model
        if from_model == 'FUSE' or fuse_routing == 'mizuRoute':
            self.create_fuse_control_file()
        elif from_model == 'GR' or gr_routing == 'mizuRoute':
            self.create_gr_control_file()
        else:
            self.create_control_file()

        self.logger.info("mizuRoute spatial preprocessing completed")


    def copy_base_settings(self, source_dir: Optional[Path] = None, file_patterns: Optional[List[str]] = None):
        """
        Copy mizuRoute base settings from package resources.

        Copies template configuration files (parameter anchors, routing method
        settings) from symfluence resources to the setup directory, providing
        starting points that will be customized during preprocessing.
        """
        if source_dir:
            return super().copy_base_settings(source_dir, file_patterns)

        self.logger.info("Copying mizuRoute base settings")
        from symfluence.resources import get_base_settings_dir
        base_settings_path = get_base_settings_dir('mizuRoute')
        self.setup_dir.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, self.setup_dir / file)
        self.logger.info("mizuRoute base settings copied")

    def create_area_weighted_remap_file(self):
        """Create remapping file with area-based weights from delineated catchments"""
        self.logger.info("Creating area-weighted remapping file")

        # Load topology to get HRU information
        topology_file = self.setup_dir / self.mizu_topology_file
        with xr.open_dataset(topology_file) as topo:
            hru_ids = topo['hruId'].values

        n_hrus = len(hru_ids)

        # Use the weights stored during topology creation
        if hasattr(self, 'subcatchment_weights') and hasattr(self, 'subcatchment_gru_ids'):
            weights = self.subcatchment_weights
            gru_ids = self.subcatchment_gru_ids
        else:
            # Fallback: load from delineated catchments shapefile
            catchment_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.domain_name}_catchment_delineated.shp"
            shp_catchments = gpd.read_file(catchment_path)
            weights = shp_catchments['avg_subbas'].values
        remap_name = self.mizu_remap_file
        if not remap_name:
            remap_name = "remap_file.nc"
            self.logger.warning(f"SETTINGS_MIZU_REMAP not found in config, using default: {remap_name}")

        with nc4.Dataset(self.setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            # Set attributes
            ncid.setncattr('Author', "Created by SUMMA workflow scripts")
            ncid.setncattr('Purpose', 'Area-weighted remapping for lumped to distributed routing')

            # Create dimensions
            ncid.createDimension('hru', n_hrus)  # One entry per HRU
            ncid.createDimension('data', n_hrus)  # One data entry per HRU

            # Create variables
            # RN_hruId: The routing HRU IDs (from delineated catchments)
            rn_hru = ncid.createVariable('RN_hruId', 'i4', ('hru',))
            rn_hru[:] = gru_ids
            rn_hru.long_name = 'River network HRU ID'

            # nOverlaps: Each HRU gets input from 1 SUMMA GRU
            noverlaps = ncid.createVariable('nOverlaps', 'i4', ('hru',))
            noverlaps[:] = [1] * n_hrus  # Each HRU has 1 overlap (with SUMMA GRU 1)
            noverlaps.long_name = 'Number of overlapping HM_HRUs for each RN_HRU'

            # HM_hruId: The SUMMA GRU ID (1) for each entry
            hm_hru = ncid.createVariable('HM_hruId', 'i4', ('data',))
            hm_hru[:] = [1] * n_hrus  # All entries point to SUMMA GRU 1
            hm_hru.long_name = 'ID of overlapping HM_HRUs'

            # weight: Area-based weights from delineated catchments
            weight_var = ncid.createVariable('weight', 'f8', ('data',))
            weight_var[:] = weights
            weight_var.long_name = 'Areal weights based on delineated subcatchment areas'

        self.logger.info(f"Area-weighted remapping file created with {n_hrus} HRUs")
        self.logger.info(f"Weight range: {weights.min():.4f} to {weights.max():.4f}")
        self.logger.info(f"Weight sum: {weights.sum():.4f}")



    def create_gr_control_file(self):
        """Create mizuRoute control file specifically for GR4J input."""
        writer = self._get_control_writer()
        mizu_config = self._get_mizu_config()
        writer.write_control_file(model_type='gr', mizu_config=mizu_config)



    def _check_if_headwater_basin(self, shp_river):
        """
        Check if this is a headwater basin with None/invalid river network data.

        Args:
            shp_river: GeoDataFrame of river network

        Returns:
            bool: True if this appears to be a headwater basin with invalid network data
        """
        # Check for critical None values in key columns
        seg_id_col = self.river_segid_col
        downseg_id_col = self.river_downsegid_col

        if seg_id_col in shp_river.columns and downseg_id_col in shp_river.columns:
            # Check if all segment IDs are None/null
            seg_ids_null = shp_river[seg_id_col].isna().all()
            downseg_ids_null = shp_river[downseg_id_col].isna().all()

            if seg_ids_null and downseg_ids_null:
                self.logger.info("Detected headwater basin: all river network IDs are None/null")
                return True

            # Also check for string 'None' values (sometimes shapefiles store None as string)
            if shp_river[seg_id_col].dtype == 'object':
                seg_ids_none_str = (shp_river[seg_id_col] == 'None').all()
                downseg_ids_none_str = (shp_river[downseg_id_col] == 'None').all()

                if seg_ids_none_str and downseg_ids_none_str:
                    self.logger.info("Detected headwater basin: all river network IDs are 'None' strings")
                    return True

        return False

    def _create_synthetic_river_network(self, shp_river, hru_ids):
        """
        Create a synthetic single-segment river network for headwater basins.

        Args:
            shp_river: Original GeoDataFrame (with None values)
            hru_ids: Array of HRU IDs from delineated catchments

        Returns:
            GeoDataFrame: Modified river network with synthetic single segment
        """
        self.logger.info("Creating synthetic river network for headwater basin")

        # Use the first HRU ID as the segment ID (should be reasonable identifier)
        synthetic_seg_id = int(hru_ids[0]) if len(hru_ids) > 0 else 1

        # Create synthetic values for the single segment
        synthetic_data = {
            self.river_segid_col: synthetic_seg_id,
            self.river_downsegid_col: 0,  # Outlet (downstream ID = 0)
            self.river_length_col: 1000.0,  # Default 1 km length
            self.river_slope_col: 0.001,  # Default 0.1% slope
        }

        # Get the geometry column name (usually 'geometry')
        geom_col = shp_river.geometry.name

        # Create a simple point geometry at the centroid of the original (if it exists)
        if not shp_river.empty and shp_river.geometry.iloc[0] is not None:
            # Use the centroid of the first geometry, handling CRS projection via mixin
            synthetic_geom = self.calculate_feature_centroids(shp_river.iloc[[0]]).iloc[0]
        else:
            # Create a default point geometry (this won't be used for actual routing)
            from shapely.geometry import Point
            synthetic_geom = Point(0, 0)

        synthetic_data[geom_col] = synthetic_geom

        # Create new GeoDataFrame with single row
        synthetic_gdf = gpd.GeoDataFrame([synthetic_data], crs=shp_river.crs)

        self.logger.info(f"Created synthetic river network: segment ID {synthetic_seg_id} (outlet)")

        return synthetic_gdf

    def create_network_topology_file(self):
        """
        Create the network topology NetCDF file for mizuRoute.

        Generates a topology file containing river segment IDs, downstream connectivity,
        HRU assignments, and channel properties. Supports multiple modes:
        - Standard distributed: Uses river network and basin shapefiles
        - Lumped-to-distributed: Creates synthetic network for single-GRU to multi-segment
        - Grid-based: Creates topology from regular grid cells
        - Point-scale: Creates minimal single-segment topology

        The topology file is required by mizuRoute to route water through the network.
        """
        self.logger.info("Creating network topology file")

        # Check for grid-based distribute mode
        is_grid_distribute = self.domain_definition_method == 'distribute'
        if is_grid_distribute:
            self._create_grid_topology_file()
            return

        # Check for point-scale mode
        is_point_scale = self.domain_definition_method == 'point'
        if is_point_scale:
            self._create_point_topology_file()
            return

        river_network_path = self.config_dict.get('RIVER_NETWORK_SHP_PATH')
        river_network_name = self.config_dict.get('RIVER_NETWORK_SHP_NAME')
        method_suffix = self._get_method_suffix()

        # Check if this is lumped domain with distributed routing
        # If so, use the delineated river network (from distributed delineation)
        is_lumped_to_distributed = (
            self.domain_definition_method == 'lumped' and
            self.config_dict.get('ROUTING_DELINEATION', 'river_network') == 'river_network'
        )

        # For lumped-to-distributed, use delineated river network and catchments
        routing_suffix = 'delineate' if is_lumped_to_distributed else method_suffix

        if river_network_name == 'default':
            river_network_name = f"{self.domain_name}_riverNetwork_{routing_suffix}.shp"

        if river_network_path == 'default':
            river_network_path = self.project_dir / 'shapefiles/river_network'
        else:
            river_network_path = Path(river_network_path)

        river_basin_path = self.config_dict.get('RIVER_BASINS_PATH')
        river_basin_name = self.config_dict.get('RIVER_BASINS_NAME')

        if river_basin_name == 'default':
            river_basin_name = f"{self.domain_name}_riverBasins_{routing_suffix}.shp"

        if river_basin_path == 'default':
            river_basin_path = self.project_dir / 'shapefiles/river_basins'
        else:
            river_basin_path = Path(river_basin_path)

        topology_name = self.mizu_topology_file
        if not topology_name:
            topology_name = "mizuRoute_topology.nc"
            self.logger.warning(f"SETTINGS_MIZU_TOPOLOGY not found in config, using default: {topology_name}")

        # Load shapefiles
        shp_river = gpd.read_file(river_network_path / river_network_name)
        shp_basin = gpd.read_file(river_basin_path / river_basin_name)

        if is_lumped_to_distributed:
            self.logger.info("Using delineated catchments for lumped-to-distributed routing")

            # For lumped-to-distributed, SUMMA output is converted to gru/gruId format
            # by the spatial_orchestrator, so mizuRoute control file should use gru/gruId
            self.summa_uses_gru_runoff = True

            # Enable remapping: map single lumped SUMMA GRU to 25 routing HRUs with area weights
            self.needs_remap_lumped_distributed = True

            # Load the delineated catchments shapefile
            catchment_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.domain_name}_catchment_delineated.shp"
            if not catchment_path.exists():
                raise FileNotFoundError(f"Delineated catchment shapefile not found: {catchment_path}")

            shp_catchments = gpd.read_file(catchment_path)
            self.logger.info(f"Loaded {len(shp_catchments)} delineated subcatchments")

            # Extract HRU data from delineated catchments
            hru_ids = shp_catchments['GRU_ID'].values.astype(int)

            # Check if we have a headwater basin (None values in river network)
            if self._check_if_headwater_basin(shp_river):
                # Create synthetic river network for headwater basin
                shp_river = self._create_synthetic_river_network(shp_river, hru_ids)

            # Use the delineated catchments as HRUs
            num_seg = len(shp_river)
            num_hru = len(shp_catchments)

            hru_to_seg_ids = shp_catchments['GRU_ID'].values.astype(int)  # Each GRU drains to segment with same ID

            # Convert fractional areas to actual areas (multiply by total basin area)
            total_basin_area = shp_basin[self.basin_area_col].sum()
            hru_areas = shp_catchments['avg_subbas'].values * total_basin_area

            # Store fractional areas for remapping
            self.subcatchment_weights = shp_catchments['avg_subbas'].values
            self.subcatchment_gru_ids = hru_ids

            self.logger.info(f"Created {num_hru} HRUs from delineated catchments")
            self.logger.info(f"Weight range: {self.subcatchment_weights.min():.4f} to {self.subcatchment_weights.max():.4f}")

        else:
            # Check if we have SUMMA attributes file with multiple HRUs per GRU
            attributes_path = self.project_dir / 'settings' / 'SUMMA' / 'attributes.nc'

            if attributes_path.exists():
                with nc4.Dataset(attributes_path, 'r') as attrs:
                    n_hrus = len(attrs.dimensions['hru'])
                    n_grus = len(attrs.dimensions['gru'])

                    if n_hrus > n_grus:
                        # Multiple HRUs per GRU - SUMMA will output GRU-level runoff
                        # mizuRoute should route at GRU level
                        self.logger.info(f"Distributed SUMMA with {n_hrus} HRUs across {n_grus} GRUs")
                        self.logger.info("Creating GRU-level topology for mizuRoute (SUMMA outputs averageRoutedRunoff at GRU level)")

                        # Read GRU information from SUMMA attributes file
                        gru_ids = attrs.variables['gruId'][:].astype(int)

                        # For distributed SUMMA, GRU IDs should match segment IDs
                        hru_ids = gru_ids  # mizuRoute will read GRU-level data
                        hru_to_seg_ids = gru_ids  # Each GRU drains to segment with same ID

                        # Calculate GRU areas by summing HRU areas within each GRU
                        hru2gru = attrs.variables['hru2gruId'][:].astype(int)
                        hru_areas_all = attrs.variables['HRUarea'][:].astype(float)

                        # Sum areas for each GRU
                        gru_areas = np.zeros(n_grus)
                        for i, gru_id in enumerate(gru_ids):
                            gru_mask = hru2gru == gru_id
                            gru_areas[i] = hru_areas_all[gru_mask].sum()

                        hru_areas = gru_areas

                        num_seg = len(shp_river)
                        num_hru = n_grus  # mizuRoute sees GRUs as HRUs

                        # Store flag for control file generation
                        self.summa_uses_gru_runoff = True

                        self.logger.info(f"Created topology with {num_hru} GRUs for mizuRoute routing")
                    else:
                        # Lumped modeling: use original logic
                        self.summa_uses_gru_runoff = False
                        closest_segment_id = self._find_closest_segment_to_pour_point(shp_river)

                        if len(shp_basin) == 1:
                            shp_basin.loc[0, self.basin_hru_to_seg_col] = closest_segment_id
                            self.logger.info(f"Set single HRU to drain to closest segment: {closest_segment_id}")

                        num_seg = len(shp_river)
                        num_hru = len(shp_basin)

                        hru_ids = shp_basin[self.basin_gruid_col].values.astype(int)
                        hru_to_seg_ids = shp_basin[self.basin_hru_to_seg_col].values.astype(int)
                        hru_areas = shp_basin[self.basin_area_col].values.astype(float)
            else:
                # No attributes file: use original logic
                self.summa_uses_gru_runoff = False
                closest_segment_id = self._find_closest_segment_to_pour_point(shp_river)

                if len(shp_basin) == 1:
                    shp_basin.loc[0, self.basin_hru_to_seg_col] = closest_segment_id
                    self.logger.info(f"Set single HRU to drain to closest segment: {closest_segment_id}")

                num_seg = len(shp_river)
                num_hru = len(shp_basin)

                hru_ids = shp_basin[self.basin_gruid_col].values.astype(int)
                hru_to_seg_ids = shp_basin[self.basin_hru_to_seg_col].values.astype(int)
                hru_areas = shp_basin[self.basin_area_col].values.astype(float)

        # Ensure minimum segment length - now safe from None values
        length_col = self.river_length_col
        if length_col in shp_river.columns:
            # Convert None/null values to 0 first, then set minimum
            shp_river[length_col] = shp_river[length_col].fillna(0)
            shp_river.loc[shp_river[length_col] == 0, length_col] = 1

        # Ensure slope column has valid values
        slope_col = self.river_slope_col
        if slope_col in shp_river.columns:
            shp_river[slope_col] = shp_river[slope_col].fillna(0.001)  # Default slope
            shp_river.loc[shp_river[slope_col] == 0, slope_col] = 0.001

        # Enforce outlets if specified
        make_outlet = self.mizu_make_outlet
        if make_outlet and make_outlet != 'n/a':
            river_outlet_ids = [int(id) for id in make_outlet.split(',')]
            seg_id_col = self.river_segid_col
            downseg_id_col = self.river_downsegid_col

            for outlet_id in river_outlet_ids:
                if outlet_id in shp_river[seg_id_col].values:
                    shp_river.loc[shp_river[seg_id_col] == outlet_id, downseg_id_col] = 0
                else:
                    self.logger.warning(f"Outlet ID {outlet_id} not found in river network")

        # Create the netCDF file
        with nc4.Dataset(self.setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, num_seg, num_hru)

            # Create segment variables (now safe from None values)
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', shp_river[self.river_segid_col].values.astype(int), 'Unique ID of each stream segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', shp_river[self.river_downsegid_col].values.astype(int), 'ID of the downstream segment', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', shp_river[self.river_slope_col].values.astype(float), 'Segment slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', shp_river[self.river_length_col].values.astype(float), 'Segment length', 'm')

            # Create HRU variables (using our computed values)
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', hru_ids, 'Unique hru ID', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', hru_to_seg_ids, 'ID of the stream segment to which the HRU discharges', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', hru_areas, 'HRU area', 'm^2')

        self.logger.info(f"Network topology file created at {self.setup_dir / topology_name}")

    def _find_closest_segment_to_pour_point(self, shp_river):
        """
        Find the river segment closest to the pour point.

        Args:
            shp_river: GeoDataFrame of river network

        Returns:
            int: Segment ID of closest segment to pour point
        """

        # Find pour point shapefile
        pour_point_dir = self.project_dir / 'shapefiles' / 'pour_point'
        pour_point_files = list(pour_point_dir.glob('*.shp'))

        if not pour_point_files:
            self.logger.error(f"No pour point shapefiles found in {pour_point_dir}")
            # Fallback: use outlet segment (downSegId == 0)
            outlet_mask = shp_river[self.river_downsegid_col] == 0
            if outlet_mask.any():
                outlet_seg = shp_river.loc[outlet_mask, self.river_segid_col].iloc[0]
                self.logger.warning(f"Using outlet segment as fallback: {outlet_seg}")
                return outlet_seg
            else:
                # Last resort: use first segment
                fallback_seg = shp_river[self.river_segid_col].iloc[0]
                self.logger.warning(f"Using first segment as fallback: {fallback_seg}")
                return fallback_seg

        # Load first pour point file
        pour_point_file = pour_point_files[0]
        self.logger.info(f"Loading pour point from {pour_point_file}")

        try:
            shp_pour_point = gpd.read_file(pour_point_file)

            # Ensure both are in the same CRS
            if shp_river.crs != shp_pour_point.crs:
                shp_pour_point = shp_pour_point.to_crs(shp_river.crs)

            # Get pour point coordinates (assume first/only point)
            shp_pour_point.geometry.iloc[0]

            # Calculate distances from pour point to all river segments
            shp_river_proj = shp_river.to_crs(shp_river.estimate_utm_crs())
            # Use mixin to get pour point centroid safely if needed (though it's a point)
            pour_point_centroids = self.calculate_feature_centroids(shp_pour_point.iloc[[0]])
            pour_point_proj = pour_point_centroids.to_crs(shp_river_proj.crs)
            distances = shp_river_proj.geometry.distance(pour_point_proj.iloc[0])

            # Find closest segment
            closest_idx = distances.idxmin()
            closest_segment_id = shp_river.loc[closest_idx, self.river_segid_col]

            self.logger.info(f"Closest segment to pour point: {closest_segment_id} (distance: {distances.iloc[closest_idx]:.1f} units)")

            return closest_segment_id

        except Exception as e:
            self.logger.error(f"Error finding closest segment: {str(e)}")
            # Fallback to outlet segment
            outlet_mask = shp_river[self.river_downsegid_col] == 0
            if outlet_mask.any():
                outlet_seg = shp_river.loc[outlet_mask, self.river_segid_col].iloc[0]
                self.logger.warning(f"Using outlet segment as fallback: {outlet_seg}")
                return outlet_seg
            else:
                fallback_seg = shp_river[self.river_segid_col].iloc[0]
                self.logger.warning(f"Using first segment as fallback: {fallback_seg}")
                return fallback_seg

    def create_equal_weight_remap_file(self):
        """
        Create remapping file with equal weights for all routing HRUs.

        This method creates a NetCDF remapping file that distributes runoff
        equally from a single lumped hydrological model GRU to multiple
        routing HRUs. Used when routing a lumped model through a distributed
        river network (e.g., single SUMMA GRU routed through multiple
        mizuRoute segments).

        The remapping file contains:
        - RN_hruId: Routing network HRU IDs from topology
        - nOverlaps: Number of source GRUs per routing HRU (always 1)
        - HM_hruId: Source hydrological model GRU ID (always 1)
        - weight: Equal areal weight (1/n_hrus) for each HRU

        File is written to: {setup_dir}/{SETTINGS_MIZU_REMAP}

        Note:
            This equal-weight approach assumes uniform runoff distribution
            and is appropriate only for lumped-to-distributed routing.
            For area-weighted remapping, use create_area_weighted_remap_file.
        """
        self.logger.info("Creating equal-weight remapping file")

        # Load topology to get segment information
        topology_file = self.setup_dir / self.mizu_topology_file
        with xr.open_dataset(topology_file) as topo:
            seg_ids = topo['segId'].values
            hru_ids = topo['hruId'].values  # Now we have multiple HRUs

        len(seg_ids)
        n_hrus = len(hru_ids)
        equal_weight = 1.0 / n_hrus  # Equal weight for each HRU

        remap_name = self.mizu_remap_file
        if not remap_name:
            remap_name = "remap_file.nc"
            self.logger.warning(f"SETTINGS_MIZU_REMAP not found in config, using default: {remap_name}")

        with nc4.Dataset(self.setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            # Set attributes
            ncid.setncattr('Author', "Created by SUMMA workflow scripts")
            ncid.setncattr('Purpose', 'Equal-weight remapping for lumped to distributed routing')

            # Create dimensions
            ncid.createDimension('hru', n_hrus)  # One entry per HRU
            ncid.createDimension('data', n_hrus)  # One data entry per HRU

            # Create variables
            # RN_hruId: The routing HRU IDs (1, 2, 3, ..., n_hrus)
            rn_hru = ncid.createVariable('RN_hruId', 'i4', ('hru',))
            rn_hru[:] = hru_ids
            rn_hru.long_name = 'River network HRU ID'

            # nOverlaps: Each HRU gets input from 1 SUMMA GRU
            noverlaps = ncid.createVariable('nOverlaps', 'i4', ('hru',))
            noverlaps[:] = [1] * n_hrus  # Each HRU has 1 overlap (with SUMMA GRU 1)
            noverlaps.long_name = 'Number of overlapping HM_HRUs for each RN_HRU'

            # HM_hruId: The SUMMA GRU ID (1) for each entry
            hm_hru = ncid.createVariable('HM_hruId', 'i4', ('data',))
            hm_hru[:] = [1] * n_hrus  # All entries point to SUMMA GRU 1
            hm_hru.long_name = 'ID of overlapping HM_HRUs'

            # weight: Equal weights for all HRUs
            weights = ncid.createVariable('weight', 'f8', ('data',))
            weights[:] = [equal_weight] * n_hrus
            weights.long_name = f'Equal areal weights ({equal_weight:.4f}) for all HRUs'

        self.logger.info(f"Equal-weight remapping file created with {n_hrus} HRUs, weight = {equal_weight:.4f}")

    def remap_summa_catchments_to_routing(self):
        """
        Create remapping file from SUMMA catchments to routing network HRUs.

        Computes spatial intersection between hydrological model (HM) catchments
        and routing model (RM) basins to create area-weighted remapping. This
        enables routing when the source model uses different spatial units than
        the river network topology.

        For lumped domains with river_network delineation, creates area-weighted
        remapping that distributes runoff proportionally to HRU areas. For
        distributed domains, performs full spatial intersection using EASYMORE.

        The workflow:
        1. Load HM catchment and RM basin shapefiles
        2. Perform spatial intersection (reproject to EPSG:6933 for accuracy)
        3. Calculate area weights for each HM-RM overlap
        4. Write remapping NetCDF with overlap counts and weights

        Configuration keys used:
        - CATCHMENT_PATH, CATCHMENT_SHP_NAME: Source model catchments
        - RIVER_BASINS_PATH, RIVER_BASINS_NAME: Routing network basins
        - INTERSECT_ROUTING_PATH, INTERSECT_ROUTING_NAME: Intersection output
        - SETTINGS_MIZU_REMAP: Output remapping file name

        Note:
            Requires geopandas and EASYMORE for spatial intersection operations.
        """
        self.logger.info("Remapping SUMMA catchments to routing catchments")
        if self.domain_definition_method == 'lumped' and self.config_dict.get('ROUTING_DELINEATION') == 'river_network':
            self.logger.info("Area-weighted mapping for SUMMA catchments to routing catchments")
            self.create_area_weighted_remap_file()  # Changed from create_equal_weight_remap_file
            return

        hm_catchment_path = Path(self.config_dict.get('CATCHMENT_PATH'))
        hm_catchment_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if hm_catchment_name == 'default':
            hm_catchment_name = f"{self.domain_name}_HRUs_{self.config_dict.get('DOMAIN_DISCRETIZATION')}.shp"

        rm_catchment_path = Path(self.config_dict.get('RIVER_BASINS_PATH'))
        rm_catchment_name = self.config_dict.get('RIVER_BASINS_NAME')

        intersect_path = Path(self.config_dict.get('INTERSECT_ROUTING_PATH'))
        intersect_name = self.config_dict.get('INTERSECT_ROUTING_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_routing_basins.shp'

        if intersect_path == 'default':
            intersect_path = self.project_dir / 'shapefiles/catchment_intersection'
        else:
            intersect_path = Path(intersect_path)

        remap_name = self.mizu_remap_file
        if not remap_name:
            remap_name = "remap_file.nc"
            self.logger.warning(f"SETTINGS_MIZU_REMAP not found in config, using default: {remap_name}")

        if hm_catchment_path == 'default':
            hm_catchment_path = self.project_dir / 'shapefiles/catchment'
        else:
            hm_catchment_path = Path(hm_catchment_path)

        if rm_catchment_path == 'default':
            rm_catchment_path = self.project_dir / 'shapefiles/catchment'
        else:
            rm_catchment_path = Path(rm_catchment_path)

        # Load shapefiles
        hm_shape = gpd.read_file(hm_catchment_path / hm_catchment_name)
        rm_shape = gpd.read_file(rm_catchment_path / rm_catchment_name)

        # Create intersection
        esmr_obj = _create_easymore_instance()
        hm_shape = hm_shape.to_crs('EPSG:6933')
        rm_shape = rm_shape.to_crs('EPSG:6933')
        intersected_shape = esmr_obj.intersection_shp(rm_shape, hm_shape)
        intersected_shape = intersected_shape.to_crs('EPSG:4326')
        intersected_shape.to_file(intersect_path / intersect_name)

        # Process variables for remapping file
        self._process_remap_variables(intersected_shape)

        # Create remapping netCDF file
        self._create_remap_file(intersected_shape, remap_name)

        self.logger.info(f"Remapping file created at {self.setup_dir / remap_name}")

    def create_control_file(self):
        """
        Create mizuRoute control file for SUMMA runoff input.

        Generates the mizuRoute control file (*.control) that configures the
        routing simulation when using SUMMA as the source hydrological model.
        The control file specifies input/output paths, topology files, routing
        scheme parameters, and simulation time controls.

        The control file includes sections for:
        - Directory and file paths (topology, runoff input, output)
        - Simulation period (start/end times from config)
        - Routing scheme selection (IRF, KWT, DW)
        - Spatial configuration (segments, HRUs, remapping)
        - Output variable selection and frequency

        Uses ControlFileWriter to generate SUMMA-specific settings that account
        for SUMMA's GRU-level runoff output format and time conventions.

        File is written to: {setup_dir}/{experiment_id}.control

        See Also:
            create_fuse_control_file: For FUSE model input configuration.
        """
        writer = self._get_control_writer()
        mizu_config = self._get_mizu_config()
        writer.write_control_file(model_type='summa', mizu_config=mizu_config)

    def _set_topology_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a river network .nc file for mizuRoute routing')

    def _create_topology_dimensions(self, ncid, num_seg, num_hru):
        ncid.createDimension('seg', num_seg)
        ncid.createDimension('hru', num_hru)

    def _fix_routing_cycles(self, seg_ids, down_seg_ids, elevations):
        """
        Detect and fix cycles in the routing graph.

        For each cycle found, the node with the lowest elevation is forced
        to be an outlet (downSegId = 0).

        Args:
            seg_ids: Array of segment IDs
            down_seg_ids: Array of downstream segment IDs
            elevations: Array of segment elevations

        Returns:
            Fixed down_seg_ids array
        """
        self.logger.info("Checking for cycles in routing topology...")

        # Create mapping from ID to index
        id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

        # Adjacency list (node_idx -> downstream_node_idx)
        # Use -1 for outlet/external
        adj = {}
        for i, down_sid in enumerate(down_seg_ids):
            if down_sid in id_to_idx:
                adj[i] = id_to_idx[down_sid]
            else:
                adj[i] = -1

        visited = set()
        path_set = set()
        path_stack = []
        cycles_found = 0
        fixed_down_ids = down_seg_ids.copy()

        def visit(u):
            nonlocal cycles_found

            [(u, iter(adj.get(u, []) if u in adj and adj[u] != -1 else []))]
            path_set.add(u)
            path_stack.append(u)
            visited.add(u)

            # Iterative DFS to avoid recursion depth issues
            curr = u
            while True:
                neighbor = adj.get(curr, -1)

                if neighbor == -1:
                    # End of path
                    path_set.remove(curr)
                    path_stack.pop()
                    if not path_stack:
                        break
                    curr = path_stack[-1]
                    continue

                if neighbor in path_set:
                    # Cycle detected
                    cycle_nodes_idx = []
                    # Extract cycle from path_stack
                    try:
                        start_pos = path_stack.index(neighbor)
                        cycle_nodes_idx = path_stack[start_pos:]
                    except ValueError:
                        pass # Should not happen

                    if cycle_nodes_idx:
                        cycles_found += 1

                        # Find node with lowest elevation in cycle
                        min_elev = float('inf')
                        sink_node_idx = -1

                        for idx in cycle_nodes_idx:
                            elev = elevations[idx]
                            if elev < min_elev:
                                min_elev = elev
                                sink_node_idx = idx

                        # Break cycle: make sink_node an outlet
                        if sink_node_idx != -1:
                            fixed_down_ids[sink_node_idx] = 0
                            # Update adjacency to reflect break for future traversals
                            adj[sink_node_idx] = -1

                    # Backtrack
                    path_set.remove(curr)
                    path_stack.pop()
                    if not path_stack:
                        break
                    curr = path_stack[-1]
                    continue

                if neighbor not in visited:
                    visited.add(neighbor)
                    path_set.add(neighbor)
                    path_stack.append(neighbor)
                    curr = neighbor
                else:
                    # Already visited, not a cycle
                    path_set.remove(curr)
                    path_stack.pop()
                    if not path_stack:
                        break
                    curr = path_stack[-1]

        # Iterative DFS wrapper
        # The above nested function approach was a bit mix of recursive/iterative thinking.
        # Let's implement a clean iterative DFS.

        visited = set()
        path_set = set()

        for start_node_idx in range(len(seg_ids)):
            if start_node_idx in visited:
                continue

            stack = [(start_node_idx, 0)] # node_idx, state (0: enter, 1: exit)

            while stack:
                u, state = stack[-1]

                if state == 0:
                    visited.add(u)
                    path_set.add(u)
                    stack[-1] = (u, 1) # Next time we see u, we are exiting

                    v = adj.get(u, -1)
                    if v != -1:
                        if v in path_set:
                            # Cycle detected
                            cycles_found += 1

                            # Trace back stack to find cycle
                            cycle_indices = []
                            for node, _ in reversed(stack):
                                cycle_indices.append(node)
                                if node == v:
                                    break

                            # Find lowest elevation
                            min_elev = float('inf')
                            sink_idx = -1
                            for idx in cycle_indices:
                                if elevations[idx] < min_elev:
                                    min_elev = elevations[idx]
                                    sink_idx = idx

                            # Break cycle
                            fixed_down_ids[sink_idx] = 0
                            adj[sink_idx] = -1 # Update graph

                            # No need to continue this path as it's broken
                            # But we continue DFS to find other components

                        elif v not in visited:
                            stack.append((v, 0))
                else:
                    path_set.remove(u)
                    stack.pop()

        if cycles_found > 0:
            self.logger.warning(f"Detected and fixed {cycles_found} cycles in routing topology.")
        else:
            self.logger.info("No cycles detected in routing topology.")

        return fixed_down_ids

    def _create_grid_topology_file(self):
        """
        Create mizuRoute topology for grid-based distributed modeling.

        Each grid cell becomes both an HRU and a segment. D8 flow direction
        determines segment connectivity.
        """
        self.logger.info("Creating grid-based network topology for distributed mode")

        # Load grid shapefile with D8 topology
        grid_path = self.project_dir / 'shapefiles' / 'river_basins' / f"{self.domain_name}_riverBasins_distribute.shp"

        if not grid_path.exists():
            self.logger.error(f"Grid basins shapefile not found: {grid_path}")
            raise FileNotFoundError(f"Grid basins not found: {grid_path}")

        grid_gdf = gpd.read_file(grid_path)
        num_cells = len(grid_gdf)

        self.logger.info(f"Loaded {num_cells} grid cells from {grid_path}")

        topology_name = self.mizu_topology_file

        # Extract topology data from grid shapefile
        seg_ids = grid_gdf['GRU_ID'].values.astype(int)

        # Get downstream IDs from D8 topology
        # Note: shapefile truncates column names to 10 chars, so downstream_id becomes downstream
        if 'downstream_id' in grid_gdf.columns:
            down_seg_ids = grid_gdf['downstream_id'].values.astype(int)
        elif 'downstream' in grid_gdf.columns:
            down_seg_ids = grid_gdf['downstream'].values.astype(int)
        elif 'DSLINKNO' in grid_gdf.columns:
            down_seg_ids = grid_gdf['DSLINKNO'].values.astype(int)
        else:
            self.logger.warning("No D8 topology found, setting all cells as outlets")
            down_seg_ids = np.zeros(num_cells, dtype=int)

        # Get slopes from grid
        if 'slope' in grid_gdf.columns:
            slopes = grid_gdf['slope'].values.astype(float)
            # Ensure minimum slope
            slopes = np.maximum(slopes, 0.001)
        else:
            self.logger.warning("No slope data found, using default 0.01")
            slopes = np.full(num_cells, 0.01)

        # Get elevations from grid (for cycle breaking)
        if 'elev_mean' in grid_gdf.columns:
            elevations = grid_gdf['elev_mean'].values.astype(float)
        else:
            self.logger.warning("No elevation data found, using 0.0")
            elevations = np.zeros(num_cells)

        # Fix cycles in topology
        down_seg_ids = self._fix_routing_cycles(seg_ids, down_seg_ids, elevations)

        # Get cell size for segment length
        grid_cell_size = self.config_dict.get('GRID_CELL_SIZE', 1000.0)
        lengths = np.full(num_cells, float(grid_cell_size))

        # HRU variables (each cell is also an HRU)
        hru_ids = seg_ids.copy()
        hru_to_seg_ids = seg_ids.copy()  # Each HRU drains to its own segment

        # Get HRU areas
        if 'GRU_area' in grid_gdf.columns:
            hru_areas = grid_gdf['GRU_area'].values.astype(float)
        else:
            self.logger.warning("No area data found, using cell size squared")
            hru_areas = np.full(num_cells, grid_cell_size ** 2)

        # Create the netCDF topology file
        with nc4.Dataset(self.setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, num_cells, num_cells)

            # Create segment variables
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', seg_ids,
                                         'Unique ID of each grid cell segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', down_seg_ids,
                                         'ID of downstream grid cell (0=outlet)', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', slopes,
                                         'Grid cell slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', lengths,
                                         'Grid cell length (cell size)', 'm')

            # Create HRU variables
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', hru_ids,
                                         'Unique HRU ID (=grid cell ID)', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', hru_to_seg_ids,
                                         'Segment to which HRU drains (=cell ID)', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', hru_areas,
                                         'HRU area', 'm^2')

        # Count outlets for logging
        n_outlets = np.sum(down_seg_ids == 0)
        self.logger.info(f"Grid topology created: {num_cells} cells, {n_outlets} outlets")
        self.logger.info(f"Topology file: {self.setup_dir / topology_name}")

        # Set flag for control file - grid cells use GRU-level runoff
        self.summa_uses_gru_runoff = True

    def _create_point_topology_file(self):
        """
        Create mizuRoute topology for point-scale modeling.

        Point-scale domains have a single HRU and a single segment (outlet).
        """
        self.logger.info("Creating point-scale network topology")

        topology_name = self.mizu_topology_file
        if not topology_name:
            topology_name = "mizuRoute_topology.nc"

        # Single segment and HRU for point-scale domain
        seg_id = 1
        down_seg_id = 0  # Outlet
        hru_id = 1

        # Default values for point-scale
        slope = 0.01  # 1% slope default
        length = 100.0  # 100m default segment length
        area = 10000.0  # 1 hectare default area

        # Create the netCDF topology file
        with nc4.Dataset(self.setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, 1, 1)  # 1 segment, 1 HRU

            # Create segment variables
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', np.array([seg_id]),
                                         'Unique ID of segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', np.array([down_seg_id]),
                                         'ID of downstream segment (0=outlet)', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', np.array([slope]),
                                         'Segment slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', np.array([length]),
                                         'Segment length', 'm')

            # Create HRU variables
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', np.array([hru_id]),
                                         'Unique HRU ID', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', np.array([seg_id]),
                                         'Segment to which HRU drains', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', np.array([area]),
                                         'HRU area', 'm^2')

        self.logger.info("Point-scale topology created: 1 HRU, 1 outlet segment")
        self.logger.info(f"Topology file: {self.setup_dir / topology_name}")

        # Set flag for control file - point-scale uses GRU-level runoff
        self.summa_uses_gru_runoff = True

    def create_fuse_control_file(self):
        """
        Create mizuRoute control file for FUSE runoff input.

        Generates the mizuRoute control file (*.control) configured for routing
        FUSE model output. FUSE produces runoff in a different format than SUMMA,
        requiring specific variable mappings and time handling in the control file.

        Key FUSE-specific configurations:
        - Variable name mapping for FUSE runoff output variables
        - Time dimension handling (FUSE uses different time conventions)
        - Appropriate runoff flux units conversion if needed

        The control file includes the same structural sections as SUMMA routing:
        - Directory and file paths (topology, runoff input, output)
        - Simulation period matching the FUSE run
        - Routing scheme selection (IRF, KWT, DW)
        - Spatial configuration with appropriate remapping

        File is written to: {setup_dir}/{experiment_id}.control

        See Also:
            create_control_file: For SUMMA model input configuration.
        """
        writer = self._get_control_writer()
        mizu_config = self._get_mizu_config()
        writer.write_control_file(model_type='fuse', mizu_config=mizu_config)

    def _get_control_writer(self) -> ControlFileWriter:
        """Get a configured ControlFileWriter instance."""
        writer = ControlFileWriter(
            config=self.config_dict,
            setup_dir=self.setup_dir,
            project_dir=self.project_dir,
            experiment_id=self.experiment_id,
            domain_name=self.domain_name,
            logger=self.logger
        )
        # Transfer state flags
        writer.summa_uses_gru_runoff = getattr(self, 'summa_uses_gru_runoff', False)
        writer.needs_remap_lumped_distributed = getattr(self, 'needs_remap_lumped_distributed', False)
        return writer

    def _get_mizu_config(self) -> dict:
        """Get mizuRoute configuration values for the control writer."""
        return {
            'topology_file': self.mizu_topology_file,
            'remap_file': self.mizu_remap_file,
            'parameters_file': self.mizu_parameters_file,
            'within_basin': self.mizu_within_basin,
        }

    # Legacy method kept for backwards compatibility - now unused
    def _write_control_file_simulation_controls(self, cf):
        """Enhanced simulation control writing with proper time handling"""
        # Get simulation dates from config
        sim_start = self.time_start
        sim_end = self.time_end

        # Determine source model
        from_model = self.mizu_from_model.upper()
        gr_routing = self.config_dict.get('GR_ROUTING_INTEGRATION', 'none').lower() == 'mizuroute'

        # Special handling for GR: force midnight alignment for daily data
        if from_model == 'GR' or gr_routing:
            if isinstance(sim_start, str):
                # Replace any time part with 00:00
                sim_start = sim_start.split(' ')[0] + " 00:00"
            if isinstance(sim_end, str):
                # Replace any time part with 00:00 (or keep as is if daily)
                sim_end = sim_end.split(' ')[0] + " 00:00"
            self.logger.debug(f"Forced GR simulation period to midnight: {sim_start} to {sim_end}")

        # Ensure dates are in proper format
        if isinstance(sim_start, str) and len(sim_start) == 10:  # YYYY-MM-DD format
            sim_start = f"{sim_start} 00:00"
        if isinstance(sim_end, str) and len(sim_end) == 10:  # YYYY-MM-DD format
            sim_end = f"{sim_end} 23:00"

        cf.write("!\n! --- DEFINE SIMULATION CONTROLS \n")
        cf.write(f"<case_name>             {self.experiment_id}    ! Simulation case name \n")
        cf.write(f"<sim_start>             {sim_start}    ! Time of simulation start \n")
        cf.write(f"<sim_end>               {sim_end}    ! Time of simulation end \n")
        cf.write(f"<route_opt>             {self.mizu_output_vars}    ! Option for routing schemes \n")
        cf.write(f"<newFileFrequency>      {self.mizu_output_freq}    ! Frequency for new output files \n")

    def _create_topology_variables(self, ncid, shp_river, shp_basin):
        self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', shp_river[self.river_segid_col].values.astype(int), 'Unique ID of each stream segment', '-')
        self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', shp_river[self.river_downsegid_col].values.astype(int), 'ID of the downstream segment', '-')
        self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', shp_river[self.river_slope_col].values.astype(float), 'Segment slope', '-')
        self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', shp_river[self.river_length_col].values.astype(float), 'Segment length', 'm')
        self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', shp_basin[self.basin_gruid_col].values.astype(int), 'Unique hru ID', '-')
        self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', shp_basin[self.basin_hru_to_seg_col].values.astype(int), 'ID of the stream segment to which the HRU discharges', '-')
        self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', shp_basin[self.basin_area_col].values.astype(float), 'HRU area', 'm^2')

    def _process_remap_variables(self, intersected_shape):
        int_rm_id = f"S_1_{self.config_dict.get('RIVER_BASIN_SHP_RM_HRUID')}"
        int_hm_id = f"S_2_{self.config_dict.get('CATCHMENT_SHP_GRUID')}"
        int_weight = 'AP1N'

        intersected_shape = intersected_shape.sort_values(by=[int_rm_id, int_hm_id])

        self.nc_rnhruid = intersected_shape.groupby(int_rm_id).agg({int_rm_id: pd.unique}).values.astype(int)
        self.nc_noverlaps = intersected_shape.groupby(int_rm_id).agg({int_hm_id: 'count'}).values.astype(int)

        multi_nested_list = intersected_shape.groupby(int_rm_id).agg({int_hm_id: list}).values.tolist()
        self.nc_hmgruid = [item for sublist in multi_nested_list for item in sublist[0]]

        multi_nested_list = intersected_shape.groupby(int_rm_id).agg({int_weight: list}).values.tolist()
        self.nc_weight = [item for sublist in multi_nested_list for item in sublist[0]]

    def _create_remap_file(self, intersected_shape, remap_name):
        num_hru = len(intersected_shape[f"S_1_{self.config_dict.get('RIVER_BASIN_SHP_RM_HRUID')}"].unique())
        num_data = len(intersected_shape)

        with nc4.Dataset(self.setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            self._set_remap_attributes(ncid)
            self._create_remap_dimensions(ncid, num_hru, num_data)
            self._create_remap_variables(ncid)

    def _set_remap_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a remapping .nc file for mizuRoute routing')

    def _create_remap_dimensions(self, ncid, num_hru, num_data):
        ncid.createDimension('hru', num_hru)
        ncid.createDimension('data', num_data)

    def _create_remap_variables(self, ncid):
        self._create_and_fill_nc_var(ncid, 'RN_hruId', 'int', 'hru', self.nc_rnhruid, 'River network HRU ID', '-')
        self._create_and_fill_nc_var(ncid, 'nOverlaps', 'int', 'hru', self.nc_noverlaps, 'Number of overlapping HM_HRUs for each RN_HRU', '-')
        self._create_and_fill_nc_var(ncid, 'HM_hruId', 'int', 'data', self.nc_hmgruid, 'ID of overlapping HM_HRUs. Note that SUMMA calls these GRUs', '-')
        self._create_and_fill_nc_var(ncid, 'weight', 'f8', 'data', self.nc_weight, 'Areal weight of overlapping HM_HRUs. Note that SUMMA calls these GRUs', '-')

    def _create_and_fill_nc_var(self, ncid, var_name, var_type, dim, fill_data, long_name, units):
        ncvar = ncid.createVariable(var_name, var_type, (dim,))
        ncvar[:] = fill_data
        ncvar.long_name = long_name
        ncvar.units = units

    def _write_control_file_header(self, cf):
        cf.write("! mizuRoute control file generated by SUMMA public workflow scripts \n")

    def _write_control_file_directories(self, cf):
        experiment_output_summa = self.config_dict.get('EXPERIMENT_OUTPUT_SUMMA')
        experiment_output_mizuroute = self.config_dict.get('EXPERIMENT_OUTPUT_MIZUROUTE')

        if experiment_output_summa == 'default':
            experiment_output_summa = self.project_dir / f"simulations/{self.experiment_id}" / 'SUMMA'
        else:
            experiment_output_summa = Path(experiment_output_summa)

        if experiment_output_mizuroute == 'default' or not experiment_output_mizuroute:
            experiment_output_mizuroute = self.project_dir / f"simulations/{self.experiment_id}" / 'mizuRoute'
        else:
            experiment_output_mizuroute = Path(experiment_output_mizuroute)

        # Ensure output directory exists
        experiment_output_mizuroute.mkdir(parents=True, exist_ok=True)

        cf.write("!\n! --- DEFINE DIRECTORIES \n")
        cf.write(f"<ancil_dir>             {self.setup_dir}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n")
        cf.write(f"<input_dir>             {experiment_output_summa}/    ! Folder that contains runoff data from SUMMA \n")
        cf.write(f"<output_dir>            {experiment_output_mizuroute}/    ! Folder that will contain mizuRoute simulations \n")

    def _write_control_file_parameters(self, cf):
        cf.write("!\n! --- NAMELIST FILENAME \n")
        cf.write(f"<param_nml>             {self.mizu_parameters_file}    ! Spatially constant parameter namelist (should be stored in the ancil_dir) \n")


    def _write_control_file_topology(self, cf):
        cf.write("!\n! --- DEFINE TOPOLOGY FILE \n")
        cf.write(f"<fname_ntopOld>         {self.mizu_topology_file}    ! Name of input netCDF for River Network \n")
        cf.write("<dname_sseg>            seg    ! Dimension name for reach in river network netCDF \n")
        cf.write("<dname_nhru>            hru    ! Dimension name for RN_HRU in river network netCDF \n")
        cf.write("<seg_outlet>            -9999    ! Outlet reach ID at which to stop routing (i.e. use subset of full network). -9999 to use full network \n")
        cf.write("<varname_area>          area    ! Name of variable holding hru area \n")
        cf.write("<varname_length>        length    ! Name of variable holding segment length \n")
        cf.write("<varname_slope>         slope    ! Name of variable holding segment slope \n")
        cf.write("<varname_HRUid>         hruId    ! Name of variable holding HRU id \n")
        cf.write("<varname_hruSegId>      hruToSegId    ! Name of variable holding the stream segment below each HRU \n")
        cf.write("<varname_segId>         segId    ! Name of variable holding the ID of each stream segment \n")
        cf.write("<varname_downSegId>     downSegId    ! Name of variable holding the ID of the next downstream segment \n")

    def _write_control_file_remapping(self, cf):
        cf.write("!\n! --- DEFINE RUNOFF MAPPING FILE \n")
        # Check both config flag and lumped-to-distributed flag
        remap_flag = (
            self.config_dict.get('SETTINGS_MIZU_NEEDS_REMAP', '') or
            getattr(self, 'needs_remap_lumped_distributed', False)
        )
        cf.write(f"<is_remap>              {'T' if remap_flag else 'F'}    ! Logical to indicate runoff needs to be remapped to RN_HRU. T or F \n")

        if remap_flag:
            cf.write(f"<fname_remap>           {self.mizu_remap_file}    ! netCDF name of runoff remapping \n")
            cf.write("<vname_hruid_in_remap>  RN_hruId    ! Variable name for RN_HRUs \n")
            cf.write("<vname_weight>          weight    ! Variable name for areal weights of overlapping HM_HRUs \n")
            cf.write("<vname_qhruid>          HM_hruId    ! Variable name for HM_HRU ID \n")
            cf.write("<vname_num_qhru>        nOverlaps    ! Variable name for a numbers of overlapping HM_HRUs with RN_HRUs \n")
            cf.write("<dname_hru_remap>       hru    ! Dimension name for HM_HRU \n")
            cf.write("<dname_data_remap>      data    ! Dimension name for data \n")

    def _write_control_file_miscellaneous(self, cf):
        cf.write("!\n! --- MISCELLANEOUS \n")
        cf.write(f"<doesBasinRoute>        {self.mizu_within_basin}    ! Hillslope routing options. 0 -> no (already routed by SUMMA), 1 -> use IRF \n")

    def _get_default_time(self, time_key, default_year):
        time_value = self.config_dict.get(time_key)
        if time_value == 'default':
            raw_time = [
                    self.time_start.split('-')[0],  # Get year from full datetime
                    self.time_end.split('-')[0]
                ]
            year = raw_time[0] if default_year == 'start' else raw_time[1]
            return f"{year}-{'01-01 00:00' if default_year == 'start' else '12-31 23:00'}"
        return time_value

    def _pad_string(self, string, pad_to=20):
        return f"{string:{pad_to}}"
