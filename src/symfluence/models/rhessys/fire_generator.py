"""
RHESSys WMFire Input Generator

Handles generation of WMFire (Wildfire Module) inputs for RHESSys, including:
- Fire grid files (patch_grid.txt, dem_grid.txt)
- Fire default parameter files (fire.def)
- Ignition point processing

Extracted from RHESSysPreProcessor for modularity.
"""
import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)


class RHESSysFireGenerator:
    """
    Generates WMFire fire spread inputs for RHESSys.

    Creates the fire grid files and parameter defaults required for
    the RHESSys -firespread flag. Supports georeferenced grid generation,
    distributed multi-patch grids, and lumped single-patch fallbacks.

    Args:
        preprocessor: Parent RHESSysPreProcessor instance providing access
            to configuration, paths, and helper methods.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor

    def setup_wmfire_inputs(self):
        """
        Setup WMFire fire spread inputs for RHESSys.

        Creates the fire grid files required for the -firespread flag:
        - patch_grid.txt: Grid of patch IDs for fire tracking
        - dem_grid.txt: DEM values for fire spread calculations

        Uses FireGridManager for georeferenced grid generation with
        proper CRS and transform metadata. Supports multiple resolutions
        (30, 60, 90m) and optional GeoTIFF output for visualization.
        """
        logger.info("WMFire is enabled. Setting up fire spread inputs...")

        fire_dir = self.pp.fire_dir
        fire_dir.mkdir(parents=True, exist_ok=True)

        patch_grid_file = fire_dir / "patch_grid.txt"
        dem_grid_file = fire_dir / "dem_grid.txt"

        # Get WMFire configuration
        wmfire_config = None
        try:
            if hasattr(self.pp.config.model.rhessys, 'wmfire'):
                wmfire_config = self.pp.config.model.rhessys.wmfire
        except AttributeError:
            pass

        # Try to use FireGridManager for georeferenced grid generation
        use_geospatial = False
        try:
            catchment_path = self.pp.get_catchment_path()
            if catchment_path.exists():
                catchment_gdf = gpd.read_file(catchment_path)
                if len(catchment_gdf) > 0:
                    use_geospatial = True
        except Exception as e:
            logger.warning(f"Could not load catchment for fire grids: {e}")

        if use_geospatial:
            self._setup_geospatial_fire_grids(
                catchment_gdf, patch_grid_file, dem_grid_file, wmfire_config
            )
        elif hasattr(self.pp, '_distributed_patches') and len(self.pp._distributed_patches) > 1:
            # Fallback: use distributed patches from worldfile generation
            self._setup_distributed_fire_grids(patch_grid_file, dem_grid_file)
        else:
            # Fallback: lumped single-patch grid
            self._setup_lumped_fire_grids(patch_grid_file, dem_grid_file)

        # Generate fire.def with correct grid dimensions
        self._generate_fire_defaults()

        logger.info(f"WMFire input files created in {fire_dir}")

    def _setup_geospatial_fire_grids(
        self,
        catchment_gdf: gpd.GeoDataFrame,
        patch_grid_file: Path,
        dem_grid_file: Path,
        wmfire_config
    ):
        """
        Setup georeferenced fire grids using FireGridManager.

        Creates properly georeferenced grids with correct CRS, transform,
        and resolution. Optionally writes GeoTIFF outputs for visualization.
        Also handles ignition point processing.

        Args:
            catchment_gdf: GeoDataFrame with catchment/HRU polygons
            patch_grid_file: Output path for patch grid text file
            dem_grid_file: Output path for DEM grid text file
            wmfire_config: WMFireConfig object with settings
        """
        from symfluence.models.wmfire import FireGridManager

        # Create grid manager
        grid_manager = FireGridManager(self.pp.config, self.pp.logger)
        resolution = grid_manager.resolution
        logger.info(f"Creating fire grids at {resolution}m resolution")

        # Look for DEM if available
        dem_path = None
        dem_candidates = [
            self.pp.project_attributes_dir / 'elevation' / 'dem.tif',
            self.pp.project_attributes_dir / 'dem' / 'dem.tif',
            self.pp.project_dir / 'domain' / 'dem.tif',
        ]
        for candidate in dem_candidates:
            if candidate.exists():
                dem_path = candidate
                logger.info(f"Found DEM: {dem_path}")
                break

        # Generate georeferenced grids
        try:
            patch_grid, dem_grid = grid_manager.create_fire_grid(
                catchment_gdf, dem_path
            )

            # Write text format for RHESSys
            patch_grid_file.write_text(patch_grid.to_text(), encoding='utf-8')
            dem_grid_file.write_text(dem_grid.to_text(), encoding='utf-8')

            # Store grid dimensions for fire.def generation
            self._fire_grid_nrows = patch_grid.nrows
            self._fire_grid_ncols = patch_grid.ncols
            self._fire_grid = patch_grid  # Store for fire.def generation

            logger.info(f"Georeferenced fire grids created: "
                       f"{patch_grid.nrows}x{patch_grid.ncols} @ {resolution}m")

            # Process ignition point
            self._process_ignition_point(patch_grid, wmfire_config)

            # Write GeoTIFF outputs if requested
            write_geotiff = True
            if wmfire_config and hasattr(wmfire_config, 'write_geotiff'):
                write_geotiff = wmfire_config.write_geotiff

            if write_geotiff:
                fire_dir = patch_grid_file.parent
                try:
                    patch_grid.to_geotiff(fire_dir / "patch_grid.tif")
                    dem_grid.to_geotiff(fire_dir / "dem_grid.tif")
                    logger.info("GeoTIFF outputs written for visualization")
                except Exception as e:
                    logger.warning(f"Could not write GeoTIFF outputs: {e}")

        except Exception as e:
            logger.warning(f"Geospatial grid generation failed: {e}, falling back to simple grid")
            import traceback
            logger.debug(traceback.format_exc())
            # Fall back to simple grid generation
            if hasattr(self.pp, '_distributed_patches') and len(self.pp._distributed_patches) > 1:
                self._setup_distributed_fire_grids(patch_grid_file, dem_grid_file)
            else:
                self._setup_lumped_fire_grids(patch_grid_file, dem_grid_file)

    def _process_ignition_point(self, patch_grid, wmfire_config):
        """
        Process ignition point from configuration.

        Loads ignition point from shapefile or coordinates, converts to
        grid indices, and optionally writes ignition shapefile.

        Args:
            patch_grid: FireGrid object with grid metadata
            wmfire_config: WMFireConfig object with ignition settings
        """
        from symfluence.models.wmfire import IgnitionManager

        ignition_mgr = IgnitionManager(self.pp.config, self.pp.logger)
        ignition = ignition_mgr.get_ignition_point()

        if ignition is None:
            logger.info("No ignition point specified, using random ignition (-1, -1)")
            self._ignition_row = -1
            self._ignition_col = -1
            return

        logger.info(f"Processing ignition point: {ignition.name} "
                   f"({ignition.latitude:.4f}, {ignition.longitude:.4f})")

        # Convert to grid indices
        row, col = ignition_mgr.convert_to_grid_indices(
            ignition,
            patch_grid.transform,
            patch_grid.crs,
            patch_grid.nrows,
            patch_grid.ncols
        )

        self._ignition_row = row
        self._ignition_col = col

        # Write ignition shapefile if coordinates were from config (not already a shapefile)
        if ignition.source == 'config':
            ignition_dir = self.pp.domain_path / 'shapefiles' / 'ignitions'
            ignition_mgr.write_ignition_shapefile(
                ignition,
                ignition_dir,
                filename=f"{ignition.name}.shp"
            )

        logger.info(f"Ignition grid indices: row={row}, col={col}")

    def _setup_distributed_fire_grids(self, patch_grid_file: Path, dem_grid_file: Path):
        """
        Setup fire grids for distributed multi-patch domains.

        Creates a grid representing the spatial arrangement of patches.
        For elevation-based discretization, arranges patches in a column
        from lowest to highest elevation.
        """
        patches = self.pp._distributed_patches
        num_patches = len(patches)

        # Sort patches by elevation (lowest at bottom/south)
        patches_sorted = sorted(patches, key=lambda p: p['elev'])

        # For simplicity, create a 3-column grid with patches arranged by elevation
        # This gives each patch 3 cells for more realistic fire spread
        ncols = 3
        nrows = num_patches

        logger.info(f"Creating distributed fire grid: {nrows}x{ncols} ({num_patches} patches)")

        # Build patch grid (each row is one patch)
        # RHESSys expects NO header line and tab-separated values
        patch_lines = []
        dem_lines = []

        for patch_info in patches_sorted:
            patch_id = patch_info['patch_id']
            elev = patch_info['elev']
            # Each patch gets a full row (3 cells), tab-separated
            patch_lines.append(f"{patch_id}\t{patch_id}\t{patch_id}")
            dem_lines.append(f"{elev:.1f}\t{elev:.1f}\t{elev:.1f}")

        patch_grid_file.write_text('\n'.join(patch_lines) + '\n', encoding='utf-8')
        dem_grid_file.write_text('\n'.join(dem_lines) + '\n', encoding='utf-8')

        # Store grid dimensions for fire.def generation
        self._fire_grid_nrows = nrows
        self._fire_grid_ncols = ncols

        logger.info(f"Distributed fire grids created with {num_patches} patches")
        for i, p in enumerate(patches_sorted):
            logger.debug(f"  Row {i+1}: patch_id={p['patch_id']}, elev={p['elev']:.0f}m")

    def _setup_lumped_fire_grids(self, patch_grid_file: Path, dem_grid_file: Path):
        """
        Setup fire grids for lumped single-patch domains.
        """
        # Get elevation if available
        try:
            catchment_path = self.pp.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                elev = float(gdf.get('elev_mean', [1500])[0]) if 'elev_mean' in gdf.columns else 1500.0
            else:
                elev = 1500.0
        except (FileNotFoundError, KeyError, IndexError, ValueError):
            elev = 1500.0

        # Simple 3x3 grid for single patch
        # RHESSys expects NO header line and tab-separated values
        patch_content = """1\t1\t1
1\t1\t1
1\t1\t1
"""
        dem_content = f"""{elev:.1f}\t{elev:.1f}\t{elev:.1f}
{elev:.1f}\t{elev:.1f}\t{elev:.1f}
{elev:.1f}\t{elev:.1f}\t{elev:.1f}
"""

        patch_grid_file.write_text(patch_content, encoding='utf-8')
        dem_grid_file.write_text(dem_content, encoding='utf-8')

        # Store grid dimensions for fire.def generation
        self._fire_grid_nrows = 3
        self._fire_grid_ncols = 3

        logger.info("Lumped fire grids created (3x3, single patch)")

    def _generate_fire_defaults(self):
        """
        Generate fire.def file with grid dimensions for WMFire.

        The fire.def file specifies fire spread parameters including
        n_rows and n_cols which must match the fire grid dimensions.
        Uses FireDefGenerator for dynamic parameter generation based
        on WMFire configuration.
        """
        fire_def_file = self.pp.defs_dir / "fire.def"

        # Get grid dimensions
        nrows = getattr(self, '_fire_grid_nrows', 3)
        ncols = getattr(self, '_fire_grid_ncols', 3)

        # Get ignition indices (default -1 for random)
        ignition_row = getattr(self, '_ignition_row', -1)
        ignition_col = getattr(self, '_ignition_col', -1)

        # Try to use FireDefGenerator for enhanced generation
        try:
            from symfluence.models.wmfire import FireDefGenerator

            fire_def_gen = FireDefGenerator(self.pp.config, self.pp.logger)

            # Use stored grid if available, otherwise create dummy
            if hasattr(self, '_fire_grid'):
                fire_def_gen.write_fire_def(
                    fire_def_file,
                    self._fire_grid,
                    ignition_row=ignition_row,
                    ignition_col=ignition_col
                )
            else:
                # Generate with just dimensions
                content = fire_def_gen.generate_default_fire_def(
                    n_rows=nrows,
                    n_cols=ncols,
                    output_path=fire_def_file
                )
            return

        except ImportError:
            logger.debug("FireDefGenerator not available, using fallback")
        except Exception as e:
            logger.warning(f"FireDefGenerator failed: {e}, using fallback")

        # Fallback: Generate fire.def manually
        # Fire defaults content based on RHESSys WMFire construct_fire_defaults.c
        content = f"""1    fire_parm_ID
30.0    ndays_average
3.9    load_k1
0.07    load_k2
0.91    slope_k1
1.0    slope_k2
3.8    moisture_k1
0.27    moisture_k2
0.87    winddir_k1
0.48    winddir_k2
3.8    moisture_ign_k1
0.27    moisture_ign_k2
1.0    windmax
-1    ignition_col
-1    ignition_row
10.0    ignition_tmin
0    fire_verbose
0    fire_write
0    fire_in_buffer
{nrows}    n_rows
{ncols}    n_cols
9    spread_calc_type
0.494    mean_log_wind
0.654    sd_log_wind
1.71    mean1_rvm
-1.91    mean2_rvm
2.37    kappa1_rvm
2.38    kappa2_rvm
0.411    p_rvm
1.0    ign_def_mod
0.8    veg_k1
10.0    veg_k2
1.0    mean_ign
0    ran_seed
0    calc_fire_effects
0    include_wui
0    fire_size_name
0.0    wind_shift
"""
        fire_def_file.write_text(content, encoding='utf-8')
        logger.info(f"Fire defaults written: {fire_def_file} (grid: {nrows}x{ncols})")
