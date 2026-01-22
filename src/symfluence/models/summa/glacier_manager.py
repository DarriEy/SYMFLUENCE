"""
SUMMA Glacier Attributes Manager.

This module provides the GlacierAttributesManager class for processing glacier
data into SUMMA-compatible NetCDF files for glacier simulations.

Based on Ashley Medin's glacier preprocessing workflow.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import geopandas as gpd
import netCDF4 as nc4
import numpy as np
import rasterio
import xarray as xr

from symfluence.core.mixins import ConfigMixin


class GlacierAttributesManager(ConfigMixin):
    """
    Manager for SUMMA glacier preprocessing.

    Handles the transformation of glacier data into SUMMA-compatible
    NetCDF files including:
    - attributes_glac.nc (HRU-scale glacier attributes)
    - attributes_glacBedTopo.nc (grid-scale bed topography)
    - coldState_glac.nc (HRU-scale initial glacier state)
    - coldState_glacSurfTopo.nc (grid-scale surface topography)

    Based on Ashley Medin's glacier preprocessing workflow.
    """

    # Domain types following Ashley's convention
    DOMAIN_UPLAND = 1
    DOMAIN_GLACIER_CLEAN_1 = 2  # Accumulation zone
    DOMAIN_GLACIER_CLEAN_2 = 3  # Ablation zone (clean)
    DOMAIN_GLACIER_DEBRIS = 4
    DOMAIN_WETLAND = 5

    # Ice layer depths (meters) - from Ashley's code
    ICE_LAYER_DEPTHS = np.array([0.15, 0.45, 2.25, 7.0, 30.0])
    N_GLCE_LAYERS = 5  # Number of glacier ice layers
    N_SOIL_GLAC = 3    # Number of soil layers for debris-covered glacier

    # Glacier inclusion threshold
    GLACIER_THRESHOLD = 0.8

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        domain_name: str,
        dem_path: Path,
        project_dir: Optional[Path] = None
    ):
        """
        Initialize the GlacierAttributesManager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            domain_name: Name of the domain (e.g., 'Wolverine')
            dem_path: Path to DEM file for elevation data
            project_dir: Path to project directory (for shapefiles)
        """
        # Import here to avoid circular imports

        from symfluence.core.config.models import SymfluenceConfig



        # Auto-convert dict to typed config for backward compatibility

        if isinstance(config, dict):

            try:

                self._config = SymfluenceConfig(**config)

            except Exception:

                # Fallback for partial configs (e.g., in tests)

                self._config = config

        else:

            self._config = config
        self.logger = logger
        self.domain_name = domain_name
        self.dem_path = dem_path
        self.project_dir = project_dir

    def has_glacier_data(self, glacier_dir: Path) -> bool:
        """
        Check if glacier preprocessing data exists.

        Checks for either:
        1. Domain type intersection shapefiles (Ashley's workflow)
        2. Glacier raster files (fallback)

        Args:
            glacier_dir: Directory containing glacier rasters

        Returns:
            True if glacier data is available
        """
        # First check for intersection shapefiles (preferred)
        if self.project_dir:
            intersect_base = self.project_dir / 'shapefiles' / 'catchment_intersection'
            domain_type_shp = intersect_base / 'with_domain_type' / 'catchment_with_domain_type.shp'
            dem_domain_shp = intersect_base / 'with_dem_domain' / 'catchment_with_dem_domain.shp'

            if domain_type_shp.exists() and dem_domain_shp.exists():
                # Check if these have domain type columns
                try:
                    gdf = gpd.read_file(domain_type_shp)
                    if any(col.startswith('domType_') for col in gdf.columns):
                        self.logger.debug("Found glacier domain type shapefiles")
                        return True
                except Exception:
                    pass

        # Fallback: check for raster files
        if glacier_dir.exists():
            required = ['domain_type.tif', 'hru_id.tif', 'rgi_id.tif']
            for raster_name in required:
                raster_file = self._get_raster_path(glacier_dir, raster_name)
                if raster_file is None:
                    return False
            return True

        return False

    def _get_raster_path(self, glacier_dir: Path, raster_name: str) -> Optional[Path]:
        """Get raster file path, checking naming conventions."""
        for prefix in [f"domain_{self.domain_name}_", f"{self.domain_name}_"]:
            raster_file = glacier_dir / f"{prefix}{raster_name}"
            if raster_file.exists():
                return raster_file
        return None

    def process_glacier_attributes(
        self,
        glacier_dir: Path,
        settings_dir: Path,
        base_attributes_file: Path,
        base_coldstate_file: Optional[Path] = None
    ) -> bool:
        """
        Process glacier data into SUMMA NetCDF files.

        Args:
            glacier_dir: Directory containing glacier rasters
            settings_dir: SUMMA settings directory for output
            base_attributes_file: Base attributes.nc file
            base_coldstate_file: Base coldState.nc file (optional)

        Returns:
            True if processing succeeded
        """
        self.logger.info("Processing glacier attributes (Ashley's methodology)")

        try:
            # Load base attributes for HRU info
            with xr.open_dataset(base_attributes_file) as base_attrs:
                hru_ids = base_attrs['hruId'].values
                gru_ids = base_attrs['gruId'].values
                hru_areas = base_attrs['HRUarea'].values
                hru_elevs = base_attrs['elevation'].values
                hru2gru = base_attrs['hru2gruId'].values

            # Try shapefile-based processing first (Ashley's workflow)
            if self.project_dir:
                intersect_base = self.project_dir / 'shapefiles' / 'catchment_intersection'
                domain_type_shp = intersect_base / 'with_domain_type' / 'catchment_with_domain_type.shp'
                dem_domain_shp = intersect_base / 'with_dem_domain' / 'catchment_with_dem_domain.shp'

                if domain_type_shp.exists() and dem_domain_shp.exists():
                    return self._process_from_shapefiles(
                        domain_type_shp, dem_domain_shp, glacier_dir,
                        settings_dir, base_attributes_file, base_coldstate_file,
                        hru_ids, gru_ids, hru_areas, hru_elevs, hru2gru
                    )

            # Fallback to raster-based processing
            return self._process_from_rasters(
                glacier_dir, settings_dir, base_attributes_file,
                hru_ids, gru_ids, hru_areas
            )

        except Exception as e:
            self.logger.error(f"Error processing glacier attributes: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _process_from_shapefiles(
        self,
        domain_type_shp: Path,
        dem_domain_shp: Path,
        glacier_dir: Path,
        settings_dir: Path,
        base_attributes_file: Path,
        base_coldstate_file: Optional[Path],
        hru_ids: np.ndarray,
        gru_ids: np.ndarray,
        hru_areas: np.ndarray,
        hru_elevs: np.ndarray,
        hru2gru: np.ndarray
    ) -> bool:
        """
        Process glacier attributes from intersection shapefiles (Ashley's workflow).
        """
        self.logger.info("Using shapefile-based glacier preprocessing")

        # Load shapefiles
        shp_area = gpd.read_file(domain_type_shp)
        shp_elev = gpd.read_file(dem_domain_shp)

        hru_id_col = self._get_config_value(lambda: self.config.paths.catchment_hruid, default='HRU_ID', dict_key='CATCHMENT_SHP_HRUID')
        num_hru = len(hru_ids)
        len(gru_ids)

        # Determine available domain types from columns
        domain_cols = [c for c in shp_area.columns if c.startswith('domType_')]
        available_domains = sorted([int(c.split('_')[1]) for c in domain_cols])
        self.logger.info(f"Found domain types: {available_domains}")

        # Check for debris and clean zones
        has_debris = 4 in available_domains
        has_clean = 2 in available_domains or 3 in available_domains

        # Build domain arrays following Ashley's logic
        ndom0 = 5  # Maximum possible domains
        DOMarea = np.zeros((1, num_hru, ndom0), dtype='f8')
        DOMelev = np.full((1, num_hru, ndom0), -9999.0, dtype='f8')
        domType = np.zeros((1, num_hru, ndom0), dtype='i4')

        ndom = 0
        for i, hru_id in enumerate(hru_ids):
            ind = 0
            shp_mask = shp_elev[hru_id_col].astype(int) == hru_id
            shp_mask_count = shp_area[hru_id_col].astype(int) == hru_id

            if not shp_mask.any():
                continue

            for domain_type in range(1, ndom0 + 1):
                elev_col = f'elv_mean_{domain_type}'
                area_col = f'domType_{domain_type}'

                # Check if column exists and has valid value
                valid = False
                if elev_col in shp_elev.columns:
                    val = shp_elev.loc[shp_mask, elev_col].values
                    if len(val) > 0 and val[0] is not None:
                        if not np.isnan(val[0]) and val[0] >= 0:
                            valid = True

                if valid:
                    DOMelev[0, i, ind] = shp_elev.loc[shp_mask, elev_col].values[0]
                    if area_col in shp_area.columns:
                        DOMarea[0, i, ind] = shp_area.loc[shp_mask_count, area_col].values[0]
                    domType[0, i, ind] = domain_type
                    ind += 1
                else:
                    # Ashley's rules for including empty domains
                    if domain_type == 1:  # Always include upland
                        DOMelev[0, i, ind] = -9999.0
                        DOMarea[0, i, ind] = 0
                        domType[0, i, ind] = domain_type
                        ind += 1
                    elif domain_type == 2 and has_debris and not has_clean:
                        # Need clean zone if debris exists
                        DOMelev[0, i, ind] = -9999.0
                        DOMarea[0, i, ind] = 0
                        domType[0, i, ind] = domain_type
                        ind += 1
                    elif domain_type == 4 and has_debris:
                        # Need debris zone if debris exists
                        DOMelev[0, i, ind] = -9999.0
                        DOMarea[0, i, ind] = 0
                        domType[0, i, ind] = domain_type
                        ind += 1

                ndom = max(ndom, ind)

        # Scale areas by HRU area
        for i in range(num_hru):
            total_count = np.sum(DOMarea[0, i, :])
            if total_count > 0:
                DOMarea[0, i, :] = hru_areas[i] * DOMarea[0, i, :] / total_count

        # Apply Ashley's domain merging logic:
        # If two clean domains exist and HRU is not the only one in glacier, merge them
        # Also relabel domain 3 to domain 2 when there's no domain 2
        scalarAblFrac = np.zeros((1, num_hru, ndom0), dtype='f8')
        scalarAblFrac[(domType == self.DOMAIN_GLACIER_CLEAN_2) | (domType == self.DOMAIN_GLACIER_DEBRIS)] = 1.0

        for i in range(num_hru):
            # Find indices of domains 2 and 3
            dom2_idx = np.where(domType[0, i, :] == self.DOMAIN_GLACIER_CLEAN_1)[0]
            dom3_idx = np.where(domType[0, i, :] == self.DOMAIN_GLACIER_CLEAN_2)[0]

            if len(dom2_idx) > 0 and len(dom3_idx) > 0:
                # Both clean zones exist - merge them into zone 2
                idx2, idx3 = dom2_idx[0], dom3_idx[0]
                if DOMarea[0, i, idx2] > 0 and DOMarea[0, i, idx3] > 0:
                    # Weighted average of elevation and ablation fraction
                    total_area = DOMarea[0, i, idx2] + DOMarea[0, i, idx3]
                    DOMelev[0, i, idx2] = (DOMelev[0, i, idx2] * DOMarea[0, i, idx2] +
                                           DOMelev[0, i, idx3] * DOMarea[0, i, idx3]) / total_area
                    scalarAblFrac[0, i, idx2] = (scalarAblFrac[0, i, idx2] * DOMarea[0, i, idx2] +
                                                  scalarAblFrac[0, i, idx3] * DOMarea[0, i, idx3]) / total_area
                    DOMarea[0, i, idx2] = total_area
                elif DOMarea[0, i, idx3] > 0:
                    DOMelev[0, i, idx2] = DOMelev[0, i, idx3]
                    scalarAblFrac[0, i, idx2] = scalarAblFrac[0, i, idx3]
                    DOMarea[0, i, idx2] = DOMarea[0, i, idx3]

                # Remove domain 3 by shifting remaining domains
                for j in range(idx3, ndom - 1):
                    DOMelev[0, i, j] = DOMelev[0, i, j + 1]
                    DOMarea[0, i, j] = DOMarea[0, i, j + 1]
                    domType[0, i, j] = domType[0, i, j + 1]
                    scalarAblFrac[0, i, j] = scalarAblFrac[0, i, j + 1]
                # Clear last slot
                DOMelev[0, i, ndom - 1] = -9999.0
                DOMarea[0, i, ndom - 1] = 0
                domType[0, i, ndom - 1] = 0
                scalarAblFrac[0, i, ndom - 1] = 0

            elif len(dom3_idx) > 0 and len(dom2_idx) == 0:
                # Only domain 3 exists - relabel to domain 2
                idx3 = dom3_idx[0]
                domType[0, i, idx3] = self.DOMAIN_GLACIER_CLEAN_1

        # Remove empty trailing domains (where domType = 0 for all HRUs)
        while ndom > 1 and np.all(domType[0, :, ndom - 1] == 0):
            ndom -= 1

        # Trim to actual number of domains
        DOMarea = DOMarea[:, :, :ndom]
        DOMelev = DOMelev[:, :, :ndom]
        domType = domType[:, :, :ndom]
        scalarAblFrac = scalarAblFrac[:, :, :ndom]

        self.logger.info(f"Using {ndom} domain types")

        # Calculate glacier grid info from rasters
        grid_info = self._calculate_grid_info(glacier_dir, hru_ids, gru_ids, hru2gru)

        # Create output files
        self._create_attributes_glac_from_shp(
            settings_dir, base_attributes_file, grid_info['nGrid']
        )
        self._create_attributes_glacBedTopo_from_rasters(
            settings_dir, glacier_dir, gru_ids, grid_info
        )
        self._create_coldState_glac_from_shp(
            settings_dir, base_coldstate_file,
            hru_ids, gru_ids, ndom, domType, DOMarea, DOMelev, scalarAblFrac, grid_info
        )
        self._create_coldState_glacSurfTopo_from_rasters(
            settings_dir, glacier_dir, gru_ids
        )

        self.logger.info("Successfully created all glacier NetCDF files")
        return True

    def _calculate_grid_info(
        self,
        glacier_dir: Path,
        hru_ids: np.ndarray,
        gru_ids: np.ndarray,
        hru2gru: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate glacier grid information from rasters."""
        num_gru = len(gru_ids)

        # Load rasters
        rgi_path = self._get_raster_path(glacier_dir, 'rgi_id.tif')
        hru_path = self._get_raster_path(glacier_dir, 'hru_id.tif')

        if rgi_path is None or hru_path is None:
            # Return defaults
            return {
                'nGrid': np.ones(num_gru, dtype='i4'),
                'gridId': np.zeros((1, num_gru), dtype='i8'),
                'basin__GlacierStorage': np.zeros(num_gru, dtype='f8')
            }

        with rasterio.open(rgi_path) as rgi_src, rasterio.open(hru_path) as hru_src:
            rgi_data = rgi_src.read(1)
            hru_data = hru_src.read(1)
            cell_area = abs(rgi_src.transform[0] * rgi_src.transform[4])

            # For geographic CRS, convert to approximate m^2
            if rgi_src.crs and rgi_src.crs.is_geographic:
                bounds = rgi_src.bounds
                center_lat = (bounds.bottom + bounds.top) / 2
                m_per_deg = 111000 * np.cos(np.radians(center_lat))
                cell_area = cell_area * m_per_deg * 111000

        nGrid = np.zeros(num_gru, dtype='i4')
        max_glac = len(np.unique(rgi_data[rgi_data > 0]))
        gridId = np.zeros((max(max_glac, 1), num_gru), dtype='i8')
        basin__GlacierStorage = np.zeros(num_gru, dtype='f8')

        # Load surface elevation from DEM for ice volume calculation
        surf_data = None
        if self.dem_path.exists():
            with rasterio.open(self.dem_path) as src:
                surf_data = src.read(1).astype(np.float64)
                if surf_data.shape != rgi_data.shape:
                    surf_data = self._resample_to_grid(surf_data, rgi_data.shape)

        for i, gru_id in enumerate(gru_ids):
            # Get HRUs in this GRU
            hrus_in_gru = hru_ids[hru2gru == gru_id]

            # Get cells in GRU
            gru_mask = np.isin(hru_data, hrus_in_gru)

            # Calculate glacier storage (ice volume in Gt)
            # Ashley's formula: cell_area * sum(surface - bed) * ice_density * 1e-9
            # For now, estimate ice thickness as 10% of surface elevation (rough approximation)
            # Real calculation would need bed elevation raster
            if surf_data is not None:
                surf_data[gru_mask]
                # Estimate ice thickness ~ 50m average for glaciers (rough approximation)
                # Better: use actual bed elevation if available
                ice_thickness = np.where(rgi_data[gru_mask] > 0, 50.0, 0.0)  # 50m average
                basin__GlacierStorage[i] = cell_area * np.sum(ice_thickness) * 916.7e-9  # Gt

            # Get glaciers in GRU
            glac_ids_in_gru = rgi_data[gru_mask]
            unique_glacs = np.unique(glac_ids_in_gru[glac_ids_in_gru > 0])

            # Filter by threshold
            valid_glacs = []
            for gid in unique_glacs:
                gru_count = np.sum(glac_ids_in_gru == gid)
                total_count = np.sum(rgi_data == gid)
                if gru_count >= self.GLACIER_THRESHOLD * total_count:
                    valid_glacs.append(gid)

            nGrid[i] = len(valid_glacs)
            for j, gid in enumerate(valid_glacs[:max_glac]):
                gridId[j, i] = gid

        return {
            'nGrid': nGrid,
            'gridId': gridId,
            'basin__GlacierStorage': basin__GlacierStorage
        }

    def _create_attributes_glac_from_shp(
        self,
        settings_dir: Path,
        base_attributes_file: Path,
        nGrid: np.ndarray
    ) -> None:
        """Create attributes_glac.nc by extending base attributes."""
        output_file = settings_dir / 'attributes_glac.nc'
        shutil.copy2(base_attributes_file, output_file)

        with nc4.Dataset(output_file, 'r+') as ds:
            len(ds.dimensions['gru'])

            # nGlac = nGrid (number of glacier grids per GRU)
            if 'nGlac' not in ds.variables:
                ds.createVariable('nGlac', 'i4', ('gru',), fill_value=-999)
            ds['nGlac'][:] = nGrid

            # nWtld = 0 (no wetlands)
            if 'nWtld' not in ds.variables:
                ds.createVariable('nWtld', 'i4', ('gru',), fill_value=-999)
            ds['nWtld'][:] = 0

            ds.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")} - Glacier attributes added')

        self.logger.info(f"Created attributes_glac.nc with nGlac={nGrid}")

    def _create_attributes_glacBedTopo_from_rasters(
        self,
        settings_dir: Path,
        glacier_dir: Path,
        gru_ids: np.ndarray,
        grid_info: Dict[str, Any]
    ) -> None:
        """Create attributes_glacBedTopo.nc from raster data."""
        output_file = settings_dir / 'attributes_glacBedTopo.nc'

        # Load rasters
        domain_type_path = self._get_raster_path(glacier_dir, 'domain_type.tif')
        hru_path = self._get_raster_path(glacier_dir, 'hru_id.tif')

        if domain_type_path is None:
            self.logger.warning("No domain_type raster, skipping glacBedTopo")
            return

        with rasterio.open(domain_type_path) as src:
            domain_data = src.read(1)
            transform = src.transform
            ygrid, xgrid = src.shape
            dx = abs(transform[0])
            dy = abs(transform[4])

            # Convert to meters if geographic
            if src.crs and src.crs.is_geographic:
                bounds = src.bounds
                center_lat = (bounds.bottom + bounds.top) / 2
                m_per_deg_x = 111000 * np.cos(np.radians(center_lat))
                m_per_deg_y = 111000
                dx = dx * m_per_deg_x
                dy = dy * m_per_deg_y

        # Glacier mask (domain_type > 1 is glacier)
        glacier_mask = (domain_data > 1).astype(np.int32)

        # Load HRU mapping
        if hru_path:
            with rasterio.open(hru_path) as src:
                cell2hru = src.read(1).astype(np.int64)
        else:
            cell2hru = np.ones((ygrid, xgrid), dtype=np.int64)

        # Load bed elevation from DEM
        if self.dem_path.exists():
            with rasterio.open(self.dem_path) as src:
                bed_elev = src.read(1).astype(np.float64)
                if bed_elev.shape != (ygrid, xgrid):
                    bed_elev = self._resample_to_grid(bed_elev, (ygrid, xgrid))
        else:
            bed_elev = np.zeros((ygrid, xgrid), dtype=np.float64)

        num_gru = len(gru_ids)
        num_grid = 1

        with nc4.Dataset(output_file, 'w', format='NETCDF4') as ds:
            ds.createDimension('gru', num_gru)
            ds.createDimension('grid', num_grid)
            ds.createDimension('xgrid', xgrid)
            ds.createDimension('ygrid', ygrid)

            gruId_var = ds.createVariable('gruId', 'i8', ('gru',), fill_value=-999)
            gruId_var.long_name = 'GRU ID'
            gruId_var[:] = gru_ids

            nGrid_var = ds.createVariable('nGrid', 'i4', ('gru',), fill_value=-999)
            nGrid_var[:] = grid_info['nGrid']

            gridId_var = ds.createVariable('gridId', 'i8', ('grid', 'gru'), fill_value=-999)
            gridId_var[:, :] = grid_info['gridId'][0:1, :]

            nx_var = ds.createVariable('nx', 'i4', ('grid', 'gru'), fill_value=-999)
            nx_var[:, :] = xgrid

            ny_var = ds.createVariable('ny', 'i4', ('grid', 'gru'), fill_value=-999)
            ny_var[:, :] = ygrid

            dx_var = ds.createVariable('dx', 'f8', ('grid', 'gru'), fill_value=-999.)
            dx_var[:, :] = dx

            dy_var = ds.createVariable('dy', 'f8', ('grid', 'gru'), fill_value=-999.)
            dy_var[:, :] = dy

            bed_var = ds.createVariable('bed_elev', 'f8', ('ygrid', 'xgrid', 'grid', 'gru'), fill_value=-999.)
            bed_var[:, :, 0, 0] = bed_elev

            mask_var = ds.createVariable('glacierMask', 'i4', ('ygrid', 'xgrid', 'grid', 'gru'), fill_value=-999)
            mask_var[:, :, 0, 0] = glacier_mask

            cell2hru_var = ds.createVariable('cell2hruId', 'i8', ('ygrid', 'xgrid', 'grid', 'gru'), fill_value=-999)
            cell2hru_var[:, :, 0, 0] = cell2hru

        self.logger.info(f"Created attributes_glacBedTopo.nc: grid={xgrid}x{ygrid}")

    def _create_coldState_glac_from_shp(
        self,
        settings_dir: Path,
        base_coldstate_file: Optional[Path],
        hru_ids: np.ndarray,
        gru_ids: np.ndarray,
        ndom: int,
        domType: np.ndarray,
        DOMarea: np.ndarray,
        DOMelev: np.ndarray,
        scalarAblFrac: np.ndarray,
        grid_info: Dict[str, Any]
    ) -> None:
        """Create coldState_glac.nc following Ashley's methodology."""
        output_file = settings_dir / 'coldState_glac.nc'

        num_hru = len(hru_ids)
        num_gru = len(gru_ids)
        num_glac = max(int(grid_info['nGrid'].max()), 1)

        # Layer dimensions
        scalarv = 1
        midSoil = 8
        midToto = 8
        ifcToto = 9

        # Initialize layer counts per domain
        nSnow = np.zeros((1, num_hru, ndom), dtype='f8')
        nLake = np.zeros((1, num_hru, ndom), dtype='f8')
        nSoil = np.zeros((1, num_hru, ndom), dtype='f8')
        nGlce = np.zeros((1, num_hru, ndom), dtype='f8')

        # Set layer counts based on domain type
        for i in range(num_hru):
            for j in range(ndom):
                dt = domType[0, i, j]
                if dt == self.DOMAIN_UPLAND:
                    nSoil[0, i, j] = 8
                elif dt in [self.DOMAIN_GLACIER_CLEAN_1, self.DOMAIN_GLACIER_CLEAN_2]:
                    nGlce[0, i, j] = self.N_GLCE_LAYERS
                elif dt == self.DOMAIN_GLACIER_DEBRIS:
                    nSoil[0, i, j] = self.N_SOIL_GLAC
                    nGlce[0, i, j] = self.N_GLCE_LAYERS
                elif dt == self.DOMAIN_WETLAND:
                    nLake[0, i, j] = 5
                    nSoil[0, i, j] = 3

        # Read upland layer structure from base coldstate file if available
        upland_iLayerHeight = None
        if base_coldstate_file and base_coldstate_file.exists():
            with nc4.Dataset(base_coldstate_file) as base_cold:
                if 'iLayerHeight' in base_cold.variables:
                    upland_iLayerHeight = base_cold['iLayerHeight'][:, :].T  # Transpose to (hru, layer)

        # Initialize layer heights
        iLayerHeight = np.zeros((ifcToto, num_hru, ndom), dtype='f8')
        for i in range(num_hru):
            for j in range(ndom):
                dt = domType[0, i, j]
                if dt == self.DOMAIN_UPLAND:
                    # Use upland structure from base coldstate if available
                    if upland_iLayerHeight is not None and i < upland_iLayerHeight.shape[0]:
                        n_layers = min(ifcToto, upland_iLayerHeight.shape[1])
                        iLayerHeight[:n_layers, i, j] = upland_iLayerHeight[i, :n_layers]
                elif dt in [self.DOMAIN_GLACIER_CLEAN_1, self.DOMAIN_GLACIER_CLEAN_2]:
                    # Ice layers only - cumulative heights
                    iLayerHeight[:self.N_GLCE_LAYERS + 1, i, j] = np.concatenate([[0], np.cumsum(self.ICE_LAYER_DEPTHS)])
                elif dt == self.DOMAIN_GLACIER_DEBRIS:
                    # Soil + ice layers - use Ashley's debris soil depths
                    debris_soil_depths = np.array([0.014, 0.042, 0.07])  # Approximate Ashley's values
                    depths = np.concatenate([debris_soil_depths, self.ICE_LAYER_DEPTHS])
                    n_layers = min(ifcToto, len(depths) + 1)
                    iLayerHeight[:n_layers, i, j] = np.concatenate([[0], np.cumsum(depths)])[:n_layers]

        mLayerDepth = np.diff(iLayerHeight, axis=0)
        mLayerDepth[mLayerDepth < 0] = 0

        # Calculate glacier areas using scalarAblFrac for weighting
        # scalarAblFrac = 1 for ablation zone, 0 for accumulation zone
        glacierAblArea = np.zeros((num_glac, num_gru), dtype='f8')
        glacierAccArea = np.zeros((num_glac, num_gru), dtype='f8')

        # Sum ablation/accumulation areas from DOMarea weighted by scalarAblFrac
        for i in range(num_hru):
            for j in range(ndom):
                if domType[0, i, j] in [self.DOMAIN_GLACIER_CLEAN_1, self.DOMAIN_GLACIER_CLEAN_2, self.DOMAIN_GLACIER_DEBRIS]:
                    abl_frac = scalarAblFrac[0, i, j]
                    glacierAblArea[0, 0] += DOMarea[0, i, j] * abl_frac
                    glacierAccArea[0, 0] += DOMarea[0, i, j] * (1.0 - abl_frac)

        # Default state values
        states = {
            'scalarCanopyIce': 0.0,
            'scalarCanopyLiq': 0.0,
            'scalarSnowDepth': 0.0,
            'scalarSWE': 0.0,
            'scalarSfcMeltPond': 0.0,
            'scalarAquiferStorage': 1.0,
            'scalarSnowAlbedo': 0.8,
            'scalarCanairTemp': 283.16,
            'scalarCanopyTemp': 283.16,
            'glacMass4AreaChange': 0.0,
            'scalarGlceWE': 0.0,
            'dt_init': 3600.0,
        }

        with nc4.Dataset(output_file, 'w', format='NETCDF4') as ds:
            # Create dimensions
            ds.createDimension('hru', num_hru)
            ds.createDimension('gru', num_gru)
            ds.createDimension('scalarv', scalarv)
            ds.createDimension('midSoil', midSoil)
            ds.createDimension('midToto', midToto)
            ds.createDimension('ifcToto', ifcToto)
            ds.createDimension('dom', ndom)
            ds.createDimension('glac', num_glac)

            # ID variables
            hruId_var = ds.createVariable('hruId', 'i8', ('hru',), fill_value=-999)
            hruId_var.long_name = 'USGS HUC12 ID'
            hruId_var[:] = hru_ids

            gruId_var = ds.createVariable('gruId', 'i8', ('gru',), fill_value=-999)
            gruId_var.long_name = 'GRU ID'
            gruId_var[:] = gru_ids

            # Layer structure
            ds.createVariable('iLayerHeight', 'f8', ('ifcToto', 'hru', 'dom'), fill_value=-999.)[:] = iLayerHeight
            ds.createVariable('mLayerDepth', 'f8', ('midToto', 'hru', 'dom'), fill_value=-999.)[:] = mLayerDepth

            # Domain counts
            ds.createVariable('nSnow', 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)[:] = nSnow
            ds.createVariable('nLake', 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)[:] = nLake
            ds.createVariable('nSoil', 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)[:] = nSoil
            ds.createVariable('nGlce', 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)[:] = nGlce

            # Domain attributes
            ds.createVariable('domType', 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)[:] = domType.astype('f8')
            ds.createVariable('DOMarea', 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)[:] = DOMarea
            ds.createVariable('DOMelev', 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)[:] = DOMelev
            ds.createVariable('scalarAblFrac', 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)[:] = scalarAblFrac

            # Glacier variables
            ds.createVariable('glacierAblArea', 'f8', ('glac', 'gru'), fill_value=-999.)[:] = glacierAblArea
            ds.createVariable('glacierAccArea', 'f8', ('glac', 'gru'), fill_value=-999.)[:] = glacierAccArea
            ds.createVariable('glacId', 'i8', ('glac', 'gru'), fill_value=-999)[:] = grid_info['gridId'][:num_glac, :]
            ds.createVariable('basin__GlacierStorage', 'f8', ('gru',), fill_value=-999.)[:] = grid_info['basin__GlacierStorage']

            # State variables
            for var_name, var_value in states.items():
                var = ds.createVariable(var_name, 'f8', ('scalarv', 'hru', 'dom'), fill_value=-999.)
                var[:] = var_value

            # Layer state variables
            mLayerTemp = np.full((midToto, num_hru, ndom), 283.16, dtype='f8')
            mLayerVolFracIce = np.zeros((midToto, num_hru, ndom), dtype='f8')
            mLayerVolFracLiq = np.full((midToto, num_hru, ndom), 0.2, dtype='f8')

            # Set ice layer temperatures (268.16 K = -5C) as per Ashley's code
            for i in range(num_hru):
                for j in range(ndom):
                    if nGlce[0, i, j] > 0:
                        start_ice = int(nSnow[0, i, j] + nLake[0, i, j] + nSoil[0, i, j])
                        end_ice = start_ice + int(nGlce[0, i, j])
                        mLayerTemp[start_ice:end_ice, i, j] = 268.16
                        mLayerVolFracIce[start_ice:end_ice, i, j] = 0.9

            ds.createVariable('mLayerTemp', 'f8', ('midToto', 'hru', 'dom'), fill_value=-999.)[:] = mLayerTemp
            ds.createVariable('mLayerVolFracIce', 'f8', ('midToto', 'hru', 'dom'), fill_value=-999.)[:] = mLayerVolFracIce
            ds.createVariable('mLayerVolFracLiq', 'f8', ('midToto', 'hru', 'dom'), fill_value=-999.)[:] = mLayerVolFracLiq
            ds.createVariable('mLayerMatricHead', 'f8', ('midSoil', 'hru', 'dom'), fill_value=-999.)[:] = -1.0

        self.logger.info(f"Created coldState_glac.nc: {num_hru} HRUs, {ndom} domains, {num_glac} glaciers")

    def _create_coldState_glacSurfTopo_from_rasters(
        self,
        settings_dir: Path,
        glacier_dir: Path,
        gru_ids: np.ndarray
    ) -> None:
        """Create coldState_glacSurfTopo.nc from raster data."""
        output_file = settings_dir / 'coldState_glacSurfTopo.nc'

        # Load domain type for grid dimensions
        domain_type_path = self._get_raster_path(glacier_dir, 'domain_type.tif')
        if domain_type_path is None:
            self.logger.warning("No domain_type raster, skipping glacSurfTopo")
            return

        with rasterio.open(domain_type_path) as src:
            ygrid, xgrid = src.shape

        # Load surface elevation from DEM
        if self.dem_path.exists():
            with rasterio.open(self.dem_path) as src:
                surface_elev = src.read(1).astype(np.float64)
                if surface_elev.shape != (ygrid, xgrid):
                    surface_elev = self._resample_to_grid(surface_elev, (ygrid, xgrid))
        else:
            surface_elev = np.zeros((ygrid, xgrid), dtype=np.float64)

        # Load debris thickness
        debris_path = self._get_raster_path(glacier_dir, 'debris_thickness.tif')
        if debris_path:
            with rasterio.open(debris_path) as src:
                debris_thick = src.read(1).astype(np.float64)
                if debris_thick.shape != (ygrid, xgrid):
                    debris_thick = self._resample_to_grid(debris_thick, (ygrid, xgrid))
        else:
            debris_thick = np.zeros((ygrid, xgrid), dtype=np.float64)

        num_gru = len(gru_ids)
        num_grid = 1

        with nc4.Dataset(output_file, 'w', format='NETCDF4') as ds:
            ds.createDimension('gru', num_gru)
            ds.createDimension('grid', num_grid)
            ds.createDimension('xgrid', xgrid)
            ds.createDimension('ygrid', ygrid)

            gruId_var = ds.createVariable('gruId', 'i8', ('gru',), fill_value=-999)
            gruId_var.long_name = 'GRU ID'
            gruId_var[:] = gru_ids

            gridId_var = ds.createVariable('gridId', 'i8', ('grid', 'gru'), fill_value=-999)
            gridId_var[:, :] = 1

            surf_var = ds.createVariable('surface_elev', 'f8', ('ygrid', 'xgrid', 'grid', 'gru'), fill_value=-999.)
            surf_var[:, :, 0, 0] = surface_elev

            debris_var = ds.createVariable('debris_thick', 'f8', ('ygrid', 'xgrid', 'grid', 'gru'), fill_value=-999.)
            debris_var[:, :, 0, 0] = debris_thick

        self.logger.info(f"Created coldState_glacSurfTopo.nc: grid={xgrid}x{ygrid}")

    def _process_from_rasters(
        self,
        glacier_dir: Path,
        settings_dir: Path,
        base_attributes_file: Path,
        hru_ids: np.ndarray,
        gru_ids: np.ndarray,
        hru_areas: np.ndarray
    ) -> bool:
        """Fallback: process glacier attributes directly from rasters."""
        self.logger.info("Using raster-based glacier preprocessing (fallback)")
        # This is the simpler raster-only approach from the original implementation
        # Used when shapefile intersections are not available
        return False  # Not implemented - requires shapefiles for proper preprocessing

    def _resample_to_grid(
        self,
        data: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Resample array to target shape using bilinear interpolation."""
        from scipy import ndimage
        zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
        return ndimage.zoom(data, zoom_factors, order=1)
