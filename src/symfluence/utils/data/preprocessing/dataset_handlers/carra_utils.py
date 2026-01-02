"""
CARRA Dataset Handler for SYMFLUENCE

This module provides the CARRA-specific implementation for forcing data processing.
CARRA uses a polar stereographic projection and requires special coordinate handling.
"""

from pathlib import Path
from typing import Dict, Tuple
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj
from pyproj import CRS, Transformer

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('carra')
class CARRAHandler(BaseDatasetHandler):
    """Handler for CARRA (Copernicus Arctic Regional Reanalysis) dataset."""
    
    def get_variable_mapping(self) -> Dict[str, str]:
        """
        CARRA variable name mapping to standard names.
        
        Returns:
            Dictionary mapping CARRA variable names to standard names
        """
        return {
            # CARRA typically uses standard names already
            't2m': 'airtemp',
            'tp': 'pptrate',
            'sp': 'airpres',
            'q': 'spechum',
            'u10': 'windspd_u',
            'v10': 'windspd_v',
            'ws10': 'windspd',
            'ssrd': 'SWRadAtm',
            'strd': 'LWRadAtm',
        }
    
    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process CARRA dataset with variable renaming if needed.

        CARRA data typically comes in standard units.

        Args:
            ds: Input CARRA dataset

        Returns:
            Processed dataset with standardized variables
        """
        # Rename variables using mapping
        variable_mapping = self.get_variable_mapping()
        existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}

        if existing_vars:
            ds = ds.rename(existing_vars)

        # Apply standard CF-compliant attributes (uses centralized definitions)
        # CARRA precipitation is in m/s, override the default
        ds = self.apply_standard_attributes(ds, overrides={
            'pptrate': {'units': 'm s-1', 'standard_name': 'precipitation_rate'}
        })

        return ds
    
    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        CARRA uses latitude/longitude coordinates.
        
        Returns:
            Tuple of ('latitude', 'longitude')
        """
        return ('latitude', 'longitude')
    
    def needs_merging(self) -> bool:
        """CARRA data typically doesn't require merging."""
        return False
    
    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        CARRA typically doesn't require merging.
        
        This method is a no-op for CARRA but is required by the interface.
        """
        self.logger.info("CARRA data does not require merging. Skipping merge step.")
        pass
    
    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create CARRA grid shapefile.
        
        CARRA uses a polar stereographic projection which requires special handling.
        The grid is defined in stereographic coordinates but must be converted to lat/lon.
        
        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to CARRA data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics
            
        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating CARRA grid shapefile")
        
        output_shapefile = shapefile_path / f"forcing_{self.config.get('FORCING_DATASET')}.shp"
        
        try:
            # Find a processed CARRA file
            carra_files = list(merged_forcing_path.glob('*.nc'))
            if not carra_files:
                raise FileNotFoundError("No processed CARRA files found")
            carra_file = carra_files[0]
            
            self.logger.info(f"Using CARRA file: {carra_file}")
            
            # Read CARRA data
            with xr.open_dataset(carra_file) as ds:
                lats = ds.latitude.values
                lons = ds.longitude.values
            
            self.logger.info(f"CARRA dimensions: {lats.shape}")

            # Get HRU bounding box for spatial filtering
            # Read the HRU shapefile to get its extent
            # shapefile_path is .../shapefiles/forcing, so parent is .../shapefiles
            hru_shapefile_dir = shapefile_path.parent / 'catchment'
            domain_name = self.config.get('DOMAIN_NAME', 'domain')
            hru_shapefile = hru_shapefile_dir / f"{domain_name}_HRUs_GRUs.shp"

            bbox_filter = None
            if hru_shapefile.exists():
                try:
                    import geopandas as gpd
                    hru_gdf = gpd.read_file(hru_shapefile)
                    # Get bounding box with small buffer to ensure we capture nearby cells
                    bbox = hru_gdf.total_bounds  # [minx, miny, maxx, maxy]
                    buffer = 0.1  # ~10km buffer
                    bbox_filter = {
                        'lon_min': bbox[0] - buffer,
                        'lon_max': bbox[2] + buffer,
                        'lat_min': bbox[1] - buffer,
                        'lat_max': bbox[3] + buffer
                    }
                    self.logger.info(f"Applying spatial filter based on HRU extent:")
                    self.logger.info(f"  Lon: {bbox_filter['lon_min']:.2f} to {bbox_filter['lon_max']:.2f}")
                    self.logger.info(f"  Lat: {bbox_filter['lat_min']:.2f} to {bbox_filter['lat_max']:.2f}")
                except Exception as e:
                    self.logger.warning(f"Could not read HRU shapefile for spatial filtering: {e}")
                    self.logger.warning("Will create shapefile for full domain (may be slow)")

            # Define CARRA projection (polar stereographic)
            carra_proj = CRS('+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6356752.3142 +units=m +no_defs')
            wgs84 = CRS('EPSG:4326')

            transformer = Transformer.from_crs(carra_proj, wgs84, always_xy=True)
            transformer_to_carra = Transformer.from_crs(wgs84, carra_proj, always_xy=True)

            # Create geometries
            self.logger.info("Creating CARRA grid cell geometries")

            geometries = []
            ids = []
            center_lats = []
            center_lons = []

            # Flatten 2D lat/lon arrays to 1D for iteration
            lats_flat = lats.flatten()
            lons_flat = lons.flatten()

            # Process in batches
            batch_size = 100
            total_cells = len(lats_flat)
            num_batches = (total_cells + batch_size - 1) // batch_size
            cells_created = 0

            self.logger.info(f"Processing {total_cells} CARRA grid cells in {num_batches} batches")

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_cells)

                self.logger.info(f"Processing grid cell batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})")

                for i in range(start_idx, end_idx):
                    # Get cell center coordinates
                    center_lat_raw = float(lats_flat[i])
                    center_lon_raw = float(lons_flat[i])

                    # Convert longitude from 0-360 to -180/180 range for consistency
                    if center_lon_raw > 180:
                        center_lon_normalized = center_lon_raw - 360
                    else:
                        center_lon_normalized = center_lon_raw

                    # Apply spatial filter if available (using normalized longitude)
                    if bbox_filter is not None:
                        if (center_lon_normalized < bbox_filter['lon_min'] or center_lon_normalized > bbox_filter['lon_max'] or
                            center_lat_raw < bbox_filter['lat_min'] or center_lat_raw > bbox_filter['lat_max']):
                            continue

                    # Convert lat/lon to CARRA coordinates (scalars now)
                    x, y = transformer_to_carra.transform(center_lon_raw, center_lat_raw)
                    
                    # Define grid cell (assuming 2.5 km resolution)
                    half_dx = 1250  # meters
                    half_dy = 1250  # meters
                    
                    vertices = [
                        (x - half_dx, y - half_dy),
                        (x - half_dx, y + half_dy),
                        (x + half_dx, y + half_dy),
                        (x + half_dx, y - half_dy),
                        (x - half_dx, y - half_dy)
                    ]
                    
                    # Convert vertices back to lat/lon
                    # Ensure values are scalars, not arrays
                    lat_lon_vertices = []
                    for vx, vy in vertices:
                        lon, lat = transformer.transform(vx, vy)
                        # Extract scalar values if they're arrays
                        if hasattr(lon, 'item'):
                            lon = lon.item()
                        if hasattr(lat, 'item'):
                            lat = lat.item()
                        lat_lon_vertices.append((float(lon), float(lat)))

                    geometries.append(Polygon(lat_lon_vertices))
                    ids.append(i)

                    center_lon, center_lat = transformer.transform(x, y)
                    # Extract scalar values if they're arrays
                    if hasattr(center_lon, 'item'):
                        center_lon = center_lon.item()
                    if hasattr(center_lat, 'item'):
                        center_lat = center_lat.item()
                    center_lats.append(float(center_lat))
                    center_lons.append(float(center_lon))
                    cells_created += 1

            if bbox_filter is not None:
                self.logger.info(f"Spatial filtering: created {cells_created} cells (from {total_cells} total)")

            # Create GeoDataFrame
            self.logger.info(f"Creating GeoDataFrame with {cells_created} grid cells")
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self.config.get('FORCING_SHAPE_LAT_NAME'): center_lats,
                self.config.get('FORCING_SHAPE_LON_NAME'): center_lons,
            }, crs='EPSG:4326')
            
            # Calculate elevation using the safe method
            self.logger.info("Calculating elevation values using safe method")
            elevations = elevation_calculator(gdf, dem_path, batch_size=50)
            gdf['elev_m'] = elevations
            
            # Save the shapefile
            self.logger.info(f"Saving CARRA shapefile to {output_shapefile}")
            gdf.to_file(output_shapefile)
            self.logger.info(f"CARRA grid shapefile created and saved to {output_shapefile}")
            
            return output_shapefile
            
        except Exception as e:
            self.logger.error(f"Error in create_carra_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
