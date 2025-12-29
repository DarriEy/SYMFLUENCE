"""
CERRA Dataset Handler for SYMFLUENCE

This module provides the CERRA-specific implementation for forcing data processing.
CERRA (Copernicus European Regional Reanalysis) covers Europe at 5.5 km resolution.
"""

from pathlib import Path
from typing import Dict, Tuple
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('cerra')
class CERRAHandler(BaseDatasetHandler):
    """Handler for CERRA (Copernicus European Regional Reanalysis) dataset."""

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        CERRA variable name mapping to standard names.

        Returns:
            Dictionary mapping CERRA variable names to standard names
        """
        return {
            # CERRA variable names from CDS
            't2m': 'airtemp',               # 2m temperature
            'r2': 'relhum',                 # 2m relative humidity
            'tp': 'pptrate',                # total precipitation
            'sp': 'airpres',                # surface pressure
            'q': 'spechum',                 # specific humidity (if available)
            'u10': 'windspd_u',             # 10m U wind component
            'v10': 'windspd_v',             # 10m V wind component
            'ws10': 'windspd',              # 10m wind speed
            'ssrd': 'SWRadAtm',             # surface solar radiation downwards
            'strd': 'LWRadAtm',             # surface thermal radiation downwards
            '2m_temperature': 'airtemp',
            '2m_relative_humidity': 'relhum',
            'total_precipitation': 'pptrate',
            'surface_pressure': 'airpres',
            '10m_u_component_of_wind': 'windspd_u',
            '10m_v_component_of_wind': 'windspd_v',
            'surface_solar_radiation_downwards': 'SWRadAtm',
            'surface_thermal_radiation_downwards': 'LWRadAtm',
        }

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process CERRA dataset with variable renaming and unit conversions.

        CERRA data typically comes in standard units but may need some adjustments.

        Args:
            ds: Input CERRA dataset

        Returns:
            Processed dataset with standardized variables
        """
        # Rename variables
        variable_mapping = self.get_variable_mapping()
        existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}

        if existing_vars:
            ds = ds.rename(existing_vars)

        # Calculate wind speed from components if not present
        if 'windspd' not in ds and 'windspd_u' in ds and 'windspd_v' in ds:
            u = ds['windspd_u']
            v = ds['windspd_v']
            windspd = np.sqrt(u**2 + v**2)
            windspd.name = 'windspd'
            windspd.attrs = {
                'units': 'm s-1',
                'long_name': 'wind speed',
                'standard_name': 'wind_speed',
            }
            ds['windspd'] = windspd

        # Convert total precipitation from accumulated to rate if needed
        if 'pptrate' in ds:
            # CERRA typically provides accumulated precipitation
            # Need to convert to rate (m/s)
            p = ds['pptrate']
            if 'units' in p.attrs:
                units = p.attrs['units'].lower()
                if 'm' in units and 'hour' not in units and 's-1' not in units:
                    # Accumulated meters - convert to rate
                    # CERRA is 3-hourly, so divide by 10800 seconds
                    time_diff = ds.time.diff('time').median()
                    if time_diff:
                        # Convert timedelta to seconds
                        seconds = time_diff.values / np.timedelta64(1, 's')
                        ds['pptrate'] = p / float(seconds)
                        ds['pptrate'].attrs['units'] = 'm s-1'

            ds['pptrate'].attrs.update({
                'long_name': 'precipitation rate',
                'standard_name': 'precipitation_rate',
            })

        # Calculate specific humidity from relative humidity if needed
        if 'spechum' not in ds and 'relhum' in ds and 'airtemp' in ds and 'airpres' in ds:
            # q = 0.622 * e / (p - 0.378 * e)
            # where e = (RH/100) * e_sat
            # and e_sat = 611.2 * exp(17.67 * (T-273.15) / (T-29.65))
            T = ds['airtemp']
            RH = ds['relhum']
            P = ds['airpres']

            # Saturation vapor pressure (Pa)
            e_sat = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))

            # Actual vapor pressure (Pa)
            e = (RH / 100.0) * e_sat

            # Specific humidity (kg/kg)
            q = 0.622 * e / (P - 0.378 * e)
            q.name = 'spechum'
            q.attrs = {
                'units': 'kg kg-1',
                'long_name': 'specific humidity',
                'standard_name': 'specific_humidity',
            }
            ds['spechum'] = q

        # Ensure standard attributes
        if 'airpres' in ds:
            ds['airpres'].attrs.update({
                'units': 'Pa',
                'long_name': 'air pressure',
                'standard_name': 'air_pressure'
            })

        if 'airtemp' in ds:
            ds['airtemp'].attrs.update({
                'units': 'K',
                'long_name': 'air temperature',
                'standard_name': 'air_temperature'
            })

        if 'LWRadAtm' in ds:
            ds['LWRadAtm'].attrs.update({
                'units': 'W m-2',
                'long_name': 'downward longwave radiation at the surface',
                'standard_name': 'surface_downwelling_longwave_flux_in_air'
            })

        if 'SWRadAtm' in ds:
            ds['SWRadAtm'].attrs.update({
                'units': 'W m-2',
                'long_name': 'downward shortwave radiation at the surface',
                'standard_name': 'surface_downwelling_shortwave_flux_in_air'
            })

        # Add metadata via base helpers
        ds = self.setup_time_encoding(ds)
        ds = self.add_metadata(
            ds,
            "CERRA data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
        )
        ds = self.clean_variable_attributes(ds)

        return ds

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        CERRA uses latitude/longitude coordinates.

        Returns:
            Tuple of ('latitude', 'longitude') or ('lat', 'lon')
        """
        # CERRA from CDS typically uses 'latitude' and 'longitude'
        return ('latitude', 'longitude')

    def needs_merging(self) -> bool:
        """CERRA data requires standardization/processing."""
        return True

    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        Standardize CERRA forcings.

        This method processes CERRA data by:
        - Applying variable mapping
        - Converting units
        - Deriving additional variables (wind speed, specific humidity)
        """
        self.logger.info("Standardizing CERRA forcing files")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        patterns = [
            f"{self.domain_name}_CERRA_*.nc",
            f"domain_{self.domain_name}_CERRA_*.nc",
            "*CERRA*.nc",
        ]

        files = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} CERRA file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            msg = f"No CERRA forcing files found in {raw_forcing_path} with patterns {patterns}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        for f in files:
            self.logger.info(f"Processing CERRA file: {f}")
            try:
                ds = xr.open_dataset(f)
            except Exception as e:
                self.logger.error(f"Error opening CERRA file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)
                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed CERRA forcing: {out_name}")
            except Exception as e:
                self.logger.error(f"Error processing CERRA dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("CERRA forcing standardization completed")

    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create CERRA grid shapefile.

        CERRA uses a regular latitude-longitude grid over Europe.

        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to CERRA data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics

        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating CERRA grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"

        try:
            # Find a processed CERRA file
            cerra_files = list(merged_forcing_path.glob('*.nc'))
            if not cerra_files:
                raise FileNotFoundError("No processed CERRA files found")
            cerra_file = cerra_files[0]

            self.logger.info(f"Using CERRA file: {cerra_file}")

            # Read CERRA data
            with xr.open_dataset(cerra_file) as ds:
                var_lat, var_lon = self.get_coordinate_names()

                # Handle both 1D and 2D coordinates
                if var_lat in ds.coords:
                    lats = ds.coords[var_lat].values
                elif var_lat in ds.variables:
                    lats = ds[var_lat].values
                else:
                    # Try alternative names
                    if 'lat' in ds.coords:
                        lats = ds.coords['lat'].values
                    elif 'lat' in ds.variables:
                        lats = ds['lat'].values
                    else:
                        raise KeyError(f"Latitude coordinate not found in CERRA file")

                if var_lon in ds.coords:
                    lons = ds.coords[var_lon].values
                elif var_lon in ds.variables:
                    lons = ds[var_lon].values
                else:
                    # Try alternative names
                    if 'lon' in ds.coords:
                        lons = ds.coords['lon'].values
                    elif 'lon' in ds.variables:
                        lons = ds['lon'].values
                    else:
                        raise KeyError(f"Longitude coordinate not found in CERRA file")

            self.logger.info(f"CERRA grid dimensions: lat={lats.shape}, lon={lons.shape}")

            # Get HRU bounding box for spatial filtering
            # Read the HRU shapefile to get its extent
            # shapefile_path is .../shapefiles/forcing, so parent is .../shapefiles
            hru_shapefile_dir = shapefile_path.parent / 'catchment'
            domain_name = self.config.get('DOMAIN_NAME', 'domain')
            hru_shapefile = hru_shapefile_dir / f"{domain_name}_HRUs_GRUs.shp"

            self.logger.info(f"Looking for HRU shapefile at: {hru_shapefile}")

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
                    self.logger.info(f"✓ Applying spatial filter based on HRU extent:")
                    self.logger.info(f"  Lon: {bbox_filter['lon_min']:.2f} to {bbox_filter['lon_max']:.2f}")
                    self.logger.info(f"  Lat: {bbox_filter['lat_min']:.2f} to {bbox_filter['lat_max']:.2f}")
                except Exception as e:
                    self.logger.warning(f"Could not read HRU shapefile for spatial filtering: {e}")
                    self.logger.warning("Will create shapefile for full domain (may be slow)")
            else:
                self.logger.warning(f"HRU shapefile not found: {hru_shapefile}")
                self.logger.warning("Will create shapefile for full domain (may be slow)")

            # Create geometries
            self.logger.info("Creating CERRA grid cell geometries")

            geometries = []
            ids = []
            center_lats = []
            center_lons = []

            # Handle 1D grid (regular lat/lon)
            if lats.ndim == 1 and lons.ndim == 1:
                # Regular grid - create cell boundaries
                half_dlat = abs(lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.025
                half_dlon = abs(lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.025

                cell_id = 0
                total_cells = len(lats) * len(lons)
                cells_created = 0

                for i, center_lon in enumerate(lons):
                    for j, center_lat in enumerate(lats):
                        # Apply spatial filter if available
                        if bbox_filter is not None:
                            if (center_lon < bbox_filter['lon_min'] or center_lon > bbox_filter['lon_max'] or
                                center_lat < bbox_filter['lat_min'] or center_lat > bbox_filter['lat_max']):
                                continue

                        verts = [
                            [float(center_lon) - half_dlon, float(center_lat) - half_dlat],
                            [float(center_lon) - half_dlon, float(center_lat) + half_dlat],
                            [float(center_lon) + half_dlon, float(center_lat) + half_dlat],
                            [float(center_lon) + half_dlon, float(center_lat) - half_dlat],
                            [float(center_lon) - half_dlon, float(center_lat) - half_dlat],
                        ]
                        geometries.append(Polygon(verts))
                        ids.append(cell_id)
                        center_lats.append(float(center_lat))
                        center_lons.append(float(center_lon))
                        cells_created += 1

                        if cells_created % 1000 == 0:
                            self.logger.info(f"Created {cells_created} CERRA grid cells (filtered from {cell_id}/{total_cells})")

                        cell_id += 1

                if bbox_filter is not None:
                    self.logger.info(f"Spatial filtering: created {cells_created} cells (from {total_cells} total)")
            else:
                # 2D grid (Lambert Conformal - CERRA uses this)
                ny, nx = lats.shape
                total_cells = ny * nx
                cells_created = 0

                # Pre-filter grid indices if bounding box is available
                if bbox_filter is not None:
                    # Create mask for cells within bounding box
                    mask = ((lons >= bbox_filter['lon_min']) & (lons <= bbox_filter['lon_max']) &
                            (lats >= bbox_filter['lat_min']) & (lats <= bbox_filter['lat_max']))
                    # Get indices of cells within bbox
                    indices = np.where(mask)
                    indices_list = list(zip(indices[0], indices[1]))
                    self.logger.info(f"Spatial filtering: {len(indices_list)} cells within bbox (from {total_cells} total)")
                else:
                    # No filter - use all indices
                    indices_list = [(i, j) for i in range(ny) for j in range(nx)]

                for i, j in indices_list:
                    # Get cell center coordinates
                    center_lat = float(lats[i, j])
                    center_lon = float(lons[i, j])

                    # Create cell from corners
                    lat_corners = [
                        lats[i, j],
                        lats[i, j + 1] if j + 1 < nx else lats[i, j],
                        lats[i + 1, j + 1] if i + 1 < ny and j + 1 < nx else lats[i, j],
                        lats[i + 1, j] if i + 1 < ny else lats[i, j],
                    ]
                    lon_corners = [
                        lons[i, j],
                        lons[i, j + 1] if j + 1 < nx else lons[i, j],
                        lons[i + 1, j + 1] if i + 1 < ny and j + 1 < nx else lons[i, j],
                        lons[i + 1, j] if i + 1 < ny else lons[i, j],
                    ]

                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * nx + j)
                    center_lats.append(center_lat)
                    center_lons.append(center_lon)
                    cells_created += 1

                    if cells_created % 1000 == 0:
                        self.logger.info(f"Created {cells_created} CERRA grid cells")

                if bbox_filter is not None:
                    self.logger.info(f"Spatial filtering: created {cells_created} cells (from {total_cells} total)")

            # Create GeoDataFrame
            self.logger.info("Creating GeoDataFrame")
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self.config.get('FORCING_SHAPE_LAT_NAME'): center_lats,
                self.config.get('FORCING_SHAPE_LON_NAME'): center_lons,
            }, crs='EPSG:4326')

            # Calculate elevation
            self.logger.info("Calculating elevation values")
            elevations = elevation_calculator(gdf, dem_path, batch_size=50)
            gdf['elev_m'] = elevations

            # Save the shapefile
            self.logger.info(f"Saving CERRA shapefile to {output_shapefile}")
            shapefile_path.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_shapefile)
            self.logger.info(f"CERRA grid shapefile created and saved to {output_shapefile}")

            return output_shapefile

        except Exception as e:
            self.logger.error(f"Error in create_cerra_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
