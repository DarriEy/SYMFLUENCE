"""
Elevation band management for FUSE model.

This module contains the FuseElevationBandManager class which handles creation
of elevation bands for both lumped and distributed spatial configurations.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd


class FuseElevationBandManager:
    """
    Manager for FUSE elevation band creation in lumped and distributed modes.

    This class handles:
    - Lumped elevation band generation from catchment statistics
    - Distributed elevation band generation for multiple HRUs
    - NetCDF file creation with proper formatting
    - Elevation and area fraction calculations

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        project_dir: Root project directory
        forcing_fuse_path: Path to FUSE forcing output directory
        catchment_path: Path to catchment shapefile
        domain_name: Domain name for file naming
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        project_dir: Path,
        forcing_fuse_path: Path,
        catchment_path: Path,
        domain_name: str,
        calculate_catchment_centroid_callback
    ):
        """
        Initialize the elevation band manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Root project directory
            forcing_fuse_path: Path to FUSE forcing output directory
            catchment_path: Path to catchment shapefile
            domain_name: Domain name for file naming
            calculate_catchment_centroid_callback: Callback to parent's calculate_catchment_centroid method
        """
        self.config = config
        self.logger = logger
        self.project_dir = Path(project_dir)
        self.forcing_fuse_path = Path(forcing_fuse_path)
        self.catchment_path = Path(catchment_path)
        self.domain_name = domain_name
        self.calculate_catchment_centroid = calculate_catchment_centroid_callback

    def create_elevation_bands(self) -> Path:
        """
        Create elevation bands based on spatial mode.

        Returns:
            Path to created elevation bands file

        Raises:
            ValueError: If unknown spatial mode specified
        """
        spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')

        self.logger.info(f"Creating elevation bands for {spatial_mode} mode")

        if spatial_mode == 'lumped':
            return self._create_lumped_elevation_bands()
        elif spatial_mode in ['semi_distributed', 'distributed']:
            return self._create_distributed_elevation_bands()
        else:
            raise ValueError(f"Unknown FUSE spatial mode: {spatial_mode}")

    def _get_forcing_spatial_info(self) -> Optional[Tuple[list, Dict[str, np.ndarray]]]:
        forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
        if not forcing_file.exists():
            return None

        try:
            with xr.open_dataset(forcing_file) as ds:
                data_var = next(iter(ds.data_vars.values()), None)
                if data_var is None:
                    return None

                spatial_dims: List[Any] = []
                spatial_coords: Dict[str, np.ndarray] = {}

                for dim in data_var.dims:
                    if dim == 'time':
                        continue
                    dim_str = str(dim)
                    spatial_dims.append(dim_str)
                    if dim in ds.coords:
                        spatial_coords[dim_str] = ds[dim].values
                    else:
                        spatial_coords[dim_str] = np.arange(ds.sizes[dim])

                return spatial_dims, spatial_coords
        except Exception as e:
            self.logger.warning(f"Unable to read spatial info from {forcing_file}: {e}")
            return None

    def _create_lumped_elevation_bands(self) -> Path:
        """
        Create elevation bands for lumped mode.

        Uses catchment statistics to generate representative elevation bands.

        Returns:
            Path to created elevation bands file
        """
        self.logger.info("Creating lumped elevation bands file")

        try:
            # Load catchment shapefile
            catchment = gpd.read_file(self.catchment_path)

            # Get elevation statistics from shapefile or use defaults
            if 'elev_mean' in catchment.columns and 'elev_range' in catchment.columns:
                elev_mean = catchment['elev_mean'].iloc[0]
                elev_range = catchment['elev_range'].iloc[0]
                elev_min = elev_mean - (elev_range / 2)
                elev_max = elev_mean + (elev_range / 2)
                self.logger.info(f"Using shapefile elevation stats: mean={elev_mean}m, range={elev_range}m")
            else:
                # Use default values if elevation data not in shapefile
                elev_mean = 1000.0
                elev_range = 500.0
                elev_min = 750.0
                elev_max = 1250.0
                self.logger.warning(f"Elevation data not found in shapefile, using defaults: mean={elev_mean}m, range={elev_range}m")

            # Prefer spatial info from forcing file to match FUSE expectations
            spatial_info = self._get_forcing_spatial_info()
            if spatial_info:
                spatial_dims, spatial_coords = spatial_info
            else:
                # Get centroid for lat/lon fallback
                mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
                spatial_dims = ['latitude', 'longitude']
                spatial_coords = {
                    'latitude': np.array([float(mean_lat)]),
                    'longitude': np.array([float(mean_lon)])
                }

            # Create simple elevation bands (e.g., 5 bands)
            n_bands = self.config.get('FUSE_N_ELEVATION_BANDS', 5)
            elevations = np.linspace(elev_min, elev_max, n_bands)

            # Equal area fractions for each band
            area_fractions = np.ones(n_bands) / n_bands

            # Create dataset matching FUSE expectations
            coords = {dim: (dim, spatial_coords[dim]) for dim in spatial_dims}
            coords['elevation_band'] = ('elevation_band', np.arange(1, n_bands + 1))
            ds = xr.Dataset(coords=coords, attrs={
                'title': 'FUSE elevation bands (lumped mode)',
                'source': 'Generated by SYMFLUENCE',
                'history': f'Created {pd.Timestamp.now()}',
                'spatial_mode': 'lumped',
                'n_bands': n_bands
            })

            # Use xarray broadcasting for cleaner and more efficient array creation
            xr.DataArray(area_fractions, dims=['elevation_band'])
            xr.DataArray(elevations, dims=['elevation_band'])

            band_shape = (n_bands,) + tuple(len(spatial_coords[dim]) for dim in spatial_dims)
            band_dims = ['elevation_band'] + spatial_dims
            broadcast_shape = (n_bands,) + (1,) * len(spatial_dims)

            ds['area_frac'] = xr.DataArray(
                np.broadcast_to(area_fractions.reshape(broadcast_shape), band_shape),
                dims=band_dims,
                attrs={
                    'long_name': 'Fraction of catchment area in each elevation band',
                    'units': '-'
                }
            ).astype('float32')

            ds['mean_elev'] = xr.DataArray(
                np.broadcast_to(elevations.reshape(broadcast_shape), band_shape),
                dims=band_dims,
                attrs={
                    'long_name': 'Band mid-point elevation',
                    'units': 'm',
                    'standard_name': 'height_above_reference_ellipsoid'
                }
            ).astype('float32')

            ds['prec_frac'] = xr.DataArray(
                np.broadcast_to(area_fractions.reshape(broadcast_shape), band_shape),
                dims=band_dims,
                attrs={
                    'long_name': 'Fraction of catchment precipitation that falls on each elevation band',
                    'units': '-'
                }
            ).astype('float32')

            # Save to file
            output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"

            # Use strict encoding for FUSE compatibility
            encoding = {
                'area_frac': {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': False},
                'mean_elev': {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': False},
                'prec_frac': {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': False},
                'elevation_band': {'dtype': 'float64', '_FillValue': None}
            }
            for dim in spatial_dims:
                encoding[dim] = {'dtype': 'float64', '_FillValue': None}

            ds.to_netcdf(output_file, encoding=encoding)

            self.logger.info(f"Created lumped elevation bands file with {n_bands} bands: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error creating lumped elevation bands: {str(e)}")
            raise

    def _create_distributed_elevation_bands(self) -> Path:
        """
        Create elevation bands for distributed/semi-distributed mode.

        Uses per-HRU elevation data from catchment shapefile.

        Returns:
            Path to created elevation bands file
        """
        self.logger.info("Creating distributed elevation bands file")

        try:
            # Load catchment shapefile
            catchment = gpd.read_file(self.catchment_path)
            n_hrus = len(catchment)

            # Get elevation data per HRU
            if 'elev_mean' in catchment.columns:
                elevations = catchment['elev_mean'].values
                self.logger.info("Using HRU-specific elevation data from shapefile")
            else:
                # Use uniform elevation if not available
                elevations = np.full(n_hrus, 1000.0)
                self.logger.warning("No HRU elevation data found, using default 1000m for all HRUs")

            # Calculate area fractions
            if 'area' in catchment.columns or 'Area' in catchment.columns:
                area_col = 'area' if 'area' in catchment.columns else 'Area'
                areas = catchment[area_col].values
                total_area = areas.sum()
                area_fractions = areas / total_area
            elif 'GRU_area' in catchment.columns:
                areas = catchment['GRU_area'].values
                total_area = areas.sum()
                area_fractions = areas / total_area
            else:
                # Equal area fractions
                area_fractions = np.ones(n_hrus) / n_hrus
                self.logger.warning("No area data found, using equal area fractions")

            # Get lat/lon for each HRU
            if 'latitude' in catchment.columns and 'longitude' in catchment.columns:
                latitudes = catchment['latitude'].values
                longitudes = catchment['longitude'].values
            else:
                # Calculate centroids in a way that avoids the geographic CRS warning
                if catchment.crs and catchment.crs.is_geographic:
                    # Project to EPSG:3857 for centroid calculation, then back to WGS84
                    # This avoids the UserWarning: Geometry is in a geographic CRS
                    centroids_proj = catchment.to_crs(epsg=3857).geometry.centroid
                    centroids = centroids_proj.to_crs(epsg=4326)
                else:
                    # Already projected, calculate centroid then convert to WGS84
                    centroids = catchment.geometry.centroid.to_crs(epsg=4326)

                latitudes = centroids.y.values
                longitudes = centroids.x.values

            spatial_info = self._get_forcing_spatial_info()
            if spatial_info:
                spatial_dims, spatial_coords = spatial_info
            else:
                spatial_dims = ['latitude', 'longitude']
                spatial_coords = {
                    'latitude': np.array([latitudes.mean()]),
                    'longitude': np.array([longitudes.mean()])
                }

            n_bands = self.config.get('FUSE_N_ELEVATION_BANDS', 5)
            elevations = np.linspace(elevations.min(), elevations.max(), n_bands)
            area_fractions = np.ones(n_bands) / n_bands

            coords = {dim: (dim, spatial_coords[dim]) for dim in spatial_dims}
            coords['elevation_band'] = ('elevation_band', np.arange(1, n_bands + 1))
            ds = xr.Dataset(coords=coords, attrs={
                'title': 'FUSE elevation bands (distributed mode)',
                'source': 'Generated by SYMFLUENCE',
                'history': f'Created {pd.Timestamp.now()}',
                'spatial_mode': 'distributed',
                'n_bands': n_bands
            })

            band_shape = (n_bands,) + tuple(len(spatial_coords[dim]) for dim in spatial_dims)
            band_dims = ['elevation_band'] + spatial_dims
            broadcast_shape = (n_bands,) + (1,) * len(spatial_dims)

            ds['area_frac'] = xr.DataArray(
                np.broadcast_to(area_fractions.reshape(broadcast_shape), band_shape),
                dims=band_dims,
                attrs={
                    'long_name': 'Fraction of catchment area in each elevation band',
                    'units': '-'
                }
            ).astype('float32')

            ds['mean_elev'] = xr.DataArray(
                np.broadcast_to(elevations.reshape(broadcast_shape), band_shape),
                dims=band_dims,
                attrs={
                    'long_name': 'Band mid-point elevation',
                    'units': 'm',
                    'standard_name': 'height_above_reference_ellipsoid'
                }
            ).astype('float32')

            ds['prec_frac'] = xr.DataArray(
                np.broadcast_to(area_fractions.reshape(broadcast_shape), band_shape),
                dims=band_dims,
                attrs={
                    'long_name': 'Fraction of catchment precipitation that falls on each elevation band',
                    'units': '-'
                }
            ).astype('float32')

            # Save to file
            output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"

            # Use strict encoding for FUSE compatibility
            encoding = {
                'area_frac': {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': False},
                'mean_elev': {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': False},
                'prec_frac': {'dtype': 'float32', '_FillValue': -9999.0, 'zlib': False},
                'elevation_band': {'dtype': 'float64', '_FillValue': None}
            }
            for dim in spatial_dims:
                encoding[dim] = {'dtype': 'float64', '_FillValue': None}

            ds.to_netcdf(output_file, encoding=encoding)

            self.logger.info(f"Created distributed elevation bands file with {n_hrus} HRUs: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error creating distributed elevation bands: {str(e)}")
            raise
