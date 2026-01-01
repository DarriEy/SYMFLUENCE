"""
Point Scale Forcing Extractor

Simplified forcing extraction for point-scale or small grid domains.
"""

import logging
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import List, Optional


class PointScaleForcingExtractor:
    """
    Extracts forcing data for point-scale or tiny grid domains.

    Bypasses EASYMORE remapping for 1x1 or small grids where intersection
    calculations would fail or be meaningless.
    """

    def __init__(
        self,
        config: dict,
        project_dir: Path,
        dataset_handler,
        logger: logging.Logger = None
    ):
        """
        Initialize point scale extractor.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            dataset_handler: Dataset-specific handler for coordinate names
            logger: Optional logger instance
        """
        self.config = config
        self.project_dir = project_dir
        self.dataset_handler = dataset_handler
        self.logger = logger or logging.getLogger(__name__)

    def should_use_point_scale(self, merged_forcing_path: Path) -> bool:
        """
        Check if the forcing grid is too small for EASYMORE remapping.

        Args:
            merged_forcing_path: Path to merged forcing files

        Returns:
            True if point-scale extraction should be used
        """
        try:
            # Find a sample forcing file
            exclude_patterns = ['attributes', 'metadata', 'static', 'constants', 'params']
            all_nc_files = list(merged_forcing_path.glob('*.nc'))
            forcing_files = [
                f for f in all_nc_files
                if not any(pattern in f.name.lower() for pattern in exclude_patterns)
            ]

            if not forcing_files:
                return False

            sample_file = forcing_files[0]
            var_lat, var_lon = self.dataset_handler.get_coordinate_names()

            with xr.open_dataset(sample_file) as ds:
                lat_vals = ds[var_lat].values
                lon_vals = ds[var_lon].values

                # Determine grid size
                if lat_vals.ndim == 1:
                    lat_size = len(lat_vals)
                    lon_size = len(lon_vals)
                elif lat_vals.ndim == 2:
                    lat_size, lon_size = lat_vals.shape
                else:
                    lat_size = lon_size = 1

                # EASYMORE requires at least 3 values in each dimension
                is_tiny = (lat_size <= 2 or lon_size <= 2)

                if is_tiny:
                    self.logger.info(
                        f"Detected small forcing grid: {lat_size}x{lon_size} "
                        f"(EASYMORE requires >= 3 in each dimension)"
                    )
                    return True

                return False

        except Exception as e:
            self.logger.warning(f"Could not check forcing grid size: {e}")
            return False

    def process(
        self,
        forcing_files: List[Path],
        output_dir: Path,
        catchment_path: Path,
        catchment_name: str,
        output_filename_func
    ) -> None:
        """
        Process forcing files using simplified point-scale extraction.

        Args:
            forcing_files: List of forcing files to process
            output_dir: Output directory for processed files
            catchment_path: Path to catchment shapefile directory
            catchment_name: Name of catchment shapefile
            output_filename_func: Function to determine output filename
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)

        # Create minimal intersection CSV
        case_name = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}"
        intersect_csv = intersect_path / f"{case_name}_intersected_shapefile.csv"

        if not intersect_csv.exists():
            self._create_intersection_csv(intersect_csv, catchment_path, catchment_name)

        # Process each file
        for file in forcing_files:
            output_file = output_filename_func(file)
            if output_file.exists() and not self.config.get('FORCE_RUN_ALL_STEPS', False):
                continue

            self._process_single_file(file, output_file, intersect_csv)

    def _create_intersection_csv(
        self,
        intersect_csv: Path,
        catchment_path: Path,
        catchment_name: str
    ) -> None:
        """Create minimal intersection artifact for SUMMA preprocessor."""
        self.logger.info(f"Creating minimal intersection artifact: {intersect_csv.name}")

        target_shp_path = catchment_path / catchment_name
        target_gdf = gpd.read_file(target_shp_path)
        hru_id_field = self.config.get('CATCHMENT_SHP_HRUID')

        hru_id_field_val = target_gdf[hru_id_field].values
        df_int = pd.DataFrame({
            hru_id_field: hru_id_field_val,
            'S_1_HRU_ID': hru_id_field_val,
            'S_1_GRU_ID': target_gdf['GRU_ID'].values if 'GRU_ID' in target_gdf.columns else [1],
            'ID': [1] * len(target_gdf),
            'weight': [1.0] * len(target_gdf),
            'S_1_elev_m': target_gdf['elev_mean'].values if 'elev_mean' in target_gdf.columns else [1600.0],
            'S_2_elev_m': [1600.0] * len(target_gdf)
        })
        df_int.to_csv(intersect_csv, index=False)

    def _process_single_file(
        self,
        file: Path,
        output_file: Path,
        intersect_csv: Path
    ) -> None:
        """Process a single forcing file."""
        self.logger.info(f"Extracting point forcing: {file.name}")

        with xr.open_dataset(file) as ds:
            # Pick the first cell if it's a grid
            spatial_dims = {d: 0 for d in ds.dims if d not in ['time', 'hru']}

            # Check for empty spatial dimensions
            for dim_name, idx in spatial_dims.items():
                if dim_name in ds.dims and ds.sizes[dim_name] == 0:
                    raise ValueError(
                        f"Cannot extract point forcing from {file.name}: "
                        f"dimension '{dim_name}' has size 0."
                    )

            ds_point = ds.isel(spatial_dims)

            # Determine HRU IDs
            hru_ids = [1]
            if intersect_csv.exists():
                try:
                    df_int = pd.read_csv(intersect_csv)
                    hru_ids = df_int[self.config.get('CATCHMENT_SHP_HRUID')].values.astype('int32')
                except Exception:
                    pass

            n_hrus = len(hru_ids)

            # Add HRU dimension if missing
            if 'hru' not in ds_point.dims:
                ds_point = ds_point.expand_dims(hru=range(n_hrus))

            if 'hruId' not in ds_point.data_vars:
                ds_point['hruId'] = (('hru',), hru_ids)

            # Ensure correct dimension order (time, hru)
            for var in ds_point.data_vars:
                if 'time' in ds_point[var].dims and 'hru' in ds_point[var].dims:
                    ds_point[var] = ds_point[var].transpose('time', 'hru')

            # Drop irrelevant coordinates
            coords_to_drop = ['latitude', 'longitude', 'lat', 'lon', 'expver']
            ds_point = ds_point.drop_vars(
                [c for c in coords_to_drop if c in ds_point.coords or c in ds_point.data_vars]
            )

            # Clear encoding
            for var in ds_point.variables:
                ds_point[var].encoding = {}
                if 'missing_value' in ds_point[var].attrs:
                    del ds_point[var].attrs['missing_value']
                if '_FillValue' in ds_point[var].attrs:
                    del ds_point[var].attrs['_FillValue']

            ds_point.to_netcdf(output_file)
            self.logger.info(f"Created point forcing: {output_file.name}")
