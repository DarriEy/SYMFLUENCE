"""
ShapefileManager - CRS alignment and HRU ID management for forcing remapping.

This module handles:
- Ensuring shapefiles are in WGS84 (EPSG:4326) for EASYMORE compatibility
- Ensuring HRU IDs are unique for proper weight calculation
- Longitude frame alignment (0-360 vs -180/180)
- Geometry validation and repair

Extracted from ForcingResampler to improve testability and reduce coupling.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
from shapely.affinity import translate

from symfluence.core.config.coercion import coerce_config
from symfluence.core.mixins import ConfigMixin


class ShapefileManager(ConfigMixin):
    """
    Manages shapefile CRS conversions and HRU ID uniqueness for forcing remapping.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize ShapefileManager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self._config = coerce_config(config, warn=False)
        self.logger = logger

    def ensure_wgs84(
        self,
        shapefile_path: Path,
        output_suffix: str = "_wgs84",
        ensure_unique_ids: bool = False,
        hru_id_field: Optional[str] = None
    ) -> Union[Path, Tuple[Path, str]]:
        """
        Ensure shapefile is in WGS84 (EPSG:4326) for EASYMORE compatibility.

        Args:
            shapefile_path: Path to the shapefile
            output_suffix: Suffix for WGS84 version filename
            ensure_unique_ids: If True, also ensure HRU IDs are unique
            hru_id_field: HRU ID field name (required if ensure_unique_ids=True)

        Returns:
            If ensure_unique_ids=False: Path to WGS84 shapefile
            If ensure_unique_ids=True: Tuple of (path, actual_hru_field_used)
        """
        shapefile_path = Path(shapefile_path)

        try:
            gdf = gpd.read_file(shapefile_path)
            current_crs = gdf.crs

            self.logger.info(f"Checking CRS for {shapefile_path.name}: {current_crs}")

            # Handle unique HRU IDs if requested
            actual_hru_field = hru_id_field or 'hru_id'
            if ensure_unique_ids and hru_id_field:
                shapefile_path, actual_hru_field = self.ensure_unique_hru_ids(
                    shapefile_path, hru_id_field
                )
                gdf = gpd.read_file(shapefile_path)
                current_crs = gdf.crs

            # Check if already in WGS84
            if current_crs is not None and current_crs.to_epsg() == 4326:
                self.logger.info(f"Shapefile {shapefile_path.name} already in WGS84")
                if ensure_unique_ids:
                    return shapefile_path, actual_hru_field
                return shapefile_path

            # Create WGS84 version
            wgs84_shapefile = shapefile_path.parent / f"{shapefile_path.stem}{output_suffix}.shp"

            # Check if valid WGS84 version already exists
            if wgs84_shapefile.exists():
                existing = self._validate_existing_wgs84(
                    wgs84_shapefile, ensure_unique_ids, actual_hru_field
                )
                if existing is not None:
                    return existing  # type: ignore[return-value]

            # Convert to WGS84
            self.logger.info(f"Converting {shapefile_path.name} from {current_crs} to WGS84")
            gdf_wgs84 = gdf.to_crs('EPSG:4326')
            gdf_wgs84.to_file(wgs84_shapefile)
            self.logger.info(f"WGS84 shapefile created: {wgs84_shapefile}")

            if ensure_unique_ids:
                saved_gdf = gpd.read_file(wgs84_shapefile)
                possible_fields = [col for col in saved_gdf.columns if col.startswith('hru_id')]
                if possible_fields:
                    actual_saved_field = possible_fields[0]
                    self.logger.info(f"Using field '{actual_saved_field}' from WGS84 shapefile")
                    return wgs84_shapefile, actual_saved_field
                # actual_hru_field is guaranteed to be str at this point due to earlier initialization
                return wgs84_shapefile, actual_hru_field

            return wgs84_shapefile

        except Exception as e:
            self.logger.error(f"Error ensuring WGS84 for {shapefile_path}: {str(e)}")
            raise

    def _validate_existing_wgs84(
        self,
        wgs84_path: Path,
        check_unique_ids: bool,
        hru_field: Optional[str]
    ) -> Optional[Union[Path, Tuple[Path, str]]]:
        """Check if existing WGS84 file is valid and return it if so."""
        try:
            wgs84_gdf = gpd.read_file(wgs84_path)
            if wgs84_gdf.crs is None or wgs84_gdf.crs.to_epsg() != 4326:
                self.logger.warning("Existing WGS84 file has wrong CRS. Recreating.")
                return None

            if check_unique_ids:
                possible_fields = [col for col in wgs84_gdf.columns if col.startswith('hru_id')]
                if possible_fields and wgs84_gdf[possible_fields[0]].nunique() == len(wgs84_gdf):
                    self.logger.info(f"WGS84 version with unique IDs exists: {wgs84_path.name}")
                    return wgs84_path, possible_fields[0]
                self.logger.warning("Existing WGS84 file missing unique ID field. Recreating.")
                return None

            self.logger.info(f"WGS84 version already exists: {wgs84_path.name}")
            return wgs84_path

        except Exception as e:
            self.logger.warning(f"Error reading existing WGS84 file: {str(e)}. Recreating.")
            return None

    def ensure_unique_hru_ids(
        self,
        shapefile_path: Path,
        hru_id_field: str
    ) -> Tuple[Path, str]:
        """
        Ensure HRU IDs are unique in the shapefile.

        Args:
            shapefile_path: Path to the shapefile
            hru_id_field: Name of the HRU ID field

        Returns:
            Tuple of (updated_shapefile_path, actual_hru_id_field_used)

        Raises:
            ValueError: If HRU ID field not found or cannot create unique IDs
        """
        try:
            gdf = gpd.read_file(shapefile_path)
            self.logger.info(f"Checking HRU ID uniqueness in {shapefile_path.name}")
            self.logger.info(f"Available fields: {list(gdf.columns)}")

            if hru_id_field not in gdf.columns:
                raise ValueError(f"HRU ID field '{hru_id_field}' not found in shapefile")

            original_count = len(gdf)
            unique_count = gdf[hru_id_field].nunique()

            self.logger.info(
                f"Shapefile has {original_count} rows, {unique_count} unique {hru_id_field} values"
            )

            if unique_count == original_count:
                self.logger.info(f"All {hru_id_field} values are unique")
                return shapefile_path, hru_id_field

            # Handle duplicate IDs
            self.logger.warning(f"Found {original_count - unique_count} duplicate {hru_id_field} values")

            new_hru_field = "hru_id_new"  # 10 chars max for shapefile compatibility

            gdf_updated = gdf.copy()
            if new_hru_field in gdf.columns and gdf[new_hru_field].nunique() == len(gdf):
                self.logger.info(f"Using existing unique field: {new_hru_field}")
            else:
                self.logger.info(f"Creating new unique IDs in field: {new_hru_field}")
                gdf_updated[new_hru_field] = range(1, len(gdf_updated) + 1)

            output_path = shapefile_path.parent / f"{shapefile_path.stem}_unique_ids.shp"
            gdf_updated.to_file(output_path)
            self.logger.info(f"Updated shapefile with unique IDs saved to: {output_path}")

            # Verify the fix
            verify_gdf = gpd.read_file(output_path)
            possible_fields = [col for col in verify_gdf.columns if col.startswith('hru_id')]

            if not possible_fields:
                raise ValueError("Could not find unique HRU ID field in saved shapefile")

            actual_saved_field = possible_fields[0]
            if verify_gdf[actual_saved_field].nunique() != len(verify_gdf):
                raise ValueError("Could not create unique HRU IDs")

            self.logger.info(f"Verification successful: All {actual_saved_field} values are unique")
            return output_path, actual_saved_field

        except Exception as e:
            self.logger.error(f"Error ensuring unique HRU IDs: {str(e)}")
            raise

    def align_longitude_frame(
        self,
        target_shapefile: Path,
        source_shapefile: Path,
        output_dir: Path
    ) -> Tuple[Path, bool]:
        """
        Align target shapefile longitudes to match source grid frame (0-360 or -180/180).

        Args:
            target_shapefile: Path to target (catchment) shapefile
            source_shapefile: Path to source (forcing grid) shapefile
            output_dir: Directory for output files

        Returns:
            Tuple of (aligned_shapefile_path, correction_disabled)
        """
        try:
            source_gdf = gpd.read_file(source_shapefile)
            lon_field = self._get_config_value(lambda: self.config.forcing.shape_lon_name, dict_key='FORCING_SHAPE_LON_NAME')

            if lon_field not in source_gdf.columns:
                return target_shapefile, False

            source_lon_max = float(source_gdf[lon_field].max())

            if source_lon_max <= 180:
                return target_shapefile, False

            # Source uses 0-360 frame, check if target needs shifting
            target_gdf = gpd.read_file(target_shapefile)
            minx, _, maxx, _ = target_gdf.total_bounds

            if minx >= 0 and maxx >= 0:
                return target_shapefile, False

            # Shift target to 0-360 frame
            self.logger.info("Shifting target shapefile longitudes to 0-360 for EASYMORE")

            target_gdf = target_gdf.copy()
            target_gdf["geometry"] = target_gdf["geometry"].apply(
                lambda geom: translate(geom, xoff=360) if geom is not None else geom
            )

            target_lon_field = self._get_config_value(lambda: self.config.paths.catchment_lon, dict_key='CATCHMENT_SHP_LON')
            if target_lon_field in target_gdf.columns:
                target_gdf[target_lon_field] = target_gdf[target_lon_field].apply(
                    lambda v: v + 360 if v < 0 else v
                )

            shifted_path = output_dir / f"{target_shapefile.stem}_lon360.shp"
            target_gdf.to_file(shifted_path)

            return shifted_path, True

        except Exception as e:
            self.logger.warning(f"Failed to align target longitudes: {e}")
            return target_shapefile, False
