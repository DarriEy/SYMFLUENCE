"""
Domain discretization core module for Hydrologic Response Unit (HRU) creation.

Provides the DomainDiscretizer class for subdividing catchments into HRUs
based on elevation bands, soil classes, land cover, aspect, or radiation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd  # type: ignore
import numpy as np  # type: ignore
import rasterio  # type: ignore
import rasterstats  # type: ignore
from pyproj import CRS  # type: ignore
from rasterio.mask import mask  # type: ignore
from shapely.geometry import MultiPolygon, Polygon, shape  # type: ignore

from symfluence.core.path_resolver import PathResolverMixin
from symfluence.geospatial.geometry_utils import clean_geometry
from symfluence.geospatial.raster_utils import analyze_raster_values
from .artifacts import DiscretizationArtifacts
from .attributes import aspect, combined, elevation, grus, landclass, radiation, soilclass


class DomainDiscretizer(PathResolverMixin):
    """
    A class for discretizing a domain into Hydrologic Response Units (HRUs).

    This class provides methods for various types of domain discretization,
    including elevation-based, soil class-based, land class-based, and
    radiation-based discretization. HRUs are allowed to be MultiPolygons,
    meaning spatially disconnected areas with the same attributes are
    grouped into single HRUs.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        logger: Logger object for logging information and errors.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # Standard paths from ProjectContextMixin
        self.catchment_dir = self.ensure_dir(self.project_shapefiles_dir / "catchment")
        
        self.dem_path = self._get_file_path(
            path_key="DEM_PATH",
            name_key="DEM_NAME",
            default_subpath="attributes/elevation/dem",
            default_name=f"domain_{self.domain_name}_elv.tif"
        )

        delineation_method = self._get_config_value("DOMAIN_DEFINITION_METHOD", "delineate")

        if delineation_method == "delineate":
            self.delineation_suffix = "delineate"
        elif delineation_method == "lumped":
            self.delineation_suffix = "lumped"
        elif delineation_method == "subset":
            self.delineation_suffix = f"subset_{self._get_config_value('GEOFABRIC_TYPE')}"

    def sort_catchment_shape(self):
        """
        Sort the catchment shapefile based on GRU and HRU IDs.

        This method performs the following steps:
        1. Loads the catchment shapefile
        2. Sorts the shapefile based on GRU and HRU IDs
        3. Saves the sorted shapefile back to the original location

        The method uses GRU and HRU ID column names specified in the configuration.

        Raises:
            FileNotFoundError: If the catchment shapefile is not found.
            ValueError: If the required ID columns are not present in the shapefile.
        """
        self.logger.debug("Sorting catchment shape")

        self.catchment_path = self.config.get("CATCHMENT_PATH")
        self.catchment_name = self.config.get("CATCHMENT_SHP_NAME")
        if self.catchment_name == "default":
            discretization_method = self.config.get("DOMAIN_DISCRETIZATION")
            # Handle comma-separated attributes for output filename
            if "," in discretization_method:
                method_suffix = discretization_method.replace(",", "_")
            else:
                method_suffix = discretization_method
            self.catchment_name = (
                f"{self.config.get('DOMAIN_NAME')}_HRUs_{method_suffix}.shp"
            )
        self.gruId = self.config.get("CATCHMENT_SHP_GRUID")
        self.hruId = self.config.get("CATCHMENT_SHP_HRUID")

        if self.catchment_path == "default":
            self.catchment_path = self.project_dir / "shapefiles" / "catchment"
        else:
            self.catchment_path = Path(self.catchment_path)

        catchment_file = self.catchment_path / self.catchment_name

        try:
            # Open the shape
            shp = gpd.read_file(catchment_file)

            # Check if required columns exist
            if self.gruId not in shp.columns or self.hruId not in shp.columns:
                raise ValueError(
                    f"Required columns {self.gruId} and/or {self.hruId} not found in shapefile"
                )

            # Sort
            shp = shp.sort_values(by=[self.gruId, self.hruId])

            # Save
            shp.to_file(catchment_file)

            self.logger.debug(f"Catchment shape sorted and saved to {catchment_file}")
            return catchment_file
        except FileNotFoundError:
            self.logger.error(f"Catchment shapefile not found at {catchment_file}")
            raise
        except ValueError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error sorting catchment shape: {str(e)}")
            raise

    def discretize_domain(self) -> Optional[Path]:
        """
        Discretize the domain based on the method specified in the configuration.
        If CATCHMENT_SHP_NAME is provided and not 'default', it uses the provided shapefile instead.
        Supports both single attributes and comma-separated multiple attributes.
        """
        with self.time_limit("Domain Discretization"):
            # Check if a custom catchment shapefile is provided
            catchment_name = self.config.get("CATCHMENT_SHP_NAME")
            if catchment_name != "default":
                self.logger.debug(f"Using provided catchment shapefile: {catchment_name}")

                # Just sort the existing shapefile
                return self.sort_catchment_shape()

            # Parse discretization method to check for multiple attributes
            discretization_config = self.config.get("DOMAIN_DISCRETIZATION")
            attributes = [attr.strip() for attr in discretization_config.split(",")]

            self.logger.debug(f"Discretizing using: {', '.join(attributes)}")

            # Handle single vs multiple attributes
            if len(attributes) == 1:
                # Single attribute - use existing logic
                discretization_method = attributes[0].lower()
                method_map = {
                    "grus": grus.discretize,
                    "elevation": elevation.discretize,
                    "aspect": aspect.discretize,
                    "soilclass": soilclass.discretize,
                    "landclass": landclass.discretize,
                    "radiation": radiation.discretize,
                }

                if discretization_method not in method_map:
                    self.logger.error(
                        f"Invalid discretization method: {discretization_method}"
                    )
                    raise ValueError(
                        f"Invalid discretization method: {discretization_method}"
                    )

                method_map[discretization_method](self)
            else:
                # Multiple attributes - use combined discretization
                combined.discretize(self, attributes)

            return self.sort_catchment_shape()

    def _read_and_prepare_data(self, shapefile_path, raster_path, band_size=None):
        """
        Read and prepare data with chunking for large rasters.

        Args:
            shapefile_path: Path to the GRU shapefile
            raster_path: Path to the raster file
            band_size: Optional band size for discretization

        Returns:
            tuple: (gru_gdf, thresholds) where:
                - gru_gdf is the GeoDataFrame containing GRU data
                - thresholds are the class boundaries for discretization
        """
        # Read the GRU shapefile
        gru_gdf = self._read_shapefile(shapefile_path)

        # Use shared utility for raster analysis
        thresholds = analyze_raster_values(
            raster_path=raster_path,
            band_size=band_size,
            logger=self.logger
        )

        return gru_gdf, thresholds

    def _create_multipolygon_hrus(self, gru_gdf, raster_path, thresholds, attribute_name):
        """
        Create HRUs by discretizing each GRU based on raster values within it.
        Each unique raster value within a GRU becomes an HRU (Polygon or MultiPolygon).

        Args:
            gru_gdf: GeoDataFrame containing GRU data
            raster_path: Path to the classification raster
            thresholds: Array of threshold values for classification
            attribute_name: Name of the attribute column

        Returns:
            GeoDataFrame containing HRUs
        """
        self.logger.info(
            f"Creating HRUs within {len(gru_gdf)} GRUs based on {attribute_name}"
        )

        all_hrus = []
        hru_id_counter = 1

        # Process each GRU individually
        for gru_idx, gru_row in gru_gdf.iterrows():
            self.logger.info(f"Processing GRU {gru_idx + 1}/{len(gru_gdf)}")

            gru_geometry = gru_row.geometry
            gru_id = gru_row.get("GRU_ID", gru_idx + 1)

            # Extract raster data within this GRU
            with rasterio.open(raster_path) as src:
                try:
                    # Mask the raster to this GRU's geometry
                    out_image, out_transform = mask(
                        src,
                        [gru_geometry],
                        crop=True,
                        all_touched=True,
                        filled=False,
                    )
                    out_image = out_image[0]
                    nodata_value = src.nodata
                except Exception as e:
                    self.logger.warning(
                        f"Could not extract raster data for GRU {gru_id}: {str(e)}"
                    )
                    continue

            # Create mask for valid pixels
            if nodata_value is not None:
                valid_mask = out_image != nodata_value
            else:
                valid_mask = (
                    ~np.isnan(out_image)
                    if out_image.dtype == np.float64
                    else np.ones_like(out_image, dtype=bool)
                )

            if not np.any(valid_mask):
                self.logger.warning(f"No valid pixels found in GRU {gru_id}")
                continue

            # Find unique values within this GRU
            valid_values = out_image[valid_mask]

            if attribute_name in ["elevClass", "radiationClass"]:
                # For continuous data, classify into bands
                gru_hrus = self._create_hrus_from_bands(
                    out_image,
                    valid_mask,
                    out_transform,
                    thresholds,
                    attribute_name,
                    gru_geometry,
                    gru_row,
                    hru_id_counter,
                )
            else:
                # For discrete classes, use unique values
                unique_values = np.unique(valid_values)
                gru_hrus = self._create_hrus_from_classes(
                    out_image,
                    valid_mask,
                    out_transform,
                    unique_values,
                    attribute_name,
                    gru_geometry,
                    gru_row,
                    hru_id_counter,
                )

            all_hrus.extend(gru_hrus)
            hru_id_counter += len(gru_hrus)

        self.logger.info(f"Created {len(all_hrus)} HRUs across all GRUs")
        return gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)

    def _create_hrus_from_bands(
        self,
        raster_data,
        valid_mask,
        transform,
        thresholds,
        attribute_name,
        gru_geometry,
        gru_row,
        start_hru_id,
    ):
        """Create HRUs from elevation/radiation bands within a single GRU."""
        hrus = []
        current_hru_id = start_hru_id

        for i in range(len(thresholds) - 1):
            lower, upper = thresholds[i : i + 2]

            # Make the last band inclusive of the upper bound
            if i == len(thresholds) - 2:  # Last band
                class_mask = valid_mask & (raster_data >= lower) & (raster_data <= upper)
            else:
                class_mask = valid_mask & (raster_data >= lower) & (raster_data < upper)

            if np.any(class_mask):
                hru = self._create_hru_from_mask(
                    class_mask,
                    transform,
                    raster_data,
                    gru_geometry,
                    gru_row,
                    current_hru_id,
                    attribute_name,
                    i + 1,
                )
                if hru:
                    hrus.append(hru)
                    current_hru_id += 1

        return hrus

    def _create_hrus_from_classes(
        self,
        raster_data,
        valid_mask,
        transform,
        unique_values,
        attribute_name,
        gru_geometry,
        gru_row,
        start_hru_id,
    ):
        """Create HRUs from discrete classes within a single GRU."""
        hrus = []
        current_hru_id = start_hru_id

        for class_value in unique_values:
            class_mask = valid_mask & (raster_data == class_value)

            if np.any(class_mask):
                hru = self._create_hru_from_mask(
                    class_mask,
                    transform,
                    raster_data,
                    gru_geometry,
                    gru_row,
                    current_hru_id,
                    attribute_name,
                    class_value,
                )
                if hru:
                    hrus.append(hru)
                    current_hru_id += 1

        return hrus

    def _create_hru_from_mask(
        self,
        class_mask,
        transform,
        raster_data,
        gru_geometry,
        gru_row,
        hru_id,
        attribute_name,
        class_value,
    ):
        """Create a single HRU from a class mask within a GRU."""
        try:
            # Extract shapes from the mask
            shapes = list(
                rasterio.features.shapes(
                    class_mask.astype(np.uint8),
                    mask=class_mask,
                    transform=transform,
                    connectivity=4,
                )
            )

            if not shapes:
                return None

            # Create polygons from shapes
            polygons = []
            for shp, _ in shapes:
                try:
                    geom = shape(shp)
                    if geom.is_valid and not geom.is_empty and geom.area > 0:
                        polygons.append(geom)
                except Exception:
                    continue

            if not polygons:
                return None

            # Create final geometry (naturally Polygon or MultiPolygon)
            if len(polygons) == 1:
                final_geometry = polygons[0]
            else:
                # This naturally creates a MultiPolygon if there are disconnected areas
                final_geometry = MultiPolygon(polygons)

            # Clean the geometry
            if not final_geometry.is_valid:
                final_geometry = final_geometry.buffer(0)

            if final_geometry.is_empty or not final_geometry.is_valid:
                return None

            # Ensure it's within the GRU boundary
            clipped_geometry = final_geometry.intersection(gru_geometry)

            if clipped_geometry.is_empty or not clipped_geometry.is_valid:
                return None

            # Calculate average attribute value
            avg_value = (
                np.mean(raster_data[class_mask]) if np.any(class_mask) else class_value
            )

            # Create HRU data
            hru_data = {
                "geometry": clipped_geometry,
                "GRU_ID": gru_row.get("GRU_ID", gru_row.name),
                "HRU_ID": hru_id,
                attribute_name: class_value,
                f"avg_{attribute_name.lower()}": avg_value,
                "hru_type": f"{attribute_name}_within_gru",
            }

            # Copy relevant GRU attributes (excluding geometry)
            for col in gru_row.index:
                if col not in ["geometry", "GRU_ID"] and col not in hru_data:
                    hru_data[col] = gru_row[col]

            return hru_data

        except Exception as e:
            self.logger.warning(
                f"Error creating HRU for class {class_value} in GRU: {str(e)}"
            )
            return None

    def _create_single_multipolygon_hru(
        self,
        class_mask,
        out_transform,
        domain_boundary,
        class_value,
        out_image,
        attribute_name,
        gru_gdf,
    ):
        """
        Create a single MultiPolygon HRU from a class mask.

        Args:
            class_mask: Boolean mask for the class
            out_transform: Raster transform
            domain_boundary: Boundary of the domain
            class_value: Value of the class
            out_image: Original raster data
            attribute_name: Name of the attribute
            gru_gdf: Original GRU GeoDataFrame

        Returns:
            Dictionary representing the HRU
        """
        try:
            # Extract shapes from the mask
            shapes = list(
                rasterio.features.shapes(
                    class_mask.astype(np.uint8),
                    mask=class_mask,
                    transform=out_transform,
                    connectivity=8,
                )
            )

            if not shapes:
                return None

            # Create polygons from shapes
            polygons = []
            for shp, _ in shapes:
                geom = shape(shp)
                if geom.is_valid and not geom.is_empty:
                    # Intersect with domain boundary to ensure it's within the domain
                    intersected = geom.intersection(domain_boundary)
                    if not intersected.is_empty:
                        if isinstance(intersected, (Polygon, MultiPolygon)):
                            polygons.append(intersected)

            if not polygons:
                return None

            # Create a single MultiPolygon from all polygons
            if len(polygons) == 1:
                multipolygon = polygons[0]
            else:
                multipolygon = MultiPolygon(polygons)

            # Clean the geometry
            multipolygon = multipolygon.buffer(0)  # Fix any topology issues

            if multipolygon.is_empty or not multipolygon.is_valid:
                return None

            # Calculate average attribute value
            avg_value = np.mean(out_image[class_mask])

            # Get a representative GRU for metadata (use the first one)
            representative_gru = gru_gdf.iloc[0]

            return {
                "geometry": multipolygon,
                "GRU_ID": 1,  # Single domain-wide unit
                attribute_name: class_value,
                f"avg_{attribute_name.lower()}": avg_value,
                "HRU_ID": class_value,  # Use class value as HRU ID
                "hru_type": f"{attribute_name}_multipolygon",
            }

        except Exception as e:
            self.logger.warning(
                f"Error creating MultiPolygon HRU for class {class_value}: {str(e)}"
            )
            return None

    def _clean_and_prepare_hru_gdf(self, hru_gdf):
        """
        Clean and prepare the HRU GeoDataFrame for output.
        """
        # Ensure all geometries are valid
        hru_gdf["geometry"] = hru_gdf["geometry"].apply(lambda g: clean_geometry(g, self.logger))
        hru_gdf = hru_gdf[hru_gdf["geometry"].notnull()]

        # Final check: ensure only Polygon or MultiPolygon geometries
        valid_rows = []
        for idx, row in hru_gdf.iterrows():
            geom = row["geometry"]
            if (
                isinstance(geom, (Polygon, MultiPolygon))
                and geom.is_valid
                and not geom.is_empty
            ):
                valid_rows.append(row)
            else:
                self.logger.warning(
                    f"Removing HRU {idx} with invalid geometry type: {type(geom)}"
                )

        if not valid_rows:
            self.logger.error("No valid HRUs after final geometry validation")
            return gpd.GeoDataFrame(columns=hru_gdf.columns, crs=hru_gdf.crs)

        hru_gdf = gpd.GeoDataFrame(valid_rows, crs=hru_gdf.crs)

        self.logger.info(f"Retained {len(hru_gdf)} HRUs after geometry validation")

        # Calculate areas and centroids
        self.logger.info("Calculating HRU areas and centroids")

        # Project to UTM for accurate area calculation
        utm_crs = hru_gdf.estimate_utm_crs()
        hru_gdf_utm = hru_gdf.to_crs(utm_crs)
        hru_gdf_utm["HRU_area"] = hru_gdf_utm.geometry.area

        # Calculate centroids (use representative point for MultiPolygons)
        centroids_utm = hru_gdf_utm.geometry.representative_point()
        centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))

        hru_gdf_utm["center_lon"] = centroids_wgs84.x
        hru_gdf_utm["center_lat"] = centroids_wgs84.y

        # Convert back to original CRS
        hru_gdf = hru_gdf_utm.to_crs(hru_gdf.crs)

        # Calculate mean elevation for each HRU with proper CRS handling
        self.logger.info("Calculating mean elevation for each HRU")
        try:
            # Get CRS information
            with rasterio.open(self.dem_path) as src:
                dem_crs = src.crs
                dem_array = src.read(1)
                dem_transform = src.transform
                dem_nodata = src.nodata

            shapefile_crs = hru_gdf.crs

            # Check if CRS match
            if dem_crs != shapefile_crs:
                self.logger.info(
                    f"CRS mismatch detected. Reprojecting HRUs from {shapefile_crs} to {dem_crs}"
                )
                hru_gdf_projected = hru_gdf.to_crs(dem_crs)
            else:
                hru_gdf_projected = hru_gdf.copy()

            # Use rasterstats with the raster array and transform
            zs = rasterstats.zonal_stats(
                hru_gdf_projected.geometry,
                dem_array,
                affine=dem_transform,
                stats=["mean"],
                nodata=dem_nodata if dem_nodata is not None else -9999,
            )
            hru_gdf["elev_mean"] = [
                item["mean"] if item["mean"] is not None else -9999 for item in zs
            ]

        except Exception as e:
            self.logger.error(f"Error calculating mean elevation: {str(e)}")
            hru_gdf["elev_mean"] = -9999

        # Ensure HRU_ID is sequential if not already set properly
        if "HRU_ID" in hru_gdf.columns:
            # Reset HRU_ID to be sequential
            hru_gdf = hru_gdf.sort_values(["GRU_ID", "HRU_ID"])
            hru_gdf["HRU_ID"] = range(1, len(hru_gdf) + 1)
        else:
            hru_gdf["HRU_ID"] = range(1, len(hru_gdf) + 1)

        return hru_gdf



    def _read_shapefile(self, shapefile_path):
        """
        Read a shapefile and return it as a GeoDataFrame.

        Args:
            shapefile_path (str or Path): Path to the shapefile.

        Returns:
            gpd.GeoDataFrame: The shapefile content as a GeoDataFrame.
        """
        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs is None:
                self.logger.warning(
                    f"CRS is not defined for {shapefile_path}. Setting to EPSG:4326."
                )
                gdf = gdf.set_crs("EPSG:4326")
            return gdf
        except Exception as e:
            self.logger.error(f"Error reading shapefile {shapefile_path}: {str(e)}")
            raise




class DomainDiscretizationRunner:
    """
    Wraps domain discretization with explicit artifact tracking.
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.discretizer = DomainDiscretizer(self.config, self.logger)

    def discretize_domain(
        self,
    ) -> Tuple[Optional[Union[object, Dict[str, object]]], DiscretizationArtifacts]:
        method = self.config.get("DOMAIN_DISCRETIZATION")
        hru_paths = self.discretizer.discretize_domain()
        artifacts = DiscretizationArtifacts(method=method, hru_paths=hru_paths)

        # Check if we need to discretize delineated catchments for lumped+river_network case
        domain_method = self.config.get("DOMAIN_DEFINITION_METHOD", "lumped")
        routing_delineation = self.config.get("ROUTING_DELINEATION", "lumped")

        if domain_method == "lumped" and routing_delineation == "river_network":
            self.logger.info("Discretizing delineated catchments for lumped-to-distributed routing")
            delineated_hru_path = self._discretize_delineated_domain(method)
            if delineated_hru_path:
                artifacts.metadata['delineated_hru_path'] = str(delineated_hru_path)
                self.logger.info(f"Delineated catchments discretized: {delineated_hru_path}")
            else:
                self.logger.warning("Failed to discretize delineated domain")

        return hru_paths, artifacts

    def _discretize_delineated_domain(self, method: str) -> Optional[Path]:
        """
        Discretize the delineated river basins and create catchment_delineated shapefile.

        Args:
            method: Discretization method (e.g., 'elevation', 'grus')

        Returns:
            Path to the discretized catchment_delineated shapefile, or None if failed
        """
        try:
            import geopandas as gpd

            # Get paths using the discretizer's path utilities
            domain_name = self.discretizer.domain_name
            project_dir = self.discretizer.project_dir

            # Find delineated river basins file from delineation step
            delineated_basins_path = project_dir / "shapefiles" / "river_basins" / f"{domain_name}_riverBasins_delineate.shp"

            if not delineated_basins_path.exists():
                self.logger.warning(f"Delineated river basins not found at {delineated_basins_path}")
                return None

            # Create output path for delineated catchments
            catchment_delineated_path = project_dir / "shapefiles" / "catchment" / f"{domain_name}_catchment_delineated.shp"
            catchment_delineated_path.parent.mkdir(parents=True, exist_ok=True)

            # Read the delineated river basins
            delineated_gdf = gpd.read_file(delineated_basins_path)

            self.logger.info(f"Read {len(delineated_gdf)} delineated subcatchments from {delineated_basins_path}")

            # For now, treat delineated subcatchments as HRUs
            # Add HRU_ID and other required fields
            delineated_gdf['HRU_ID'] = range(1, len(delineated_gdf) + 1)
            delineated_gdf['GRU_ID'] = range(1, len(delineated_gdf) + 1)  # Each subcatchment gets its own ID

            # Calculate fractional areas (avg_subbas) - sum to 1.0
            utm_crs = delineated_gdf.estimate_utm_crs()
            delineated_utm = delineated_gdf.to_crs(utm_crs)
            areas_utm = delineated_utm.geometry.area
            total_area = areas_utm.sum()

            delineated_gdf['avg_subbas'] = (areas_utm / total_area).values

            # Save the delineated catchments
            delineated_gdf.to_file(catchment_delineated_path)
            self.logger.info(f"Saved delineated catchments to {catchment_delineated_path}")
            self.logger.info(f"Fractional areas (avg_subbas): min={delineated_gdf['avg_subbas'].min():.4f}, max={delineated_gdf['avg_subbas'].max():.4f}, sum={delineated_gdf['avg_subbas'].sum():.4f}")

            return catchment_delineated_path

        except Exception as e:
            self.logger.error(f"Error discretizing delineated domain: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
