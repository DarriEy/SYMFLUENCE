"""
GDAL raster processing operations.

Provides raster to vector conversion for watershed delineation.
Used by both distributed and lumped delineators.

Refactored from geofabric_utils.py (2026-01-01)
"""

import os
import subprocess
import glob
from pathlib import Path
from typing import Any
import geopandas as gpd

try:
    from osgeo import gdal, ogr
except ImportError:
    gdal = None
    ogr = None


class GDALProcessor:
    """
    GDAL operations for raster processing.

    Handles raster-to-vector conversion using both GDAL library
    and command-line tools as fallback.
    """

    def __init__(self, logger: Any):
        """
        Initialize GDAL processor.

        Args:
            logger: Logger instance
        """
        if gdal is None:
            raise ImportError(
                "GDAL Python bindings (osgeo) are required for GDALProcessor. "
                "Install with: pip install GDAL"
            )
        self.logger = logger

    def run_gdal_processing(self, interim_dir: Path):
        """
        Convert watershed raster to polygon shapefile.

        Used by distributed delineation. Attempts direct GDAL polygonization
        first, falls back to command-line tool if that fails.

        Args:
            interim_dir: Directory containing interim TauDEM outputs

        Raises:
            RuntimeError: If all polygonization attempts fail
        """
        # Ensure output directory exists
        interim_dir.mkdir(parents=True, exist_ok=True)

        input_raster = str(interim_dir / "elv-watersheds.tif")
        output_shapefile = str(interim_dir / "basin-watersheds.shp")

        try:
            # First attempt: Using gdal.Polygonize directly
            src_ds = gdal.Open(input_raster)
            if src_ds is None:
                raise RuntimeError(f"Could not open input raster: {input_raster}")

            srcband = src_ds.GetRasterBand(1)

            # Create output shapefile
            drv = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(output_shapefile):
                drv.DeleteDataSource(output_shapefile)

            dst_ds = drv.CreateDataSource(output_shapefile)
            if dst_ds is None:
                raise RuntimeError(f"Could not create output shapefile: {output_shapefile}")

            dst_layer = dst_ds.CreateLayer("watersheds", srs=None)
            if dst_layer is None:
                raise RuntimeError("Could not create output layer")

            # Add field for raster value
            fd = ogr.FieldDefn("DN", ogr.OFTInteger)
            dst_layer.CreateField(fd)

            # Run polygonize
            gdal.Polygonize(srcband, srcband.GetMaskBand(), dst_layer, 0)

            # Cleanup
            dst_ds = None
            src_ds = None

            self.logger.info("Completed GDAL polygonization using direct method")

        except Exception as e:
            self.logger.warning(f"Direct polygonization failed: {str(e)}, trying command line method...")
            try:
                # Second attempt: Using command line tool without MPI
                command = [
                    "gdal_polygonize.py",
                    "-f", "ESRI Shapefile",
                    str(input_raster),
                    str(output_shapefile)
                ]
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.logger.info("Completed GDAL polygonization using command line method")

            except subprocess.CalledProcessError as cmd_err:
                error_msg = f"gdal_polygonize.py failed with exit code {cmd_err.returncode}"
                if cmd_err.stderr:
                    error_msg += f"\nstderr: {cmd_err.stderr}"
                if cmd_err.stdout:
                    error_msg += f"\nstdout: {cmd_err.stdout}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from cmd_err
            except FileNotFoundError:
                self.logger.error("gdal_polygonize.py not found in PATH")
                raise RuntimeError(
                    "gdal_polygonize.py not found. Ensure GDAL is installed and in PATH. "
                    "On macOS, try: brew install gdal"
                )
            except Exception as fallback_err:
                self.logger.error(f"All polygonization attempts failed: {str(fallback_err)}")
                raise

    def raster_to_polygon(self, raster_path: Path, output_shp_path: Path):
        """
        Convert a raster to a polygon shapefile.

        Used by lumped delineation. Filters to keep only polygon with ID = 1.

        Args:
            raster_path: Path to the input raster file
            output_shp_path: Path to save the output shapefile

        Raises:
            ValueError: If no polygon with ID = 1 is found
        """
        gdal.UseExceptions()
        ogr.UseExceptions()

        # Open the raster
        raster = gdal.Open(str(raster_path))
        band = raster.GetRasterBand(1)

        # Create a temporary shapefile
        temp_shp_path = output_shp_path.with_name(output_shp_path.stem + "_temp.shp")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        temp_ds = driver.CreateDataSource(str(temp_shp_path))
        temp_layer = temp_ds.CreateLayer("watershed", srs=None)

        # Add a field to the layer
        field_def = ogr.FieldDefn("ID", ogr.OFTInteger)
        temp_layer.CreateField(field_def)

        # Polygonize the raster
        gdal.Polygonize(band, None, temp_layer, 0, [], callback=None)

        # Close the temporary datasource
        temp_ds = None
        raster = None

        # Read the temporary shapefile with geopandas
        gdf = gpd.read_file(temp_shp_path)

        # Filter to keep only the shape with ID = 1
        filtered_gdf = gdf[gdf['ID'] == 1]
        filtered_gdf = filtered_gdf.set_crs('epsg:4326')

        if filtered_gdf.empty:
            self.logger.error("No polygon with ID = 1 found in the watershed shapefile.")
            raise ValueError("No polygon with ID = 1 found in the watershed shapefile.")

        # Save the filtered GeoDataFrame to the final shapefile
        filtered_gdf.to_file(output_shp_path)

        # Remove all temporary files
        temp_files = glob.glob(str(temp_shp_path.with_suffix(".*")))
        for temp_file in temp_files:
            Path(temp_file).unlink()
            self.logger.debug(f"Removed temporary file: {temp_file}")

        self.logger.debug(f"Filtered watershed shapefile created at: {output_shp_path}")
