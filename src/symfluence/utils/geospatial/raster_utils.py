# In utils/geospatial/raster_utils.py
import numpy as np
import rasterio # type: ignore
from scipy import stats
import glob
import pandas as pd
import pvlib
from pathlib import Path
from logging import Logger
from typing import Optional

def calculate_landcover_mode(input_dir, output_file, start_year, end_year, domain_name):
    """
    Calculate the mode of land cover data across multiple years.
    
    Args:
        input_dir (Path): Directory containing the yearly land cover files
        output_file (Path): Path to save the output mode raster
        start_year (int): Start year for mode calculation
        end_year (int): End year for mode calculation
        domain_name (str): Name of the domain
    """
    
    # Create a list to store the data from each year
    yearly_data = []
    meta = None
    
    # Get a list of files matching the pattern for the specified years
    file_pattern = f"{input_dir}/domain_{domain_name}_*_{start_year}*.tif"
    files = glob.glob(str(file_pattern))
    
    if not files:
        # If no files match the start year, try to find any files in the directory
        file_pattern = f"{input_dir}/domain_{domain_name}_*.tif"
        files = glob.glob(str(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No land cover files found matching pattern: {file_pattern}")
    
    # Read metadata from the first file
    with rasterio.open(files[0]) as src:
        meta = src.meta.copy()
        shape = (src.height, src.width)
    
    # Read data for each year
    for year in range(start_year, end_year + 1):
        pattern = f"{input_dir}/domain_{domain_name}_*_{year}*.tif"
        year_files = glob.glob(str(pattern))
        
        if year_files:
            with rasterio.open(year_files[0]) as src:
                # Read the data and append to our list
                data = src.read(1)
                yearly_data.append(data)
    
    if not yearly_data:
        # If no yearly data was found, use the first file we found
        with rasterio.open(files[0]) as src:
            data = src.read(1)
            yearly_data.append(data)
    
    # Check if we have only one year of data
    if len(yearly_data) == 1:
        # Just use that single year's data
        mode_data = yearly_data[0]
    else:
        # Stack the arrays
        stacked_data = np.stack(yearly_data, axis=0)
        
        # Calculate the mode along the year axis (axis=0)
        # Using scipy.stats.mode with keepdims=False for newer scipy versions
        try:
            mode_data, _ = stats.mode(stacked_data, axis=0, keepdims=False)
        except TypeError:
            # For older scipy versions that don't have keepdims parameter
            mode_result = stats.mode(stacked_data, axis=0)
            mode_data = mode_result[0][0]  # Extract the mode values
    
    # Update the metadata for the output file
    meta.update({
        'count': 1,
        'nodata': 0
    })
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the result
    with rasterio.open(output_file, 'w', **meta) as dst:
        # Make sure mode_data has the right shape
        if mode_data.ndim == 1 or mode_data.shape != shape:
            # If the shape doesn't match, reshape it to the expected dimensions
            if mode_data.size == shape[0] * shape[1]:
                mode_data = mode_data.reshape(shape)
            else:
                # Create a new array with the correct shape
                new_mode_data = np.zeros(shape, dtype=meta['dtype'])
                
                # If mode_data is 1D but should be 2D
                if mode_data.ndim == 1:
                    # Take as many values as we can from mode_data
                    size = min(mode_data.size, shape[0] * shape[1])
                    new_mode_data.flat[:size] = mode_data[:size]
                    mode_data = new_mode_data
                else:
                    # If dimensions don't match but we can copy partial data
                    min_h = min(mode_data.shape[0], shape[0])
                    min_w = min(mode_data.shape[1], shape[1])
                    new_mode_data[:min_h, :min_w] = mode_data[:min_h, :min_w]
                    mode_data = new_mode_data
        
        # Now write the data
        dst.write(mode_data, 1)


def calculate_aspect(dem_raster: Path, aspect_raster: Path, aspect_class_number: int, logger: Logger) -> Optional[Path]:
    """
    Calculate aspect (slope direction) from DEM and classify into directional classes.
    
    Args:
        dem_raster: Path to the DEM raster
        aspect_raster: Path where the aspect raster will be saved
        aspect_class_number: Number of aspect classes to create
        logger: Logger object
        
    Returns:
        Path to the created aspect raster or None if failed
    """
    logger.info(f"Calculating aspect from DEM: {dem_raster}")
    
    try:
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        # Calculate gradients
        dy, dx = np.gradient(dem.astype(float))
        
        # Calculate aspect in radians, then convert to degrees
        aspect_rad = np.arctan2(-dx, dy)  # Note the negative sign for dx
        aspect_deg = np.degrees(aspect_rad)
        
        # Convert to compass bearing (0-360 degrees, 0 = North)
        aspect_deg = (90 - aspect_deg) % 360
        
        # Handle flat areas (where both dx and dy are near zero)
        slope_magnitude = np.sqrt(dx*dx + dy*dy)
        flat_threshold = 1e-6  # Adjust as needed
        flat_mask = slope_magnitude < flat_threshold
        
        # Classify aspect into directional classes
        classified_aspect = classify_aspect_into_classes(aspect_deg, flat_mask, aspect_class_number)
        
        # Handle nodata values from original DEM
        if nodata is not None:
            dem_nodata_mask = dem == nodata
            classified_aspect[dem_nodata_mask] = -9999
        
        # Save the classified aspect raster
        aspect_raster.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(aspect_raster, 'w', driver='GTiff',
                        height=classified_aspect.shape[0], width=classified_aspect.shape[1],
                        count=1, dtype=classified_aspect.dtype,
                        crs=crs, transform=transform, nodata=-9999) as dst:
            dst.write(classified_aspect, 1)
        
        logger.info(f"Aspect raster saved to: {aspect_raster}")
        logger.info(f"Aspect classes: {np.unique(classified_aspect[classified_aspect != -9999])}")
        return aspect_raster
    
    except Exception as e:
        logger.error(f"Error calculating aspect: {str(e)}", exc_info=True)
        return None

def classify_aspect_into_classes(aspect_deg: np.ndarray, flat_mask: np.ndarray, 
                                num_classes: int) -> np.ndarray:
    """
    Classify aspect degrees into directional classes.
    
    Args:
        aspect_deg: Aspect in degrees (0-360)
        flat_mask: Boolean mask for flat areas
        num_classes: Number of aspect classes to create
        
    Returns:
        Classified aspect array
    """
    classified = np.zeros_like(aspect_deg, dtype=int)
    
    if num_classes == 8:
        # Standard 8-direction classification
        # N, NE, E, SE, S, SW, W, NW
        bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 1]  # Last one wraps to North
        
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:  # Last bin (337.5 to 360)
                mask = (aspect_deg >= bins[i]) & (aspect_deg <= bins[i+1])
            else:
                mask = (aspect_deg >= bins[i]) & (aspect_deg < bins[i+1])
            classified[mask] = labels[i]
            
    elif num_classes == 4:
        # 4-direction classification (N, E, S, W)
        bins = [0, 45, 135, 225, 315, 360]
        labels = [1, 2, 3, 4, 1]  # N, E, S, W, N
        
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:  # Last bin
                mask = (aspect_deg >= bins[i]) & (aspect_deg <= bins[i+1])
            else:
                mask = (aspect_deg >= bins[i]) & (aspect_deg < bins[i+1])
            classified[mask] = labels[i]
    
    else:
        # Custom number of classes - divide 360 degrees evenly
        class_width = 360.0 / num_classes
        for i in range(num_classes):
            lower = i * class_width
            upper = (i + 1) * class_width
            
            if i == num_classes - 1:  # Last class includes 360
                mask = (aspect_deg >= lower) & (aspect_deg <= upper)
            else:
                mask = (aspect_deg >= lower) & (aspect_deg < upper)
            classified[mask] = i + 1
    
    # Set flat areas to a special class (0)
    classified[flat_mask] = 0
    
    # Set areas that don't fall into any class to -9999 (shouldn't happen but safety)
    classified[classified == 0] = 0  # Keep flat areas as 0
    
    return classified

def calculate_annual_radiation(dem_raster: Path, radiation_raster: Path, logger: Logger) -> Optional[Path]:
    """
    Calculate annual radiation from DEM.
    
    Args:
        dem_raster: Path to the DEM raster
        radiation_raster: Path where the radiation raster will be saved
        logger: Logger object
        
    Returns:
        Path to the created radiation raster or None if failed
    """
    logger.info(f"Calculating annual radiation from DEM: {dem_raster}")
    
    try:
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
        
        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2
        
        # Calculate slope and aspect
        dy, dx = np.gradient(dem)
        slope = np.arctan(np.sqrt(dx*dx + dy*dy))
        aspect = np.arctan2(-dx, dy)
        
        # Create a DatetimeIndex for the entire year (daily)
        times = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D')
        
        # Create location object
        location = pvlib.location.Location(latitude=center_lat, longitude=center_lon, altitude=np.mean(dem))
        
        # Calculate solar position
        solar_position = location.get_solarposition(times=times)
        
        # Calculate clear sky radiation
        clearsky = location.get_clearsky(times=times)
        
        # Initialize the radiation array
        radiation = np.zeros_like(dem)
        
        logger.info("Calculating radiation for each pixel...")
        for i in range(dem.shape[0]):
            for j in range(dem.shape[1]):
                surface_tilt = np.degrees(slope[i, j])
                surface_azimuth = np.degrees(aspect[i, j])
                
                total_irrad = pvlib.irradiance.get_total_irradiance(
                    surface_tilt, surface_azimuth,
                    solar_position['apparent_zenith'], solar_position['azimuth'],
                    clearsky['dni'], clearsky['ghi'], clearsky['dhi']
                )
                
                radiation[i, j] = total_irrad['poa_global'].sum()
        
        # Save the radiation raster
        radiation_raster.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(radiation_raster, 'w', driver='GTiff',
                        height=radiation.shape[0], width=radiation.shape[1],
                        count=1, dtype=radiation.dtype,
                        crs=crs, transform=transform) as dst:
            dst.write(radiation, 1)
        
        logger.info(f"Radiation raster saved to: {radiation_raster}")
        return radiation_raster
    
    except Exception as e:
        logger.error(f"Error calculating annual radiation: {str(e)}", exc_info=True)
        return None