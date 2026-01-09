"""
DEPRECATED: Legacy HYPE flow functions.

This module is deprecated and maintained only for backward compatibility.
Please use the new manager classes instead:

- HYPEForcingProcessor: For forcing data processing
- HYPEConfigManager: For configuration files (info.txt, par.txt, filedir.txt)
- HYPEGeoDataManager: For geographic data files (GeoData.txt, GeoClass.txt, ForcKey.txt)

Example migration:
    # Old way (deprecated):
    from symfluence.models.hype.hypeFlow import (
        write_hype_forcing,
        write_hype_geo_files,
        write_hype_par_file,
        write_hype_info_filedir_files
    )

    # New way (recommended):
    from symfluence.models.hype import (
        HYPEForcingProcessor,
        HYPEConfigManager,
        HYPEGeoDataManager
    )

    # Use manager classes
    forcing_processor = HYPEForcingProcessor(config, logger, ...)
    forcing_processor.process_forcing()

    geodata_manager = HYPEGeoDataManager(config, logger, output_path, geofabric_mapping)
    land_uses = geodata_manager.create_geofiles(...)

    config_manager = HYPEConfigManager(config, logger, output_path)
    config_manager.write_par_file(params=params, land_uses=land_uses)
    config_manager.write_info_filedir(spinup_days, results_dir, ...)
"""

import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd

# Import the new manager classes
from .geodata_manager import HYPEGeoDataManager
from .config_manager import HYPEConfigManager

# Emit deprecation warning on import
warnings.warn(
    "The hypeFlow module is deprecated. Please use HYPEForcingProcessor, "
    "HYPEConfigManager, and HYPEGeoDataManager instead.",
    DeprecationWarning,
    stacklevel=2
)


def sort_geodata(geodata: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED: Sort sub-basins from upstream to downstream.

    Use HYPEGeoDataManager.sort_geodata() instead.
    """
    warnings.warn(
        "sort_geodata is deprecated. Use HYPEGeoDataManager.sort_geodata() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a minimal manager to use its method
    manager = HYPEGeoDataManager(
        config={},
        logger=None,
        output_path=Path('.'),
        geofabric_mapping={}
    )
    return manager.sort_geodata(geodata)


def write_hype_forcing(
    easymore_output,
    timeshift,
    forcing_units,
    geofabric_mapping,
    path_to_save,
    cache_path
):
    """
    DEPRECATED: Write HYPE forcing from easymore nc files.

    Use HYPEForcingProcessor.process_forcing() instead.
    """
    warnings.warn(
        "write_hype_forcing is deprecated. Use HYPEForcingProcessor.process_forcing() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # This function is now fully handled by HYPEForcingProcessor
    # For backward compatibility, raise an error directing to the new class
    raise NotImplementedError(
        "write_hype_forcing has been replaced by HYPEForcingProcessor. "
        "Please use:\n"
        "    from symfluence.models.hype import HYPEForcingProcessor\n"
        "    processor = HYPEForcingProcessor(config, logger, forcing_input_dir, ...)\n"
        "    processor.process_forcing()"
    )


def write_hype_geo_files(
    gistool_output,
    subbasins_shapefile,
    rivers_shapefile,
    frac_threshold,
    geofabric_mapping,
    path_to_save,
    intersect_base_path=None
) -> np.ndarray:
    """
    DEPRECATED: Write GeoData and GeoClass files.

    Use HYPEGeoDataManager.create_geofiles() instead.
    """
    warnings.warn(
        "write_hype_geo_files is deprecated. Use HYPEGeoDataManager.create_geofiles() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    manager = HYPEGeoDataManager(
        config={},
        logger=None,
        output_path=Path(path_to_save),
        geofabric_mapping=geofabric_mapping
    )
    return manager.create_geofiles(
        gistool_output=Path(gistool_output),
        subbasins_shapefile=Path(subbasins_shapefile),
        rivers_shapefile=Path(rivers_shapefile),
        frac_threshold=frac_threshold,
        intersect_base_path=Path(intersect_base_path) if intersect_base_path else None
    )


def write_hype_par_file(
    path_to_save,
    params: Optional[Dict[str, Any]] = None,
    template_file: Optional[str] = None,
    land_uses: Optional[np.ndarray] = None
) -> None:
    """
    DEPRECATED: Write par.txt file.

    Use HYPEConfigManager.write_par_file() instead.
    """
    warnings.warn(
        "write_hype_par_file is deprecated. Use HYPEConfigManager.write_par_file() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    manager = HYPEConfigManager(
        config={},
        logger=None,
        output_path=Path(path_to_save)
    )
    manager.write_par_file(
        params=params,
        template_file=Path(template_file) if template_file else None,
        land_uses=land_uses
    )


def write_hype_info_filedir_files(
    path_to_save,
    spinup_days: int,
    hype_results_dir: str,
    experiment_start: Optional[str] = None,
    experiment_end: Optional[str] = None
) -> None:
    """
    DEPRECATED: Write info.txt and filedir.txt files.

    Use HYPEConfigManager.write_info_filedir() instead.
    """
    warnings.warn(
        "write_hype_info_filedir_files is deprecated. Use HYPEConfigManager.write_info_filedir() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    manager = HYPEConfigManager(
        config={},
        logger=None,
        output_path=Path(path_to_save)
    )
    manager.write_info_filedir(
        spinup_days=spinup_days,
        results_dir=hype_results_dir,
        experiment_start=experiment_start,
        experiment_end=experiment_end
    )


def write_geoclass(slc_df: pd.DataFrame, path_to_save) -> None:
    """
    DEPRECATED: Write GeoClass.txt file.

    This is now handled internally by HYPEGeoDataManager.create_geofiles().
    """
    warnings.warn(
        "write_geoclass is deprecated. It's now handled internally by "
        "HYPEGeoDataManager.create_geofiles().",
        DeprecationWarning,
        stacklevel=2
    )
    manager = HYPEGeoDataManager(
        config={},
        logger=None,
        output_path=Path(path_to_save),
        geofabric_mapping={}
    )
    manager._write_geoclass(slc_df)


# Legacy helper function - now internal to HYPEGeoDataManager
def _get_projected_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    DEPRECATED: Calculate centroids in a projected CRS.

    This is now handled internally by HYPEGeoDataManager.
    """
    warnings.warn(
        "_get_projected_centroids is deprecated. It's now handled internally by "
        "HYPEGeoDataManager._get_projected_centroids().",
        DeprecationWarning,
        stacklevel=2
    )
    manager = HYPEGeoDataManager(
        config={},
        logger=None,
        output_path=Path('.'),
        geofabric_mapping={}
    )
    return manager._get_projected_centroids(gdf)


# Legacy helper function - now internal to HYPEConfigManager
def _generate_landuse_params(land_uses: np.ndarray) -> tuple[Dict[str, str], int]:
    """
    DEPRECATED: Generate land-use-dependent parameter values.

    This is now handled internally by HYPEConfigManager.
    """
    warnings.warn(
        "_generate_landuse_params is deprecated. It's now handled internally by "
        "HYPEConfigManager._generate_landuse_params().",
        DeprecationWarning,
        stacklevel=2
    )
    manager = HYPEConfigManager(
        config={},
        logger=None,
        output_path=Path('.')
    )
    return manager._generate_landuse_params(land_uses)
