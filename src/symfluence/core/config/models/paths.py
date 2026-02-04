"""
Path configuration models.

Contains ShapefilePathConfig and PathsConfig for file paths and directory structure.
"""

from pydantic import BaseModel, Field

from .base import FROZEN_CONFIG


class ShapefilePathConfig(BaseModel):
    """Shapefile path and field mapping configuration"""
    model_config = FROZEN_CONFIG

    path: str = Field(default='default')
    name: str = Field(default='default')


class PathsConfig(BaseModel):
    """File paths and directory structure configuration"""
    model_config = FROZEN_CONFIG

    # Shapefile paths with field mappings
    catchment_path: str = Field(default='default', alias='CATCHMENT_PATH')
    catchment_name: str = Field(default='default', alias='CATCHMENT_SHP_NAME')
    catchment_lat: str = Field(default='center_lat', alias='CATCHMENT_SHP_LAT')
    catchment_lon: str = Field(default='center_lon', alias='CATCHMENT_SHP_LON')
    catchment_area: str = Field(default='HRU_area', alias='CATCHMENT_SHP_AREA')
    catchment_elev: str = Field(default='elev_mean', alias='CATCHMENT_SHP_ELEV')
    catchment_slope: str = Field(default='slope', alias='CATCHMENT_SHP_SLOPE')
    catchment_hruid: str = Field(default='HRU_ID', alias='CATCHMENT_SHP_HRUID')
    catchment_gruid: str = Field(default='GRU_ID', alias='CATCHMENT_SHP_GRUID')

    river_basins_path: str = Field(default='default', alias='RIVER_BASINS_PATH')
    river_basins_name: str = Field(default='default', alias='RIVER_BASINS_NAME')
    river_basin_rm_gruid: str = Field(default='GRU_ID', alias='RIVER_BASIN_SHP_RM_GRUID')
    river_basin_hru_to_seg: str = Field(default='gru_to_seg', alias='RIVER_BASIN_SHP_HRU_TO_SEG')
    river_basin_area: str = Field(default='GRU_area', alias='RIVER_BASIN_SHP_AREA')

    river_network_path: str = Field(default='default', alias='RIVER_NETWORK_SHP_PATH')
    river_network_name: str = Field(default='default', alias='RIVER_NETWORK_SHP_NAME')
    river_network_length: str = Field(default='Length', alias='RIVER_NETWORK_SHP_LENGTH')
    river_network_segid: str = Field(default='LINKNO', alias='RIVER_NETWORK_SHP_SEGID')
    river_network_downsegid: str = Field(default='DSLINKNO', alias='RIVER_NETWORK_SHP_DOWNSEGID')
    river_network_slope: str = Field(default='Slope', alias='RIVER_NETWORK_SHP_SLOPE')

    pour_point_path: str = Field(default='default', alias='POUR_POINT_SHP_PATH')
    pour_point_name: str = Field(default='default', alias='POUR_POINT_SHP_NAME')

    # Common paths
    forcing_path: str = Field(default='default', alias='FORCING_PATH')
    observations_path: str = Field(default='default', alias='OBSERVATIONS_PATH')
    simulations_path: str = Field(default='default', alias='SIMULATIONS_PATH')
    intersect_soil_path: str = Field(default='default', alias='INTERSECT_SOIL_PATH')
    intersect_soil_name: str = Field(default='default', alias='INTERSECT_SOIL_NAME')
    intersect_routing_path: str = Field(default='default', alias='INTERSECT_ROUTING_PATH')
    intersect_routing_name: str = Field(default='default', alias='INTERSECT_ROUTING_NAME')
    intersect_dem_path: str = Field(default='default', alias='INTERSECT_DEM_PATH')
    intersect_dem_name: str = Field(default='default', alias='INTERSECT_DEM_NAME')
    intersect_land_path: str = Field(default='default', alias='INTERSECT_LAND_PATH')
    intersect_land_name: str = Field(default='default', alias='INTERSECT_LAND_NAME')
    output_basins_path: str = Field(default='default', alias='OUTPUT_BASINS_PATH')
    output_rivers_path: str = Field(default='default', alias='OUTPUT_RIVERS_PATH')
    dem_path: str = Field(default='default', alias='DEM_PATH')
    dem_name: str = Field(default='default', alias='DEM_NAME')
    source_geofabric_basins_path: str = Field(default='default', alias='SOURCE_GEOFABRIC_BASINS_PATH')
    source_geofabric_rivers_path: str = Field(default='default', alias='SOURCE_GEOFABRIC_RIVERS_PATH')
    taudem_dir: str = Field(default='default', alias='TAUDEM_DIR')
    output_dir: str = Field(default='default', alias='OUTPUT_DIR')
    catchment_plot_dir: str = Field(default='default', alias='CATCHMENT_PLOT_DIR')
    soil_class_path: str = Field(default='default', alias='SOIL_CLASS_PATH')
    soil_class_name: str = Field(default='default', alias='SOIL_CLASS_NAME')
    land_class_path: str = Field(default='default', alias='LAND_CLASS_PATH')
    radiation_path: str = Field(default='default', alias='RADIATION_PATH')

    # Tool paths
    datatool_path: str = Field(default='default', alias='DATATOOL_PATH')
    gistool_path: str = Field(default='default', alias='GISTOOL_PATH')
    easymore_client: str = Field(default='easymore cli', alias='EASYMORE_CLIENT')
    datatool_dataset_root: str = Field(default='/path/to/meteorological-data/', alias='DATATOOL_DATASET_ROOT')
    gistool_dataset_root: str = Field(default='/path/to/geospatial-data/', alias='GISTOOL_DATASET_ROOT')
    tool_cache: str = Field(default='/path/to/cache/dir', alias='TOOL_CACHE')
    easymore_cache: str = Field(default='/path/to/cache/dir', alias='EASYMORE_CACHE')
    easymore_job_conf: str = Field(default='/path/to/esmr/job_config', alias='EASYMORE_JOB_CONF')
    cluster_json: str = Field(default='/path/to/cluster.json', alias='CLUSTER_JSON')
    gistool_lib_path: str = Field(default='/path/to/r-env/', alias='GISTOOL_LIB_PATH')
