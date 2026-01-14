"""
Shapefile access mixin for SYMFLUENCE modules.

Provides standardized shapefile column name access from configuration.
"""

from typing import Any

from .config import ConfigMixin


class ShapefileColumnProperty:
    """
    Descriptor for shapefile column properties.

    Provides lazy access to configuration values with defaults,
    reducing boilerplate for column name properties.
    """

    def __init__(self, config_accessor: str, default: str, doc: str):
        """
        Initialize the property descriptor.

        Args:
            config_accessor: Attribute path on config.paths (e.g., 'catchment_name')
            default: Default value if config value is not set
            doc: Documentation string for the property
        """
        self.config_accessor = config_accessor
        self.default = default
        self.__doc__ = doc

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type = None) -> str:
        if obj is None:
            return self  # type: ignore
        return obj._get_config_value(
            lambda: getattr(obj.config.paths, self.config_accessor),
            default=self.default
        )


def shapefile_column(config_attr: str, default: str, doc: str) -> ShapefileColumnProperty:
    """
    Factory function for creating shapefile column properties.

    Args:
        config_attr: Attribute name on config.paths
        default: Default column name
        doc: Property documentation

    Returns:
        ShapefileColumnProperty descriptor
    """
    return ShapefileColumnProperty(config_attr, default, doc)


class ShapefileAccessMixin(ConfigMixin):
    """
    Mixin providing standardized shapefile column name access.

    Provides properties for accessing shapefile column names from the typed
    config, with sensible defaults for common geofabric conventions.
    """

    # =========================================================================
    # Catchment Shapefile Columns
    # =========================================================================

    catchment_name_col = shapefile_column(
        'catchment_name', 'HRU_ID',
        "Name/ID column in catchment shapefile from config.paths.catchment_name."
    )

    catchment_hruid_col = shapefile_column(
        'catchment_hruid', 'HRU_ID',
        "HRU ID column in catchment shapefile from config.paths.catchment_hruid."
    )

    catchment_gruid_col = shapefile_column(
        'catchment_gruid', 'GRU_ID',
        "GRU ID column in catchment shapefile from config.paths.catchment_gruid."
    )

    catchment_area_col = shapefile_column(
        'catchment_area', 'HRU_area',
        "Area column in catchment shapefile from config.paths.catchment_area."
    )

    catchment_lat_col = shapefile_column(
        'catchment_lat', 'center_lat',
        "Latitude column in catchment shapefile from config.paths.catchment_lat."
    )

    catchment_lon_col = shapefile_column(
        'catchment_lon', 'center_lon',
        "Longitude column in catchment shapefile from config.paths.catchment_lon."
    )

    # =========================================================================
    # River Network Shapefile Columns
    # =========================================================================

    river_network_name_col = shapefile_column(
        'river_network_name', 'LINKNO',
        "Name column in river network shapefile from config.paths.river_network_name."
    )

    river_segid_col = shapefile_column(
        'river_network_segid', 'LINKNO',
        "Segment ID column in river network from config.paths.river_network_segid."
    )

    river_downsegid_col = shapefile_column(
        'river_network_downsegid', 'DSLINKNO',
        "Downstream segment ID column from config.paths.river_network_downsegid."
    )

    river_length_col = shapefile_column(
        'river_network_length', 'Length',
        "Length column in river network from config.paths.river_network_length."
    )

    river_slope_col = shapefile_column(
        'river_network_slope', 'Slope',
        "Slope column in river network from config.paths.river_network_slope."
    )

    # =========================================================================
    # River Basin Shapefile Columns
    # =========================================================================

    basin_name_col = shapefile_column(
        'river_basins_name', 'GRU_ID',
        "Name column in river basins shapefile from config.paths.river_basins_name."
    )

    basin_gruid_col = shapefile_column(
        'river_basin_rm_gruid', 'GRU_ID',
        "GRU ID column in river basins from config.paths.river_basin_rm_gruid."
    )

    basin_hru_to_seg_col = shapefile_column(
        'river_basin_hru_to_seg', 'gru_to_seg',
        "HRU to segment mapping column from config.paths.river_basin_hru_to_seg."
    )

    basin_area_col = shapefile_column(
        'river_basin_area', 'GRU_area',
        "Area column in river basins from config.paths.river_basin_area."
    )
