"""
Shapefile access mixin for SYMFLUENCE modules.

Provides standardized shapefile column name access from configuration.
"""

from .config import ConfigMixin


class ShapefileAccessMixin(ConfigMixin):
    """
    Mixin providing standardized shapefile column name access.

    Provides properties for accessing shapefile column names from the typed
    config, with sensible defaults for common geofabric conventions.
    """

    # =========================================================================
    # Catchment Shapefile Columns
    # =========================================================================

    @property
    def catchment_name_col(self) -> str:
        """Name/ID column in catchment shapefile from config.paths.catchment_name."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_name,
            default='HRU_ID'
        )

    @property
    def catchment_hruid_col(self) -> str:
        """HRU ID column in catchment shapefile from config.paths.catchment_hruid."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_hruid,
            default='HRU_ID'
        )

    @property
    def catchment_gruid_col(self) -> str:
        """GRU ID column in catchment shapefile from config.paths.catchment_gruid."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_gruid,
            default='GRU_ID'
        )

    @property
    def catchment_area_col(self) -> str:
        """Area column in catchment shapefile from config.paths.catchment_area."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_area,
            default='HRU_area'
        )

    @property
    def catchment_lat_col(self) -> str:
        """Latitude column in catchment shapefile from config.paths.catchment_lat."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_lat,
            default='center_lat'
        )

    @property
    def catchment_lon_col(self) -> str:
        """Longitude column in catchment shapefile from config.paths.catchment_lon."""
        return self._get_config_value(
            lambda: self.config.paths.catchment_lon,
            default='center_lon'
        )

    # =========================================================================
    # River Network Shapefile Columns
    # =========================================================================

    @property
    def river_network_name_col(self) -> str:
        """Name column in river network shapefile from config.paths.river_network_name."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_name,
            default='LINKNO'
        )

    @property
    def river_segid_col(self) -> str:
        """Segment ID column in river network from config.paths.river_network_segid."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_segid,
            default='LINKNO'
        )

    @property
    def river_downsegid_col(self) -> str:
        """Downstream segment ID column from config.paths.river_network_downsegid."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_downsegid,
            default='DSLINKNO'
        )

    @property
    def river_length_col(self) -> str:
        """Length column in river network from config.paths.river_network_length."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_length,
            default='Length'
        )

    @property
    def river_slope_col(self) -> str:
        """Slope column in river network from config.paths.river_network_slope."""
        return self._get_config_value(
            lambda: self.config.paths.river_network_slope,
            default='Slope'
        )

    # =========================================================================
    # River Basin Shapefile Columns
    # =========================================================================

    @property
    def basin_name_col(self) -> str:
        """Name column in river basins shapefile from config.paths.river_basins_name."""
        return self._get_config_value(
            lambda: self.config.paths.river_basins_name,
            default='GRU_ID'
        )

    @property
    def basin_gruid_col(self) -> str:
        """GRU ID column in river basins from config.paths.river_basin_rm_gruid."""
        return self._get_config_value(
            lambda: self.config.paths.river_basin_rm_gruid,
            default='GRU_ID'
        )

    @property
    def basin_hru_to_seg_col(self) -> str:
        """HRU to segment mapping column from config.paths.river_basin_hru_to_seg."""
        return self._get_config_value(
            lambda: self.config.paths.river_basin_hru_to_seg,
            default='gru_to_seg'
        )

    @property
    def basin_area_col(self) -> str:
        """Area column in river basins from config.paths.river_basin_area."""
        return self._get_config_value(
            lambda: self.config.paths.river_basin_area,
            default='GRU_area'
        )
