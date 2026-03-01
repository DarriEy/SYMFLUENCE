# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Worldfile Generator

Handles generation of RHESSys worldfiles, which describe the hierarchical
spatial structure: world > basin > hillslope > zone > patch > stratum.

Extracted from RHESSysPreProcessor for modularity.
"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RHESSysWorldfileGenerator:
    """
    Generates RHESSys worldfiles and associated header files.

    The worldfile is the primary spatial descriptor for RHESSys, defining
    the hierarchy of spatial objects and their initial state variables.

    Args:
        preprocessor: Parent RHESSysPreProcessor instance providing access
            to configuration, paths, and helper methods.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor

    @staticmethod
    def _parse_optional_float(value):
        """Parse optional float config values, allowing None/empty/0 to disable."""
        if value is None:
            return None
        if isinstance(value, str):
            val = value.strip().lower()
            if val in {"none", "null", ""}:
                return None
            try:
                value = float(val)
            except ValueError:
                return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    def generate_worldfile(self):
        """
        Generate the RHESSys worldfile from domain data.

        The worldfile describes the hierarchical structure:
        world > basin > hillslope > zone > patch > canopy_stratum

        If the domain has multiple HRUs, generates a distributed worldfile
        with one patch per HRU for proper TOPMODEL behavior.
        """
        logger.info("Generating worldfile...")

        world_file = self.pp.worldfiles_dir / f"{self.pp.domain_name}.world"
        start_date, end_date = self.pp._get_simulation_dates()

        # Check if this is a distributed domain with multiple HRUs
        try:
            catchment_path = self.pp.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                num_hrus = len(gdf)

                if num_hrus > 1:
                    logger.info(f"Detected {num_hrus} HRUs - generating distributed worldfile")
                    self.generate_distributed_worldfile(gdf, world_file)
                    return
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not check for distributed domain: {e}")

        # Fall back to single-patch worldfile
        logger.info("Generating single-patch worldfile")

        # Get domain properties from shapefile
        try:
            catchment_path = self.pp.get_catchment_path()

            # If catchment path doesn't exist, search other experiment dirs
            if not catchment_path.exists():
                catchment_dir = self.pp.project_dir / 'shapefiles' / 'catchment'
                if catchment_dir.exists():
                    for shp_file in catchment_dir.rglob('*.shp'):
                        if self.pp.domain_name in shp_file.name:
                            catchment_path = shp_file
                            logger.info(f"Found catchment shapefile in alternate location: {shp_file}")
                            break

            if not catchment_path.exists():
                raise FileNotFoundError(
                    f"Catchment shapefile not found at {catchment_path} or any alternate location. "
                    f"Searched: {self.pp.project_dir / 'shapefiles' / 'catchment'}. "
                    f"Cannot determine basin area - this would cause incorrect unit conversions. "
                    f"Run geospatial preprocessing first or verify shapefile paths."
                )

            gdf = gpd.read_file(catchment_path)

            # Compute centroid in projected CRS to avoid geographic centroid warnings
            utm_crs = self.pp._get_utm_crs_from_bounds(gdf)
            lon, lat = self.pp._get_centroid_lon_lat(gdf, utm_crs)

            # Project to UTM for accurate area calculation
            gdf_proj = (gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf).to_crs(utm_crs)
            area_m2 = gdf_proj.geometry.area.sum()

            if area_m2 <= 0:
                raise ValueError(
                    f"Catchment area computed as {area_m2} m² from {catchment_path}. "
                    f"Area must be positive. Check shapefile geometry."
                )

            logger.info(f"Catchment area: {area_m2:.0f} m² ({area_m2/1e6:.2f} km²)")

            # Try to get elevation stats
            elev_col = getattr(self.pp, 'catchment_elev_col', 'elev_mean')
            slope_col = getattr(self.pp, 'catchment_slope_col', 'slope')
            elev = float(gdf[elev_col].mean()) if elev_col in gdf.columns else 1500.0
            slope_raw = float(gdf[slope_col].mean()) if slope_col in gdf.columns else 10.0
            try:
                slope_units = str(self.pp.config.paths.catchment_shp_slope_units).lower()
            except (AttributeError, TypeError):
                slope_units = 'degrees'
            if slope_units in {'deg', 'degree', 'degrees'}:
                slope = np.tan(np.radians(slope_raw))
            elif slope_units in {'rad', 'radian', 'radians'}:
                slope = np.tan(slope_raw)
            elif slope_units in {'ratio', 'fraction', 'tan'}:
                slope = slope_raw
            else:
                logger.warning(
                    f"Unknown CATCHMENT_SHP_SLOPE_UNITS='{slope_units}'. "
                    "Falling back to degrees."
                )
                slope = np.tan(np.radians(slope_raw))
            slope = max(0.01, min(slope, 2.0))
        except (FileNotFoundError, ValueError):
            raise  # Re-raise area-related errors - these must not be silently ignored
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise RuntimeError(
                f"Failed to read catchment properties from shapefile: {e}. "
                f"Cannot determine basin area for RHESSys worldfile generation."
            ) from e

        # IDs
        world_id = 1
        basin_id = 1
        hillslope_id = 1
        zone_id = 1
        patch_id = 1
        stratum_id = 1

        # Calculate topographic wetness index (lna) for TOPMODEL-based runoff
        # lna = ln(a/tan(beta)) where a = contributing area per unit contour
        # Approximation: a ~ sqrt(area), so lna = 0.5*ln(area) - ln(tan(slope))
        # slope is already tan(beta) from above, so use it directly
        tan_slope = max(slope, 0.001)  # Minimum to avoid log issues
        # Optional lna controls (can be disabled via config)
        lna_area_cap = self._parse_optional_float(self.pp._get_config_value(
            lambda: self.pp.config.model.rhessys.lna_area_cap_m2 if self.pp.config.model and self.pp.config.model.rhessys else None,
            default=1e5
        ))
        lna_min = self._parse_optional_float(self.pp._get_config_value(
            lambda: self.pp.config.model.rhessys.lna_min if self.pp.config.model and self.pp.config.model.rhessys else None,
            default=5.0
        ))
        lna_max = self._parse_optional_float(self.pp._get_config_value(
            lambda: self.pp.config.model.rhessys.lna_max if self.pp.config.model and self.pp.config.model.rhessys else None,
            default=15.0
        ))

        # For very large basins, optionally cap the contributing area effect
        effective_area = min(area_m2, lna_area_cap) if lna_area_cap else area_m2
        lna = 0.5 * np.log(effective_area) - np.log(tan_slope)
        if lna_min is not None:
            lna = max(lna_min, lna)
        if lna_max is not None:
            lna = min(lna_max, lna)

        logger.info(
            f"Calculated lna (TWI) = {lna:.2f} for area={area_m2/1e6:.1f} km², slope={slope:.3f} "
            f"(lna_area_cap={lna_area_cap}, lna_min={lna_min}, lna_max={lna_max})"
        )

        # Use configurable initial conditions for faster spinup
        init_sat_deficit = self.pp.init_sat_deficit
        init_gw_storage = self.pp.init_gw_storage
        init_rz_storage = self.pp.init_rz_storage
        init_unsat_storage = self.pp.init_unsat_storage

        # Build worldfile content (simplified single-patch world)
        # When using a separate .hdr header file, the worldfile should NOT include
        # num_world_base_stations or dates - those come from header and command line
        # Format follows RHESSys v5.x conventions with short parameter names
        content = f"""{world_id}    world_ID
1    num_basins
   {basin_id}    basin_ID
   {lon:.8f}    x
   {lat:.8f}    y
   {elev:.8f}    z
   1    basin_parm_ID
   {lat:.8f}    latitude
   0    basin_n_basestations
   1    num_hillslopes
      {hillslope_id}    hillslope_ID
      {lon:.8f}    x
      {lat:.8f}    y
      {elev:.8f}    z
      1    hill_parm_ID
      {init_gw_storage:.8f}    gw.storage
      0.00000000    gw.NO3
      0    hillslope_n_basestations
      1    num_zones
         {zone_id}    zone_ID
         {lon:.8f}    x
         {lat:.8f}    y
         {elev:.8f}    z
         1    zone_parm_ID
         {area_m2:.8f}    area
         {slope:.8f}    slope
         180.00000000    aspect
         1.00000000    precip_lapse_rate
         0.00000000    e_horizon
         0.00000000    w_horizon
         1    zone_n_basestations
         1    zone_basestation_ID
         1    num_patches
            {patch_id}    patch_ID
            {lon:.8f}    x
            {lat:.8f}    y
            {elev:.8f}    z
            1    soil_parm_ID
            1    landuse_parm_ID
            {area_m2:.8f}    area
            {slope:.8f}    slope
            {lna:.8f}    lna
            1.00000000    Ksat_vertical
            0.00000000    mpar
            {init_rz_storage:.8f}    rz_storage
            {init_unsat_storage:.8f}    unsat_storage
            {init_sat_deficit:.8f}    sat_deficit
            0.00000000    snowpack.water_equivalent_depth
            0.00000000    snowpack.water_depth
            0.00000000    snowpack.T
            0.00000000    snowpack.surface_age
            0.00000000    snowpack.energy_deficit
            1.00000000    litter.cover_fraction
            0.00100000    litter.rain_stored
            0.03000000    litter_cs.litr1c
            0.00100000    litter_ns.litr1n
            0.20000000    litter_cs.litr2c
            0.80000000    litter_cs.litr3c
            0.70000000    litter_cs.litr4c
            0.05000000    soil_cs.soil1c
            0.00010000    soil_ns.sminn
            0.00200000    soil_ns.nitrate
            0.40000000    soil_cs.soil2c
            6.00000000    soil_cs.soil3c
            35.00000000    soil_cs.soil4c
            0    patch_n_basestations
            1    num_canopy_strata
               {stratum_id}    canopy_strata_ID
               1    veg_parm_ID
               0.70000000    cover_fraction
               0.30000000    gap_fraction
               2.00000000    rootzone.depth
               0.00000000    snow_stored
               0.00000000    cs.stem_density
               0.00200000    rain_stored
               1.00000000    cs.cpool
               0.44000000    cs.leafc
               0.05000000    cs.dead_leafc
               0.71000000    cs.live_stemc
               4.00000000    cs.dead_stemc
               0.22000000    cs.live_crootc
               1.20000000    cs.dead_crootc
               0.58000000    cs.frootc
               0.50000000    cs.cwdc
               0.10000000    ns.npool
               0.00980000    ns.leafn
               0.00100000    ns.dead_leafn
               0.00360000    ns.live_stemn
               0.02000000    ns.dead_stemn
               0.00110000    ns.live_crootn
               0.00600000    ns.dead_crootn
               0.00420000    ns.frootn
               0.00250000    ns.cwdn
               0.01000000    ns.retransn
               0.10000000    epv.prev_leafcalloc
               10.00000000    epv.height
               0    canopy_strata_n_basestations
"""

        # Add fire_parm_ID for WMFire support if enabled
        if self.pp.wmfire_enabled:
            content = content.replace(
                "            1    landuse_parm_ID\n            ",
                "            1    landuse_parm_ID\n            1    fire_parm_ID\n            "
            )

        world_file.write_text(content, encoding='utf-8')
        logger.info(f"Worldfile written: {world_file}")

        # Generate the header file with default file paths
        self.generate_world_header(world_file)

    def generate_distributed_worldfile(self, gdf: gpd.GeoDataFrame, world_file: Path):
        """
        Generate a distributed RHESSys worldfile with multiple patches (one per HRU).

        This enables proper TOPMODEL behavior with variable source areas by providing
        spatial variability in TWI/lna values across patches.

        Args:
            gdf: GeoDataFrame with HRU polygons and attributes
            world_file: Output path for worldfile
        """
        from shapely.validation import make_valid

        logger.info(f"Generating distributed worldfile with {len(gdf)} patches...")

        # Fix invalid geometries
        if not gdf.is_valid.all():
            invalid_count = (~gdf.is_valid).sum()
            logger.info(f"Fixing {invalid_count} invalid geometries in catchment")
            gdf = gdf.copy()
            gdf['geometry'] = gdf['geometry'].apply(
                lambda g: make_valid(g) if g is not None and not g.is_valid else g
            )

        # Load additional attributes if available
        attrs_file = self.pp.project_attributes_dir / f'{self.pp.domain_name}_attributes.csv'
        hru_attrs = {}
        if attrs_file.exists():
            try:
                attrs_df = pd.read_csv(attrs_file)
                for _, row in attrs_df.iterrows():
                    hru_id = int(row.get('hru_id', row.get('HRU_ID', 0)))
                    hru_attrs[hru_id] = {
                        'elev_mean': row.get('dem.mean', 1500.0),
                        'slope_mean': row.get('slope.mean', 10.0),
                        'aspect_mean': row.get('aspect.circmean', 180.0),
                        'porosity': row.get('soil.porosity', 0.45),
                        'ksat': row.get('soil.ksat', 1e-6),
                    }
                logger.info(f"Loaded attributes for {len(hru_attrs)} HRUs from {attrs_file}")
            except Exception as e:  # noqa: BLE001 — model execution resilience
                logger.warning(f"Could not load HRU attributes: {e}")

        # Project to UTM for accurate area calculation
        utm_crs = self.pp._get_utm_crs_from_bounds(gdf)
        gdf_proj = (gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf).to_crs(utm_crs)

        # Calculate total basin area
        total_area = gdf_proj.geometry.area.sum()
        logger.info(f"Total basin area: {total_area/1e6:.2f} km²")

        # Get basin centroid for header (in lon/lat)
        basin_lon, basin_lat = self.pp._get_centroid_lon_lat(gdf, utm_crs)

        # Get column names for HRU attributes
        hru_id_col = 'HRU_ID' if 'HRU_ID' in gdf.columns else 'hru_id'
        elev_col = 'elev_mean' if 'elev_mean' in gdf.columns else None

        # Sort HRUs by ID for consistent ordering
        gdf = gdf.sort_values(by=hru_id_col).reset_index(drop=True)
        gdf_proj = gdf_proj.sort_values(by=hru_id_col).reset_index(drop=True)

        num_patches = len(gdf)
        world_id = 1
        basin_id = 1
        hillslope_id = 1

        # Get mean elevation for basin
        if elev_col and elev_col in gdf.columns:
            basin_elev = gdf[elev_col].mean()
        else:
            basin_elev = 2000.0

        # Build worldfile content
        lines = []
        lines.append(f"{world_id}    world_ID")
        lines.append("1    num_basins")
        lines.append(f"   {basin_id}    basin_ID")
        lines.append(f"   {basin_lon:.8f}    x")
        lines.append(f"   {basin_lat:.8f}    y")
        lines.append(f"   {basin_elev:.8f}    z")
        lines.append("   1    basin_parm_ID")
        lines.append(f"   {basin_lat:.8f}    latitude")
        lines.append("   0    basin_n_basestations")
        lines.append("   1    num_hillslopes")
        lines.append(f"      {hillslope_id}    hillslope_ID")
        lines.append(f"      {basin_lon:.8f}    x")
        lines.append(f"      {basin_lat:.8f}    y")
        lines.append(f"      {basin_elev:.8f}    z")
        lines.append("      1    hill_parm_ID")
        lines.append(f"      {self.pp.init_gw_storage:.8f}    gw.storage")
        lines.append("      0.00000000    gw.NO3")
        lines.append("      0    hillslope_n_basestations")
        lines.append(f"      {num_patches}    num_zones")

        # Optional lna controls (can be disabled via config)
        lna_area_cap = self._parse_optional_float(self.pp._get_config_value(
            lambda: self.pp.config.model.rhessys.lna_area_cap_m2 if self.pp.config.model and self.pp.config.model.rhessys else None,
            default=1e5
        ))
        lna_min = self._parse_optional_float(self.pp._get_config_value(
            lambda: self.pp.config.model.rhessys.lna_min if self.pp.config.model and self.pp.config.model.rhessys else None,
            default=5.0
        ))
        lna_max = self._parse_optional_float(self.pp._get_config_value(
            lambda: self.pp.config.model.rhessys.lna_max if self.pp.config.model and self.pp.config.model.rhessys else None,
            default=15.0
        ))

        # Generate each zone/patch/stratum (one per HRU)
        for idx, (_, row) in enumerate(gdf.iterrows()):
            hru_id = int(row[hru_id_col])
            zone_id = hru_id
            patch_id = hru_id
            stratum_id = hru_id

            # Get HRU geometry properties
            proj_row = gdf_proj.iloc[idx]
            area_m2 = proj_row.geometry.area

            centroid_proj = proj_row.geometry.centroid
            centroid_ll = gpd.GeoSeries([centroid_proj], crs=utm_crs).to_crs("EPSG:4326").iloc[0]
            lon, lat = float(centroid_ll.x), float(centroid_ll.y)

            # Get elevation from shapefile or attributes
            if elev_col and elev_col in gdf.columns:
                elev = float(row[elev_col])
            elif hru_id in hru_attrs:
                elev = hru_attrs[hru_id].get('elev_mean', 2000.0)
            else:
                elev = 2000.0

            # Get slope from attributes (convert from degrees if needed)
            if hru_id in hru_attrs:
                slope_deg = hru_attrs[hru_id].get('slope_mean', 10.0)
                # The slope.mean in attributes appears to be in degrees already
                # but some values are ~90 which suggests radians - check and convert
                if slope_deg > 60:  # Likely in radians or error
                    slope_deg = min(slope_deg, 45.0)  # Cap at reasonable value
            else:
                slope_deg = 10.0

            # Convert slope to fraction for RHESSys (tan of angle)
            slope_frac = np.tan(np.radians(slope_deg))
            slope_frac = max(0.01, min(slope_frac, 2.0))  # Reasonable bounds

            # Get aspect from attributes
            if hru_id in hru_attrs:
                aspect = hru_attrs[hru_id].get('aspect_mean', 180.0)
            else:
                aspect = 180.0

            # Calculate TWI/lna for this patch
            # lna = ln(a/tan(beta)) where a = contributing area, beta = slope
            # For each HRU, use its own area as contributing area approximation
            tan_slope = max(np.tan(np.radians(slope_deg)), 0.01)
            # Use sqrt of area as contour length approximation
            effective_area = min(area_m2, lna_area_cap) if lna_area_cap else area_m2
            contrib_area = np.sqrt(effective_area)
            lna = np.log(contrib_area / tan_slope)
            if lna_min is not None:
                lna = max(lna_min, lna)
            if lna_max is not None:
                lna = min(lna_max, lna)

            logger.debug(f"HRU {hru_id}: area={area_m2/1e6:.2f}km², elev={elev:.0f}m, slope={slope_deg:.1f}°, lna={lna:.2f}")

            # Zone block
            lines.append(f"         {zone_id}    zone_ID")
            lines.append(f"         {lon:.8f}    x")
            lines.append(f"         {lat:.8f}    y")
            lines.append(f"         {elev:.8f}    z")
            lines.append("         1    zone_parm_ID")
            lines.append(f"         {area_m2:.8f}    area")
            lines.append(f"         {slope_frac:.8f}    slope")
            lines.append(f"         {aspect:.8f}    aspect")
            lines.append("         1.00000000    precip_lapse_rate")
            # Horizons must be 0 for basin daylength to be assigned to zones
            lines.append("         0.00000000    e_horizon")
            lines.append("         0.00000000    w_horizon")
            lines.append("         1    zone_n_basestations")
            lines.append("         1    zone_basestation_ID")
            lines.append("         1    num_patches")

            # Patch block
            lines.append(f"            {patch_id}    patch_ID")
            lines.append(f"            {lon:.8f}    x")
            lines.append(f"            {lat:.8f}    y")
            lines.append(f"            {elev:.8f}    z")
            lines.append("            1    soil_parm_ID")
            lines.append("            1    landuse_parm_ID")
            # Add fire_parm_ID for WMFire support
            if self.pp.wmfire_enabled:
                lines.append("            1    fire_parm_ID")
            lines.append(f"            {area_m2:.8f}    area")
            lines.append(f"            {slope_frac:.8f}    slope")
            lines.append(f"            {lna:.8f}    lna")
            lines.append("            1.00000000    Ksat_vertical")
            lines.append("            0.00000000    mpar")
            lines.append(f"            {self.pp.init_rz_storage:.8f}    rz_storage")
            lines.append(f"            {self.pp.init_unsat_storage:.8f}    unsat_storage")
            lines.append(f"            {self.pp.init_sat_deficit:.8f}    sat_deficit")
            # Initialize snowpack to zero - model will build snowpack from precipitation
            lines.append("            0.00000000    snowpack.water_equivalent_depth")
            lines.append("            0.00000000    snowpack.water_depth")
            lines.append("            0.00000000    snowpack.T")
            lines.append("            0.00000000    snowpack.surface_age")
            lines.append("            0.00000000    snowpack.energy_deficit")
            lines.append("            1.00000000    litter.cover_fraction")
            lines.append("            0.00100000    litter.rain_stored")
            lines.append("            0.03000000    litter_cs.litr1c")
            lines.append("            0.00100000    litter_ns.litr1n")
            lines.append("            0.20000000    litter_cs.litr2c")
            lines.append("            0.80000000    litter_cs.litr3c")
            lines.append("            0.70000000    litter_cs.litr4c")
            lines.append("            0.05000000    soil_cs.soil1c")
            lines.append("            0.00010000    soil_ns.sminn")
            lines.append("            0.00200000    soil_ns.nitrate")
            lines.append("            0.40000000    soil_cs.soil2c")
            lines.append("            6.00000000    soil_cs.soil3c")
            lines.append("            35.00000000    soil_cs.soil4c")
            lines.append("            0    patch_n_basestations")
            lines.append("            1    num_canopy_strata")

            # Stratum block
            lines.append(f"               {stratum_id}    canopy_strata_ID")
            lines.append("               1    veg_parm_ID")
            lines.append("               0.70000000    cover_fraction")
            lines.append("               0.30000000    gap_fraction")
            lines.append("               2.00000000    rootzone.depth")
            lines.append("               0.00000000    snow_stored")
            # stem_density must be 0 to prevent horizon recalculation in update_phenology.c
            # which would override zone daylength to 0 and disable photosynthesis/transpiration
            lines.append("               0.00000000    cs.stem_density")
            lines.append("               0.00200000    rain_stored")
            lines.append("               1.00000000    cs.cpool")
            lines.append("               0.44000000    cs.leafc")
            lines.append("               0.05000000    cs.dead_leafc")
            lines.append("               0.71000000    cs.live_stemc")
            lines.append("               4.00000000    cs.dead_stemc")
            lines.append("               0.22000000    cs.live_crootc")
            lines.append("               1.20000000    cs.dead_crootc")
            lines.append("               0.58000000    cs.frootc")
            lines.append("               0.50000000    cs.cwdc")
            lines.append("               0.10000000    ns.npool")
            lines.append("               0.00980000    ns.leafn")
            lines.append("               0.00100000    ns.dead_leafn")
            lines.append("               0.00360000    ns.live_stemn")
            lines.append("               0.02000000    ns.dead_stemn")
            lines.append("               0.00110000    ns.live_crootn")
            lines.append("               0.00600000    ns.dead_crootn")
            lines.append("               0.00420000    ns.frootn")
            lines.append("               0.00250000    ns.cwdn")
            lines.append("               0.01000000    ns.retransn")
            lines.append("               0.10000000    epv.prev_leafcalloc")
            lines.append("               10.00000000    epv.height")
            lines.append("               0    canopy_strata_n_basestations")

        content = '\n'.join(lines)
        world_file.write_text(content, encoding='utf-8')
        logger.info(f"Distributed worldfile written: {world_file} ({num_patches} patches)")

        # Generate the header file
        self.generate_world_header(world_file)

        # Store patch info for flow table generation
        self.pp._distributed_patches = []
        for idx, (_, row) in enumerate(gdf.iterrows()):
            hru_id = int(row[hru_id_col])
            proj_row = gdf_proj.iloc[idx]
            centroid_proj = proj_row.geometry.centroid
            centroid_ll = gpd.GeoSeries([centroid_proj], crs=utm_crs).to_crs("EPSG:4326").iloc[0]

            if elev_col and elev_col in gdf.columns:
                elev = float(row[elev_col])
            elif hru_id in hru_attrs:
                elev = hru_attrs[hru_id].get('elev_mean', 2000.0)
            else:
                elev = 2000.0

            self.pp._distributed_patches.append({
                'patch_id': hru_id,
                'zone_id': hru_id,
                'hill_id': hillslope_id,
                'lon': float(centroid_ll.x),
                'lat': float(centroid_ll.y),
                'elev': elev,
                'area': proj_row.geometry.area,
            })

    def generate_world_header(self, world_file: Path):
        """
        Generate the RHESSys world header file (.hdr) with default file paths.

        The header file lists all default parameter files that are referenced
        by ID in the worldfile.
        """
        header_file = world_file.with_suffix('.world.hdr')

        # Build header content with default file paths
        content = f"""1    num_basin_default_files
{self.pp.defs_dir / 'basin.def'}
1    num_hillslope_default_files
{self.pp.defs_dir / 'hillslope.def'}
1    num_zone_default_files
{self.pp.defs_dir / 'zone.def'}
1    num_soil_default_files
{self.pp.defs_dir / 'soil.def'}
1    num_landuse_default_files
{self.pp.defs_dir / 'landuse.def'}
1    num_stratum_default_files
{self.pp.defs_dir / 'stratum.def'}
"""

        # Add fire defaults if WMFire is enabled
        if self.pp.wmfire_enabled:
            content += f"""1    num_fire_default_files
{self.pp.defs_dir / 'fire.def'}
"""

        # Add base stations
        content += f"""1    num_base_stations
{self.pp.climate_dir / f'{self.pp.domain_name}_base'}
"""
        header_file.write_text(content, encoding='utf-8')

        # Also create the landuse.def file if it doesn't exist
        landuse_def = self.pp.defs_dir / 'landuse.def'
        if not landuse_def.exists():
            landuse_content = """1    landuse_default_ID
1.0    irrigation_fraction
1.0    septic_water_load
1.0    septic_NO3_load
1.0    fertilizer_NO3_load
1.0    fertilizer_NH4_load
1    fertilizer_day_of_year
0.0    grazing_Closs
0.0    impervious_fraction
"""
            landuse_def.write_text(landuse_content, encoding='utf-8')

        logger.info(f"World header written: {header_file}")
