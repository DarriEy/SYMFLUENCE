"""
RHESSys Model Preprocessor

Handles preparation of RHESSys model inputs including:
- Climate forcing files (daily precipitation, temperature)
- Worldfile generation (domain structure)
- TEC file generation (temporal event control)
- Flow table generation (routing)
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry
from symfluence.models.mixins import ObservationLoaderMixin
from symfluence.models.rhessys.climate_generator import RHESSysClimateGenerator

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("RHESSys")
class RHESSysPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):
    """
    Prepares inputs for a RHESSys model run.

    RHESSys requires:
    - Climate station files with daily forcing (P, Tmax, Tmin)
    - Worldfile describing domain hierarchy (world > basin > hillslope > zone > patch > stratum)
    - TEC file with simulation dates and output events
    - Optional: Flow table for hillslope routing, fire grids for WMFire
    """

    def __init__(self, config, logger):
        """
        Initialize the RHESSys preprocessor.

        Sets up RHESSys-specific directory structure and checks for optional
        WMFire (wildfire spread) module configuration.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                RHESSys settings, domain paths, and simulation parameters.
            logger: Logger instance for status messages and debugging.

        Note:
            Creates input directories for worldfiles, tecfiles, climate data,
            routing tables, and definition files under {project_dir}/RHESSys_input/.
        """
        super().__init__(config, logger)
        # Check for WMFire support (handles both wmfire and legacy vmfire config names)
        self.wmfire_enabled = self._check_wmfire_enabled()

        # Setup RHESSys-specific directories
        self.rhessys_input_dir = self.project_dir / "RHESSys_input"
        self.worldfiles_dir = self.rhessys_input_dir / "worldfiles"
        self.tecfiles_dir = self.rhessys_input_dir / "tecfiles"
        self.climate_dir = self.rhessys_input_dir / "clim"
        self.routing_dir = self.rhessys_input_dir / "routing"
        self.defs_dir = self.rhessys_input_dir / "defs"

        # Note: experiment_id and forcing_dataset are inherited as properties from ShapefileAccessMixin

    def _check_wmfire_enabled(self) -> bool:
        """Check if WMFire fire spread is enabled (supports both new and legacy config names)."""
        try:
            # Try new naming first
            if hasattr(self.config.model.rhessys, 'use_wmfire'):
                return self.config.model.rhessys.use_wmfire
            # Fall back to legacy vmfire naming
            if hasattr(self.config.model.rhessys, 'use_vmfire'):
                return self.config.model.rhessys.use_vmfire
        except AttributeError:
            pass
        return False

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "RHESSys"

    def run_preprocessing(self) -> bool:
        """
        Run the complete RHESSys preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting RHESSys preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Generate climate files from forcing data
            self._generate_climate_files()

            # Generate worldfile
            self._generate_worldfile()

            # Generate TEC file
            self._generate_tec_file()

            # Generate flow table
            self._generate_flow_table()

            # Generate default files
            self._generate_default_files()

            # Setup WMFire inputs if enabled
            if self.wmfire_enabled:
                self._setup_wmfire_inputs()

            logger.info("RHESSys preprocessing complete.")
            return True
        except Exception as e:
            logger.error(f"RHESSys preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self):
        """Create RHESSys input directory structure."""
        dirs = [
            self.rhessys_input_dir,
            self.worldfiles_dir,
            self.tecfiles_dir,
            self.climate_dir,
            self.routing_dir,
            self.defs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created RHESSys input directories at {self.rhessys_input_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """
        Get simulation start and end dates from configuration.

        Returns:
            Tuple[datetime, datetime]: Start and end dates for the simulation
                period, parsed from EXPERIMENT_TIME_START and EXPERIMENT_TIME_END.
        """
        start_str = self._get_config_value(
            lambda: self.config.domain.time_start
        )
        end_str = self._get_config_value(
            lambda: self.config.domain.time_end
        )

        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def _load_forcing_data(self) -> xr.Dataset:
        """
        Load basin-averaged forcing data from NetCDF files.

        Searches for forcing data in multiple locations (basin_averaged_data,
        merged_path, SUMMA_input, raw_data) and combines using xarray.

        Returns:
            xr.Dataset: Forcing dataset subset to simulation period.

        Raises:
            FileNotFoundError: If no forcing files found in any search path.
        """
        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

        if not forcing_files:
            logger.warning(f"No forcing files found in {self.forcing_basin_path}")
            # Try merged_path (common for SUMMA-preprocessed domains)
            merged_path = self.project_dir / 'forcing' / 'merged_path'
            if merged_path.exists():
                forcing_files = list(merged_path.glob("*.nc"))
                if forcing_files:
                    logger.info(f"Found {len(forcing_files)} forcing files in merged_path")

        if not forcing_files:
            # Try SUMMA_input (contains processed forcing from SUMMA preprocessing)
            summa_input = self.project_dir / 'forcing' / 'SUMMA_input'
            if summa_input.exists():
                forcing_files = list(summa_input.glob("*.nc"))
                if forcing_files:
                    logger.info(f"Found {len(forcing_files)} forcing files in SUMMA_input")

        if not forcing_files:
            # Try raw data
            forcing_files = list(self.forcing_raw_path.glob("*.nc"))

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in {self.forcing_basin_path} or {self.forcing_raw_path}")

        logger.info(f"Loading forcing from {len(forcing_files)} files")

        try:
            # Try standard combination by coordinates (works for time-split files)
            ds = xr.open_mfdataset(forcing_files, combine='by_coords')
        except ValueError as e:
            logger.warning(f"Failed to open with combine='by_coords': {e}. Retrying with combine='nested'...")
            try:
                # Try nested concatenation (works if files are strictly ordered)
                ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time')
            except (FileNotFoundError, OSError, ValueError, KeyError):
                # Fallback: Open individually and merge (works for variable-split files)
                logger.warning("Failed to concat. Attempting to merge variable-split files...")
                datasets = [xr.open_dataset(f) for f in forcing_files]
                ds = xr.merge(datasets)

        # Ensure time is a coordinate
        if 'time' in ds.data_vars and 'time' not in ds.coords:
            ds = ds.set_coords('time')

        # Subset to simulation period
        ds = self.subset_to_simulation_time(ds, "Forcing")

        return ds

    def _generate_climate_files(self):
        """
        Generate RHESSys-compatible climate input files from forcing data.

        Delegates to RHESSysClimateGenerator for the actual file generation.
        """
        start_date, end_date = self._get_simulation_dates()

        generator = RHESSysClimateGenerator(
            config=self.config_dict,
            project_dir=self.project_dir,
            domain_name=self.domain_name,
            logger=self.logger
        )

        catchment_path = self.get_catchment_path() if hasattr(self, 'get_catchment_path') else None
        generator.generate_climate_files(start_date, end_date, catchment_path)

    def _find_variable(self, ds: xr.Dataset, candidates: list) -> Optional[str]:
        """Find first matching variable name in dataset."""
        for var in candidates:
            if var in ds.data_vars:
                return var
        return None

    def _write_base_station_file(self, base_name: str, station_id: int, start_date: pd.Timestamp):
        """Write RHESSys base station file."""
        base_file = self.climate_dir / f"{base_name}"

        # Get centroid coordinates from basin shapefile
        try:
            basin_path = self.get_catchment_path()
            if basin_path.exists():
                gdf = gpd.read_file(basin_path)
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y
                # Get elevation (rough estimate)
                elev = float(gdf.get('elev_mean', [1000])[0]) if 'elev_mean' in gdf.columns else 1000.0
            else:
                lon, lat, elev = -115.0, 51.0, 1500.0
        except (FileNotFoundError, KeyError, IndexError, ValueError):
            lon, lat, elev = -115.0, 51.0, 1500.0

        # Full path to climate file prefix for RHESSys to find the daily files
        climate_prefix = self.climate_dir / base_name

        # Build list of non-critical daily sequences based on available files
        # These are additional climate variables beyond the required tmax, tmin, rain
        non_critical_sequences = []

        # Check which optional climate files exist and add them
        if (self.climate_dir / f"{base_name}.wind").exists():
            non_critical_sequences.append("wind")
        if (self.climate_dir / f"{base_name}.relative_humidity").exists():
            non_critical_sequences.append("relative_humidity")
        if (self.climate_dir / f"{base_name}.Kdown_direct").exists():
            non_critical_sequences.append("Kdown_direct")
        if (self.climate_dir / f"{base_name}.Ldown").exists():
            non_critical_sequences.append("Ldown")
        if (self.climate_dir / f"{base_name}.tavg").exists():
            non_critical_sequences.append("tavg")

        # Build sequence section
        num_sequences = len(non_critical_sequences)
        sequence_lines = "\n".join(non_critical_sequences) if non_critical_sequences else ""

        # RHESSys base station format: value<tab>label
        # Must include all temporal prefixes with number_non_critical_sequences
        content = f"""{station_id}\tbase_station_id
{lon:.4f}\tx_coordinate
{lat:.4f}\ty_coordinate
{elev:.1f}\tz_coordinate
3.5\teffective_lai
2.0\tscreen_height
none\tannual_climate_prefix
0\tnumber_non_critical_annual_sequences
none\tmonthly_climate_prefix
0\tnumber_non_critical_monthly_sequences
{climate_prefix}\tdaily_climate_prefix
{num_sequences}\tnumber_non_critical_daily_sequences
{sequence_lines}
none\thourly_climate_prefix
0\tnumber_non_critical_hourly_sequences
"""
        base_file.write_text(content)
        logger.info(f"Base station file written: {base_file} ({num_sequences} non-critical sequences: {', '.join(non_critical_sequences)})")

    def _write_climate_file(self, filename: str, dates: pd.DatetimeIndex, values: np.ndarray):
        """
        Write a single RHESSys climate file.

        RHESSys climate file format:
        - Line 1: start date (year month day hour)
        - Lines 2+: one value per line (one per day for daily data)
        """
        filepath = self.climate_dir / filename

        with open(filepath, 'w') as f:
            # First line: start date only
            start_date = dates[0]
            f.write(f"{start_date.year} {start_date.month} {start_date.day} 1\n")

            # Subsequent lines: just the values (one per day)
            for value in values:
                f.write(f"{value:.4f}\n")

        logger.debug(f"Climate file written: {filepath}")

    def _create_synthetic_climate(self):
        """Create synthetic climate data for testing when real forcing is unavailable."""
        start_date, end_date = self._get_simulation_dates()
        dates = pd.date_range(start_date, end_date, freq='D')

        # Simple synthetic data
        precip = np.random.exponential(2, len(dates))  # mm/day
        temp = 10 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates))
        tmax = temp + 5 + np.random.normal(0, 1, len(dates))
        tmin = temp - 5 + np.random.normal(0, 1, len(dates))

        base_name = f"{self.domain_name}_base"
        self._write_base_station_file(base_name, 1, dates[0])
        self._write_climate_file(f"{base_name}.rain", dates, precip)
        self._write_climate_file(f"{base_name}.tmax", dates, tmax)
        self._write_climate_file(f"{base_name}.tmin", dates, tmin)
        self._write_climate_file(f"{base_name}.tavg", dates, temp)

        logger.info("Synthetic climate files created")

    def _generate_worldfile(self):
        """
        Generate the RHESSys worldfile from domain data.

        The worldfile describes the hierarchical structure:
        world > basin > hillslope > zone > patch > canopy_stratum

        If the domain has multiple HRUs, generates a distributed worldfile
        with one patch per HRU for proper TOPMODEL behavior.
        """
        logger.info("Generating worldfile...")

        world_file = self.worldfiles_dir / f"{self.domain_name}.world"
        start_date, end_date = self._get_simulation_dates()

        # Check if this is a distributed domain with multiple HRUs
        try:
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                num_hrus = len(gdf)

                if num_hrus > 1:
                    logger.info(f"Detected {num_hrus} HRUs - generating distributed worldfile")
                    self._generate_distributed_worldfile(gdf, world_file)
                    return
        except Exception as e:
            logger.warning(f"Could not check for distributed domain: {e}")

        # Fall back to single-patch worldfile
        logger.info("Generating single-patch worldfile")

        # Get domain properties from shapefile
        try:
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)

                # Get centroid in original CRS (geographic)
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y

                # Project to UTM for accurate area calculation
                # Estimate UTM zone from longitude
                utm_zone = int((lon + 180) / 6) + 1
                hemisphere = 'north' if lat >= 0 else 'south'
                utm_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
                gdf_proj = gdf.to_crs(utm_crs)
                area_m2 = gdf_proj.geometry.area.sum()

                logger.info(f"Catchment area: {area_m2:.0f} m² ({area_m2/1e6:.2f} km²)")

                # Try to get elevation stats
                elev = float(gdf.get('elev_mean', [1500])[0]) if 'elev_mean' in gdf.columns else 1500.0
                slope_deg = float(gdf.get('slope_mean', [10.0])[0]) if 'slope_mean' in gdf.columns else 10.0
                # Convert slope from degrees to fraction (tan) for RHESSys
                slope = np.tan(np.radians(slope_deg))
                slope = max(0.01, min(slope, 2.0))
            else:
                area_m2 = 1e8  # 100 km2
                lon, lat = -115.0, 51.0
                elev, slope = 1500.0, 0.176  # ~10 degrees
        except Exception as e:
            logger.warning(f"Could not read catchment properties: {e}")
            area_m2 = 1e8
            lon, lat = -115.0, 51.0
            elev, slope = 1500.0, 0.1

        # IDs
        world_id = 1
        basin_id = 1
        hillslope_id = 1
        zone_id = 1
        patch_id = 1
        stratum_id = 1

        # Calculate topographic wetness index (lna) for TOPMODEL-based runoff
        # lna = ln(a/tan(beta)) where a = contributing area per unit contour
        # Approximation: a ≈ sqrt(area), so lna = 0.5*ln(area) - ln(tan(slope))
        # Ensure slope is reasonable (minimum 0.01 = 1% to avoid division issues)
        slope_rad = max(slope, 0.01)  # Minimum slope
        tan_slope = np.tan(np.radians(slope_rad * 100))  # Convert from fraction to degrees
        if tan_slope < 0.001:
            tan_slope = 0.001  # Minimum to avoid log issues
        # For very large basins, cap the contributing area effect
        effective_area = min(area_m2, 1e8)  # Cap at 100 km²
        lna = 0.5 * np.log(effective_area) - np.log(tan_slope)
        lna = max(5.0, min(lna, 15.0))  # Constrain to reasonable range (5-15)

        logger.info(f"Calculated lna (TWI) = {lna:.2f} for area={area_m2/1e6:.1f} km², slope={slope:.3f}")

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
      0.10000000    gw.storage
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
         0.20000000    e_horizon
         0.20000000    w_horizon
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
            0.10000000    rz_storage
            0.05000000    unsat_storage
            0.50000000    sat_deficit
            0.00000000    snowpack.water_equivalent_depth
            0.00000000    snowpack.water_depth
            -10.00000000    snowpack.T
            0.00000000    snowpack.surface_age
            500.00000000    snowpack.energy_deficit
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
               0.00000000    gap_fraction
               2.00000000    rootzone.depth
               0.00000000    snow_stored
               0.02000000    cs.stem_density
               0.00200000    rain_stored
               20.00000000    cs.cpool
               8.00000000    cs.leafc
               1.00000000    cs.dead_leafc
               5.00000000    cs.live_stemc
               10.00000000    cs.dead_stemc
               2.00000000    cs.live_crootc
               5.00000000    cs.dead_crootc
               1.00000000    cs.frootc
               0.50000000    cs.cwdc
               0.50000000    ns.npool
               0.01000000    ns.leafn
               0.03000000    ns.dead_leafn
               0.10000000    ns.live_stemn
               0.20000000    ns.dead_stemn
               0.05000000    ns.live_crootn
               0.10000000    ns.dead_crootn
               0.02000000    ns.frootn
               0.01000000    ns.cwdn
               0.01000000    ns.retransn
               0.10000000    epv.prev_leafcalloc
               0    canopy_strata_n_basestations
"""

        # Add fire_parm_ID for WMFire support if enabled
        if self.wmfire_enabled:
            # Insert fire_parm_ID line after landuse_parm_ID
            content = content.replace(
                "            1    landuse_parm_ID\n            ",
                "            1    landuse_parm_ID\n            1    fire_parm_ID\n            "
            )

        world_file.write_text(content)
        logger.info(f"Worldfile written: {world_file}")

        # Generate the header file with default file paths
        self._generate_world_header(world_file)

    def _generate_distributed_worldfile(self, gdf: gpd.GeoDataFrame, world_file: Path):
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
        attrs_file = self.project_dir / 'attributes' / f'{self.domain_name}_attributes.csv'
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
            except Exception as e:
                logger.warning(f"Could not load HRU attributes: {e}")

        # Project to UTM for accurate area calculation
        sample_centroid = gdf.geometry.centroid.iloc[0]
        lon_sample = sample_centroid.x
        lat_sample = sample_centroid.y
        utm_zone = int((lon_sample + 180) / 6) + 1
        hemisphere = 'north' if lat_sample >= 0 else 'south'
        utm_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
        gdf_proj = gdf.to_crs(utm_crs)

        # Calculate total basin area
        total_area = gdf_proj.geometry.area.sum()
        logger.info(f"Total basin area: {total_area/1e6:.2f} km²")

        # Get basin centroid for header
        basin_centroid = gdf.unary_union.centroid
        basin_lon, basin_lat = basin_centroid.x, basin_centroid.y

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
        lines.append("      0.10000000    gw.storage")
        lines.append("      0.00000000    gw.NO3")
        lines.append("      0    hillslope_n_basestations")
        lines.append(f"      {num_patches}    num_zones")

        # Generate each zone/patch/stratum (one per HRU)
        for idx, (_, row) in enumerate(gdf.iterrows()):
            hru_id = int(row[hru_id_col])
            zone_id = hru_id
            patch_id = hru_id
            stratum_id = hru_id

            # Get HRU geometry properties
            proj_row = gdf_proj.iloc[idx]
            area_m2 = proj_row.geometry.area

            centroid = row.geometry.centroid
            lon, lat = centroid.x, centroid.y

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
            # lna = ln(a/tan(β)) where a = contributing area, β = slope
            # For each HRU, use its own area as contributing area approximation
            tan_slope = max(np.tan(np.radians(slope_deg)), 0.01)
            # Use sqrt of area as contour length approximation
            contrib_area = np.sqrt(area_m2)
            lna = np.log(contrib_area / tan_slope)
            lna = max(5.0, min(lna, 15.0))  # Constrain to reasonable range

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
            lines.append("         0.20000000    e_horizon")
            lines.append("         0.20000000    w_horizon")
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
            if self.wmfire_enabled:
                lines.append("            1    fire_parm_ID")
            lines.append(f"            {area_m2:.8f}    area")
            lines.append(f"            {slope_frac:.8f}    slope")
            lines.append(f"            {lna:.8f}    lna")
            lines.append("            1.00000000    Ksat_vertical")
            lines.append("            0.00000000    mpar")
            lines.append("            0.10000000    rz_storage")
            lines.append("            0.05000000    unsat_storage")
            lines.append("            0.50000000    sat_deficit")
            lines.append("            0.00000000    snowpack.water_equivalent_depth")
            lines.append("            0.00000000    snowpack.water_depth")
            lines.append("            -10.00000000    snowpack.T")
            lines.append("            0.00000000    snowpack.surface_age")
            lines.append("            500.00000000    snowpack.energy_deficit")
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
            lines.append("               0.00000000    gap_fraction")
            lines.append("               2.00000000    rootzone.depth")
            lines.append("               0.00000000    snow_stored")
            lines.append("               0.02000000    cs.stem_density")
            lines.append("               0.00200000    rain_stored")
            lines.append("               20.00000000    cs.cpool")
            lines.append("               8.00000000    cs.leafc")
            lines.append("               1.00000000    cs.dead_leafc")
            lines.append("               5.00000000    cs.live_stemc")
            lines.append("               10.00000000    cs.dead_stemc")
            lines.append("               2.00000000    cs.live_crootc")
            lines.append("               5.00000000    cs.dead_crootc")
            lines.append("               1.00000000    cs.frootc")
            lines.append("               0.50000000    cs.cwdc")
            lines.append("               0.50000000    ns.npool")
            lines.append("               0.01000000    ns.leafn")
            lines.append("               0.03000000    ns.dead_leafn")
            lines.append("               0.10000000    ns.live_stemn")
            lines.append("               0.20000000    ns.dead_stemn")
            lines.append("               0.05000000    ns.live_crootn")
            lines.append("               0.10000000    ns.dead_crootn")
            lines.append("               0.02000000    ns.frootn")
            lines.append("               0.01000000    ns.cwdn")
            lines.append("               0.01000000    ns.retransn")
            lines.append("               0.10000000    epv.prev_leafcalloc")
            lines.append("               0    canopy_strata_n_basestations")

        content = '\n'.join(lines)
        world_file.write_text(content)
        logger.info(f"Distributed worldfile written: {world_file} ({num_patches} patches)")

        # Generate the header file
        self._generate_world_header(world_file)

        # Store patch info for flow table generation
        self._distributed_patches = []
        for idx, (_, row) in enumerate(gdf.iterrows()):
            hru_id = int(row[hru_id_col])
            proj_row = gdf_proj.iloc[idx]
            centroid = row.geometry.centroid

            if elev_col and elev_col in gdf.columns:
                elev = float(row[elev_col])
            elif hru_id in hru_attrs:
                elev = hru_attrs[hru_id].get('elev_mean', 2000.0)
            else:
                elev = 2000.0

            self._distributed_patches.append({
                'patch_id': hru_id,
                'zone_id': hru_id,
                'hill_id': hillslope_id,
                'lon': centroid.x,
                'lat': centroid.y,
                'elev': elev,
                'area': proj_row.geometry.area,
            })

    def _generate_world_header(self, world_file: Path):
        """
        Generate the RHESSys world header file (.hdr) with default file paths.

        The header file lists all default parameter files that are referenced
        by ID in the worldfile.
        """
        header_file = world_file.with_suffix('.world.hdr')

        # Build header content with default file paths
        content = f"""1    num_basin_default_files
{self.defs_dir / 'basin.def'}
1    num_hillslope_default_files
{self.defs_dir / 'hillslope.def'}
1    num_zone_default_files
{self.defs_dir / 'zone.def'}
1    num_soil_default_files
{self.defs_dir / 'soil.def'}
1    num_landuse_default_files
{self.defs_dir / 'landuse.def'}
1    num_stratum_default_files
{self.defs_dir / 'stratum.def'}
"""

        # Add fire defaults if WMFire is enabled
        if self.wmfire_enabled:
            content += f"""1    num_fire_default_files
{self.defs_dir / 'fire.def'}
"""

        # Add base stations
        content += f"""1    num_base_stations
{self.climate_dir / f'{self.domain_name}_base'}
"""
        header_file.write_text(content)

        # Also create the landuse.def file if it doesn't exist
        landuse_def = self.defs_dir / 'landuse.def'
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
            landuse_def.write_text(landuse_content)

        logger.info(f"World header written: {header_file}")

    def _generate_tec_file(self):
        """
        Generate the RHESSys Temporal Event Control (TEC) file.

        The TEC file specifies simulation events and output timing.
        When WMFire is enabled, adds fire ignition event.
        """
        logger.info("Generating TEC file...")

        tec_file = self.tecfiles_dir / f"{self.domain_name}.tec"
        start_date, end_date = self._get_simulation_dates()

        # Build TEC content
        # Note: print_daily_on at hour 1, print_daily_growth_on at hour 2
        # Don't use print_daily_off - let output continue until simulation ends
        lines = [
            f"{start_date.year} {start_date.month} {start_date.day} 1 print_daily_on",
            f"{start_date.year} {start_date.month} {start_date.day} 2 print_daily_growth_on",
        ]

        # Add fire ignition event if WMFire is enabled
        if self.wmfire_enabled:
            fire_date = self._get_fire_ignition_date(start_date, end_date)
            if fire_date:
                # fire_grid_on triggers WMFire fire spread at the specified date
                lines.append(f"{fire_date.year} {fire_date.month} {fire_date.day} 1 fire_grid_on")
                logger.info(f"Added fire ignition event for {fire_date.strftime('%Y-%m-%d')}")

        content = "\n".join(lines) + "\n"
        tec_file.write_text(content)
        logger.info(f"TEC file written: {tec_file}")

    def _get_fire_ignition_date(self, start_date, end_date):
        """
        Get the fire ignition date from config or calculate default.

        Returns:
            datetime: The ignition date, or None if fire should not be triggered.
        """
        from datetime import datetime, timedelta

        # Try to get ignition date from WMFire config
        try:
            wmfire_config = self.config.model.rhessys.wmfire
            if wmfire_config and hasattr(wmfire_config, 'ignition_date') and wmfire_config.ignition_date:
                # Parse date string (format: YYYY-MM-DD)
                ignition_date = datetime.strptime(wmfire_config.ignition_date, '%Y-%m-%d')
                logger.info(f"Using configured ignition date: {ignition_date.strftime('%Y-%m-%d')}")
                return ignition_date
        except (AttributeError, ValueError) as e:
            logger.debug(f"Could not get ignition date from config: {e}")

        # Default: use spinup end date + 1 day, or 30 days after start if no spinup
        try:
            spinup_period = self.config.experiment.spinup_period
            if spinup_period:
                # spinup_period is typically (start, end) tuple or string
                if isinstance(spinup_period, (list, tuple)) and len(spinup_period) >= 2:
                    spinup_end_str = spinup_period[1]
                    if isinstance(spinup_end_str, str):
                        spinup_end = datetime.strptime(spinup_end_str.strip(), '%Y-%m-%d')
                        ignition_date = spinup_end + timedelta(days=1)
                        logger.info(f"Using post-spinup ignition date: {ignition_date.strftime('%Y-%m-%d')}")
                        return ignition_date
        except (AttributeError, ValueError) as e:
            logger.debug(f"Could not parse spinup period: {e}")

        # Fallback: 30 days after simulation start
        ignition_date = start_date + timedelta(days=30)
        if ignition_date > end_date:
            ignition_date = start_date + timedelta(days=7)  # Use 7 days if simulation is short
        logger.info(f"Using default ignition date: {ignition_date.strftime('%Y-%m-%d')}")
        return ignition_date

    def _generate_flow_table(self):
        """
        Generate the RHESSys flow table for routing.

        For distributed models, creates routing based on elevation gradient.
        For lumped models, single patch drains to stream outlet.

        Flow table format (per construct_routing_topology.c):
        <num_patches>
        <patch_ID zone_ID hill_ID x y z area area drainage_type gamma num_neighbors>
        [<neighbor_patch_ID neighbor_zone_ID neighbor_hill_ID gamma> for each neighbor]

        drainage_type: 0=LAND, 1=STREAM, 2=ROAD
        """
        logger.info("Generating flow table...")

        flow_file = self.routing_dir / f"{self.domain_name}.routing"

        # Check if we have distributed patches from worldfile generation
        if hasattr(self, '_distributed_patches') and len(self._distributed_patches) > 1:
            self._generate_distributed_flow_table(flow_file)
            return

        # Fall back to single-patch flow table
        try:
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y
                elev = float(gdf.get('elev_mean', [1500])[0]) if 'elev_mean' in gdf.columns else 1500.0

                utm_zone = int((lon + 180) / 6) + 1
                hemisphere = 'north' if lat >= 0 else 'south'
                utm_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
                gdf_proj = gdf.to_crs(utm_crs)
                area_m2 = gdf_proj.geometry.area.sum()
            else:
                lon, lat = -115.0, 51.0
                elev = 1500.0
                area_m2 = 1e8
        except Exception as e:
            logger.warning(f"Could not read catchment for flow table: {e}")
            lon, lat = -115.0, 51.0
            elev = 1500.0
            area_m2 = 1e8

        patch_id = 1
        zone_id = 1
        hill_id = 1
        num_hillslopes = 1
        num_patches = 1

        content = f"""{num_hillslopes}
{hill_id}
{num_patches}
{patch_id} {zone_id} {hill_id} {lon:.8f} {lat:.8f} {elev:.8f} {area_m2:.8f} {area_m2:.8f} 1 0.0 0
"""

        flow_file.write_text(content)
        logger.info(f"Flow table written: {flow_file}")

    def _generate_distributed_flow_table(self, flow_file: Path):
        """
        Generate flow table for distributed domain with multiple patches.

        Routes water based on elevation gradient - each patch drains to the
        lowest elevation neighbor, with the lowest overall patch being the outlet.

        Args:
            flow_file: Output path for flow table
        """
        patches = self._distributed_patches
        num_patches = len(patches)
        logger.info(f"Generating distributed flow table for {num_patches} patches")

        # Sort patches by elevation (lowest = outlet)
        sorted_patches = sorted(patches, key=lambda p: p['elev'])

        # Find outlet (lowest elevation patch)
        outlet_patch = sorted_patches[0]
        outlet_id = outlet_patch['patch_id']

        # Build adjacency based on elevation - each patch drains to outlet
        # For simple approach: all patches drain directly to outlet
        # More sophisticated: chain drainage based on elevation
        lines = []
        lines.append("1")  # num_hillslopes
        lines.append("1")  # hillslope_ID
        lines.append(str(num_patches))

        for patch in patches:
            pid = patch['patch_id']
            zid = patch['zone_id']
            hid = patch['hill_id']
            lon = patch['lon']
            lat = patch['lat']
            elev = patch['elev']
            area = patch['area']

            if pid == outlet_id:
                # Outlet patch: drainage_type=1 (STREAM), no neighbors
                lines.append(
                    f"{pid} {zid} {hid} {lon:.8f} {lat:.8f} {elev:.8f} "
                    f"{area:.8f} {area:.8f} 1 0.0 0"
                )
            else:
                # Find downstream neighbor (next lower elevation patch)
                # Simple approach: drain directly to outlet
                downstream_id = outlet_id
                downstream_zone = outlet_patch['zone_id']
                downstream_hill = outlet_patch['hill_id']

                # More sophisticated: find nearest lower elevation patch
                for p in sorted_patches:
                    if p['elev'] < elev and p['patch_id'] != pid:
                        downstream_id = p['patch_id']
                        downstream_zone = p['zone_id']
                        downstream_hill = p['hill_id']
                        break

                # drainage_type=0 (LAND), gamma=1.0 (100% to neighbor), 1 neighbor
                lines.append(
                    f"{pid} {zid} {hid} {lon:.8f} {lat:.8f} {elev:.8f} "
                    f"{area:.8f} {area:.8f} 0 1.0 1"
                )
                # Neighbor line: patch_id zone_id hill_id gamma
                lines.append(f"{downstream_id} {downstream_zone} {downstream_hill} 1.0")

        content = '\n'.join(lines)
        flow_file.write_text(content)
        logger.info(f"Distributed flow table written: {flow_file} ({num_patches} patches, outlet={outlet_id})")

    def _generate_default_files(self):
        """Generate RHESSys default parameter files."""
        logger.info("Generating default files...")

        # Basin defaults
        # Note: sat_to_gw_coeff set for moderate groundwater recharge
        # gw_loss_coeff set low to allow groundwater storage and baseflow
        basin_def = self.defs_dir / "basin.def"
        basin_content = """1    basin_default_ID
0.2    psi_air_entry
0.02    pore_size_index
0.00005    sat_to_gw_coeff
25.0    gw_loss_coeff
0.2    n_routing_power
1.0    m_pai
"""
        basin_def.write_text(basin_content)

        # Hillslope defaults
        hillslope_def = self.defs_dir / "hillslope.def"
        hillslope_content = """1    hillslope_default_ID
0.2    gw_loss_coeff
"""
        hillslope_def.write_text(hillslope_content)

        # Zone defaults
        zone_def = self.defs_dir / "zone.def"
        zone_content = """1    zone_default_ID
0.0    atm_trans_lapse_rate
-0.006    dewpoint_lapse_rate
-0.0065    lapse_rate_tmax
-0.0065    lapse_rate_tmin
0.0    max_effective_lai
2.0    max_snow_temp
-2.0    min_rain_temp
0.7    ndep_NO3
-0.004    wet_lapse_rate
0.0    lapse_rate_precip_default
"""
        zone_def.write_text(zone_content)

        # Soil (patch) defaults
        # Note: soil_depth increased to 2.5m for sufficient storage
        # m set to 2.0 for moderate Ksat decay with depth
        # Ksat_0 set to allow reasonable infiltration
        soil_def = self.defs_dir / "soil.def"
        soil_content = """1    patch_default_ID
-6.0    psi_air_entry
0.12    pore_size_index
0.45    porosity_0
0.45    porosity_decay
0.0001    Ksat_0
200.0    Ksat_0_v
2.0    m
2.0    m_z
2.0    N_decay
2.5    soil_depth
0.3    active_zone_z
1500.0    albedo
1500.0    maximum_snow_energy_deficit
0.5    snow_melt_Tcoef
3.0    snow_water_capacity
0.5    wilting_point
0.15    theta_mean_std_p1
0.0    theta_mean_std_p2
0.0    gl_c
0.00012    gsurf_slope
0.00001    gsurf_intercept
"""
        soil_def.write_text(soil_content)

        # Vegetation (stratum) defaults
        veg_def = self.defs_dir / "stratum.def"
        veg_content = """1    stratum_default_ID
1    epc.veg_type
0    epc.phenology_type
static    epc.phenology_flag
4.0    epc.max_lai
0.5    epc.proj_sla
0.5    epc.shade_sla
0.5    epc.proj_sla_shade_sla_ratio
0.03    epc.lai_stomatal_fraction
1.0    epc.flnr
1.0    epc.ppfd_coef
0.5    epc.topt
0.5    epc.tcoef
0.5    epc.tmax
0.0012    epc.psi_open
-0.0065    epc.psi_close
0.5    epc.vpd_open
4.0    epc.vpd_close
0.05    epc.gl_smax
0.00006    epc.gl_c
0.0    epc.specific_rain_capacity
0.0    epc.specific_snow_capacity
0.5    epc.wind_attenuation_coef
1.0    epc.max_height
0.5    epc.max_root_depth
0.5    epc.root_growth_direction
0.25    epc.root_distrib_parm
0.0    epc.resprout_leaf_carbon
0.7    epc.min_percent_leafg
0.0    epc.dickenson_pa
0.0    epc.waring_pa
0.0    epc.chen_pa
waring    epc.allocation_flag
0.0    epc.storage_transfer_prop
"""
        veg_def.write_text(veg_content)

        logger.info(f"Default files written to {self.defs_dir}")

    def _setup_wmfire_inputs(self):
        """
        Setup WMFire fire spread inputs for RHESSys.

        Creates the fire grid files required for the -firespread flag:
        - patch_grid.txt: Grid of patch IDs for fire tracking
        - dem_grid.txt: DEM values for fire spread calculations

        Uses FireGridManager for georeferenced grid generation with
        proper CRS and transform metadata. Supports multiple resolutions
        (30, 60, 90m) and optional GeoTIFF output for visualization.
        """
        logger.info("WMFire is enabled. Setting up fire spread inputs...")

        fire_dir = self.rhessys_input_dir / "fire"
        fire_dir.mkdir(parents=True, exist_ok=True)

        patch_grid_file = fire_dir / "patch_grid.txt"
        dem_grid_file = fire_dir / "dem_grid.txt"

        # Get WMFire configuration
        wmfire_config = None
        try:
            if hasattr(self.config.model.rhessys, 'wmfire'):
                wmfire_config = self.config.model.rhessys.wmfire
        except AttributeError:
            pass

        # Try to use FireGridManager for georeferenced grid generation
        use_geospatial = False
        try:
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                catchment_gdf = gpd.read_file(catchment_path)
                if len(catchment_gdf) > 0:
                    use_geospatial = True
        except Exception as e:
            logger.warning(f"Could not load catchment for fire grids: {e}")

        if use_geospatial:
            self._setup_geospatial_fire_grids(
                catchment_gdf, patch_grid_file, dem_grid_file, wmfire_config
            )
        elif hasattr(self, '_distributed_patches') and len(self._distributed_patches) > 1:
            # Fallback: use distributed patches from worldfile generation
            self._setup_distributed_fire_grids(patch_grid_file, dem_grid_file)
        else:
            # Fallback: lumped single-patch grid
            self._setup_lumped_fire_grids(patch_grid_file, dem_grid_file)

        # Generate fire.def with correct grid dimensions
        self._generate_fire_defaults()

        logger.info(f"WMFire input files created in {fire_dir}")

    def _setup_geospatial_fire_grids(
        self,
        catchment_gdf: gpd.GeoDataFrame,
        patch_grid_file: Path,
        dem_grid_file: Path,
        wmfire_config
    ):
        """
        Setup georeferenced fire grids using FireGridManager.

        Creates properly georeferenced grids with correct CRS, transform,
        and resolution. Optionally writes GeoTIFF outputs for visualization.
        Also handles ignition point processing.

        Args:
            catchment_gdf: GeoDataFrame with catchment/HRU polygons
            patch_grid_file: Output path for patch grid text file
            dem_grid_file: Output path for DEM grid text file
            wmfire_config: WMFireConfig object with settings
        """
        from symfluence.models.wmfire import FireGridManager

        # Create grid manager
        grid_manager = FireGridManager(self.config, self.logger)
        resolution = grid_manager.resolution
        logger.info(f"Creating fire grids at {resolution}m resolution")

        # Look for DEM if available
        dem_path = None
        dem_candidates = [
            self.project_dir / 'attributes' / 'elevation' / 'dem.tif',
            self.project_dir / 'attributes' / 'dem' / 'dem.tif',
            self.project_dir / 'domain' / 'dem.tif',
        ]
        for candidate in dem_candidates:
            if candidate.exists():
                dem_path = candidate
                logger.info(f"Found DEM: {dem_path}")
                break

        # Generate georeferenced grids
        try:
            patch_grid, dem_grid = grid_manager.create_fire_grid(
                catchment_gdf, dem_path
            )

            # Write text format for RHESSys
            patch_grid_file.write_text(patch_grid.to_text())
            dem_grid_file.write_text(dem_grid.to_text())

            # Store grid dimensions for fire.def generation
            self._fire_grid_nrows = patch_grid.nrows
            self._fire_grid_ncols = patch_grid.ncols
            self._fire_grid = patch_grid  # Store for fire.def generation

            logger.info(f"Georeferenced fire grids created: "
                       f"{patch_grid.nrows}x{patch_grid.ncols} @ {resolution}m")

            # Process ignition point
            self._process_ignition_point(patch_grid, wmfire_config)

            # Write GeoTIFF outputs if requested
            write_geotiff = True
            if wmfire_config and hasattr(wmfire_config, 'write_geotiff'):
                write_geotiff = wmfire_config.write_geotiff

            if write_geotiff:
                fire_dir = patch_grid_file.parent
                try:
                    patch_grid.to_geotiff(fire_dir / "patch_grid.tif")
                    dem_grid.to_geotiff(fire_dir / "dem_grid.tif")
                    logger.info("GeoTIFF outputs written for visualization")
                except Exception as e:
                    logger.warning(f"Could not write GeoTIFF outputs: {e}")

        except Exception as e:
            logger.warning(f"Geospatial grid generation failed: {e}, falling back to simple grid")
            import traceback
            logger.debug(traceback.format_exc())
            # Fall back to simple grid generation
            if hasattr(self, '_distributed_patches') and len(self._distributed_patches) > 1:
                self._setup_distributed_fire_grids(patch_grid_file, dem_grid_file)
            else:
                self._setup_lumped_fire_grids(patch_grid_file, dem_grid_file)

    def _process_ignition_point(self, patch_grid, wmfire_config):
        """
        Process ignition point from configuration.

        Loads ignition point from shapefile or coordinates, converts to
        grid indices, and optionally writes ignition shapefile.

        Args:
            patch_grid: FireGrid object with grid metadata
            wmfire_config: WMFireConfig object with ignition settings
        """
        from symfluence.models.wmfire import IgnitionManager

        ignition_mgr = IgnitionManager(self.config, self.logger)
        ignition = ignition_mgr.get_ignition_point()

        if ignition is None:
            logger.info("No ignition point specified, using random ignition (-1, -1)")
            self._ignition_row = -1
            self._ignition_col = -1
            return

        logger.info(f"Processing ignition point: {ignition.name} "
                   f"({ignition.latitude:.4f}, {ignition.longitude:.4f})")

        # Convert to grid indices
        row, col = ignition_mgr.convert_to_grid_indices(
            ignition,
            patch_grid.transform,
            patch_grid.crs,
            patch_grid.nrows,
            patch_grid.ncols
        )

        self._ignition_row = row
        self._ignition_col = col

        # Write ignition shapefile if coordinates were from config (not already a shapefile)
        if ignition.source == 'config':
            ignition_dir = self.domain_path / 'shapefiles' / 'ignitions'
            ignition_mgr.write_ignition_shapefile(
                ignition,
                ignition_dir,
                filename=f"{ignition.name}.shp"
            )

        logger.info(f"Ignition grid indices: row={row}, col={col}")

    def _setup_distributed_fire_grids(self, patch_grid_file: Path, dem_grid_file: Path):
        """
        Setup fire grids for distributed multi-patch domains.

        Creates a grid representing the spatial arrangement of patches.
        For elevation-based discretization, arranges patches in a column
        from lowest to highest elevation.
        """
        patches = self._distributed_patches
        num_patches = len(patches)

        # Sort patches by elevation (lowest at bottom/south)
        patches_sorted = sorted(patches, key=lambda p: p['elev'])

        # For simplicity, create a 3-column grid with patches arranged by elevation
        # This gives each patch 3 cells for more realistic fire spread
        ncols = 3
        nrows = num_patches

        logger.info(f"Creating distributed fire grid: {nrows}x{ncols} ({num_patches} patches)")

        # Build patch grid (each row is one patch)
        # RHESSys expects NO header line and tab-separated values
        patch_lines = []
        dem_lines = []

        for patch_info in patches_sorted:
            patch_id = patch_info['patch_id']
            elev = patch_info['elev']
            # Each patch gets a full row (3 cells), tab-separated
            patch_lines.append(f"{patch_id}\t{patch_id}\t{patch_id}")
            dem_lines.append(f"{elev:.1f}\t{elev:.1f}\t{elev:.1f}")

        patch_grid_file.write_text('\n'.join(patch_lines) + '\n')
        dem_grid_file.write_text('\n'.join(dem_lines) + '\n')

        # Store grid dimensions for fire.def generation
        self._fire_grid_nrows = nrows
        self._fire_grid_ncols = ncols

        logger.info(f"Distributed fire grids created with {num_patches} patches")
        for i, p in enumerate(patches_sorted):
            logger.debug(f"  Row {i+1}: patch_id={p['patch_id']}, elev={p['elev']:.0f}m")

    def _setup_lumped_fire_grids(self, patch_grid_file: Path, dem_grid_file: Path):
        """
        Setup fire grids for lumped single-patch domains.
        """
        # Get elevation if available
        try:
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                elev = float(gdf.get('elev_mean', [1500])[0]) if 'elev_mean' in gdf.columns else 1500.0
            else:
                elev = 1500.0
        except (FileNotFoundError, KeyError, IndexError, ValueError):
            elev = 1500.0

        # Simple 3x3 grid for single patch
        # RHESSys expects NO header line and tab-separated values
        patch_content = """1\t1\t1
1\t1\t1
1\t1\t1
"""
        dem_content = f"""{elev:.1f}\t{elev:.1f}\t{elev:.1f}
{elev:.1f}\t{elev:.1f}\t{elev:.1f}
{elev:.1f}\t{elev:.1f}\t{elev:.1f}
"""

        patch_grid_file.write_text(patch_content)
        dem_grid_file.write_text(dem_content)

        # Store grid dimensions for fire.def generation
        self._fire_grid_nrows = 3
        self._fire_grid_ncols = 3

        logger.info("Lumped fire grids created (3x3, single patch)")

    def _generate_fire_defaults(self):
        """
        Generate fire.def file with grid dimensions for WMFire.

        The fire.def file specifies fire spread parameters including
        n_rows and n_cols which must match the fire grid dimensions.
        Uses FireDefGenerator for dynamic parameter generation based
        on WMFire configuration.
        """
        fire_def_file = self.defs_dir / "fire.def"

        # Get grid dimensions
        nrows = getattr(self, '_fire_grid_nrows', 3)
        ncols = getattr(self, '_fire_grid_ncols', 3)

        # Get ignition indices (default -1 for random)
        ignition_row = getattr(self, '_ignition_row', -1)
        ignition_col = getattr(self, '_ignition_col', -1)

        # Try to use FireDefGenerator for enhanced generation
        try:
            from symfluence.models.wmfire import FireDefGenerator

            fire_def_gen = FireDefGenerator(self.config, self.logger)

            # Use stored grid if available, otherwise create dummy
            if hasattr(self, '_fire_grid'):
                fire_def_gen.write_fire_def(
                    fire_def_file,
                    self._fire_grid,
                    ignition_row=ignition_row,
                    ignition_col=ignition_col
                )
            else:
                # Generate with just dimensions
                content = fire_def_gen.generate_default_fire_def(
                    n_rows=nrows,
                    n_cols=ncols,
                    output_path=fire_def_file
                )
            return

        except ImportError:
            logger.debug("FireDefGenerator not available, using fallback")
        except Exception as e:
            logger.warning(f"FireDefGenerator failed: {e}, using fallback")

        # Fallback: Generate fire.def manually
        # Fire defaults content based on RHESSys WMFire construct_fire_defaults.c
        content = f"""1    fire_parm_ID
30.0    ndays_average
3.9    load_k1
0.07    load_k2
0.91    slope_k1
1.0    slope_k2
3.8    moisture_k1
0.27    moisture_k2
0.87    winddir_k1
0.48    winddir_k2
3.8    moisture_ign_k1
0.27    moisture_ign_k2
1.0    windmax
-1    ignition_col
-1    ignition_row
10.0    ignition_tmin
0    fire_verbose
0    fire_write
0    fire_in_buffer
{nrows}    n_rows
{ncols}    n_cols
9    spread_calc_type
0.494    mean_log_wind
0.654    sd_log_wind
1.71    mean1_rvm
-1.91    mean2_rvm
2.37    kappa1_rvm
2.38    kappa2_rvm
0.411    p_rvm
1.0    ign_def_mod
0.8    veg_k1
10.0    veg_k2
1.0    mean_ign
0    ran_seed
0    calc_fire_effects
0    include_wui
0    fire_size_name
0.0    wind_shift
"""
        fire_def_file.write_text(content)
        logger.info(f"Fire defaults written: {fire_def_file} (grid: {nrows}x{ncols})")

    def preprocess(self, **kwargs):
        """
        Alternative entry point for preprocessing.
        """
        return self.run_preprocessing()
