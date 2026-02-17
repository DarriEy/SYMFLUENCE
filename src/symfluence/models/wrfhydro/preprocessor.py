"""
WRF-Hydro Model Preprocessor

Handles preparation of WRF-Hydro model inputs including:
- HRLDAS namelist (namelist.hrldas) for Noah-MP LSM
- Hydro namelist (hydro.namelist) for routing configuration
- Geogrid/wrfinput files (domain definition)
- Fulldom routing grid (channel network)
- Forcing files (LDASIN NetCDF)
"""
import logging
from typing import Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry
from symfluence.models.mixins import ObservationLoaderMixin

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("WRFHYDRO")
class WRFHydroPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):  # type: ignore[misc]
    """
    Prepares inputs for a WRF-Hydro model run.

    WRF-Hydro requires:
    - namelist.hrldas: HRLDAS/Noah-MP control file
    - hydro.namelist: Hydrological routing control file
    - wrfinput_d01.nc / geo_em.d01.nc: Domain definition files
    - Fulldom_hires.nc: High-resolution routing grid
    - LDASIN forcing files: Meteorological forcing in NetCDF format
    - Restart files (optional): For warm-start simulations
    """

    def __init__(self, config, logger):
        """
        Initialize the WRF-Hydro preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Setup WRF-Hydro-specific directories
        self.wrfhydro_input_dir = self.project_dir / "WRFHydro_input"
        self.settings_dir = self.wrfhydro_input_dir / "settings"
        self.forcing_dir = self.wrfhydro_input_dir / "forcing"
        self.routing_dir = self.wrfhydro_input_dir / "routing"
        self.restart_dir = self.wrfhydro_input_dir / "restart"

        # Resolve spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.wrfhydro.spatial_mode,
            default=None,
            dict_key='WRFHYDRO_SPATIAL_MODE'
        )
        if configured_mode and configured_mode not in (None, 'auto', 'default'):
            self.spatial_mode = configured_mode
        else:
            self.spatial_mode = 'distributed'
        logger.info(f"WRF-Hydro spatial mode: {self.spatial_mode}")

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "WRFHYDRO"

    def run_preprocessing(self) -> bool:
        """
        Run the complete WRF-Hydro preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting WRF-Hydro preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Get simulation dates
            start_date, end_date = self._get_simulation_dates()

            # Generate forcing files
            self._generate_forcing_files(start_date, end_date)

            # Generate domain/geogrid files
            self._generate_domain_files()

            # Generate routing files
            self._generate_routing_files()

            # Generate HRLDAS namelist
            self._generate_hrldas_namelist(start_date, end_date)

            # Generate hydro namelist
            self._generate_hydro_namelist()

            logger.info("WRF-Hydro preprocessing completed successfully")
            return True

        except Exception as e:
            logger.error(f"WRF-Hydro preprocessing failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create WRF-Hydro input directory structure."""
        for d in [self.wrfhydro_input_dir, self.settings_dir,
                  self.forcing_dir, self.routing_dir, self.restart_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created WRF-Hydro directory structure in {self.wrfhydro_input_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from config."""
        start = self._get_config_value(
            lambda: self.config.domain.time_start,
            default='2000-01-01'
        )
        end = self._get_config_value(
            lambda: self.config.domain.time_end,
            default='2001-12-31'
        )
        if isinstance(start, str):
            start = pd.Timestamp(start).to_pydatetime()
        if isinstance(end, str):
            end = pd.Timestamp(end).to_pydatetime()
        return start, end

    def _generate_forcing_files(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate WRF-Hydro LDASIN forcing files from ERA5 data.

        WRF-Hydro expects hourly NetCDF forcing files named:
        YYYYMMDDHH.LDASIN_DOMAIN1

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
        """
        logger.info("Generating WRF-Hydro forcing files...")

        forcing_data = self._load_forcing_data()

        if forcing_data is not None:
            self._write_ldasin_files(forcing_data, start_date, end_date)
        else:
            logger.warning("No forcing data found, generating synthetic forcing")
            self._generate_synthetic_forcing(start_date, end_date)

    def _load_forcing_data(self):
        """
        Check that basin-averaged ERA5 forcing data exists.

        Uses the same basin-averaged forcing data as all other models in the
        ensemble (from self.forcing_basin_path inherited from BaseModelPreProcessor).

        Returns:
            True if forcing data available, None otherwise.
            Actual data is processed file-by-file in _write_ldasin_files.
        """
        forcing_path = self.forcing_basin_path
        if not forcing_path.exists():
            logger.warning(f"Forcing path does not exist: {forcing_path}")
            return None

        forcing_files = sorted(forcing_path.glob("*.nc"))
        if not forcing_files:
            logger.warning(f"No NetCDF files found in {forcing_path}")
            return None

        logger.info(f"Found ERA5 forcing in {forcing_path} ({len(forcing_files)} files)")
        return True  # Data processed file-by-file in _write_ldasin_files

    def _write_ldasin_files(self, forcing_data, start_date: datetime, end_date: datetime) -> None:
        """
        Write LDASIN NetCDF forcing files from ERA5 basin-averaged data.

        Maps ERA5 variables to WRF-Hydro LDASIN format:
          airtemp (K)      → T2D (K)
          spechum (kg/kg)  → Q2D (kg/kg)
          windspd (m/s)    → U2D (m/s), V2D = 0
          airpres (Pa)     → PSFC (Pa)
          pptrate (mm/s)   → RAINRATE (mm/s = kg/m²/s)
          SWRadAtm (W/m²)  → SWDOWN (W/m²)
          LWRadAtm (W/m²)  → LWDOWN (W/m²)

        Processes one monthly ERA5 file at a time to avoid memory issues.
        Uses netCDF4 directly for performance (70k+ files).
        """
        import xarray as xr
        from netCDF4 import Dataset as NC4Dataset

        # Variable mapping: ERA5 name → (LDASIN name, default_value)
        var_map = {
            'airtemp': ('T2D', 280.0),
            'spechum': ('Q2D', 0.005),
            'windspd': ('U2D', 2.0),
            'airpres': ('PSFC', 101325.0),
            'pptrate': ('RAINRATE', 0.0),
            'SWRadAtm': ('SWDOWN', 0.0),
            'LWRadAtm': ('LWDOWN', 300.0),
        }

        grid_shape = (3, 3)
        count = 0

        # Process one monthly file at a time (memory efficient)
        forcing_files = sorted(self.forcing_basin_path.glob("*.nc"))
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        for fpath in forcing_files:
            ds = xr.open_dataset(fpath).load()

            # Subset to simulation window
            times = pd.DatetimeIndex(ds['time'].values)
            mask = (times >= start_ts) & (times <= end_ts)
            if not mask.any():
                ds.close()
                continue

            ds = ds.isel(time=mask)
            times = pd.DatetimeIndex(ds['time'].values)

            # Pre-extract arrays for this month
            arrays = {}
            for era5_var, (ldasin_var, default) in var_map.items():
                if era5_var in ds:
                    arr = ds[era5_var].values.squeeze()
                    arrays[ldasin_var] = (arr, default)
                else:
                    arrays[ldasin_var] = (None, default)

            for i, t in enumerate(times):
                t_pd = pd.Timestamp(t)
                fname = t_pd.strftime('%Y%m%d%H') + '.LDASIN_DOMAIN1'
                out_path = str(self.forcing_dir / fname)

                nc = NC4Dataset(out_path, 'w', format='NETCDF4')
                nc.createDimension('south_north', grid_shape[0])
                nc.createDimension('west_east', grid_shape[1])

                for ldasin_var, (arr, default) in arrays.items():
                    val = float(arr[i]) if arr is not None else default
                    if np.isnan(val):
                        val = default
                    var = nc.createVariable(ldasin_var, 'f4',
                                            ('south_north', 'west_east'))
                    var[:] = np.full(grid_shape, val, dtype=np.float32)

                var = nc.createVariable('V2D', 'f4',
                                        ('south_north', 'west_east'))
                var[:] = np.zeros(grid_shape, dtype=np.float32)
                nc.close()
                count += 1

            ds.close()
            logger.info(f"  Processed {fpath.name}: {len(times)} timesteps (total: {count})")

        logger.info(f"Generated {count} LDASIN forcing files from ERA5 data")

    def _generate_synthetic_forcing(self, start_date: datetime, end_date: datetime) -> None:
        """Raise error — synthetic forcing should never be used."""
        raise FileNotFoundError(
            f"No basin-averaged ERA5 forcing data found in {self.forcing_basin_path}. "
            "WRF-Hydro requires real forcing data. Ensure the domain has been set up "
            "with forcing data before running preprocessing."
        )

    def _generate_domain_files(self) -> None:
        """Generate wrfinput and geogrid domain files."""
        logger.info("Generating WRF-Hydro domain files...")

        from netCDF4 import Dataset as NC4Dataset

        # Bow at Banff basin parameters
        lat = 51.17
        lon = -115.57
        elev = 1500.0
        dx = 1000.0  # Grid spacing in meters (3x3 lumped grid)

        wrfinput_path = self.settings_dir / 'wrfinput_d01.nc'
        ds = NC4Dataset(str(wrfinput_path), 'w', format='NETCDF4')

        # Global attributes required by WRF-Hydro
        ds.DX = dx
        ds.DY = dx
        ds.TRUELAT1 = lat
        ds.TRUELAT2 = lat
        ds.STAND_LON = lon
        ds.CEN_LAT = lat
        ds.CEN_LON = lon
        ds.MOAD_CEN_LAT = lat
        ds.MAP_PROJ = 1  # Lambert Conformal
        ds.ISWATER = 17
        ds.ISLAKE = 21
        ds.ISICE = 15
        ds.ISURBAN = 13
        ds.ISOILWATER = 14
        ds.GRID_ID = 1
        ds.MMINLU = "MODIFIED_IGBP_MODIS_NOAH"
        ds.NUM_LAND_CAT = 21
        ds.WEST_EAST_GRID_DIMENSION = 4
        ds.SOUTH_NORTH_GRID_DIMENSION = 4
        ds.TITLE = "WRF-Hydro domain for Bow at Banff (lumped)"

        # Dimensions
        ds.createDimension('south_north', 3)
        ds.createDimension('west_east', 3)
        ds.createDimension('soil_layers_stag', 4)
        ds.createDimension('Time', None)

        # Variables
        # WRF-Hydro reads XLAT/XLONG/HGT (without _M suffix)
        hgt = ds.createVariable('HGT', 'f4', ('south_north', 'west_east'))
        hgt[:] = np.full((3, 3), elev)

        xlat = ds.createVariable('XLAT', 'f4', ('south_north', 'west_east'))
        xlat[:] = np.full((3, 3), lat)

        xlong = ds.createVariable('XLONG', 'f4', ('south_north', 'west_east'))
        xlong[:] = np.full((3, 3), lon)

        # Also write _M versions for routing modules
        v = ds.createVariable('HGT_M', 'f4', ('south_north', 'west_east'))
        v[:] = np.full((3, 3), elev)
        v = ds.createVariable('XLAT_M', 'f4', ('south_north', 'west_east'))
        v[:] = np.full((3, 3), lat)
        v = ds.createVariable('XLONG_M', 'f4', ('south_north', 'west_east'))
        v[:] = np.full((3, 3), lon)

        ivgtyp = ds.createVariable('IVGTYP', 'i4', ('south_north', 'west_east'))
        ivgtyp[:] = np.full((3, 3), 14, dtype=np.int32)  # 14=Evergreen Needleleaf

        isltyp = ds.createVariable('ISLTYP', 'i4', ('south_north', 'west_east'))
        isltyp[:] = np.full((3, 3), 4, dtype=np.int32)  # 4=Silt Loam

        tmn = ds.createVariable('TMN', 'f4', ('south_north', 'west_east'))
        tmn[:] = np.full((3, 3), 275.0)  # Deep soil temperature ~2°C

        # LAI and vegetation fraction (needed by Noah-MP)
        lai = ds.createVariable('LAI', 'f4', ('south_north', 'west_east'))
        lai[:] = np.full((3, 3), 3.0)

        vegfra = ds.createVariable('VEGFRA', 'f4', ('south_north', 'west_east'))
        vegfra[:] = np.full((3, 3), 60.0)

        # Initial soil moisture and temperature
        smois = ds.createVariable('SMOIS', 'f4', ('soil_layers_stag', 'south_north', 'west_east'))
        smois[:] = np.full((4, 3, 3), 0.3)

        tslb = ds.createVariable('TSLB', 'f4', ('soil_layers_stag', 'south_north', 'west_east'))
        tslb[:] = np.full((4, 3, 3), 275.0)

        # Snow cover (initially zero)
        snow = ds.createVariable('SNOW', 'f4', ('south_north', 'west_east'))
        snow[:] = np.full((3, 3), 0.0)

        canwat = ds.createVariable('CANWAT', 'f4', ('south_north', 'west_east'))
        canwat[:] = np.full((3, 3), 0.0)

        tsk = ds.createVariable('TSK', 'f4', ('south_north', 'west_east'))
        tsk[:] = np.full((3, 3), 275.0)  # Skin temperature (K)

        xland = ds.createVariable('XLAND', 'f4', ('south_north', 'west_east'))
        xland[:] = np.full((3, 3), 1.0)  # 1=land

        shdmin = ds.createVariable('SHDMIN', 'f4', ('south_north', 'west_east'))
        shdmin[:] = np.full((3, 3), 10.0)  # Min green vegetation fraction %

        shdmax = ds.createVariable('SHDMAX', 'f4', ('south_north', 'west_east'))
        shdmax[:] = np.full((3, 3), 80.0)  # Max green vegetation fraction %

        ds.close()
        logger.info("Generated wrfinput_d01.nc domain file")

    def _generate_routing_files(self) -> None:
        """Generate Fulldom routing grid and channel routing files."""
        logger.info("Generating WRF-Hydro routing files...")

        from netCDF4 import Dataset as NC4Dataset

        # Fulldom_hires.nc (routing grid - same resolution as LSM for lumped)
        fulldom_path = self.routing_dir / 'Fulldom_hires.nc'
        ds = NC4Dataset(str(fulldom_path), 'w', format='NETCDF4')
        ds.DX = 1000.0
        ds.DY = 1000.0

        ds.createDimension('y', 3)
        ds.createDimension('x', 3)

        topo = ds.createVariable('TOPOGRAPHY', 'f4', ('y', 'x'))
        topo[:] = np.full((3, 3), 1500.0)

        flowdir = ds.createVariable('FLOWDIRECTION', 'i4', ('y', 'x'))
        # Flow direction: center cell flows out (value=0 for outlet)
        flowdir[:] = np.array([[4, 8, 8], [4, 0, 8], [2, 2, 4]], dtype=np.int32)

        chgrid = ds.createVariable('CHANNELGRID', 'i4', ('y', 'x'))
        chgrid[:] = np.full((3, 3), -1, dtype=np.int32)
        chgrid[1, 1] = 0  # Channel at center cell

        strmord = ds.createVariable('STREAMORDER', 'i4', ('y', 'x'))
        strmord[:] = np.full((3, 3), -1, dtype=np.int32)
        strmord[1, 1] = 1

        lakegrid = ds.createVariable('LAKEGRID', 'i4', ('y', 'x'))
        lakegrid[:] = np.full((3, 3), -1, dtype=np.int32)

        lat = ds.createVariable('LATITUDE', 'f4', ('y', 'x'))
        lat[:] = np.full((3, 3), 51.17)

        lon = ds.createVariable('LONGITUDE', 'f4', ('y', 'x'))
        lon[:] = np.full((3, 3), -115.57)

        ds.close()

        # Route_Link.nc (channel routing parameters)
        rl_path = self.routing_dir / 'Route_Link.nc'
        ds = NC4Dataset(str(rl_path), 'w', format='NETCDF4')

        ds.createDimension('feature_id', 1)

        link = ds.createVariable('link', 'i4', ('feature_id',))
        link[:] = [1]

        frm = ds.createVariable('from', 'i4', ('feature_id',))
        frm[:] = [0]

        to = ds.createVariable('to', 'i4', ('feature_id',))
        to[:] = [0]

        length = ds.createVariable('Length', 'f4', ('feature_id',))
        length[:] = [1000.0]

        n = ds.createVariable('n', 'f4', ('feature_id',))
        n[:] = [0.035]

        chslp = ds.createVariable('ChSlp', 'f4', ('feature_id',))
        chslp[:] = [0.01]

        btmwdth = ds.createVariable('BtmWdth', 'f4', ('feature_id',))
        btmwdth[:] = [5.0]

        ds.close()
        logger.info("Generated Fulldom_hires.nc and Route_Link.nc routing files")

    def _generate_hrldas_namelist(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate the HRLDAS namelist (namelist.hrldas).

        This controls the Noah-MP land surface model component.
        """
        namelist_file = self._get_config_value(
            lambda: self.config.model.wrfhydro.namelist_file,
            default='namelist.hrldas'
        )

        restart_freq = self._get_config_value(
            lambda: self.config.model.wrfhydro.restart_frequency,
            default='monthly'
        )

        # Map restart frequency to output steps
        restart_minutes = {'hourly': 60, 'daily': 1440, 'monthly': 43200}.get(
            restart_freq, 43200
        )

        content = f"""&NOAHLSM_OFFLINE

 HRLDAS_SETUP_FILE = '{self.settings_dir}/wrfinput_d01.nc'
 INDIR = '{self.forcing_dir}'
 OUTDIR = '{self.wrfhydro_input_dir}'

 START_YEAR  = {start_date.year}
 START_MONTH = {start_date.month:02d}
 START_DAY   = {start_date.day:02d}
 START_HOUR  = {start_date.hour:02d}
 START_MIN   = 00

 ! Simulation length in hours
 KHOUR = {(end_date - start_date).days * 24}

 ! Physics options
 DYNAMIC_VEG_OPTION                = 4
 CANOPY_STOMATAL_RESISTANCE_OPTION = 1
 BTR_OPTION                        = 1
 RUNOFF_OPTION                     = 3
 SURFACE_DRAG_OPTION               = 1
 SUPERCOOLED_WATER_OPTION          = 1
 FROZEN_SOIL_OPTION                = 1
 RADIATIVE_TRANSFER_OPTION         = 3
 SNOW_ALBEDO_OPTION                = 2
 PCP_PARTITION_OPTION              = 1
 TBOT_OPTION                       = 2
 TEMP_TIME_SCHEME_OPTION           = 3
 GLACIER_OPTION                    = 2
 SURFACE_RESISTANCE_OPTION         = 4

 ! Output
 OUTPUT_TIMESTEP = 3600
 RESTART_FREQUENCY_HOURS = {restart_minutes // 60}
 SPLIT_OUTPUT_COUNT = 1

 ! Forcing
 FORCING_TIMESTEP = 3600
 NOAH_TIMESTEP    = 3600

 ! Soil layers
 NSOIL = 4
 soil_thick_input(1) = 0.10
 soil_thick_input(2) = 0.30
 soil_thick_input(3) = 0.60
 soil_thick_input(4) = 1.00

 ZLVL = 10.0

/

&WRF_HYDRO_OFFLINE
 finemesh        = 0
 finemesh_factor = 1
 forc_typ        = 1
 snow_assim      = 0
/
"""
        out_path = self.settings_dir / namelist_file
        out_path.write_text(content)
        logger.info(f"Generated HRLDAS namelist: {out_path}")

    def _generate_hydro_namelist(self) -> None:
        """
        Generate the hydro namelist (hydro.namelist).

        This controls the hydrological routing component.
        """
        hydro_namelist_file = self._get_config_value(
            lambda: self.config.model.wrfhydro.hydro_namelist,
            default='hydro.namelist'
        )

        content = f"""&HYDRO_nlist

 ! System coupling
 sys_cpl = 1
 IGRID = 1

 ! Routing: disabled for lumped basin
 CHANRTSWCRT    = 0
 channel_option = 0
 SUBRTSWCRT     = 0
 OVRTSWCRT      = 0
 GWBASESWCRT    = 0

 ! Routing grid parameters
 AGGFACTRT = 1
 dtrt_ter  = 10
 dtrt_ch   = 10
 dxrt      = 1000.0
 NSOIL     = 4
 ZSOIL8(1) = -0.10
 ZSOIL8(2) = -0.40
 ZSOIL8(3) = -1.00
 ZSOIL8(4) = -2.00

 ! File paths
 GEO_STATIC_FLNM  = '{self.settings_dir}/wrfinput_d01.nc'
 GEO_FINEGRID_FLNM = '{self.routing_dir}/Fulldom_hires.nc'
 route_link_f      = '{self.routing_dir}/Route_Link.nc'

 ! Output control
 SPLIT_OUTPUT_COUNT = 1
 out_dt             = 60
 rst_dt             = 1440
 rst_typ            = 1
 rst_bi_in          = 0
 rst_bi_out         = 0
 RSTRT_SWC          = 0
 GW_RESTART         = 0
 order_to_write     = 1
 io_form_outputs    = 0
 io_config_outputs  = 0
 t0OutputFlag       = 1
 output_channelBucket_influx = 0
 TERADJ_SOLAR       = 0
 bucket_loss        = 0
 UDMP_OPT           = 0
 imperv_adj         = 0

 ! Output switches
 CHRTOUT_DOMAIN     = 0
 CHANOBS_DOMAIN     = 0
 CHRTOUT_GRID       = 0
 LSMOUT_DOMAIN      = 1
 RTOUT_DOMAIN       = 0
 output_gw          = 0
 outlake            = 0
 frxst_pts_out      = 0
 GW_RESTART  = 0

/

&NUDGING_nlist
 nudgingParamFile = ''
 netwkReExFile    = ''
/
"""
        out_path = self.settings_dir / hydro_namelist_file
        out_path.write_text(content)
        logger.info(f"Generated hydro namelist: {out_path}")
