"""
SWAT Model Preprocessor

Handles preparation of SWAT model inputs including:
- TxtInOut directory structure
- Forcing files (.pcp and .tmp) from ERA5 NetCDF data
- Basin file (.bsn) with default snow/surface parameters
- file.cio master control file
"""
import logging
from typing import Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry
from symfluence.models.mixins import ObservationLoaderMixin

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("SWAT")
class SWATPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):  # type: ignore[misc]
    """
    Prepares inputs for a SWAT model run.

    SWAT requires a TxtInOut directory containing:
    - file.cio: Master control file
    - .pcp files: Precipitation data
    - .tmp files: Temperature data (min/max)
    - .bsn: Basin-level parameters
    - .sub: Sub-basin files
    - .hru: HRU files
    - .gw: Groundwater files
    - .mgt: Management files
    - .sol: Soil files
    """

    def __init__(self, config, logger):
        """
        Initialize the SWAT preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Setup SWAT-specific directories
        self.swat_input_dir = self.project_dir / "SWAT_input"
        txtinout_name = self._get_config_value(
            lambda: self.config.model.swat.txtinout_dir,
            default='TxtInOut'
        )
        self.txtinout_dir = self.swat_input_dir / txtinout_name

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "SWAT"

    def run_preprocessing(self) -> bool:
        """
        Run the complete SWAT preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting SWAT preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Get simulation dates
            start_date, end_date = self._get_simulation_dates()

            # Generate forcing files from ERA5
            self._generate_forcing_files(start_date, end_date)

            # Generate basin file
            self._generate_basin_file()

            # Generate sub-basin, HRU, groundwater, management, and soil files
            self._generate_subbasin_files()

            # Generate watershed routing file (fig.fig)
            self._generate_fig_file()

            # Generate minimal database stub files
            self._generate_database_stubs()

            # Generate file.cio (must be last -- references all other files)
            self._generate_file_cio(start_date, end_date)

            logger.info("SWAT preprocessing complete.")
            return True

        except Exception as e:
            logger.error(f"SWAT preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create SWAT TxtInOut directory structure."""
        self.txtinout_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created SWAT TxtInOut directory at {self.txtinout_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from configuration."""
        start_str = self._get_config_value(lambda: self.config.domain.time_start)
        end_str = self._get_config_value(lambda: self.config.domain.time_end)

        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def _get_catchment_properties(self) -> Dict:
        """
        Get catchment properties from shapefile.

        Returns:
            Dict with centroid lat/lon, area, and elevation
        """
        try:
            import geopandas as gpd
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)

                # Get centroid
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y

                # Project to UTM for accurate area
                utm_zone = int((lon + 180) / 6) + 1
                hemisphere = 'north' if lat >= 0 else 'south'
                utm_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
                gdf_proj = gdf.to_crs(utm_crs)
                area_m2 = gdf_proj.geometry.area.sum()

                elev = float(gdf.get('elev_mean', [1000])[0]) if 'elev_mean' in gdf.columns else 1000.0

                return {
                    'lat': lat,
                    'lon': lon,
                    'area_m2': area_m2,
                    'area_km2': area_m2 / 1e6,
                    'elev': elev
                }
        except Exception as e:
            logger.warning(f"Could not read catchment properties: {e}")

        return {
            'lat': 51.0,
            'lon': -115.0,
            'area_m2': 1e8,
            'area_km2': 100.0,
            'elev': 1000.0
        }

    def _load_forcing_data(self):
        """Load basin-averaged forcing data from ERA5 NetCDF files."""
        import xarray as xr

        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

        if not forcing_files:
            merged_path = self.project_dir / 'forcing' / 'merged_path'
            if merged_path.exists():
                forcing_files = list(merged_path.glob("*.nc"))

        if not forcing_files:
            raise FileNotFoundError(f"No forcing data found in {self.forcing_basin_path}")

        logger.info(f"Loading forcing from {len(forcing_files)} files")

        try:
            ds = xr.open_mfdataset(forcing_files, combine='by_coords')
        except ValueError:
            try:
                ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time')
            except Exception:
                datasets = [xr.open_dataset(f) for f in forcing_files]
                ds = xr.merge(datasets)

        ds = self.subset_to_simulation_time(ds, "Forcing")
        return ds

    def _generate_forcing_files(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate SWAT forcing files (.pcp and .tmp) from ERA5 data.

        SWAT .pcp format: Fixed-width text with daily precipitation [mm]
        SWAT .tmp format: Fixed-width text with daily max/min temperature [deg C]
        """
        logger.info("Generating SWAT forcing files...")

        try:
            forcing_ds = self._load_forcing_data()
            self._write_pcp_file(forcing_ds, start_date, end_date)
            self._write_tmp_file(forcing_ds, start_date, end_date)
        except Exception as e:
            logger.warning(f"Could not load forcing data: {e}, generating synthetic")
            self._generate_synthetic_forcing(start_date, end_date)

    def _extract_variable(self, ds, candidates, default_val=0.0):
        """Extract a variable from dataset by trying multiple candidate names."""
        for candidate in candidates:
            if candidate in ds:
                data = ds[candidate].values
                # Average over spatial dims if present
                while data.ndim > 1:
                    data = np.nanmean(data, axis=-1)
                return data, candidate
        return None, None

    def _write_pcp_file(self, forcing_ds, start_date, end_date) -> None:
        """Write SWAT precipitation file (.pcp)."""
        precip_data, src_var = self._extract_variable(
            forcing_ds,
            ['pptrate', 'precipitation', 'pr', 'precip', 'tp', 'PREC']
        )

        if precip_data is None:
            logger.warning("No precipitation variable found, using zeros")
            dates = pd.date_range(start_date, end_date, freq='D')
            precip_data = np.zeros(len(dates))
        else:
            # Unit conversion
            src_units = ''
            if src_var and src_var in forcing_ds:
                src_units = forcing_ds[src_var].attrs.get('units', '')

            if 'mm/s' in src_units or src_var == 'pptrate':
                precip_data = precip_data * 86400  # mm/s -> mm/day
                logger.info(f"Converted {src_var} from mm/s to mm/day")
            elif src_units == 'm' or src_var == 'tp':
                precip_data = precip_data * 1000.0  # m -> mm
                logger.info(f"Converted {src_var} from m to mm")
            elif 'kg' in src_units and 'm-2' in src_units and 's-1' in src_units:
                precip_data = precip_data * 86400  # kg/m2/s -> mm/day
                logger.info(f"Converted {src_var} from kg/m2/s to mm/day")

        # Resample to daily if sub-daily
        times = forcing_ds['time'].values if 'time' in forcing_ds else pd.date_range(start_date, end_date, freq='D')
        precip_series = pd.Series(precip_data[:len(times)], index=pd.DatetimeIndex(times))
        precip_daily = precip_series.resample('D').sum()

        # Ensure non-negative
        precip_daily = precip_daily.clip(lower=0.0)

        # Write .pcp file
        # Header: 4 lines (title, column headers, lat/lon/elev, elevation integers)
        # Data: format (i4,i3,1800f5.1) -- year(4), julday(3), precip(5.1) per gage
        pcp_path = self.txtinout_dir / 'pcp1.pcp'
        lines = []
        lines.append("Station  1: SYMFLUENCE generated")
        lines.append("Lati    Long  Elev")
        props = self._get_catchment_properties()
        lines.append(f"{props['lat']:8.3f}{props['lon']:8.3f}{props['elev']:8.1f}")
        # Line 4: elevation as integer, format (7x,1800i5) -- 7 spaces + 5-char int per gage
        lines.append(f"{'':7s}{int(props['elev']):5d}")

        for date, precip in precip_daily.items():
            year = date.year
            jday = date.timetuple().tm_yday
            lines.append(f"{year:4d}{jday:3d}{precip:5.1f}")

        pcp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Precipitation file written: {pcp_path}")

    def _write_tmp_file(self, forcing_ds, start_date, end_date) -> None:
        """Write SWAT temperature file (.tmp) with daily max/min temperatures."""
        # Try to find temperature variable
        temp_data, src_var = self._extract_variable(
            forcing_ds,
            ['airtemp', 'temperature', 'tas', 'temp', 't2m', 'AIR_TEMP']
        )

        if temp_data is None:
            logger.warning("No temperature variable found, using synthetic")
            dates = pd.date_range(start_date, end_date, freq='D')
            doy = dates.dayofyear
            temp_data = 10 + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
            tmax_daily = pd.Series(temp_data + 5, index=dates)
            tmin_daily = pd.Series(temp_data - 5, index=dates)
        else:
            # Unit conversion
            src_units = ''
            if src_var and src_var in forcing_ds:
                src_units = forcing_ds[src_var].attrs.get('units', '')

            if src_units == 'K' or np.nanmean(temp_data) > 100:
                temp_data = temp_data - 273.15
                logger.info(f"Converted {src_var} from K to deg C")

            times = forcing_ds['time'].values if 'time' in forcing_ds else pd.date_range(start_date, end_date, freq='D')
            temp_series = pd.Series(temp_data[:len(times)], index=pd.DatetimeIndex(times))
            tmax_daily = temp_series.resample('D').max()
            tmin_daily = temp_series.resample('D').min()

        # Write .tmp file
        # Header: 4 lines (title, column headers, lat/lon/elev, elevation integers)
        # Data: format (i4,i3,3600f5.1) -- year(4), julday(3), tmax(5.1), tmin(5.1) per gage
        tmp_path = self.txtinout_dir / 'tmp1.tmp'
        lines = []
        lines.append("Station  1: SYMFLUENCE generated")
        lines.append("Lati    Long  Elev")
        props = self._get_catchment_properties()
        lines.append(f"{props['lat']:8.3f}{props['lon']:8.3f}{props['elev']:8.1f}")
        # Line 4: elevation as integer, format (7x,1800i10) -- 7 spaces + 10-char int per gage
        lines.append(f"{'':7s}{int(props['elev']):10d}")

        for date in tmax_daily.index:
            year = date.year
            jday = date.timetuple().tm_yday
            tmax = tmax_daily[date]
            tmin = tmin_daily[date]
            lines.append(f"{year:4d}{jday:3d}{tmax:5.1f}{tmin:5.1f}")

        tmp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Temperature file written: {tmp_path}")

    def _generate_synthetic_forcing(self, start_date: datetime, end_date: datetime) -> None:
        """Generate synthetic forcing data for testing."""
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)
        props = self._get_catchment_properties()

        # Synthetic precipitation
        precip = np.random.exponential(2.0, n)

        pcp_path = self.txtinout_dir / 'pcp1.pcp'
        lines = []
        lines.append("Station  1: SYMFLUENCE synthetic")
        lines.append("Lati    Long  Elev")
        lines.append(f"{props['lat']:8.3f}{props['lon']:8.3f}{props['elev']:8.1f}")
        # Line 4: elevation as integer, format (7x,1800i5)
        lines.append(f"{'':7s}{int(props['elev']):5d}")
        for i, date in enumerate(dates):
            year = date.year
            jday = date.timetuple().tm_yday
            lines.append(f"{year:4d}{jday:3d}{precip[i]:5.1f}")
        pcp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

        # Synthetic temperature
        doy = dates.dayofyear
        tmean = 10 + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
        tmax = tmean + 5.0
        tmin = tmean - 5.0

        tmp_path = self.txtinout_dir / 'tmp1.tmp'
        lines = []
        lines.append("Station  1: SYMFLUENCE synthetic")
        lines.append("Lati    Long  Elev")
        lines.append(f"{props['lat']:8.3f}{props['lon']:8.3f}{props['elev']:8.1f}")
        # Line 4: elevation as integer, format (7x,1800i10)
        lines.append(f"{'':7s}{int(props['elev']):10d}")
        for i, date in enumerate(dates):
            year = date.year
            jday = date.timetuple().tm_yday
            lines.append(f"{year:4d}{jday:3d}{tmax[i]:5.1f}{tmin[i]:5.1f}")
        tmp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

        logger.info(f"Synthetic forcing files written to {self.txtinout_dir}")

    def _generate_basin_file(self) -> None:
        """Generate SWAT basin file (.bsn) matching readbsn.f read order.

        readbsn.f reads the .bsn file in a very specific order after 3
        title lines.  Every value must appear on its own line, in exactly
        the order that the Fortran reads consume them.  String values
        (petfile, wwqfile) use format 1000 = (a), reading the first
        len(variable) characters.  Integer values use format 1001 = (i4).
        Real values use free-format (read *).
        """
        from .parameters import DEFAULT_PARAMS

        bsn_path = self.txtinout_dir / 'basins.bsn'

        # Default values for basin-level parameters
        sftmp = DEFAULT_PARAMS['SFTMP']
        smtmp = DEFAULT_PARAMS['SMTMP']
        smfmx = DEFAULT_PARAMS['SMFMX']
        smfmn = DEFAULT_PARAMS['SMFMN']
        timp = DEFAULT_PARAMS['TIMP']
        snocovmx = 1.0       # Minimum snow water content for 100% snow cover [mm]
        sno50cov = 0.5       # Snow water content for 50% snow cover [mm]
        ipet = 1             # PET method (0=P-M, 1=Hargreaves, 2=Priestley-Taylor)
        petfile = ""         # PET input file (blank = not used)
        esco_bsn = DEFAULT_PARAMS['ESCO']
        epco_bsn = 0.95      # Plant uptake compensation factor
        evlai = 3.0          # Leaf area index at which no evap from water surface
        ffcb = 0.0           # Initial soil water storage as fraction of FC

        ievent = 0           # Rainfall/runoff code (0=daily)
        icrk = 0             # Crack flow code
        surlag_bsn = DEFAULT_PARAMS['SURLAG']
        adj_pkr = 0.0        # Peak rate adjustment for sediment routing
        prf_bsn = 1.0        # Peak rate adjustment for channel sediment routing
        spcon_bsn = 0.0001   # Linear coeff for channel sed re-entrainment
        spexp_bsn = 1.0      # Exponent for channel sed re-entrainment

        rcn_sub_bsn = 1.0    # Concentration of N in rainfall [mg/l]
        cmn_bsn = 0.0003     # Rate coefficient for humus mineralization
        n_updis = 20.0       # N uptake distribution parameter
        p_updis = 20.0       # P uptake distribution parameter
        nperco_bsn = 0.20    # N percolation coefficient
        pperco_bsn = 10.0    # P percolation coefficient
        phoskd_bsn = 175.0   # P soil partitioning coefficient
        psp_bsn = 0.4        # P sorption coefficient
        rsdco = 0.05         # Residue decomposition coefficient

        percop = 0.5         # Pesticide percolation coefficient
        isubwq = 0           # Subbasin water quality code

        # 16 bacteria parameters (all 0.0)
        bact_params = [0.0] * 16
        ised_det = 0         # Sediment detachment method

        irte = 0             # Channel routing (0=variable-storage)
        msk_co1 = 0.0        # Muskingum calibration coeff 1
        msk_co2 = 3.5        # Muskingum calibration coeff 2
        msk_x = 0.2          # Muskingum weighting factor
        ideg = 0             # Channel degradation code
        iwq = 0              # In-stream water quality code
        wwqfile = "basins.wwq"  # Water quality file (char*13)
        trnsrch = 0.0        # Fraction of transmission losses to deep aquifer
        evrch = 1.0          # Reach evaporation coefficient
        irtpest = 0          # Pesticide routing flag
        icn = 0              # Daily CN calculation method (0=traditional)
        cncoef = 0.0         # CN coefficient (for icn=2)
        cdn_bsn = 1.4        # Denitrification exponential rate coefficient
        sdnco_bsn = 1.1      # Denitrification threshold water content
        bact_swf = 0.0       # Fraction of bacteria in solution

        # Optional params (read with iostat=eof)
        bactmx = 0.0
        bactminlp = 0.0
        bactminp = 0.0
        wdlprch = 0.0
        wdprch = 0.0
        wdlpres = 0.0
        wdpres = 0.0
        tb_adj = 0.0
        depimp_bsn = 6000.0
        ddrain_bsn = 0.0
        tdrain_bsn = 0.0
        gdrain_bsn = 0.0
        cn_froz = 0.000862

        lines = []

        # -- 3 title lines --
        lines.append(" Basin parameters -- SYMFLUENCE generated")
        lines.append(" SWAT basin (.bsn) file")
        lines.append(" ")

        # -- Block 1: snow, PET, evap params (read *, then title) --
        lines.append(f"  {sftmp:14.4f}    | SFTMP : Snowfall temperature [deg C]")
        lines.append(f"  {smtmp:14.4f}    | SMTMP : Snow melt base temperature [deg C]")
        lines.append(f"  {smfmx:14.4f}    | SMFMX : Max melt rate [mm/deg C/day]")
        lines.append(f"  {smfmn:14.4f}    | SMFMN : Min melt rate [mm/deg C/day]")
        lines.append(f"  {timp:14.4f}    | TIMP : Snow pack temperature lag factor")
        lines.append(f"  {snocovmx:14.4f}    | SNOCOVMX : Min snow for 100% cover [mm]")
        lines.append(f"  {sno50cov:14.4f}    | SNO50COV : Snow for 50% cover")
        # ipet: integer format (i4) -- first 4 chars
        lines.append(f"{ipet:4d}                | IPET : PET method code")
        # petfile: format (a) -- first len chars; blank = not used
        lines.append(f"{petfile:13s}     | PETFILE : PET input file")
        lines.append(f"  {esco_bsn:14.4f}    | ESCO : Soil evaporation compensation factor")
        lines.append(f"  {epco_bsn:14.4f}    | EPCO : Plant uptake compensation factor")
        lines.append(f"  {evlai:14.4f}    | EVLAI : LAI at which no evap from water surface")
        lines.append(f"  {ffcb:14.4f}    | FFCB : Initial soil water as fraction of FC")
        lines.append(" Runoff/Sediment:")

        # -- Block 2: runoff, sediment (read *, then title) --
        lines.append(f"{ievent:4d}                | IEVENT : Rainfall/runoff code")
        lines.append(f"{icrk:4d}                | ICRK : Crack flow code")
        lines.append(f"  {surlag_bsn:14.4f}    | SURLAG : Surface runoff lag coefficient")
        lines.append(f"  {adj_pkr:14.4f}    | ADJ_PKR : Peak rate adj for sediment routing")
        lines.append(f"  {prf_bsn:14.4f}    | PRF : Peak rate adj for channel sediment routing")
        lines.append(f"  {spcon_bsn:14.4f}    | SPCON : Linear coeff for channel sed re-entrainment")
        lines.append(f"  {spexp_bsn:14.4f}    | SPEXP : Exponent for channel sed re-entrainment")
        lines.append(" Nutrients:")

        # -- Block 3: nutrients (read *, then title) --
        lines.append(f"  {rcn_sub_bsn:14.4f}    | RCN : Concentration of N in rainfall [mg/l]")
        lines.append(f"  {cmn_bsn:14.4f}    | CMN : Rate coeff for humus mineralization")
        lines.append(f"  {n_updis:14.4f}    | N_UPDIS : N uptake distribution parameter")
        lines.append(f"  {p_updis:14.4f}    | P_UPDIS : P uptake distribution parameter")
        lines.append(f"  {nperco_bsn:14.4f}    | NPERCO : N percolation coefficient")
        lines.append(f"  {pperco_bsn:14.4f}    | PPERCO : P percolation coefficient")
        lines.append(f"  {phoskd_bsn:14.4f}    | PHOSKD : P soil partitioning coefficient")
        lines.append(f"  {psp_bsn:14.4f}    | PSP : P sorption coefficient")
        lines.append(f"  {rsdco:14.4f}    | RSDCO : Residue decomposition coefficient")
        lines.append(" Pesticides:")

        # -- Block 4: percop, then title, isubwq, then title --
        lines.append(f"  {percop:14.4f}    | PERCOP : Pesticide percolation coefficient")
        lines.append(" Algae/CBOD/DO:")
        lines.append(f"{isubwq:4d}                | ISUBWQ : Subbasin water quality code")
        lines.append(" Bacteria:")

        # -- 16 bacteria params (all 0), then ised_det, then title --
        bact_names = [
            "WDPQ", "WGPQ", "WDLPQ", "WGLPQ",
            "WDPS", "WGPS", "WDLPS", "WGLPS",
            "WDPF", "WGPF", "WDLPF", "WGLPF",
            "WDPSC", "WGPSC", "WDLPSC", "WGLPSC",
        ]
        for i, bname in enumerate(bact_names):
            lines.append(f"  {bact_params[i]:14.4f}    | {bname}")
        lines.append(f"{ised_det:4d}                | ISED_DET : Sediment detachment method")
        lines.append(" Channel Routing:")

        # -- Block 5: routing params (read *) --
        lines.append(f"{irte:4d}                | IRTE : Channel routing method")
        lines.append(f"  {msk_co1:14.4f}    | MSK_CO1 : Muskingum calib coeff 1")
        lines.append(f"  {msk_co2:14.4f}    | MSK_CO2 : Muskingum calib coeff 2")
        lines.append(f"  {msk_x:14.4f}    | MSK_X : Muskingum weighting factor")
        lines.append(f"{ideg:4d}                | IDEG : Channel degradation code")
        lines.append(f"{iwq:4d}                | IWQ : In-stream water quality code")
        # wwqfile: format (a), char*13
        lines.append(f"{wwqfile:13s}     | WWQFILE : Water quality input file")
        lines.append(f"  {trnsrch:14.4f}    | TRNSRCH : Fraction of trans losses to deep aq")
        lines.append(f"  {evrch:14.4f}    | EVRCH : Reach evaporation coefficient")
        lines.append(f"{irtpest:4d}                | IRTPEST : Pesticide routing flag")
        lines.append(f"{icn:4d}                | ICN : Daily CN calculation method")
        lines.append(f"  {cncoef:14.4f}    | CNCOEF : CN coefficient")
        lines.append(f"  {cdn_bsn:14.4f}    | CDN : Denitrification exponential rate coeff")
        lines.append(f"  {sdnco_bsn:14.4f}    | SDNCO : Denitrification threshold water content")
        lines.append(f"  {bact_swf:14.4f}    | BACT_SWF : Fraction of bacteria in solution")

        # -- Optional params (read with iostat=eof) --
        lines.append(f"  {bactmx:14.4f}    | BACTMX")
        lines.append(f"  {bactminlp:14.4f}    | BACTMINLP")
        lines.append(f"  {bactminp:14.4f}    | BACTMINP")
        lines.append(f"  {wdlprch:14.4f}    | WDLPRCH")
        lines.append(f"  {wdprch:14.4f}    | WDPRCH")
        lines.append(f"  {wdlpres:14.4f}    | WDLPRES")
        lines.append(f"  {wdpres:14.4f}    | WDPRES")
        lines.append(f"  {tb_adj:14.4f}    | TB_ADJ")
        lines.append(f"  {depimp_bsn:14.4f}    | DEPIMP_BSN")
        lines.append(f"  {ddrain_bsn:14.4f}    | DDRAIN_BSN")
        lines.append(f"  {tdrain_bsn:14.4f}    | TDRAIN_BSN")
        lines.append(f"  {gdrain_bsn:14.4f}    | GDRAIN_BSN")
        lines.append(f"  {cn_froz:14.6f}    | CN_FROZ")

        bsn_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Basin file written: {bsn_path}")

        # Write dummy basins.wwq file (required by readbsn.f when iwq=0)
        wwq_path = self.txtinout_dir / 'basins.wwq'
        wwq_path.write_text(
            " Water quality file -- SYMFLUENCE\n", encoding='utf-8')
        logger.info(f"Water quality file written: {wwq_path}")

    def _generate_subbasin_files(self) -> None:
        """Generate sub-basin, HRU, groundwater, management, and soil files.

        The .sub file must conform to the layout expected by both
        ``getallo.f`` and ``readsub.f``.

        ``getallo.f`` does::

            do j = 1, 52
              read(25,6000) titldum      ! skip 52 lines
            end do
            read(25,*) numhru            ! line 53 = number of HRUs
            do j = 1, 8
              read(25,6000) titldum      ! skip 8 lines (54-61)
            end do
            call hruallo                 ! reads HRU file refs at line 62+

        ``readsub.f`` reads every line sequentially (title, SUB_KM,
        lat, elev, gage codes, WGN filename, elevation bands, lapse
        rates, channel params, pond/wus files, climate-change
        adjustments, then hrutot on line 53, 8 skip lines, and HRU
        file references on line 62+).

        ``hruallo.f`` reads each HRU line with format ``(4a13,52x,i6)``
        (4 filenames of 13 chars, 52-char gap, then an integer).
        ``readsub.f`` reads with format ``(8a13,i6)`` (8 filenames of
        13 chars, then an integer).  Both consume the same physical
        line, so filenames must sit in exact 13-char columns.
        """
        from .parameters import DEFAULT_PARAMS

        props = self._get_catchment_properties()

        # Helper: format a 10-value elevation-band line (format 10f8.1)
        def fmt_10f81(vals):
            return ''.join(f"{v:8.1f}" for v in vals)

        # Helper: format a 6-value monthly-adjustment line (format 10f8.1)
        def fmt_6f81(vals):
            return ''.join(f"{v:8.1f}" for v in vals)

        # ------------------------------------------------------------------
        # Generate auxiliary files referenced by .sub
        # ------------------------------------------------------------------
        self._generate_wgn_file(props)
        self._generate_pnd_stub()
        self._generate_wus_stub()
        self._generate_chm_stub()

        # ------------------------------------------------------------------
        # Build .sub file -- exactly 62+ lines
        # ------------------------------------------------------------------
        sub_path = self.txtinout_dir / '000010001.sub'

        zeros10 = [0.0] * 10
        zeros6  = [0.0] * 6

        sub_lines = []

        # Line  1: Title
        sub_lines.append(" Subbasin: 1 -- SYMFLUENCE generated")

        # Line  2: SUB_KM (read with read *)
        sub_lines.append(
            f"  {props['area_km2']:14.4f}    | SUB_KM : Subbasin area [km2]")

        # Line  3: skip line (titldum) -- readsub reads this as titldum
        #          (in isproj==3 branch it reads harg_petco etc; else just titldum)
        sub_lines.append(
            f"  {0.0023:14.4f}    | HARG_PETCO / or skip line")

        # Line  4: skip line (titldum)
        sub_lines.append(" Subbasin climate and channel data")

        # Line  5: SUB_LAT
        sub_lines.append(
            f"  {props['lat']:14.4f}    | SUB_LAT : Latitude [deg]")

        # Line  6: SUB_ELEV
        sub_lines.append(
            f"  {props['elev']:14.4f}    | SUB_ELEV : Mean elevation [m]")

        # Line  7: IRGAGE (rain gage code)
        sub_lines.append(
            f"  {1:14d}    | IRGAGE : Precipitation gage data code")

        # Line  8: ITGAGE (temp gage code)
        sub_lines.append(
            f"  {1:14d}    | ITGAGE : Temperature gage data code")

        # Line  9: ISGAGE (solar gage code)
        sub_lines.append(
            f"  {0:14d}    | ISGAGE : Solar radiation gage data code")

        # Line 10: IHGAGE (humidity gage code)
        sub_lines.append(
            f"  {0:14d}    | IHGAGE : Relative humidity gage data code")

        # Line 11: IWGAGE (wind gage code)
        sub_lines.append(
            f"  {0:14d}    | IWGAGE : Wind speed gage data code")

        # Line 12: WGNFILE -- read with format (8a13,i6) => first 13 chars
        sub_lines.append(
            f"{'000010001.wgn':13s}"
            f"             | WGNFILE : Weather generator file")

        # Line 13: FCST_REG (forecast region)
        sub_lines.append(
            f"  {0:14d}    | FCST_REG : Forecast region")

        # Line 14: skip (elevation band header)
        sub_lines.append(" Elevation Band Data")

        # Line 15: skip (titldum before elevb values)
        sub_lines.append(" Elevation of band center [m]:")

        # Line 16: ELEVB(1:10) -- format 10f8.1
        sub_lines.append(fmt_10f81(zeros10))

        # Line 17: skip (titldum)
        sub_lines.append(" Fraction of subbasin area in band:")

        # Line 18: ELEVB_FR(1:10) -- format 10f8.1
        sub_lines.append(fmt_10f81(zeros10))

        # Line 19: skip (titldum)
        sub_lines.append(" Snow water equivalent in band [mm]:")

        # Line 20: SNOEB(1:10) -- format 10f8.1
        sub_lines.append(fmt_10f81(zeros10))

        # Line 21: PLAPS (precip lapse rate)
        sub_lines.append(
            f"  {0.0:14.4f}    | PLAPS : Precip lapse rate [mm H2O/km]")

        # Line 22: TLAPS (temp lapse rate)
        sub_lines.append(
            f"  {0.0:14.4f}    | TLAPS : Temp lapse rate [deg C/km]")

        # Line 23: SNO_SUB (initial snow)
        sub_lines.append(
            f"  {0.0:14.4f}    | SNO_SUB : Initial snow water [mm H2O]")

        # Line 24: skip (titldum -- channel header)
        sub_lines.append(" Channel data:")

        # Line 25: CH_L1 (longest tributary channel length)
        sub_lines.append(
            f"  {1.0:14.4f}    | CH_L1 : Longest trib channel length [km]")

        # Line 26: CH_S1 (average slope of tributary channel)
        sub_lines.append(
            f"  {0.05:14.4f}    | CH_S1 : Avg slope of trib channel [m/m]")

        # Line 27: CH_W1 (average width of tributary channel)
        sub_lines.append(
            f"  {10.0:14.4f}    | CH_W1 : Avg width of trib channel [m]")

        # Line 28: CH_K1 (effective hydraulic conductivity)
        sub_lines.append(
            f"  {0.0:14.4f}    | CH_K1 : Eff hydraulic conductivity [mm/hr]")

        # Line 29: CH_N1 (Manning's n for tributary channel)
        sub_lines.append(
            f"  {0.014:14.4f}    | CH_N1 : Manning's n for trib channel")

        # Line 30: skip (titldum -- pond header)
        sub_lines.append(" Pond/Wetland data:")

        # Line 31: PNDFILE -- read with format (8a13,i6) => first 13 chars
        sub_lines.append(
            f"{'000010001.pnd':13s}"
            f"             | PNDFILE : Pond input file")

        # Line 32: skip (titldum -- water use header)
        sub_lines.append(" Water Use data:")

        # Line 33: WUSFILE -- read with format (8a13,i6) => first 13 chars
        sub_lines.append(
            f"{'000010001.wus':13s}"
            f"             | WUSFILE : Water use file")

        # Line 34: SNOFILE / Climate Change -- read as titldum/snofile
        sub_lines.append("              ")  # 13+ blank chars = no snow file

        # Line 35: CO2
        sub_lines.append(
            f"  {330.0:14.4f}    | CO2 : CO2 concentration [ppmv]")

        # Line 36: skip (titldum -- rainfall adj header)
        sub_lines.append(" Rainfall adjustment (months 1-6):")

        # Line 37: RFINC(1:6) -- format 10f8.1
        sub_lines.append(fmt_6f81(zeros6))

        # Line 38: skip (titldum)
        sub_lines.append(" Rainfall adjustment (months 7-12):")

        # Line 39: RFINC(7:12)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 40: skip (titldum -- temp adj header)
        sub_lines.append(" Temperature adjustment (months 1-6):")

        # Line 41: TMPINC(1:6)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 42: skip (titldum)
        sub_lines.append(" Temperature adjustment (months 7-12):")

        # Line 43: TMPINC(7:12)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 44: skip (titldum -- radiation adj header)
        sub_lines.append(" Solar radiation adjustment (months 1-6):")

        # Line 45: RADINC(1:6)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 46: skip (titldum)
        sub_lines.append(" Solar radiation adjustment (months 7-12):")

        # Line 47: RADINC(7:12)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 48: skip (titldum -- humidity adj header)
        sub_lines.append(" Humidity adjustment (months 1-6):")

        # Line 49: HUMINC(1:6)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 50: skip (titldum)
        sub_lines.append(" Humidity adjustment (months 7-12):")

        # Line 51: HUMINC(7:12)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 52: skip (titldum -- HRU section header)
        sub_lines.append(" HRU data:")

        # ---- Line 53: NUMBER OF HRUs (critical for getallo.f) ----
        sub_lines.append(
            f"  {1:14d}    | HRUTOT : Number of HRUs in subbasin")

        # Lines 54-61: 8 skip/header lines before HRU file references
        sub_lines.append(
            " HRU: Num  Luse   Soil         Slope        Fraction")
        sub_lines.append(
            "  1  FRST  SoilA  0-9999       1.000")
        sub_lines.append(
            f"  {'FRST':14s}    | LUSE : Land use")
        sub_lines.append(
            f"  {'SoilA':14s}    | SOIL : Soil type")
        sub_lines.append(
            f"  {'0-9999':14s}    | SLOPE_CD : Slope class")
        sub_lines.append(
            f"  {1.0:14.4f}    | HRU_FR : Fraction of subbasin in HRU")
        sub_lines.append(
            f"  {0.05:14.4f}    | HRU_SLP : Average slope steepness [m/m]")
        sub_lines.append(
            f"  {50.0:14.4f}    | OV_N : Manning's n for overland flow")

        # ---- Line 62: HRU file references ----
        # Format (8a13,i6) for readsub: hrufile, mgtfile, solfile,
        #   chmfile, gwfile, opsfile, septfile, sdrfile, ils2
        # Format (4a13,52x,i6) for hruallo: hrufile, mgtfile, solfile,
        #   chmfile, (skip 52 covering gw+ops+sep+sdr), ils2
        # Both read the same 110-char line.
        hru_ref_line = (
            f"{'000010001.hru':13s}"
            f"{'000010001.mgt':13s}"
            f"{'000010001.sol':13s}"
            f"{'000010001.chm':13s}"
            f"{'000010001.gw':13s}"
            f"{'':13s}"  # opsfile (blank)
            f"{'':13s}"  # septfile (blank)
            f"{'':13s}"  # sdrfile (blank)
            f"{0:6d}"    # ils2 (landscape routing flag)
        )
        sub_lines.append(hru_ref_line)

        # Verify line count (should be at least 62)
        assert len(sub_lines) >= 62, (
            f".sub file has {len(sub_lines)} lines, need >= 62"
        )

        sub_path.write_text('\n'.join(sub_lines) + '\n', encoding='utf-8')
        logger.info(
            f"Sub-basin file written: {sub_path} ({len(sub_lines)} lines)")

        # --------------------------------------------------------------
        # HRU file (.hru) -- readhru.f reads ~53 values
        # Line 1 is title, then all remaining are free-format (* reads).
        # --------------------------------------------------------------
        hru_path = self.txtinout_dir / '000010001.hru'
        hru_lines = [
            " HRU:1 Subbasin:1 -- SYMFLUENCE generated",
            # Line  2: HRU_FR
            f"  {1.0:14.4f}    | HRU_FR : Fraction of subbasin area",
            # Line  3: SLSUBBSN
            f"  {91.46:14.4f}    | SLSUBBSN : Avg slope length [m]",
            # Line  4: HRU_SLP
            f"  {0.05:14.4f}    | HRU_SLP : Avg slope steepness [m/m]",
            # Line  5: OV_N
            f"  {0.14:14.4f}    | OV_N : Manning's n for overland flow",
            # Line  6: LAT_TTIME
            f"  {0.0:14.4f}    | LAT_TTIME : Lateral flow travel time [days]",
            # Line  7: LAT_SED
            f"  {0.0:14.4f}    | LAT_SED : Sediment conc in lateral flow [mg/l]",
            # Line  8: SLSOIL
            f"  {0.0:14.4f}    | SLSOIL : Slope length for lateral subsurface flow [m]",
            # Line  9: CANMX
            f"  {0.0:14.4f}    | CANMX : Maximum canopy storage [mm H2O]",
            # Line 10: ESCO
            f"  {DEFAULT_PARAMS['ESCO']:14.4f}    | ESCO : Soil evaporation compensation factor",
            # Line 11: EPCO
            f"  {0.95:14.4f}    | EPCO : Plant uptake compensation factor",
            # Line 12: RSDIN
            f"  {0.0:14.4f}    | RSDIN : Initial residue cover [kg/ha]",
            # Line 13: ERORGN
            f"  {0.0:14.4f}    | ERORGN : Organic N enrichment ratio",
            # Line 14: ERORGP
            f"  {0.0:14.4f}    | ERORGP : Organic P enrichment ratio",
            # Line 15: POT_FR
            f"  {0.0:14.4f}    | POT_FR : Fraction draining to pothole",
            # Line 16: FLD_FR
            f"  {0.0:14.4f}    | FLD_FR : Fraction of HRU in floodplain",
            # Line 17: RIP_FR
            f"  {0.0:14.4f}    | RIP_FR : Fraction of HRU in riparian zone",
            # Line 18: title
            " Pothole data:",
            # Line 19: POT_TILEMM
            f"  {0.0:14.4f}    | POT_TILEMM",
            # Line 20: POT_VOLXMM
            f"  {0.0:14.4f}    | POT_VOLXMM",
            # Line 21: POT_VOLMM
            f"  {0.0:14.4f}    | POT_VOLMM",
            # Line 22: POT_NSED
            f"  {0.0:14.4f}    | POT_NSED",
            # Line 23: POT_NO3L
            f"  {0.0:14.4f}    | POT_NO3L",
            # Line 24: DEP_IMP
            f"  {6000.0:14.4f}    | DEP_IMP : Depth to impervious layer [mm]",
            # Line 25: title
            " Urban data:",
            # Line 26: title
            " Consumptive water use:",
            # Line 27: title
            " Tile drain:",
            # Line 28: EVPOT
            f"  {0.0:14.4f}    | EVPOT",
            # Line 29: DIS_STREAM
            f"  {0.0:14.4f}    | DIS_STREAM : Distance to stream [m]",
            # Line 30: CF (concentration factor)
            f"  {0.0:14.4f}    | CF",
            # Line 31: CFH
            f"  {0.0:14.4f}    | CFH",
            # Line 32: CFDEC
            f"  {0.0:14.4f}    | CFDEC",
            # Line 33: SED_CON
            f"  {0.0:14.4f}    | SED_CON : Sediment concentration [mg/l]",
            # Line 34: ORGN_CON
            f"  {0.0:14.4f}    | ORGN_CON : Organic N concentration [mg/l]",
            # Line 35: ORGP_CON
            f"  {0.0:14.4f}    | ORGP_CON : Organic P concentration [mg/l]",
            # Line 36: SOLN_CON
            f"  {0.0:14.4f}    | SOLN_CON : Soluble N concentration [mg/l]",
            # Line 37: SOLP_CON
            f"  {0.0:14.4f}    | SOLP_CON : Soluble P concentration [mg/l]",
            # Line 38: POT_SOLPL
            f"  {0.0:14.4f}    | POT_SOLPL",
            # Line 39: POT_K
            f"  {0.0:14.4f}    | POT_K",
            # Line 40: N_REDUC
            f"  {0.0:14.4f}    | N_REDUC",
            # Line 41: N_LAG
            f"  {0.0:14.4f}    | N_LAG",
            # Line 42: N_LN
            f"  {0.0:14.4f}    | N_LN",
            # Line 43: N_LNCO
            f"  {0.0:14.4f}    | N_LNCO",
            # Line 44: SURLAG (HRU-level override)
            f"  {DEFAULT_PARAMS['SURLAG']:14.4f}    | SURLAG : Surface runoff lag coefficient",
            # Line 45: R2ADJ
            f"  {0.0:14.4f}    | R2ADJ",
            # Line 46: CMN
            f"  {0.0003:14.4f}    | CMN",
            # Line 47: CDN
            f"  {1.4:14.4f}    | CDN",
            # Line 48: NPERCO
            f"  {0.20:14.4f}    | NPERCO",
            # Line 49: PHOSKD
            f"  {175.0:14.4f}    | PHOSKD",
            # Line 50: PSP
            f"  {0.4:14.4f}    | PSP",
            # Line 51: SDNCO
            f"  {1.1:14.4f}    | SDNCO",
            # Line 52: IWETILE (integer)
            f"  {0:14d}    | IWETILE",
            # Line 53: IWETGW (integer)
            f"  {0:14d}    | IWETGW",
        ]
        hru_path.write_text('\n'.join(hru_lines) + '\n', encoding='utf-8')

        # --------------------------------------------------------------
        # Groundwater file (.gw) -- readgw.f reads title + 17 data values
        # Line 1: title; Lines 2-18: free-format read * values
        # --------------------------------------------------------------
        gw_path = self.txtinout_dir / '000010001.gw'
        gw_lines = [
            " Groundwater parameters -- SYMFLUENCE generated",
            # Line  2: SHALLST
            f"  {0.5:14.4f}    | SHALLST : Initial shallow aquifer storage [mm]",
            # Line  3: DEEPST
            f"  {1000.0:14.4f}    | DEEPST : Initial deep aquifer storage [mm]",
            # Line  4: GW_DELAY (= delay)
            f"  {DEFAULT_PARAMS['GW_DELAY']:14.4f}    | GW_DELAY : Groundwater delay time [days]",
            # Line  5: ALPHA_BF
            f"  {DEFAULT_PARAMS['ALPHA_BF']:14.4f}    | ALPHA_BF : Baseflow alpha factor [1/days]",
            # Line  6: GWQMN
            f"  {DEFAULT_PARAMS['GWQMN']:14.4f}    | GWQMN : Threshold depth for return flow [mm]",
            # Line  7: GW_REVAP
            f"  {DEFAULT_PARAMS['GW_REVAP']:14.4f}    | GW_REVAP : Groundwater revap coefficient",
            # Line  8: REVAPMN
            f"  {500.0:14.4f}    | REVAPMN : Threshold depth for revap [mm]",
            # Line  9: RCHRG_DP
            f"  {0.05:14.4f}    | RCHRG_DP : Deep aquifer percolation fraction",
            # Line 10: GWHT
            f"  {10.0:14.4f}    | GWHT : Initial groundwater height [m]",
            # Line 11: GW_SPYLD
            f"  {0.003:14.4f}    | GW_SPYLD : Specific yield of shallow aquifer [m3/m3]",
            # Line 12: SHALLST_N
            f"  {0.0:14.4f}    | SHALLST_N : Initial NO3 in shallow aquifer [mg/l]",
            # Line 13: GWMINP (=gwsolp in some versions)
            f"  {0.0:14.4f}    | GWMINP : Minimum P in groundwater [mg/l]",
            # Line 14: HLIFE_NGW
            f"  {0.0:14.4f}    | HLIFE_NGW : Half-life of NO3 in shallow aquifer [days]",
            # Line 15: LAT_ORGN (NEW)
            f"  {0.0:14.4f}    | LAT_ORGN : Organic N in lateral flow [mg/l]",
            # Line 16: LAT_ORGP (NEW)
            f"  {0.0:14.4f}    | LAT_ORGP : Organic P in lateral flow [mg/l]",
            # Line 17: ALPHA_BF_D (NEW)
            f"  {0.0:14.4f}    | ALPHA_BF_D : Baseflow alpha factor for deep aquifer [1/days]",
            # Line 18: GWNO3_AQ (NEW -- read with iostat=eof, optional)
            f"  {0.0:14.4f}    | GWNO3_AQ : NO3 concentration in groundwater [mg/l]",
        ]
        gw_path.write_text('\n'.join(gw_lines) + '\n', encoding='utf-8')

        # --------------------------------------------------------------
        # Management file (.mgt) -- readmgt.f reads exactly 30 lines
        # hruallo.f skips 30 lines before reading management operations.
        # Line 1: title, then pairs of (data, title) or (data) reads.
        # All data reads use free-format (read *).
        # --------------------------------------------------------------
        mgt_path = self.txtinout_dir / '000010001.mgt'
        mgt_lines = [
            # Line  1: title
            " Management parameters -- SYMFLUENCE generated",
            # Line  2: NMGT (management schedule number)
            f"  {0:14d}    | NMGT : Management schedule number",
            # Line  3: title
            " Plant growth:",
            # Line  4: IGRO (land cover status, 0=no crop growing)
            f"  {1:14d}    | IGRO : Land cover status (1=growing)",
            # Line  5: NCRP (land cover code -- ID in plant.dat)
            f"  {0:14d}    | NCRP : Land cover code from crop.dat",
            # Line  6: LAIDAY (leaf area index)
            f"  {0.0:14.4f}    | LAIDAY : Current leaf area index",
            # Line  7: BIO_MS (biomass, kg/ha)
            f"  {0.0:14.4f}    | BIO_MS : Biomass [kg/ha]",
            # Line  8: PHU_PLT (heat units to maturity)
            f"  {0.0:14.4f}    | PHU_PLT : Heat units to maturity",
            # Line  9: title
            " General management:",
            # Line 10: BIOMIX
            f"  {0.20:14.4f}    | BIOMIX : Biological mixing efficiency",
            # Line 11: CN2
            f"  {78.0:14.4f}    | CN2 : Initial SCS CN for moisture condition II",
            # Line 12: USLE_P
            f"  {1.0:14.4f}    | USLE_P : USLE support practice factor",
            # Line 13: BIO_MIN
            f"  {0.0:14.4f}    | BIO_MIN : Min biomass for grazing [kg/ha]",
            # Line 14: FILTERW
            f"  {0.0:14.4f}    | FILTERW : Width of edge-of-field filter strip [m]",
            # Line 15: title
            " Urban:",
            # Line 16: IURBAN
            f"  {0:14d}    | IURBAN : Urban simulation code",
            # Line 17: URBLU
            f"  {0:14d}    | URBLU : Urban land type",
            # Line 18: title
            " Irrigation:",
            # Line 19: IRRSC
            f"  {0:14d}    | IRRSC : Irrigation code",
            # Line 20: IRRNO
            f"  {0:14d}    | IRRNO : Irrigation source code",
            # Line 21: FLOWMIN
            f"  {0.0:14.4f}    | FLOWMIN : Min in-stream flow for irrigation [m3/s]",
            # Line 22: DIVMAX
            f"  {0.0:14.4f}    | DIVMAX : Max irrigation diversion [mm]",
            # Line 23: FLOWFR
            f"  {0.0:14.4f}    | FLOWFR : Fraction of flow allowed for diversion",
            # Line 24: title
            " Tile drain:",
            # Line 25: DDRAIN
            f"  {0.0:14.4f}    | DDRAIN : Depth to subsurface drain [mm]",
            # Line 26: TDRAIN
            f"  {0.0:14.4f}    | TDRAIN : Time to drain to field capacity [hr]",
            # Line 27: GDRAIN
            f"  {0.0:14.4f}    | GDRAIN : Drain tile lag time [hr]",
            # Line 28: title
            " Management operations:",
            # Line 29: title (was NROT in older versions, now just title)
            " Rotation:",
            # Line 30: title
            " Schedule:",
        ]
        mgt_path.write_text('\n'.join(mgt_lines) + '\n', encoding='utf-8')

        # --------------------------------------------------------------
        # Soil file (.sol) -- readsol.f reads with fixed-width formats
        # Line 1: title (a80)
        # Line 2: SNAM -- format (12x,a16)
        # Line 3: HYDGRP -- format (24x,a1)
        # Line 4: SOL_ZMX -- format (28x,f12.2)
        # Line 5: ANION_EXCL -- format (51x,f5.3)
        # Line 6: title (skip)
        # Lines 7+: data rows -- format (27x,15f12.2)
        #   Each row has a 27-char label then 12.2-float values per layer
        # Variables: SOL_Z, SOL_BD, SOL_AWC, SOL_K, SOL_CBN, CLAY,
        #   SILT, SAND, ROCK, SOL_ALB, USLE_K, SOL_EC, PH, CACO3
        # (14 data rows for 1 soil layer)
        # --------------------------------------------------------------
        sol_path = self.txtinout_dir / '000010001.sol'

        # Single soil layer values (mountain catchment defaults)
        sol_z = 500.0       # Depth to bottom of layer [mm]
        sol_bd = 1.45       # Moist bulk density [Mg/m3]
        sol_awc = 0.18      # Available water capacity [mm/mm]
        sol_k = 12.0        # Saturated hydraulic conductivity [mm/hr]
        sol_cbn = 1.5       # Organic carbon content [%]
        clay_pct = 15.0     # Clay content [%]
        silt_pct = 35.0     # Silt content [%]
        sand_pct = 50.0     # Sand content [%]
        rock_pct = 10.0     # Rock fragment content [%]
        sol_alb = 0.10      # Moist soil albedo
        usle_k = 0.28       # USLE soil erodibility factor
        sol_ec = 0.0        # Electrical conductivity [dS/m]
        ph = 6.5            # Soil pH
        caco3 = 0.0         # CaCO3 content [%]

        def sol_data_row(label, value):
            """Format a .sol data row: 27-char label + 12.2 float per layer."""
            return f"{label:27s}{value:12.2f}"

        sol_lines = [
            # Line 1: title (a80)
            " Soil data -- SYMFLUENCE generated                                             ",
            # Line 2: SNAM -- format (12x,a16) -- 12 skip chars then 16-char name
            f"{'':12s}{'MountainSoil':16s}",
            # Line 3: HYDGRP -- format (24x,a1) -- 24 skip chars then 1 char
            f"{'':24s}{'B':1s}",
            # Line 4: SOL_ZMX -- format (28x,f12.2) -- 28 skip chars then f12.2
            f"{'':28s}{sol_z:12.2f}",
            # Line 5: ANION_EXCL -- format (51x,f5.3) -- 51 skip chars then f5.3
            f"{'':51s}{0.500:5.3f}",
            # Line 6: title (skip line)
            " Soil layer data:",
            # Lines 7+: data rows -- format (27x,15f12.2)
            sol_data_row(" SOL_Z(mm)               :", sol_z),
            sol_data_row(" SOL_BD(Mg/m**3)         :", sol_bd),
            sol_data_row(" SOL_AWC(mm/mm)           :", sol_awc),
            sol_data_row(" SOL_K(mm/hr)            :", sol_k),
            sol_data_row(" SOL_CBN(%)              :", sol_cbn),
            sol_data_row(" CLAY(%)                 :", clay_pct),
            sol_data_row(" SILT(%)                 :", silt_pct),
            sol_data_row(" SAND(%)                 :", sand_pct),
            sol_data_row(" ROCK(%)                 :", rock_pct),
            sol_data_row(" SOL_ALB                 :", sol_alb),
            sol_data_row(" USLE_K                  :", usle_k),
            sol_data_row(" SOL_EC(dS/m)            :", sol_ec),
            sol_data_row(" PH                      :", ph),
            sol_data_row(" CACO3                   :", caco3),
        ]
        sol_path.write_text('\n'.join(sol_lines) + '\n', encoding='utf-8')

        logger.info(f"Generated sub-basin, HRU, GW, MGT, SOL files in {self.txtinout_dir}")

    # ------------------------------------------------------------------
    # Auxiliary file generators referenced by the .sub file
    # ------------------------------------------------------------------

    def _generate_wgn_file(self, props: Dict) -> None:
        """Generate a weather generator file (.wgn) for the sub-basin.

        ``readwgn.f`` reads the file on unit 114 with these formats:

        - Line 1: title (format a)
        - Line 2: station latitude  (format 12x,f7.2)
        - Line 3: station elevation (format 12x,f7.2)
        - Line 4: rain_yrs          (format 12x,f7.2)
        - Lines 5-18: 14 rows of 12 monthly values (format 12f6.2,
          except line 9 which uses 12f6.1 for pcpmm)

        The 14 monthly-data rows are (in order):
          TMPMX, TMPMN, TMPSTDMX, TMPSTDMN, PCPMM, PCPSTD, PCPSKW,
          PR_W1, PR_W2, PCPD, RAINHHMX, SOLARAV, DEWPT, WNDAV
        """
        wgn_path = self.txtinout_dir / '000010001.wgn'

        lat = props.get('lat', 51.0)
        elev = props.get('elev', 1000.0)

        # Reasonable monthly climate normals for a mid-latitude mountain
        # catchment (e.g. Canadian Rockies ~51N).  These are placeholder
        # values that keep the weather generator stable.
        # fmt: off
        tmpmx  = [ -5.0, -2.0,  3.0, 10.0, 16.0, 20.0, 23.0, 22.0, 17.0, 10.0,  2.0, -4.0]
        tmpmn  = [-15.0,-13.0, -8.0, -2.0,  3.0,  7.0,  9.0,  8.0,  4.0, -1.0, -8.0,-14.0]
        tmpstdmx = [6.0,  6.0,  5.5,  5.0,  4.5,  4.0,  3.5,  3.5,  4.5,  5.0,  5.5,  6.0]
        tmpstdmn = [6.5,  6.5,  5.5,  4.5,  3.5,  3.0,  2.5,  2.5,  3.5,  4.5,  5.5,  6.5]
        pcpmm  = [30.0, 25.0, 30.0, 40.0, 55.0, 70.0, 65.0, 55.0, 45.0, 35.0, 30.0, 30.0]
        pcpstd = [ 5.0,  4.5,  5.0,  6.0,  7.0,  8.0,  8.5,  7.5,  6.5,  5.5,  5.0,  5.0]
        pcpskw = [ 1.5,  1.5,  1.5,  1.2,  1.0,  0.8,  0.8,  1.0,  1.2,  1.5,  1.5,  1.5]
        pr_w1  = [ 0.15, 0.15, 0.18, 0.22, 0.25, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15, 0.15]
        pr_w2  = [ 0.55, 0.55, 0.58, 0.60, 0.62, 0.65, 0.62, 0.60, 0.58, 0.55, 0.55, 0.55]
        pcpd   = [ 8.0,  7.0,  9.0, 10.0, 12.0, 13.0, 11.0, 10.0,  9.0,  8.0,  8.0,  8.0]
        rainhhmx = [10.0, 10.0, 12.0, 15.0, 20.0, 25.0, 28.0, 25.0, 20.0, 15.0, 12.0, 10.0]
        solarav = [ 6.0,  9.0, 13.0, 17.0, 20.0, 22.0, 22.0, 19.0, 15.0, 10.0,  7.0,  5.0]
        dewpt  = [-18.0,-16.0,-11.0, -4.0,  1.0,  5.0,  7.0,  6.0,  2.0, -3.0,-10.0,-17.0]
        wndav  = [ 3.5,  3.5,  3.8,  4.0,  3.5,  3.2,  2.8,  2.8,  3.0,  3.5,  3.5,  3.5]
        # fmt: on

        def fmt_12f62(vals):
            """Format 12 monthly values in Fortran 12f6.2."""
            return ''.join(f"{v:6.2f}" for v in vals)

        def fmt_12f61(vals):
            """Format 12 monthly values in Fortran 12f6.1."""
            return ''.join(f"{v:6.1f}" for v in vals)

        lines = [
            " Weather generator data -- SYMFLUENCE generated",
            f"  LATI (deg){lat:7.2f}",
            f"  ELEV (m)  {elev:7.2f}",
            f"  RAIN_YRS  {10.00:7.2f}",
            fmt_12f62(tmpmx),       # TMPMX
            fmt_12f62(tmpmn),       # TMPMN
            fmt_12f62(tmpstdmx),    # TMPSTDMX
            fmt_12f62(tmpstdmn),    # TMPSTDMN
            fmt_12f61(pcpmm),       # PCPMM (format 12f6.1)
            fmt_12f62(pcpstd),      # PCPSTD
            fmt_12f62(pcpskw),      # PCPSKW
            fmt_12f62(pr_w1),       # PR_W1
            fmt_12f62(pr_w2),       # PR_W2
            fmt_12f62(pcpd),        # PCPD
            fmt_12f62(rainhhmx),    # RAINHHMX
            fmt_12f62(solarav),     # SOLARAV
            fmt_12f62(dewpt),       # DEWPT
            fmt_12f62(wndav),       # WNDAV
        ]
        wgn_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Weather generator file written: {wgn_path}")

    def _generate_pnd_stub(self) -> None:
        """Generate a minimal pond/wetland stub file (.pnd).

        ``readpnd.f`` reads every value with ``iostat=eof``, so an
        empty or minimal file is safe -- EOF stops the reads
        gracefully and all pond variables retain their default (zero)
        values.
        """
        pnd_path = self.txtinout_dir / '000010001.pnd'
        lines = [
            " Pond/Wetland data -- SYMFLUENCE stub (no ponds/wetlands)",
            " Pond data:",
        ]
        pnd_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Pond stub file written: {pnd_path}")

    def _generate_wus_stub(self) -> None:
        """Generate a minimal water-use stub file (.wus).

        ``readwus.f`` reads with ``iostat=eof``; an empty/minimal
        file is safe.
        """
        wus_path = self.txtinout_dir / '000010001.wus'
        lines = [
            " Water Use data -- SYMFLUENCE stub (no external water use)",
        ]
        wus_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Water use stub file written: {wus_path}")

    def _generate_chm_stub(self) -> None:
        """Generate a minimal soil-chemistry stub file (.chm).

        ``hruallo.f`` opens the .chm file and reads groups of 11
        header lines then pesticide IDs until EOF.  A minimal file
        with just a title line causes the ``iostat=eof`` to fire
        immediately, which is the correct behaviour for a model with
        no pesticides.
        """
        chm_path = self.txtinout_dir / '000010001.chm'
        lines = [
            " Soil chemistry data -- SYMFLUENCE stub (no pesticides)",
        ]
        chm_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Soil chemistry stub file written: {chm_path}")

    def _generate_file_cio(self, start_date: datetime, end_date: datetime) -> None:
        """
        Generate SWAT master control file (file.cio).

        The file.cio is read line-by-line by SWAT's getallo.f and readfile.f
        routines. Each line must be in exactly the right position. Values use
        the first 16 columns (right-justified numbers or left-justified
        filenames) followed by '| VARNAME : description'.

        Filenames are read with Fortran format (6a) which reads the first
        ~13 characters. Numeric values are read with list-directed I/O
        (read *) which extracts the first valid number from the line.
        """
        warmup_years = self._get_config_value(
            lambda: self.config.model.swat.warmup_years,
            default=2
        )

        nbyr = end_date.year - start_date.year + 1
        iyr = start_date.year
        idaf = start_date.timetuple().tm_yday
        idal = end_date.timetuple().tm_yday

        # Helper: format a filename value line (left-justified, 13 chars).
        # Fortran format (6a) reads first 13 chars as the filename.
        def fline(fname, comment):
            return f"{fname:13s} | {comment}"

        # Helper: format an integer value line (right-justified, 16 chars).
        # Fortran list-directed read (*) extracts the first valid number.
        def iline(val, comment):
            return f"{val:16d}    | {comment}"

        # Helper: format a float value line (right-justified, 16 chars).
        def rline(val, comment):
            return f"{val:16.3f}    | {comment}"

        # Helper: format a gage file array line.
        # Fortran format (6a) with character(len=13) reads 6 filenames
        # of 13 chars each per record.  For 18 filenames that is 3 lines.
        def gage_file_lines(filenames):
            """Return 3 lines of 6 x 13-char filename slots (total 18)."""
            padded = list(filenames) + [''] * (18 - len(filenames))
            result = []
            for row in range(3):
                parts = padded[row * 6:(row + 1) * 6]
                result.append(''.join(f"{p:13s}" for p in parts))
            return result

        cio_path = self.txtinout_dir / 'file.cio'

        # ----------------------------------------------------------------
        # Build file.cio line by line matching readfile.f / getallo.f read
        # order.  Line numbers (FL) are the sequential file-line positions
        # consumed by both Fortran routines reading the same file.
        #
        # CRITICAL: readfile.f reads rfile(1:18) and tfile(1:18) with
        # Fortran format (6a).  With character(len=13) variables and 18
        # items, this consumes 3 records (lines) per array -- 6 filenames
        # of 13 chars each per line.  getallo.f simply skips all these
        # lines with read(23,6000) titldum, so the line count must match.
        # ----------------------------------------------------------------
        lines = []

        # --- Title lines (FL 1-5) ---
        # Both getallo.f and readfile.f use `use parm` where
        # title is declared as character(len=4) :: title(60).
        # Format 5100 = (20a4) reads 20 items per record, so
        # 60 items = 3 records (lines).  Both routines read:
        #   read(unit,6000/5101) titldum   -> FL 1
        #   read(unit,6000/5101) titldum   -> FL 2
        #   read(unit,5100) title          -> FL 3, FL 4, FL 5
        lines.append(" Master watershed file: file.cio")                     # FL 1
        lines.append(" SYMFLUENCE auto-generated SWAT input")               # FL 2
        lines.append(" Simulation generated by SYMFLUENCE preprocessor")    # FL 3
        lines.append("")                                                     # FL 4
        lines.append("")                                                     # FL 5

        # --- General Information (FL 6-11) ---
        lines.append(" General Information/Watershed Configuration:")        # FL 6
        lines.append(fline("fig.fig", "FIGFILE"))                           # FL 7
        lines.append(iline(nbyr, "NBYR : Number of years simulated"))       # FL 8
        lines.append(iline(iyr, "IYR : Beginning year of simulation"))      # FL 9
        lines.append(iline(idaf, "IDAF : Beginning julian day"))            # FL 10
        lines.append(iline(idal, "IDAL : Ending julian day"))               # FL 11

        # --- Climate (FL 12-33) ---
        lines.append(" Climate:")                                            # FL 12
        lines.append(iline(0, "IGEN : Random generator seed"))              # FL 13
        lines.append(iline(1, "PCPSIM : Precip input (1=measured)"))        # FL 14
        lines.append(iline(0, "IDT : Sub-daily timestep (0=daily)"))        # FL 15
        lines.append(iline(0, "IDIST : Rainfall distribution code"))        # FL 16
        lines.append(rline(0.0, "REXP : Mixed exponential exponent"))       # FL 17
        lines.append(iline(1, "NRGAGE : Number of precip gage files"))      # FL 18
        lines.append(iline(1, "NRTOT : Total number of precip gages"))      # FL 19
        lines.append(iline(1, "NRGFIL : Precip gages per file"))            # FL 20
        lines.append(iline(1, "TMPSIM : Temp input (1=measured)"))          # FL 21
        lines.append(iline(1, "NTGAGE : Number of temp gage files"))        # FL 22
        lines.append(iline(1, "NTTOT : Total number of temp gages"))        # FL 23
        lines.append(iline(1, "NTGFIL : Temp gages per file"))              # FL 24
        lines.append(iline(2, "SLRSIM : Solar rad (2=simulated)"))          # FL 25
        lines.append(iline(0, "NSTOT : Number of solar rad records"))       # FL 26
        lines.append(iline(2, "RHSIM : Rel humidity (2=simulated)"))        # FL 27
        lines.append(iline(0, "NHTOT : Number of humidity records"))        # FL 28
        lines.append(iline(2, "WNDSIM : Wind speed (2=simulated)"))         # FL 29
        lines.append(iline(0, "NWTOT : Number of wind speed records"))      # FL 30
        lines.append(iline(0, "FCSTYR : Forecast begin year (0=off)"))      # FL 31
        lines.append(iline(0, "FCSTDAY : Forecast begin julian day"))       # FL 32
        lines.append(iline(0, "FCSTCYCLES : Forecast cycles"))              # FL 33

        # --- Precipitation Input Files (FL 34-37) ---
        lines.append(" Precipitation Input Files:")                          # FL 34
        lines.extend(gage_file_lines(["pcp1.pcp"]))                         # FL 35-37

        # --- Temperature Input Files (FL 38-41) ---
        lines.append(" Temperature Input Files:")                            # FL 38
        lines.extend(gage_file_lines(["tmp1.tmp"]))                         # FL 39-41

        # --- Solar / Humidity / Wind / Forecast files (FL 42-45) ---
        lines.append(fline("", "SLRFILE : Solar radiation file"))           # FL 42
        lines.append(fline("", "RHFILE : Relative humidity file"))          # FL 43
        lines.append(fline("", "WNDFILE : Wind speed file"))               # FL 44
        lines.append(fline("", "FCSTFILE : Weather forecast file"))         # FL 45

        # --- Watershed Modeling Options (FL 46-47) ---
        lines.append(" Watershed Modeling Options:")                         # FL 46
        lines.append(fline("basins.bsn", "BSNFILE : Basin input file"))    # FL 47

        # --- Database Files (FL 48-53) ---
        lines.append(" Database Files:")                                     # FL 48
        lines.append(fline("plant.dat", "PLANTDB : Plant database"))        # FL 49
        lines.append(fline("till.dat", "TILLDB : Tillage database"))        # FL 50
        lines.append(fline("pest.dat", "PESTDB : Pesticide database"))      # FL 51
        lines.append(fline("fert.dat", "FERTDB : Fertilizer database"))     # FL 52
        lines.append(fline("urban.dat", "URBANDB : Urban database"))        # FL 53

        # --- Special Projects (FL 54-57) ---
        lines.append(" Special Projects:")                                   # FL 54
        lines.append(iline(0, "ISPROJ : Special project code"))             # FL 55
        lines.append(iline(0, "ICLB : Auto-calibration flag"))             # FL 56
        lines.append(fline("", "CALFILE : Calibration input file"))         # FL 57

        # --- Output Information (FL 58-63) ---
        lines.append(" Output Information:")                                 # FL 58
        lines.append(iline(1, "IPRINT : Output (0=mon,1=day,2=yr)"))       # FL 59
        lines.append(iline(warmup_years, "NYSKIP : Years to skip"))         # FL 60
        lines.append(iline(0, "ILOG : Streamflow output code"))             # FL 61
        lines.append(iline(0, "IPRP : Pesticide output code"))             # FL 62
        lines.append(" Reach (output.rch) print-frequency option:")          # FL 63

        # --- Output Variable Selection (FL 64-71) ---
        lines.append(" Reach output variables:")                             # FL 64
        lines.append(" 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")          # FL 65
        lines.append(" Subbasin output variables:")                          # FL 66
        lines.append(" 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")                     # FL 67
        lines.append(" HRU output variables:")                               # FL 68
        lines.append(" 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")          # FL 69
        lines.append(" HRU printout selection:")                             # FL 70
        lines.append(" 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")          # FL 71

        # --- Atmospheric Deposition (FL 72-73) ---
        lines.append(" Atmospheric deposition:")                             # FL 72
        lines.append(fline("", "ATMOFILE : Atmospheric deposition"))        # FL 73

        # --- Additional optional outputs (FL 74-77) ---
        lines.append(iline(0, "IPHR : Hourly output (0=off)"))              # FL 74
        lines.append(iline(0, "ISTO : Soil storage output (0=off)"))        # FL 75
        lines.append(iline(0, "ISOL : Soil nutrient output (0=off)"))       # FL 76
        lines.append(iline(0, "I_SUBHW : Headwater routing (0=off)"))       # FL 77

        # --- Septic Database (FL 78) ---
        lines.append(fline("", "SEPTDB : Septic database file"))            # FL 78

        # --- Binary output flag (FL 79) ---
        lines.append(iline(0, "IA_B : ASCII(0) or Binary(1) output"))       # FL 79

        # --- Additional optional flags (FL 80+) ---
        lines.append(iline(0, "IHUMUS : Water quality output (0=off)"))     # FL 80
        lines.append(iline(0, "ITEMP : Velocity/depth output (0=off)"))     # FL 81
        lines.append(iline(0, "ISNOW : Snowband output (0=off)"))           # FL 82
        lines.append(iline(0, "IMGT : Management output (0=off)"))          # FL 83
        lines.append(iline(0, "IWTR : Water balance output (0=off)"))       # FL 84
        lines.append(iline(0, "ICALEN : Calendar day (0=julian)"))          # FL 85

        cio_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Master control file written: {cio_path}")

    def _generate_fig_file(self) -> None:
        """
        Generate SWAT watershed routing file (fig.fig).

        The fig.fig file defines the routing structure. For a lumped
        (single sub-basin) model, it contains a single SUBBASIN command
        followed by an END command.
        """
        fig_path = self.txtinout_dir / 'fig.fig'
        # Format: a1, 9x, 5i6 per line (format 5001 in getallo.f)
        # Col 1: comment flag (space = active, * = comment)
        # Col 2-10: padding
        # Col 11-16: icd (command code, 1=subbasin)
        # Col 17-22: iht (hydrograph storage location)
        # Col 23-28: inm1
        # Col 29-34: inm2
        # Col 35-40: inm3 (subbasin number if icd=1)
        lines = []
        # Subbasin command: icd=1, iht=1, inm1=0, inm2=1, inm3=1
        lines.append(f" {'':9s}{1:6d}{1:6d}{0:6d}{1:6d}{1:6d}")
        # Sub-basin file reference line (format 6100: 10x, a13)
        lines.append(f"{'':10s}{'000010001.sub':13s}")
        # End command: icd=0 signals end of routing
        lines.append(f" {'':9s}{0:6d}{0:6d}{0:6d}{0:6d}{0:6d}")

        fig_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Routing file written: {fig_path}")

    def _generate_database_stubs(self) -> None:
        """
        Generate minimal SWAT database stub files.

        SWAT requires plant.dat, till.dat, pest.dat, fert.dat, and
        urban.dat to be present. These minimal stubs contain a single
        dummy record so that getallo.f can parse them without error.
        Database files only need to exist; the single-HRU lumped model
        uses default parameters.
        """
        # plant.dat (crop database): record format = id, then 4 lines of data
        plant_path = self.txtinout_dir / 'plant.dat'
        plant_lines = [
            "   1",
            "AGRL         WARM SEASON ANNUAL LEGUME",
            "   0.000   0.000   0.000   0.000   0.000   0.000   0.000",
            "   0.000   0.000   0.000   0.000   0.000   0.000   0.000",
            "   0.000   0.000   0.000   0.000   0.000   0.000   0.000",
        ]
        plant_path.write_text('\n'.join(plant_lines) + '\n', encoding='utf-8')

        # till.dat (tillage database): format 6300 = i4
        till_path = self.txtinout_dir / 'till.dat'
        till_path.write_text("   1  Generic tillage\n", encoding='utf-8')

        # pest.dat (pesticide database): format 6200 = i3
        pest_path = self.txtinout_dir / 'pest.dat'
        pest_path.write_text("  1  Generic pesticide\n", encoding='utf-8')

        # fert.dat (fertilizer database): format 6300 = i4
        fert_path = self.txtinout_dir / 'fert.dat'
        fert_path.write_text("   1  Generic fertilizer\n", encoding='utf-8')

        # urban.dat (urban land type database): format 6200 = i3
        urban_path = self.txtinout_dir / 'urban.dat'
        urban_lines = [
            "  1",
            "Urban land type 1",
        ]
        urban_path.write_text('\n'.join(urban_lines) + '\n', encoding='utf-8')

        logger.info(f"Database stub files written to {self.txtinout_dir}")

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
