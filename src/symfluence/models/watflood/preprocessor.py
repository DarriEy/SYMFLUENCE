"""
WATFLOOD Pre-Processor.

Generates a complete WATFLOOD/CHARM input file suite from ERA5 forcing
for a lumped single-cell basin:
  - Watershed definition  (_shd.r2c)
  - Parameter file        (.par)
  - Event files           (.evt)  — one per month, chained
  - Forcing files         (.rag / .tag)  — one per month
  - Output spec           (wfo_spec.txt)
  - Streamflow obs        (_str.tb0)
  - Directory structure   (basin/, event/, raing/, tempg/, strfw/, results/, debug/)
"""

import calendar
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor

logger = logging.getLogger(__name__)


class WATFLOODPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Pre-processor for WATFLOOD model setup (lumped 1-cell basin)."""

    MODEL_NAME = "WATFLOOD"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.watflood_dir = self.project_dir / 'WATFLOOD_input'
        self.settings_dir = self.watflood_dir / 'settings'

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run_preprocessing(self) -> bool:
        """Generate all WATFLOOD input files from scratch."""
        try:
            # Create directory tree
            for d in ('basin', 'event', 'raing', 'tempg', 'strfw',
                      'results', 'debug', 'moist', 'snow1'):
                (self.settings_dir / d).mkdir(parents=True, exist_ok=True)

            start, end = self._get_simulation_dates()
            logger.info(f"WATFLOOD preprocessing: {start:%Y-%m-%d} to {end:%Y-%m-%d}")

            # Load ERA5 forcing
            hourly = self._load_era5_forcing(start, end)

            # 1. Watershed definition
            self._generate_shd_file()

            # 2. Parameter file
            self._generate_par_file()

            # 3. Monthly forcing + event files
            self._generate_monthly_files(hourly, start, end)

            # 4. Output spec
            self._generate_wfo_spec()

            # 5. Observation streamflow (for WATFLOOD stats)
            self._generate_streamflow_tb0(start, end)

            logger.info(f"WATFLOOD preprocessing complete: {self.settings_dir}")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"WATFLOOD preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------
    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        start = self._get_config_value(
            lambda: self.config.domain.time_start, default='2002-01-01')
        end = self._get_config_value(
            lambda: self.config.domain.time_end, default='2009-12-31')
        if isinstance(start, str):
            start = pd.Timestamp(start).to_pydatetime()
        if isinstance(end, str):
            end = pd.Timestamp(end).to_pydatetime()
        return start, end

    # ------------------------------------------------------------------
    # ERA5 loading
    # ------------------------------------------------------------------
    def _load_era5_forcing(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Load ERA5 basin-averaged forcing → hourly P (mm) and T (°C)."""
        forcing_path = self.forcing_basin_path
        if not forcing_path.exists():
            raise FileNotFoundError(f"Forcing not found: {forcing_path}")

        forcing_files = sorted(forcing_path.glob("*.nc"))
        if not forcing_files:
            raise FileNotFoundError(f"No NetCDF files in {forcing_path}")

        logger.info(f"Loading ERA5 forcing ({len(forcing_files)} files)")
        try:
            ds = xr.open_mfdataset(forcing_files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override')
        except Exception:  # noqa: BLE001 — model execution resilience
            datasets = [xr.open_dataset(f) for f in forcing_files]
            ds = xr.concat(datasets, dim='time')

        ds = ds.sel(time=slice(str(start), str(end)))

        airtemp = ds['airtemp'].values.squeeze()   # K
        pptrate = ds['pptrate'].values.squeeze()    # mm/s
        times = pd.DatetimeIndex(ds['time'].values)

        hourly = pd.DataFrame({
            'temp_C': airtemp - 273.15,
            'precip_mm': pptrate * 3600.0,
        }, index=times)
        ds.close()

        logger.info(f"ERA5: {len(hourly)} hours, "
                     f"P [{hourly['precip_mm'].min():.2f}–{hourly['precip_mm'].max():.2f}] mm/h, "
                     f"T [{hourly['temp_C'].min():.1f}–{hourly['temp_C'].max():.1f}] °C")
        return hourly

    # ------------------------------------------------------------------
    # 1. Watershed definition  (_shd.r2c)
    # ------------------------------------------------------------------
    def _generate_shd_file(self) -> None:
        """Generate a lumped watershed definition in r2c format.

        Uses a 3x3 grid with only the center cell (row=2, col=2) active.
        Data is written as 2D grid blocks matching the standard EnSim r2c
        format that CHARM expects:
          rank, next, DA, bankfull, slope, elevation, channel_length,
          IAK, int_slope, chnl, reach, then one grid per land class.
        """
        area_km2 = 2210.0
        cell_m = 5000.0   # 5 km cells
        elev = 1600.0
        x_origin = 560000.0
        y_origin = 5670000.0
        da = area_km2 / ((cell_m / 1000.0) ** 2)  # ~88.4 grid units
        bankfull = 20.0
        slope = 0.005
        ch_len = cell_m
        nc = 3  # grid dimension

        def _grid_line(vals):
            """Format one row of a 3x3 grid."""
            return ' '.join(f'{v:5d}' for v in vals) + '\n'

        def _grid_line_f(vals, fmt='.7E'):
            """Format one row of a 3x3 float grid."""
            return ' '.join(f' {v:{fmt}}' for v in vals) + ' \n'

        def _grid_line_fx(vals, fmt='10.3f'):
            """Format one row of a 3x3 float grid (fixed)."""
            return ' '.join(f'{v:{fmt}}' for v in vals) + ' \n'

        # Active cell is (row=2, col=2) in 1-indexed → index (1,1) in 0-indexed
        z3 = [0, 0, 0]

        out = self.settings_dir / 'basin' / 'bow_shd.r2c'
        with open(out, 'w') as f:
            # Header
            f.write("########################################\n")
            f.write(":FileType r2c  ASCII  EnSim 1.0         \n")
            f.write("#                                       \n")
            f.write("# DataType               2D Rect Cell   \n")
            f.write("#                                       \n")
            f.write(":Application             EnSimHydrologic\n")
            f.write(":Version                 2.1.23         \n")
            f.write(":WrittenBy          SYMFLUENCE          \n")
            f.write(":CreationDate       2026-01-01  00:00\n")
            f.write("#                                       \n")
            f.write(":SourceFileName                bow.map  \n")
            f.write(f":NominalGridSize_AL     {cell_m:.3f}\n")
            f.write(":ContourInterval           1.000\n")
            f.write(":ImperviousArea            0.000\n")
            f.write(":ClassCount                    1\n")
            f.write(":NumRiverClasses               1\n")
            f.write(":ElevConversion            1.000\n")
            f.write(":TotalNumOfGrids               1\n")
            f.write(":numGridsInBasin               1\n")
            f.write(":DebugGridNo                   1\n")
            f.write("#                                       \n")
            f.write(":Projection         CARTESIAN \n")
            f.write(":Ellipsoid          unknown   \n")
            f.write("#                                       \n")
            f.write(f":xOrigin              {x_origin:.6f}\n")
            f.write(f":yOrigin             {y_origin:.6f}\n")
            f.write("#                                       \n")
            f.write(":AttributeName 1 Rank         \n")
            f.write(":AttributeName 2 Next         \n")
            f.write(":AttributeName 3 DA           \n")
            f.write(":AttributeName 4 Bankfull     \n")
            f.write(":AttributeName 5 ChnlSlope    \n")
            f.write(":AttributeName 6 Elev         \n")
            f.write(":AttributeName 7 ChnlLength   \n")
            f.write(":AttributeName 8 IAK          \n")
            f.write(":AttributeName 9 IntSlope     \n")
            f.write(":AttributeName 10 Chnl        \n")
            f.write(":AttributeName 11 Reach       \n")
            f.write(":AttributeName 12 conifer     \n")
            f.write("#                                       \n")
            f.write(f":xCount                       {nc}\n")
            f.write(f":yCount                       {nc}\n")
            f.write(f":xDelta                 {cell_m:.6f}\n")
            f.write(f":yDelta                 {cell_m:.6f}\n")
            f.write("#                                       \n")
            f.write(":EndHeader                              \n")

            # 1. Rank grid (3x3 integers) — cell (2,2) = 1
            f.write(_grid_line(z3))
            f.write(_grid_line([0, 1, 0]))
            f.write(_grid_line(z3))

            # 2. Next grid (downstream cell) — 0 everywhere (outlet)
            for _ in range(nc):
                f.write(_grid_line(z3))

            # 3. DA grid (drainage area in grid units, scientific notation)
            z3f = [0.0, 0.0, 0.0]
            f.write(_grid_line_f(z3f))
            f.write(_grid_line_f([0.0, da, 0.0]))
            f.write(_grid_line_f(z3f))

            # 4. Bankfull grid (fixed format)
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))
            f.write(_grid_line_fx([0.0, bankfull, 0.0]))
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))

            # 5. Channel slope grid
            f.write(_grid_line_fx([0.0, 0.0, 0.0], '10.7f'))
            f.write(_grid_line_fx([0.0, slope, 0.0], '10.7f'))
            f.write(_grid_line_fx([0.0, 0.0, 0.0], '10.7f'))

            # 6. Elevation grid
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))
            f.write(_grid_line_fx([0.0, elev, 0.0]))
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))

            # 7. Channel length grid
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))
            f.write(_grid_line_fx([0.0, ch_len, 0.0]))
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))

            # 8. IAK grid (interflow active key: 1=active)
            f.write(_grid_line(z3))
            f.write(_grid_line([0, 1, 0]))
            f.write(_grid_line(z3))

            # 9. Internal slope grid
            f.write(_grid_line_fx([0.0, 0.0, 0.0], '10.7f'))
            f.write(_grid_line_fx([0.0, slope, 0.0], '10.7f'))
            f.write(_grid_line_fx([0.0, 0.0, 0.0], '10.7f'))

            # 10. Channel class grid (1=river)
            f.write(_grid_line(z3))
            f.write(_grid_line([0, 1, 0]))
            f.write(_grid_line(z3))

            # 11. Reach grid (0=no reach)
            for _ in range(nc):

                f.write(_grid_line(z3))

            # 12. Land class fraction: conifer = 1.0 at active cell
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))
            f.write(_grid_line_fx([0.0, 1.0, 0.0]))
            f.write(_grid_line_fx([0.0, 0.0, 0.0]))

        logger.info(f"Wrote watershed file: {out}")

    # ------------------------------------------------------------------
    # 2. Parameter file (.par)
    # ------------------------------------------------------------------
    def _generate_par_file(self) -> None:
        """Generate WATFLOOD .par file for lumped 1-class basin.

        Format mirrors the demo.par structure exactly — CHARM's Fortran parser
        is positional and expects parameters in a specific order with the
        correct number of per-class values on each line.
        """
        out = self.settings_dir / 'basin' / 'bow.par'
        lat = 51.17

        with open(out, 'w') as f:
            # Header / global parameters
            f.write("# WATFLOOD parameter file generated by SYMFLUENCE\n")
            f.write("#       Bow at Banff — lumped 1-cell, 1-class\n")
            f.write("ver       9.300     parameter file version number\n")
            f.write("iopt          0     debug level\n")
            f.write("itype         0\n")
            f.write("numa          0     optimization 0=no 1=yes\n")
            f.write("nper          1     opt delta 1-absolute\n")
            f.write("kc            5     no of times delta halved\n")
            f.write("maxn       2000     max no of trials\n")
            f.write("ddsfl         0     DDS optimization 0=no 1=yes\n")
            f.write("itrce       100\n")
            f.write("iiout       100\n")
            f.write("typeo         1     no of land classes optimized\n")
            f.write("nbsn          1     no of river classes optimized\n")
            # Global coefficients
            f.write("a1     -999.999\n")
            f.write("a2     -999.999\n")
            f.write("a3     -999.999\n")
            f.write("a4     -999.999\n")
            f.write("a5        0.983     API coefficient\n")
            f.write("a6      900.000     Minimum routing time step in seconds\n")
            f.write("a7        0.750     weighting factor - old vs. new sca value\n")
            f.write("a8        0.000     min temperature time offset\n")
            f.write("a9        0.500\n")
            f.write("a10       1.500\n")
            f.write("a11    -999.999\n")
            f.write("a12       0.000     min precip rate for smearing\n")
            # River class parameters (meander section)
            f.write("      meander\n")
            f.write("lzf   0.100E-04\n")
            f.write("pwr   0.200E+01\n")
            f.write("r1n   0.100E+00\n")
            f.write("R2n   0.400E-01\n")
            f.write("mndr  0.120E+01\n")
            f.write("aa2   0.110E+00\n")
            f.write("aa3   0.430E-01\n")
            f.write("aa4   0.100E+01\n")
            f.write("theta 0.500E+00\n")
            f.write("widep 0.200E+02\n")
            f.write("kcond 0.100E+01\n")
            # Per-class parameters — 1 landcover class (conifer)
            f.write("     conifer\n")
            f.write("ds    0.500E+01\n")
            f.write("dsfs  0.500E+01\n")
            f.write("Re    0.300E+00\n")
            f.write("AK    0.300E+02\n")
            f.write("AKfs  0.200E+02\n")
            f.write("retn  0.100E+03\n")
            f.write("ak2   0.500E-01\n")
            f.write("ak2fs 0.100E-01\n")
            f.write("R3    0.300E+02\n")
            f.write("r3fs  0.300E+02\n")
            f.write("r4    0.100E+02\n")
            f.write("ch    0.800E+00\n")
            f.write("MF    0.900E-01\n")
            f.write("BASE -0.100E+01\n")
            f.write("NMF   0.000E+00\n")
            f.write("UADJ  0.000E+00\n")
            f.write("TIPM  0.100E+00\n")
            f.write("RHO   0.333E+00\n")
            f.write("WHCL  0.350E-01\n")
            # ET and climate (per-class values — 1 class)
            f.write("fmadj     0.000\n")
            f.write("fmlow     0.000\n")
            f.write("fmhgh     0.000\n")
            f.write("gladj     0.000\n")
            f.write("rlaps     0.000\n")
            f.write("elvrf     0.000\n")
            f.write("flgev      2.00       2 = Hargreaves\n")
            f.write("albed      1.00\n")
            f.write("allw       1.00\n")
            f.write("fpet       3.00\n")
            f.write("ftal       0.50\n")
            f.write("flint        1.\n")
            f.write("fcap       0.25\n")
            f.write("ffcap      0.10\n")
            f.write("spore      0.40\n")
            f.write("tempa       40.\n")
            f.write("tempa       50.\n")
            f.write("tempa      500.\n")
            f.write("tton       200.\n")
            f.write(f"lat.    {lat:.1f}\n")
            # Monthly temperature range (Bow at Banff climate normals)
            f.write("dif-m 10.0 11.0 11.5 11.0 12.0 12.5 13.0 13.0 11.5 10.0  9.0  9.5\n")
            # Monthly humidity (approximate)
            f.write("humid 65.0 60.0 55.0 50.0 55.0 60.0 55.0 55.0 60.0 65.0 70.0 68.0\n")
            # Monthly mean pressure (kPa at ~1600m elevation)
            f.write("meanp 83.5 83.5 83.4 83.3 83.2 83.0 83.2 83.3 83.3 83.2 83.3 83.4\n")
            # Monthly vegetation height/interception parameters
            f.write("ti2    jan  feb  mar  apr  may  jun  jul  aug  sep  oct  nov  dec\n")
            f.write("h1    1.80 1.80 1.80 1.80 1.80 1.80 1.80 1.80 1.80 1.80 1.80 1.80\n")

        logger.info(f"Wrote parameter file: {out}")

    # ------------------------------------------------------------------
    # 3. Monthly event + forcing files
    # ------------------------------------------------------------------
    def _generate_monthly_files(self, hourly: pd.DataFrame,
                                start: datetime, end: datetime) -> None:
        """Generate per-month .evt, .rag, .tag files."""
        # Coordinate info for forcing headers (UTM zone 11N, km)
        y_km = 5670
        x_km = 560
        ymin, ymax = y_km, y_km + 15  # 3 cells * 5km = 15km span
        xmin, xmax = x_km, x_km + 15

        # Build list of months
        months = pd.date_range(start, end, freq='MS')
        logger.info(f"Generating {len(months)} monthly event files")

        evt_files = []
        for i, month_start in enumerate(months):
            year = month_start.year
            month = month_start.month
            ndays = calendar.monthrange(year, month)[1]
            nhours = ndays * 24
            datestr = f"{year:04d}{month:02d}01"

            # Extract this month's hourly data
            month_end = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta('23:59:59')
            mdata = hourly.loc[month_start:month_end]

            if len(mdata) == 0:
                logger.warning(f"No data for {datestr}, skipping")
                continue

            # Pad/trim to exact nhours
            precip_vals = mdata['precip_mm'].values[:nhours]
            temp_vals = mdata['temp_C'].values[:nhours]
            if len(precip_vals) < nhours:
                precip_vals = np.pad(precip_vals, (0, nhours - len(precip_vals)),
                                     constant_values=0.0)
                temp_vals = np.pad(temp_vals, (0, nhours - len(temp_vals)),
                                   constant_values=temp_vals[-1] if len(temp_vals) > 0 else 0.0)

            # Write .rag file (precipitation)
            rag_path = self.settings_dir / 'raing' / f'{datestr}.rag'
            with open(rag_path, 'w') as f:
                f.write(f"    2 {ymin} {ymax}  {xmin}  {xmax}\n")
                f.write(f"    1  {nhours} 1.00\n")
                # 1 station at basin centroid (y, x order per WATFLOOD convention)
                f.write(f" {y_km + 7}  {x_km + 7} SYMFLUENCE\n")
                for h in range(nhours):
                    f.write(f"    {precip_vals[h]:.2f}\n")

            # Write .tag file (temperature)
            tag_path = self.settings_dir / 'tempg' / f'{datestr}.tag'
            with open(tag_path, 'w') as f:
                f.write(f"    2 {ymin} {ymax}  {xmin}  {xmax}\n")
                f.write(f"    1  {nhours}    1\n")
                f.write(f" {y_km + 7}  {x_km + 7} SYMFLUENCE\n")
                for h in range(nhours):
                    f.write(f"    {temp_vals[h]:.2f}\n")

            # Write .evt file
            evt_path = self.settings_dir / 'event' / f'{datestr}.evt'
            is_last = (i == len(months) - 1)
            next_month = months[i + 1] if not is_last else None
            self._write_evt_file(evt_path, year, month, nhours,
                                 datestr, is_last, next_month)
            evt_files.append(evt_path)

        # Write the master event.evt pointing to the first month
        if evt_files:
            first_datestr = f"{months[0].year:04d}{months[0].month:02d}01"
            master_evt = self.settings_dir / 'event' / 'event.evt'
            first_month = months[0]
            ndays_first = calendar.monthrange(first_month.year, first_month.month)[1]
            nhours_first = ndays_first * 24
            self._write_evt_file(master_evt, first_month.year, first_month.month,
                                 nhours_first, first_datestr,
                                 len(months) <= 1,
                                 months[1] if len(months) > 1 else None)

        logger.info(f"Wrote {len(evt_files)} monthly event/forcing files")

    def _write_evt_file(self, path: Path, year: int, month: int,
                        nhours: int, datestr: str,
                        is_last: bool, next_month) -> None:
        """Write a single .evt file."""
        with open(path, 'w') as f:
            f.write("#\n")
            f.write(":filetype                     .evt\n")
            f.write(":fileversionno                9.4\n")
            f.write(f":year                         {year}\n")
            f.write(f":month                        {month:02d}\n")
            f.write(":day                          01\n")
            f.write(":hour                          0\n")
            f.write("#\n")
            f.write(":snwflg                       y\n")
            f.write(":sedflg                       n\n")
            f.write(":vapflg                       y\n")
            f.write(":smrflg                       n\n")
            f.write(":resinflg                     n\n")
            f.write(":tbcflg                       n\n")
            f.write(":resumflg                     n\n")
            # Continue from previous month (except first month)
            is_continuation = (path.name != 'event.evt' and
                               not (year == 2002 and month == 1))
            f.write(f":contflg                      {'y' if is_continuation else 'n'}\n")
            f.write(":routeflg                     n\n")
            f.write(":crseflg                      n\n")
            f.write(":ensimflg                     n\n")
            f.write(":picflg                       n\n")
            f.write(":wetflg                       n\n")
            f.write(":modelflg                     n\n")
            f.write(":shdflg                       n\n")
            f.write(":trcflg                       n\n")
            f.write(":frcflg                       n\n")
            f.write("#\n")
            f.write(":intsoilmoisture              0.25\n")
            f.write(":rainconvfactor                1.00\n")
            f.write(":eventprecipscalefactor        1.00\n")
            f.write(":precipscalefactor             0.00\n")
            f.write(":eventsnowscalefactor          0.00\n")
            f.write(":snowscalefactor               0.00\n")
            f.write(":eventtempscalefactor          0.00\n")
            f.write(":tempscalefactor               0.00\n")
            f.write("#\n")
            f.write(f":hoursraindata                 {nhours}\n")
            f.write(f":hoursflowdata                 {nhours}\n")
            f.write("#\n")
            f.write(":basinfilename                basin\\bow_shd.r2c\n")
            f.write(":parfilename                  basin\\bow.par\n")
            f.write("#\n")
            f.write(f":pointprecip                  raing\\{datestr}.rag\n")
            f.write(f":pointtemps                   tempg\\{datestr}.tag\n")
            f.write(":pointnetradiation\n")
            f.write(":pointhumidity\n")
            f.write(":pointwind\n")
            f.write(":pointlongwave\n")
            f.write(":pointshortwave\n")
            f.write(":pointatmpressure\n")
            f.write("#\n")
            f.write(f":streamflowdatafile           strfw\\{datestr}_str.tb0\n")
            f.write("#\n")
            if is_last:
                f.write(":noeventstofollow                 00\n")
            else:
                next_datestr = f"{next_month.year:04d}{next_month.month:02d}01"
                f.write(":noeventstofollow                 01\n")
                f.write("#\n")
                f.write(f"event\\{next_datestr}.evt\n")
            f.write("eof\n")

    # ------------------------------------------------------------------
    # 4. Output spec
    # ------------------------------------------------------------------
    def _generate_wfo_spec(self) -> None:
        """Generate wfo_spec.txt controlling WATFLOOD output."""
        out = self.settings_dir / 'wfo_spec.txt'
        with open(out, 'w') as f:
            f.write("  5.0 Version Number\n")
            f.write("   10 AttributeCount\n")
            f.write("   24 ReportingTimeStep Hours\n")
            f.write("    0 Start Reporting Time for GreenKenue (hr)\n")
            f.write("    0 End Reporting Time for GreenKenue (hr)\n")
            f.write("1   1 Temperature\n")
            f.write("1   2 Precipitation\n")
            f.write("1   3 Cumulative Precipitation\n")
            f.write("0   4 Lower Zone Storage Class\n")
            f.write("0   5 Ground Water Discharge m^3/s\n")
            f.write("0   6 Grid Runoff\n")
            f.write("1   7 Observed Outflow\n")
            f.write("1   8 Computed Outflow\n")
            f.write("1   9 Weighted SWE\n")
            f.write("1  10 Cumulative ET\n")
        logger.info(f"Wrote output spec: {out}")

    # ------------------------------------------------------------------
    # 5. Streamflow observation .tb0
    # ------------------------------------------------------------------
    def _generate_streamflow_tb0(self, start: datetime, end: datetime) -> None:
        """Generate streamflow .tb0 files from observations for each month."""
        try:
            obs_path = self._find_observation_file()
            if obs_path is None:
                logger.warning("No observation file found, skipping .tb0 generation")
                return

            obs_df = pd.read_csv(obs_path, parse_dates=[0], index_col=0)
            flow_col = None
            for col in obs_df.columns:
                if 'discharge' in col.lower() or 'flow' in col.lower():
                    flow_col = col
                    break
            if flow_col is None and len(obs_df.columns) > 0:
                flow_col = obs_df.columns[0]

            if flow_col is None:
                logger.warning("No flow column found in observations")
                return

            obs_daily = obs_df[flow_col].resample('D').mean()

            months = pd.date_range(start, end, freq='MS')
            for month_start in months:
                datestr = f"{month_start.year:04d}{month_start.month:02d}01"
                month_end = month_start + pd.offsets.MonthEnd(0)
                month_obs = obs_daily.loc[month_start:month_end]

                tb0_path = self.settings_dir / 'strfw' / f'{datestr}_str.tb0'
                with open(tb0_path, 'w') as f:
                    f.write("########################################\n")
                    f.write(":FileType tb0  ASCII  EnSim 1.0\n")
                    f.write("#\n")
                    f.write("# DataType               EnSim Table\n")
                    f.write("#\n")
                    f.write(":Application             EnSimHydrologic\n")
                    f.write(":Version                 2.1.23\n")
                    f.write(":WrittenBy          SYMFLUENCE\n")
                    f.write(f":CreationDate       {datetime.now():%Y-%m-%d  %H:%M}\n")
                    f.write("#\n")
                    f.write(":Name               Streamflow\n")
                    f.write("#\n")
                    f.write(":Projection         UTM\n")
                    f.write(":Ellipsoid          WGS84\n")
                    f.write(":Zone                       11\n")
                    f.write("#\n")
                    f.write(":StartTime         00:00:00.00\n")
                    f.write(f":StartDate            {month_start:%Y/%m/%d}\n")
                    f.write(":DeltaT                        1\n")
                    f.write(":RoutingDeltaT                 1\n")
                    f.write("#\n")
                    f.write(":ColumnMetaData\n")
                    f.write("   :ColumnUnits             m3/s\n")
                    f.write("   :ColumnType             float\n")
                    f.write("   :ColumnName          05BB001\n")
                    f.write("   :ColumnLocationX      583000\n")
                    f.write("   :ColumnLocationY     5673000\n")
                    f.write(":EndColumnMetaData\n")
                    f.write("#\n")
                    f.write(":endHeader\n")
                    # Write daily obs at hour 0
                    ndays = calendar.monthrange(month_start.year, month_start.month)[1]
                    for day in range(1, ndays + 1):
                        date = pd.Timestamp(year=month_start.year,
                                            month=month_start.month, day=day)
                        val = month_obs.get(date, -1.0)
                        if pd.isna(val):
                            val = -1.0
                        f.write(f" {month_start.year} {month_start.month:2d}"
                                f" {day:2d}  0 {val:10.3f}\n")

            logger.info(f"Wrote {len(months)} streamflow .tb0 files")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.warning(f"Could not generate streamflow .tb0: {e}")

    def _find_observation_file(self):
        """Find observation streamflow file."""
        search_dirs = [
            self.project_observations_dir / 'streamflow' / 'preprocessed',
            self.project_observations_dir / 'streamflow',
            self.project_observations_dir,
        ]
        for obs_dir in search_dirs:
            if not obs_dir.exists():
                continue
            for pattern in ['*streamflow*.csv', '*discharge*.csv', '*.csv']:
                matches = list(obs_dir.glob(pattern))
                if matches:
                    return matches[0]
        return None
