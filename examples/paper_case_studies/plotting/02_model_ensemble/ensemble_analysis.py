#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Multi-model ensemble analysis for SYMFLUENCE paper Section 4.2.
Generates three publication-quality figures for the Bow River at Banff case study.

Models included (calibration KGE > 0.5):
  SUMMA, FUSE, GR4J, HBV, HYPE, VIC, LSTM, RHESSys, SACSMA
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
# Data root from SYMFLUENCE_DATA_DIR (default: sibling SYMFLUENCE_data of the
# repo root). The 02_model_ensemble configs all use domain
# Bow_at_Banff_lumped_era5 with experiment_id run_1.
import os
_REPO_ROOT = Path(__file__).resolve().parents[4]
SYMFLUENCE_CODE_DIR = _REPO_ROOT
_DATA_DIR = Path(os.environ.get('SYMFLUENCE_DATA_DIR', _REPO_ROOT.parent / 'SYMFLUENCE_data'))
DATA_ROOT = _DATA_DIR / "domain_Bow_at_Banff_lumped_era5"
FIG_DIR = Path(__file__).resolve().parents[1] / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Current layout keeps observations under data/; the original flat layout is
# the fallback.
_OBS_CANDIDATES = [
    DATA_ROOT / "data" / "observations" / "streamflow" / "preprocessed" /
    "Bow_at_Banff_lumped_era5_streamflow_processed.csv",
    DATA_ROOT / "observations" / "streamflow" / "preprocessed" /
    "Bow_at_Banff_lumped_era5_streamflow_processed.csv",
]
OBS_FILE = next((c for c in _OBS_CANDIDATES if c.exists()), _OBS_CANDIDATES[0])
OPT_DIR = DATA_ROOT / "optimization"

# Basin area (m²) — from SUMMA attributes.nc / shapefiles
BASIN_AREA_M2 = 2_209_951_307.64
BASIN_AREA_KM2 = BASIN_AREA_M2 / 1e6  # 2209.95 km²

# Model specification
# Units key:
#   "m_per_s"  — SUMMA/FUSE native output (m/s), multiply by BASIN_AREA_M2
#   "mm_per_d" — GR4J output (mm/day), multiply by BASIN_AREA_KM2 / 86.4
#   "cms"      — already m³/s
MODEL_SPEC = {
    "SUMMA": {
        "file": OPT_DIR / "SUMMA/dds_run_1/final_evaluation/run_1_timestep.nc",
        "fmt": "netcdf",
        "var": "averageRoutedRunoff",
        "units": "m_per_s",
    },
    "FUSE": {
        "file": OPT_DIR / "FUSE/dds_run_1/fuse_dds_best_clean.nc",
        "fmt": "netcdf",
        "var": "q_instnt",
        "units": "mm_per_d",
    },
    "GR4J": {
        "file": OPT_DIR / "GR/dds_run_1/final_evaluation/GR_results.csv",
        "fmt": "csv",
        "var": "q_sim",
        "units": "mm_per_d",
    },
    "HBV": {
        "file": OPT_DIR / "HBV/dds_run_1/final_evaluation/Bow_at_Banff_lumped_era5_hbv_output.csv",
        "fmt": "csv",
        "var": "streamflow_cms",
        "units": "cms",
    },
    "HYPE": {
        "file": OPT_DIR / "HYPE/dds_hype_seed42_cold/final_evaluation/timeCOUT.txt",
        "fmt": "tsv",
        "var": "1",
        "units": "cms",
    },
    "VIC": {
        "file": OPT_DIR / "VIC/dds_run_1/final_evaluation/vic_output.2002-01-01.nc",
        "fmt": "netcdf_vic",
        "var": "OUT_RUNOFF+OUT_BASEFLOW",
        "units": "mm_per_d",
    },
    "LSTM": {
        "file": DATA_ROOT / "results/run_1_results.csv",
        "fmt": "csv_hourly",
        "var": "LSTM_discharge_cms",
        "units": "cms",
    },
    "RHESSys": {
        "file": OPT_DIR / "RHESSys/dds_run_1/final_evaluation/rhessys_basin.daily",
        "fmt": "rhessys",
        "var": "routedstreamflow",
        "units": "mm_per_d",
    },
    "NGEN": {
        "file": OPT_DIR / "NGEN/dds_run_1/final_evaluation/nex-1_output.csv",
        "fmt": "ngen",
        "var": "flow",
        "units": "cms",
    },
    "MESH": {
        "file": OPT_DIR / "MESH/dds_run_1/final_evaluation/Basin_average_water_balance.csv",
        "fmt": "mesh",
        "var": "RFF+DRAINSOL",
        "units": "mm_per_d",
    },
    "SACSMA": {
        "file": OPT_DIR / "SACSMA/dds_run_1/final_evaluation/Bow_at_Banff_lumped_era5_sacsma_output.csv",
        "fmt": "csv",
        "var": "streamflow_cms",
        "units": "cms",
    },
    "XAJ": {
        "file": OPT_DIR / "XINANJIANG/__no_snow_run__/final_evaluation/Bow_at_Banff_lumped_era5_xinanjiang_output.csv",
        "fmt": "csv",
        "var": "streamflow_cms",
        "units": "cms",
    },
    "XAJ+Snow17": {
        "file": OPT_DIR / "XINANJIANG/dds_run_1/final_evaluation/Bow_at_Banff_lumped_era5_xinanjiang_output.csv",
        "fmt": "csv",
        "var": "streamflow_cms",
        "units": "cms",
    },
    "CLM": {
        "file": OPT_DIR / "CLM/dds_run_1/final_evaluation/simulated_streamflow.csv",
        "fmt": "csv",
        "var": "sim_discharge_m3s",
        "units": "cms",
    },
    "SWAT": {
        "file": OPT_DIR / "SWAT/dds_run_1/final_evaluation/output.rch",
        "fmt": "swat_rch",
        "var": "FLOW_OUTcms",
        "units": "cms",
    },
    "MHM": {
        "file": OPT_DIR / "MHM/dds_run_1/final_evaluation",
        "fmt": "mhm_nc",
        "var": "Qsim",
        "units": "cms",
    },
    "CRHM": {
        "file": OPT_DIR / "CRHM/dds_run_1/final_evaluation",
        "fmt": "crhm_csv",
        "var": "flow",
        "units": "cms",
    },
    "WRFHYDRO": {
        "file": OPT_DIR / "WRFHYDRO/dds_wrfhydro_calibration_dds/final_evaluation",
        "fmt": "netcdf_wrfhydro",
        "var": "streamflow",
        "units": "cms",
    },
    "PRMS": {
        "file": OPT_DIR / "PRMS/dds_prms_calibration_dds/final_evaluation",
        "fmt": "prms_statvar",
        "var": "basin_cfs",
        "units": "cfs",
    },
    "ParFlow+Snow17": {
        "file": OPT_DIR / "PARFLOW/dds_run_1/final_evaluation",
        "fmt": "parflow_pfb",
        "var": "overland_flow",
        "units": "cms",
    },
    "PIHM": {
        "file": OPT_DIR / "PIHM/dds_pihm_calibration_dds_v2/final_evaluation/output/pihm_lumped/pihm_lumped.river.flx1.txt",
        "fmt": "pihm_river",
        "var": "river_flux",
        "units": "cms",
    },
    "HECHMS": {
        "file": OPT_DIR / "HECHMS/dds_run_1/final_evaluation/Bow_at_Banff_lumped_era5_hechms_output.csv",
        "fmt": "csv",
        "var": "streamflow_cms",
        "units": "cms",
    },
    "TOPMODEL": {
        "file": OPT_DIR / "TOPMODEL/dds_run_1/final_evaluation/Bow_at_Banff_lumped_era5_topmodel_output.csv",
        "fmt": "csv",
        "var": "streamflow_cms",
        "units": "cms",
    },
    "SUMMA+MODFLOW": {
        "file": OPT_DIR / "COUPLED_GW/dds_run_5/final_evaluation",
        "fmt": "coupled_gw",
        "var": "combined_flow",
        "units": "cms",
    },
    "GSFLOW": {
        "file": OPT_DIR / "GSFLOW/dds_run_1/final_evaluation",
        "fmt": "prms_statvar",
        "var": "basin_cfs",
        "units": "cfs",
    },
    "CLM+ParFlow": {
        "file": OPT_DIR / "CLMPARFLOW/dds_run_1/final_evaluation",
        "fmt": "clmparflow_pfb",
        "var": "overland_flow",
        "units": "cms",
    },
    "WFLOW": {
        "file": OPT_DIR / "WFLOW/dds_run_1/final_evaluation/output.csv",
        "fmt": "wflow_csv",
        "var": "Q",
        "units": "cms",  # load_wflow_csv handles mm/hr→m³/s internally (before routing)
    },
    "WATFLOOD": {
        "file": OPT_DIR / "WATFLOOD/dds_run_1/final_evaluation/CHARM_dly.csv",
        "fmt": "watflood_csv",
        "var": "05BB001_SIM",
        "units": "cms",
    },
}

# JSON metric files per model
METRIC_FILES = {
    "SUMMA":   OPT_DIR / "SUMMA/dds_run_1/run_1_dds_final_evaluation.json",
    "FUSE":    OPT_DIR / "FUSE/dds_run_1/run_1_dds_final_evaluation.json",
    "GR4J":    OPT_DIR / "GR/dds_run_1/run_1_dds_final_evaluation.json",
    "HBV":     OPT_DIR / "HBV/dds_run_1/run_1_dds_final_evaluation.json",
    "HYPE":    OPT_DIR / "HYPE/dds_hype_seed42_cold/hype_seed42_cold_dds_final_evaluation.json",
    "VIC":     OPT_DIR / "VIC/dds_run_1/run_1_dds_final_evaluation.json",
    "LSTM":    OPT_DIR / "LSTM/pso_run_1/run_1_pso_final_evaluation.json",
    "RHESSys": OPT_DIR / "RHESSys/dds_run_1/run_1_dds_final_evaluation.json",
    "NGEN":    OPT_DIR / "NGEN/dds_run_1/run_1_dds_final_evaluation.json",
    "MESH":    OPT_DIR / "MESH/dds_run_1/run_1_dds_final_evaluation.json",
    "SACSMA":  OPT_DIR / "SACSMA/dds_run_1/run_1_dds_final_evaluation.json",
    "XAJ":     OPT_DIR / "XINANJIANG/__no_snow_run__/run_1_dds_final_evaluation.json",
    "XAJ+Snow17": OPT_DIR / "XINANJIANG/dds_run_1/run_1_dds_final_evaluation.json",
    "CLM":     OPT_DIR / "CLM/dds_run_1/final_evaluation/final_evaluation_results.json",
    "SWAT":    OPT_DIR / "SWAT/dds_run_1/run_1_dds_final_evaluation.json",
    "MHM":     OPT_DIR / "MHM/dds_run_1/run_1_dds_final_evaluation.json",
    "CRHM":    OPT_DIR / "CRHM/dds_run_1/run_1_dds_final_evaluation.json",
    "WRFHYDRO": OPT_DIR / "WRFHYDRO/dds_wrfhydro_calibration_dds/wrfhydro_calibration_dds_dds_final_evaluation.json",
    "PRMS":    OPT_DIR / "PRMS/dds_prms_calibration_dds/prms_calibration_dds_dds_final_evaluation.json",
    "ParFlow+Snow17": OPT_DIR / "PARFLOW/dds_run_1/run_1_dds_final_evaluation.json",
    "PIHM":    OPT_DIR / "PIHM/dds_pihm_calibration_dds_v2/pihm_calibration_dds_v2_dds_final_evaluation.json",
    "HECHMS":  OPT_DIR / "HECHMS/dds_run_1/run_1_dds_final_evaluation.json",
    "TOPMODEL": OPT_DIR / "TOPMODEL/dds_run_1/run_1_dds_final_evaluation.json",
    "SUMMA+MODFLOW": OPT_DIR / "COUPLED_GW/dds_run_5/run_5_dds_final_evaluation.json",
    "GSFLOW": OPT_DIR / "GSFLOW/dds_run_1/run_1_dds_final_evaluation.json",
    "WFLOW":  OPT_DIR / "WFLOW/dds_run_1/run_1_dds_final_evaluation.json",
    "CLM+ParFlow": OPT_DIR / "CLMPARFLOW/dds_run_1/run_1_dds_final_evaluation.json",
    "WATFLOOD": OPT_DIR / "WATFLOOD/dds_run_1/run_1_dds_final_evaluation.json",
}

# Iteration-results CSVs for crash-rate extraction
CRASH_CSV_FILES = {
    "SUMMA":   OPT_DIR / "SUMMA/dds_run_1/run_1_parallel_iteration_results.csv",
    "FUSE":    OPT_DIR / "FUSE/dds_run_1/run_1_parallel_iteration_results.csv",
    "GR4J":    OPT_DIR / "GR/dds_run_1/run_1_parallel_iteration_results.csv",
    "HBV":     OPT_DIR / "HBV/dds_run_1/run_1_parallel_iteration_results.csv",
    "HYPE":    OPT_DIR / "HYPE/dds_hype_seed42_cold/hype_seed42_cold_parallel_iteration_results.csv",
    "VIC":     OPT_DIR / "VIC/dds_run_1/run_1_parallel_iteration_results.csv",
    "RHESSys": OPT_DIR / "RHESSys/dds_run_1/run_1_parallel_iteration_results.csv",
    "NGEN":    OPT_DIR / "NGEN/dds_run_1/run_1_parallel_iteration_results.csv",
    "MESH":    OPT_DIR / "MESH/dds_run_1/run_1_parallel_iteration_results.csv",
    "SACSMA":  OPT_DIR / "SACSMA/dds_run_1/run_1_parallel_iteration_results.csv",
    "XAJ":     OPT_DIR / "XINANJIANG/__no_snow_run__/run_1_parallel_iteration_results.csv",
    "XAJ+Snow17": OPT_DIR / "XINANJIANG/dds_run_1/run_1_parallel_iteration_results.csv",
    "CLM":     OPT_DIR / "CLM/dds_run_1/run_1_parallel_iteration_results.csv",
    "SWAT":    OPT_DIR / "SWAT/dds_run_1/run_1_parallel_iteration_results.csv",
    "MHM":     OPT_DIR / "MHM/dds_run_1/run_1_parallel_iteration_results.csv",
    "CRHM":    OPT_DIR / "CRHM/dds_run_1/run_1_parallel_iteration_results.csv",
    "WRFHYDRO": OPT_DIR / "WRFHYDRO/dds_wrfhydro_calibration_dds/wrfhydro_calibration_dds_parallel_iteration_results.csv",
    "PRMS":    OPT_DIR / "PRMS/dds_prms_calibration_dds/prms_calibration_dds_parallel_iteration_results.csv",
    "ParFlow+Snow17": OPT_DIR / "PARFLOW/dds_run_1/run_1_parallel_iteration_results.csv",
    "PIHM":    OPT_DIR / "PIHM/dds_pihm_calibration_dds_v2/pihm_calibration_dds_v2_parallel_iteration_results.csv",
    "HECHMS":  OPT_DIR / "HECHMS/dds_run_1/run_1_parallel_iteration_results.csv",
    "TOPMODEL": OPT_DIR / "TOPMODEL/dds_run_1/run_1_parallel_iteration_results.csv",
    "SUMMA+MODFLOW": OPT_DIR / "COUPLED_GW/dds_run_5/run_5_parallel_iteration_results.csv",
    "GSFLOW":  OPT_DIR / "GSFLOW/dds_run_1/run_1_parallel_iteration_results.csv",
    "WFLOW":   OPT_DIR / "WFLOW/dds_run_1/run_1_parallel_iteration_results.csv",
    "CLM+ParFlow": OPT_DIR / "CLMPARFLOW/dds_run_1/run_1_parallel_iteration_results.csv",
    "WATFLOOD": OPT_DIR / "WATFLOOD/dds_run_1/run_1_parallel_iteration_results.csv",
}

KGE_THRESHOLD = 0.5

# Common analysis period (2004 start — all models have data from 2004 onward;
# MHM, SWAT and others use 2003 or earlier for spinup)
PERIOD_START = "2004-01-01"
PERIOD_END   = "2009-12-31"

# Calibration / evaluation split (aligned with compare_ensemble.py)
CALIB_START = "2004-01-01"
CALIB_END   = "2007-12-31"
EVAL_START  = "2008-01-01"
EVAL_END    = "2009-12-31"

# Plot period
PLOT_START  = "2004-01-01"
PLOT_END    = "2009-12-31"

# Zoom water years
ZOOM_CAL_START = "2005-04-01"
ZOOM_CAL_END   = "2005-10-31"
# Evaluation detail: Apr-Oct 2008 (same seasonal window as calibration)
ZOOM_EVAL_START = "2008-04-01"
ZOOM_EVAL_END   = "2008-10-31"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "SUMMA":   "#1f77b4",
    "FUSE":    "#ff7f0e",
    "GR4J":    "#2ca02c",
    "HBV":     "#d62728",
    "HYPE":    "#9467bd",
    "VIC":     "#e377c2",
    "LSTM":    "#8c564b",
    "RHESSys": "#17becf",
    "NGEN":    "#7f7f7f",
    "MESH":    "#bcbd22",
    "SACSMA":  "#ff9896",
    "XAJ":     "#aec7e8",
    "XAJ+Snow17": "#c7c7c7",
    "CLM":     "#e6550d",
    "SWAT":    "#98df8a",   # Light green
    "MHM":     "#c49c94",   # Tan
    "CRHM":    "#dbdb8d",   # Light olive
    "WRFHYDRO": "#c5b0d5",  # Light purple
    "PRMS":    "#ffbb78",   # Light orange
    "ParFlow+Snow17": "#e31a1c",   # Bright red
    "PIHM":    "#006d2c",           # Dark green
    "HECHMS":  "#ff6600",           # Orange-red
    "TOPMODEL": "#8B4513",          # Saddle brown
    "SUMMA+MODFLOW": "#b2182b",      # Dark red — highlighted coupled model
    "GSFLOW":        "#4a0082",       # Deep purple — coupled GW-SW model
    "WFLOW":         "#00ced1",       # Dark turquoise — Deltares distributed model
    "CLM+ParFlow":   "#006400",       # Dark green — tightly coupled LSM+subsurface
    "WATFLOOD":      "#8B0000",       # Dark red — distributed flood model
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_obs(path: Path) -> pd.Series:
    """Load observed streamflow, resample hourly -> daily mean."""
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    daily = df["discharge_cms"].resample("D").mean()
    return daily.loc[PERIOD_START:PERIOD_END]


def load_netcdf(path: Path, var: str) -> pd.Series:
    """Load a streamflow variable from a NetCDF file, resample to daily."""
    if not HAS_XARRAY:
        raise ImportError(
            "xarray is required for NetCDF files. "
            "Install with: pip install xarray netCDF4"
        )
    ds = xr.open_dataset(path)
    da = ds[var]
    # Squeeze extra dimensions
    while da.ndim > 1:
        for dim in da.dims:
            if dim != "time":
                da = da.isel({dim: 0})
    s = da.to_series()
    s.index = pd.DatetimeIndex(s.index)
    # Convert to daily if sub-daily
    if len(s) > 1 and (s.index[1] - s.index[0]) < pd.Timedelta("1D"):
        s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_csv_daily(path: Path, var: str) -> pd.Series:
    """Load daily CSV (GR4J, HBV)."""
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    s = df[var]
    s.index = pd.DatetimeIndex(s.index)
    return s.loc[PERIOD_START:PERIOD_END]


def load_csv_hourly(path: Path, var: str) -> pd.Series:
    """Load hourly CSV (LSTM), resample to daily mean."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df[var].dropna()
    s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_netcdf_vic(path: Path, var: str) -> pd.Series:
    """Load VIC NetCDF output, summing OUT_RUNOFF + OUT_BASEFLOW (mm/day)."""
    if not HAS_XARRAY:
        raise ImportError(
            "xarray is required for NetCDF files. "
            "Install with: pip install xarray netCDF4"
        )
    ds = xr.open_dataset(path)
    runoff = ds["OUT_RUNOFF"].squeeze()
    baseflow = ds["OUT_BASEFLOW"].squeeze()
    total = (runoff + baseflow).to_series()
    total.index = pd.DatetimeIndex(total.index)
    return total.loc[PERIOD_START:PERIOD_END]


def load_tsv_hype(path: Path, var: str) -> pd.Series:
    """Load HYPE timeCOUT.txt (tab-separated, 1-line header comment)."""
    df = pd.read_csv(
        path, sep="\t", skiprows=1, index_col="DATE", parse_dates=True
    )
    s = df[var].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    return s.loc[PERIOD_START:PERIOD_END]


def load_rhessys(path: Path, var: str) -> pd.Series:
    """Load RHESSys basin.daily output (whitespace-separated)."""
    df = pd.read_csv(path, sep=r"\s+")
    # Build datetime index from day, month, year columns
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day"]].rename(
            columns={"year": "year", "month": "month", "day": "day"}
        )
    )
    df = df.set_index("datetime")
    s = df[var].astype(float)
    return s.loc[PERIOD_START:PERIOD_END]


def load_ngen(path: Path, var: str) -> pd.Series:
    """Load NGEN nexus output CSV (hourly), resample to daily mean."""
    # Format: index, datetime, flow_value
    df = pd.read_csv(path, header=None, names=["idx", "datetime", "flow"])
    df["datetime"] = pd.to_datetime(df["datetime"].str.strip())
    df = df.set_index("datetime")
    s = df["flow"].astype(float)
    # Resample hourly to daily mean
    s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_mesh(path: Path, var: str) -> pd.Series:
    """Load MESH Basin_average_water_balance.csv (YEAR/JDAY format).

    *var* may be a '+'-separated list of columns (e.g. 'RFF+DRAINSOL')
    which will be summed to give total runoff.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["YEAR"] = df["YEAR"].astype(int)
    df["JDAY"] = df["JDAY"].astype(int)
    df["datetime"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["JDAY"].astype(str).str.zfill(3),
        format="%Y%j",
    )
    df = df.set_index("datetime")
    # Sum multiple columns if var contains '+'
    cols = [c.strip() for c in var.split("+")]
    s = sum(df[c].astype(float) for c in cols)
    return s.loc[PERIOD_START:PERIOD_END]


def load_netcdf_clm(path: Path, var: str) -> pd.Series:
    """Load CLM history files (*.clm2.h0.*.nc) from a directory.

    CLM outputs QRUNOFF in mm/s.  If *path* is a directory, all matching
    ``*.clm2.h0.*.nc`` files are opened with ``open_mfdataset``.
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for NetCDF files.")
    p = Path(path)
    if p.is_dir():
        hist_files = sorted(p.glob("*.clm2.h0.*.nc"))
        if not hist_files:
            raise FileNotFoundError(f"No CLM history files in {p}")
        ds = xr.open_mfdataset(hist_files, combine="by_coords")
    else:
        ds = xr.open_dataset(p)

    if var in ds:
        da = ds[var]
    elif "QOVER" in ds and "QDRAI" in ds:
        da = ds["QOVER"] + ds["QDRAI"]
    else:
        raise KeyError(f"Variable '{var}' not in CLM output")

    # Squeeze spatial dims
    while da.ndim > 1:
        for dim in da.dims:
            if dim != "time":
                da = da.isel({dim: 0})

    s = da.to_series()
    s.index = pd.DatetimeIndex(s.index)
    if len(s) > 1 and (s.index[1] - s.index[0]) < pd.Timedelta("1D"):
        s = s.resample("D").mean()
    ds.close()
    return s.loc[PERIOD_START:PERIOD_END]


def load_swat_rch(path: Path, var: str) -> pd.Series:
    """Load SWAT output.rch fixed-width format, extract reach 1 daily flow.

    The output.rch file has a 9-line header, then fixed-width rows with:
    REACH, GIS, MON, AREAkm2, FLOW_INcms, FLOW_OUTcms, EVAPcms, ...
    We extract reach 1 daily records (MON values 1-366 indicate daily output).
    """
    # Read all lines, skip the 9-line header
    with open(path) as f:
        lines = f.readlines()

    # Find header line (contains "REACH")
    header_end = 0
    for i, line in enumerate(lines):
        if "RCH" in line and "GIS" in line:
            header_end = i + 1
            break
    if header_end == 0:
        header_end = 9  # Default SWAT header length

    data_lines = lines[header_end:]
    records = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            reach = int(parts[1])
            mon = int(float(parts[3]))
            flow_out = float(parts[6])  # FLOW_OUTcms column
        except (ValueError, IndexError):
            continue
        # Reach 1, daily records (MON <= 366)
        if reach == 1 and mon <= 366:
            records.append(flow_out)

    if not records:
        raise ValueError(f"No reach-1 daily records found in {path}")

    # Build date index from experiment start (skip warmup years)
    # SWAT output starts after warmup; dates must align with simulation period
    from datetime import datetime, timedelta
    start = datetime(2004, 1, 1)  # After 2-year warmup from 2002
    dates = [start + timedelta(days=i) for i in range(len(records))]
    s = pd.Series(records, index=pd.DatetimeIndex(dates), name=var)
    return s.loc[PERIOD_START:PERIOD_END]


def load_mhm_nc(path: Path, var: str) -> pd.Series:
    """Load mHM discharge NetCDF output from a directory.

    mHM outputs discharge_*.nc files with the variable Qrouted (m³/s).
    If *path* is a directory, all matching discharge_*.nc files are opened.
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for NetCDF files.")
    p = Path(path)
    if p.is_dir():
        nc_files = sorted(p.glob("discharge*.nc"))
        if not nc_files:
            # Fallback: try any .nc file (exclude Fluxes_States)
            nc_files = sorted(f for f in p.glob("*.nc") if "Fluxes" not in f.name)
        if not nc_files:
            raise FileNotFoundError(f"No mHM discharge files in {p}")
        ds = xr.open_mfdataset(nc_files, combine="by_coords")
    else:
        ds = xr.open_dataset(p)

    if var in ds:
        da = ds[var]
    elif "Q" in ds:
        da = ds["Q"]
    else:
        # mHM names simulated discharge as Qsim_XXXXXXXXXX (gauge-specific)
        qsim_vars = [v for v in ds.data_vars if v.startswith("Qsim")]
        if qsim_vars:
            da = ds[qsim_vars[0]]
        else:
            raise KeyError(f"Variable '{var}' not in mHM output. Available: {list(ds.data_vars)}")

    # Squeeze spatial dims
    while da.ndim > 1:
        for dim in da.dims:
            if dim != "time":
                da = da.isel({dim: 0})

    s = da.to_series()
    s.index = pd.DatetimeIndex(s.index)
    if len(s) > 1 and (s.index[1] - s.index[0]) < pd.Timedelta("1D"):
        s = s.resample("D").mean()
    ds.close()
    return s.loc[PERIOD_START:PERIOD_END]


def load_crhm_csv(path: Path, var: str) -> pd.Series:
    """Load CRHM output from a directory or file.

    CRHM outputs tab-separated .txt or CSV files with datetime index.
    The flow column is 'basinflow(1)' in m^3/interval (hourly).
    """
    p = Path(path)
    if p.is_dir():
        # CRHM output can be .txt or .csv
        target_files = sorted(p.glob("CRHM_output*.txt"))
        if not target_files:
            target_files = sorted(p.glob("*output*.csv"))
        if not target_files:
            target_files = sorted(p.glob("*.csv")) + sorted(p.glob("*.txt"))
        if not target_files:
            raise FileNotFoundError(f"No CRHM output files in {p}")
        target = target_files[0]
    else:
        target = p

    # CRHM .txt files are tab-separated with a units row
    if target.suffix == '.txt':
        df = pd.read_csv(target, sep='\t', parse_dates=True, index_col=0,
                         skiprows=[1], encoding='latin-1')
    else:
        df = pd.read_csv(target, parse_dates=True, index_col=0, encoding='latin-1')

    # Find flow column: basinflow(1), or flow-like column
    flow_col = None
    for col in df.columns:
        if 'basinflow' in col.lower():
            flow_col = col
            break
    if flow_col is None:
        for col in df.columns:
            if any(kw in col.lower() for kw in ['flow', 'discharge', 'streamflow']):
                flow_col = col
                break
    if flow_col is None:
        if var in df.columns:
            flow_col = var
        else:
            raise KeyError(f"No flow column found in CRHM output. Available: {list(df.columns)}")

    s = df[flow_col].astype(float)
    s.index = pd.DatetimeIndex(s.index)

    # CRHM basinflow is in m^3/interval; convert to m^3/s
    if len(s) > 1:
        dt_seconds = (s.index[1] - s.index[0]).total_seconds()
        if dt_seconds > 0:
            s = s / dt_seconds

    if len(s) > 1 and (s.index[1] - s.index[0]) < pd.Timedelta("1D"):
        s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_netcdf_wrfhydro(path: Path, var: str) -> pd.Series:
    """Load WRF-Hydro output from a directory or file.

    Tries CHRTOUT files first (contain streamflow in m³/s directly).
    Falls back to LDASOUT files (accumulated runoff in mm) when routing
    is disabled (CHRTOUT_DOMAIN=0, LSMOUT_DOMAIN=1).
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for NetCDF files.")
    p = Path(path)

    # --- Try CHRTOUT files first ---
    chrtout_files = sorted(p.glob("*CHRTOUT*")) if p.is_dir() else []
    if chrtout_files:
        ds = xr.open_mfdataset(chrtout_files, combine="by_coords")
        if var in ds:
            da = ds[var]
        elif "streamflow" in ds:
            da = ds["streamflow"]
        else:
            raise KeyError(f"Variable '{var}' not in CHRTOUT. Available: {list(ds.data_vars)}")
        while da.ndim > 1:
            for dim in da.dims:
                if dim != "time":
                    da = da.isel({dim: 0})
        s = da.to_series()
        s.index = pd.DatetimeIndex(s.index)
        if len(s) > 1 and (s.index[1] - s.index[0]) < pd.Timedelta("1D"):
            s = s.resample("D").mean()
        ds.close()
        return s.loc[PERIOD_START:PERIOD_END]

    # --- Fallback: LDASOUT files (routing disabled) ---
    ldasout_files = sorted(p.glob("*LDASOUT*")) if p.is_dir() else []
    if not ldasout_files and not p.is_dir():
        ds = xr.open_dataset(p)
        if var in ds:
            da = ds[var]
            while da.ndim > 1:
                for dim in da.dims:
                    if dim != "time":
                        da = da.isel({dim: 0})
            s = da.to_series()
            s.index = pd.DatetimeIndex(s.index)
            ds.close()
            return s.loc[PERIOD_START:PERIOD_END]
        ds.close()

    if not ldasout_files:
        raise FileNotFoundError(f"No CHRTOUT or LDASOUT files in {p}")

    # Read accumulated runoff from each LDASOUT file
    times, runoff_mm = [], []
    for f in ldasout_files:
        ds = xr.open_dataset(f)
        t = pd.Timestamp(ds["time"].values[0]) if "time" in ds else None
        sfcrnoff = float(ds["SFCRNOFF"].values.mean()) if "SFCRNOFF" in ds else 0.0
        ugdrnoff = float(ds["UGDRNOFF"].values.mean()) if "UGDRNOFF" in ds else 0.0
        if t is not None:
            times.append(t)
            runoff_mm.append(sfcrnoff + ugdrnoff)
        ds.close()

    if not times:
        raise ValueError(f"No valid timesteps in LDASOUT files at {p}")

    acc = pd.Series(runoff_mm, index=pd.DatetimeIndex(times)).sort_index()
    # Difference consecutive accumulated values to get per-step runoff (mm)
    delta_mm = acc.diff().iloc[1:]
    delta_mm = delta_mm.clip(lower=0.0)
    # Infer timestep in seconds
    dt_s = (delta_mm.index[1] - delta_mm.index[0]).total_seconds() if len(delta_mm) > 1 else 86400.0
    # Convert mm -> m³/s: Q = delta_mm * area_m² / (dt_s * 1000)
    q_cms = delta_mm * BASIN_AREA_M2 / (dt_s * 1000.0)

    if len(q_cms) > 1 and (q_cms.index[1] - q_cms.index[0]) < pd.Timedelta("1D"):
        q_cms = q_cms.resample("D").mean()
    # Normalize to midnight timestamps (LDASOUT uses start-hour, e.g. 01:00)
    q_cms.index = q_cms.index.normalize()
    return q_cms.loc[PERIOD_START:PERIOD_END]


def load_prms_statvar(path: Path, var: str) -> pd.Series:
    """Load PRMS statvar output from a directory or file.

    PRMS statvar output is a text file with date columns and variable columns.
    The first 6 columns are: index, year, month, day, hour, minute, second.
    Remaining columns are output variables.
    """
    p = Path(path)
    if p.is_dir():
        target_files = sorted(p.glob("*statvar*"))
        if not target_files:
            target_files = sorted(p.glob("*prms*.csv")) + sorted(p.glob("*prms*.txt"))
        if not target_files:
            target_files = sorted(p.glob("*.csv")) + sorted(p.glob("*.dat"))
        if not target_files:
            raise FileNotFoundError(f"No PRMS statvar files in {p}")
        target = target_files[0]
    else:
        target = p

    # Try reading as structured statvar format
    try:
        # PRMS statvar: first line is number of variables, then variable names,
        # then data rows with: index year month day hour minute second values...
        with open(target) as f:
            first_line = f.readline().strip()

        if first_line.isdigit():
            # Standard statvar format
            n_vars = int(first_line)
            var_names = []
            with open(target) as f:
                f.readline()  # skip count line
                for _ in range(n_vars):
                    var_names.append(f.readline().strip())

            skip_rows = 1 + n_vars
            df = pd.read_csv(target, sep=r"\s+", skiprows=skip_rows, header=None)
            # Columns: index, year, month, day, hour, minute, second, var1, var2, ...
            date_cols = df.iloc[:, 1:4]
            date_cols.columns = ["year", "month", "day"]
            df["datetime"] = pd.to_datetime(date_cols)
            df = df.set_index("datetime")

            # Map variable names to columns (starting from column 7)
            for i, vname in enumerate(var_names):
                df = df.rename(columns={7 + i: vname})

            if var in df.columns:
                s = df[var].astype(float)
            else:
                # Try partial match
                matched = [c for c in df.columns if var in str(c)]
                if matched:
                    s = df[matched[0]].astype(float)
                else:
                    raise KeyError(f"Variable '{var}' not in PRMS output. Available: {var_names}")
        else:
            # CSV-like format
            df = pd.read_csv(target, parse_dates=True, index_col=0)
            if var in df.columns:
                s = df[var].astype(float)
            else:
                raise KeyError(f"Variable '{var}' not in PRMS output. Available: {list(df.columns)}")

    except Exception:
        # Fallback: try generic CSV
        df = pd.read_csv(target, sep=r"\s+", parse_dates=True, index_col=0, engine="python")
        if var in df.columns:
            s = df[var].astype(float)
        else:
            # Use first numeric column
            for col in df.columns:
                try:
                    s = df[col].astype(float)
                    break
                except ValueError:
                    continue
            else:
                raise KeyError(f"No numeric column found in PRMS output {target}")

    s.index = pd.DatetimeIndex(s.index)
    if len(s) > 1 and (s.index[1] - s.index[0]) < pd.Timedelta("1D"):
        s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_parflow_pfb(path: Path, var: str) -> pd.Series:
    """Load ParFlow .pfb overland-flow output, route, return daily m³/s.

    Self-contained reader — reads PFB binary files directly, applies area
    scaling and calibrated routing without depending on SYMFLUENCE target
    infrastructure (which can mis-index dates when the config is a raw dict).
    """
    import sys as _sys
    import json as _json
    import numpy as _np
    _sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

    from symfluence.models.parflow.extractor import _read_pfb
    from symfluence.models.parflow.calibration.targets import _linear_reservoir_routing

    output_dir = Path(path)

    # --- Simulation timing (from config_Bow_ParFlow_era5.yaml) ---
    SIM_START = '2002-01-01'
    DUMP_INTERVAL_H = 24

    # --- Grid geometry ---
    NX, NY, DX, DY = 3, 1, 1000.0, 1000.0
    CATCHMENT_AREA_M2 = 2_209_951_308.0          # from HRU shapefile

    domain_area_m2 = NX * DX * NY * DY            # 3 000 000 m²
    area_scale = CATCHMENT_AREA_M2 / domain_area_m2

    # --- Read all overlandsum PFB files ---
    pfb_files = sorted(output_dir.glob('*.out.overlandsum.*.pfb'))
    if not pfb_files:
        raise FileNotFoundError(f"No overlandsum PFB files in {output_dir}")

    values = []
    for f in pfb_files:
        data = _read_pfb(str(f))
        values.append(_np.sum(data) / (DUMP_INTERVAL_H * 3600.0))  # → m³/s

    # --- Build date index (first dump is at SIM_START + 24 h) ---
    dates = pd.date_range(
        start=pd.Timestamp(SIM_START) + pd.Timedelta(hours=DUMP_INTERVAL_H),
        periods=len(values),
        freq=f'{DUMP_INTERVAL_H}h',
    )

    streamflow = pd.Series(values, index=dates, name='streamflow_m3s')
    streamflow = streamflow * area_scale
    streamflow = streamflow.resample('D').mean()

    # --- Apply routing with *calibrated* params ---
    best_params_json = output_dir.parent / 'run_1_dds_best_params.json'
    if best_params_json.exists():
        params = _json.loads(best_params_json.read_text())['best_params']
        alpha    = params.get('ROUTE_ALPHA',    0.3)
        k_slow   = params.get('ROUTE_K_SLOW',   20.0)
        baseflow = params.get('ROUTE_BASEFLOW',  5.0)
        raw = streamflow.values.copy()
        routed = _linear_reservoir_routing(raw, alpha, k_slow, baseflow)
        streamflow = pd.Series(routed, index=streamflow.index,
                               name='streamflow_m3s')

    return streamflow.loc[PERIOD_START:PERIOD_END]


def load_pihm_river(path: Path, var: str) -> pd.Series:
    """Load PIHM river flux output (tab-separated, quoted datetime).

    Format: ``"YYYY-MM-DD HH:MM"\\tvalue``  (m³/s, daily).
    """
    df = pd.read_csv(
        path, sep="\t", header=None, names=["datetime", "value"],
        engine="python",
    )
    df["datetime"] = pd.to_datetime(
        df["datetime"].astype(str).str.strip('" ')
    )
    df = df.set_index("datetime").sort_index()
    s = df["value"].astype(float).abs()
    if len(s) > 1 and (s.index[1] - s.index[0]) < pd.Timedelta("1D"):
        s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_coupled_gw(path: Path, var: str) -> pd.Series:
    """Load SUMMA+MODFLOW coupled output (surface runoff + GW baseflow).

    *path* is the final_evaluation directory containing SUMMA/ and MODFLOW/.
    Combined flow = (averageRoutedRunoff - scalarSoilDrainage) * area
                    + MODFLOW_drain_discharge / 86400
    """
    if not HAS_XARRAY:
        raise ImportError("xarray required for coupled GW loading")

    summa_dir = Path(path) / "SUMMA"
    modflow_dir = Path(path) / "MODFLOW"

    # --- Read averageRoutedRunoff (hourly -> daily mean, m/s) ---
    ts_files = sorted(summa_dir.glob("*_timestep.nc"))
    if not ts_files:
        raise FileNotFoundError(f"No timestep .nc files in {summa_dir}")
    ds_ts = xr.open_dataset(ts_files[0])
    routed = ds_ts["averageRoutedRunoff"].squeeze().to_series()
    routed.index = pd.DatetimeIndex(routed.index)
    routed_daily = routed.resample("D").mean()
    ds_ts.close()

    # --- Read scalarSoilDrainage (daily, m/s) ---
    day_files = sorted(summa_dir.glob("*_day.nc"))
    if not day_files:
        raise FileNotFoundError(f"No day .nc files in {summa_dir}")
    ds_day = xr.open_dataset(day_files[0])
    drainage = ds_day["scalarSoilDrainage"].squeeze().to_series()
    drainage.index = pd.DatetimeIndex(drainage.index)
    drainage = drainage.fillna(0.0).clip(lower=0.0)
    ds_day.close()

    # --- Fast runoff (m/s) = total routed - drainage sent to GW ---
    common = routed_daily.index.intersection(drainage.index)
    fast_runoff = (routed_daily.loc[common] - drainage.loc[common]).clip(lower=0.0)
    surface_m3s = fast_runoff * BASIN_AREA_M2

    # --- MODFLOW drain discharge (m^3/d -> m^3/s) via flopy ---
    import flopy.utils.binaryfile as bf
    bud_file = modflow_dir / "gwf.bud"
    cbb = bf.CellBudgetFile(str(bud_file))
    drain_records = cbb.get_data(text="DRN")

    drain_m3d = []
    for rec in drain_records:
        q_total = abs(float(np.sum(rec["q"])))
        drain_m3d.append(q_total)
    cbb.close()

    # Build drain date index: daily stress periods from simulation start
    # SUMMA simulation starts 2002-01-01; first MODFLOW output = end of day 1
    sim_start = pd.Timestamp("2002-01-01")
    drain_dates = pd.date_range(
        start=sim_start + pd.Timedelta(days=1),
        periods=len(drain_m3d),
        freq="D",
    )
    baseflow_m3s = pd.Series(drain_m3d, index=drain_dates) / 86400.0

    # --- Combine surface + baseflow ---
    surface_m3s.index = surface_m3s.index.normalize()
    baseflow_m3s.index = baseflow_m3s.index.normalize()
    common_final = surface_m3s.index.intersection(baseflow_m3s.index)
    total = surface_m3s.loc[common_final] + baseflow_m3s.loc[common_final]

    return total.loc[PERIOD_START:PERIOD_END]


def load_clmparflow_pfb(path: Path, var: str) -> pd.Series:
    """Load CLMParFlow .pfb overland-flow output, route, return daily m³/s.

    Self-contained reader — reads PFB binary files directly, applies area
    scaling and calibrated routing without depending on SYMFLUENCE target
    infrastructure (which can mis-index dates when the config is a raw dict).
    """
    import sys as _sys
    import json as _json
    import numpy as _np
    _sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

    from symfluence.models.parflow.extractor import _read_pfb
    from symfluence.models.parflow.calibration.targets import _linear_reservoir_routing

    output_dir = Path(path)

    # --- Simulation timing (from config_Bow_CLMParFlow_era5.yaml) ---
    SIM_START = '2002-01-01'
    DUMP_INTERVAL_H = 24

    # --- Grid geometry ---
    NX, NY, DX, DY = 3, 1, 1000.0, 1000.0
    CATCHMENT_AREA_M2 = 2_209_951_308.0          # from HRU shapefile

    domain_area_m2 = NX * DX * NY * DY            # 3 000 000 m²
    area_scale = CATCHMENT_AREA_M2 / domain_area_m2

    # --- Read all overlandsum PFB files ---
    pfb_files = sorted(output_dir.glob('*.out.overlandsum.*.pfb'))
    if not pfb_files:
        raise FileNotFoundError(f"No overlandsum PFB files in {output_dir}")

    values = []
    for f in pfb_files:
        data = _read_pfb(str(f))
        values.append(_np.sum(data) / (DUMP_INTERVAL_H * 3600.0))  # → m³/s

    # --- Build date index (first dump is at SIM_START + 24 h) ---
    dates = pd.date_range(
        start=pd.Timestamp(SIM_START) + pd.Timedelta(hours=DUMP_INTERVAL_H),
        periods=len(values),
        freq=f'{DUMP_INTERVAL_H}h',
    )

    streamflow = pd.Series(values, index=dates, name='streamflow_m3s')
    streamflow = streamflow * area_scale
    streamflow = streamflow.resample('D').mean()

    # --- Apply routing with *calibrated* params ---
    best_params_json = output_dir.parent / 'run_1_dds_best_params.json'
    if best_params_json.exists():
        params = _json.loads(best_params_json.read_text())['best_params']
        alpha    = params.get('ROUTE_ALPHA',    0.3)
        k_slow   = params.get('ROUTE_K_SLOW',   20.0)
        baseflow = params.get('ROUTE_BASEFLOW',  5.0)
        raw = streamflow.values.copy()
        routed = _linear_reservoir_routing(raw, alpha, k_slow, baseflow)
        streamflow = pd.Series(routed, index=streamflow.index,
                               name='streamflow_m3s')

    return streamflow.loc[PERIOD_START:PERIOD_END]


def load_wflow_csv(path: Path, var: str) -> pd.Series:
    """Load Wflow CSV output (hourly), convert mm/hr→m³/s, apply routing, resample to daily.

    Conversion and routing happen at the native hourly timestep so that
    ROUTE_BASEFLOW (in m³/s) is added in consistent units.  The caller
    (load_model) should set units="cms" for WFLOW to skip double-conversion.
    """
    import json as _json
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = var if var in df.columns else df.columns[0]
    s = df[col]
    # Convert mm/hr → m³/s at hourly resolution (before routing)
    s = s * BASIN_AREA_M2 / 3.6e6
    # Apply post-hoc routing if best params contain ROUTE_* keys
    best_params_json = path.parent.parent / 'run_3_dds_best_params.json'
    if best_params_json.exists():
        params = _json.loads(best_params_json.read_text()).get('best_params', {})
        if 'ROUTE_ALPHA' in params:
            alpha = params['ROUTE_ALPHA']
            beta = params['ROUTE_BETA']
            split = params['ROUTE_SPLIT']
            baseflow = params.get('ROUTE_BASEFLOW', 0.0)
            vals = s.values.astype(float)
            n = len(vals)
            s_fast = s_slow = 0.0
            out = np.empty(n)
            for i in range(n):
                q_in = max(vals[i], 0.0)
                s_fast = alpha * s_fast + split * q_in
                s_slow = beta * s_slow + (1.0 - split) * q_in
                out[i] = (1.0 - alpha) * s_fast + (1.0 - beta) * s_slow + baseflow
            s = pd.Series(out, index=s.index, name=col)
    s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_watflood_csv(path: Path, var: str) -> pd.Series:
    """Load WATFLOOD CHARM_dly.csv output (daily, m³/s).

    Format: ``date, 05BB001_obs, 05BB001_SIM,`` with -1.0 as no-data.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.rstrip(",")
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].str.strip())
    df = df.set_index("date")
    col = var.strip()
    if col not in df.columns:
        # Fallback: first SIM column
        sim_cols = [c for c in df.columns if "SIM" in c.upper()]
        col = sim_cols[0] if sim_cols else df.columns[-1]
    s = df[col].astype(float)
    s = s.replace(-1.0, np.nan)
    return s.loc[PERIOD_START:PERIOD_END]


LOADERS = {
    "netcdf":          load_netcdf,
    "netcdf_vic":      load_netcdf_vic,
    "netcdf_clm":      load_netcdf_clm,
    "netcdf_wrfhydro": load_netcdf_wrfhydro,
    "csv":             load_csv_daily,
    "csv_hourly":      load_csv_hourly,
    "tsv":             load_tsv_hype,
    "rhessys":         load_rhessys,
    "ngen":            load_ngen,
    "mesh":            load_mesh,
    "swat_rch":        load_swat_rch,
    "mhm_nc":          load_mhm_nc,
    "crhm_csv":        load_crhm_csv,
    "prms_statvar":    load_prms_statvar,
    "parflow_pfb":     load_parflow_pfb,
    "pihm_river":      load_pihm_river,
    "coupled_gw":      load_coupled_gw,
    "wflow_csv":       load_wflow_csv,
    "clmparflow_pfb":  load_clmparflow_pfb,
    "watflood_csv":    load_watflood_csv,
}


def load_model(spec: dict) -> pd.Series:
    """Dispatch to the right loader and apply unit conversion to m³/s."""
    loader = LOADERS[spec["fmt"]]
    s = loader(spec["file"], spec["var"])
    units = spec.get("units", "cms")
    if units == "m_per_s":
        s = s * BASIN_AREA_M2
    elif units == "mm_per_d":
        s = s * BASIN_AREA_KM2 / 86.4
    elif units == "mm_per_s":
        # CLM outputs mm/s → m³/s: Q = QRUNOFF * area_m2 / 1000
        s = s * BASIN_AREA_M2 / 1000.0
    elif units == "mm_per_hr":
        # mm/hr → m³/s: Q = mm_hr * area_m2 / (3600 * 1000)
        s = s * BASIN_AREA_M2 / 3.6e6
    elif units == "cfs":
        # cubic feet per second → cubic metres per second
        s = s * 0.028316846592
    return s


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def load_metrics(path: Path) -> dict:
    """Load JSON metrics, normalising HBV-style prefixed keys."""
    with open(path) as f:
        raw = json.load(f)

    out = {"calibration": {}, "evaluation": {}}
    for period, key_prefix in [
        ("calibration", "calibration_metrics"),
        ("evaluation", "evaluation_metrics"),
    ]:
        block = raw[key_prefix]
        for metric in ("KGE", "r", "alpha", "beta", "NSE", "RMSE", "PBIAS"):
            val = block.get(metric)
            # Try lowercase (e.g. CLM uses 'kge', 'nse')
            if val is None:
                val = block.get(metric.lower())
            if val is None:
                prefix = "Calib_" if period == "calibration" else "Eval_"
                val = block.get(prefix + metric)
            if val is None:
                suffix = "_Calib" if period == "calibration" else "_Eval"
                val = block.get(metric + suffix)
            if val is not None:
                out[period][metric] = val
    return out


def kge_from_series(sim: pd.Series, obs: pd.Series) -> dict:
    """Compute KGE and its components from aligned series."""
    common = sim.dropna().index.intersection(obs.dropna().index)
    s, o = sim.loc[common].values, obs.loc[common].values
    r = np.corrcoef(s, o)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"KGE": kge, "r": r, "alpha": alpha, "beta": beta}


def load_crash_rates() -> dict:
    """Load average crash rate (%) for each model from iteration-results CSVs.

    Returns a dict mapping model display name to average crash rate as a
    percentage (0–100).  Models without a ``crash_rate`` column or without
    a CSV get ``None``.
    """
    import csv as _csv

    rates = {}
    for name, path in CRASH_CSV_FILES.items():
        if not path.exists():
            rates[name] = None
            continue
        try:
            with open(path) as f:
                reader = list(_csv.DictReader(f))
            if not reader:
                rates[name] = None
                continue
            if "crash_rate" in reader[0]:
                vals = [float(r["crash_rate"]) for r in reader]
                rates[name] = sum(vals) / len(vals) * 100  # → %
            else:
                # Fallback: infer crash rate from score/kge column
                # (score == -9999 or kge <= -9998 indicates a crash)
                score_col = "score" if "score" in reader[0] else (
                    "kge" if "kge" in reader[0] else None)
                if score_col:
                    scores = [float(r[score_col]) for r in reader]
                    n_crash = sum(1 for s in scores if s <= -9998)
                    rates[name] = n_crash / len(scores) * 100
                else:
                    rates[name] = None
        except Exception:
            rates[name] = None
    return rates


def load_calib_runtimes() -> dict:
    """Load total calibration wall-clock time for each model.

    Derives runtime from first and last timestamps in iteration-results CSVs.
    Returns a dict mapping model display name to total hours.
    """
    import csv as _csv
    from datetime import datetime

    runtimes = {}
    for name, path in CRASH_CSV_FILES.items():
        if not path.exists():
            runtimes[name] = None
            continue
        try:
            with open(path) as f:
                reader = list(_csv.DictReader(f))
            if len(reader) < 2:
                runtimes[name] = None
                continue
            if "timestamp" in reader[0]:
                t0 = datetime.fromisoformat(reader[0]["timestamp"])
                t1 = datetime.fromisoformat(reader[-1]["timestamp"])
                runtimes[name] = (t1 - t0).total_seconds() / 3600
            else:
                # Fallback: use file creation/modification timestamps
                import os
                stat = os.stat(path)
                ctime = stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_ctime
                mtime = stat.st_mtime
                delta_h = (mtime - ctime) / 3600
                runtimes[name] = delta_h if delta_h > 0.001 else None
        except Exception:
            runtimes[name] = None
    # Manual overrides for models without iteration CSVs
    # LSTM: gradient-based training (Run 2), 29165s from work log
    if "LSTM" not in runtimes or runtimes.get("LSTM") is None:
        runtimes["LSTM"] = 29165 / 3600  # 8.1 hours
    return runtimes


# Number of calibrated/trainable parameters per model
PARAM_COUNTS = {
    "SUMMA":          11,
    "FUSE":           13,
    "GR4J":            4,
    "HBV":            14,
    "HYPE":           10,
    "VIC":            13,
    "LSTM":       115105,  # trainable weights (2-layer LSTM, hidden=96)
    "RHESSys":        19,
    "NGEN":           11,
    "MESH":           13,
    "SACSMA":         26,
    "XAJ":            15,
    "XAJ+Snow17":     25,
    "CLM":            29,
    "SWAT":           17,
    "MHM":            21,
    "CRHM":           16,
    "WRFHYDRO":        7,
    "PRMS":           10,
    "ParFlow+Snow17": 14,
    "PIHM":            9,
    "HECHMS":         14,
    "TOPMODEL":       11,
    "SUMMA+MODFLOW":  17,
    "GSFLOW":         15,
    "CLM+ParFlow":    14,
    "WFLOW":          19,
    "WATFLOOD":       16,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading observed streamflow...")
    obs = load_obs(OBS_FILE)
    print(
        f"  Obs period: {obs.index[0].date()} to {obs.index[-1].date()}, "
        f"n={len(obs)}"
    )

    # Load all models
    simulations = {}
    for name, spec in MODEL_SPEC.items():
        try:
            s = load_model(spec)
            print(
                f"  {name}: loaded {len(s)} daily values "
                f"({s.index[0].date()} – {s.index[-1].date()})"
            )
            simulations[name] = s
        except Exception as e:
            print(f"  {name}: FAILED to load – {e}")

    # Load JSON metrics (used as fallback only; timeseries-computed values
    # override below because some JSONs store full-period KGE in both fields)
    all_metrics = {}
    for name, path in METRIC_FILES.items():
        if name in simulations:
            try:
                jm = load_metrics(path)
                all_metrics[name] = jm
                # Warn if JSON has identical cal/eval KGE (suspect)
                cal_kge_j = jm.get("calibration", {}).get("KGE")
                eval_kge_j = jm.get("evaluation", {}).get("KGE")
                if (cal_kge_j is not None and eval_kge_j is not None
                        and abs(cal_kge_j - eval_kge_j) < 1e-6):
                    print(f"  WARNING: {name} JSON has identical Cal/Eval KGE "
                          f"({cal_kge_j:.4f}) — will override from timeseries")
            except Exception as e:
                print(f"  {name}: metrics load failed – {e}")

    # Compute period-specific KGE components from time series for all models
    # (more reliable than JSON which may store full-period KGE in both fields)
    for name, sim in simulations.items():
        if name not in all_metrics:
            all_metrics[name] = {}
        for period, (pstart, pend) in [
            ("calibration", (CALIB_START, CALIB_END)),
            ("evaluation", (EVAL_START, EVAL_END)),
        ]:
            mask = (sim.index >= pstart) & (sim.index <= pend)
            obs_p = obs.loc[sim.index[mask]].dropna()
            sim_p = sim.loc[obs_p.index].dropna()
            common = sim_p.index.intersection(obs_p.index)
            if len(common) > 10:
                computed = kge_from_series(sim_p.loc[common], obs_p.loc[common])
                all_metrics[name][period] = computed
            else:
                all_metrics[name][period] = {}

    # Apply KGE filter
    included = {}
    for name, sim in simulations.items():
        calib_kge = all_metrics.get(name, {}).get("calibration", {}).get("KGE", None)
        if calib_kge is not None and calib_kge > KGE_THRESHOLD:
            included[name] = sim
            print(f"  {name}: INCLUDED (Calib KGE = {calib_kge:.3f})")
        else:
            print(f"  {name}: EXCLUDED (Calib KGE = {calib_kge})")

    if not included:
        print("No models passed the KGE filter. Exiting.")
        return

    # Align to common daily index
    common_idx = obs.index
    for s in included.values():
        common_idx = common_idx.intersection(s.index)
    print(
        f"\nCommon period: {common_idx[0].date()} to {common_idx[-1].date()}, "
        f"n={len(common_idx)}"
    )

    obs_aligned = obs.loc[common_idx]
    sim_aligned = {k: v.loc[common_idx] for k, v in included.items()}

    # Build ensemble
    ensemble_df = pd.DataFrame(sim_aligned, index=common_idx)
    ens_mean = ensemble_df.mean(axis=1)
    ens_median = ensemble_df.median(axis=1)
    ens_min = ensemble_df.min(axis=1)
    ens_max = ensemble_df.max(axis=1)
    ens_q10 = ensemble_df.quantile(0.10, axis=1)
    ens_q25 = ensemble_df.quantile(0.25, axis=1)
    ens_q75 = ensemble_df.quantile(0.75, axis=1)
    ens_q90 = ensemble_df.quantile(0.90, axis=1)

    # Ensemble metrics (full period)
    ens_mean_metrics = kge_from_series(ens_mean, obs_aligned)
    ens_median_metrics = kge_from_series(ens_median, obs_aligned)

    # Ensemble metrics by period (calibration and evaluation)
    cal_mask = (ens_mean.index >= CALIB_START) & (ens_mean.index <= CALIB_END)
    eval_mask = (ens_mean.index >= EVAL_START) & (ens_mean.index <= EVAL_END)
    ens_mean_cal_kge = kge_from_series(ens_mean.loc[cal_mask], obs_aligned.loc[cal_mask])["KGE"]
    ens_mean_eval_kge = kge_from_series(ens_mean.loc[eval_mask], obs_aligned.loc[eval_mask])["KGE"]
    ens_med_cal_kge = kge_from_series(ens_median.loc[cal_mask], obs_aligned.loc[cal_mask])["KGE"]
    ens_med_eval_kge = kge_from_series(ens_median.loc[eval_mask], obs_aligned.loc[eval_mask])["KGE"]

    print(f"\nEnsemble mean  KGE: {ens_mean_metrics['KGE']:.3f} (Cal: {ens_mean_cal_kge:.3f}, Eval: {ens_mean_eval_kge:.3f})")
    print(f"Ensemble median KGE: {ens_median_metrics['KGE']:.3f} (Cal: {ens_med_cal_kge:.3f}, Eval: {ens_med_eval_kge:.3f})")

    # ==================================================================
    # FIGURE A: Multi-Model Hydrograph (4-panel + metrics table)
    # ==================================================================
    print("\nGenerating Figure A: Multi-Model Hydrograph...")

    # Compute model cal/eval KGE from time series (more reliable than JSON
    # which may store full-period KGE in both cal and eval fields)
    model_eval_kge = {}
    model_cal_kge = {}
    model_table_metrics = {}
    for name in sim_aligned.keys():
        sim = sim_aligned[name]
        em = (sim.index >= EVAL_START) & (sim.index <= EVAL_END)
        cm = (sim.index >= CALIB_START) & (sim.index <= CALIB_END)
        ckge = kge_from_series(sim.loc[cm], obs_aligned.loc[cm])
        ekge = kge_from_series(sim.loc[em], obs_aligned.loc[em])
        model_cal_kge[name] = ckge["KGE"]
        model_eval_kge[name] = ekge["KGE"]
        # Additional eval-period metrics for table
        s_e, o_e = sim.loc[em].values, obs_aligned.loc[em].values
        nse_e = 1 - np.sum((s_e - o_e) ** 2) / np.sum((o_e - np.mean(o_e)) ** 2)
        pbias_e = 100 * np.sum(s_e - o_e) / np.sum(o_e)
        model_table_metrics[name] = {
            "cal_kge": ckge["KGE"], "eval_kge": ekge["KGE"],
            "eval_r": ekge["r"], "eval_alpha": ekge["alpha"],
            "eval_beta": ekge["beta"], "eval_nse": nse_e, "eval_pbias": pbias_e,
        }

    # FDC helper
    def fdc(series):
        sorted_vals = np.sort(series.dropna().values)[::-1]
        exceedance = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        return exceedance, sorted_vals

    # Helper: plot ensemble shading + lines on a given axis for a date range
    def plot_ensemble_panel(ax, idx):
        ax.fill_between(idx, ens_q10.loc[idx], ens_q90.loc[idx],
                         color="#d1e5f0", alpha=0.7, zorder=1, edgecolor="none")
        ax.fill_between(idx, ens_q25.loc[idx], ens_q75.loc[idx],
                         color="#92c5de", alpha=0.7, zorder=1.5, edgecolor="none")
        for name in sim_aligned:
            ax.plot(idx, sim_aligned[name].loc[idx], color="#aaaaaa",
                    linewidth=0.4, alpha=0.5, zorder=2)
        ax.plot(idx, ens_median.loc[idx], color="#e66101", linewidth=1.2,
                linestyle="--", zorder=3.5)
        ax.plot(idx, ens_mean.loc[idx], color="#1f78b4", linewidth=1.5, zorder=3)
        ax.plot(idx, obs_aligned.loc[idx], color="black", linewidth=1.3, zorder=4)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Layout: wider + taller figure to accommodate 25-model table
    fig = plt.figure(figsize=(24, 13))

    # Top-left: hydrograph
    ax_full = fig.add_axes([0.04, 0.40, 0.42, 0.56])

    # Top-right: table — full height to fit 25 model rows
    ax_tab = fig.add_axes([0.48, 0.40, 0.51, 0.56])

    # Bottom row: three panels
    ax_cal  = fig.add_axes([0.05, 0.05, 0.27, 0.28])
    ax_eval = fig.add_axes([0.38, 0.05, 0.27, 0.28])
    ax_fdc  = fig.add_axes([0.71, 0.05, 0.27, 0.28])

    # --- Panel (a): Full period ---
    plot_mask = (obs_aligned.index >= PLOT_START) & (obs_aligned.index <= PLOT_END)
    plot_idx = obs_aligned.index[plot_mask]
    plot_ensemble_panel(ax_full, plot_idx)

    # Background shading to distinguish calibration vs evaluation periods
    ax_full.axvspan(pd.Timestamp(PLOT_START), pd.Timestamp(CALIB_END),
                    color="#e8f4fd", alpha=0.35, zorder=0)
    ax_full.axvspan(pd.Timestamp(EVAL_START), pd.Timestamp(PLOT_END),
                    color="#fde8e8", alpha=0.35, zorder=0)
    ax_full.axvline(pd.Timestamp(EVAL_START), color="#555555", linestyle="--",
                    linewidth=1.2, zorder=5)
    ax_full.text(pd.Timestamp("2005-12-01"), 305, "Calibration (2004\u20132007)",
                 fontsize=10, color="#2166ac", ha="center", fontweight="semibold")
    ax_full.text(pd.Timestamp("2009-01-01"), 305, "Evaluation (2008\u20132009)",
                 fontsize=10, color="#b2182b", ha="center", fontweight="semibold")

    ax_full.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_full.set_title("(a) Multi-model ensemble hydrograph \u2014 Bow River at Banff (2004\u20132009)",
                      fontsize=13, fontweight="semibold", pad=8)
    ax_full.xaxis.set_major_locator(mdates.YearLocator())
    ax_full.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_full.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    ax_full.set_ylim(bottom=0, top=320)

    leg_handles = [
        Line2D([], [], color="black", lw=1.3, label="Observed"),
        Line2D([], [], color="#1f78b4", lw=1.5, label="Ens. mean"),
        Line2D([], [], color="#e66101", lw=1.2, ls="--", label="Ens. median"),
        Patch(facecolor="#92c5de", alpha=0.7, label="IQR (25\u201375th)"),
        Patch(facecolor="#d1e5f0", alpha=0.7, label="10\u201390th pctl"),
        Line2D([], [], color="#aaaaaa", lw=0.6, alpha=0.6, label="Individual models"),
    ]
    ax_full.legend(handles=leg_handles, loc="lower center", fontsize=9,
                   framealpha=0.95, ncol=6, handlelength=1.5, columnspacing=1.0,
                   bbox_to_anchor=(0.5, -0.08), borderaxespad=0)

    # --- Table panel: model metrics ranked by eval KGE (monospace text) ---
    ax_tab.set_axis_off()
    ax_tab.set_title("(b) Model performance summary (ranked by evaluation KGE)",
                      fontsize=13, fontweight="semibold", pad=8, loc="left")
    ranked = sorted(model_table_metrics.keys(),
                    key=lambda m: -model_table_metrics[m]["eval_kge"])

    # Load crash rates and calibration runtimes
    crash_rates = load_crash_rates()
    calib_runtimes = load_calib_runtimes()

    # Models with asterisk: SYMFLUENCE native JAX re-implementations
    JAX_MODELS = {"TOPMODEL", "HBV", "SACSMA", "XAJ", "XAJ+Snow17", "HECHMS"}
    SHORT_NAMES = {
        "SUMMA+MODFLOW": "SUMMA+MOD",
        "ParFlow+Snow17": "PF+Snow17",
        "XAJ+Snow17": "XAJ+S17*",
        "CLM+ParFlow": "CLM+PF",
    }

    # Format runtime as compact string (must be exactly 6 chars)
    def _fmt_rt(hrs):
        if hrs is None:
            return "    --"
        if hrs < 0.05:
            return f"{hrs*3600:5.0f}s"
        if hrs < 1.0:
            return f"{hrs*60:5.1f}m"
        return f"{hrs:5.1f}h"

    # Format parameter count as compact string (must be exactly 5 chars)
    def _fmt_np(count):
        if count is None:
            return "   --"
        if count >= 1000:
            return f"{count/1000:4.0f}k"
        return f"{count:5d}"

    # Three-space column separator for generous whitespace
    sp = "   "
    # Eval decoration must span exactly 46 chars (KGE+r+α+β+NSE+PB% with separators)
    eval_deco = "----------------- Evaluation -----------------"  # 46 chars
    hdr1 = (f"{'':>2}  {'':17}{sp}{'':>5}{sp}{'Cal':>5}{sp}"
            f"{eval_deco}{sp}{'Fl%':>4}{sp}{'Time':>6}")
    hdr2 = (f"{'#':>2}  {'Model':17}{sp}{'nP':>5}{sp}{'KGE':>5}{sp}"
            f"{'KGE':>5}{sp}{'r':>5}{sp}"
            f"{chr(945):>5}{sp}{chr(946):>5}{sp}"
            f"{'NSE':>5}{sp}{'PB%':>6}{sp}{'%':>4}{sp}{'DDS':>6}")
    sep = "\u2500" * len(hdr2)
    lines = [hdr1, hdr2, sep]
    for i, name in enumerate(ranked, 1):
        m = model_table_metrics[name]
        raw_disp = SHORT_NAMES.get(name, name)
        disp = raw_disp + "*" if name in JAX_MODELS and "*" not in raw_disp else raw_disp
        cr = crash_rates.get(name)
        cr_str = f"{cr:4.0f}" if cr is not None else "  --"
        rt = calib_runtimes.get(name)
        rt_str = _fmt_rt(rt)
        np_str = _fmt_np(PARAM_COUNTS.get(name))
        lines.append(
            f"{i:>2}  {disp:17}{sp}{np_str}{sp}{m['cal_kge']:5.2f}{sp}"
            f"{m['eval_kge']:5.2f}{sp}{m['eval_r']:5.2f}{sp}"
            f"{m['eval_alpha']:5.2f}{sp}{m['eval_beta']:5.2f}{sp}"
            f"{m['eval_nse']:5.2f}{sp}{m['eval_pbias']:>6.1f}{sp}"
            f"{cr_str}{sp}{rt_str}"
        )
    lines.append(sep)
    lines.append(
        f"{'':>2}  {'Ens. mean':17}{sp}{'':>5}{sp}{ens_mean_cal_kge:5.2f}{sp}"
        f"{ens_mean_eval_kge:5.2f}"
    )
    lines.append(
        f"{'':>2}  {'Ens. median':17}{sp}{'':>5}{sp}{ens_med_cal_kge:5.2f}{sp}"
        f"{ens_med_eval_kge:5.2f}"
    )
    lines.append("")
    lines.append("* JAX re-impl    nP: # params    Fl%: crash rate    Time: DDS calibration")

    table_text = "\n".join(lines)
    ax_tab.text(0.02, 0.98, table_text, fontsize=13, fontfamily="monospace",
                va="top", ha="left", transform=ax_tab.transAxes,
                linespacing=1.22,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fafafa",
                          edgecolor="#cccccc", linewidth=0.6))

    # --- Panel (b): Calibration zoom (2005) ---
    cal_zoom_mask = (obs_aligned.index >= ZOOM_CAL_START) & (obs_aligned.index <= ZOOM_CAL_END)
    cal_idx = obs_aligned.index[cal_zoom_mask]
    plot_ensemble_panel(ax_cal, cal_idx)
    ax_cal.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_cal.set_title("(c) Calibration: Apr\u2013Oct 2005", fontsize=12, fontweight="semibold", pad=6)
    ax_cal.xaxis.set_major_locator(mdates.MonthLocator())
    ax_cal.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_cal.set_xlim(pd.Timestamp(ZOOM_CAL_START), pd.Timestamp(ZOOM_CAL_END))

    # --- Panel (c): Evaluation zoom (2008) ---
    eval_zoom_mask = (obs_aligned.index >= ZOOM_EVAL_START) & (obs_aligned.index <= ZOOM_EVAL_END)
    eval_idx = obs_aligned.index[eval_zoom_mask]
    plot_ensemble_panel(ax_eval, eval_idx)
    ax_eval.set_ylabel("")
    ax_eval.set_title("(d) Evaluation: Apr\u2013Oct 2008", fontsize=12, fontweight="semibold", pad=6)
    ax_eval.xaxis.set_major_locator(mdates.MonthLocator())
    ax_eval.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_eval.set_xlim(pd.Timestamp(ZOOM_EVAL_START), pd.Timestamp(ZOOM_EVAL_END))

    # --- Panel (d): Flow Duration Curve ---
    exc_obs, val_obs = fdc(obs_aligned)
    ax_fdc.plot(exc_obs, val_obs, color="black", linewidth=1.5, label="Observed", zorder=4)
    exc_mean, val_mean = fdc(ens_mean)
    ax_fdc.plot(exc_mean, val_mean, color="#1f78b4", linewidth=1.5, label="Ens. mean", zorder=3)
    exc_med, val_med = fdc(ens_median)
    ax_fdc.plot(exc_med, val_med, color="#e66101", linewidth=1.2, linestyle="--",
                label="Ens. median", zorder=3.5)

    _, v_q10 = fdc(ens_q10)
    _, v_q25 = fdc(ens_q25)
    _, v_q75 = fdc(ens_q75)
    _, v_q90 = fdc(ens_q90)
    ax_fdc.fill_between(exc_mean, v_q10, v_q90, color="#d1e5f0", alpha=0.7,
                         zorder=1, edgecolor="none")
    ax_fdc.fill_between(exc_mean, v_q25, v_q75, color="#92c5de", alpha=0.7,
                         zorder=1.5, edgecolor="none")
    for name in sim_aligned:
        exc_m, val_m = fdc(sim_aligned[name])
        ax_fdc.plot(exc_m, val_m, color="#aaaaaa", linewidth=0.4, alpha=0.5, zorder=2)

    ax_fdc.set_xlabel("Exceedance probability (%)")
    ax_fdc.set_ylabel("")
    ax_fdc.set_title("(e) Flow duration curve", fontsize=12, fontweight="semibold", pad=6)
    ax_fdc.set_yscale("log")
    ax_fdc.set_xlim(0, 100)
    ax_fdc.set_ylim(bottom=1)
    ax_fdc.legend(loc="upper right", fontsize=7.5, framealpha=0.95)

    fig.savefig(FIG_DIR / "fig_ensemble_hydrograph.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_ensemble_hydrograph.png'}")

    # ==================================================================
    # FIGURE B: KGE Decomposition (Improved - slope graph with ensemble)
    # ==================================================================
    print("Generating Figure B: KGE Decomposition...")
    models_ordered = sorted(all_metrics.keys())
    components = ["r", "alpha", "beta"]
    comp_labels = ["$r$", r"$\alpha$", r"$\beta$"]
    comp_names = ["Correlation", "Variability ratio", "Bias ratio"]
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    # Add ensemble metrics to the comparison
    ens_metrics = {
        "calibration": kge_from_series(ens_mean.loc[cal_mask], obs_aligned.loc[cal_mask]),
        "evaluation": kge_from_series(ens_mean.loc[eval_mask], obs_aligned.loc[eval_mask]),
    }

    # Create figure with better proportions
    fig = plt.figure(figsize=(13, 4.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.85], wspace=0.32)

    # Colors
    cal_color = "#2166ac"      # Blue for calibration
    eval_color = "#d6604d"     # Orange-red for evaluation
    model_color = "#737373"    # Gray for individual models
    ens_color = "#1a9641"      # Green for ensemble
    optimal_color = "#404040"  # Dark gray for optimal line

    for ci, (comp, clabel, cname) in enumerate(zip(components, comp_labels, comp_names)):
        ax = fig.add_subplot(gs[ci])

        # Determine if this is the "highlight" panel (alpha)
        is_highlight = (comp == "alpha")

        # Highlight panel with subtle background
        if is_highlight:
            ax.set_facecolor("#fef6f6")

        # Optimal line at 1.0 (draw first, behind everything)
        ax.axhline(1.0, color=optimal_color, linestyle="-", linewidth=1.0, zorder=0, alpha=0.3)

        # Plot individual models as slope lines
        for mi, model in enumerate(models_ordered):
            cal_val = all_metrics[model].get("calibration", {}).get(comp, np.nan)
            eval_val = all_metrics[model].get("evaluation", {}).get(comp, np.nan)

            if not np.isnan(cal_val) and not np.isnan(eval_val):
                line_alpha = 0.5 if is_highlight else 0.35
                line_width = 1.2 if is_highlight else 0.9
                ax.plot([0, 1], [cal_val, eval_val], color=model_color,
                        alpha=line_alpha, linewidth=line_width, zorder=1)
                ax.scatter([0], [cal_val], color=cal_color, s=40, zorder=2,
                          edgecolor="white", linewidth=0.5, alpha=0.8)
                ax.scatter([1], [eval_val], color=eval_color, s=40, zorder=2,
                          edgecolor="white", linewidth=0.5, alpha=0.8)

        # Plot ensemble as thick highlighted line
        ens_cal = ens_metrics["calibration"].get(comp, np.nan)
        ens_eval = ens_metrics["evaluation"].get(comp, np.nan)
        if not np.isnan(ens_cal) and not np.isnan(ens_eval):
            ax.plot([0, 1], [ens_cal, ens_eval], color=ens_color,
                    linewidth=2.5, zorder=3)
            ax.scatter([0], [ens_cal], color=ens_color, s=80, zorder=4,
                      edgecolor="white", linewidth=1.0)
            ax.scatter([1], [ens_eval], color=ens_color, s=80, zorder=4,
                      edgecolor="white", linewidth=1.0)
            # Value labels for ensemble only
            ax.text(-0.12, ens_cal, f"{ens_cal:.2f}", ha="right", va="center",
                   fontsize=8, color=ens_color, fontweight="semibold")
            ax.text(1.12, ens_eval, f"{ens_eval:.2f}", ha="left", va="center",
                   fontsize=8, color=ens_color, fontweight="semibold")

        # Axis styling
        ax.set_xlim(-0.25, 1.25)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Cal", "Eval"], fontsize=9)
        ax.tick_params(axis='x', length=0, pad=6)

        # Y-axis
        ax.set_ylim(0.70, 1.35)
        ax.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        if ci == 0:
            ax.set_ylabel("Component value", fontsize=10)
        else:
            ax.set_yticklabels([])

        # Panel title: letter + component
        title_weight = "bold" if is_highlight else "normal"
        ax.set_title(f"{panel_letters[ci]}  {clabel}  {cname}", fontsize=10,
                    fontweight=title_weight, loc="left", pad=10)

        # Add "optimal = 1" annotation only on first panel
        if ci == 0:
            ax.annotate("optimal", xy=(0.5, 1.0), xytext=(0.5, 0.92),
                       fontsize=8, color=optimal_color, ha="center",
                       arrowprops=dict(arrowstyle="-", color=optimal_color, alpha=0.5, lw=0.8))

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend only on first panel
        if ci == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cal_color,
                       markersize=7, label='Calibration'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=eval_color,
                       markersize=7, label='Evaluation'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=ens_color,
                       markersize=8, label='Ensemble', markeredgecolor='white', markeredgewidth=0.5),
            ]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=7.5,
                     framealpha=0.95, borderpad=0.6, handletextpad=0.4)

    # Fourth panel: Summary of changes (Δ)
    ax_delta = fig.add_subplot(gs[3])

    # Calculate changes across models
    delta_data = {comp: [] for comp in components}
    for model in models_ordered:
        for comp in components:
            cal_val = all_metrics[model].get("calibration", {}).get(comp, np.nan)
            eval_val = all_metrics[model].get("evaluation", {}).get(comp, np.nan)
            if not np.isnan(cal_val) and not np.isnan(eval_val):
                delta = abs(eval_val - 1.0) - abs(cal_val - 1.0)
                delta_data[comp].append(delta)

    # Ensemble deltas
    ens_deltas = {}
    for comp in components:
        ens_cal = ens_metrics["calibration"].get(comp, np.nan)
        ens_eval = ens_metrics["evaluation"].get(comp, np.nan)
        ens_deltas[comp] = abs(ens_eval - 1.0) - abs(ens_cal - 1.0)

    # Box plot
    positions = [0, 1, 2]
    bp = ax_delta.boxplot(
        [delta_data[c] for c in components],
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color="#666666", linewidth=1),
        capprops=dict(color="#666666", linewidth=1),
        flierprops=dict(marker='o', markerfacecolor='#999999', markersize=4, alpha=0.6),
    )

    # Color boxes - highlight alpha
    box_colors = ["#2166ac", "#b2182b", "#2166ac"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")
        patch.set_linewidth(1)

    # Ensemble points
    for i, comp in enumerate(components):
        ax_delta.scatter([i], [ens_deltas[comp]], color=ens_color, s=90,
                        zorder=5, edgecolor="white", linewidth=1.5, marker="D")

    # Zero line (no degradation)
    ax_delta.axhline(0, color=optimal_color, linestyle="-", linewidth=1.0, alpha=0.3, zorder=0)

    # Styling
    ax_delta.set_xticks(positions)
    ax_delta.set_xticklabels(["$r$", r"$\alpha$", r"$\beta$"], fontsize=10)
    ax_delta.tick_params(axis='x', length=0, pad=6)
    ax_delta.set_title(f"{panel_letters[3]}  Change (Cal → Eval)", fontsize=10, loc="left", pad=10)

    # Move y-axis to right side
    ax_delta.yaxis.set_label_position("right")
    ax_delta.yaxis.tick_right()
    ax_delta.set_ylabel("Degradation from optimal", fontsize=10)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["left"].set_visible(False)

    # Add annotation for interpretation
    ax_delta.text(0.5, -0.08, "↑ worse in evaluation", ha="center", va="top",
                 transform=ax_delta.transAxes, fontsize=7.5, color="#666666", style="italic")

    # Ensemble legend
    from matplotlib.lines import Line2D
    ens_marker = Line2D([0], [0], marker='D', color='w', markerfacecolor=ens_color,
                        markersize=8, label='Ensemble', markeredgecolor='white')
    ax_delta.legend(handles=[ens_marker], loc="upper right", fontsize=7.5,
                   framealpha=0.95, borderpad=0.5)

    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98)
    fig.savefig(FIG_DIR / "fig_kge_decomposition.png", dpi=300)
    fig.savefig(FIG_DIR / "fig_kge_decomposition.pdf")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_kge_decomposition.png'}")

    # ==================================================================
    # FIGURE C: Ensemble Spread / Envelope + FDC
    # ==================================================================
    print("Generating Figure C: Ensemble Envelope + FDC...")
    fig, (ax_env, ax_fdc) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.30},
    )

    # Left: envelope time series (evaluation period)
    env_mask = (common_idx >= EVAL_START) & (common_idx <= EVAL_END)
    idx_env = common_idx[env_mask]

    ax_env.fill_between(
        idx_env, ens_min.loc[idx_env], ens_max.loc[idx_env],
        color="#a6cee3", alpha=0.45, label="Ensemble range",
    )
    ax_env.plot(
        idx_env, obs_aligned.loc[idx_env],
        color="black", linewidth=1.0, linestyle="--", label="Observed",
    )
    ax_env.plot(
        idx_env, ens_mean.loc[idx_env],
        color="#1f78b4", linewidth=0.9,
        label=f"Ensemble mean (KGE={ens_mean_metrics['KGE']:.2f})",
    )
    ax_env.plot(
        idx_env, ens_median.loc[idx_env],
        color="#e31a1c", linewidth=0.9,
        label=f"Ensemble median (KGE={ens_median_metrics['KGE']:.2f})",
    )

    ax_env.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_env.set_title("(a) Ensemble envelope — evaluation period")
    ax_env.legend(
        loc="upper left", fontsize=7.5, framealpha=0.95,
        borderaxespad=0.5,
    )
    ax_env.xaxis.set_major_locator(mdates.YearLocator())
    ax_env.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Right: Flow Duration Curve
    def fdc(series):
        sorted_vals = np.sort(series.dropna().values)[::-1]
        exceedance = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        return exceedance, sorted_vals

    exc_obs, val_obs = fdc(obs_aligned)
    ax_fdc.plot(
        exc_obs, val_obs, color="black", linewidth=1.2,
        linestyle="--", label="Observed",
    )
    exc_mean, val_mean = fdc(ens_mean)
    ax_fdc.plot(exc_mean, val_mean, color="#1f78b4", linewidth=1.0, label="Ens. mean")
    exc_med, val_med = fdc(ens_median)
    ax_fdc.plot(exc_med, val_med, color="#e31a1c", linewidth=1.0, label="Ens. median")

    for name in sorted(sim_aligned.keys()):
        exc_m, val_m = fdc(sim_aligned[name])
        _hl = (name == "SUMMA+MODFLOW")
        ax_fdc.plot(
            exc_m, val_m, color=MODEL_COLORS[name],
            linewidth=1.5 if _hl else 0.6,
            alpha=0.95 if _hl else 0.6,
            label=name,
        )

    ax_fdc.set_xlabel("Exceedance probability (%)")
    ax_fdc.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_fdc.set_title("(b) Flow duration curve")
    ax_fdc.set_yscale("log")
    ax_fdc.legend(
        loc="lower left", fontsize=6.5, framealpha=0.95, ncol=1,
        borderaxespad=0.5,
    )
    ax_fdc.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ensemble_envelope.png")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_ensemble_envelope.png'}")

    # ==================================================================
    # Summary table (timeseries-computed values)
    # ==================================================================
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (metrics computed from timeseries)")
    print("=" * 90)
    header = (
        f"{'Model':<16} {'Cal KGE':>8} {'Cal r':>7} {'Cal α':>7} {'Cal β':>7} "
        f"{'Eval KGE':>9} {'Eval r':>7} {'Eval α':>8} {'Eval β':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in models_ordered:
        cal = all_metrics[m].get("calibration", {})
        evl = all_metrics[m].get("evaluation", {})
        print(
            f"{m:<16} {cal.get('KGE', 0):8.3f} {cal.get('r', 0):7.3f} "
            f"{cal.get('alpha', 0):7.3f} {cal.get('beta', 0):7.3f} "
            f"{evl.get('KGE', 0):9.3f} {evl.get('r', 0):7.3f} "
            f"{evl.get('alpha', 0):8.3f} {evl.get('beta', 0):8.3f}"
        )
    print("-" * len(header))
    print(
        f"{'Ens. mean':<16} {ens_mean_metrics['KGE']:8.3f} {ens_mean_metrics['r']:7.3f} "
        f"{ens_mean_metrics['alpha']:7.3f} {ens_mean_metrics['beta']:7.3f} "
        f"{ens_mean_cal_kge:9.3f}"
    )
    print(
        f"{'Ens. median':<16} {ens_median_metrics['KGE']:8.3f} {ens_median_metrics['r']:7.3f} "
        f"{ens_median_metrics['alpha']:7.3f} {ens_median_metrics['beta']:7.3f}"
    )

    # ==================================================================
    # Model ranking by evaluation KGE
    # ==================================================================
    print("\n" + "=" * 90)
    print("MODEL RANKING (sorted by Evaluation KGE, best first)")
    print("=" * 90)
    ranking = []
    for m in sim_aligned:
        cal = all_metrics[m].get("calibration", {})
        evl = all_metrics[m].get("evaluation", {})
        ranking.append((m, cal.get("KGE", np.nan), evl.get("KGE", np.nan),
                         evl.get("r", np.nan), evl.get("alpha", np.nan),
                         evl.get("beta", np.nan)))
    ranking.sort(key=lambda x: -x[2] if not np.isnan(x[2]) else 999)

    print(f"{'Rank':>4} {'Model':<16} {'Cal KGE':>8} {'Eval KGE':>9} "
          f"{'Eval r':>7} {'Eval α':>8} {'Eval β':>8}  {'KGE Δ':>7}")
    print("-" * 80)
    for i, (m, ck, ek, er, ea, eb) in enumerate(ranking, 1):
        delta = ek - ck if not (np.isnan(ek) or np.isnan(ck)) else np.nan
        print(f"{i:4d} {m:<16} {ck:8.3f} {ek:9.3f} {er:7.3f} {ea:8.3f} "
              f"{eb:8.3f}  {delta:+7.3f}")
    # Ensemble mean reference
    print("-" * 80)
    ens_delta = ens_mean_eval_kge - ens_mean_cal_kge
    print(f"{'':>4} {'Ens. mean':<16} {ens_mean_cal_kge:8.3f} {ens_mean_eval_kge:9.3f} "
          f"{'':>7} {'':>8} {'':>8}  {ens_delta:+7.3f}")
    print("=" * 90)

    print("\nDone. All figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
