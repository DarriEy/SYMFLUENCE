#!/usr/bin/env python3
"""
Data Processing Pipeline Analysis for SYMFLUENCE Paper Section 4.12

Analyses the outputs of the pipeline experiment across three canonical
scales (Paradise point, Bow watershed, Iceland regional) to produce:
  1. Stage-by-stage data volume inventory (per domain)
  2. NetCDF CF-convention compliance check
  3. Remapping weight characterisation (sparsity, coverage)
  4. Observation coverage and quality report
  5. Variable standardisation audit (CFIF compliance)
  6. Forcing preprocessing profiling (per-file cost estimate)
  7. Data compression analysis (raw vs basin-averaged)
  8. Stage dependency mapping (DAG structure)
  9. Variable mapping audit (ERA5 -> CFIF complete table)
 10. Cross-scale summary (totals, compression ratios, scaling)
 11. Data shape tracking (array dimensions at each pipeline stage)

Usage:
    python analyze_pipeline.py [--data-dir DIR] [--output-dir DIR]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add SYMFLUENCE to path
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

# Three canonical domains spanning the full range of scales in the paper
# n_hrus = number of hydrological response units (forcing remapping target)
# n_grus = number of grouped response units (sub-basins); GRUs contain one or more HRUs
# raw_grid_cells = approximate number of ERA5 grid cells intersecting the domain
DOMAINS = {
    "paradise": {
        "dir_name": "paradise",
        "dir_name_fallbacks": ["paradise_snotel_wa_era5", "paradise_multivar"],
        "label": "Paradise SNOTEL",
        "scale": "point",
        "n_hrus": 1,
        "n_grus": 1,
        "raw_grid_cells": 9,
        "area_km2": 0.01,
        "sections": ["4.3", "4.10"],
    },
    "bow": {
        "dir_name": "Bow_at_Banff_semi_distributed",
        "dir_name_fallbacks": ["Bow_at_Banff_lumped_era5", "Bow_at_Banff_multivar"],
        "label": "Bow at Banff",
        "scale": "watershed",
        "n_hrus": 49,
        "n_grus": 49,
        "raw_grid_cells": 42,
        "area_km2": 2210,
        "sections": ["4.2", "4.4-4.7"],
    },
    "iceland": {
        "dir_name": "Iceland",
        "dir_name_fallbacks": ["ellioaar_iceland", "Iceland_multivar"],
        "label": "Iceland",
        "scale": "regional",
        "n_hrus": 21474,
        "n_grus": 6600,
        "raw_grid_cells": 954,
        "area_km2": 103000,
        "sections": ["4.8", "4.9"],
    },
}


def resolve_domain_dir(data_dir: Path, domain_info: Dict) -> Path:
    """Resolve the actual domain directory, trying fallbacks for observations etc."""
    primary = data_dir / f"domain_{domain_info['dir_name']}"
    if primary.exists():
        return primary
    for fb in domain_info.get("dir_name_fallbacks", []):
        fallback = data_dir / f"domain_{fb}"
        if fallback.exists():
            logger.info(f"  Using fallback directory: {fallback.name}")
            return fallback
    return primary  # return primary even if missing, let callers handle


def resolve_observation_dir(data_dir: Path, domain_info: Dict) -> Path:
    """Find the best directory for observation data, scanning fallbacks for richest obs."""
    candidates = [domain_info["dir_name"]] + domain_info.get("dir_name_fallbacks", [])
    best_dir = None
    best_count = -1
    for name in candidates:
        obs_dir = data_dir / f"domain_{name}" / "observations"
        if obs_dir.exists():
            # Count observation subdirectories that contain actual files
            count = sum(1 for d in obs_dir.iterdir()
                        if d.is_dir() and any(d.rglob("*.csv")) or any(d.rglob("*.nc")))
            if count > best_count:
                best_count = count
                best_dir = obs_dir.parent
    if best_dir is not None:
        return best_dir
    return data_dir / f"domain_{domain_info['dir_name']}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline_analysis")


# ------------------------------------------------------------------ helpers
def format_bytes(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def scan_directory(path: Path) -> Dict[str, Any]:
    """Return file counts and sizes grouped by extension."""
    ext_stats: Dict[str, Dict] = {}
    total_bytes = 0
    total_files = 0
    if not path.exists():
        return {"total_bytes": 0, "total_files": 0, "by_extension": {}}
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower() or "(no ext)"
        sz = f.stat().st_size
        total_bytes += sz
        total_files += 1
        if ext not in ext_stats:
            ext_stats[ext] = {"count": 0, "bytes": 0}
        ext_stats[ext]["count"] += 1
        ext_stats[ext]["bytes"] += sz
    return {
        "total_bytes": total_bytes,
        "total_files": total_files,
        "by_extension": ext_stats,
    }


# ------------------------------------------------ 1. Data volume inventory
def analyze_data_volumes(data_dir: Path, domain_info: Dict) -> Dict[str, Any]:
    """Inventory all data produced by the pipeline for one domain."""
    domain_dir = resolve_domain_dir(data_dir, domain_info)
    obs_domain_dir = resolve_observation_dir(data_dir, domain_info)
    if not domain_dir.exists():
        logger.warning(f"Domain directory not found: {domain_dir}")
        return {}

    categories = [
        ("attributes/elevation", "Elevation (DEM)"),
        ("attributes/soilclass", "Soil class"),
        ("attributes/landclass", "Land cover"),
        ("attributes/aspect", "Aspect"),
        ("forcing/raw_data", "Forcing (raw)"),
        ("forcing/basin_averaged_data", "Forcing (basin-averaged)"),
        ("forcing/merged_path", "Forcing (merged)"),
        ("shapefiles/catchment_intersection/with_forcing", "Remapping weights"),
        ("shapefiles/catchment", "Catchment shapefiles"),
        ("shapefiles/river_network", "River network"),
        ("observations/streamflow", "Streamflow obs"),
        ("observations/snow", "Snow obs"),
        ("observations/et", "ET obs"),
        ("observations/grace", "GRACE obs"),
        ("observations/soil_moisture", "Soil moisture obs"),
        ("settings", "Model settings"),
    ]

    inventory = {}
    for subpath, label in categories:
        # For observation categories, try the obs fallback directory first
        if subpath.startswith("observations/"):
            full = obs_domain_dir / subpath
            if not full.exists():
                full = domain_dir / subpath
        else:
            full = domain_dir / subpath
        stats = scan_directory(full)
        inventory[label] = {
            "path": str(full),
            "exists": full.exists(),
            "total_bytes": stats["total_bytes"],
            "total_human": format_bytes(stats["total_bytes"]),
            "n_files": stats["total_files"],
            "by_extension": {
                k: {"count": v["count"], "size": format_bytes(v["bytes"])}
                for k, v in stats["by_extension"].items()
            },
        }
    return inventory


# ------------------------------------------- 2. NetCDF CF compliance check
def check_cf_compliance(data_dir: Path, domain_info: Dict) -> List[Dict]:
    """Validate CF-convention attributes on forcing NetCDF files."""
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not available; skipping CF compliance check")
        return []

    nc_dir = resolve_domain_dir(data_dir, domain_info) / "forcing" / "basin_averaged_data"
    if not nc_dir.exists():
        logger.info(f"  No basin-averaged forcing for {domain_info['label']}; skipping CF check")
        return []

    required_attrs = ["units", "long_name"]
    recommended_attrs = ["standard_name", "cell_methods"]
    results = []

    for nc_file in sorted(nc_dir.glob("*.nc"))[:20]:
        try:
            ds = xr.open_dataset(nc_file)
            file_report = {
                "file": nc_file.name,
                "n_variables": len(ds.data_vars),
                "dimensions": {k: v for k, v in ds.dims.items()},
                "time_range": None,
                "variables": {},
            }
            if "time" in ds.coords:
                file_report["time_range"] = [
                    str(ds.time.values[0]),
                    str(ds.time.values[-1]),
                ]

            for var_name in ds.data_vars:
                var = ds[var_name]
                var_report = {
                    "dtype": str(var.dtype),
                    "shape": list(var.shape),
                    "has_required_attrs": all(a in var.attrs for a in required_attrs),
                    "has_recommended_attrs": all(a in var.attrs for a in recommended_attrs),
                    "units": var.attrs.get("units", "MISSING"),
                    "standard_name": var.attrs.get("standard_name", "MISSING"),
                }
                file_report["variables"][var_name] = var_report
            ds.close()
            results.append(file_report)
        except Exception as e:
            results.append({"file": nc_file.name, "error": str(e)})

    return results


# -------------------------------------- 3. Remapping weight characterisation
def analyze_remapping_weights(data_dir: Path, domain_info: Dict) -> Dict[str, Any]:
    """Characterise remapping weight matrices (sparsity, coverage)."""
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not available; skipping weight analysis")
        return {}

    weight_dir = (
        resolve_domain_dir(data_dir, domain_info)
        / "shapefiles" / "catchment_intersection" / "with_forcing"
    )
    if not weight_dir.exists():
        logger.info(f"  No remapping weights for {domain_info['label']}; skipping")
        return {"exists": False}

    weight_files = list(weight_dir.glob("*.nc")) + list(weight_dir.glob("*.csv"))
    if not weight_files:
        return {"exists": True, "n_files": 0}

    report = {
        "exists": True,
        "n_files": len(weight_files),
        "files": [],
        "domain_label": domain_info["label"],
        "n_hrus": domain_info["n_hrus"],
        "n_grus": domain_info["n_grus"],
        "scale": domain_info["scale"],
    }

    for wf in weight_files[:5]:
        entry = {"file": wf.name, "size": format_bytes(wf.stat().st_size)}
        if wf.suffix == ".nc":
            try:
                ds = xr.open_dataset(wf)
                entry["dimensions"] = {k: v for k, v in ds.dims.items()}
                entry["variables"] = list(ds.data_vars)
                for vname in ds.data_vars:
                    arr = ds[vname].values.flatten()
                    n_total = arr.size
                    n_nonzero = np.count_nonzero(arr)
                    entry["sparsity"] = round(1.0 - n_nonzero / max(n_total, 1), 4)
                    entry["n_source_cells"] = int(n_nonzero)
                    entry["n_total_elements"] = int(n_total)
                    # Store downsampled sparsity pattern for visualization
                    if arr.ndim == 1 and n_total > 0:
                        # Reshape if possible
                        pass
                    break
                ds.close()
            except Exception as e:
                entry["error"] = str(e)
        elif wf.suffix == ".csv":
            try:
                df = pd.read_csv(wf)
                entry["n_rows"] = len(df)
                entry["columns"] = list(df.columns)
            except Exception as e:
                entry["error"] = str(e)
        report["files"].append(entry)

    return report


# ------------------------------------------ 4. Observation coverage report
def analyze_observation_coverage(data_dir: Path, domain_info: Dict) -> Dict[str, Any]:
    """Assess temporal coverage and gap fraction for each observation type."""
    obs_dir = resolve_observation_dir(data_dir, domain_info) / "observations"
    if not obs_dir.exists():
        logger.info(f"  No observations directory for {domain_info['label']}; skipping")
        return {}

    coverage = {}
    for obs_type_dir in sorted(obs_dir.iterdir()):
        if not obs_type_dir.is_dir():
            continue
        obs_type = obs_type_dir.name
        csv_files = list(obs_type_dir.rglob("*.csv"))
        nc_files = list(obs_type_dir.rglob("*.nc"))

        type_report = {
            "n_csv": len(csv_files),
            "n_nc": len(nc_files),
            "total_size": format_bytes(scan_directory(obs_type_dir)["total_bytes"]),
            "domain": domain_info["label"],
            "scale": domain_info["scale"],
        }

        if csv_files:
            try:
                df = pd.read_csv(csv_files[0], parse_dates=True, index_col=0)
                type_report["n_records"] = len(df)
                type_report["date_range"] = [str(df.index[0]), str(df.index[-1])]
                type_report["columns"] = list(df.columns)
                n_missing = int(df.iloc[:, 0].isna().sum()) if len(df.columns) > 0 else 0
                type_report["gap_fraction"] = round(n_missing / max(len(df), 1), 4)
            except Exception:
                pass

        coverage[obs_type] = type_report
    return coverage


# -------------------------------------------- 5. Variable standardisation audit
def audit_variable_standardisation(data_dir: Path, domain_info: Dict) -> Dict[str, Any]:
    """
    Compare variable names and units between raw and basin-averaged forcing
    to verify that the CFIF standardisation pipeline ran correctly.
    """
    try:
        import xarray as xr
    except ImportError:
        return {}

    domain_dir = resolve_domain_dir(data_dir, domain_info)
    raw_dir = domain_dir / "forcing" / "raw_data"
    processed_dir = domain_dir / "forcing" / "basin_averaged_data"

    audit = {"raw": {}, "processed": {}, "domain": domain_info["label"]}
    for label, fdir in [("raw", raw_dir), ("processed", processed_dir)]:
        if not fdir.exists():
            logger.info(f"  No {label} forcing for {domain_info['label']}; skipping audit")
            continue
        nc_files = sorted(fdir.glob("*.nc"))[:5]
        for nc in nc_files:
            try:
                ds = xr.open_dataset(nc)
                for vname in ds.data_vars:
                    audit[label][vname] = {
                        "units": ds[vname].attrs.get("units", "MISSING"),
                        "standard_name": ds[vname].attrs.get("standard_name", "MISSING"),
                        "long_name": ds[vname].attrs.get("long_name", "MISSING"),
                        "dtype": str(ds[vname].dtype),
                    }
                ds.close()
            except Exception:
                continue
    return audit


# -------------------------------------------- 6. Forcing preprocessing profiling
def profile_forcing_preprocessing(data_dir: Path, domain_info: Dict) -> Dict[str, Any]:
    """
    Profile per-file processing cost by measuring read time for raw vs
    basin-averaged forcing files.
    """
    try:
        import xarray as xr
    except ImportError:
        return {}

    domain_dir = resolve_domain_dir(data_dir, domain_info)
    raw_dir = domain_dir / "forcing" / "raw_data"
    ba_dir = domain_dir / "forcing" / "basin_averaged_data"

    profiling = {
        "raw_files_sampled": 0,
        "processed_files_sampled": 0,
        "raw_avg_size_bytes": 0,
        "processed_avg_size_bytes": 0,
        "raw_avg_read_ms": 0,
        "processed_avg_read_ms": 0,
        "raw_avg_timesteps": 0,
        "processed_avg_timesteps": 0,
        "per_variable": {},
        "domain": domain_info["label"],
        "n_hrus": domain_info["n_hrus"],
    }

    raw_files = sorted(raw_dir.glob("*.nc"))[:5] if raw_dir.exists() else []
    raw_sizes, raw_times, raw_ntime = [], [], []
    for rf in raw_files:
        raw_sizes.append(rf.stat().st_size)
        t0 = time.perf_counter()
        try:
            ds = xr.open_dataset(rf)
            _ = {v: ds[v].values for v in ds.data_vars}
            nt = ds.dims.get("time", 0)
            raw_ntime.append(nt)
            ds.close()
        except Exception:
            raw_ntime.append(0)
        raw_times.append((time.perf_counter() - t0) * 1000)

    ba_files = sorted(ba_dir.glob("*.nc"))[:5] if ba_dir.exists() else []
    ba_sizes, ba_times, ba_ntime = [], [], []
    for bf in ba_files:
        ba_sizes.append(bf.stat().st_size)
        t0 = time.perf_counter()
        try:
            ds = xr.open_dataset(bf)
            _ = {v: ds[v].values for v in ds.data_vars}
            nt = ds.dims.get("time", 0)
            ba_ntime.append(nt)

            for vname in ds.data_vars:
                if vname in ("latitude", "longitude", "hruId", "time"):
                    continue
                arr = ds[vname].values
                if arr.size > 0:
                    profiling["per_variable"][vname] = {
                        "mean": float(np.nanmean(arr)),
                        "std": float(np.nanstd(arr)),
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "dtype": str(arr.dtype),
                        "shape": list(arr.shape),
                    }
            ds.close()
        except Exception:
            ba_ntime.append(0)
        ba_times.append((time.perf_counter() - t0) * 1000)

    profiling["raw_files_sampled"] = len(raw_files)
    profiling["processed_files_sampled"] = len(ba_files)
    if raw_sizes:
        profiling["raw_avg_size_bytes"] = int(np.mean(raw_sizes))
        profiling["raw_avg_read_ms"] = round(float(np.mean(raw_times)), 1)
        profiling["raw_avg_timesteps"] = int(np.mean(raw_ntime))
    if ba_sizes:
        profiling["processed_avg_size_bytes"] = int(np.mean(ba_sizes))
        profiling["processed_avg_read_ms"] = round(float(np.mean(ba_times)), 1)
        profiling["processed_avg_timesteps"] = int(np.mean(ba_ntime))

    return profiling


# -------------------------------------------- 7. Data compression analysis
def analyze_compression(data_dir: Path, domain_info: Dict) -> Dict[str, Any]:
    """
    Compare raw vs basin-averaged NetCDF file sizes to quantify the
    compression achieved by spatial aggregation from grid to HRUs.
    """
    domain_dir = resolve_domain_dir(data_dir, domain_info)
    raw_dir = domain_dir / "forcing" / "raw_data"
    ba_dir = domain_dir / "forcing" / "basin_averaged_data"

    raw_total = 0
    ba_total = 0
    raw_count = 0
    ba_count = 0
    per_variable_compression = {}

    if raw_dir.exists():
        for f in raw_dir.glob("*.nc"):
            raw_total += f.stat().st_size
            raw_count += 1

    if ba_dir.exists():
        for f in ba_dir.glob("*.nc"):
            ba_total += f.stat().st_size
            ba_count += 1

    try:
        import xarray as xr
        raw_files = sorted(raw_dir.glob("*.nc"))[:10] if raw_dir.exists() else []
        for rf in raw_files:
            try:
                ds = xr.open_dataset(rf)
                for vname in ds.data_vars:
                    raw_var_bytes = ds[vname].values.nbytes
                    per_variable_compression[vname] = per_variable_compression.get(vname, {
                        "raw_bytes": 0, "processed_bytes": 0,
                    })
                    per_variable_compression[vname]["raw_bytes"] += raw_var_bytes
                ds.close()
            except Exception:
                continue

        ba_files = sorted(ba_dir.glob("*.nc"))[:10] if ba_dir.exists() else []
        for bf in ba_files:
            try:
                ds = xr.open_dataset(bf)
                for vname in ds.data_vars:
                    if vname in per_variable_compression:
                        proc_var_bytes = ds[vname].values.nbytes
                        per_variable_compression[vname]["processed_bytes"] += proc_var_bytes
                ds.close()
            except Exception:
                continue
    except ImportError:
        pass

    for vname, info in per_variable_compression.items():
        raw_b = info["raw_bytes"]
        proc_b = info["processed_bytes"]
        if proc_b > 0:
            info["compression_ratio"] = round(raw_b / proc_b, 2)
        else:
            info["compression_ratio"] = None
        info["raw_human"] = format_bytes(raw_b)
        info["processed_human"] = format_bytes(proc_b)

    compression = {
        "raw_total_bytes": raw_total,
        "raw_total_human": format_bytes(raw_total),
        "raw_n_files": raw_count,
        "processed_total_bytes": ba_total,
        "processed_total_human": format_bytes(ba_total),
        "processed_n_files": ba_count,
        "overall_compression_ratio": round(raw_total / max(ba_total, 1), 2),
        "per_variable": per_variable_compression,
        "domain": domain_info["label"],
        "n_hrus": domain_info["n_hrus"],
        "n_grus": domain_info["n_grus"],
        "area_km2": domain_info["area_km2"],
    }
    return compression


# -------------------------------------------- 8. Stage dependency mapping (DAG)
def build_stage_dependency_map() -> Dict[str, Any]:
    """
    Document the directed acyclic graph (DAG) of pipeline stages,
    showing which stages produce outputs consumed by downstream stages.
    """
    stages = [
        {
            "id": "setup_project",
            "label": "Project initialisation",
            "category": "setup",
            "description": "Create directory structure, validate configuration YAML",
            "outputs": ["project_dir", "config_validated"],
            "inputs": ["config_yaml"],
        },
        {
            "id": "create_pour_point",
            "label": "Pour-point creation",
            "category": "geospatial",
            "description": "Define watershed outlet from coordinates or station ID",
            "outputs": ["pour_point_shapefile"],
            "inputs": ["config_yaml"],
        },
        {
            "id": "define_domain",
            "label": "Domain delineation",
            "category": "geospatial",
            "description": "Delineate catchment boundary from DEM and pour point",
            "outputs": ["catchment_shapefile", "river_network_shapefile"],
            "inputs": ["pour_point_shapefile", "DEM_raster"],
        },
        {
            "id": "discretize_domain",
            "label": "Domain discretisation",
            "category": "geospatial",
            "description": "Subdivide catchment into GRUs (elevation bands, land cover, soil)",
            "outputs": ["gru_shapefile"],
            "inputs": ["catchment_shapefile", "DEM_raster", "landcover_raster", "soilclass_raster"],
        },
        {
            "id": "acquire_attributes",
            "label": "Attribute acquisition",
            "category": "geospatial",
            "description": "Download DEM, soil class, land cover rasters for domain bounding box",
            "outputs": ["DEM_raster", "soilclass_raster", "landcover_raster"],
            "inputs": ["catchment_shapefile"],
        },
        {
            "id": "compute_zonal_statistics",
            "label": "Zonal statistics",
            "category": "geospatial",
            "description": "Aggregate raster attributes (elevation, soil, land cover) to GRU polygons",
            "outputs": ["gru_attributes_table"],
            "inputs": ["gru_shapefile", "DEM_raster", "soilclass_raster", "landcover_raster"],
        },
        {
            "id": "acquire_forcings",
            "label": "Forcing acquisition",
            "category": "forcing",
            "description": "Download ERA5 reanalysis (7 variables, hourly) via CDS API",
            "outputs": ["raw_forcing_nc"],
            "inputs": ["catchment_shapefile", "config_yaml"],
        },
        {
            "id": "generate_weights",
            "label": "Weight generation",
            "category": "forcing",
            "description": "EASYMORE intersection of forcing grid cells with GRU polygons",
            "outputs": ["remapping_weights_nc", "intersected_shapefile"],
            "inputs": ["gru_shapefile", "raw_forcing_nc"],
        },
        {
            "id": "apply_weights",
            "label": "Weight application",
            "category": "forcing",
            "description": "Matrix multiply raw forcing by remapping weights to produce basin-averaged values",
            "outputs": ["basin_averaged_forcing_nc"],
            "inputs": ["raw_forcing_nc", "remapping_weights_nc"],
        },
        {
            "id": "standardize_variables",
            "label": "Variable standardisation",
            "category": "forcing",
            "description": "Convert dataset-native names/units to CF-Intermediate Format (CFIF)",
            "outputs": ["cfif_forcing_nc"],
            "inputs": ["basin_averaged_forcing_nc"],
        },
        {
            "id": "lapse_rate_correction",
            "label": "Lapse-rate correction",
            "category": "forcing",
            "description": "Apply elevation-based temperature correction per GRU",
            "outputs": ["corrected_forcing_nc"],
            "inputs": ["cfif_forcing_nc", "gru_attributes_table"],
        },
        {
            "id": "merge_forcing",
            "label": "Forcing merge",
            "category": "forcing",
            "description": "Merge per-variable forcing into single time-series per GRU",
            "outputs": ["merged_forcing_nc"],
            "inputs": ["corrected_forcing_nc"],
        },
        {
            "id": "process_streamflow",
            "label": "Streamflow processing",
            "category": "observation",
            "description": "Download and QC WSC streamflow for calibration/evaluation",
            "outputs": ["streamflow_csv"],
            "inputs": ["config_yaml"],
        },
        {
            "id": "process_snow",
            "label": "Snow observation processing",
            "category": "observation",
            "description": "Download SNOTEL SWE, MODIS SCA for validation",
            "outputs": ["snow_obs"],
            "inputs": ["catchment_shapefile"],
        },
        {
            "id": "process_et_grace",
            "label": "ET/GRACE processing",
            "category": "observation",
            "description": "Download MODIS ET, GRACE TWS anomalies for validation",
            "outputs": ["et_obs", "grace_obs"],
            "inputs": ["catchment_shapefile"],
        },
        {
            "id": "model_specific_preprocessing",
            "label": "Model-format conversion",
            "category": "model_setup",
            "description": "Convert CFIF to model-specific format (e.g. SUMMA cold state, forcing structure)",
            "outputs": ["model_ready_inputs"],
            "inputs": ["merged_forcing_nc", "gru_attributes_table", "streamflow_csv"],
        },
    ]

    edges = []
    output_to_stage = {}
    for stage in stages:
        for out in stage["outputs"]:
            output_to_stage[out] = stage["id"]

    for stage in stages:
        for inp in stage["inputs"]:
            if inp in output_to_stage:
                src = output_to_stage[inp]
                if src != stage["id"]:
                    edges.append({"from": src, "to": stage["id"], "data_product": inp})

    return {
        "stages": stages,
        "edges": edges,
        "n_stages": len(stages),
        "n_edges": len(edges),
        "categories": {
            "setup": {"color": "#999999", "label": "Setup"},
            "geospatial": {"color": "#4C72B0", "label": "Geospatial processing"},
            "forcing": {"color": "#DD8452", "label": "Forcing preprocessing"},
            "observation": {"color": "#55A868", "label": "Observation processing"},
            "model_setup": {"color": "#C44E52", "label": "Model setup"},
        },
    }


# -------------------------------------------- 9. Complete variable mapping table
def build_variable_mapping_table(data_dir: Path, domain_info: Dict) -> List[Dict]:
    """
    Build a complete ERA5 -> CFIF variable mapping table with actual values
    observed in the data files.
    """
    try:
        import xarray as xr
    except ImportError:
        return []

    era5_mappings = {
        "airtemp": {
            "era5_name": "t2m",
            "era5_long_name": "2 metre temperature",
            "era5_units": "K",
            "cfif_name": "airtemp",
            "cfif_units": "K",
            "cf_standard_name": "air_temperature",
            "description": "Near-surface air temperature",
        },
        "pptrate": {
            "era5_name": "tp",
            "era5_long_name": "Total precipitation",
            "era5_units": "m (accumulated)",
            "cfif_name": "pptrate",
            "cfif_units": "kg m-2 s-1",
            "cf_standard_name": "precipitation_flux",
            "description": "Precipitation rate (de-accumulated, converted)",
        },
        "LWRadAtm": {
            "era5_name": "strd",
            "era5_long_name": "Surface thermal radiation downwards",
            "era5_units": "J m-2 (accumulated)",
            "cfif_name": "LWRadAtm",
            "cfif_units": "W m-2",
            "cf_standard_name": "surface_downwelling_longwave_flux_in_air",
            "description": "Downward longwave radiation (de-accumulated)",
        },
        "SWRadAtm": {
            "era5_name": "ssrd",
            "era5_long_name": "Surface solar radiation downwards",
            "era5_units": "J m-2 (accumulated)",
            "cfif_name": "SWRadAtm",
            "cfif_units": "W m-2",
            "cf_standard_name": "surface_downwelling_shortwave_flux_in_air",
            "description": "Downward shortwave radiation (de-accumulated)",
        },
        "spechum": {
            "era5_name": "d2m / t2m",
            "era5_long_name": "Specific humidity (derived)",
            "era5_units": "K (dew point) -> kg kg-1",
            "cfif_name": "spechum",
            "cfif_units": "kg kg-1",
            "cf_standard_name": "specific_humidity",
            "description": "Specific humidity (derived from dew-point temperature)",
        },
        "windspd": {
            "era5_name": "u10, v10",
            "era5_long_name": "Wind speed (derived)",
            "era5_units": "m s-1 (components)",
            "cfif_name": "windspd",
            "cfif_units": "m s-1",
            "cf_standard_name": "wind_speed",
            "description": "Wind speed (from U and V components)",
        },
        "airpres": {
            "era5_name": "sp",
            "era5_long_name": "Surface pressure",
            "era5_units": "Pa",
            "cfif_name": "airpres",
            "cfif_units": "Pa",
            "cf_standard_name": "surface_air_pressure",
            "description": "Surface air pressure",
        },
    }

    domain_dir = resolve_domain_dir(data_dir, domain_info)
    raw_dir = domain_dir / "forcing" / "raw_data"
    ba_dir = domain_dir / "forcing" / "basin_averaged_data"

    raw_actual = {}
    ba_actual = {}

    if raw_dir.exists():
        for rf in sorted(raw_dir.glob("*.nc"))[:3]:
            try:
                ds = xr.open_dataset(rf)
                for vname in ds.data_vars:
                    if vname not in raw_actual:
                        raw_actual[vname] = {
                            "units": ds[vname].attrs.get("units", "MISSING"),
                            "standard_name": ds[vname].attrs.get("standard_name", "MISSING"),
                        }
                ds.close()
            except Exception:
                continue

    if ba_dir.exists():
        for bf in sorted(ba_dir.glob("*.nc"))[:3]:
            try:
                ds = xr.open_dataset(bf)
                for vname in ds.data_vars:
                    if vname not in ba_actual:
                        ba_actual[vname] = {
                            "units": ds[vname].attrs.get("units", "MISSING"),
                            "standard_name": ds[vname].attrs.get("standard_name", "MISSING"),
                        }
                ds.close()
            except Exception:
                continue

    table = []
    for cfif_name, mapping in era5_mappings.items():
        row = dict(mapping)
        if cfif_name in raw_actual:
            row["raw_actual_units"] = raw_actual[cfif_name]["units"]
            row["raw_actual_standard_name"] = raw_actual[cfif_name]["standard_name"]
        if cfif_name in ba_actual:
            row["processed_actual_units"] = ba_actual[cfif_name]["units"]
            row["processed_actual_standard_name"] = ba_actual[cfif_name]["standard_name"]
        proc_sname = ba_actual.get(cfif_name, {}).get("standard_name", "MISSING")
        proc_units = ba_actual.get(cfif_name, {}).get("units", "MISSING")
        row["has_standard_name"] = proc_sname not in ("MISSING", "n/a", "")
        row["has_units"] = proc_units not in ("MISSING", "")
        table.append(row)

    return table


# -------------------------------------------- 10. Cross-scale summary
def build_cross_scale_summary(full_report: Dict) -> Dict[str, Any]:
    """
    Compute per-domain totals, compression ratios, and scaling relationships
    across the three canonical domains.
    """
    summary = {"domains": {}, "scaling": {}}

    for domain_key, domain_info in DOMAINS.items():
        dr = full_report.get(domain_key, {})
        if not dr:
            continue

        # Total data volume across all categories
        vols = dr.get("data_volumes", {})
        total_bytes = sum(v.get("total_bytes", 0) for v in vols.values())
        total_files = sum(v.get("n_files", 0) for v in vols.values())

        # Forcing-specific volumes
        raw_forcing = vols.get("Forcing (raw)", {}).get("total_bytes", 0)
        ba_forcing = vols.get("Forcing (basin-averaged)", {}).get("total_bytes", 0)

        # Compression
        comp = dr.get("compression", {})
        overall_ratio = comp.get("overall_compression_ratio", None)

        # Observation count
        obs = dr.get("observation_coverage", {})
        n_obs_types = len(obs)

        domain_summary = {
            "label": domain_info["label"],
            "scale": domain_info["scale"],
            "n_hrus": domain_info["n_hrus"],
            "n_grus": domain_info["n_grus"],
            "raw_grid_cells": domain_info["raw_grid_cells"],
            "area_km2": domain_info["area_km2"],
            "sections": domain_info["sections"],
            "total_bytes": total_bytes,
            "total_human": format_bytes(total_bytes),
            "total_files": total_files,
            "raw_forcing_bytes": raw_forcing,
            "ba_forcing_bytes": ba_forcing,
            "compression_ratio": overall_ratio,
            "n_obs_types": n_obs_types,
            "bytes_per_hru": round(total_bytes / max(domain_info["n_hrus"], 1), 0),
            "bytes_per_km2": round(total_bytes / max(domain_info["area_km2"], 0.001), 0),
        }
        summary["domains"][domain_key] = domain_summary

    # Scaling relationships (log-log HRU count vs data volume)
    hru_counts = []
    total_volumes = []
    compression_ratios = []
    domain_labels = []
    for dk, ds in summary["domains"].items():
        hru_counts.append(ds["n_hrus"])
        total_volumes.append(ds["total_bytes"])
        compression_ratios.append(ds.get("compression_ratio"))
        domain_labels.append(ds["label"])

    summary["scaling"] = {
        "hru_counts": hru_counts,
        "total_volumes_bytes": total_volumes,
        "compression_ratios": compression_ratios,
        "domain_labels": domain_labels,
    }

    return summary


# -------------------------------------------- 11. Data shape tracking
def track_data_shapes(data_dir: Path, domain_info: Dict) -> Dict[str, Any]:
    """
    Record array dimensions at each pipeline stage (raw grid -> remapped -> merged)
    for Sankey/flow figure.
    """
    try:
        import xarray as xr
    except ImportError:
        return {}

    domain_dir = resolve_domain_dir(data_dir, domain_info)
    shapes = {
        "domain": domain_info["label"],
        "scale": domain_info["scale"],
        "n_hrus": domain_info["n_hrus"],
        "stages": {},
    }

    stage_dirs = {
        "raw": domain_dir / "forcing" / "raw_data",
        "basin_averaged": domain_dir / "forcing" / "basin_averaged_data",
        "merged": domain_dir / "forcing" / "merged_path",
    }

    for stage_name, stage_dir in stage_dirs.items():
        if not stage_dir.exists():
            shapes["stages"][stage_name] = {"exists": False}
            continue

        nc_files = sorted(stage_dir.glob("*.nc"))[:3]
        if not nc_files:
            shapes["stages"][stage_name] = {"exists": True, "n_files": 0}
            continue

        stage_info = {
            "exists": True,
            "n_files": len(list(stage_dir.glob("*.nc"))),
            "total_bytes": sum(f.stat().st_size for f in stage_dir.glob("*.nc")),
            "sample_shapes": {},
        }

        for nc in nc_files[:1]:
            try:
                ds = xr.open_dataset(nc)
                stage_info["dimensions"] = {k: int(v) for k, v in ds.dims.items()}
                for vname in ds.data_vars:
                    if vname not in ("latitude", "longitude", "hruId", "time"):
                        stage_info["sample_shapes"][vname] = {
                            "shape": list(ds[vname].shape),
                            "dtype": str(ds[vname].dtype),
                            "nbytes": int(ds[vname].values.nbytes),
                        }
                        break
                ds.close()
            except Exception:
                continue

        shapes["stages"][stage_name] = stage_info

    return shapes


# ----------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Analyse SYMFLUENCE data processing pipeline outputs (Section 4.12)"
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_report: Dict[str, Any] = {}

    # ---- Stage dependency map (shared across domains) ----
    logger.info("Building stage dependency map ...")
    stage_dag = build_stage_dependency_map()
    full_report["stage_dag"] = stage_dag

    for domain_key, domain_info in DOMAINS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Analysing domain: {domain_info['label']} ({domain_key}, "
                     f"scale={domain_info['scale']}, HRUs={domain_info['n_hrus']}, GRUs={domain_info['n_grus']}, "
                     f"area={domain_info['area_km2']} km²)")
        logger.info(f"{'='*60}")

        domain_report: Dict[str, Any] = {
            "domain_info": domain_info,
        }

        # 1. Data volumes
        logger.info("1. Data volume inventory ...")
        domain_report["data_volumes"] = analyze_data_volumes(data_dir, domain_info)

        # 2. CF compliance
        logger.info("2. CF compliance check ...")
        domain_report["cf_compliance"] = check_cf_compliance(data_dir, domain_info)

        # 3. Remapping weights
        logger.info("3. Remapping weight characterisation ...")
        domain_report["remapping_weights"] = analyze_remapping_weights(data_dir, domain_info)

        # 4. Observation coverage
        logger.info("4. Observation coverage ...")
        domain_report["observation_coverage"] = analyze_observation_coverage(data_dir, domain_info)

        # 5. Variable standardisation audit
        logger.info("5. Variable standardisation audit ...")
        domain_report["variable_audit"] = audit_variable_standardisation(data_dir, domain_info)

        # 6. Forcing preprocessing profiling
        logger.info("6. Forcing preprocessing profiling ...")
        domain_report["forcing_profiling"] = profile_forcing_preprocessing(data_dir, domain_info)

        # 7. Data compression analysis
        logger.info("7. Data compression analysis ...")
        domain_report["compression"] = analyze_compression(data_dir, domain_info)

        # 8. Variable mapping table
        logger.info("8. Variable mapping table ...")
        domain_report["variable_mapping"] = build_variable_mapping_table(data_dir, domain_info)

        # 9. Data shape tracking
        logger.info("9. Data shape tracking ...")
        domain_report["data_shapes"] = track_data_shapes(data_dir, domain_info)

        full_report[domain_key] = domain_report

    # ---- Cross-scale summary ----
    logger.info("\nBuilding cross-scale summary ...")
    cross_scale = build_cross_scale_summary(full_report)
    full_report["cross_scale_summary"] = cross_scale

    # ---- Comparative summary ----
    summary_lines = []
    summary_lines.append("SYMFLUENCE Data Processing Pipeline -- Cross-Scale Analysis Summary")
    summary_lines.append("=" * 70)

    for domain_key in DOMAINS:
        dr = full_report.get(domain_key, {})
        di = DOMAINS[domain_key]
        summary_lines.append(f"\n--- {di['label'].upper()} ({di['scale']}, "
                             f"{di['n_hrus']:,} HRUs, {di['n_grus']:,} GRUs, {di['area_km2']} km²) ---")
        summary_lines.append(f"  Paper sections: {', '.join(di['sections'])}")

        vols = dr.get("data_volumes", {})
        for cat, info in vols.items():
            if info.get("exists"):
                summary_lines.append(
                    f"  {cat:<30} {info['total_human']:>10}  ({info['n_files']} files)"
                )

        obs = dr.get("observation_coverage", {})
        if obs:
            summary_lines.append("\n  Observation coverage:")
            for obs_type, info in obs.items():
                gap = info.get("gap_fraction", "N/A")
                rng = info.get("date_range", ["?", "?"])
                summary_lines.append(
                    f"    {obs_type:<20} {rng[0]} - {rng[1]}  gap: {gap}"
                )

        weights = dr.get("remapping_weights", {})
        if weights.get("exists"):
            summary_lines.append(f"\n  Remapping weights: {weights.get('n_files', 0)} file(s)")
            for wf in weights.get("files", []):
                sp = wf.get("sparsity", "N/A")
                summary_lines.append(f"    {wf['file']}: sparsity={sp}, size={wf['size']}")

        comp = dr.get("compression", {})
        if comp:
            summary_lines.append("\n  Compression:")
            summary_lines.append(f"    Raw forcing:       {comp.get('raw_total_human', 'N/A')} "
                                 f"({comp.get('raw_n_files', 0)} files)")
            summary_lines.append(f"    Basin-averaged:    {comp.get('processed_total_human', 'N/A')} "
                                 f"({comp.get('processed_n_files', 0)} files)")
            summary_lines.append(f"    Overall ratio:     {comp.get('overall_compression_ratio', 'N/A')}:1")

        shapes = dr.get("data_shapes", {})
        if shapes.get("stages"):
            summary_lines.append("\n  Data shapes:")
            for stage_name, sinfo in shapes["stages"].items():
                if sinfo.get("exists") and sinfo.get("dimensions"):
                    dims_str = ", ".join(f"{k}={v}" for k, v in sinfo["dimensions"].items())
                    summary_lines.append(f"    {stage_name:<20} dims: {dims_str}")

    # Cross-scale summary
    cs = full_report.get("cross_scale_summary", {})
    if cs.get("domains"):
        summary_lines.append("\n--- CROSS-SCALE COMPARISON ---")
        summary_lines.append(f"  {'Domain':<20} {'Scale':<12} {'HRUs':>8} {'GRUs':>8} {'Area (km²)':>12} "
                             f"{'Total Data':>12} {'Compress.':>10}")
        for dk, ds in cs["domains"].items():
            cr = ds.get("compression_ratio")
            cr_str = f"{cr:.2f}:1" if cr else "N/A"
            summary_lines.append(
                f"  {ds['label']:<20} {ds['scale']:<12} {ds['n_hrus']:>8,} {ds['n_grus']:>8,} "
                f"{ds['area_km2']:>12,.0f} {ds['total_human']:>12} {cr_str:>10}"
            )

    # Stage DAG summary
    dag = full_report.get("stage_dag", {})
    if dag:
        summary_lines.append("\n--- PIPELINE DAG ---")
        summary_lines.append(f"  Stages: {dag.get('n_stages', 0)}")
        summary_lines.append(f"  Edges:  {dag.get('n_edges', 0)}")
        for stage in dag.get("stages", []):
            deps = [e["from"] for e in dag.get("edges", []) if e["to"] == stage["id"]]
            dep_str = f" <- [{', '.join(deps)}]" if deps else " (root)"
            summary_lines.append(f"    {stage['id']:<35}{dep_str}")

    summary_text = "\n".join(summary_lines)
    logger.info("\n" + summary_text)

    # Save outputs
    json_path = output_dir / f"pipeline_analysis_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info(f"Full report: {json_path}")

    txt_path = output_dir / f"pipeline_analysis_{timestamp}.txt"
    txt_path.write_text(summary_text)
    logger.info(f"Summary: {txt_path}")

    # CSV summary table with Domain/Scale/GRUs/Area columns
    rows = []
    for domain_key in DOMAINS:
        dr = full_report.get(domain_key, {})
        di = DOMAINS[domain_key]
        for cat, info in dr.get("data_volumes", {}).items():
            if info.get("exists"):
                rows.append({
                    "Domain": di["label"],
                    "Scale": di["scale"],
                    "HRUs": di["n_hrus"],
                    "GRUs": di["n_grus"],
                    "Area_km2": di["area_km2"],
                    "Category": cat,
                    "Size (bytes)": info["total_bytes"],
                    "Size": info["total_human"],
                    "Files": info["n_files"],
                })
    if rows:
        csv_path = output_dir / f"pipeline_data_volumes_{timestamp}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Volume table: {csv_path}")

    # Compression CSV
    comp_rows = []
    for domain_key in DOMAINS:
        dr = full_report.get(domain_key, {})
        di = DOMAINS[domain_key]
        comp = dr.get("compression", {})
        for vname, vinfo in comp.get("per_variable", {}).items():
            comp_rows.append({
                "Domain": di["label"],
                "Scale": di["scale"],
                "GRUs": di["n_grus"],
                "Area_km2": di["area_km2"],
                "Variable": vname,
                "Raw (bytes)": vinfo["raw_bytes"],
                "Processed (bytes)": vinfo["processed_bytes"],
                "Compression ratio": vinfo.get("compression_ratio", "N/A"),
            })
    if comp_rows:
        csv_path = output_dir / f"pipeline_compression_{timestamp}.csv"
        pd.DataFrame(comp_rows).to_csv(csv_path, index=False)
        logger.info(f"Compression table: {csv_path}")

    # Cross-scale summary CSV
    cs_domains = cs.get("domains", {})
    if cs_domains:
        cs_rows = []
        for dk, ds in cs_domains.items():
            cs_rows.append(ds)
        csv_path = output_dir / f"pipeline_cross_scale_{timestamp}.csv"
        pd.DataFrame(cs_rows).to_csv(csv_path, index=False)
        logger.info(f"Cross-scale summary: {csv_path}")

    # Variable mapping CSV (use first domain that has data)
    for domain_key in DOMAINS:
        dr = full_report.get(domain_key, {})
        mapping = dr.get("variable_mapping", [])
        if mapping:
            csv_path = output_dir / f"pipeline_variable_mapping_{timestamp}.csv"
            pd.DataFrame(mapping).to_csv(csv_path, index=False)
            logger.info(f"Variable mapping: {csv_path}")
            break

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
