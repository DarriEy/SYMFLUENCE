#!/usr/bin/env python3
"""
Multi-Model Ensemble Analysis Script for SYMFLUENCE Paper Section 4.2

This script analyzes and compares results from all model calibrations,
computing performance metrics and generating summary statistics.

Usage:
    python analyze_ensemble.py [--output-dir DIR]

Output:
    - ensemble_metrics.csv: Performance metrics for all models
    - ensemble_comparison.csv: Side-by-side comparison table
    - model_rankings.csv: Models ranked by different metrics
"""

import argparse
import glob
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("Warning: xarray not available, NetCDF reading disabled")

# Add SYMFLUENCE to path
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

# Configuration
BASE_DIR = Path(__file__).parent.parent
SIMULATIONS_DIR = BASE_DIR / "simulations"
ANALYSIS_DIR = BASE_DIR / "analysis"

# Model configurations - Full 10-model ensemble
# Models may be in different experiment directories
MODELS = ["HBV", "GR4J", "FUSE", "jFUSE", "SUMMA", "HYPE", "RHESSys", "MESH", "ngen", "LSTM"]
EXPERIMENT_IDS = ["run_1", "run_2", "run_dds", "test_run"]  # All experiment IDs to search
EXPERIMENT_ID = "run_1"  # Primary experiment ID

# Periods for evaluation
CALIBRATION_PERIOD = ("2004-01-01", "2007-12-31")
EVALUATION_PERIOD = ("2008-01-01", "2009-12-31")

# Catchment area for unit conversion (mm/day to cms)
# Bow River at Banff catchment area
CATCHMENT_AREA_KM2 = 2210.0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ensemble_analysis")


def convert_mm_to_cms(values: np.ndarray, area_km2: float = CATCHMENT_AREA_KM2) -> np.ndarray:
    """
    Convert streamflow from mm/day to m³/s (cms).

    Formula: Q_cms = Q_mm/day * Area_km2 / 86.4
    (86.4 = 86400 seconds/day / 1000 mm/m / 1000000 m2/km2)

    Args:
        values: Streamflow in mm/day
        area_km2: Catchment area in km²

    Returns:
        Streamflow in m³/s
    """
    return values * area_km2 / 86.4


def detect_and_convert_units(sim: np.ndarray, obs: np.ndarray, area_km2: float = CATCHMENT_AREA_KM2) -> Tuple[np.ndarray, bool]:
    """
    Detect if simulation is in mm/day and convert to cms if needed.

    Uses the ratio between means to detect unit mismatch.

    Args:
        sim: Simulated streamflow array
        obs: Observed streamflow array (assumed to be in cms)
        area_km2: Catchment area for conversion

    Returns:
        Tuple of (converted_sim, was_converted)
    """
    # Remove NaN for calculation
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs_clean = obs[mask]
    sim_clean = sim[mask]

    if len(obs_clean) == 0 or np.mean(sim_clean) == 0:
        return sim, False

    ratio = np.mean(obs_clean) / np.mean(sim_clean)

    # If ratio is close to the expected conversion factor, likely mm/day
    # Expected ratio for mm -> cms conversion: area_km2 / 86.4 ≈ 25.6 for this catchment
    expected_ratio = area_km2 / 86.4

    # Allow some tolerance (within factor of 2)
    if expected_ratio / 2 < ratio < expected_ratio * 2:
        logger.info(f"Detected mm/day units (ratio={ratio:.1f}, expected={expected_ratio:.1f}), converting to cms")
        return convert_mm_to_cms(sim, area_km2), True

    return sim, False


def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """
    Compute performance metrics between observed and simulated streamflow.

    Args:
        obs: Observed streamflow array
        sim: Simulated streamflow array

    Returns:
        Dictionary of metric names to values
    """
    # Remove NaN values
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return {
            "KGE": np.nan,
            "NSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "PBIAS": np.nan,
            "r": np.nan,
        }

    # KGE components
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # NSE
    nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)

    # RMSE
    rmse = np.sqrt(np.mean((obs - sim)**2))

    # MAE
    mae = np.mean(np.abs(obs - sim))

    # Percent Bias
    pbias = 100 * np.sum(sim - obs) / np.sum(obs)

    return {
        "KGE": kge,
        "NSE": nse,
        "RMSE": rmse,
        "MAE": mae,
        "PBIAS": pbias,
        "r": r,
        "alpha": alpha,
        "beta": beta,
    }


def identify_model_from_params(params: Dict) -> Optional[str]:
    """
    Identify which model a set of parameters belongs to.

    Args:
        params: Dictionary of parameter names and values

    Returns:
        Model name or None if not identified
    """
    # Convert param names to lowercase for case-insensitive matching
    param_names_lower = set(k.lower() for k in params.keys())

    # SUMMA parameters (case-insensitive)
    summa_params = {'maxwatr_1', 'maxwatr_2', 'baserte', 'qb_powr', 'timedelay', 'percrte', 'fracten'}
    if param_names_lower & summa_params:
        return "SUMMA"

    # RHESSys parameters (case-insensitive)
    rhessys_params = {'sat_to_gw_coeff', 'gw_loss_coeff', 'ksat_0', 'porosity_0', 'soil_depth'}
    if param_names_lower & rhessys_params:
        return "RHESSys"

    # HBV parameters (case-insensitive)
    hbv_params = {'fc', 'lp', 'beta', 'k0', 'k1', 'k2', 'maxbas', 'perc'}
    if param_names_lower & hbv_params:
        return "HBV"

    # GR4J parameters (case-insensitive)
    gr4j_params = {'x1', 'x2', 'x3', 'x4'}
    if param_names_lower & gr4j_params:
        return "GR4J"

    # FUSE parameters (case-insensitive)
    fuse_params = {'rferr_add', 'rferr_mlt', 'frchzne', 'fracten', 'maxwatr_1', 'percfrac', 'fprimqb', 'qbrate_2a', 'qbrate_2b', 'qb_prms', 'maxwatr_2', 'baserte', 'rtfrac1', 'percrte', 'percexp', 'sacpmlt', 'sacpexp', 'iflwrte', 'axv_bexp', 'saession', 'timedelay', 'loglamb', 'tishape', 'qb_powr'}
    if param_names_lower & fuse_params and not (param_names_lower & summa_params):
        # Distinguish FUSE from SUMMA by checking for FUSE-specific params
        fuse_specific = {'rferr_add', 'rferr_mlt', 'frchzne', 'percfrac', 'fprimqb', 'qbrate_2a', 'qbrate_2b', 'qb_prms', 'rtfrac1', 'percexp', 'sacpmlt', 'sacpexp', 'iflwrte', 'axv_bexp', 'saession', 'loglamb', 'tishape'}
        if param_names_lower & fuse_specific:
            return "FUSE"

    # jFUSE parameters (Java FUSE implementation - similar to FUSE but may have different naming)
    jfuse_params = {'jfuse_', 'maxwatr1', 'maxwatr2', 'fracten', 'frchzne', 'fprimqb'}
    # Check if any param starts with jfuse_ or matches jFUSE-specific patterns
    if any(p.startswith('jfuse') for p in param_names_lower):
        return "jFUSE"
    # jFUSE often uses slightly different naming conventions
    jfuse_specific = {'rferr_add', 'rferr_mlt'}
    if param_names_lower & jfuse_specific and param_names_lower & {'maxwatr_1', 'maxwatr_2'}:
        # If it has FUSE-like params but not identified as FUSE, could be jFUSE
        pass  # Will be caught by FUSE or path-based detection

    # HYPE parameters (case-insensitive)
    hype_params = {'cevp', 'cmlt', 'ttmp', 'cfmax', 'fc', 'lp', 'alfa', 'beta', 'k4', 'perc', 'cap', 'kperc', 'wp', 'wcep', 'wcfc', 'cevpam', 'cevpph', 'deadl', 'denitr', 'denitw', 'decay', 'depthrel', 'rrcs1', 'rrcs2', 'rrcs3', 'cevpcorr', 'cmltcorr', 'sfdlim', 'ttpi', 'cmrad'}
    hype_specific = {'cevp', 'cmlt', 'ttmp', 'cfmax', 'alfa', 'k4', 'cap', 'kperc', 'wcep', 'wcfc', 'cevpam', 'cevpph', 'deadl', 'denitr', 'denitw', 'decay', 'depthrel', 'rrcs1', 'rrcs2', 'rrcs3'}
    if param_names_lower & hype_specific:
        return "HYPE"

    # MESH parameters (case-insensitive)
    mesh_params = {'r2n', 'r1n', 'flz', 'pwr', 'dd', 'zsnl', 'zpls', 'zplg', 'sdep', 'xslp', 'grkf', 'mann', 'ksat', 'wfci', 'mid', 'fare', 'dd', 'drn', 'sand', 'clay', 'orgm'}
    mesh_specific = {'r2n', 'r1n', 'flz', 'pwr', 'zsnl', 'zpls', 'zplg', 'sdep', 'xslp', 'grkf', 'wfci', 'fare', 'drn', 'orgm'}
    if param_names_lower & mesh_specific:
        return "MESH"

    # LSTM parameters (neural network hyperparameters)
    lstm_params = {'hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size', 'seq_length', 'epochs', 'patience', 'lookback', 'lstm_hidden', 'lstm_layers', 'lr', 'weight_decay'}
    if param_names_lower & lstm_params:
        return "LSTM"

    # ngen CFE parameters (NextGen Conceptual Functional Equivalent)
    ngen_cfe_params = {'maxsmc', 'satdk', 'bb', 'slop', 'smcmax', 'smcwlt', 'refkdt', 'dksat', 'slope', 'expon', 'max_gw_storage', 'cgw', 'k_nash', 'klf', 'kn', 'nash_n', 'giuh'}
    if param_names_lower & ngen_cfe_params:
        return "ngen"

    return None


def normalize_metrics(data: Dict) -> Dict:
    """
    Normalize metric keys to standard format (KGE, NSE, etc.).

    Handles different formats like Calib_KGE -> KGE, Eval_KGE -> KGE.
    """
    result = {}

    cal_metrics = data.get("calibration_metrics", {})
    eval_metrics = data.get("evaluation_metrics", {})

    # Normalize calibration metrics
    normalized_cal = {}
    for key, value in cal_metrics.items():
        # Remove prefixes like "Calib_"
        clean_key = key.replace("Calib_", "").replace("Cal_", "")
        normalized_cal[clean_key] = value
    result["calibration_metrics"] = normalized_cal

    # Normalize evaluation metrics
    normalized_eval = {}
    for key, value in eval_metrics.items():
        # Remove prefixes like "Eval_"
        clean_key = key.replace("Eval_", "").replace("Val_", "")
        normalized_eval[clean_key] = value
    result["evaluation_metrics"] = normalized_eval

    # Copy other fields
    for key in data:
        if key not in ["calibration_metrics", "evaluation_metrics"]:
            result[key] = data[key]

    return result


def scan_all_evaluations(data_dir: Path) -> Dict[str, Dict]:
    """
    Scan all evaluation JSON files and identify which model each belongs to.

    Keeps track of best results for each model (highest evaluation KGE).

    Args:
        data_dir: Base data directory

    Returns:
        Dictionary mapping model names to their metrics
    """
    exp_id = EXPERIMENT_ID
    domain_dir = data_dir / "domain_Bow_at_Banff_lumped_era5"
    model_metrics = {}
    model_best_kge = {}  # Track best KGE for each model

    # Find all evaluation JSON files with comprehensive patterns
    patterns = [
        str(domain_dir / "optimization" / "*" / "*final_evaluation.json"),
        str(domain_dir / "optimization" / "*" / "*" / "*final_evaluation.json"),
        str(domain_dir / "optimization" / "*" / "final_evaluation.json"),
        str(domain_dir / "optimization" / "*" / "*evaluation*.json"),
        str(domain_dir / "optimization" / "**" / "*evaluation*.json"),
        str(domain_dir / "simulations" / "*" / "*" / "*evaluation*.json"),
        str(domain_dir / "simulations" / "**" / "*evaluation*.json"),
    ]

    all_files = set()
    for pattern in patterns:
        all_files.update(glob.glob(pattern, recursive=True))

    for json_path in all_files:
        try:
            with open(json_path) as f:
                data = json.load(f)

            model_name = None

            # Try to identify the model from parameters
            if "best_params" in data:
                model_name = identify_model_from_params(data["best_params"])

            # Fallback: try to identify from path name
            if not model_name:
                path_str = str(json_path).lower()
                for m in MODELS:
                    if m.lower() in path_str:
                        model_name = m
                        logger.debug(f"Identified {m} from path: {json_path}")
                        break

            if model_name:
                # Normalize metrics to standard format
                normalized = normalize_metrics(data)

                # Get evaluation KGE
                eval_kge = normalized.get("evaluation_metrics", {}).get("KGE")
                if eval_kge is None:
                    eval_kge = -999

                # Keep best result for each model
                if model_name not in model_best_kge or eval_kge > model_best_kge[model_name]:
                    model_best_kge[model_name] = eval_kge
                    model_metrics[model_name] = normalized
                    logger.info(f"Found {model_name} results (KGE={eval_kge:.3f}) in: {json_path}")
        except Exception as e:
            logger.debug(f"Failed to parse {json_path}: {e}")

    return model_metrics


def load_evaluation_json(model_name: str, data_dir: Path) -> Optional[Dict]:
    """
    Load pre-computed metrics from JSON evaluation files.

    Args:
        model_name: Name of the model
        data_dir: Base data directory

    Returns:
        Dictionary with calibration_metrics and evaluation_metrics, or None
    """
    exp_id = EXPERIMENT_ID
    domain_dir = data_dir / "domain_Bow_at_Banff_lumped_era5"

    # Search paths for evaluation JSON files
    # Include backup directories and model-specific directories
    search_patterns = [
        domain_dir / "optimization" / f"dds_{exp_id}" / f"{exp_id}_dds_final_evaluation.json",
        domain_dir / "optimization" / f"dds_{model_name}_{exp_id}" / f"{exp_id}_dds_final_evaluation.json",
        domain_dir / "optimization" / f"dds_{model_name}_run" / "*final_evaluation.json",
        domain_dir / "optimization" / exp_id / f"{exp_id}_dds_final_evaluation.json",
    ]

    # Also search backup directories (sorted by timestamp, newest first)
    backup_pattern = str(domain_dir / "optimization" / f"dds_{exp_id}_backup_*")
    backup_dirs = sorted(glob.glob(backup_pattern), reverse=True)
    for backup_dir in backup_dirs:
        search_patterns.append(Path(backup_dir) / f"{exp_id}_dds_final_evaluation.json")

    for pattern in search_patterns:
        if isinstance(pattern, Path) and pattern.exists():
            try:
                with open(pattern) as f:
                    data = json.load(f)

                # Verify this is the right model by checking parameters
                if "best_params" in data:
                    detected_model = identify_model_from_params(data["best_params"])
                    if detected_model and detected_model != model_name:
                        logger.debug(f"Skipping {pattern}: belongs to {detected_model}, not {model_name}")
                        continue

                logger.info(f"Loaded {model_name} metrics from: {pattern}")
                return data
            except Exception as e:
                logger.debug(f"Failed to load {pattern}: {e}")
        elif "*" in str(pattern):
            files = glob.glob(str(pattern))
            for f in files:
                try:
                    with open(f) as fp:
                        data = json.load(fp)

                    # Verify this is the right model
                    if "best_params" in data:
                        detected_model = identify_model_from_params(data["best_params"])
                        if detected_model and detected_model != model_name:
                            continue

                    logger.info(f"Loaded {model_name} metrics from: {f}")
                    return data
                except Exception as e:
                    logger.debug(f"Failed to load {f}: {e}")

    return None


def load_summa_netcdf(nc_path: Path) -> Optional[pd.DataFrame]:
    """
    Load SUMMA simulation results from NetCDF file.

    Args:
        nc_path: Path to NetCDF file

    Returns:
        DataFrame with datetime index and 'sim' column (streamflow in cms)
    """
    if not HAS_XARRAY:
        logger.warning("xarray not available, cannot read NetCDF files")
        return None

    try:
        ds = xr.open_dataset(nc_path)

        # Look for streamflow variable (averageRoutedRunoff in timestep files)
        streamflow_vars = ['averageRoutedRunoff', 'scalarTotalRunoff', 'IRFroutedRunoff']

        sim_var = None
        for var in streamflow_vars:
            if var in ds.data_vars:
                sim_var = var
                break

        if sim_var is None:
            logger.debug(f"No streamflow variable found in {nc_path}")
            ds.close()
            return None

        # Extract time series
        sim_data = ds[sim_var].values
        time_data = pd.to_datetime(ds['time'].values)

        # Handle multi-dimensional data (time, hru)
        if sim_data.ndim > 1:
            sim_data = sim_data[:, 0]  # Take first HRU for lumped model

        df = pd.DataFrame({
            'sim': sim_data
        }, index=time_data)

        ds.close()
        return df

    except Exception as e:
        logger.debug(f"Failed to load NetCDF {nc_path}: {e}")
        return None


def load_lstm_netcdf(nc_path: Path) -> Optional[pd.DataFrame]:
    """
    Load LSTM simulation results from NetCDF file.

    Args:
        nc_path: Path to NetCDF file

    Returns:
        DataFrame with datetime index and 'sim' column (streamflow in cms)
    """
    if not HAS_XARRAY:
        logger.warning("xarray not available, cannot read NetCDF files")
        return None

    try:
        ds = xr.open_dataset(nc_path)

        # LSTM uses 'predicted_streamflow' variable
        if 'predicted_streamflow' not in ds.data_vars:
            logger.debug(f"No predicted_streamflow variable found in {nc_path}")
            ds.close()
            return None

        sim_data = ds['predicted_streamflow'].values
        time_data = pd.to_datetime(ds['time'].values)

        df = pd.DataFrame({
            'sim': sim_data
        }, index=time_data)

        ds.close()
        return df

    except Exception as e:
        logger.debug(f"Failed to load LSTM NetCDF {nc_path}: {e}")
        return None


def load_hype_output(txt_path: Path) -> Optional[pd.DataFrame]:
    """
    Load HYPE simulation results from timeCOUT.txt file.

    Args:
        txt_path: Path to timeCOUT.txt file

    Returns:
        DataFrame with datetime index and 'sim' column (streamflow in cms)
    """
    try:
        # HYPE timeCOUT.txt format:
        # Line 1: Comment starting with !!
        # Line 2: Header (DATE, subbasin IDs)
        # Line 3+: Data (date, values)
        df = pd.read_csv(
            txt_path,
            sep='\t',
            skiprows=1,  # Skip the !! comment line
            parse_dates=['DATE'],
            index_col='DATE'
        )

        # Rename the first value column to 'sim'
        # HYPE may have multiple subbasins, take the first one
        df = df.rename(columns={df.columns[0]: 'sim'})
        df = df[['sim']]  # Keep only the sim column

        # HYPE outputs are already in m³/s
        return df

    except Exception as e:
        logger.debug(f"Failed to load HYPE output {txt_path}: {e}")
        return None


def load_model_results(model_name: str, data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load simulation results for a model.

    Args:
        model_name: Name of the model
        data_dir: Base data directory

    Returns:
        DataFrame with datetime index, obs, and sim columns
    """
    domain_dir = data_dir / "domain_Bow_at_Banff_lumped_era5"

    # Look for calibration results in various locations across all experiment IDs
    search_paths = []

    for exp_id in EXPERIMENT_IDS:
        # Model-specific simulation directories
        search_paths.extend([
            domain_dir / "simulations" / exp_id / model_name,
            domain_dir / "simulations" / exp_id / model_name.upper(),
            domain_dir / "simulations" / exp_id / model_name.lower(),
            # Optimization output directories
            domain_dir / "optimization" / f"dds_{exp_id}" / "final_evaluation",
            domain_dir / "optimization" / f"dds_{model_name}_{exp_id}" / "final_evaluation",
            domain_dir / "optimization" / f"dds_{model_name}_run" / "final_evaluation",
            domain_dir / "optimization" / exp_id / "final_evaluation",
            domain_dir / "optimization" / f"dds_{exp_id}",
            # Legacy paths
            domain_dir / "simulations" / exp_id,
            domain_dir / "optimization" / exp_id,
        ])

        # Add backup directories to search paths
        backup_pattern = str(domain_dir / "optimization" / f"dds_{exp_id}_backup_*")
        for backup_dir in glob.glob(backup_pattern):
            search_paths.append(Path(backup_dir) / "final_evaluation")
            search_paths.append(Path(backup_dir))

    # Add general paths
    search_paths.extend([
        SIMULATIONS_DIR / f"{model_name}_run",
        domain_dir / "optimization",
    ])

    results_df = None
    searched_paths = []

    for path in search_paths:
        if not path.exists():
            continue
        searched_paths.append(str(path))

        # Model-specific file patterns
        model_patterns = {
            "HBV": ["*hbv_output*.csv", "*HBV*.csv"],
            "GR4J": ["GR_results.csv", "*GR*.csv", "*gr4j*.csv"],
            "FUSE": ["*fuse*.csv", "*FUSE*.csv"],
            "SUMMA": ["run_*_timestep.nc", "run_*_day.nc", "*summa*.nc"],
            "HYPE": ["timeCOUT.txt", "*hype*.csv", "*HYPE*.csv"],
            "RHESSys": ["rhessys_basin.daily", "*rhessys*.csv"],
            "MESH": ["*mesh*.csv", "*MESH*.csv"],
            "LSTM": ["*LSTM_output.nc", "*lstm*.nc", "*lstm*.csv", "*LSTM*.csv"],
            "jFUSE": ["*jfuse*.csv", "*jFUSE*.csv"],
            "ngen": ["*ngen*.csv", "*ngen_output*.csv", "*cfe*.csv", "*CFE*.csv"],
        }

        # Get patterns for this model, plus generic patterns
        patterns = model_patterns.get(model_name, [])
        patterns.extend([
            f"*{model_name}*output*.csv",
            f"*{model_name}*results*.csv",
            f"*{model_name.lower()}*.csv",
            "simulation*.csv",
            "*best_simulation*.csv",
        ])

        for pattern in patterns:
            files = list(path.glob(pattern)) + list(path.glob(f"**/{pattern}"))
            if not files:
                continue

            for file_path in files:
                try:
                    # Handle NetCDF files
                    if file_path.suffix == '.nc':
                        if model_name == "LSTM" or 'lstm' in file_path.name.lower():
                            results_df = load_lstm_netcdf(file_path)
                        else:
                            results_df = load_summa_netcdf(file_path)
                        if results_df is not None:
                            logger.info(f"Loaded {model_name} results from NetCDF: {file_path}")
                            break
                    # Handle HYPE timeCOUT.txt files
                    elif file_path.name == 'timeCOUT.txt':
                        results_df = load_hype_output(file_path)
                        if results_df is not None:
                            logger.info(f"Loaded {model_name} results from HYPE output: {file_path}")
                            break
                    # Handle CSV files
                    elif file_path.suffix == '.csv':
                        df = pd.read_csv(file_path, parse_dates=True, index_col=0)

                        # Standardize column names
                        col_mapping = {}
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'sim' in col_lower or 'q_sim' in col_lower:
                                col_mapping[col] = 'sim'
                            elif 'streamflow_cms' in col_lower or 'discharge' in col_lower:
                                col_mapping[col] = 'sim'
                            elif 'streamflow_mm' in col_lower and 'cms' not in col_lower:
                                continue  # Skip mm columns if cms exists
                            elif 'obs' in col_lower or 'observed' in col_lower:
                                col_mapping[col] = 'obs'

                        if col_mapping:
                            df = df.rename(columns=col_mapping)

                        # If no 'sim' column found, try to identify the streamflow column
                        # Prefer cms columns over mm columns
                        if 'sim' not in df.columns:
                            # Priority order: cms > q_sim > generic flow/runoff
                            best_col = None
                            best_priority = 999
                            for col in df.columns:
                                col_lower = col.lower()
                                priority = 999
                                # Highest priority: explicit cms
                                if 'cms' in col_lower:
                                    priority = 1
                                # Second priority: q_sim or sim
                                elif 'q_sim' in col_lower or col_lower == 'sim':
                                    priority = 2
                                # Third priority: generic flow/runoff (but not mm)
                                elif any(x in col_lower for x in ['flow', 'runoff']) and 'mm' not in col_lower:
                                    priority = 3
                                # Lowest priority: mm columns (only use if nothing else)
                                elif any(x in col_lower for x in ['flow', 'runoff', 'q_']):
                                    priority = 4

                                if priority < best_priority:
                                    best_priority = priority
                                    best_col = col

                            if best_col:
                                df = df.rename(columns={best_col: 'sim'})
                                logger.debug(f"Selected column '{best_col}' as sim (priority {best_priority})")

                        if 'sim' in df.columns or len(df.columns) == 1:
                            if 'sim' not in df.columns:
                                df.columns = ['sim']
                            results_df = df
                            logger.info(f"Loaded {model_name} results from CSV: {file_path}")
                            break

                except Exception as e:
                    logger.debug(f"Failed to load {file_path}: {e}")

            if results_df is not None:
                break

        if results_df is not None:
            break

    if results_df is None and searched_paths:
        logger.debug(f"Searched paths for {model_name}: {searched_paths}")

    return results_df


def load_observations(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load observed streamflow data."""
    search_paths = [
        data_dir / "domain_Bow_at_Banff_lumped_era5" / "observations" / "streamflow" / "preprocessed",
        data_dir / "domain_Bow_at_Banff_lumped_era5" / "observations" / "streamflow",
        data_dir / "observations" / "streamflow" / "preprocessed",
        data_dir / "observations" / "streamflow",
    ]

    for path in search_paths:
        if not path.exists():
            continue
        for pattern in ["*processed*.csv", "*obs*.csv", "*streamflow*.csv"]:
            files = list(path.glob(pattern))
            if files:
                try:
                    obs_df = pd.read_csv(files[0], parse_dates=True, index_col=0)

                    # Standardize column names
                    for col in obs_df.columns:
                        col_lower = col.lower()
                        if 'discharge' in col_lower or 'flow' in col_lower or col_lower == 'q':
                            obs_df = obs_df.rename(columns={col: 'obs'})
                            break

                    logger.info(f"Loaded observations from: {files[0]}")
                    return obs_df
                except Exception as e:
                    logger.warning(f"Failed to load {files[0]}: {e}")

    return None


def analyze_model(
    model_name: str,
    data_dir: Path,
    obs_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze a single model's performance.

    First tries to load pre-computed metrics from JSON files,
    then falls back to computing metrics from time series data.

    Args:
        model_name: Name of the model
        data_dir: Base data directory
        obs_df: Pre-loaded observations DataFrame (optional)

    Returns:
        Dictionary with calibration and evaluation metrics
    """
    results = {
        "calibration": {},
        "evaluation": {},
        "full_period": {},
    }

    # First, try to load pre-computed metrics from JSON
    json_metrics = load_evaluation_json(model_name, data_dir)
    if json_metrics is not None:
        if "calibration_metrics" in json_metrics:
            results["calibration"] = json_metrics["calibration_metrics"]
        if "evaluation_metrics" in json_metrics:
            results["evaluation"] = json_metrics["evaluation_metrics"]
        logger.info(f"Using pre-computed metrics for {model_name}")
        return results

    # Fall back to loading time series and computing metrics
    sim_df = load_model_results(model_name, data_dir)
    if sim_df is None:
        logger.warning(f"No results found for {model_name}")
        return results

    # Load observations if not provided
    if obs_df is None:
        obs_df = load_observations(data_dir)
    if obs_df is None:
        logger.warning("No observations found")
        return results

    # Align time series
    try:
        if "sim" in sim_df.columns and "obs" in sim_df.columns:
            # Results already have obs column
            merged = sim_df
        else:
            # Need to merge with observations
            # Find the right columns
            obs_col = None
            for c in obs_df.columns:
                if c == "obs" or "flow" in c.lower() or "discharge" in c.lower() or c.lower() == "q":
                    obs_col = c
                    break

            # Check for 'sim' column first (already renamed in load_model_results)
            if 'sim' in sim_df.columns:
                sim_col = 'sim'
            else:
                sim_col = None
                for c in sim_df.columns:
                    if "sim" in c.lower() or "flow" in c.lower():
                        sim_col = c
                        break

            if obs_col is None:
                obs_col = obs_df.columns[0]
            if sim_col is None:
                sim_col = sim_df.columns[0]

            # Resample to daily if needed (for hourly data)
            if hasattr(obs_df.index, 'freq') or len(obs_df) > 10000:
                obs_daily = obs_df[[obs_col]].resample('D').mean()
            else:
                obs_daily = obs_df[[obs_col]]

            if hasattr(sim_df.index, 'freq') or len(sim_df) > 10000:
                sim_daily = sim_df[[sim_col]].resample('D').mean()
            else:
                sim_daily = sim_df[[sim_col]]

            merged = pd.merge(
                obs_daily.rename(columns={obs_col: "obs"}),
                sim_daily.rename(columns={sim_col: "sim"}),
                left_index=True,
                right_index=True,
                how="inner",
            )

        # Check for unit mismatch and convert if needed
        sim_values, was_converted = detect_and_convert_units(
            merged["sim"].values,
            merged["obs"].values
        )
        if was_converted:
            merged["sim"] = sim_values

        # Compute metrics for each period
        cal_mask = (merged.index >= CALIBRATION_PERIOD[0]) & (merged.index <= CALIBRATION_PERIOD[1])
        eval_mask = (merged.index >= EVALUATION_PERIOD[0]) & (merged.index <= EVALUATION_PERIOD[1])

        if cal_mask.any():
            results["calibration"] = compute_metrics(
                merged.loc[cal_mask, "obs"].values,
                merged.loc[cal_mask, "sim"].values,
            )

        if eval_mask.any():
            results["evaluation"] = compute_metrics(
                merged.loc[eval_mask, "obs"].values,
                merged.loc[eval_mask, "sim"].values,
            )

        results["full_period"] = compute_metrics(
            merged["obs"].values,
            merged["sim"].values,
        )

    except Exception as e:
        logger.error(f"Error analyzing {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return results


def create_comparison_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a side-by-side comparison table of all models."""
    rows = []

    for model_name, results in all_results.items():
        row = {"Model": model_name}

        # Add calibration metrics
        for metric, value in results.get("calibration", {}).items():
            row[f"Cal_{metric}"] = value

        # Add evaluation metrics
        for metric, value in results.get("evaluation", {}).items():
            row[f"Eval_{metric}"] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("Model")

    return df


def create_rankings(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Create rankings of models by different metrics."""
    if comparison_df.empty:
        return pd.DataFrame()

    rankings = {}

    # Metrics where higher is better
    higher_better = ["KGE", "NSE", "r"]
    # Metrics where lower is better
    lower_better = ["RMSE", "MAE"]
    # PBIAS - closer to 0 is better
    pbias_cols = [c for c in comparison_df.columns if "PBIAS" in c]

    for col in comparison_df.columns:
        metric = col.split("_")[-1]
        if metric in higher_better:
            rankings[col] = comparison_df[col].rank(ascending=False)
        elif metric in lower_better:
            rankings[col] = comparison_df[col].rank(ascending=True)
        elif col in pbias_cols:
            rankings[col] = comparison_df[col].abs().rank(ascending=True)

    rankings_df = pd.DataFrame(rankings, index=comparison_df.index)

    # Add average rank
    rankings_df["Avg_Rank"] = rankings_df.mean(axis=1)

    return rankings_df.sort_values("Avg_Rank")


def run_analysis(data_dir: Path, output_dir: Path) -> None:
    """
    Run the full ensemble analysis.

    Args:
        data_dir: SYMFLUENCE data directory
        output_dir: Directory to save analysis results
    """
    logger.info("Starting Multi-Model Ensemble Analysis")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, scan all evaluation files to see what's available
    logger.info("\nScanning for available model results...")
    available_models = scan_all_evaluations(data_dir)
    if available_models:
        logger.info(f"Found pre-computed results for: {list(available_models.keys())}")
    else:
        logger.info("No pre-computed evaluation JSON files found")

    # Load observations once
    obs_df = load_observations(data_dir)
    if obs_df is not None:
        logger.info(f"Observations loaded: {len(obs_df)} records from {obs_df.index.min()} to {obs_df.index.max()}")

    # Analyze each model
    all_results = {}
    for model_name in MODELS:
        logger.info(f"\nAnalyzing {model_name}...")

        # First check if we found this model in the scan
        if model_name in available_models:
            data = available_models[model_name]
            results = {
                "calibration": data.get("calibration_metrics", {}),
                "evaluation": data.get("evaluation_metrics", {}),
                "full_period": {},
            }
            logger.info(f"Using pre-scanned metrics for {model_name}")
        else:
            results = analyze_model(model_name, data_dir, obs_df)

        all_results[model_name] = results

    # Create comparison table
    comparison_df = create_comparison_table(all_results)

    # Create rankings
    rankings_df = create_rankings(comparison_df)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed metrics
    metrics_file = output_dir / f"ensemble_metrics_{timestamp}.csv"
    comparison_df.to_csv(metrics_file)
    logger.info(f"Saved metrics to: {metrics_file}")

    # Save rankings
    rankings_file = output_dir / f"model_rankings_{timestamp}.csv"
    rankings_df.to_csv(rankings_file)
    logger.info(f"Saved rankings to: {rankings_file}")

    # Save summary report
    report_file = output_dir / f"analysis_report_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("Multi-Model Ensemble Analysis Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")

        f.write("Models Analyzed:\n")
        for model in MODELS:
            f.write(f"  - {model}\n")
        f.write("\n")

        f.write(f"Calibration Period: {CALIBRATION_PERIOD[0]} to {CALIBRATION_PERIOD[1]}\n")
        f.write(f"Evaluation Period: {EVALUATION_PERIOD[0]} to {EVALUATION_PERIOD[1]}\n")
        f.write("\n")

        f.write("Performance Comparison (Evaluation Period):\n")
        f.write("-" * 60 + "\n")
        if not comparison_df.empty:
            eval_cols = [c for c in comparison_df.columns if c.startswith("Eval_")]
            if eval_cols:
                f.write(comparison_df[eval_cols].to_string())
        f.write("\n\n")

        f.write("Model Rankings (lower is better):\n")
        f.write("-" * 60 + "\n")
        if not rankings_df.empty:
            f.write(rankings_df.to_string())
        f.write("\n")

    logger.info(f"Saved report to: {report_file}")

    # Print summary to console
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    if not comparison_df.empty:
        print("\nEvaluation Period Metrics:")
        eval_cols = [c for c in comparison_df.columns if c.startswith("Eval_")]
        if eval_cols:
            print(comparison_df[eval_cols].round(3).to_string())

    if not rankings_df.empty:
        print("\nOverall Rankings:")
        print(rankings_df[["Avg_Rank"]].round(2).to_string())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze multi-model ensemble results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data",
        help="SYMFLUENCE data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR

    run_analysis(data_dir, output_dir)


if __name__ == "__main__":
    main()
