#!/usr/bin/env python3
"""
Multi-Model Ensemble Visualization Script for SYMFLUENCE Paper Section 4.2

This script creates publication-quality figures comparing all models:
1. Hydrograph comparison (observed vs all models)
2. Performance metrics bar chart
3. Flow duration curves
4. Scatter plots (obs vs sim)
5. Monthly/seasonal performance
6. Parameter sensitivity (if available)

Usage:
    python visualize_ensemble.py [--output-dir DIR] [--format png|pdf|svg]

Output:
    - fig_hydrograph_comparison.{format}: Time series comparison
    - fig_performance_metrics.{format}: Bar chart of KGE, NSE, etc.
    - fig_flow_duration_curves.{format}: FDC comparison
    - fig_scatter_plots.{format}: Obs vs Sim scatter
    - fig_monthly_performance.{format}: Monthly boxplots
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

# Add SYMFLUENCE to path
SYMFLUENCE_CODE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
sys.path.insert(0, str(SYMFLUENCE_CODE_DIR / "src"))

# Configuration
BASE_DIR = Path(__file__).parent.parent
SIMULATIONS_DIR = BASE_DIR / "simulations"
FIGURES_DIR = BASE_DIR / "figures"

# Model configurations - Full 10-model ensemble
# Models may be in different experiment directories
MODELS = ["HBV", "GR4J", "FUSE", "jFUSE", "SUMMA", "HYPE", "RHESSys", "MESH", "ngen", "LSTM"]
EXPERIMENT_IDS = ["run_1", "run_2", "run_dds", "test_run"]  # All experiment IDs to search
EXPERIMENT_ID = "run_1"  # Primary experiment ID

# Visual styling - 10 distinct colors for all models
MODEL_COLORS = {
    # Conceptual models (warm colors)
    "HBV": "#1f77b4",      # Blue
    "GR4J": "#ff7f0e",     # Orange
    "FUSE": "#2ca02c",     # Green
    "jFUSE": "#d62728",    # Red
    # Process-based models (cool colors)
    "SUMMA": "#9467bd",    # Purple
    "HYPE": "#8c564b",     # Brown
    "RHESSys": "#e377c2",  # Pink
    "MESH": "#bcbd22",     # Yellow-green
    # NextGen framework
    "ngen": "#7f7f7f",     # Gray
    # Machine learning models
    "LSTM": "#17becf",     # Cyan
    "Observed": "black",
}

MODEL_LINESTYLES = {
    "HBV": "-",
    "GR4J": "--",
    "FUSE": "-.",
    "jFUSE": ":",
    "SUMMA": "-",
    "HYPE": "--",
    "RHESSys": "-.",
    "MESH": "-",
    "ngen": "--",
    "LSTM": ":",
    "Observed": "-",
}

# Periods
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
logger = logging.getLogger("ensemble_visualization")

# Set publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def convert_mm_to_cms(values: np.ndarray, area_km2: float = CATCHMENT_AREA_KM2) -> np.ndarray:
    """
    Convert mm/day to m³/s (cms).

    Formula: Q (m³/s) = (P (mm/day) * A (km²) * 1000) / (86400)
                      = P * A / 86.4
    """
    return values * area_km2 / 86.4


def detect_and_convert_units(sim_values: np.ndarray, obs_values: np.ndarray,
                              area_km2: float = CATCHMENT_AREA_KM2) -> Tuple[np.ndarray, bool]:
    """
    Detect if simulation is in mm/day and convert to cms if needed.

    Returns:
        Tuple of (converted_values, was_converted)
    """
    # Remove NaNs for comparison
    sim_clean = sim_values[~np.isnan(sim_values)]
    obs_clean = obs_values[~np.isnan(obs_values)]

    if len(sim_clean) == 0 or len(obs_clean) == 0:
        return sim_values, False

    # Calculate ratio of means
    sim_mean = np.mean(sim_clean)
    obs_mean = np.mean(obs_clean)

    if sim_mean <= 0:
        return sim_values, False

    ratio = obs_mean / sim_mean

    # Expected ratio for mm/day to cms conversion
    # Q_cms = Q_mm * area_km2 / 86.4
    expected_ratio = area_km2 / 86.4  # ~25.6 for 2210 km²

    # If ratio is close to expected conversion factor, data is likely in mm/day
    if 0.5 * expected_ratio < ratio < 2.0 * expected_ratio:
        logger.debug(f"Detected mm/day units (ratio={ratio:.1f}, expected={expected_ratio:.1f}), converting to cms")
        return convert_mm_to_cms(sim_values, area_km2), True

    return sim_values, False


def identify_model_from_params(params: Dict) -> Optional[str]:
    """Identify which model a set of parameters belongs to.

    Order matters - more specific/unique parameter sets should be checked first.
    """
    param_names_lower = set(k.lower() for k in params.keys())

    # SUMMA parameters (unique: maxwatr, baserte, qb_powr)
    summa_params = {'maxwatr_1', 'maxwatr_2', 'baserte', 'qb_powr', 'timedelay', 'percrte', 'fracten'}
    if param_names_lower & summa_params:
        return "SUMMA"

    # RHESSys parameters (unique: sat_to_gw_coeff, gw_loss_coeff)
    rhessys_params = {'sat_to_gw_coeff', 'gw_loss_coeff', 'ksat_0', 'porosity_0', 'soil_depth'}
    if param_names_lower & rhessys_params:
        return "RHESSys"

    # GR4J parameters (unique: x1, x2, x3, x4)
    gr4j_params = {'x1', 'x2', 'x3', 'x4'}
    if param_names_lower & gr4j_params:
        return "GR4J"

    # FUSE parameters (unique: rferr_add, frchzne, percfrac, etc.)
    fuse_specific = {'rferr_add', 'rferr_mlt', 'frchzne', 'percfrac', 'fprimqb', 'qbrate_2a', 'qbrate_2b', 'qb_prms', 'rtfrac1', 'percexp', 'sacpmlt', 'sacpexp', 'iflwrte', 'axv_bexp', 'saession', 'loglamb', 'tishape'}
    if param_names_lower & fuse_specific:
        return "FUSE"

    # jFUSE parameters
    if any(p.startswith('jfuse') for p in param_names_lower):
        return "jFUSE"

    # HYPE parameters - CHECK BEFORE HBV because they share 'lp'
    # Use unique HYPE params: cevp, cmlt, ttmp, rrcs1, rrcs2
    hype_unique = {'cevp', 'cmlt', 'ttmp', 'rrcs1', 'rrcs2', 'rcgrw', 'rivvel', 'epotdist'}
    if param_names_lower & hype_unique:
        return "HYPE"

    # HBV parameters - use unique HBV params (not 'lp' which is shared with HYPE)
    hbv_unique = {'fc', 'beta', 'k0', 'k1', 'k2', 'maxbas', 'perc'}
    if param_names_lower & hbv_unique:
        return "HBV"

    # MESH parameters
    mesh_specific = {'r2n', 'r1n', 'flz', 'pwr', 'zsnl', 'zpls', 'zplg', 'sdep', 'xslp', 'grkf', 'wfci', 'fare', 'drn', 'orgm'}
    if param_names_lower & mesh_specific:
        return "MESH"

    # LSTM parameters
    lstm_params = {'hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size', 'seq_length', 'epochs', 'patience', 'lookback', 'lstm_hidden', 'lstm_layers', 'lr', 'weight_decay'}
    if param_names_lower & lstm_params:
        return "LSTM"

    # ngen CFE parameters (NextGen Conceptual Functional Equivalent)
    ngen_cfe_params = {'maxsmc', 'satdk', 'bb', 'slop', 'smcmax', 'smcwlt', 'refkdt', 'dksat', 'slope', 'expon', 'max_gw_storage', 'cgw', 'k_nash', 'klf', 'kn', 'nash_n', 'giuh'}
    if param_names_lower & ngen_cfe_params:
        return "ngen"

    return None





def normalize_metrics(data: Dict) -> Dict:


    """Normalize metric keys to standard format."""


    result = {}


    cal_metrics = data.get("calibration_metrics", {})


    eval_metrics = data.get("evaluation_metrics", {})





    normalized_cal = {k.replace("Calib_", "").replace("Cal_", ""): v for k, v in cal_metrics.items()}


    result["calibration_metrics"] = normalized_cal





    normalized_eval = {k.replace("Eval_", "").replace("Val_", ""): v for k, v in eval_metrics.items()}


    result["evaluation_metrics"] = normalized_eval





    for key in data:


        if key not in ["calibration_metrics", "evaluation_metrics"]:


            result[key] = data[key]


    return result


def load_summa_netcdf(nc_path: Path) -> Optional[pd.DataFrame]:
    """Load SUMMA simulation results from NetCDF file."""
    try:
        ds = xr.open_dataset(nc_path)

        # Look for streamflow variable
        streamflow_vars = ['averageRoutedRunoff', 'scalarTotalRunoff', 'IRFroutedRunoff',
                          'basin_runoff', 'runoff', 'streamflow', 'discharge']

        sim_var = None
        for var in streamflow_vars:
            if var in ds.data_vars:
                sim_var = var
                break

        if sim_var is None:
            logger.debug(f"No streamflow variable found in {nc_path}")
            ds.close()
            return None

        sim_data = ds[sim_var].values.squeeze()  # Squeeze all extra dimensions
        time_data = pd.to_datetime(ds['time'].values)

        # Check units and apply area conversion if needed (m/s to m³/s)
        units = ds[sim_var].attrs.get('units', '')
        if 'm s-1' in units or 'm/s' in units:
            # SUMMA outputs runoff as depth rate (m/s), need to multiply by area
            # Use catchment area for Bow at Banff (2210 km² = 2.21e9 m²)
            area_m2 = CATCHMENT_AREA_KM2 * 1e6
            sim_data = sim_data * area_m2
            logger.debug(f"Converted SUMMA {sim_var} from m/s to m³/s using area {CATCHMENT_AREA_KM2} km²")

        # Skip first 24 hours (spin-up artifact)
        if len(sim_data) > 24:
            sim_data = sim_data[24:]
            time_data = time_data[24:]

        df = pd.DataFrame({'sim': sim_data}, index=time_data)
        ds.close()
        return df

    except Exception as e:
        logger.debug(f"Failed to load NetCDF {nc_path}: {e}")
        return None


def load_lstm_netcdf(nc_path: Path) -> Optional[pd.DataFrame]:
    """Load LSTM simulation results from NetCDF file."""
    try:
        ds = xr.open_dataset(nc_path)

        # LSTM possible variable names
        lstm_vars = ['predicted_streamflow', 'streamflow', 'prediction', 'output',
                    'q_sim', 'discharge', 'flow']

        sim_var = None
        for var in lstm_vars:
            if var in ds.data_vars:
                sim_var = var
                break

        if sim_var is None:
            logger.debug(f"No streamflow variable found in LSTM output {nc_path}")
            ds.close()
            return None

        sim_data = ds[sim_var].values
        time_data = pd.to_datetime(ds['time'].values)

        # Handle multi-dimensional data
        if sim_data.ndim > 1:
            sim_data = sim_data.flatten()

        df = pd.DataFrame({'sim': sim_data}, index=time_data)
        ds.close()
        return df

    except Exception as e:
        logger.debug(f"Failed to load LSTM NetCDF {nc_path}: {e}")
        return None


def load_hype_output(txt_path: Path) -> Optional[pd.DataFrame]:
    """Load HYPE simulation results from timeCOUT.txt file."""
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
        df = df.rename(columns={df.columns[0]: 'sim'})
        df = df[['sim']]  # Keep only the sim column

        return df

    except Exception as e:
        logger.debug(f"Failed to load HYPE output {txt_path}: {e}")
        return None


def load_fuse_netcdf(nc_path: Path) -> Optional[pd.DataFrame]:
    """Load FUSE simulation results from NetCDF file."""
    try:
        ds = xr.open_dataset(nc_path)

        # FUSE streamflow variables (in order of preference)
        fuse_vars = ['q_routed', 'q_instnt', 'qsurf', 'oflow_1', 'oflow_2']

        sim_var = None
        for var in fuse_vars:
            if var in ds.data_vars:
                sim_var = var
                break

        if sim_var is None:
            logger.debug(f"No streamflow variable found in FUSE output {nc_path}")
            ds.close()
            return None

        # Get data and squeeze extra dimensions
        sim_data = ds[sim_var].values.squeeze()
        time_data = pd.to_datetime(ds['time'].values)

        df = pd.DataFrame({'sim': sim_data}, index=time_data)
        ds.close()
        return df

    except Exception as e:
        logger.debug(f"Failed to load FUSE NetCDF {nc_path}: {e}")
        return None


def scan_all_evaluations(data_dir: Path) -> Dict[str, Path]:


    """Scan all evaluation JSON files and return path to BEST result file for each model."""


    import glob


    import json


    domain_dir = data_dir / "domain_Bow_at_Banff_lumped_era5"


    model_best_kge = {}


    model_best_path = {}





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


            if "best_params" in data:


                model_name = identify_model_from_params(data["best_params"])





            # Fallback: try to guess from path


            if not model_name:


                path_str = str(json_path).lower()


                for m in MODELS:


                    if m.lower() in path_str:


                        model_name = m


                        break





            if model_name:


                normalized = normalize_metrics(data)


                eval_kge = normalized.get("evaluation_metrics", {}).get("KGE", -999)





                if model_name not in model_best_kge or eval_kge > model_best_kge[model_name]:
                    # Look for corresponding result file in the same directory and subdirectories
                    json_dir = Path(json_path).parent

                    # Search for likely result files in this directory AND final_evaluation subdir
                    result_files = []
                    search_dirs = [json_dir]
                    # Add final_evaluation subdirectory if it exists
                    final_eval_dir = json_dir / "final_evaluation"
                    if final_eval_dir.exists():
                        search_dirs.insert(0, final_eval_dir)  # Prioritize final_evaluation

                    # Model-specific file patterns - search model-specific patterns FIRST
                    model_file_patterns = {
                        "GR4J": ["GR_results.csv", "*GR*.csv", "*gr4j*.csv"],
                        "HBV": ["*hbv_output*.csv", "*HBV*.csv"],
                        "HYPE": ["timeCOUT.txt"],
                        "SUMMA": ["*_timestep.nc", "*_day.nc", "*summa*.nc"],
                        "RHESSys": ["rhessys_basin.daily", "*rhessys*.csv"],
                        "LSTM": ["*lstm*.csv", "*LSTM*.csv"],
                        "FUSE": ["*_runs_best.nc", "*fuse*.nc", "*FUSE*.nc", "*fuse*.csv", "*FUSE*.csv"],
                        "jFUSE": ["*jfuse*.csv"],
                        "MESH": ["*mesh*.csv", "*MESH*.csv"],
                        "ngen": ["*ngen*.csv", "*ngen_output*.csv", "*cfe*.csv", "*CFE*.csv"],
                    }

                    # First search for model-specific patterns
                    specific_patterns = model_file_patterns.get(model_name, [])
                    for search_dir in search_dirs:
                        for pat in specific_patterns:
                            result_files.extend(list(search_dir.glob(pat)))

                    # If no model-specific files found, fall back to generic patterns
                    if not result_files:
                        generic_patterns = ["*.csv", "*.nc", "*.txt"]
                        for search_dir in search_dirs:
                            for pat in generic_patterns:
                                result_files.extend(list(search_dir.glob(pat)))





                    # Filter out the json itself and likely non-result files


                    result_files = [f for f in result_files if f.name != Path(json_path).name and "params" not in f.name]





                    if result_files:


                        # Pick the largest file or one matching model name


                        best_file = result_files[0]


                        for rf in result_files:


                            if model_name.lower() in rf.name.lower():


                                best_file = rf


                                break





                        model_best_kge[model_name] = eval_kge


                        model_best_path[model_name] = best_file


                        logger.info(f"Found candidate for {model_name} (KGE={eval_kge:.3f}) at {best_file}")





        except Exception:


            continue





    return model_best_path





def load_all_results(data_dir: Path) -> Dict[str, pd.DataFrame]:


    """Load simulation results for all models using robust search logic."""


    results = {}
    domain_dir = data_dir / "domain_Bow_at_Banff_lumped_era5"





    # Pre-scan for best optimization results


    best_paths = scan_all_evaluations(data_dir)





    for model_name in MODELS:
        # Prioritize the best path found via scanning
        search_paths = []

        # Model-specific optimization directories (new structure: optimization/{model}/dds_{exp_id})
        for exp_id in EXPERIMENT_IDS:
            model_opt_dir = domain_dir / "optimization" / model_name / f"dds_{exp_id}" / "final_evaluation"
            if model_opt_dir.exists():
                search_paths.append(model_opt_dir)
            # Also check uppercase/lowercase variants
            for variant in [model_name.upper(), model_name.lower()]:
                variant_dir = domain_dir / "optimization" / variant / f"dds_{exp_id}" / "final_evaluation"
                if variant_dir.exists():
                    search_paths.append(variant_dir)

        # Legacy shared directory (for backward compatibility)
        shared_final_eval = domain_dir / "optimization" / "dds_run_1" / "final_evaluation"
        if shared_final_eval.exists():
            search_paths.append(shared_final_eval)

        if model_name in best_paths:
            search_paths.append(best_paths[model_name].parent)
            logger.info(f"Prioritizing best found path for {model_name}: {best_paths[model_name]}")





        # Add search paths for all experiment IDs
        for exp_id in EXPERIMENT_IDS:
            search_paths.extend([
                # NEW: Model-specific optimization directories (optimization/{model}/dds_{exp_id})
                domain_dir / "optimization" / model_name / f"dds_{exp_id}" / "final_evaluation",
                domain_dir / "optimization" / model_name / f"dds_{exp_id}",
                domain_dir / "optimization" / model_name.upper() / f"dds_{exp_id}" / "final_evaluation",
                domain_dir / "optimization" / model_name.upper() / f"dds_{exp_id}",
                # Model-specific simulation directories
                domain_dir / "simulations" / exp_id / model_name,
                domain_dir / "simulations" / exp_id / model_name.upper(),
                domain_dir / "simulations" / exp_id / model_name.lower(),
                # Legacy optimization output directories
                domain_dir / "optimization" / f"dds_{exp_id}" / "final_evaluation",
                domain_dir / "optimization" / f"dds_{model_name}_{exp_id}" / "final_evaluation",
                domain_dir / "optimization" / f"dds_{model_name}_run" / "final_evaluation",
                domain_dir / "optimization" / exp_id / "final_evaluation",
                domain_dir / "optimization" / f"dds_{exp_id}",
                # Direct optimization directories
                domain_dir / "optimization" / f"dds_{model_name}_{exp_id}",
                domain_dir / "optimization" / f"dds_{model_name}_run",
                domain_dir / "optimization" / exp_id,
                # Legacy paths
                domain_dir / "simulations" / exp_id,
                # Backup directories (wildcard)
                domain_dir / "optimization" / f"dds_{exp_id}_backup_*",
            ])
        # Add general paths
        search_paths.extend([
            SIMULATIONS_DIR / f"{model_name}_run",
            domain_dir / "optimization",
            # Results directory (contains LSTM output)
            domain_dir / "results",
        ])





        # Expand wildcards in search paths


        expanded_paths = []


        for p in search_paths:
            if "*" in str(p):
                import glob
                for expanded in glob.glob(str(p)):
                    expanded_path = Path(expanded)
                    expanded_paths.append(expanded_path)
                    # Also add final_evaluation subdirectory if it exists
                    final_eval = expanded_path / "final_evaluation"
                    if final_eval.exists():
                        expanded_paths.append(final_eval)
            else:
                expanded_paths.append(p)
        search_paths = expanded_paths





        model_patterns = {


            "HBV": ["*hbv_output*.csv", "*HBV*.csv"],


            "GR4J": ["GR_results.csv", "*GR*.csv", "*gr4j*.csv"],


            "FUSE": ["*_runs_best.nc", "*fuse*.nc", "*FUSE*.nc", "*fuse*.csv", "*FUSE*.csv"],

            "SUMMA": ["run_*_timestep.nc", "run_*_day.nc", "*summa*.nc"],


            "HYPE": ["timeCOUT.txt", "*hype*.csv", "*HYPE*.csv"],


            "RHESSys": ["rhessys_basin.daily", "*rhessys*.csv", "*basin.daily"],


            "MESH": ["*mesh*.csv", "*MESH*.csv"],


            "LSTM": ["*LSTM_output.nc", "*lstm*.nc", "*lstm*.csv", "*LSTM*.csv", "run_2_results.csv", "run_*_results.csv"],


            "jFUSE": ["*jfuse*.csv", "*jFUSE*.csv"],


            "ngen": ["*ngen*.csv", "*ngen_output*.csv", "*cfe*.csv", "*CFE*.csv"],


        }





        patterns = model_patterns.get(model_name, [])


        # Add generic patterns


        patterns.extend([


            f"*{model_name}*output*.csv",


            f"*{model_name}*results*.csv",


            f"*{model_name.lower()}*.csv",


            "simulation*.csv",


            "*best_simulation*.csv",


        ])





        # Add file name from best_paths if available


        if model_name in best_paths:


             patterns.insert(0, best_paths[model_name].name)





        found_df = None


        for path in search_paths:


            if not path.exists():


                continue





            for pattern in patterns:


                files = list(path.glob(pattern)) + list(path.glob(f"**/{pattern}"))





                # Sort files to prioritize "best" or "calibrated" in name if duplicates


                files.sort(key=lambda x: ("best" not in x.name.lower(), x.name))





                for file_path in files:


                    try:


                        if file_path.suffix == '.nc':
                            if model_name == "LSTM" or 'lstm' in file_path.name.lower():
                                found_df = load_lstm_netcdf(file_path)
                            elif model_name == "FUSE" or 'fuse' in file_path.name.lower() or '_runs_' in file_path.name:
                                found_df = load_fuse_netcdf(file_path)
                            else:
                                found_df = load_summa_netcdf(file_path)


                        elif file_path.name == 'timeCOUT.txt':


                            found_df = load_hype_output(file_path)


                        elif file_path.suffix == '.csv' or 'basin.daily' in file_path.name:
                             # RHESSys basin.daily is CSV-like but with space/tab
                            sep = ','
                            if 'basin.daily' in file_path.name: sep = '\s+'

                            df = pd.read_csv(file_path, sep=sep, parse_dates=True, index_col=0, engine='python')
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index)
                            df = df.sort_index()

                            # RHESSys specific handling
                            if model_name == "RHESSys":


                                # Look for 'streamflow' or 'flow' column


                                for c in df.columns:


                                    if 'streamflow' in c or 'flow' in c:


                                        df = df.rename(columns={c: 'sim'})


                                        break





                            # Standardize column names


                            col_mapping = {}


                            for col in df.columns:


                                col_lower = col.lower()


                                if 'sim' in col_lower or 'q_sim' in col_lower or 'discharge' in col_lower:


                                    col_mapping[col] = 'sim'


                                elif 'obs' in col_lower:


                                    col_mapping[col] = 'obs'





                            if col_mapping:


                                df = df.rename(columns=col_mapping)





                            # Fallback column detection
                            if 'sim' not in df.columns:
                                # Priority: model-specific > cms > q_sim > flow/runoff > any
                                best_col = None
                                best_priority = 999
                                for col in df.columns:
                                    col_lower = col.lower()
                                    priority = 999
                                    # Model-specific column matching
                                    if model_name.lower() in col_lower: priority = 0
                                    elif 'cms' in col_lower: priority = 1
                                    elif 'q_sim' in col_lower or col_lower == 'sim': priority = 2
                                    elif any(x in col_lower for x in ['flow', 'runoff', 'streamflow']) and 'mm' not in col_lower: priority = 3





                                    if priority < best_priority:


                                        best_priority = priority


                                        best_col = col





                                if best_col:


                                    df = df.rename(columns={best_col: 'sim'})





                            if 'sim' in df.columns:


                                found_df = df


                            elif len(df.columns) == 1:


                                df.columns = ['sim']


                                found_df = df





                        if found_df is not None:


                            results[model_name] = found_df


                            logger.info(f"Loaded {model_name} from: {file_path}")


                            break


                    except Exception:


                        continue





                if model_name in results:


                    break


            if model_name in results:


                break





        if model_name not in results:


            logger.warning(f"No results found for {model_name}")





    return results


def load_observations(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load observed streamflow data."""
    search_paths = [
        data_dir / "domain_Bow_at_Banff_lumped_era5" / "observations" / "streamflow" / "preprocessed",
        data_dir / "domain_Bow_at_Banff_lumped_era5" / "observations" / "streamflow",
        data_dir / "observations" / "streamflow" / "preprocessed",
        data_dir / "observations" / "streamflow",
        data_dir / "domain_Bow_at_Banff_lumped_era5" / "observations",
    ]

    for path in search_paths:
        if not path.exists():
            continue
        for pattern in ["*processed*.csv", "*obs*.csv", "*streamflow*.csv", "*discharge*.csv", "*flow*.csv"]:
            files = list(path.glob(pattern))
            if files:
                try:
                    df = pd.read_csv(files[0], parse_dates=True, index_col=0)

                    # Standardize column names for observations
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'discharge' in col_lower or 'flow' in col_lower or col_lower == 'q':
                            df = df.rename(columns={col: 'obs'})
                            break

                    logger.info(f"Loaded observations from: {files[0]}")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load {files[0]}: {e}")

    return None


def get_aligned_data(obs: pd.Series, sim: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Align observation and simulation data, resampling obs if needed."""
    # Ensure inputs are Series
    if isinstance(obs, pd.DataFrame): obs = obs.iloc[:, 0]
    if isinstance(sim, pd.DataFrame): sim = sim.iloc[:, 0]

    # Ensure indices are datetime
    if not isinstance(obs.index, pd.DatetimeIndex):
        obs.index = pd.to_datetime(obs.index)
    if not isinstance(sim.index, pd.DatetimeIndex):
        sim.index = pd.to_datetime(sim.index)

    # Remove duplicates
    obs = obs[~obs.index.duplicated(keep='first')]
    sim = sim[~sim.index.duplicated(keep='first')]

    # Simple heuristic: if sim is much shorter than obs, resample obs to daily
    if len(sim) < len(obs) * 0.5:
        obs = obs.resample('D').mean()
    elif len(obs) < len(sim) * 0.5:
        sim = sim.resample('D').mean()

    # Intersection
    common_idx = obs.index.intersection(sim.index)

    if len(common_idx) == 0:
        return np.array([]), np.array([])

    return obs.loc[common_idx].values, sim.loc[common_idx].values


def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Compute performance metrics."""
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]

    if len(obs) == 0:
        return {"KGE": np.nan, "NSE": np.nan, "RMSE": np.nan, "PBIAS": np.nan}

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
    rmse = np.sqrt(np.mean((obs - sim)**2))
    pbias = 100 * np.sum(sim - obs) / np.sum(obs)

    return {"KGE": kge, "NSE": nse, "RMSE": rmse, "PBIAS": pbias, "r": r}


def get_obs_column(df: pd.DataFrame) -> str:
    """Find the observation column in a DataFrame."""
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'obs' or 'discharge' in col_lower or 'flow' in col_lower or col_lower == 'q':
            return col
    # Fallback to first column
    return df.columns[0]


def get_sim_column(df: pd.DataFrame) -> str:
    """Find the simulation column in a DataFrame, prioritizing cms units."""
    # First priority: columns explicitly in cms
    for col in df.columns:
        col_lower = col.lower()
        if 'cms' in col_lower or 'm3/s' in col_lower or 'm³/s' in col_lower:
            return col

    # Second priority: sim column or discharge/flow without mm
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'sim':
            return col
        if ('discharge' in col_lower or 'flow' in col_lower or 'q_sim' in col_lower) and 'mm' not in col_lower:
            return col

    # Third priority: any sim-like column
    for col in df.columns:
        col_lower = col.lower()
        if 'sim' in col_lower or 'discharge' in col_lower or 'flow' in col_lower:
            return col

    # Fallback to first column
    return df.columns[0]


def plot_hydrograph_comparison(
    all_data: Dict[str, pd.DataFrame],
    obs_df: pd.DataFrame,
    output_path: Path,
    period: str = "evaluation",
) -> None:
    """Create hydrograph comparison plot."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Determine period
    if period == "evaluation":
        start, end = EVALUATION_PERIOD
        title = "Evaluation Period"
    else:
        start, end = CALIBRATION_PERIOD
        title = "Calibration Period"

    # Get observations
    obs_col = get_obs_column(obs_df)
    obs_data = obs_df.loc[start:end, obs_col]

    # Top panel: Full time series
    ax1 = axes[0]
    ax1.plot(obs_data.index, obs_data.values, color=MODEL_COLORS["Observed"],
             linewidth=1.5, label="Observed", zorder=10)

    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        sim_data = df.loc[start:end, sim_col]
        ax1.plot(sim_data.index, sim_data.values, color=MODEL_COLORS.get(model_name, "gray"),
                 linestyle=MODEL_LINESTYLES.get(model_name, "-"),
                 linewidth=1.0, label=model_name, alpha=0.8)

    ax1.set_ylabel("Streamflow (m³/s)")
    ax1.set_title(f"Streamflow Comparison - {title}")
    ax1.legend(loc="upper right", ncol=3)
    ax1.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Log scale for low flows
    ax2 = axes[1]
    ax2.semilogy(obs_data.index, obs_data.values, color=MODEL_COLORS["Observed"],
                 linewidth=1.5, label="Observed", zorder=10)

    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        sim_data = df.loc[start:end, sim_col]
        ax2.semilogy(sim_data.index, sim_data.values, color=MODEL_COLORS.get(model_name, "gray"),
                     linestyle=MODEL_LINESTYLES.get(model_name, "-"),
                     linewidth=1.0, label=model_name, alpha=0.8)

    ax2.set_ylabel("Streamflow (m³/s, log scale)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved hydrograph comparison to: {output_path}")


def plot_performance_metrics(
    all_data: Dict[str, pd.DataFrame],
    obs_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create bar chart of performance metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics_cal = {}
    metrics_eval = {}

    obs_col = get_obs_column(obs_df)

    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)

        # Calibration period
        cal_obs = obs_df.loc[CALIBRATION_PERIOD[0]:CALIBRATION_PERIOD[1], obs_col]
        cal_sim = df.loc[CALIBRATION_PERIOD[0]:CALIBRATION_PERIOD[1], sim_col]
        metrics_cal[model_name] = compute_metrics(*get_aligned_data(cal_obs, cal_sim))

        # Evaluation period
        eval_obs = obs_df.loc[EVALUATION_PERIOD[0]:EVALUATION_PERIOD[1], obs_col]
        eval_sim = df.loc[EVALUATION_PERIOD[0]:EVALUATION_PERIOD[1], sim_col]
        metrics_eval[model_name] = compute_metrics(*get_aligned_data(eval_obs, eval_sim))

    # Plot calibration metrics
    ax1 = axes[0]
    metrics_to_plot = ["KGE", "NSE", "r"]
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    for i, model_name in enumerate(all_data.keys()):
        values = [metrics_cal[model_name].get(m, 0) for m in metrics_to_plot]
        ax1.bar(x + i * width, values, width, label=model_name,
                color=MODEL_COLORS.get(model_name, "gray"))

    ax1.set_ylabel("Metric Value")
    ax1.set_title("Calibration Period Performance")
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(metrics_to_plot)
    ax1.legend()
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_ylim(-0.5, 1.1)
    ax1.grid(True, axis="y", alpha=0.3)

    # Plot evaluation metrics
    ax2 = axes[1]
    for i, model_name in enumerate(all_data.keys()):
        values = [metrics_eval[model_name].get(m, 0) for m in metrics_to_plot]
        ax2.bar(x + i * width, values, width, label=model_name,
                color=MODEL_COLORS.get(model_name, "gray"))

    ax2.set_ylabel("Metric Value")
    ax2.set_title("Evaluation Period Performance")
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(metrics_to_plot)
    ax2.legend()
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_ylim(-0.5, 1.1)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved performance metrics to: {output_path}")


def plot_flow_duration_curves(
    all_data: Dict[str, pd.DataFrame],
    obs_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create flow duration curves comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use evaluation period
    start, end = EVALUATION_PERIOD
    obs_col = get_obs_column(obs_df)
    obs_values = obs_df.loc[start:end, obs_col].dropna().values

    # Calculate observed FDC
    obs_sorted = np.sort(obs_values)[::-1]
    obs_exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
    ax.semilogy(obs_exceedance, obs_sorted, color=MODEL_COLORS["Observed"],
                linewidth=2, label="Observed")

    # Plot each model's FDC
    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        sim_values = df.loc[start:end, sim_col].dropna().values
        sim_sorted = np.sort(sim_values)[::-1]
        sim_exceedance = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted) * 100
        ax.semilogy(sim_exceedance, sim_sorted, color=MODEL_COLORS.get(model_name, "gray"),
                    linestyle=MODEL_LINESTYLES.get(model_name, "-"),
                    linewidth=1.5, label=model_name, alpha=0.8)

    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_title("Flow Duration Curves - Evaluation Period")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved FDC to: {output_path}")


def plot_scatter_comparison(
    all_data: Dict[str, pd.DataFrame],
    obs_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create scatter plots (obs vs sim) for all models."""
    n_models = len(all_data)
    ncols = min(2, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    start, end = EVALUATION_PERIOD
    obs_col = get_obs_column(obs_df)

    for idx, (model_name, df) in enumerate(all_data.items()):
        ax = axes[idx]
        sim_col = get_sim_column(df)

        obs_series = obs_df.loc[start:end, obs_col]
        sim_series = df.loc[start:end, sim_col]

        obs_values, sim_values = get_aligned_data(obs_series, sim_series)

        # Remove NaNs
        mask = ~np.isnan(obs_values) & ~np.isnan(sim_values)
        obs_clean = obs_values[mask]
        sim_clean = sim_values[mask]

        if len(obs_clean) == 0:
            ax.text(0.5, 0.5, "No overlapping data", ha='center', va='center')
            ax.set_title(model_name)
            continue

        # Plot scatter
        ax.scatter(obs_clean, sim_clean, alpha=0.5, s=20,
                   color=MODEL_COLORS.get(model_name, "gray"), edgecolors="none")

        # Add 1:1 line
        max_val = max(obs_clean.max(), sim_clean.max())
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="1:1 line")

        # Compute metrics
        metrics = compute_metrics(obs_clean, sim_clean)

        # Add metrics text
        text = f"KGE = {metrics['KGE']:.3f}\nNSE = {metrics['NSE']:.3f}\nr = {metrics['r']:.3f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("Observed (m³/s)")
        ax.set_ylabel("Simulated (m³/s)")
        ax.set_title(model_name)
        ax.set_aspect("equal")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved scatter plots to: {output_path}")


def plot_monthly_performance(
    all_data: Dict[str, pd.DataFrame],
    obs_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create monthly performance comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    start, end = EVALUATION_PERIOD
    obs_col = get_obs_column(obs_df)

    for idx, (model_name, df) in enumerate(all_data.items()):
        if idx >= 4:
            break

        ax = axes.flatten()[idx]
        sim_col = get_sim_column(df)

        # Align data
        try:
            obs_monthly = obs_df.loc[start:end, obs_col].resample("M").mean()
            sim_monthly = df.loc[start:end, sim_col].resample("M").mean()

            # Ensure 1D
            if isinstance(sim_monthly, pd.DataFrame):
                sim_monthly = sim_monthly.iloc[:, 0]
            if isinstance(obs_monthly, pd.DataFrame):
                obs_monthly = obs_monthly.iloc[:, 0]

            if len(obs_monthly) == 0 or len(sim_monthly) == 0:
                raise ValueError("Empty monthly data")

            # Merge and group by month
            merged = pd.DataFrame({"obs": obs_monthly, "sim": sim_monthly}).dropna()
            merged["month"] = merged.index.month

            # Calculate monthly metrics
            monthly_kge = []
            for month in range(1, 13):
                month_data = merged[merged["month"] == month]
                if len(month_data) > 0:
                    metrics = compute_metrics(month_data["obs"].values, month_data["sim"].values)
                    monthly_kge.append(metrics["KGE"])
                else:
                    monthly_kge.append(np.nan)
        except Exception:
            # Handle empty/missing data gracefully
            monthly_kge = [np.nan] * 12

        # Plot
        months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        bars = ax.bar(months, monthly_kge, color=MODEL_COLORS.get(model_name, "gray"), alpha=0.7)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Monthly KGE")
        ax.set_title(f"{model_name} - Monthly Performance")
        ax.set_ylim(-1, 1)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved monthly performance to: {output_path}")


def create_summary_figure(
    all_data: Dict[str, pd.DataFrame],
    obs_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a comprehensive summary figure combining multiple panels."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel A: Hydrograph (top, spans 2 columns)
    ax_hydro = fig.add_subplot(gs[0, :2])

    start, end = EVALUATION_PERIOD
    obs_col = get_obs_column(obs_df)
    obs_data = obs_df.loc[start:end, obs_col]

    ax_hydro.plot(obs_data.index, obs_data.values, color=MODEL_COLORS["Observed"],
                  linewidth=1.5, label="Observed", zorder=10)

    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        sim_data = df.loc[start:end, sim_col]
        ax_hydro.plot(sim_data.index, sim_data.values, color=MODEL_COLORS.get(model_name, "gray"),
                      linestyle=MODEL_LINESTYLES.get(model_name, "-"),
                      linewidth=1.0, label=model_name, alpha=0.8)

    ax_hydro.set_ylabel("Streamflow (m³/s)")
    ax_hydro.set_title("(a) Streamflow Comparison - Evaluation Period")
    ax_hydro.legend(loc="upper right", ncol=3)
    ax_hydro.grid(True, alpha=0.3)
    ax_hydro.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_hydro.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Panel B: Performance metrics (top right)
    ax_metrics = fig.add_subplot(gs[0, 2])

    metrics_to_plot = ["KGE", "NSE"]
    x = np.arange(len(metrics_to_plot))
    width = 0.18

    for i, (model_name, df) in enumerate(all_data.items()):
        sim_col = get_sim_column(df)
        eval_obs = obs_df.loc[start:end, obs_col]
        eval_sim = df.loc[start:end, sim_col]
        metrics = compute_metrics(*get_aligned_data(eval_obs, eval_sim))
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        ax_metrics.bar(x + i * width, values, width, label=model_name,
                       color=MODEL_COLORS.get(model_name, "gray"))

    ax_metrics.set_ylabel("Metric Value")
    ax_metrics.set_title("(b) Evaluation Metrics")
    ax_metrics.set_xticks(x + width * 1.5)
    ax_metrics.set_xticklabels(metrics_to_plot)
    ax_metrics.legend(fontsize=8)
    ax_metrics.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax_metrics.set_ylim(-0.5, 1.1)
    ax_metrics.grid(True, axis="y", alpha=0.3)

    # Panel C: Flow duration curve (bottom left)
    ax_fdc = fig.add_subplot(gs[1, 0])

    obs_values = obs_df.loc[start:end, obs_col].dropna().values
    obs_sorted = np.sort(obs_values)[::-1]
    obs_exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
    ax_fdc.semilogy(obs_exceedance, obs_sorted, color=MODEL_COLORS["Observed"],
                    linewidth=2, label="Observed")

    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        sim_values = df.loc[start:end, sim_col].dropna().values
        sim_sorted = np.sort(sim_values)[::-1]
        sim_exceedance = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted) * 100
        ax_fdc.semilogy(sim_exceedance, sim_sorted, color=MODEL_COLORS.get(model_name, "gray"),
                        linestyle=MODEL_LINESTYLES.get(model_name, "-"),
                        linewidth=1.5, alpha=0.8)

    ax_fdc.set_xlabel("Exceedance (%)")
    ax_fdc.set_ylabel("Flow (m³/s)")
    ax_fdc.set_title("(c) Flow Duration Curves")
    ax_fdc.set_xlim(0, 100)
    ax_fdc.grid(True, alpha=0.3)

    # Panel D: Scatter for best model (bottom middle)
    ax_scatter = fig.add_subplot(gs[1, 1])

    # Find best model by KGE
    best_kge = -999
    best_model = None
    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        eval_obs = obs_df.loc[start:end, obs_col]
        eval_sim = df.loc[start:end, sim_col]
        metrics = compute_metrics(*get_aligned_data(eval_obs, eval_sim))
        if metrics["KGE"] > best_kge:
            best_kge = metrics["KGE"]
            best_model = model_name

    if best_model:
        df = all_data[best_model]
        sim_col = get_sim_column(df)
        obs_series = obs_df.loc[start:end, obs_col]
        sim_series = df.loc[start:end, sim_col]

        obs_values, sim_values = get_aligned_data(obs_series, sim_series)
        mask = ~np.isnan(obs_values) & ~np.isnan(sim_values)

        ax_scatter.scatter(obs_values[mask], sim_values[mask], alpha=0.5, s=20,
                           color=MODEL_COLORS.get(best_model, "gray"), edgecolors="none")
        max_val = max(obs_values[mask].max(), sim_values[mask].max())
        ax_scatter.plot([0, max_val], [0, max_val], "k--", linewidth=1)
        ax_scatter.set_xlabel("Observed (m³/s)")
        ax_scatter.set_ylabel("Simulated (m³/s)")
        ax_scatter.set_title(f"(d) {best_model} (Best KGE={best_kge:.3f})")
        ax_scatter.set_aspect("equal")
        ax_scatter.grid(True, alpha=0.3)

    # Panel E: Model comparison table (bottom right)
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis("off")

    # Create table data
    table_data = [["Model", "KGE", "NSE", "PBIAS"]]
    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        eval_obs = obs_df.loc[start:end, obs_col]
        eval_sim = df.loc[start:end, sim_col]
        metrics = compute_metrics(*get_aligned_data(eval_obs, eval_sim))
        table_data.append([
            model_name,
            f"{metrics['KGE']:.3f}",
            f"{metrics['NSE']:.3f}",
            f"{metrics['PBIAS']:.1f}%",
        ])

    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.2, 0.2, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax_table.set_title("(e) Performance Summary", pad=20)

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved summary figure to: {output_path}")


def run_visualization(data_dir: Path, output_dir: Path, fig_format: str = "png") -> None:
    """Run all visualizations."""
    logger.info("Starting Multi-Model Ensemble Visualization")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_data = load_all_results(data_dir)
    obs_df = load_observations(data_dir)

    if not all_data:
        logger.error("No model results found!")
        return

    if obs_df is None:
        logger.error("No observations found!")
        return

    logger.info(f"Loaded results for {len(all_data)} models: {list(all_data.keys())}")

    # Apply unit conversion if needed (mm/day to m³/s)
    obs_col = get_obs_column(obs_df)
    obs_values = obs_df[obs_col].values

    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        sim_col_lower = sim_col.lower()

        # Skip conversion if column name indicates already in cms
        if 'cms' in sim_col_lower or 'm3/s' in sim_col_lower or 'm³/s' in sim_col_lower:
            logger.info(f"{model_name} column '{sim_col}' already in m³/s, skipping conversion")
            continue

        sim_values = df[sim_col].values
        converted_values, was_converted = detect_and_convert_units(sim_values, obs_values)
        if was_converted:
            df[sim_col] = converted_values
            logger.info(f"Converted {model_name} from mm/day to m³/s")

    # Filter out models with invalid (near-zero) data
    valid_models = {}
    for model_name, df in all_data.items():
        sim_col = get_sim_column(df)
        sim_values = df[sim_col].dropna().values
        if len(sim_values) > 0 and np.nanmax(np.abs(sim_values)) > 0.01:
            valid_models[model_name] = df
        else:
            logger.warning(f"Skipping {model_name}: data is invalid or near-zero (max={np.nanmax(np.abs(sim_values)) if len(sim_values) > 0 else 0:.6f})")

    all_data = valid_models
    logger.info(f"Using {len(all_data)} models with valid data: {list(all_data.keys())}")

    # Create all figures
    plot_hydrograph_comparison(
        all_data, obs_df,
        output_dir / f"fig_hydrograph_comparison.{fig_format}",
        period="evaluation"
    )

    plot_performance_metrics(
        all_data, obs_df,
        output_dir / f"fig_performance_metrics.{fig_format}"
    )

    plot_flow_duration_curves(
        all_data, obs_df,
        output_dir / f"fig_flow_duration_curves.{fig_format}"
    )

    plot_scatter_comparison(
        all_data, obs_df,
        output_dir / f"fig_scatter_plots.{fig_format}"
    )

    plot_monthly_performance(
        all_data, obs_df,
        output_dir / f"fig_monthly_performance.{fig_format}"
    )

    create_summary_figure(
        all_data, obs_df,
        output_dir / f"fig_ensemble_summary.{fig_format}"
    )

    logger.info(f"\nAll figures saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create visualizations for multi-model ensemble"
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
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Figure output format",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else FIGURES_DIR

    run_visualization(data_dir, output_dir, args.format)


if __name__ == "__main__":
    main()
