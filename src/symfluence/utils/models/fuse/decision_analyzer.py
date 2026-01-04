import os
import sys
import time
import subprocess
from shutil import rmtree, copyfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime
import rasterio # type: ignore
from scipy import ndimage
import csv
import itertools
from typing import Dict, List, Tuple, Any
import xarray as xr # type: ignore
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import PETCalculatorMixin
from symfluence.utils.common.constants import UnitConversion

sys.path.append(str(Path(__file__).resolve().parent.parent))
from symfluence.utils.common.metrics import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE
from symfluence.utils.data.utilities.variable_utils import VariableHandler # type: ignore


class FuseDecisionAnalyzer:
    def __init__(self, config, logger, reporting_manager=None):
        self.config = config
        self.logger = logger
        self.reporting_manager = reporting_manager
        self.data_dir = Path(self.config_dict.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config_dict.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_folder = self.project_dir / "reporting" / "FUSE_decision_analysis"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.model_decisions_path = self.project_dir / "settings" / "FUSE" / f"fuse_zDecisions_{self.config_dict.get('EXPERIMENT_ID')}.txt"

        # Initialize FuseRunner
        self.fuse_runner = FUSERunner(config, logger)

        # Get decision options from config or use defaults
        self.decision_options = self._initialize_decision_options()
        
        # Log the decision options being used
        self.logger.info("Initialized FUSE decision options:")
        for decision, options in self.decision_options.items():
            self.logger.info(f"{decision}: {options}")

        # Add storage for simulation results
        self.simulation_results = {}
        self.observed_streamflow = None
        self.area_km2 = None

    def _initialize_decision_options(self) -> Dict[str, List[str]]:
        """
        Initialize decision options from config file or use defaults.
        
        Returns:
            Dict[str, List[str]]: Dictionary of decision options
        """
        # Default decision options as fallback
        default_options = {
            'RFERR': ['additive_e', 'multiplc_e'],
            'ARCH1': ['tension1_1', 'tension2_1', 'onestate_1'],
            'ARCH2': ['tens2pll_2', 'unlimfrc_2', 'unlimpow_2', 'fixedsiz_2'],
            'QSURF': ['arno_x_vic', 'prms_varnt', 'tmdl_param'],
            'QPERC': ['perc_f2sat', 'perc_w2sat', 'perc_lower'],
            'ESOIL': ['sequential', 'rootweight'],
            'QINTF': ['intflwnone', 'intflwsome'],
            'Q_TDH': ['rout_gamma', 'no_routing'],
            'SNOWM': ['temp_index', 'no_snowmod']
        }

        # Try to get decision options from config
        config_options = self.config_dict.get('FUSE_DECISION_OPTIONS')
        
        if config_options:
            self.logger.info("Using decision options from config file")
            
            # Validate config options
            validated_options = {}
            for decision, options in default_options.items():
                if decision in config_options:
                    # Ensure options are in list format
                    config_decision_options = config_options[decision]
                    if isinstance(config_decision_options, list):
                        validated_options[decision] = config_decision_options
                    else:
                        self.logger.warning(
                            f"Invalid options format for decision {decision} in config. "
                            f"Using defaults: {options}"
                        )
                        validated_options[decision] = options
                else:
                    self.logger.warning(
                        f"Decision {decision} not found in config. "
                        f"Using defaults: {options}"
                    )
                    validated_options[decision] = options
            
            return validated_options
        else:
            self.logger.info("No decision options found in config. Using defaults.")
            return default_options

    def generate_combinations(self) -> List[Tuple[str, ...]]:
        """Generate all possible combinations of model decisions."""
        return list(itertools.product(*self.decision_options.values()))

    def update_model_decisions(self, combination: Tuple[str, ...]):
        """
        Update the FUSE model decisions file with a new combination.
        Only updates the decision values (first string) in lines 2-10.
        
        Args:
            combination (Tuple[str, ...]): Tuple of decision values to use
        """
        self.logger.info("Updating FUSE model decisions")
        
        try:
            with open(self.model_decisions_path, 'r') as f:
                lines = f.readlines()
            
            # The decisions are in lines 2-10 (1-based indexing)
            decision_lines = range(1, 10)  # Python uses 0-based indexing
            
            # Create a mapping of decision keys to new values
            decision_keys = list(self.decision_options.keys())
            option_map = dict(zip(decision_keys, combination))
            
            # For debugging
            self.logger.debug(f"Updating with new values: {option_map}")
            
            # Update only the first part of each decision line
            for line_idx in decision_lines:
                # Split the line into components
                line_parts = lines[line_idx].split()
                if len(line_parts) >= 2:
                    # Get the decision key (RFERR, ARCH1, etc.)
                    decision_key = line_parts[1]  # Key is the second part
                    if decision_key in option_map:
                        # Replace the first part with the new value
                        new_value = option_map[decision_key]
                        # Keep the rest of the line (key and any comments) unchanged
                        rest_of_line = ' '.join(line_parts[1:])
                        lines[line_idx] = f"{new_value:<10} {rest_of_line}\n"
                        self.logger.debug(f"Updated line {line_idx + 1}: {lines[line_idx].strip()}")
            
            # Write the updated content back to the file
            with open(self.model_decisions_path, 'w') as f:
                f.writelines(lines)
                
        except Exception as e:
            self.logger.error(f"Error updating model decisions: {str(e)}")
            raise

    def get_current_decisions(self) -> List[str]:
        """Read current decisions from the FUSE decisions file."""
        with open(self.model_decisions_path, 'r') as f:
            lines = f.readlines()
        
        # Extract the first word from lines 2-10
        decisions = []
        for line in lines[1:10]:  # Lines 2-10 in 1-based indexing
            decision = line.strip().split()[0]
            decisions.append(decision)
        
        return decisions

    def calculate_performance_metrics(self) -> Tuple[float, float, float, float, float]:
        """Calculate performance metrics comparing simulated and observed streamflow."""
        obs_file_path = self.config_dict.get('OBSERVATIONS_PATH')
        if obs_file_path == 'default':
            obs_file_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.config_dict.get('DOMAIN_NAME')}_streamflow_processed.csv"
        else:
            obs_file_path = Path(obs_file_path)

        sim_file_path = self.project_dir / 'simulations' / self.config_dict.get('EXPERIMENT_ID') / 'FUSE' / f"{self.config_dict.get('DOMAIN_NAME')}_{self.config_dict.get('EXPERIMENT_ID')}_runs_best.nc"

        # Read observations if not already loaded
        if self.observed_streamflow is None:
            dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
            self.observed_streamflow = dfObs['discharge_cms'].resample('d').mean()

        # Read simulations
        dfSim = xr.open_dataset(sim_file_path, decode_timedelta=True)
        dfSim = dfSim['q_routed'].isel(
                                param_set=0,
                                latitude=0,
                                longitude=0
                            )
        dfSim = dfSim.to_pandas()

        # Get area from river basins shapefile using GRU_area if not already calculated
        if self.area_km2 is None:
            basin_name = self.config_dict.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.config_dict.get('DOMAIN_NAME')}_riverBasins_{self.config_dict.get('DOMAIN_DEFINITION_METHOD')}.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            # Sum the GRU_area column and convert from m2 to km2
            self.area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            self.logger.info(f"Total catchment area from GRU_area: {self.area_km2:.2f} km2")
        
        # Convert units from mm/day to cms
        # Q(cms) = Q(mm/day) * Area(km2) / 86.4
        dfSim = dfSim * self.area_km2 / UnitConversion.MM_DAY_TO_CMS

        # Store this simulation result
        current_combo = tuple(self.get_current_decisions())
        self.simulation_results[current_combo] = dfSim

        # Align timestamps and handle missing values
        dfObs = self.observed_streamflow.reindex(dfSim.index).dropna()
        dfSim = dfSim.reindex(dfObs.index).dropna()

        # Calculate metrics
        obs = dfObs.values
        sim = dfSim.values
        
        kge = get_KGE(obs, sim, transfo=1)
        kgep = get_KGEp(obs, sim, transfo=1)
        nse = get_NSE(obs, sim, transfo=1)
        mae = get_MAE(obs, sim, transfo=1)
        rmse = get_RMSE(obs, sim, transfo=1)

        return kge, kgep, nse, mae, rmse

    def run_decision_analysis(self):
        """
        Run the complete FUSE decision analysis workflow, including generating plots and analyzing results.
        
        Returns:
            Tuple[Path, Dict]: Path to results file and dictionary of best combinations
        """
        self.logger.info("Starting FUSE decision analysis")
        
        combinations = self.generate_combinations()
        self.logger.info(f"Generated {len(combinations)} decision combinations")

        optimization_dir = self.project_dir / 'optimization'
        optimization_dir.mkdir(parents=True, exist_ok=True)

        master_file = optimization_dir / f"{self.config_dict.get('EXPERIMENT_ID')}_fuse_decisions_comparison.csv"

        # Write header to master file
        with open(master_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration'] + list(self.decision_options.keys()) + 
                          ['kge', 'kgep', 'nse', 'mae', 'rmse'])

        for i, combination in enumerate(combinations, 1):
            self.logger.info(f"Running combination {i} of {len(combinations)}")
            self.update_model_decisions(combination)
            
            try:
                # Run FUSE model
                self.fuse_runner.run_fuse()
                
                # Calculate performance metrics
                kge, kgep, nse, mae, rmse = self.calculate_performance_metrics()

                # Write results to master file
                with open(master_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + list(combination) + [kge, kgep, nse, mae, rmse])

                self.logger.info(f"Combination {i} completed: KGE={kge:.3f}, KGEp={kgep:.3f}, "
                               f"NSE={nse:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

            except Exception as e:
                self.logger.error(f"Error in combination {i}: {str(e)}")
                with open(master_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + list(combination) + ['erroneous combination'])

        self.logger.info("FUSE decision analysis completed")
        
        if self.reporting_manager:
            # Create hydrograph plots for different metrics
            for metric in ['kge', 'nse', 'kgep']:
                self.reporting_manager.visualize_hydrographs_with_highlight(
                    master_file, 
                    self.simulation_results, 
                    self.observed_streamflow,
                    self.decision_options,
                    self.output_folder,
                    metric
                )
            
            # Create decision impact plots
            self.reporting_manager.visualize_decision_impacts(master_file, self.output_folder)
        
        # Analyze and save best combinations
        best_combinations = self.analyze_results(master_file)
        
        return master_file, best_combinations

    def analyze_results(self, results_file: Path) -> Dict[str, Dict]:
        """Analyze the results and identify the best performing combinations."""
        self.logger.info("Analyzing FUSE decision results")
        
        df = pd.read_csv(results_file)
        metrics = ['kge', 'kgep', 'nse', 'mae', 'rmse']
        decisions = list(self.decision_options.keys())

        best_combinations = {}
        for metric in metrics:
            if metric in ['mae', 'rmse']:  # Lower is better
                best_row = df.loc[df[metric].idxmin()]
            else:  # Higher is better
                best_row = df.loc[df[metric].idxmax()]
            
            best_combinations[metric] = {
                'score': best_row[metric],
                'combination': {decision: best_row[decision] for decision in decisions}
            }

        # Save results to file
        output_file = self.project_dir / 'optimization' / 'best_fuse_decision_combinations.txt'
        with open(output_file, 'w') as f:
            for metric, data in best_combinations.items():
                f.write(f"Best combination for {metric} (score: {data['score']:.3f}):\n")
                for decision, value in data['combination'].items():
                    f.write(f"  {decision}: {value}\n")
                f.write("\n")

        self.logger.info("FUSE decision analysis results saved")
        return best_combinations

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config_dict.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config_dict.get(f'{file_type}'))
        

