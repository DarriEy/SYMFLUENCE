#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SUMMA Structure Analyzer

This module implements the Structure Ensemble Analysis for the SUMMA model,
often coupled with mizuRoute for routing.
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from symfluence.utils.evaluation.structure_ensemble import BaseStructureEnsembleAnalyzer
from symfluence.utils.evaluation.metrics import kge, kge_prime, nse, mae, rmse
from symfluence.utils.models.summa.runner import SummaRunner
from symfluence.utils.models.mizuroute.runner import MizuRouteRunner

class SummaStructureAnalyzer(BaseStructureEnsembleAnalyzer):
    """
    Structure Ensemble Analyzer for SUMMA and mizuRoute.
    
    Coordinates multiple runs of SUMMA with different model decisions,
    followed by mizuRoute routing, and evaluates the performance of each
    structural configuration.
    """
    
    def __init__(self, config: Any, logger: Any, reporting_manager: Optional[Any] = None):
        """
        Initialize the SUMMA structure analyzer.
        """
        super().__init__(config, logger, reporting_manager)
        
        # Initialize runners
        self.summa_runner = SummaRunner(config, logger)
        self.mizuroute_runner = MizuRouteRunner(config, logger)
        
        self.model_decisions_path = self.project_dir / "settings" / "SUMMA" / "modelDecisions.txt"

    def _initialize_decision_options(self) -> Dict[str, List[str]]:
        """Initialize SUMMA decision options from configuration."""
        return self.config_dict.get('SUMMA_DECISION_OPTIONS', {})

    def _initialize_output_folder(self) -> Path:
        """Initialize the output folder for SUMMA analysis results."""
        return self.project_dir / "reporting" / "decision_analysis"

    def _initialize_master_file(self) -> Path:
        """Initialize the master results file path for SUMMA."""
        return self.project_dir / 'optimization' / f"{self.experiment_id}_model_decisions_comparison.csv"

    def update_model_decisions(self, combination: Tuple[str, ...]):
        """
        Update the SUMMA modelDecisions.txt file with a new combination.
        
        Args:
            combination (Tuple[str, ...]): Tuple of decision values to use.
        """
        if not self.model_decisions_path.exists():
            self.logger.error(f"SUMMA model decisions file not found: {self.model_decisions_path}")
            raise FileNotFoundError(f"Could not find {self.model_decisions_path}")

        decision_keys = list(self.decision_options.keys())
        with open(self.model_decisions_path, 'r') as f:
            lines = f.readlines()
        
        option_map = dict(zip(decision_keys, combination))
        
        for i, line in enumerate(lines):
            for option, value in option_map.items():
                if line.strip().startswith(option):
                    # Maintain format: DecisionName  Value  ! Comment
                    lines[i] = f"{option.ljust(30)} {value.ljust(15)} ! {line.split('!')[-1].strip()}\n"
        
        with open(self.model_decisions_path, 'w') as f:
            f.writelines(lines)

    def run_model(self):
        """Execute SUMMA followed by mizuRoute."""
        self.logger.info("Executing SUMMA model run")
        self.summa_runner.run_summa()
        
        self.logger.info("Executing mizuRoute routing")
        self.mizuroute_runner.run_mizuroute()

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics comparing simulated routed runoff to observations.
        
        Returns:
            Dict: Dictionary containing KGE, NSE, MAE, and RMSE.
        """
        # Load observations
        obs_file_path = self.config_dict.get('OBSERVATIONS_PATH')
        if obs_file_path == 'default' or not obs_file_path:
            obs_file_path = self.project_dir / 'observations'/ 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
        else:
            obs_file_path = Path(obs_file_path)

        if not obs_file_path.exists():
            self.logger.error(f"Observation file not found: {obs_file_path}")
            raise FileNotFoundError(f"Missing observation file: {obs_file_path}")

        # Load simulation results (mizuRoute output)
        sim_reach_ID = self.config_dict.get('SIM_REACH_ID')
        sim_path_config = self.config_dict.get('SIMULATIONS_PATH')
        
        if sim_path_config == 'default' or not sim_path_config:
            # Construct default mizuRoute output path
            start_year = self.config_dict.get('EXPERIMENT_TIME_START', '1990').split('-')[0]
            sim_file_path = (
                self.project_dir / 'simulations' / self.experiment_id /
                'mizuRoute' / f"{self.experiment_id}.h.{start_year}-01-01-03600.nc"
            )
        else:
            sim_file_path = Path(sim_path_config)

        if not sim_file_path.exists():
            self.logger.error(f"Simulation output file not found: {sim_file_path}")
            raise FileNotFoundError(f"Missing simulation output: {sim_file_path}")

        # Process observations
        dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
        # Use discharge_cms and resample to hourly
        if 'discharge_cms' in dfObs.columns:
            obs_series = dfObs['discharge_cms'].resample('h').mean()
        else:
            # Fallback to first non-date column
            data_col = [c for c in dfObs.columns if c.lower() not in ['datetime', 'date']][0]
            obs_series = dfObs[data_col].resample('h').mean()

        # Process simulations
        with xr.open_dataset(sim_file_path, engine='netcdf4') as ds:
            # Filter by reach ID
            if 'reachID' in ds.variables:
                segment_index = ds['reachID'].values == int(sim_reach_ID)
                ds_sel = ds.sel(seg=segment_index)
            else:
                ds_sel = ds.isel(seg=0)
            
            # Extract routed runoff
            var_name = 'IRFroutedRunoff' if 'IRFroutedRunoff' in ds_sel.variables else 'KWTroutedRunoff'
            if var_name not in ds_sel.variables:
                # Fallback to any routed runoff variable
                var_name = [v for v in ds_sel.variables if 'routedRunoff' in v][0]
                
            sim_df = ds_sel[var_name].to_dataframe().reset_index()
            sim_df.set_index('time', inplace=True)
            sim_df.index = sim_df.index.round(freq='h')
            sim_series = sim_df[var_name]

        # Align series
        obs_aligned = obs_series.reindex(sim_series.index).dropna()
        sim_aligned = sim_series.reindex(obs_aligned.index).dropna()
        
        obs_vals = obs_aligned.values
        sim_vals = sim_aligned.values
        
        if len(obs_vals) == 0:
            self.logger.warning("No overlapping data between observations and simulations")
            return {'kge': np.nan, 'kgep': np.nan, 'nse': np.nan, 'mae': np.nan, 'rmse': np.nan}

        return {
            'kge': float(kge(obs_vals, sim_vals, transfo=1)),
            'kgep': float(kge_prime(obs_vals, sim_vals, transfo=1)),
            'nse': float(nse(obs_vals, sim_vals, transfo=1)),
            'mae': float(mae(obs_vals, sim_vals, transfo=1)),
            'rmse': float(rmse(obs_vals, sim_vals, transfo=1))
        }
