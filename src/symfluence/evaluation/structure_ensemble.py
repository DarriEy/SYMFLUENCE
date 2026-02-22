#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Structure Ensemble Analyzer

This module provides the abstract base class for performing Structure Ensemble Analysis.
It coordinates running a model with different structural configurations (decisions)
and evaluating their performance.
"""

import itertools
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

from symfluence.core.mixins.project import ProjectContextMixin


class BaseStructureEnsembleAnalyzer(ProjectContextMixin, ABC):
    """
    Base class for performing Structure Ensemble Analysis.

    This class manages the workflow of:
    1. Generating combinations of model structure decisions.
    2. Iteratively updating model configuration.
    3. Running the model ensemble.
    4. Calculating performance metrics for each combination.
    5. Analyzing results to identify the best model structure.
    """

    def __init__(self, config: Any, logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the analyzer.

        Args:
            config: Configuration dictionary or SymfluenceConfig instance.
            logger: Logger instance.
            reporting_manager: Optional ReportingManager instance for visualization.
        """
        self.config = config
        self.logger = logger
        self.reporting_manager = reporting_manager

        # Support both typed config and dict config
        if hasattr(config, 'to_dict'):
            self.config_dict = config.to_dict(flatten=True)
        else:
            self.config_dict = config

        self.data_dir = Path(self.config_dict.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config_dict.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config_dict.get('EXPERIMENT_ID')

        # Model-specific settings to be initialized by child classes
        self.decision_options = self._initialize_decision_options()
        self.output_folder = self._initialize_output_folder()
        self.master_file = self._initialize_master_file()

    @abstractmethod
    def _initialize_decision_options(self) -> Dict[str, List[str]]:
        """Initialize model-specific decision options."""
        pass

    @abstractmethod
    def _initialize_output_folder(self) -> Path:
        """Initialize the output folder for analysis results."""
        pass

    @abstractmethod
    def _initialize_master_file(self) -> Path:
        """Initialize the master results file path."""
        pass

    def generate_combinations(self) -> List[Tuple[str, ...]]:
        """Generate all possible combinations of model decisions."""
        return list(itertools.product(*self.decision_options.values()))

    @abstractmethod
    def update_model_decisions(self, combination: Tuple[str, ...]):
        """Update the model's decision configuration file."""
        pass

    @abstractmethod
    def run_model(self):
        """Execute the model run for the current decision set."""
        pass

    @abstractmethod
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the current run."""
        pass

    def run_analysis(self) -> Path:
        """
        Main loop for running the ensemble analysis.

        Returns:
            Path: Path to the master results file.
        """
        self.logger.info("Starting structure ensemble analysis")

        combinations = self.generate_combinations()
        self.logger.info(f"Generated {len(combinations)} decision combinations")

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.master_file.parent.mkdir(parents=True, exist_ok=True)

        # Write header to master file
        with open(self.master_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration'] + list(self.decision_options.keys()) +
                          ['kge', 'kgep', 'nse', 'mae', 'rmse'])

        for i, combination in enumerate(combinations, 1):
            self.logger.info(f"Running combination {i} of {len(combinations)}")
            self.update_model_decisions(combination)

            try:
                self.run_model()
                metrics_res = self.calculate_performance_metrics()

                # metrics_res should be a dict containing 'kge', 'kgep', 'nse', 'mae', 'rmse'
                row = [i] + list(combination) + [
                    metrics_res.get('kge', np.nan),
                    metrics_res.get('kgep', np.nan),
                    metrics_res.get('nse', np.nan),
                    metrics_res.get('mae', np.nan),
                    metrics_res.get('rmse', np.nan)
                ]

                with open(self.master_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                self.logger.info(
                    f"Combination {i} completed: "
                    f"KGE={metrics_res.get('kge', 0):.3f}, "
                    f"NSE={metrics_res.get('nse', 0):.3f}"
                )

            except Exception as e:
                self.logger.error(f"Error in combination {i}: {str(e)}")
                with open(self.master_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + list(combination) + ['erroneous combination'])

        self.logger.info("Structure ensemble analysis completed")
        return self.master_file

    def analyze_results(self, results_file: Path) -> Dict[str, Any]:
        """
        Analyze the results and identify the best performing combinations.

        Args:
            results_file: Path to the results CSV file.

        Returns:
            Dict: Dictionary of best combinations for each metric.
        """
        self.logger.info("Analyzing ensemble results")

        if not results_file.exists():
            self.logger.error(f"Results file not found: {results_file}")
            return {}

        df = pd.read_csv(results_file)

        # Filter out erroneous rows
        for col in ['kge', 'kgep', 'nse', 'mae', 'rmse']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['kge', 'nse'], how='all')

        if df.empty:
            self.logger.warning("No valid results found to analyze.")
            return {}

        metrics_cols = ['kge', 'kgep', 'nse', 'mae', 'rmse']
        decisions = list(self.decision_options.keys())

        best_combinations = {}
        for metric in metrics_cols:
            if metric not in df.columns:
                continue

            try:
                if metric in ['mae', 'rmse']:
                    best_row_idx = df[metric].idxmin()
                else:
                    best_row_idx = df[metric].idxmax()

                best_row = df.loc[best_row_idx]
                best_combinations[metric] = {
                    'score': float(best_row[metric]),
                    'combination': {decision: best_row[decision] for decision in decisions}
                }
            except Exception as e:
                self.logger.warning(f"Could not find best combination for {metric}: {str(e)}")

        # Save results summary
        summary_file = self.master_file.parent / f"best_{self.__class__.__name__.lower()}_combinations.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            for metric, data in best_combinations.items():
                f.write(f"Best combination for {metric} (score: {data['score']:.3f}):\n")
                for decision, value in data['combination'].items():
                    f.write(f"  {decision}: {value}\n")
                f.write("\n")

        self.logger.info(f"Best combinations summary saved to {summary_file}")
        return best_combinations

    def run_full_analysis(self) -> Tuple[Path, Dict[str, Any]]:
        """
        Run the complete analysis workflow.

        Returns:
            Tuple[Path, Dict]: Results file path and best combinations dictionary.
        """
        results_file = self.run_analysis()

        # Model-specific visualization
        self.visualize_results(results_file)

        best_combinations = self.analyze_results(results_file)
        return results_file, best_combinations

    def visualize_results(self, results_file: Path):
        """
        Perform visualization of analysis results.
        To be optionally overridden by child classes.
        """
        if self.reporting_manager:
            try:
                self.reporting_manager.visualize_decision_impacts(results_file, self.output_folder)
            except Exception as e:
                self.logger.warning(f"Visualization failed: {str(e)}")
