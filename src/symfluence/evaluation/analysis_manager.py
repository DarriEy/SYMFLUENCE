# In utils/analysis_utils/analysis_manager.py

from pathlib import Path
import logging
import pandas as pd
from typing import Dict, Any, Optional, TYPE_CHECKING

from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer # type: ignore
from symfluence.evaluation.sensitivity_analysis import SensitivityAnalyzer # type: ignore
from symfluence.evaluation.benchmarking import Benchmarker, BenchmarkPreprocessor # type: ignore
from symfluence.models.fuse.structure_analyzer import FuseStructureAnalyzer # type: ignore
from symfluence.evaluation.registry import EvaluationRegistry

from symfluence.core.mixins import ConfigurableMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

class AnalysisManager(ConfigurableMixin):
    """
    Manages all analysis operations including benchmarking, sensitivity, and decision analysis.
    
    The AnalysisManager is responsible for coordinating various analyses that evaluate 
    hydrological model performance, parameter sensitivity, and model structure decisions. 
    These analyses provide critical insights for understanding model behavior, improving 
    model configurations, and quantifying uncertainty.
    
    Key responsibilities:
    - Benchmarking: Comparing model performance against simple reference models
    - Sensitivity Analysis: Evaluating parameter importance and uncertainty
    - Decision Analysis: Assessing the impact of model structure choices
    - Result Visualization: Creating plots and visualizations of analysis results
    
    The AnalysisManager works with multiple hydrological models, providing consistent
    interfaces for analysis operations across different model types. It integrates
    with other SYMFLUENCE components to ensure that analyses use the correct input data
    and that results are properly stored and visualized.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, config: 'SymfluenceConfig', logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the Analysis Manager.

        Args:
            config: SymfluenceConfig instance
            logger: Logger instance
            reporting_manager: ReportingManager instance

        Raises:
            TypeError: If config is not a SymfluenceConfig instance
        """
        # Import here to avoid circular imports at module level
        from symfluence.core.config.models import SymfluenceConfig

        if not isinstance(config, SymfluenceConfig):
            raise TypeError(
                f"config must be SymfluenceConfig, got {type(config).__name__}. "
                "Use SymfluenceConfig.from_file() to load configuration."
            )

        # Set config via the ConfigMixin property
        self._config = config
        self.logger = logger
        self.reporting_manager = reporting_manager
        
    def run_benchmarking(self) -> Optional[Path]:
        """
        Run benchmarking analysis to evaluate model performance against reference models.
        
        Benchmarking compares the performance of sophisticated hydrological models
        against simple reference models (e.g., mean flow, seasonality model) to
        quantify the value added by the model's complexity. This process includes:
        
        1. Preprocessing observed data for the benchmark period
        2. Running simple benchmark models (e.g., mean, seasonality, persistence)
        3. Computing performance metrics for each benchmark
        4. Visualizing benchmark results for comparison
        
        Benchmarking provides a baseline for evaluating model performance and helps
        identify the minimum acceptable performance for a given watershed.
        
        Returns:
            Optional[Path]: Path to benchmark results file or None if benchmarking failed
            
        Raises:
            FileNotFoundError: If required observation data is missing
            ValueError: If date ranges are invalid
            Exception: For other errors during benchmarking
        """
        self.logger.info("Starting benchmarking analysis")
        
        try:
            # Use typed config if available
            # Use typed config for sub-components
            
            # Preprocess data for benchmarking
            preprocessor = BenchmarkPreprocessor(self.config, self.logger)
            
            # Extract calibration and evaluation periods
            calib_period = self._get_config_value(
                lambda: self.config.domain.calibration_period
            )
            eval_period = self._get_config_value(
                lambda: self.config.domain.evaluation_period
            )
            
            calib_start = str(calib_period).split(',')[0].strip()
            eval_end = str(eval_period).split(',')[1].strip()
            
            benchmark_data = preprocessor.preprocess_benchmark_data(calib_start, eval_end)
            
            # Run benchmarking
            benchmarker = Benchmarker(self.config, self.logger)
            benchmark_results = benchmarker.run_benchmarking()
            
            # Visualize benchmark results
            if self.reporting_manager:
                self.reporting_manager.visualize_benchmarks(benchmark_results)
            
            # Return path to benchmark results
            benchmark_file = self.project_dir / "evaluation" / "benchmark_scores.csv"
            if benchmark_file.exists():
                self.logger.info(f"Benchmarking completed successfully: {benchmark_file}")
                return benchmark_file
            else:
                self.logger.warning("Benchmarking completed but results file not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during benchmarking: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def run_sensitivity_analysis(self) -> Optional[Dict]:
        """
        Run sensitivity analysis to evaluate parameter importance and uncertainty.
        
        Sensitivity analysis quantifies how model parameters influence simulation
        results and performance metrics. This analysis helps:
        
        1. Identify which parameters have the most significant impact on model performance
        2. Quantify parameter uncertainty and its effect on predictions
        3. Guide model simplification by identifying insensitive parameters
        4. Inform calibration strategies by focusing on sensitive parameters
        
        The method iterates through configured hydrological models, running
        model-specific sensitivity analyses where supported. Currently,
        detailed sensitivity analysis is implemented for the SUMMA model.
        
        Returns:
            Optional[Dict]: Dictionary mapping model names to sensitivity results,
                          or None if the analysis was disabled or failed
                          
        Raises:
            FileNotFoundError: If required optimization results are missing
            Exception: For other errors during sensitivity analysis
        """
        self.logger.info("Starting sensitivity analysis")
        
        # Check if sensitivity analysis is enabled
        run_sensitivity = self._get_config_value(
            lambda: self.config.analysis.run_sensitivity_analysis,
            True
        )
        if not run_sensitivity:
            self.logger.info("Sensitivity analysis is disabled in configuration")
            return None
        
        sensitivity_results = {}
        
        try:
            models_str = self._get_config_value(
                lambda: self.config.model.hydrological_model,
                ''
            )
            hydrological_models = str(models_str).split(',')
            
            for model in hydrological_models:
                model = model.strip()
                
                if model == 'SUMMA':
                    sensitivity_results[model] = self._run_summa_sensitivity_analysis()
                else:
                    self.logger.info(f"Sensitivity analysis not implemented for model: {model}")
                    
            return sensitivity_results if sensitivity_results else None
            
        except Exception as e:
            self.logger.error(f"Error during sensitivity analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _run_summa_sensitivity_analysis(self) -> Optional[Dict]:
        """
        Run sensitivity analysis for SUMMA model.
        """
        self.logger.info("Running SUMMA sensitivity analysis")
        
        # Use typed config for sub-components
        sensitivity_analyzer = SensitivityAnalyzer(self.config, self.logger, self.reporting_manager)
        results_file = self.project_dir / "optimization" / f"{self.experiment_id}_parallel_iteration_results.csv"
        
        if not results_file.exists():
            self.logger.error(f"Calibration results file not found: {results_file}")
            return None
        
        return sensitivity_analyzer.run_sensitivity_analysis(results_file)
    
    def run_decision_analysis(self) -> Optional[Dict]:
        """
        Run decision analysis to assess the impact of model structure choices.
        """
        self.logger.info("Starting decision analysis")
        
        # Check if decision analysis is enabled
        run_decision = self._get_config_value(
            lambda: self.config.analysis.run_decision_analysis,
            True
        )
        if not run_decision:
            self.logger.info("Decision analysis is disabled in configuration")
            return None
        
        decision_results = {}
        
        try:
            models_str = self._get_config_value(
                lambda: self.config.model.hydrological_model,
                ''
            )
            hydrological_models = str(models_str).split(',')
            
            for model in hydrological_models:
                model = model.strip()
                
                if model == 'SUMMA':
                    decision_results[model] = self._run_summa_decision_analysis()
                elif model == 'FUSE':
                    decision_results[model] = self._run_fuse_decision_analysis()
                else:
                    self.logger.info(f"Decision analysis not implemented for model: {model}")
                    
            return decision_results if decision_results else None
            
        except Exception as e:
            self.logger.error(f"Error during decision analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _run_summa_decision_analysis(self) -> Dict:
        """
        Run decision analysis for SUMMA model.
        """
        self.logger.info("Running SUMMA structure ensemble analysis")
        
        # Use typed config for sub-components
        analyzer = SummaStructureAnalyzer(self.config, self.logger, self.reporting_manager)
        results_file, best_combinations = analyzer.run_full_analysis()
        
        self.logger.info("SUMMA structure ensemble analysis completed")
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info("Best combinations for each metric:")
        for metric, data in best_combinations.items():
            self.logger.info(f"  {metric}: score = {data['score']:.3f}")
        
        return {
            'results_file': results_file,
            'best_combinations': best_combinations
        }
    
    def _run_fuse_decision_analysis(self) -> Dict:
        """
        Run decision analysis for FUSE model.
        """
        self.logger.info("Running FUSE structure ensemble analysis")
        
        # Use typed config for sub-components
        analyzer = FuseStructureAnalyzer(self.config, self.logger, self.reporting_manager)
        results_file, best_combinations = analyzer.run_full_analysis()
        
        self.logger.info("FUSE structure ensemble analysis completed")
        return {
            'results_file': results_file,
            'best_combinations': best_combinations
        }
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """
        Get status of various analyses.
        
        This method provides a comprehensive status report on the analysis operations.
        It checks for the existence of key files and directories to determine which
        analyses have been completed successfully and which are available to run.
        
        The status information includes:
        - Whether benchmarking has been completed
        - Whether sensitivity analysis is available and its results exist
        - Whether decision analysis is available and its results exist
        - Whether optimization results (required for some analyses) exist
        
        This information is useful for tracking progress, diagnosing issues,
        and providing feedback to users.
        
        Returns:
            Dict[str, Any]: Dictionary containing analysis status information,
                          including flags for completed analyses and available results
        """
        status = {
            'benchmarking_complete': (self.project_dir / "evaluation" / "benchmark_scores.csv").exists(),
            'sensitivity_analysis_available': self._get_config_value(
                lambda: self.config.analysis.run_sensitivity_analysis,
                True
            ),
            'decision_analysis_available': self._get_config_value(
                lambda: self.config.analysis.run_decision_analysis,
                True
            ),
            'optimization_results_exist': (self.project_dir / "optimization" / f"{self.experiment_id}_parallel_iteration_results.csv").exists(),
        }
        
        # Check for analysis outputs
        if (self.project_dir / "reporting" / "sensitivity_analysis").exists():
            status['sensitivity_plots_exist'] = True
        
        if (self.project_dir / "optimization").exists():
            status['decision_analysis_results_exist'] = any(
                file.name.endswith('_model_decisions_comparison.csv')
                for file in (self.project_dir / "optimization").glob('*.csv')
            )
        
        return status
    
    def run_multivariate_evaluation(self, sim_results: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """
        Run multivariate evaluation against all available observations.
        """
        self.logger.info("Starting multivariate evaluation")
        results = {}
        
        # 1. Load observations
        # Note: ModelEvaluator can load observations from file if not provided,
        # but here we might want to load them once if shared.
        
        # 2. Evaluate each variable
        for var_type, sim_series in sim_results.items():
            # For multivariate, the var_type might be SNOW, but we need to know if it's SWE or SCA
            # We can use the mapping from config if provided
            target = var_type
            evaluator = EvaluationRegistry.get_evaluator(
                var_type, self.config, self.logger, self.project_dir, target=target
            )
            if evaluator:
                self.logger.info(f"Evaluating {var_type}")
                # calculate_metrics now handles aligning and filtering
                results[var_type] = evaluator.calculate_metrics(sim_series, calibration_only=False)
                
        return results

    def _load_all_observations(self) -> Dict[str, pd.Series]:
        """Load all preprocessed observations."""
        obs = {}
        
        # Streamflow
        sf_file = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        if sf_file.exists():
            df = pd.read_csv(sf_file, parse_dates=True, index_col=0)
            obs['STREAMFLOW'] = df.iloc[:, 0]
            
        # GRACE/TWS
        tws_file = self.project_dir / "observations" / "grace" / "preprocessed" / f"{self.domain_name}_grace_tws_processed.csv"
        if tws_file.exists():
            df = pd.read_csv(tws_file, parse_dates=True, index_col=0)
            if 'grace_jpl_anomaly' in df.columns:
                obs['TWS'] = df['grace_jpl_anomaly']
                
        # Add others...
        return obs

    def validate_analysis_requirements(self) -> Dict[str, bool]:
        """
        Validate that requirements are met for running analyses.
        
        This method checks whether the necessary files and data are available
        to run each type of analysis. It verifies:
        
        1. For benchmarking: Existence of processed observation data
        2. For sensitivity analysis: Existence of optimization results
        3. For decision analysis: Existence of model simulation outputs
        
        These validations help prevent runtime errors by ensuring that analyses
        only run when their prerequisites are met.
        
        Returns:
            Dict[str, bool]: Dictionary indicating which analyses can be run:
                          - benchmarking: Whether benchmarking can be run
                          - sensitivity_analysis: Whether sensitivity analysis can be run
                          - decision_analysis: Whether decision analysis can be run
        """
        requirements = {
            'benchmarking': True,  # Benchmarking has minimal requirements
            'sensitivity_analysis': False,
            'decision_analysis': False
        }
        
        # Check for optimization results (required for sensitivity analysis)
        optimization_results = self.project_dir / "optimization" / f"{self.experiment_id}_parallel_iteration_results.csv"
        if optimization_results.exists():
            requirements['sensitivity_analysis'] = True
        
        # Check for model outputs (required for decision analysis)
        simulation_dir = self.project_dir / "simulations" / self.experiment_id
        if simulation_dir.exists():
            requirements['decision_analysis'] = True
        
        # Check for processed observations (required for all analyses)
        obs_file = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        if not obs_file.exists():
            self.logger.warning("Processed observations not found - all analyses may fail")
            requirements = {key: False for key in requirements}
        
        return requirements
