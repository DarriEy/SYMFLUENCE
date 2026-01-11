"""
Analysis management for SYMFLUENCE model evaluation workflows.

Coordinates benchmarking, sensitivity analysis, and decision analysis for
evaluating hydrological model performance and parameter importance.
"""

from pathlib import Path
import logging
import pandas as pd
from typing import Dict, Any, Optional, TYPE_CHECKING

from symfluence.evaluation.sensitivity_analysis import SensitivityAnalyzer # type: ignore
from symfluence.evaluation.benchmarking import Benchmarker, BenchmarkPreprocessor # type: ignore
from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.analysis_registry import AnalysisRegistry

from symfluence.core.mixins import ConfigurableMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

class AnalysisManager(ConfigurableMixin):
    """Orchestrates comprehensive post-calibration analysis of model performance and sensitivity.

    Central coordinator for evaluating hydrological model performance through benchmarking,
    sensitivity analysis, and decision analysis. Provides unified interface to investigate
    model behavior, parameter importance, and structural choices. Integrates with evaluation
    framework to generate publication-ready analysis reports and visualizations.

    This class implements the Facade Pattern to manage complex analysis workflows across
    multiple hydrological models (SUMMA, FUSE, GR, HYPE, etc.). Enables systematic
    investigation of model strengths/weaknesses and parameter contributions to output
    uncertainty.

    Key Responsibilities:

        1. Benchmarking (run_benchmarking):
           Compare calibrated model against simple reference models:
               - Mean flow: Annual average discharge
               - Seasonality model: Climatological monthly patterns
               - Persistence model: Lagged observations (AR1)
           Purpose: Quantify value added by sophisticated model vs simplicity

        2. Sensitivity Analysis (run_sensitivity_analysis):
           Evaluate parameter importance:
               - Morris screening: Identify influential vs non-influential parameters
               - Sobol indices: Variance-based sensitivity (1st order, total)
               - FAST: Fourier amplitude sensitivity test
           Purpose: Prioritize parameters for observation/data requirements

        3. Decision Analysis (run_decision_analysis):
           Assess impact of model structure choices:
               - Alternative process representations
               - Alternative parameter sets (e.g., from different sources)
               - Alternative calibration targets (KGE vs NSE vs RMSE)
           Purpose: Evaluate trade-offs in model complexity vs parsimony

        4. Visualization:
           Generate analysis plots via ReportingManager:
               - Performance comparison plots
               - Sensitivity indices bar charts
               - Parameter importance rankings
               - Decision analysis trade-off plots

    Analysis Types:

        Benchmarking:
            Input:
                - Calibrated model results
                - Observed streamflow
                - Reference model outputs

            Process:
                1. Run reference models (mean, seasonality, persistence)
                2. Compute performance metrics (KGE, NSE, RMSE)
                3. Compare against calibrated model
                4. Calculate relative improvement

            Output:
                - Benchmark comparison table
                - Performance metrics for all models
                - Visualization showing performance rank

            Interpretation:
                KGE(model) > KGE(mean) ≈ Model outperforms naive reference
                KGE(model) < KGE(mean) ≈ Model worse than simple average (concerning!)
                KGE(model) >> KGE(seasonality) ≈ Model captures dynamics beyond seasonal pattern

        Sensitivity Analysis:
            Input:
                - Parameter ranges (bounds)
                - Model configuration
                - Model outputs and observations

            Sampling Methods:
                - Morris One-At-a-Time: Fast screening (~100s samples)
                - Sobol Quasi-Random: Variance-based (~1000s samples)
                - FAST: Spectral approach (~500s samples)

            Output:
                - Sensitivity indices (μ, σ, μ*, S1, ST, etc.)
                - Parameter ranking by importance
                - Grouped influence vs non-influential parameters

            Interpretation:
                μ* (modified Morris): Average |∂Y/∂Xi| - Main effect magnitude
                S1 (Sobol 1st order): Fraction of output variance from Xi alone
                ST (Sobol total): Fraction of output variance involving Xi
                Non-influential: Remove from calibration (reduce dimensionality)

        Decision Analysis:
            Input:
                - Multiple model configurations
                - Results from each configuration
                - Observations for validation

            Comparison Axes:
                - Model structure (e.g., SUMMA vs FUSE)
                - Process representation (e.g., 2-layer vs 3-layer soil)
                - Calibration target (KGE vs NSE vs RMSE)
                - Spatial discretization (lumped vs distributed)

            Output:
                - Performance metrics for each configuration
                - Trade-off analysis plots
                - Pareto frontier of non-dominated solutions

            Use Case:
                Decide between 3-layer soil (more parameters, better fit) vs 2-layer
                (simpler, more generalizable) by comparing performance

    Configuration Parameters:

        analysis.benchmarking.enabled: bool (default False)
            Enable benchmarking analysis

        analysis.benchmarking.reference_models: list
            Which reference models to include: ['mean', 'seasonality', 'persistence']

        analysis.sensitivity.method: str
            Sensitivity method: 'morris', 'sobol', 'fast', 'delsa'

        analysis.sensitivity.num_samples: int
            Number of samples for sensitivity analysis

        analysis.sensitivity.parameters: list
            Which parameters to analyze (subset of all model parameters)

        analysis.decision_analysis.configurations: list
            List of model configurations to compare

        reporting.analysis_enabled: bool
            Generate visualization plots of analysis results

    Output Structure:

        analysis_results/
            benchmarking/
                benchmark_comparison.csv  # Performance metrics table
                benchmark_plots/          # PNG plots
            sensitivity_analysis/
                sensitivity_indices.csv   # Sobol indices, Morris screening
                parameter_ranking.csv     # Ranked by importance
                sensitivity_plots/        # Bar charts, rankings
            decision_analysis/
                configuration_comparison.csv  # Metrics for each configuration
                tradeoff_analysis/        # Trade-off plots, Pareto frontier

    Example Usage:

        >>> config = SymfluenceConfig.from_file('config.yaml')
        >>> logger = setup_logger()
        >>> analysis_mgr = AnalysisManager(config, logger)
        >>>
        >>> # Run benchmarking
        >>> benchmark_results = analysis_mgr.run_benchmarking()
        >>> # Output: Performance comparison with mean, seasonality, persistence models
        >>>
        >>> # Run sensitivity analysis
        >>> sensitivity_results = analysis_mgr.run_sensitivity_analysis()
        >>> # Output: Parameter importance rankings
        >>>
        >>> # Run decision analysis
        >>> decision_results = analysis_mgr.run_decision_analysis()
        >>> # Output: Trade-off analysis comparing configurations

    Key Methods:

        run_benchmarking() → Path:
            Execute benchmarking against reference models
            Returns path to benchmark results directory

        run_sensitivity_analysis() → Path:
            Execute parameter sensitivity analysis
            Returns path to sensitivity results

        run_decision_analysis() → Path:
            Execute model structure decision analysis
            Returns path to decision analysis results

        run_all_analyses() → Dict[str, Path]:
            Execute all configured analyses
            Returns dict of {analysis_type: results_path}

    Performance:

        Benchmarking: Minutes to hours (depends on model runtime)
        Sensitivity Analysis: Hours to days (1000+ model evaluations needed)
        Decision Analysis: Days+ (multiple full model runs)

    Integration Points:

        - EvaluationRegistry: Access evaluators for metrics
        - OptimizationManager: Access calibrated parameter sets
        - ModelManager: Run reference models, configurations
        - ReportingManager: Generate analysis visualizations
        - DataManager: Access observations for validation

    See Also:

        - Benchmarker: Low-level benchmarking implementation
        - SensitivityAnalyzer: Low-level sensitivity analysis
        - EvaluationRegistry: Registry of evaluation methods
        - ReportingManager: Visualization of analysis results

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
        model-specific sensitivity analyses where supported. Uses AnalysisRegistry
        for model-specific analyzers when available, falling back to generic
        SensitivityAnalyzer for models without custom implementations.

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
                model = model.strip().upper()

                # Check registry for model-specific sensitivity analyzer
                analyzer_cls = AnalysisRegistry.get_sensitivity_analyzer(model)

                if analyzer_cls:
                    # Use registered model-specific analyzer
                    self.logger.info(f"Using registered sensitivity analyzer for {model}")
                    analyzer = analyzer_cls(self.config, self.logger, self.reporting_manager)
                    results_file = self.project_dir / "optimization" / f"{self.experiment_id}_parallel_iteration_results.csv"
                    if results_file.exists():
                        sensitivity_results[model] = analyzer.run_sensitivity_analysis(results_file)
                    else:
                        self.logger.warning(f"Calibration results file not found for {model}: {results_file}")
                else:
                    # Fall back to generic sensitivity analyzer (works for SUMMA and similar)
                    self.logger.info(f"Using generic sensitivity analyzer for {model}")
                    sensitivity_results[model] = self._run_generic_sensitivity_analysis(model)

            return sensitivity_results if sensitivity_results else None

        except Exception as e:
            self.logger.error(f"Error during sensitivity analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _run_generic_sensitivity_analysis(self, model: str) -> Optional[Dict]:
        """
        Run generic sensitivity analysis for a model using the default SensitivityAnalyzer.

        Args:
            model: Model name (e.g., 'SUMMA', 'FUSE')

        Returns:
            Sensitivity analysis results or None if failed
        """
        self.logger.info(f"Running generic sensitivity analysis for {model}")

        sensitivity_analyzer = SensitivityAnalyzer(self.config, self.logger, self.reporting_manager)
        results_file = self.project_dir / "optimization" / f"{self.experiment_id}_parallel_iteration_results.csv"

        if not results_file.exists():
            self.logger.warning(f"Calibration results file not found: {results_file}")
            return None

        return sensitivity_analyzer.run_sensitivity_analysis(results_file)
    
    def run_decision_analysis(self) -> Optional[Dict]:
        """
        Run decision analysis to assess the impact of model structure choices.

        Decision analysis evaluates different model structure configurations
        (e.g., process representations, parameterizations) to understand their
        impact on model performance.

        Uses AnalysisRegistry to discover model-specific decision analyzers.
        Each model can register its own analyzer that implements the
        `run_full_analysis()` interface returning (results_file, best_combinations).

        Returns:
            Optional[Dict]: Dictionary mapping model names to decision analysis results,
                          or None if the analysis was disabled or failed
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

        # Ensure model modules are imported to trigger analyzer registration
        self._import_model_analyzers()

        decision_results = {}

        try:
            models_str = self._get_config_value(
                lambda: self.config.model.hydrological_model,
                ''
            )
            hydrological_models = str(models_str).split(',')

            for model in hydrological_models:
                model = model.strip().upper()

                # Check registry for model-specific decision analyzer
                analyzer_cls = AnalysisRegistry.get_decision_analyzer(model)

                if analyzer_cls:
                    self.logger.info(f"Running {model} structure ensemble analysis")
                    analyzer = analyzer_cls(self.config, self.logger, self.reporting_manager)
                    results_file, best_combinations = analyzer.run_full_analysis()

                    self.logger.info(f"{model} structure ensemble analysis completed")
                    self.logger.info(f"Results saved to: {results_file}")

                    if best_combinations:
                        self.logger.info("Best combinations for each metric:")
                        for metric, data in best_combinations.items():
                            score = data.get('score', 0)
                            self.logger.info(f"  {metric}: score = {score:.3f}")

                    decision_results[model] = {
                        'results_file': results_file,
                        'best_combinations': best_combinations
                    }
                else:
                    available = AnalysisRegistry.list_decision_analyzers()
                    self.logger.info(
                        f"No decision analyzer registered for model: {model}. "
                        f"Available analyzers: {available}"
                    )

            return decision_results if decision_results else None

        except Exception as e:
            self.logger.error(f"Error during decision analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _import_model_analyzers(self) -> None:
        """
        Import model modules to trigger analyzer registration with AnalysisRegistry.

        This ensures that model-specific analyzers are registered before we try
        to look them up. The registration happens at import time via decorators.
        """
        # Import model modules that have registered analyzers
        # This is a controlled set - only models known to have decision analyzers
        try:
            import symfluence.models.summa  # noqa: F401 - triggers SUMMA analyzer registration
        except ImportError:
            pass

        try:
            import symfluence.models.fuse  # noqa: F401 - triggers FUSE analyzer registration
        except ImportError:
            pass

        # Future models can be added here as they implement decision analyzers
    
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
