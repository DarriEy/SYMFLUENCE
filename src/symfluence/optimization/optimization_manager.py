# In utils/optimization/optimization_manager.py

from pathlib import Path
import logging
import warnings
from typing import Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime
import json

import numpy as np
from symfluence.optimization.objectives import ObjectiveRegistry
from symfluence.optimization.core import TransformationManager
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.optimization_results_manager import OptimizationResultsManager

from symfluence.core.mixins import ConfigurableMixin

# Import for type checking only
try:
    from symfluence.core.config.models import SymfluenceConfig
except ImportError:
    SymfluenceConfig = None

class OptimizationManager(ConfigurableMixin):
    """
    Coordinates model optimization and calibration.
    
    The OptimizationManager is responsible for coordinating model calibration 
    within the SYMFLUENCE framework. It provides a unified interface for 
    different optimization algorithms and handles the interaction between 
    optimization components and hydrological models.
    
    Key responsibilities:
    - Coordinating model calibration using different optimization algorithms
    - Managing optimization results and performance metrics
    - Validating optimization configurations
    - Providing status information on optimization progress
    
    The OptimizationManager supports multiple optimization algorithms via
    OptimizerRegistry:
    - PSO: Particle Swarm Optimization
    - SCE-UA: Shuffled Complex Evolution
    - DDS: Dynamically Dimensioned Search
    - DE: Differential Evolution
    - NSGA-II: Non-dominated Sorting Genetic Algorithm II
    - ADAM/LBFGS: Gradient-based optimization
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, config: Union[Dict[str, Any], 'SymfluenceConfig'], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the Optimization Manager.
        
        Sets up the OptimizationManager with the necessary components for model
        calibration. This includes initializing the results manager and mappings 
        between optimization algorithms and their implementation classes.
        
        Args:
            config: Configuration dictionary or SymfluenceConfig instance
            logger: Logger instance
            reporting_manager: ReportingManager instance
            
        Raises:
            KeyError: If essential configuration values are missing
            ImportError: If required optimizer modules cannot be imported
        """
        # Support both typed config and dict config
        if SymfluenceConfig and isinstance(config, SymfluenceConfig):
            self.typed_config = config
            self.config = config.to_dict(flatten=True)
        else:
            self.typed_config = None
            self.config = config

        self.logger = logger
        self.reporting_manager = reporting_manager
        
        # Use typed config if available for components
        component_config = self.typed_config if self.typed_config else self.config

        self.experiment_id = self.config.get('EXPERIMENT_ID')

        # Initialize results manager
        self.results_manager = OptimizationResultsManager(
            self.project_dir,
            self.experiment_id,
            self.logger,
            self.reporting_manager
        )
        
        # Initialize transformation manager
        self.transformation_manager = TransformationManager(component_config, self.logger)

    def run_optimization_workflow(self) -> Dict[str, Any]:
        """
        Main entry point for optimization workflow.
        
        This method checks the OPTIMIZATION_METHODS configuration and runs
        the iterative optimization (calibration) if enabled.
        
        Returns:
            Dict[str, Any]: Results from completed workflows
        """
        results = {}
        optimization_methods = self._resolve_config_value(
            lambda: self.typed_config.optimization.methods,
            'OPTIMIZATION_METHODS',
            []
        )
        
        self.logger.info(f"Running optimization workflows: {optimization_methods}")
        
        # Run iterative optimization (calibration)
        if 'iteration' in optimization_methods:
            calibration_results = self.calibrate_model()
            if calibration_results:
                results['calibration'] = str(calibration_results)
        
        # Check for deprecated methods and warn
        deprecated_methods = [
            'differentiable_parameter_emulation',
            'large_domain_emulator',
            'emulation'
        ]
        
        for method in deprecated_methods:
            if method in optimization_methods:
                self.logger.warning(
                    f"Optimization method '{method}' is deprecated and no longer supported. "
                    "Use gradient-based optimization (ADAM/LBFGS) via standard model optimizers instead."
                )
        
        return results
    
    def calibrate_model(self) -> Optional[Path]:
        """
        Calibrate the model using the specified optimization algorithm.
        
        This method coordinates the calibration process for the configured
        hydrological model using the optimization algorithm specified in the
        configuration.
        
        The calibration process involves:
        1. Checking if iterative optimization is enabled in the configuration
        2. Determining which optimization algorithm to use
        3. Executing the calibration for each configured hydrological model using the 
           unified registry-based infrastructure
        
        The optimization algorithm is specified through the ITERATIVE_OPTIMIZATION_ALGORITHM
        configuration parameter (default: 'PSO').
        
        Returns:
            Optional[Path]: Path to calibration results file or None if calibration
                        was disabled or failed
        """
        self.logger.info("Starting model calibration")

        # Check if iterative optimization is enabled
        optimization_methods = self._resolve_config_value(
            lambda: self.typed_config.optimization.methods,
            'OPTIMIZATION_METHODS',
            []
        )
        if not 'iteration' in optimization_methods:
            self.logger.info("Iterative optimization is disabled in configuration")
            return None

        # Get the optimization algorithm from config
        opt_algorithm = self._resolve_config_value(
            lambda: self.typed_config.optimization.algorithm,
            'ITERATIVE_OPTIMIZATION_ALGORITHM',
            'PSO'
        )

        try:
            models_str = self._resolve_config_value(
                lambda: self.typed_config.model.hydrological_model,
                'HYDROLOGICAL_MODEL',
                ''
            )
            hydrological_models = str(models_str).split(',')

            for model in hydrological_models:
                model = model.strip().upper()
                return self._calibrate_with_registry(model, opt_algorithm)

            return None
            
        except Exception as e:
            self.logger.error(f"Error during model calibration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _calibrate_with_registry(self, model_name: str, algorithm: str) -> Optional[Path]:
        """
        Calibrate a model using the OptimizerRegistry.

        This method uses the new unified optimizer infrastructure based on
        BaseModelOptimizer. It provides a cleaner, more maintainable approach
        to model calibration with consistent algorithm support across all models.

        Args:
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'NGEN')
            algorithm: Optimization algorithm to use

        Returns:
            Optional[Path]: Path to results file or None if calibration failed
        """
        # Import model optimizers to trigger registration
        from symfluence.optimization import model_optimizers  # noqa: F401

        # Get optimizer class from registry
        optimizer_cls = OptimizerRegistry.get_optimizer(model_name)

        if optimizer_cls is None:
            self.logger.error(f"No optimizer registered for model: {model_name}")
            self.logger.info(f"Registered models: {OptimizerRegistry.list_models()}")
            return None

        # Create optimization directory
        opt_dir = self.project_dir / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize optimizer
            self.logger.info(f"Using {algorithm} optimization for {model_name} (registry-based)")
            optimizer = optimizer_cls(self.config, self.logger, opt_dir, reporting_manager=self.reporting_manager)

            # Map algorithm name to method
            algorithm_methods = {
                'DDS': optimizer.run_dds,
                'PSO': optimizer.run_pso,
                'SCE-UA': optimizer.run_sce,
                'DE': optimizer.run_de,
                'NSGA-II': getattr(optimizer, 'run_nsga2', None),
                'ADAM': lambda: optimizer.run_adam(
                    steps=self._resolve_config_value(
                        lambda: self.typed_config.optimization.adam_steps,
                        'ADAM_STEPS', 
                        100
                    ),
                    lr=self._resolve_config_value(
                        lambda: self.typed_config.optimization.adam_learning_rate,
                        'ADAM_LEARNING_RATE',
                        0.01
                    )
                ),
                'LBFGS': lambda: optimizer.run_lbfgs(
                    steps=self._resolve_config_value(
                        lambda: self.typed_config.optimization.lbfgs_steps,
                        'LBFGS_STEPS',
                        50
                    ),
                    lr=self._resolve_config_value(
                        lambda: self.typed_config.optimization.lbfgs_learning_rate,
                        'LBFGS_LEARNING_RATE',
                        0.1
                    )
                ),
            }

            # Get algorithm method
            run_method = algorithm_methods.get(algorithm)

            if run_method is None:
                self.logger.error(f"Algorithm {algorithm} not supported for registry-based optimization")
                self.logger.info(f"Supported algorithms: {list(algorithm_methods.keys())}")
                return None

            # Run optimization
            results_file = run_method()

            if results_file and Path(results_file).exists():
                self.logger.info(f"{model_name} calibration completed: {results_file}")
                return results_file
            else:
                self.logger.warning(f"{model_name} calibration completed but results file not found")
                return None

        except Exception as e:
            self.logger.error(f"Error during {model_name} {algorithm} optimization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get status of optimization operations.
        
        Returns:
            Dict[str, Any]: Dictionary containing optimization status information
        """
        optimization_methods = self._resolve_config_value(
            lambda: self.typed_config.optimization.methods,
            'OPTIMIZATION_METHODS',
            []
        )
        status = {
            'iterative_optimization_enabled': 'iteration' in optimization_methods,
            'optimization_algorithm': self._resolve_config_value(
                lambda: self.typed_config.optimization.algorithm,
                'ITERATIVE_OPTIMIZATION_ALGORITHM',
                'PSO'
            ),
            'optimization_metric': self._resolve_config_value(
                lambda: self.typed_config.optimization.metric,
                'OPTIMIZATION_METRIC',
                'KGE'
            ),
            'optimization_dir': str(self.project_dir / "optimization"),
            'results_exist': False,
        }
        
        # Check for optimization results
        results_file = self.project_dir / "optimization" / f"{self.experiment_id}_parallel_iteration_results.csv"
        status['results_exist'] = results_file.exists()
        
        return status
    
    def validate_optimization_configuration(self) -> Dict[str, bool]:
        """
        Validate optimization configuration settings.
        
        Returns:
            Dict[str, bool]: Dictionary containing validation results
        """
        validation = {
            'algorithm_valid': False,
            'model_supported': False,
            'parameters_defined': False,
            'metric_valid': False
        }
        
        # Check algorithm
        algorithm = self._resolve_config_value(
            lambda: self.typed_config.optimization.algorithm,
            'ITERATIVE_OPTIMIZATION_ALGORITHM',
            ''
        )
        supported_algorithms = ['DDS', 'PSO', 'SCE-UA', 'DE', 'ADAM', 'LBFGS', 'NSGA-II']
        validation['algorithm_valid'] = algorithm in supported_algorithms
        
        # Check model support
        models_str = self._resolve_config_value(
            lambda: self.typed_config.model.hydrological_model,
            'HYDROLOGICAL_MODEL',
            ''
        )
        models = str(models_str).split(',')
        validation['model_supported'] = 'SUMMA' in [m.strip() for m in models]
        
        # Check parameters to calibrate
        local_params = self._resolve_config_value(
            lambda: self.typed_config.optimization.params_to_calibrate,
            'PARAMS_TO_CALIBRATE',
            ''
        )
        basin_params = self._resolve_config_value(
            lambda: self.typed_config.optimization.basin_params_to_calibrate,
            'BASIN_PARAMS_TO_CALIBRATE',
            ''
        )
        validation['parameters_defined'] = bool(local_params or basin_params)
        
        # Check metric
        valid_metrics = ['KGE', 'NSE', 'RMSE', 'MAE', 'KGEp']
        metric = self._resolve_config_value(
            lambda: self.typed_config.optimization.metric,
            'OPTIMIZATION_METRIC',
            ''
        )
        validation['metric_valid'] = metric in valid_metrics
        
        return validation
    
    def get_available_optimizers(self) -> Dict[str, str]:
        """
        Get list of available optimization algorithms.
        
        Returns:
            Dict[str, str]: Dictionary mapping algorithm identifiers to their descriptions
        """
        return {
            'PSO': 'Particle Swarm Optimization',
            'SCE-UA': 'Shuffled Complex Evolution',
            'DDS': 'Dynamically Dimensioned Search',
            'DE': 'Differential Evolution',
            'NSGA-II': 'Non-dominated Sorting Genetic Algorithm II',
            'ASYNC-DDS': 'Asynchronous Dynamically Dimensioned Search',
            'POP-DDS': 'Population Dynamically Dimensioned Search',
        }
    
    def _apply_parameter_transformations(self, params: Dict[str, float], settings_dir: Path) -> bool:
        """
        Applies transformations to parameters (e.g., soil depth multipliers).
        """
        return self.transformation_manager.transform(params, settings_dir)

    def _calculate_multivariate_objective(self, sim_results: Dict[str, pd.Series]) -> float:
        """
        Calculates a composite objective score from multivariate simulation results.
        """
        # 1. Get the multivariate objective handler
        objective_handler = ObjectiveRegistry.get_objective('MULTIVARIATE', self.config, self.logger)
        if not objective_handler:
            return 1000.0
            
        # 2. Use AnalysisManager to evaluate variables
        from symfluence.evaluation.analysis_manager import AnalysisManager
        am = AnalysisManager(self.config, self.logger)
        eval_results = am.run_multivariate_evaluation(sim_results)
        
        # 3. Calculate scalar objective
        return objective_handler.calculate(eval_results)

    def load_optimization_results(self, filename: str = None) -> Optional[Dict]:
        """
        Load optimization results from file.
        
        Args:
            filename (str, optional): Name of results file to load. If None, uses
                                    the default filename based on experiment_id.
            
        Returns:
            Optional[Dict]: Dictionary with optimization results. Returns None if loading fails.
        """
        try:
            results_df = self.results_manager.load_optimization_results(filename)
            
            if results_df is None:
                return None

            # Convert DataFrame to dictionary format
            results = {
                'parameters': results_df.to_dict(orient='records'),
                'best_iteration': results_df.iloc[0].to_dict(),
                'columns': results_df.columns.tolist()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading optimization results: {str(e)}")
            return None
