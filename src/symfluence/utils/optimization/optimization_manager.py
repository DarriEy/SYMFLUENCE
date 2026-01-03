# In utils/optimization/optimization_manager.py

from pathlib import Path
import logging
import warnings
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
import json

import numpy as np
# Import optimizers from the optimizers package
from symfluence.utils.optimization.optimizers import (
    DEOptimizer, DDSOptimizer, AsyncDDSOptimizer, PopulationDDSOptimizer,
    PSOOptimizer, NSGA2Optimizer, SCEUAOptimizer
)
from symfluence.utils.optimization.legacy.large_domain_emulator import LargeDomainEmulator
from symfluence.utils.optimization.legacy.differentiable_parameter_emulator import DifferentiableParameterOptimizer, EmulatorConfig
from symfluence.utils.optimization.objective_registry import ObjectiveRegistry
from symfluence.utils.optimization.transformers import TransformationManager
from symfluence.utils.optimization.registry import OptimizerRegistry
from symfluence.utils.optimization.optimization_results_manager import OptimizationResultsManager

class OptimizationManager:
    """
    Coordinates model optimization, calibration, and parameter emulation.
    
    The OptimizationManager is responsible for coordinating model calibration and 
    parameter emulation within the SYMFLUENCE framework. It provides a unified 
    interface for different optimization algorithms and handles the interaction 
    between optimization components and hydrological models.
    
    Key responsibilities:
    - Coordinating model calibration using different optimization algorithms
    - Executing parameter emulation workflows
    - Managing optimization results and performance metrics
    - Validating optimization configurations
    - Providing status information on optimization progress
    
    The OptimizationManager supports multiple optimization algorithms:
    - PSO: Particle Swarm Optimization
    - SCE-UA: Shuffled Complex Evolution
    - DDS: Dynamically Dimensioned Search
    
    It also supports parameter emulation for rapid exploration of parameter space
    and uncertainty quantification.
    
    The OptimizationManager acts as a bridge between the underlying optimization
    algorithms and the SYMFLUENCE workflow, ensuring consistent execution and
    results management.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
        data_dir (Path): Path to the SYMFLUENCE data directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the project directory
        experiment_id (str): ID of the current experiment
        results_manager (OptimizationResultsManager): Manager for optimization results
        optimizers (Dict[str, Any]): Mapping of algorithm names to optimizer classes
        optimizer_methods (Dict[str, str]): Mapping of algorithm names to method names
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the Optimization Manager.
        
        Sets up the OptimizationManager with the necessary components for model
        calibration and parameter emulation. This includes initializing the results
        manager, emulation runner, and mappings between optimization algorithms
        and their implementation classes.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all settings
            logger (logging.Logger): Logger instance for recording operations
            reporting_manager (ReportingManager): ReportingManager instance
            
        Raises:
            KeyError: If essential configuration values are missing
            ImportError: If required optimizer modules cannot be imported
        """
        self.config = config
        self.logger = logger
        self.reporting_manager = reporting_manager
        
        # Validate required fields
        required_fields = ['SYMFLUENCE_DATA_DIR', 'DOMAIN_NAME', 'EXPERIMENT_ID']
        for field in required_fields:
            if field not in self.config:
                raise KeyError(f"Missing required configuration field: {field}")

        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Initialize results manager
        self.results_manager = OptimizationResultsManager(
            self.project_dir,
            self.experiment_id,
            self.logger,
            self.reporting_manager
        )
        
        # Initialize transformation manager
        self.transformation_manager = TransformationManager(self.config, self.logger)
        
        # Define optimizer mapping
        self.optimizers = {
            'DDS': DDSOptimizer,
            'ASYNC-DDS': AsyncDDSOptimizer,
            'POP-DDS': PopulationDDSOptimizer,
            'DE' : DEOptimizer,
            'PSO': PSOOptimizer,
            'NSGA-II': NSGA2Optimizer,
            'SCE-UA': SCEUAOptimizer,
        }
        
        # Define optimizer run method names
        self.optimizer_methods = {
            'DDS': 'run_optimization',
            'ASYNC-DDS': 'run_optimization',
            'POP-DDS': 'run_optimization',
            'DE': 'run_optimization',
            'PSO': 'run_optimization',
            'NSGA-II': 'run_optimization',
            'SCE-UA': 'run_optimization' 

        }

    def run_optimization_workflow(self) -> Dict[str, Any]:
        """
        Main entry point for all optimization and emulation workflows.
        
        This method checks the OPTIMIZATION_METHODS configuration and runs
        the appropriate workflows in the correct order.
        
        Returns:
            Dict[str, Any]: Results from all completed workflows
        """
        results = {}
        optimization_methods = self.config.get('OPTIMIZATION_METHODS', [])
        
        self.logger.info(f"Running optimization workflows: {optimization_methods}")
        
        # Run iterative optimization (calibration)
        if 'iteration' in optimization_methods:
            calibration_results = self.calibrate_model()
            if calibration_results:
                results['calibration'] = str(calibration_results)
        
        # Run differentiable parameter emulation
        if 'differentiable_parameter_emulation' in optimization_methods:
            dpe_results = self.run_emulation()
            if dpe_results:
                results['differentiable_parameter_emulation'] = dpe_results
        
        # Run large domain emulation
        if 'large_domain_emulator' in optimization_methods:
            lde_results = self.run_large_domain_emulation()
            if lde_results:
                results['large_domain_emulation'] = lde_results
        
        return results
    
    def run_large_domain_emulation(self) -> Optional[Dict]:
        """Run large domain emulation workflow."""
        if not 'large_domain_emulator' in self.config.get('OPTIMIZATION_METHODS', []):
            self.logger.info("Large domain emulation is disabled in configuration")
            return None
        
        self.logger.info("Starting large domain emulation workflow")
        
        try:
            # Import here to avoid circular imports
            from symfluence.utils.optimization.legacy.large_domain_emulator import LargeDomainEmulator
            
            # Initialize large domain emulator
            self.large_domain_emulator = LargeDomainEmulator(self.config, self.logger)
            
            # Run the distributed emulation workflow
            results = self.large_domain_emulator.run_distributed_emulation()  # FIXED METHOD NAME
            
            if results:
                self.logger.info("Large domain emulation completed successfully")
                
                # Save results using the results manager
                self._save_large_domain_results(results)
                
                return results
            else:
                self.logger.warning("Large domain emulation did not produce results")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during large domain emulation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _save_large_domain_results(self, results: Dict[str, Any]):
        """Save large domain emulation results to standard SYMFLUENCE location."""
        try:
            # Create large domain results directory
            lde_results_dir = self.project_dir / "optimization" / "large_domain_emulation"
            lde_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save comprehensive results
            results_file = lde_results_dir / f"{self.experiment_id}_large_domain_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Saved large domain emulation results to {results_file}")
            
            # Save best parameters in SYMFLUENCE format if available
            optimization_results = results.get('optimization', {})
            best_params = optimization_results.get('best_parameters')
            
            if best_params and isinstance(best_params, dict):
                # Convert to DataFrame format compatible with other SYMFLUENCE optimizers
                params_data = {'iteration': [0]}
                
                # Add optimization mode and loss
                mode = optimization_results.get('mode', 'unknown')
                best_loss = optimization_results.get('best_loss', float('inf'))
                params_data['optimization_mode'] = [mode]
                params_data['composite_loss'] = [best_loss]
                
                # Add parameter values
                for param_name, value in best_params.items():
                    if isinstance(value, (list, np.ndarray)):
                        params_data[param_name] = [float(np.mean(value))]
                    else:
                        params_data[param_name] = [float(value)]
                
                # Save as CSV for compatibility
                params_df = pd.DataFrame(params_data)
                params_file = lde_results_dir / f"{self.experiment_id}_large_domain_parameters.csv"
                params_df.to_csv(params_file, index=False)
                
                self.logger.info(f"Saved large domain parameters to {params_file}")
                
        except Exception as e:
            self.logger.error(f"Error saving large domain results: {str(e)}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get status of optimization operations.
        
        Enhanced to include large domain emulation status.
        """
        status = {
            'iterative_optimization_enabled': 'iteration' in self.config.get('OPTIMIZATION_METHODS', []),
            'optimization_algorithm': self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO'),
            'optimization_metric': self.config.get('OPTIMIZATION_METRIC', 'KGE'),
            'optimization_dir': str(self.project_dir / "optimization"),
            'results_exist': False,
            'emulation_enabled': 'emulation' in self.config.get('OPTIMIZATION_METHODS', []),
            'rf_emulation_enabled': 'emulation' in self.config.get('OPTIMIZATION_METHODS', []),
            'differentiable_emulation_enabled': 'differentiable_parameter_emulation' in self.config.get('OPTIMIZATION_METHODS', []),
            'large_domain_emulation_enabled': 'large_domain_emulator' in self.config.get('OPTIMIZATION_METHODS', [])
        }
        
        # Check for optimization results
        results_file = self.project_dir / "optimization" / f"{self.experiment_id}_parallel_iteration_results.csv"
        status['results_exist'] = results_file.exists()
        
        # Check for emulation outputs
        emulation_dir = self.project_dir / "emulation" / self.experiment_id
        if emulation_dir.exists():
            status['emulation_parameter_sets'] = (emulation_dir / "parameter_sets.nc").exists()
            status['ensemble_runs_exist'] = (emulation_dir / "ensemble_runs").exists()
            status['performance_metrics_exist'] = (emulation_dir / "ensemble_analysis" / "performance_metrics.csv").exists()
            status['rf_emulation_complete'] = (emulation_dir / "rf_emulation" / "optimized_parameters.csv").exists()
        
        # Check for large domain emulation outputs
        lde_dir = self.project_dir / "optimization" / "large_domain_emulation"
        if lde_dir.exists():
            status['large_domain_results_exist'] = (lde_dir / f"{self.experiment_id}_large_domain_results.json").exists()
            status['large_domain_parameters_exist'] = (lde_dir / f"{self.experiment_id}_large_domain_parameters.csv").exists()
        
        return status
    

    def calibrate_model(self) -> Optional[Path]:
        """
        Calibrate the model using the specified optimization algorithm.
        
        This method coordinates the calibration process for the configured
        hydrological model using the optimization algorithm specified in the
        configuration. It supports calibration for both SUMMA and FUSE models.
        
        The calibration process involves:
        1. Checking if iterative optimization is enabled in the configuration
        2. Determining which optimization algorithm to use
        3. Executing the calibration for each configured hydrological model
        4. Saving and returning the path to the calibration results
        
        The optimization algorithm is specified through the ITERATIVE_OPTIMIZATION_ALGORITHM
        configuration parameter (default: 'PSO'). Supported algorithms include PSO,
        SCE-UA, DDS, DE, and NSGA-II.
        
        Returns:
            Optional[Path]: Path to calibration results file or None if calibration
                        was disabled or failed
                        
        Raises:
            ValueError: If the optimization algorithm is not supported
            RuntimeError: If the calibration process fails
            Exception: For other errors during calibration
        """
        self.logger.info("Starting model calibration")

        # Check if iterative optimization is enabled
        if not 'iteration' in self.config.get('OPTIMIZATION_METHODS', []):
            self.logger.info("Iterative optimization is disabled in configuration")
            return None

        # Get the optimization algorithm from config
        opt_algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO')

        # Check if unified optimizer infrastructure should be used
        # Default to True - use the new consolidated optimizer infrastructure
        use_unified = self.config.get('USE_UNIFIED_OPTIMIZER', True)

        try:
            hydrological_models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')

            for model in hydrological_models:
                model = model.strip().upper()

                # Use registry-based unified optimizer if enabled
                if use_unified and opt_algorithm in ['DDS', 'PSO', 'SCE-UA', 'DE', 'ADAM', 'LBFGS']:
                    return self._calibrate_with_registry(model, opt_algorithm)

                # Fall back to legacy model-specific methods
                if model == 'SUMMA':
                    return self._calibrate_summa(opt_algorithm)
                elif model == 'FUSE':
                    return self._calibrate_fuse(opt_algorithm)
                elif model == 'NGEN':
                    return self._calibrate_ngen(opt_algorithm)
                else:
                    self.logger.warning(f"Calibration for model {model} not yet implemented")

            return None
            
        except Exception as e:
            self.logger.error(f"Error during model calibration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


    def _calibrate_fuse(self, algorithm: str) -> Optional[Path]:
        """
        Calibrate FUSE model using specified algorithm.

        Uses the unified optimizer infrastructure via the registry.

        Args:
            algorithm (str): Optimization algorithm to use

        Returns:
            Optional[Path]: Path to results file or None if optimization failed
        """
        return self._calibrate_with_registry('FUSE', algorithm)

    def _calibrate_ngen(self, algorithm: str) -> Optional[Path]:
        """
        Calibrate NextGen (NGEN) using the specified algorithm.

        Uses the unified optimizer infrastructure via the registry.

        Args:
            algorithm (str): Optimization algorithm to use

        Returns:
            Optional[Path]: Path to results file or None if optimization failed
        """
        return self._calibrate_with_registry('NGEN', algorithm)


    def differentiable_parameter_emulation(self):
        """
        Runs the full Differentiable Parameter Emulation (DPE) workflow.
        """
        self.logger.info("Starting Differentiable Parameter Emulation workflow...")

        try:
            # a. Configure the emulator from your main config file for flexibility
            #    Add these DPE_* parameters to your main config.yaml if you wish.
            emulator_config = EmulatorConfig(
                hidden_dims=self.config.get('DPE_HIDDEN_DIMS', [256, 128, 64]),
                n_training_samples=self.config.get('DPE_TRAINING_SAMPLES', 500),
                n_validation_samples=self.config.get('DPE_VALIDATION_SAMPLES', 100),
                n_epochs=self.config.get('DPE_EPOCHS', 300),
                optimization_steps=self.config.get('DPE_OPTIMIZATION_STEPS', 200),
                learning_rate=self.config.get('DPE_LEARNING_RATE', 1e-3),
                optimization_lr=self.config.get('DPE_OPTIMIZATION_LR', 1e-2),
            )

            dpe = DifferentiableParameterOptimizer(self.config, self.config.get('DOMAIN_NAME'), emulator_config)

            self.logger.info("Running DPE from config (EMULATOR_SETTING)…")
            optimized_params = dpe.run_from_config()

            self.logger.info("Validating optimized parameters with a final SUMMA run…")
            validation_results = dpe.validate_optimization(optimized_params)

            results_dir = Path(f"results_differentiable_{dpe.domain_name}_{datetime.now().strftime('%Y%m%d_%H%M')}")
            dpe.save_results(optimized_params, results_dir)
            return True

        except Exception as e:
            self.logger.error("Differentiable Parameter Emulation workflow failed.", exc_info=True)
            # Re-raise the exception to be caught by the main SYMFLUENCE error handler
            raise e    

    def run_emulation(self) -> Optional[Dict]:
        """
        Entry point for all emulation workflows.

        This method dispatches to the appropriate emulation workflow based on
        the OPTIMIZATION_METHODS configuration.

        .. deprecated::
            run_emulation is deprecated and will be removed in a future version.
            Consider using the unified model optimizers with gradient-based methods
            (run_adam, run_lbfgs) instead.
        """
        warnings.warn(
            "run_emulation is deprecated. Consider using the unified model optimizers "
            "(SUMMAModelOptimizer, FUSEModelOptimizer, NgenModelOptimizer) with "
            "gradient-based methods (run_adam, run_lbfgs) instead.",
            DeprecationWarning,
            stacklevel=2
        )

        optimization_methods = self.config.get('OPTIMIZATION_METHODS', [])
        results = {}
        
        # Run differentiable parameter emulation
        if 'differentiable_parameter_emulation' in optimization_methods:
            dpe_results = self.differentiable_parameter_emulation()
            if dpe_results:
                results['differentiable_parameter_emulation'] = dpe_results
        
        # Run large domain emulation
        if 'large_domain_emulator' in optimization_methods:
            lde_results = self.run_large_domain_emulation()
            if lde_results:
                results['large_domain_emulation'] = lde_results
        
        # If no emulation methods were configured
        if not results:
            if not any(method in optimization_methods for method in ['differentiable_parameter_emulation', 'large_domain_emulator']):
                self.logger.info("No emulation methods enabled in configuration.")
            return None
        
        return results

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
        from symfluence.utils.optimization import model_optimizers  # noqa: F401

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
                    steps=self.config.get('ADAM_STEPS', 100),
                    lr=self.config.get('ADAM_LEARNING_RATE', 0.01)
                ),
                'LBFGS': lambda: optimizer.run_lbfgs(
                    steps=self.config.get('LBFGS_STEPS', 50),
                    lr=self.config.get('LBFGS_LEARNING_RATE', 0.1)
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

    def _calibrate_summa(self, algorithm: str) -> Optional[Path]:
        """
        Calibrate SUMMA model using specified algorithm.
        
        This is an internal method that handles the specifics of calibrating the
        SUMMA hydrological model. It:
        1. Creates the optimization directory if it doesn't exist
        2. Loads the appropriate optimizer class based on the algorithm
        3. Executes the optimization process
        4. Saves and returns the results
        
        The method supports different optimization algorithms (PSO, SCE-UA, DDS)
        through a dynamic dispatch approach using the optimizers and optimizer_methods
        mappings.
        
        Args:
            algorithm (str): Optimization algorithm to use ('PSO', 'SCE-UA', or 'DDS')
            
        Returns:
            Optional[Path]: Path to results file or None if optimization failed
            
        Raises:
            ValueError: If the specified algorithm is not supported
            RuntimeError: If the optimization process fails
            Exception: For other errors during optimization
        """
        # Create optimization directory if it doesn't exist
        opt_dir = self.project_dir / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)
        
        # Get optimizer class
        optimizer_class = self.optimizers.get(algorithm)
        
        if optimizer_class is None:
            self.logger.error(f"Optimization algorithm {algorithm} not supported")
            return None
        
        # Get optimizer method name
        method_name = self.optimizer_methods.get(algorithm)
        
        if method_name is None:
            self.logger.error(f"No run method defined for optimizer {algorithm}")
            return None
        
        try:
            # Initialize optimizer
            self.logger.info(f"Using {algorithm} optimization")
            optimizer = optimizer_class(self.config, self.logger)
            
            # Run optimization
            result = getattr(optimizer, method_name)()
            
            # Save results to standard location
            target_metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
            results_file = self.results_manager.save_optimization_results(
                result, algorithm, target_metric
            )
            
            if results_file:
                self.logger.info(f"Calibration completed successfully: {results_file}")
                return results_file
            else:
                self.logger.warning("Calibration completed but results file not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during {algorithm} optimization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get status of optimization operations.
        
        This method provides a comprehensive status report on the optimization
        and emulation operations. It checks for the existence of key files and
        directories to determine which steps have been completed successfully.
        
        The status information includes:
        - Whether iterative optimization is enabled
        - Which optimization algorithm is configured
        - Which performance metric is being used
        - Whether optimization results exist
        - Whether emulation is enabled and its status
        - The existence of various emulation outputs
        
        This information is useful for tracking progress, diagnosing issues,
        and providing feedback to users.
        
        Returns:
            Dict[str, Any]: Dictionary containing optimization status information,
                          including configuration settings and existence of output files
        """
        status = {
            'iterative_optimization_enabled': 'iteration' in self.config.get('OPTIMIZATION_METHODS', []),
            'optimization_algorithm': self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO'),
            'optimization_metric': self.config.get('OPTIMIZATION_METRIC', 'KGE'),
            'optimization_dir': str(self.project_dir / "optimization"),
            'results_exist': False,
            'emulation_enabled': 'emulation' in self.config.get('OPTIMIZATION_METHODS', []),
            'rf_emulation_enabled': 'emulation' in self.config.get('OPTIMIZATION_METHODS', [])
        }
        
        # Check for optimization results
        results_file = self.project_dir / "optimization" / f"{self.experiment_id}_parallel_iteration_results.csv"
        status['results_exist'] = results_file.exists()
        
        # Check for emulation outputs
        emulation_dir = self.project_dir / "emulation" / self.experiment_id
        if emulation_dir.exists():
            status['emulation_parameter_sets'] = (emulation_dir / "parameter_sets.nc").exists()
            status['ensemble_runs_exist'] = (emulation_dir / "ensemble_runs").exists()
            status['performance_metrics_exist'] = (emulation_dir / "ensemble_analysis" / "performance_metrics.csv").exists()
            status['rf_emulation_complete'] = (emulation_dir / "rf_emulation" / "optimized_parameters.csv").exists()
        
        return status
    
    def validate_optimization_configuration(self) -> Dict[str, bool]:
        """
        Validate optimization configuration settings.
        
        This method checks the configuration for optimization to ensure that
        all required settings are present and valid. It verifies:
        1. Whether the specified optimization algorithm is supported
        2. Whether the model to be calibrated is supported
        3. Whether parameters to calibrate are defined
        4. Whether the performance metric is valid
        
        These validations help prevent runtime errors by catching configuration
        issues before the optimization process begins.
        
        Returns:
            Dict[str, bool]: Dictionary containing validation results for each
                           aspect of the optimization configuration:
                           - algorithm_valid: Whether the algorithm is supported
                           - model_supported: Whether the model is supported for calibration
                           - parameters_defined: Whether parameters are defined for calibration
                           - metric_valid: Whether the performance metric is valid
        """
        validation = {
            'algorithm_valid': False,
            'model_supported': False,
            'parameters_defined': False,
            'metric_valid': False
        }
        
        # Check algorithm
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', '')
        validation['algorithm_valid'] = algorithm in self.optimizers
        
        # Check model support
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        validation['model_supported'] = 'SUMMA' in [m.strip() for m in models]
        
        # Check parameters to calibrate
        local_params = self.config.get('PARAMS_TO_CALIBRATE', '')
        basin_params = self.config.get('BASIN_PARAMS_TO_CALIBRATE') or ''
        validation['parameters_defined'] = bool(local_params or basin_params)
        
        # Check metric
        valid_metrics = ['KGE', 'NSE', 'RMSE', 'MAE', 'KGEp']
        metric = self.config.get('OPTIMIZATION_METRIC', '')
        validation['metric_valid'] = metric in valid_metrics
        
        return validation
    
    def get_available_optimizers(self) -> Dict[str, str]:
        """
        Get list of available optimization algorithms.
        
        This method provides information about the optimization algorithms
        supported by the OptimizationManager. The information includes both
        the algorithm identifiers used in configuration and descriptions of
        each algorithm.
        
        This information is useful for user interfaces and documentation to
        help users select an appropriate optimization algorithm.
        
        Returns:
            Dict[str, str]: Dictionary mapping algorithm identifiers to their
                          descriptions:
                          - 'PSO': 'Particle Swarm Optimization'
                          - 'SCE-UA': 'Shuffled Complex Evolution'
                          - 'DDS': 'Dynamically Dimensioned Search'
        """
        return {
            'PSO': 'Particle Swarm Optimization',
            'SCE-UA': 'Shuffled Complex Evolution',
            'DDS': 'Dynamically Dimensioned Search'
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
        from symfluence.utils.evaluation.analysis_manager import AnalysisManager
        am = AnalysisManager(self.config, self.logger)
        eval_results = am.run_multivariate_evaluation(sim_results)
        
        # 3. Calculate scalar objective
        return objective_handler.calculate(eval_results)

    def load_optimization_results(self, filename: str = None) -> Optional[Dict]:
        """
        Load optimization results from file.
        
        This method loads and formats optimization results from a previously
        completed calibration run. It converts the results from CSV format to
        a dictionary structure that can be easily used for analysis or visualization.
        
        If no filename is provided, the method loads the results for the current
        experiment ID.
        
        Args:
            filename (str, optional): Name of results file to load. If None, uses
                                    the default filename based on experiment_id.
            
        Returns:
            Optional[Dict]: Dictionary with optimization results organized as:
                          - 'parameters': List of parameter sets as dictionaries
                          - 'best_iteration': Dictionary of the best parameter set
                          - 'columns': List of column names from the results file
                          Returns None if loading fails
                          
        Raises:
            FileNotFoundError: If the results file cannot be found
            ValueError: If the file format is invalid
            Exception: For other errors during loading
        """
        try:
            results_df = self.results_manager.load_optimization_results(filename)
            
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

        if not results_file.exists():
            from symfluence.utils.exceptions import FileOperationError
            raise FileOperationError(f"Optimization results file not found: {results_file}")
        
        return pd.read_csv(results_file)
