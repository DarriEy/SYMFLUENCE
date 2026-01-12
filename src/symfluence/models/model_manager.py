"""Model Manager

Orchestrates complete hydrological modeling workflow within SYMFLUENCE framework.
Coordinates preprocessing of forcing data, model execution, output postprocessing,
and visualization across multiple models using registry-based plugin architecture.

Architecture:
    The ModelManager acts as a facade for the entire modeling pipeline:

    1. Workflow Resolution (_resolve_model_workflow)
       - Parses HYDROLOGICAL_MODEL configuration parameter
       - Automatically adds implicit dependencies (e.g., mizuRoute)
       - Returns ordered execution list based on model relationships

    2. Model Registry System
       - Each model registers components: preprocessor, runner, postprocessor
       - Registry enables dynamic model discovery without code changes
       - Simplifies adding new models: register components, update config

    3. Preprocessing Phase (preprocess_models)
       - Iterates through resolved workflow
       - Invokes model-specific preprocessor
       - Converts generic forcing data to model input formats
       - Example: Convert ERA5 to SUMMA forcing files

    4. Execution Phase (run_models)
       - Iterates through resolved workflow
       - Retrieves registered runner for each model
       - Executes model via model-specific runner method
       - Captures outputs to model output directory

    5. Postprocessing Phase (postprocess_results)
       - Iterates through resolved workflow
       - Invokes model postprocessor to extract results
       - Standardizes outputs to common format (e.g., streamflow CSV)
       - Calculates baseline performance metrics automatically

    6. Visualization Phase (visualize_outputs)
       - Invokes registered visualizer for each primary model
       - Generates timeseries plots, diagnostics, comparisons

Supported Models:
    Primary hydrological models:
    - SUMMA: Land surface model with distributed discretization
    - FUSE: Modular/flexible semi-distributed model
    - GR4J/GR6J: Lumped conceptual rainfall-runoff
    - HYPE: Semi-distributed model with internal routing
    - MESH: Pan-Arctic hydrological model
    - NGEN: NextGen modular framework (experimental)
    - LSTM: Neural network surrogate (experimental)

    Routing/postprocessing models:
    - mizuRoute: Streamflow routing model (automatic dependency)

Dependency Injection via mizuRoute:
    Some models lack streamflow routing and require mizuRoute:
    - SUMMA + mizuRoute: Distributed + routing
    - FUSE + mizuRoute: Flexible + routing
    - GR + mizuRoute: Lumped + routing
    - HYPE: Internal routing, no mizuRoute needed
    - MESH: Internal routing, no mizuRoute needed

    Decision made by RoutingDecider.needs_routing() based on:
    - Model type
    - Configuration (routing_file, routing_config)
    - Evaluation targets (if streamflow evaluation enabled)

Registry-Based Plugin System:
    Each model registers components via ModelRegistry:

    1. Preprocessor Registration
       @ModelRegistry.register_preprocessor('MYMODEL')
       class MyPreprocessor:
           def run_preprocessing(self): ...

    2. Runner Registration
       @ModelRegistry.register_runner('MYMODEL', method_name='run_model')
       class MyRunner:
           def run_model(self): ...

    3. Postprocessor Registration
       @ModelRegistry.register_postprocessor('MYMODEL')
       class MyPostprocessor:
           def extract_streamflow(self): ...

    4. Visualizer Registration
       @ModelRegistry.register_visualizer('MYMODEL')
       def visualize_mymodel(reporting_manager, config, project_dir, ...): ...

    Benefits:
    - No changes to ModelManager to add new models
    - Easy to enable/disable models via configuration
    - Loose coupling between models and orchestrator
    - Enables third-party model contributions

Baseline Performance Metrics:
    After postprocessing, automatically calculates and logs:
    - KGE (Kling-Gupta Efficiency): Balance of correlation, variability, bias
    - KGE' (Modified KGE): Symmetric variant
    - NSE (Nash-Sutcliffe Efficiency): Correlation-based metric
    - Bias (%): Mean bias relative to observations

    Useful for:
    - Understanding initial model performance
    - Detecting model setup issues before calibration
    - Establishing baseline for improvement assessment
    - Identifying models that may need configuration changes

Examples:
    >>> # Create manager with config and logger
    >>> from symfluence.models.model_manager import ModelManager
    >>> manager = ModelManager(config, logger, reporting_manager=reporter)

    >>> # Run complete workflow
    >>> manager.preprocess_models()
    >>> manager.run_models()
    >>> manager.postprocess_results()
    >>> manager.visualize_outputs()

    >>> # Run with parameter variations (for calibration)
    >>> params = {'SAI_SV': 0.3, 'snowCriticalTemp': -1.5}
    >>> manager.preprocess_models(params=params)
    >>> manager.run_models()
    >>> manager.postprocess_results()

References:
    - Kling, H., et al. (2012). A framework for understanding structural and
      parametric uncertainty in hydrological modelling. Water Resources
      Management, 26(12), 3555-3569.
    - Gupta, H. V., et al. (2009). Decomposition of the mean squared error
      and NSE performance criteria. Journal of Hydrology, 377(1-2), 80-91.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING

import pandas as pd

from symfluence.core.base_manager import BaseManager
from symfluence.models.registry import ModelRegistry
from symfluence.models.utilities.routing_decider import RoutingDecider

if TYPE_CHECKING:
    pass


class ModelManager(BaseManager):
    """Orchestrates complete hydrological modeling workflow.

    Coordinates preprocessing, execution, postprocessing, and visualization
    of multiple hydrological models through registry-based plugin architecture.
    Automatically handles model dependencies (e.g., mizuRoute routing).

    Architecture Overview:
        Workflow Resolution → Preprocessing → Execution → Postprocessing → Visualization

        1. Workflow Resolution (_resolve_model_workflow)
           - Parses config.model.hydrological_model (comma-separated list)
           - Auto-adds mizuRoute if model requires routing
           - Returns execution order respecting dependencies

        2. Preprocessing (preprocess_models)
           - Converts generic forcing to model-specific input formats
           - Example: ERA5 → SUMMA forcings, NetCDF → GR input files
           - Invokes model-specific preprocessor from registry

        3. Execution (run_models)
           - Runs each model in workflow order
           - Invokes registered runner with model-specific method
           - Captures outputs to model output directory

        4. Postprocessing (postprocess_results)
           - Extracts streamflow and other outputs
           - Standardizes to common format (CSV with datetime index)
           - Calculates baseline performance metrics
           - Triggers visualizations

        5. Visualization (visualize_outputs)
           - Generates timeseries plots, diagnostics, comparisons
           - Requires reporting_manager to be configured

    Key Responsibilities:
        - Resolve model execution order and handle implicit dependencies
        - Preprocess forcing data into model-specific input formats
        - Execute models using registered runner classes
        - Postprocess model outputs and extract standardized results
        - Calculate baseline performance metrics (KGE, NSE, Bias)
        - Coordinate output visualization and reporting

    Supported Models:
        Primary: SUMMA, FUSE, GR4J, HYPE, MESH, NGEN, LSTM
        Routing: mizuRoute (automatically added when needed)
        Configuration: model.hydrological_model = "SUMMA,FUSE" (comma-separated)

    Model Dependencies:
        SUMMA/FUSE/GR + (routing enabled) → Adds MIZUROUTE automatically
        HYPE/MESH/NGEN → Internal routing, no mizuRoute needed
        LSTM → Data-driven, no routing needed

    Registry-Based Plugin System:
        Each model registers preprocessor, runner, postprocessor via ModelRegistry:
        - Enables dynamic model discovery without modifying this class
        - Simplifies adding new models: register components, update config
        - Example: SUMMA registers SUMMAPreprocessor, SUMMARunner, SUMMAPostprocessor

    Parameter-Based Preprocessing:
        Preprocessing accepts optional params dict for calibration:
        >>> manager.preprocess_models(params={'SAI_SV': 0.3, 'snowCriticalTemp': -1.5})
        This applies parameter values during preprocessing if needed.

    Baseline Metrics:
        Automatically logged after postprocessing:
        - KGE (Kling-Gupta Efficiency): Balanced metric combining correlation, variability, bias
        - KGE' (Modified KGE): Symmetric variant for metric comparison
        - NSE (Nash-Sutcliffe): Correlation-based efficiency metric
        - Bias (%): Mean bias relative to observations
        - Interpretation: KGE >= 0.7 indicates reasonable performance

    Attributes:
        _routing_decider: Class-level RoutingDecider instance for dependency resolution
        config: SymfluenceConfig typed configuration object
        logger: Logger instance
        project_dir: Project root directory (from BaseManager)
        reporting_manager: ReportingManager for visualization (optional)

    Inherits from:
        BaseManager: Provides config, logger, project_dir, and utility methods

    Examples:
        >>> # Standard workflow
        >>> manager = ModelManager(config, logger, reporting_manager=reporter)
        >>> manager.preprocess_models()
        >>> manager.run_models()
        >>> manager.postprocess_results()
        >>> manager.visualize_outputs()

        >>> # With parameter variations (calibration scenario)
        >>> params = {'param1': 0.5, 'param2': 100.0}
        >>> manager.preprocess_models(params=params)
        >>> manager.run_models()
        >>> manager.postprocess_results()

    See Also:
        ModelRegistry: Dynamic model component registration
        RoutingDecider: Logic for mizuRoute dependency determination
        BaseManager: Parent class providing config/logger utilities
    """

    # Shared routing decision logic (class-level for efficiency)
    _routing_decider = RoutingDecider()

    def _resolve_model_workflow(self) -> List[str]:
        """Resolve model execution order including implicit dependencies.

        Parses config.model.hydrological_model (comma-separated list) and automatically
        adds dependent models (e.g., mizuRoute) based on model capabilities and
        configuration. Enables flexible model combinations while hiding complexity.

        Workflow Resolution Algorithm:
            1. Parse HYDROLOGICAL_MODEL config parameter (e.g., "SUMMA,FUSE")
            2. Iterate through each model in specified order
            3. For routable models (SUMMA, FUSE, GR):
               - Check if routing required via RoutingDecider
               - Automatically add MIZUROUTE to workflow if needed
            4. Skip models with internal routing (HYPE, MESH, NGEN)
            5. Return execution list

        Routable vs Non-Routable Models:
            Routable (require mizuRoute if routing enabled):
            - SUMMA: Distributed land surface model (needs routing for streamflow)
            - FUSE: Flexible conceptual framework (needs routing)
            - GR: Lumped rainfall-runoff (needs routing)

            Non-Routable (internal routing, skip mizuRoute):
            - HYPE: Semi-distributed with built-in routing
            - MESH: Pan-Arctic model with built-in routing
            - NGEN: NextGen framework with built-in routing

            Non-Hydrological (no routing needed):
            - LSTM: Neural network surrogate
            - Other: Data-driven models

        Routing Decision Logic (via RoutingDecider):
            mizuRoute is added if:
            1. Model is routable (SUMMA, FUSE, GR) AND
            2. Routing configuration exists (routing_file or routing_config) AND
            3. Streamflow evaluation is enabled in optimization targets OR
               Post-processing streamflow extraction is enabled

        Returns:
            List[str]: Model names in execution order
            - Typically [primary_model] or [primary_model, 'MIZUROUTE']
            - Example: ['SUMMA', 'MIZUROUTE']
            - Example: ['HYPE']  (no mizuRoute, internal routing)
            - Example: ['FUSE', 'MIZUROUTE']

        Examples:
            >>> # Single model with routing
            >>> manager._resolve_model_workflow()
            ['SUMMA', 'MIZUROUTE']

            >>> # Single model without routing (internal)
            >>> manager._resolve_model_workflow()
            ['HYPE']

            >>> # Multiple models
            >>> manager._resolve_model_workflow()
            ['SUMMA', 'MIZUROUTE', 'LSTM']

        See Also:
            _ensure_mizuroute_in_workflow(): Add mizuRoute and log context
            RoutingDecider.needs_routing(): Determine routing requirements
            run_models(): Execute resolved workflow
        """
        models_str = self.config.model.hydrological_model or ''
        configured_models = [m.strip() for m in str(models_str).split(',') if m.strip()]
        execution_list = []

        # Models that support routing via mizuRoute
        # Note: MESH, HYPE, and NGEN have internal routing, so don't need mizuRoute
        routable_models = {'SUMMA', 'FUSE', 'GR'}

        for model in configured_models:
            if model not in execution_list:
                execution_list.append(model)

            # Check implicit dependencies (e.g. mizuRoute) using shared routing decider
            if model in routable_models:
                if self._routing_decider.needs_routing(self.config_dict, model):
                    self._ensure_mizuroute_in_workflow(execution_list, source_model=model)

        return execution_list

    def _ensure_mizuroute_in_workflow(self, execution_list: List[str], source_model: str):
        """
        Add mizuRoute to workflow and log routing context.

        Args:
            execution_list: Current list of models to execute (modified in-place)
            source_model: Name of the model that requires routing (e.g., 'SUMMA')
        """
        if 'MIZUROUTE' not in execution_list:
            execution_list.append('MIZUROUTE')
            self.logger.info(f"Automatically adding MIZUROUTE to workflow (dependency of {source_model})")

        # Check if MIZU_FROM_MODEL is set in config
        mizu_from = self._get_config_value(
            lambda: self.config.model.mizuroute.from_model if self.config.model.mizuroute else None,
            default=None
        )
        if not mizu_from:
            # Log the source model (config is immutable, so we can't update it)
            self.logger.info(f"MIZU_FROM_MODEL not set, using {source_model} as source")

    def preprocess_models(self, params: Optional[Dict[str, Any]] = None):
        """Preprocess forcing data into model-specific input formats.

        Transforms generic forcing data (from data acquisition) into model-specific
        input formats. Invokes registered preprocessor for each model in resolved
        workflow. Preprocessors handle all model-specific input requirements.

        Preprocessing Workflow:
            1. Resolve model workflow (includes implicit dependencies)
            2. For each model in workflow:
               a. Create model input directory (project_dir/forcing/{MODEL}_input/)
               b. Retrieve preprocessor class from ModelRegistry
               c. Instantiate preprocessor with config, logger, and params
               d. Run preprocessor.run_preprocessing()
            3. Preprocessor outputs go to model-specific input directories

        Model-Specific Preprocessing Examples:
            SUMMA:
            - ERA5 NetCDF → SUMMA forcing file format
            - Time step interpolation/aggregation
            - Unit conversion (SI → SUMMA units)
            - Spatial interpolation to model grid

            FUSE:
            - Catchment-averaged forcing extraction
            - Temporal aggregation to model timestep
            - Unit conversion

            GR (Rainfall-Runoff):
            - Daily precipitation, temperature aggregation
            - Missing value handling

            mizuRoute:
            - Basin delineation and network structure
            - Unit hydrograph parameters
            - Routing network initialization

        Parameter Usage:
            params dict passed to preprocessor for calibration scenarios:
            - Preprocessor may use params to adjust input processing
            - Example: Parameter-dependent unit conversion or scaling
            - If preprocessor doesn't accept params, they're ignored

        Args:
            params: Optional Dict[str, Any] with parameter values
                - Example: {'SAI_SV': 0.3, 'snowCriticalTemp': -1.5}
                - Used for calibration (different param values → different inputs)
                - If None, uses default parameter values from config
                - Preprocessor determines if params are needed (introspection)

        Raises:
            Exception: If preprocessing fails for any model (logged and re-raised)
                - Caught internally with full traceback logged
                - Enables debugging of preprocessing issues

        Side Effects:
            - Creates project_dir/forcing/{MODEL}_input/ directories
            - Generates model-specific input files
            - Logs preprocessing progress and errors to logger

        Examples:
            >>> # Standard preprocessing with default parameters
            >>> manager.preprocess_models()

            >>> # Preprocessing with parameter variations (calibration)
            >>> params = {'param1': 0.5, 'param2': 100.0}
            >>> manager.preprocess_models(params=params)

        Notes:
            - LSTM and similar data-driven models skip preprocessing
            - Registry lookup enables new models without modifying this method
            - Parameter introspection (inspect.signature) handles optional params
            - Errors in preprocessing halt workflow and raise exception

        See Also:
            ModelRegistry.get_preprocessor(): Retrieve preprocessor class
            run_models(): Execute preprocessed models
            postprocess_results(): Extract and standardize results
        """
        self.logger.debug("Starting model-specific preprocessing")

        workflow = self._resolve_model_workflow()
        self.logger.debug(f"Preprocessing workflow order: {workflow}")

        for model in workflow:
            try:
                # Create model input directory
                model_input_dir = self.project_dir / "forcing" / f"{model}_input"
                model_input_dir.mkdir(parents=True, exist_ok=True)

                # Select preprocessor for this model from registry
                preprocessor_class = ModelRegistry.get_preprocessor(model)

                if preprocessor_class is None:
                    # Models that truly don't need preprocessing (e.g., LSTM)
                    if model in ['LSTM']:
                        self.logger.debug(f"Model {model} doesn't require preprocessing")
                    else:
                        # Only warn if it's a primary model, not a utility like MIZUROUTE which definitely has one
                        self.logger.debug(f"No preprocessor registered for {model} (or not required).")
                    continue

                # Run model-specific preprocessing
                self.logger.debug(f"Running preprocessor for {model}")

                # Check preprocessor signature to determine what arguments to pass
                import inspect
                sig = inspect.signature(preprocessor_class.__init__)
                kwargs = {}

                # Add optional params if supported
                if 'params' in sig.parameters:
                    kwargs['params'] = params

                # Add LSTM-specific arguments if needed
                if model == 'LSTM':
                    if 'project_dir' in sig.parameters:
                        kwargs['project_dir'] = self.project_dir
                    if 'device' in sig.parameters:
                        # Determine device - prefer GPU if available
                        try:
                            import torch
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        except ImportError:
                            device = None
                        kwargs['device'] = device

                preprocessor = preprocessor_class(self.config, self.logger, **kwargs)

                # Call appropriate preprocessing method
                if hasattr(preprocessor, 'run_preprocessing'):
                    preprocessor.run_preprocessing()
                else:
                    # Some models like LSTM don't need preprocessing
                    self.logger.debug(f"No run_preprocessing method for {model} preprocessor")

            except Exception as e:
                self.logger.error(f"Error preprocessing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        self.logger.info("Model-specific preprocessing completed")

        

    def run_models(self):
        """Execute models in resolved workflow order using registered runners.

        Invokes each model's registered runner to execute the model with preprocessed
        inputs. Runners handle all model-specific execution details (e.g., executable
        path, command-line arguments, working directory setup).

        Execution Workflow:
            1. Resolve model execution workflow
            2. For each model in workflow:
               a. Retrieve runner class from ModelRegistry
               b. Instantiate runner with config, logger, reporting_manager
               c. Get runner method name from registry (e.g., 'run_model')
               d. Execute runner.{method_name}()
            3. Each runner generates model outputs to project_dir/{model}_output/

        Model-Specific Runners:
            SUMMA: SUMMARunner.run_model()
            - Invokes SUMMA executable with file manager and settings
            - Outputs: NetCDF files to project_dir/SUMMA_output/

            FUSE: FUSERunner.run_model()
            - Runs flexible model framework with specified structure
            - Outputs: ASCII/NetCDF to project_dir/FUSE_output/

            GR: GRRunner.run_model()
            - Executes GR lumped model
            - Outputs: Daily discharge CSV

            HYPE: HYPERunner.run_model()
            - Runs HYPE semi-distributed model
            - Outputs: Model-specific format to project_dir/HYPE_output/

            mizuRoute: mizuRouteRunner.run_model()
            - Routes SUMMA/FUSE streamflow to outlet
            - Requires SUMMA/FUSE outputs as input
            - Outputs: Routed streamflow to project_dir/MIZUROUTE_output/

        Registry Integration:
            Each model registers its runner class and method name:
            @ModelRegistry.register_runner('MYMODEL', method_name='run_model')
            class MyRunner:
                def run_model(self): ...

        Execution Dependencies:
            Order matters for models with dependencies:
            1. SUMMA runs first, produces outputs
            2. mizuRoute runs second, reads SUMMA outputs
            3. Order preserved by _resolve_model_workflow()

        Output Directories:
            Each runner saves outputs to: project_dir/{MODEL}_output/
            Example structure:
            project_dir/
            ├── SUMMA_output/
            │   ├── output.nc (NetCDF with all variables)
            │   └── ...
            ├── MIZUROUTE_output/
            │   ├── Qrouted*.txt (routed streamflow)
            │   └── ...
            └── results/
                └── {experiment_id}_results.csv (postprocessed)

        Raises:
            Exception: If model execution fails (logged with traceback and re-raised)
                - Enables debugging of model failures
                - Halts workflow to prevent cascading errors

        Side Effects:
            - Creates model output directories
            - Generates model-specific output files
            - Logs execution progress to logger
            - Modifies reporting_manager state (progress tracking)

        Examples:
            >>> # Execute complete model workflow
            >>> manager.preprocess_models()
            >>> manager.run_models()  # Both SUMMA and MIZUROUTE run

            >>> # With visualization/progress reporting
            >>> manager.run_models()  # Progress tracked via reporting_manager

        Notes:
            - Registry lookup enables new models without modifying this method
            - Runner method names retrieved from registry (flexible)
            - Execution order preserved from _resolve_model_workflow()
            - Unknown models logged as error and skipped

        See Also:
            ModelRegistry.get_runner(): Retrieve runner class
            ModelRegistry.get_runner_method(): Get method name
            preprocess_models(): Prepare inputs
            postprocess_results(): Extract and standardize outputs
        """
        self.logger.info("Starting model runs")

        workflow = self._resolve_model_workflow()
        self.logger.info(f"Execution workflow order: {workflow}")

        for model in workflow:
            try:
                self.logger.info(f"Running model: {model}")
                runner_class = ModelRegistry.get_runner(model)
                if runner_class is None:
                    self.logger.error(f"Unknown hydrological model or no runner registered: {model}")
                    continue

                runner = runner_class(self.config, self.logger, reporting_manager=self.reporting_manager)
                method_name = ModelRegistry.get_runner_method(model)
                if method_name and hasattr(runner, method_name):
                    getattr(runner, method_name)()
                else:
                    self.logger.error(f"Runner method '{method_name}' not found for model: {model}")
                    continue

            except Exception as e:
                self.logger.error(f"Error running model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

    def postprocess_results(self):
        """
        Post-process model results using the registry.

        Extracts streamflow and other relevant outputs from model-specific result
        files and converts them to a standardized format for evaluation and comparison.
        After postprocessing, calculates and logs baseline performance metrics.

        The standardized interface expects postprocessors to implement extract_streamflow()
        method, which saves results to: project_dir/results/{experiment_id}_results.csv

        Note:
            Automatically triggers visualization of timeseries results after extraction.
            Falls back to legacy extract_results() method for backward compatibility.
        """
        self.logger.info("Starting model post-processing")

        workflow = self._resolve_model_workflow()

        for model in workflow:
            try:
                # Get postprocessor class from registry
                postprocessor_class = ModelRegistry.get_postprocessor(model)

                if postprocessor_class is None:
                    continue

                self.logger.info(f"Post-processing {model}")
                # Create postprocessor instance
                postprocessor = postprocessor_class(self.config, self.logger, reporting_manager=self.reporting_manager)

                # Run postprocessing
                # Standardized interface: extract_streamflow is the main entry point
                if hasattr(postprocessor, 'extract_streamflow'):
                    postprocessor.extract_streamflow()
                elif hasattr(postprocessor, 'extract_results'):
                    # Legacy support for models that might still use extract_results (e.g. HYPE if not updated)
                    postprocessor.extract_results()
                else:
                    self.logger.warning(f"No extraction method found for {model} postprocessor")
                    continue

            except Exception as e:
                self.logger.error(f"Error post-processing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Note: visualize_timeseries_results is now triggered automatically by extract_streamflow/save_streamflow_to_results

        # Log baseline performance metrics after postprocessing
        self.log_baseline_performance()

    def log_baseline_performance(self):
        """Log baseline model performance metrics before calibration.

        Calculates and logs performance metrics comparing simulated vs observed
        streamflow after initial model run. Provides diagnostic snapshot of model
        performance before calibration, enabling users to:
        1. Assess initial model setup quality
        2. Detect configuration issues
        3. Establish baseline for improvement assessment
        4. Identify models needing attention before calibration

        Metrics Calculated:
            KGE (Kling-Gupta Efficiency):
            - Formula: KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
            - r: Correlation coefficient (0-1, perfect=1)
            - α: Ratio of simulated to observed std dev
            - β: Ratio of simulated to observed mean
            - Range: [-∞, 1], interpretation:
              * KGE ≥ 0.7: Reasonable performance
              * 0.5 ≤ KGE < 0.7: Requires calibration
              * KGE < 0.5: Needs significant improvements
              * KGE < 0: Worse than using observed mean

            KGE' (Modified KGE):
            - Symmetric variant for metric comparison
            - Useful when comparing multiple model configurations

            NSE (Nash-Sutcliffe Efficiency):
            - Formula: NSE = 1 - (Σ(Qobs-Qsim)² / Σ(Qobs-Qmean)²)
            - Correlation-based metric
            - Range: [-∞, 1], similar interpretation as KGE
            - Less sensitive to bias and variability than KGE

            Bias (%):
            - Formula: Bias = ((Mean_Sim - Mean_Obs) / Mean_Obs) × 100
            - Positive: Model overestimates
            - Negative: Model underestimates
            - Can indicate systematic model errors

        Data Sources:
            1. Simulation Results: results_dir/{experiment_id}_results.csv
               - Generated by postprocess_results()
               - Contains model discharge columns

            2. Observations: Multiple fallback strategies
               a. Results file column (if 'obs' or 'observed' in column name)
               b. Observations directory: project_dir/observations/streamflow/preprocessed/
               c. External observation file with datetime and discharge columns

        Workflow:
            1. Load simulation results CSV (index=datetime)
            2. Find observation column or load from observations directory
            3. For each simulation column (e.g., 'SUMMA_discharge_cms'):
               a. Align observations and simulation by datetime index
               b. Remove NaN pairs
               c. Calculate metrics (KGE, KGE', NSE, Bias)
               d. Log results with interpretation
            4. Log footer with metrics summary

        Output Format:
            ========================================================
            BASELINE MODEL PERFORMANCE (before calibration)
            ========================================================
              MODELNAME:
                KGE  = 0.7234
                KGE' = 0.7156
                NSE  = 0.6987
                Bias = +5.3%
                Valid data points: 1825
              Note: KGE >= 0.7 indicates reasonable baseline performance
            ========================================================

        Error Handling:
            - Results file not found: Skipped with debug message
            - No observations found: Skipped with debug message
            - Insufficient valid data (<10 points): Logged as warning
            - Metric calculation errors: Caught and logged as debug

        Side Effects:
            - Logs baseline metrics to logger.info()
            - Logs interpretation and recommendations
            - No files created or modified

        Examples:
            >>> # Called automatically by postprocess_results()
            >>> manager.postprocess_results()  # Includes baseline logging

            >>> # Or called directly
            >>> manager.log_baseline_performance()

        Notes:
            - Called automatically by postprocess_results()
            - Requires results file from postprocessing
            - Useful for QA/QC before calibration begins
            - Graceful degradation if data not available
            - KGE interpretation helps understand model biases

        See Also:
            postprocess_results(): Automatically calls log_baseline_performance()
            kge(), kge_prime(), nse(): Metric calculation functions
            evaluation.metrics: Metric library
        """
        try:
            import numpy as np
            from symfluence.evaluation.metrics import kge, nse, kge_prime

            # Get results file path
            results_file = self.project_dir / "results" / f"{self.experiment_id}_results.csv"

            if not results_file.exists():
                self.logger.debug("Results file not found - skipping baseline performance logging")
                return

            # Load results
            results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)

            # Find observation column in results, or load from observations directory
            obs_col = None
            obs_series = None
            for col in results_df.columns:
                if 'obs' in col.lower() or 'observed' in col.lower():
                    obs_col = col
                    break

            if obs_col is None:
                # Try to load observations from standard location
                obs_dir = self.project_dir / "observations" / "streamflow" / "preprocessed"
                domain_name = self.config.domain.name
                obs_files = list(obs_dir.glob(f"{domain_name}*_streamflow*.csv")) if obs_dir.exists() else []

                if obs_files:
                    try:
                        obs_df = pd.read_csv(obs_files[0])
                        # Find datetime and discharge columns
                        datetime_col = None
                        discharge_col = None
                        for col in obs_df.columns:
                            if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                                datetime_col = col
                            if 'discharge' in col.lower() or 'flow' in col.lower() or col.lower() == 'q':
                                discharge_col = col

                        if datetime_col and discharge_col:
                            obs_df[datetime_col] = pd.to_datetime(obs_df[datetime_col])
                            obs_series = obs_df.set_index(datetime_col)[discharge_col]
                            obs_series = obs_series.resample('D').mean()  # Resample to daily
                            self.logger.debug(f"Loaded observations from {obs_files[0].name}")
                    except Exception as e:
                        self.logger.debug(f"Could not load observations: {e}")

                if obs_series is None:
                    self.logger.debug("No observation data found - skipping baseline metrics")
                    return

            # Find simulation columns (model outputs)
            sim_cols = [c for c in results_df.columns if 'discharge' in c.lower()]

            if not sim_cols:
                self.logger.debug("No simulation columns found in results")
                return

            # Log header
            self.logger.info("=" * 60)
            self.logger.info("BASELINE MODEL PERFORMANCE (before calibration)")
            self.logger.info("=" * 60)

            for sim_col in sim_cols:
                sim_series = results_df[sim_col]

                # Get observations - either from results file column or externally loaded
                if obs_col is not None:
                    obs_aligned = results_df[obs_col]
                    sim_aligned = sim_series
                elif obs_series is not None:
                    # Align observations with simulation by index (datetime)
                    common_idx = sim_series.index.intersection(obs_series.index)
                    if len(common_idx) == 0:
                        self.logger.warning(f"  {sim_col}: No overlapping dates with observations")
                        continue
                    obs_aligned = obs_series.loc[common_idx]
                    sim_aligned = sim_series.loc[common_idx]
                else:
                    continue

                obs = obs_aligned.values
                sim = sim_aligned.values

                # Remove NaN pairs
                valid_mask = ~(np.isnan(obs) | np.isnan(sim))
                obs_clean = obs[valid_mask]
                sim_clean = sim[valid_mask]

                if len(obs_clean) < 10:
                    self.logger.warning(f"  {sim_col}: Insufficient valid data ({len(obs_clean)} points)")
                    continue

                # Calculate metrics
                kge_val = kge(obs_clean, sim_clean, transfo=1)
                kgep_val = kge_prime(obs_clean, sim_clean, transfo=1)
                nse_val = nse(obs_clean, sim_clean, transfo=1)

                # Calculate bias
                mean_obs = np.mean(obs_clean)
                mean_sim = np.mean(sim_clean)
                bias_pct = ((mean_sim - mean_obs) / mean_obs) * 100 if mean_obs != 0 else np.nan

                # Determine model name from column
                model_name = sim_col.replace('_discharge_cms', '').replace('_discharge', '')

                # Log metrics
                self.logger.info(f"  {model_name}:")
                self.logger.info(f"    KGE  = {kge_val:.4f}")
                self.logger.info(f"    KGE' = {kgep_val:.4f}")
                self.logger.info(f"    NSE  = {nse_val:.4f}")
                self.logger.info(f"    Bias = {bias_pct:+.1f}%")
                self.logger.info(f"    Valid data points: {len(obs_clean)}")

                # Provide interpretation
                if kge_val < 0:
                    self.logger.warning("    Note: KGE < 0 indicates model performs worse than mean observed flow")
                elif kge_val < 0.5:
                    self.logger.info("    Note: KGE < 0.5 suggests calibration may significantly improve results")
                elif kge_val >= 0.7:
                    self.logger.info("    Note: KGE >= 0.7 indicates reasonable baseline performance")

            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.debug(f"Could not calculate baseline metrics: {e}")

        

        

    def visualize_outputs(self):
        """
        Visualize model outputs using registered model visualizers.

        Invokes visualization functions for each primary model in the configuration.
        Visualizers are registered per-model and handle model-specific output formats.

        Note:
            Requires reporting_manager to be configured. Skips visualization if not available.
            Each model can register its own visualization function with the ModelRegistry.
        """
        self.logger.info('Starting model output visualisation')

        if not self.reporting_manager:
            self.logger.info("Visualization disabled or reporting manager not available.")
            return

        workflow = self._resolve_model_workflow()
        # Primary models from configuration
        models_str = self.config.model.hydrological_model or ''
        models = [m.strip() for m in str(models_str).split(',') if m.strip()]

        for model in models:
            visualizer = ModelRegistry.get_visualizer(model)
            if visualizer:
                try:
                    self.logger.info(f"Using registered visualizer for {model}")
                    visualizer(
                        self.reporting_manager,
                        self.config_dict,  # Visualizer expects flat dict
                        self.project_dir,
                        self.experiment_id,
                        workflow
                    )
                except Exception as e:
                    self.logger.error(f"Error during {model} visualization: {str(e)}")
            else:
                self.logger.info(f"Visualization for {model} not yet implemented or registered")

        
        
    