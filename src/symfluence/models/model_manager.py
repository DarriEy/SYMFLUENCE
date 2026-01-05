# src/symfluence.models/model_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Union, List, Optional
import pandas as pd

# Registry
from symfluence.models.registry import ModelRegistry

# Data management
from symfluence.data.utilities.archive_utils import tar_directory # type: ignore

from symfluence.core.mixins import ConfigurableMixin

# Import for type checking only (avoid circular imports)
try:
    from symfluence.core.config.models import SymfluenceConfig
except ImportError:
    SymfluenceConfig = None

class ModelManager(ConfigurableMixin):
    """
    Manages all hydrological model operations within the SYMFLUENCE framework.
    Uses a registry-based system for easy extension with new models.
    """
    
    def __init__(self, config: Union[Dict[str, Any], 'SymfluenceConfig'], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the Model Manager.

        Args:
            config: Configuration dictionary or SymfluenceConfig instance (Phase 2)
            logger: Logger instance
            reporting_manager: ReportingManager instance
        """
        # Phase 2: Support both typed config and dict config for backward compatibility
        if SymfluenceConfig and isinstance(config, SymfluenceConfig):
            self.typed_config = config
            self.config = config.to_dict(flatten=True)
        else:
            self.typed_config = None
            self.config = config

        self.logger = logger
        self.reporting_manager = reporting_manager
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
    def _resolve_model_workflow(self) -> List[str]:

        """

        Resolve the order of models to run, including implicit dependencies.

        

        Returns:

            List of model names to execute in order.

        """

        models_str = self._resolve_config_value(

            lambda: self.typed_config.model.hydrological_model,

            'HYDROLOGICAL_MODEL',

            ''

        )

        configured_models = [m.strip() for m in str(models_str).split(',') if m.strip()]

        execution_list = []

        

        for model in configured_models:

            if model not in execution_list:

                execution_list.append(model)

        

            # check implicit dependencies (e.g. mizuRoute)

            if model == 'SUMMA':

                domain_method = self._resolve_config_value(

                    lambda: self.typed_config.domain.definition_method,

                    'DOMAIN_DEFINITION_METHOD',

                    'lumped'

                )

                routing_delineation = self._resolve_config_value(

                    lambda: self.typed_config.routing.delineation,

                    'ROUTING_DELINEATION',

                    'lumped'

                )

                

                needs_mizuroute = False

                if domain_method not in ['point', 'lumped']:

                    needs_mizuroute = True

                elif domain_method == 'lumped' and routing_delineation == 'river_network':

                    needs_mizuroute = True

                

                if needs_mizuroute:

                    self._ensure_mizuroute_in_workflow(execution_list, source_model='SUMMA')

        

            elif model == 'FUSE':

                fuse_routing = self._resolve_config_value(

                    lambda: self.typed_config.model.fuse.routing_integration if self.typed_config.model.fuse else None,

                    'FUSE_ROUTING_INTEGRATION',

                    'none'

                )

                fuse_spatial = self._resolve_config_value(

                    lambda: self.typed_config.model.fuse.spatial_mode if self.typed_config.model.fuse else None,

                    'FUSE_SPATIAL_MODE',

                    'lumped'

                )

                routing_delineation = self._resolve_config_value(

                    lambda: self.typed_config.routing.delineation,

                    'ROUTING_DELINEATION',

                    'lumped'

                )

        

                needs_mizuroute = False

                if fuse_routing == 'mizuRoute':

                    needs_mizuroute = True

                elif fuse_spatial in ['semi_distributed', 'distributed']:

                    needs_mizuroute = True # Often implies routing needed

                elif fuse_spatial == 'lumped' and routing_delineation == 'river_network':

                    needs_mizuroute = True

        

                if needs_mizuroute:

                    self._ensure_mizuroute_in_workflow(execution_list, source_model='FUSE')

        

        return execution_list

        

    def _ensure_mizuroute_in_workflow(self, execution_list: List[str], source_model: str):
        """Helper to add mizuRoute to workflow and set context."""
        if 'MIZUROUTE' not in execution_list:
            execution_list.append('MIZUROUTE')
            self.logger.info(f"Automatically adding MIZUROUTE to workflow (dependency of {source_model})")
            
        # Ensure configuration knows the source model if not explicitly set
        mizu_from = self._resolve_config_value(
            lambda: self.typed_config.routing.mizu_from_model,
            'MIZU_FROM_MODEL'
        )
        if not mizu_from:
            # Fallback to dict update for internal tracking
            self.config['MIZU_FROM_MODEL'] = source_model
            self.logger.info(f"Setting MIZU_FROM_MODEL to {source_model}")

    def preprocess_models(self, params: Optional[Dict[str, Any]] = None):
        """
        Process the forcing data into model-specific formats.

        Args:
            params: Optional dictionary of parameter values (for calibration)
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

                

                # Use typed config if available to avoid deprecation warnings

                component_config = self.typed_config if self.typed_config else self.config

                

                # Check if preprocessor accepts params

                import inspect

                sig = inspect.signature(preprocessor_class.__init__)

                if 'params' in sig.parameters:

                    preprocessor = preprocessor_class(component_config, self.logger, params=params)

                else:

                    preprocessor = preprocessor_class(component_config, self.logger)

                    

                preprocessor.run_preprocessing()

        

            except Exception as e:

                self.logger.error(f"Error preprocessing model {model}: {str(e)}")

                import traceback

                self.logger.error(traceback.format_exc())

                raise

        

        self.logger.info("Model-specific preprocessing completed")

        

    def run_models(self):
        """Execute model runs using the registry."""
        self.logger.info("Starting model runs")

        workflow = self._resolve_model_workflow()
        self.logger.info(f"Execution workflow order: {workflow}")

        # Use typed config if available to avoid deprecation warnings
        component_config = self.typed_config if self.typed_config else self.config

        for model in workflow:
            try:
                self.logger.info(f"Running model: {model}")
                runner_class = ModelRegistry.get_runner(model)
                if runner_class is None:
                    self.logger.error(f"Unknown hydrological model or no runner registered: {model}")
                    continue

                runner = runner_class(component_config, self.logger, reporting_manager=self.reporting_manager)
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
        """Post-process model results using the registry."""
        self.logger.info("Starting model post-processing")

        workflow = self._resolve_model_workflow()

        # Use typed config if available to avoid deprecation warnings
        component_config = self.typed_config if self.typed_config else self.config

        for model in workflow:
            try:
                # Get postprocessor class from registry
                postprocessor_class = ModelRegistry.get_postprocessor(model)

                if postprocessor_class is None:
                    continue

                self.logger.info(f"Post-processing {model}")
                # Create postprocessor instance
                postprocessor = postprocessor_class(component_config, self.logger, reporting_manager=self.reporting_manager)

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

        

        

    def visualize_outputs(self):
        """Visualize model outputs using registered model visualizers."""
        self.logger.info('Starting model output visualisation')

        if not self.reporting_manager:
            self.logger.info("Visualization disabled or reporting manager not available.")
            return

        workflow = self._resolve_model_workflow()
        # Primary models from configuration
        models_str = self._resolve_config_value(
            lambda: self.typed_config.model.hydrological_model,
            'HYDROLOGICAL_MODEL',
            ''
        )
        models = [m.strip() for m in str(models_str).split(',') if m.strip()]

        for model in models:
            visualizer = ModelRegistry.get_visualizer(model)
            if visualizer:
                try:
                    self.logger.info(f"Using registered visualizer for {model}")
                    visualizer(
                        self.reporting_manager, 
                        self.config, 
                        self.project_dir, 
                        self.experiment_id, 
                        workflow
                    )
                except Exception as e:
                    self.logger.error(f"Error during {model} visualization: {str(e)}")
            else:
                self.logger.info(f"Visualization for {model} not yet implemented or registered")

        
        
    