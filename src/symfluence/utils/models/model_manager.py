# src/symfluence/utils/models/model_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Union, List, Optional
import pandas as pd

# Registry
from symfluence.utils.models.registry import ModelRegistry

# Data management
from symfluence.utils.data.utilities.archive_utils import tar_directory # type: ignore

# Import for type checking only (avoid circular imports)
try:
    from symfluence.utils.config.models_v2 import SymfluenceConfig
except ImportError:
    SymfluenceConfig = None

class ModelManager:
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
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
    def _resolve_model_workflow(self) -> List[str]:
        """
        Resolve the order of models to run, including implicit dependencies.
        
        Returns:
            List of model names to execute in order.
        """
        configured_models = [m.strip() for m in self.config.get('HYDROLOGICAL_MODEL', '').split(',') if m.strip()]
        execution_list = []

        for model in configured_models:
            if model not in execution_list:
                execution_list.append(model)

            # check implicit dependencies (e.g. mizuRoute)
            if model == 'SUMMA':
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                
                needs_mizuroute = False
                if domain_method not in ['point', 'lumped']:
                    needs_mizuroute = True
                elif domain_method == 'lumped' and routing_delineation == 'river_network':
                    needs_mizuroute = True
                
                if needs_mizuroute:
                    self._ensure_mizuroute_in_workflow(execution_list, source_model='SUMMA')

            elif model == 'FUSE':
                fuse_routing = self.config.get('FUSE_ROUTING_INTEGRATION', 'none')
                fuse_spatial = self.config.get('FUSE_SPATIAL_MODE', 'lumped')

                needs_mizuroute = False
                if fuse_routing == 'mizuRoute':
                    needs_mizuroute = True
                elif fuse_spatial in ['semi_distributed', 'distributed']:
                    needs_mizuroute = True # Often implies routing needed
                elif fuse_spatial == 'lumped' and self.config.get('ROUTING_DELINEATION') == 'river_network':
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
        if not self.config.get('MIZU_FROM_MODEL'):
            self.config['MIZU_FROM_MODEL'] = source_model
            self.logger.info(f"Setting MIZU_FROM_MODEL to {source_model}")

    def preprocess_models(self, params: Optional[Dict[str, Any]] = None):
        """
        Process the forcing data into model-specific formats.

        Args:
            params: Optional dictionary of parameter values (for calibration)
        """
        self.logger.info("Starting model-specific preprocessing")

        workflow = self._resolve_model_workflow()
        self.logger.info(f"Preprocessing workflow order: {workflow}")

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
                        self.logger.info(f"Model {model} doesn't require preprocessing")
                    else:
                        # Only warn if it's a primary model, not a utility like MIZUROUTE which definitely has one
                        self.logger.debug(f"No preprocessor registered for {model} (or not required).")
                    continue

                # Run model-specific preprocessing
                self.logger.info(f"Running preprocessor for {model}")
                
                # Check if preprocessor accepts params
                import inspect
                sig = inspect.signature(preprocessor_class.__init__)
                if 'params' in sig.parameters:
                    preprocessor = preprocessor_class(self.config, self.logger, params=params)
                else:
                    preprocessor = preprocessor_class(self.config, self.logger)
                    
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
        """Post-process model results using the registry."""
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
                if hasattr(postprocessor, 'extract_streamflow'):
                    postprocessor.extract_streamflow()
                elif hasattr(postprocessor, 'extract_results'):
                    postprocessor.extract_results()
                else:
                    self.logger.warning(f"No extraction method found for {model} postprocessor")
                    continue
                
            except Exception as e:
                self.logger.error(f"Error post-processing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Create final visualizations
        try:
            if self.reporting_manager:
                self.reporting_manager.visualize_timeseries_results()
                self.logger.info("Time series visualizations created")
        except Exception as e:
            self.logger.error(f"Error creating time series visualizations: {str(e)}")

    def visualize_outputs(self):
        """Visualize model outputs."""
        self.logger.info('Starting model output visualisation')
        
        # Note: Visualization still has significant model-specific logic
        # that ideally should be moved to a 'Reporter' or 'Visualizer' interface.
        # For now, we adapt it to check the *executed* workflow.
        
        if not self.reporting_manager:
            self.logger.info("Visualization disabled or reporting manager not available.")
            return

        workflow = self._resolve_model_workflow()
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',') # Primary models

        for model in [m.strip() for m in models]:
            if model == 'SUMMA':
                self.reporting_manager.visualize_summa_outputs(self.experiment_id)
                obs_files = [('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"))]

                # Check if MizuRoute was part of the workflow
                if 'MIZUROUTE' in workflow and self.config.get('MIZU_FROM_MODEL') == 'SUMMA':
                    self.reporting_manager.update_sim_reach_id()
                    model_outputs = [(model, str(self.project_dir / "simulations" / self.experiment_id / "mizuRoute" / f"{self.experiment_id}*.nc"))]
                    self.reporting_manager.visualize_model_outputs(model_outputs, obs_files)
                else:
                    summa_output_file = str(self.project_dir / "simulations" / self.experiment_id / "SUMMA" / f"{self.experiment_id}_timestep.nc")
                    model_outputs = [(model, summa_output_file)]
                    self.reporting_manager.visualize_lumped_model_outputs(model_outputs, obs_files)
            
            elif model == 'FUSE':
                model_outputs = [("FUSE", str(self.project_dir / "simulations" / self.experiment_id / "FUSE" / f"{self.domain_name}_{self.experiment_id}_runs_best.nc"))]
                obs_files = [('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"))]
                self.reporting_manager.visualize_fuse_outputs(model_outputs, obs_files)
            
            else:
                self.logger.info(f"Visualization for {model} not yet implemented")