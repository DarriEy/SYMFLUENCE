"""
Workflow orchestration for SYMFLUENCE hydrological modeling pipeline.

Coordinates the execution sequence of modeling steps including domain definition,
data preprocessing, model execution, optimization, and analysis phases.
"""

from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass


@dataclass
class WorkflowStep:
    """
    Represents a single step in the SYMFLUENCE workflow.
    """
    name: str
    cli_name: str
    func: Callable
    check_func: Callable
    description: str


class WorkflowOrchestrator:
    """
    Orchestrates the SYMFLUENCE workflow execution and manages the step sequence.
    
    The WorkflowOrchestrator is responsible for defining, coordinating, and executing
    the complete SYMFLUENCE modeling workflow. It integrates the various manager 
    components into a coherent sequence of operations, handling dependencies between
    steps, tracking progress, and providing status information.
    
    Key responsibilities:
    - Defining the sequence of workflow steps and their validation checks
    - Coordinating execution across different manager components
    - Handling execution flow (skipping completed steps, stopping on errors)
    - Providing status information and execution reports
    - Validating prerequisites before workflow execution
    
    This class represents the "conductor" of the SYMFLUENCE system, ensuring that
    each component performs its tasks in the correct order and with the necessary
    inputs from previous steps.
    
    Attributes:
        managers (Dict[str, Any]): Dictionary of manager instances
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
        domain_name (str): Name of the hydrological domain
        experiment_id (str): ID of the current experiment
        project_dir (Path): Path to the project directory
        logging_manager: Reference to logging manager for enhanced formatting
    """
    
    def __init__(self, managers: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger, logging_manager=None):
        """
        Initialize the workflow orchestrator.
        
        Sets up the orchestrator with references to all manager components, the 
        configuration, and the logger. This creates the central coordination point
        for the entire SYMFLUENCE workflow.
        
        Args:
            managers (Dict[str, Any]): Dictionary of manager instances for each 
                                      functional area (project, domain, data, etc.)
            config (Dict[str, Any]): Configuration dictionary with all settings
            logger (logging.Logger): Logger instance for recording operations
            logging_manager: Reference to LoggingManager for enhanced formatting
            
        Raises:
            KeyError: If essential configuration values are missing
        """
        self.managers = managers
        self.config = config
        self.logger = logger
        self.logging_manager = logging_manager
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        data_dir = config.get('SYMFLUENCE_DATA_DIR')
        if not data_dir:
            raise KeyError("SYMFLUENCE_DATA_DIR not found in config")

        self.project_dir = Path(data_dir) / f"domain_{self.domain_name}"
    
    def _check_observed_data_exists(self) -> bool:
        """
        Check if required observed data files exist based on configuration.

        Checks for any of the following based on config:
        - Streamflow data (if EVALUATION_DATA or ADDITIONAL_OBSERVATIONS includes streamflow-like sources)
        - Snow data (SWE, SCA if EVALUATION_DATA includes SWE/SCA or DOWNLOAD_MODIS_SNOW/DOWNLOAD_SNOTEL)
        - Soil moisture data (if EVALUATION_DATA includes SM_ISMN, SM_SMAP, etc.)
        - ET data (if EVALUATION_DATA includes ET)

        Returns:
            bool: True if at least one required observation type has been processed
        """
        evaluation_data = self.config.get('EVALUATION_DATA', [])
        additional_obs = self.config.get('ADDITIONAL_OBSERVATIONS', [])

        # Check for snow data (SWE, SCA)
        if any(obs_type.upper() in ['SWE', 'SCA', 'SNOW'] for obs_type in evaluation_data) or \
           self.config.get('DOWNLOAD_SNOTEL') or self.config.get('DOWNLOAD_MODIS_SNOW'):
            snow_file = self.project_dir / "observations" / "snow" / "preprocessed" / f"{self.domain_name}_snow_processed.csv"
            if snow_file.exists():
                return True

        # Check for soil moisture data
        if any('SM_' in str(obs_type).upper() for obs_type in evaluation_data):
            sm_files = [
                self.project_dir / "observations" / "soil_moisture" / "point" / "processed" / f"{self.domain_name}_sm_processed.csv",
                self.project_dir / "observations" / "soil_moisture" / "smap" / "processed" / f"{self.domain_name}_smap_processed.csv",
                self.project_dir / "observations" / "soil_moisture" / "ismn" / "processed" / f"{self.domain_name}_ismn_processed.csv",
            ]
            if any(f.exists() for f in sm_files):
                return True

        # Check for streamflow data (default)
        if any(obs_type.upper() in ['STREAMFLOW', 'DISCHARGE'] for obs_type in evaluation_data) or \
           self.config.get('DOWNLOAD_USGS_DATA') or self.config.get('DOWNLOAD_WSC_DATA'):
            streamflow_file = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
            if streamflow_file.exists():
                return True

        # Check for ET data
        if any('ET' in str(obs_type).upper() for obs_type in evaluation_data):
            et_files = [
                self.project_dir / "observations" / "et" / "preprocessed" / f"{self.domain_name}_modis_et_processed.csv",
                self.project_dir / "observations" / "et" / "preprocessed" / f"{self.domain_name}_fluxnet_et_processed.csv",
            ]
            if any(f.exists() for f in et_files):
                return True

        # If no specific evaluation data is defined, check for streamflow as default
        if not evaluation_data:
            return (self.project_dir / "observations" / "streamflow" / "preprocessed" /
                    f"{self.domain_name}_streamflow_processed.csv").exists()

        # If we got here and evaluation_data is defined but nothing found, return False
        return False

    def define_workflow_steps(self) -> List[WorkflowStep]:
        """
        Define the workflow steps with their output validation checks and descriptions.

        Returns:
            List[WorkflowStep]: List of WorkflowStep objects
        """

        # Get configured analyses
        analyses = self.config.get('ANALYSES', [])
        optimizations = self.config.get('OPTIMIZATION_METHODS', []) 

        return [
            # --- Project Initialization ---
            WorkflowStep(
                name="setup_project",
                cli_name="setup_project",
                func=self.managers['project'].setup_project,
                check_func=lambda: (self.project_dir / 'shapefiles').exists(),
                description="Setting up project structure and directories"
            ),
            
            # --- Geospatial Domain Definition and Analysis ---
            WorkflowStep(
                name="create_pour_point",
                cli_name="create_pour_point",
                func=self.managers['project'].create_pour_point,
                check_func=lambda: (self.project_dir / "shapefiles" / "pour_point" / 
                        f"{self.domain_name}_pourPoint.shp").exists(),
                description="Creating watershed pour point"
            ),
            WorkflowStep(
                name="acquire_attributes",
                cli_name="acquire_attributes",
                func=self.managers['data'].acquire_attributes,
                check_func=lambda: (self.project_dir / "attributes" / "soilclass" / 
                        f"domain_{self.domain_name}_soil_classes.tif").exists(),
                description="Acquiring geospatial attributes and data"
            ),
            WorkflowStep(
                name="define_domain",
                cli_name="define_domain",
                func=self.managers['domain'].define_domain,
                check_func=lambda: (self.project_dir / "shapefiles" / "river_basins" / 
                        f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp").exists(),
                description="Defining hydrological domain boundaries"
            ),
            WorkflowStep(
                name="discretize_domain",
                cli_name="discretize_domain",
                func=self.managers['domain'].discretize_domain,
                check_func=lambda: (self.project_dir / "shapefiles" / "catchment" / 
                        f"{self.domain_name}_HRUs_{str(self.config.get('DOMAIN_DISCRETIZATION')).replace(',','_')}.shp").exists(),
                description="Discretizing domain into hydrological response units"
            ),
            
            # --- Model-Agnostic Data Preprocessing ---
            WorkflowStep(
                name="process_observed_data",
                cli_name="process_observed_data",
                func=self.managers['data'].process_observed_data,
                check_func=self._check_observed_data_exists,
                description="Processing observed data"
            ),
            WorkflowStep(
                name="acquire_forcings",
                cli_name="acquire_forcings",
                func=self.managers['data'].acquire_forcings,
                check_func=lambda: (self.project_dir / "forcing" / "raw_data").exists(),
                description="Acquiring meteorological forcing data"
            ),
            WorkflowStep(
                name="run_model_agnostic_preprocessing",
                cli_name="model_agnostic_preprocessing",
                func=self.managers['data'].run_model_agnostic_preprocessing,
                check_func=lambda: (self.project_dir / "forcing" / "basin_averaged_data").exists(),
                description="Running model-agnostic data preprocessing"
            ),
            # --- Model-Specific Preprocessing and Execution ---
            WorkflowStep(
                name="preprocess_models",
                cli_name="model_specific_preprocessing",
                func=self.managers['model'].preprocess_models,
                check_func=lambda: any((self.project_dir / "settings").glob(f"*_{self.config.get('HYDROLOGICAL_MODEL', 'SUMMA')}*")),
                description="Preprocessing model-specific input files"
            ),
            WorkflowStep(
                name="run_models",
                cli_name="run_model",
                func=self.managers['model'].run_models,
                check_func=lambda: (self.project_dir / "simulations" / 
                        f"{self.experiment_id}_{self.config.get('HYDROLOGICAL_MODEL', 'SUMMA')}_output.nc").exists(),
                description="Running hydrological model simulation"
            ),
            WorkflowStep(
                name="postprocess_results",
                cli_name="postprocess_results",
                func=self.managers['model'].postprocess_results,
                check_func=lambda: (self.project_dir / "simulations" / 
                        f"{self.experiment_id}_postprocessed.nc").exists(),
                description="Post-processing simulation results"
            ),
            
            # --- Optimization and Emulation Steps ---
            WorkflowStep(
                name="calibrate_model",
                cli_name="calibrate_model",
                func=self.managers['optimization'].calibrate_model,
                check_func=lambda: ('optimization' in optimizations and 
                        (self.project_dir / "optimization" / 
                        f"{self.experiment_id}_parallel_iteration_results.csv").exists()),
                description="Calibrating model parameters"
            ),
            
            # --- Analysis Steps ---
            WorkflowStep(
                name="run_benchmarking",
                cli_name="run_benchmarking",
                func=self.managers['analysis'].run_benchmarking,
                check_func=lambda: ('benchmarking' in analyses and
                        (self.project_dir / "evaluation" / "benchmark_scores.csv").exists()),
                description="Running model benchmarking analysis"
            ),
            
            WorkflowStep(
                name="run_decision_analysis",
                cli_name="run_decision_analysis",
                func=self.managers['analysis'].run_decision_analysis,
                check_func=lambda: ('decision' in analyses and
                        (self.project_dir / "optimization" / 
                        f"{self.experiment_id}_model_decisions_comparison.csv").exists()),
                description="Analyzing modeling decisions impact"
            ),
            
            WorkflowStep(
                name="run_sensitivity_analysis",
                cli_name="run_sensitivity_analysis",
                func=self.managers['analysis'].run_sensitivity_analysis,
                check_func=lambda: ('sensitivity' in analyses and
                        (self.project_dir / "reporting" / "sensitivity_analysis" / 
                        "all_sensitivity_results.csv").exists()),
                description="Running parameter sensitivity analysis"
            ),
            
        ]
    
    def run_workflow(self, force_run: bool = False):
        """
        Run the complete workflow according to the defined steps.
        
        This method executes each step in the workflow sequence, handling:
        - Conditional execution based on existing outputs
        - Error handling with configurable stop-on-error behavior
        - Progress tracking and timing information
        - Comprehensive logging of each operation
        
        The workflow can be configured to:
        - Skip steps that have already been completed (default)
        - Force re-execution of all steps (force_run=True)
        - Continue or stop on errors (based on STOP_ON_ERROR config)
        
        Args:
            force_run (bool): If True, forces execution of all steps even if outputs exist.
                            If False (default), skips steps with existing outputs.
        
        Raises:
            Exception: If a step fails and STOP_ON_ERROR is True in configuration
            
        Note:
            The method provides detailed logging throughout execution, including:
            - Step headers with progress indicators
            - Execution timing for each step
            - Clear success/skip/failure indicators
            - Final summary statistics
        """
        # Check prerequisites
        if not self.validate_workflow_prerequisites():
            raise ValueError("Workflow prerequisites not met")
        
        # Log workflow start
        start_time = datetime.now()
        
        # FIXED: Use direct logging instead of non-existent format_section_header()
        self.logger.info("=" * 60)
        self.logger.info("SYMFLUENCE WORKFLOW EXECUTION")
        self.logger.info(f"Domain: {self.domain_name}")
        self.logger.info(f"Experiment: {self.experiment_id}")
        self.logger.info("=" * 60)
        
        # Get workflow steps
        workflow_steps = self.define_workflow_steps()
        total_steps = len(workflow_steps)
        completed_steps = 0
        skipped_steps = 0
        failed_steps = 0
        
        # Execute each step
        for idx, step in enumerate(workflow_steps, 1):
            step_name = step.name
            
            # FIXED: Use log_step_header() instead of non-existent format_step_header()
            if self.logging_manager:
                self.logging_manager.log_step_header(idx, total_steps, step_name, step.description)
            else:
                self.logger.info(f"\nStep {idx}/{total_steps}: {step_name}")
                self.logger.info(f"{step.description}")
                self.logger.info("=" * 40)
            
            try:
                if force_run or not step.check_func():
                    step_start_time = datetime.now()
                    self.logger.info(f"Executing: {step.description}")
                    
                    step.func()
                    
                    step_end_time = datetime.now()
                    duration = (step_end_time - step_start_time).total_seconds()
                    
                    # FIXED: Use log_completion() instead of non-existent format_step_completion()
                    if self.logging_manager:
                        self.logging_manager.log_completion(
                            success=True, 
                            message=step.description, 
                            duration=duration
                        )
                    else:
                        self.logger.info(f"✓ Completed: {step_name} (Duration: {duration:.2f}s)")
                    
                    completed_steps += 1
                else:
                    # Log skip
                    if self.logging_manager:
                        self.logging_manager.log_substep(f"Skipping: {step.description} (Output already exists)")
                    else:
                        self.logger.info(f"→ Skipping: {step_name} (Output already exists)")
                    
                    skipped_steps += 1
                    
            except Exception as e:
                # Log failure
                if self.logging_manager:
                    self.logging_manager.log_completion(
                        success=False,
                        message=f"{step.description}: {str(e)}"
                    )
                else:
                    self.logger.error(f"✗ Failed: {step_name}")
                    self.logger.error(f"Error: {str(e)}")
                
                failed_steps += 1
                
                # Decide whether to continue or stop
                if self.config.get('STOP_ON_ERROR', True):
                    self.logger.error("Workflow stopped due to error (STOP_ON_ERROR=True)")
                    raise
                else:
                    self.logger.warning("Continuing despite error (STOP_ON_ERROR=False)")
        
        # Summary report
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        # FIXED: Use direct logging instead of non-existent format_section_header()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("WORKFLOW SUMMARY")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Total execution time: {total_duration}")
        self.logger.info(f"Steps completed: {completed_steps}/{total_steps}")
        self.logger.info(f"Steps skipped: {skipped_steps}")
        
        if failed_steps > 0:
            self.logger.warning(f"Steps failed: {failed_steps}")
            self.logger.warning("Workflow completed with errors")
        else:
            self.logger.info("✓ Workflow completed successfully")
        
        self.logger.info("═" * 60)
    
    def validate_workflow_prerequisites(self) -> bool:
        """
        Validate that all prerequisites are met before running the workflow.
        
        This method performs a series of checks to ensure that the workflow can be
        executed successfully:
        1. Verifies that all required configuration parameters are present
        2. Confirms that all manager components have been properly initialized
        
        These validations help prevent runtime errors by catching configuration or
        initialization issues before workflow execution begins.
        
        Returns:
            bool: True if all prerequisites are met, False otherwise
            
        Note:
            This method logs detailed information about any validation failures,
            making it useful for diagnosing configuration problems.
        """
        valid = True
        
        # Check configuration validity (support both old and new config keys)
        required_config = [
            'DOMAIN_NAME',
            'EXPERIMENT_ID',
            'HYDROLOGICAL_MODEL',
            'DOMAIN_DEFINITION_METHOD',
            'DOMAIN_DISCRETIZATION'
        ]
        
        for key in required_config:
            if not self.config.get(key):
                self.logger.error(f"Required configuration missing: {key}")
                valid = False
        
        # Check for data directory
        if not self.config.get('SYMFLUENCE_DATA_DIR'):
            self.logger.error("Required configuration missing: SYMFLUENCE_DATA_DIR")
            valid = False
        
        # Check manager initialization
        required_managers = ['project', 'domain', 'data', 'model', 'analysis', 'optimization']
        for manager_name in required_managers:
            if manager_name not in self.managers:
                self.logger.error(f"Required manager not initialized: {manager_name}")
                valid = False
        
        return valid
    
    def run_individual_steps(self, step_names: List[str], continue_on_error: bool = False) -> List[Dict[str, Any]]:
        """
        Execute a specific list of workflow steps by their CLI names.
        
        Args:
            step_names: List of step CLI names to execute
            continue_on_error: Whether to continue to next step if one fails
            
        Returns:
            List of dictionaries containing execution results for each step
        """
        # Resolve workflow steps from orchestrator
        workflow_steps = self.define_workflow_steps()
        cli_to_step = {step.cli_name: step for step in workflow_steps}
        
        results = []
        
        self.logger.info(f"Starting individual step execution: {', '.join(step_names)}")
        
        for idx, cli_name in enumerate(step_names, 1):
            step = cli_to_step.get(cli_name)
            if not step:
                self.logger.warning(f"Step '{cli_name}' not recognized; skipping")
                continue

            # Log step header
            if self.logging_manager:
                self.logging_manager.log_step_header(idx, len(step_names), step.name, step.description)
            else:
                self.logger.info(f"\nExecuting step: {cli_name} -> {step.name}")

            step_start_time = datetime.now()
            
            try:
                # Force execution; skip completion checks for individual steps
                step.func()
                
                duration = (datetime.now() - step_start_time).total_seconds()
                
                if self.logging_manager:
                    self.logging_manager.log_completion(True, step.description, duration)
                else:
                    self.logger.info(f"✓ Completed step: {cli_name}")
                    
                results.append({"cli": cli_name, "fn": step.name, "success": True, "duration": duration})
                
            except Exception as e:
                self.logger.error(f"Step '{cli_name}' failed: {e}")
                
                if self.logging_manager:
                    self.logging_manager.log_completion(False, f"{step.description}: {str(e)}")
                
                results.append({"cli": cli_name, "fn": step.name, "success": False, "error": str(e)})
                
                if not continue_on_error:
                    raise
                    
        return results

    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get the current status of the workflow execution.
        
        This method examines each step in the workflow to determine whether it has
        been completed, using the same output validation checks used during execution.
        It provides a comprehensive view of workflow progress, including which steps
        are complete and which are pending.
        
        The status information is useful for:
        - Monitoring long-running workflows
        - Generating progress reports
        - Diagnosing execution issues
        - Providing feedback to users
        
        Returns:
            Dict[str, Any]: Dictionary containing workflow status information, including:
                - total_steps: Total number of workflow steps
                - completed_steps: Number of completed steps
                - pending_steps: Number of pending steps
                - step_details: List of dictionaries with details for each step
                  (name and completion status)
        """
        workflow_steps = self.define_workflow_steps()
        
        status = {
            'total_steps': len(workflow_steps),
            'completed_steps': 0,
            'pending_steps': 0,
            'step_details': []
        }
        
        for step in workflow_steps:
            step_name = step.name
            is_complete = step.check_func()
            
            if is_complete:
                status['completed_steps'] += 1
            else:
                status['pending_steps'] += 1
            
            status['step_details'].append({
                'name': step_name,
                'cli_name': step.cli_name,
                'description': step.description,
                'complete': is_complete
            })
        
        return status
