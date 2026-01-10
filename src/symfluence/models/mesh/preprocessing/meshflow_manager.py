"""
MESH Meshflow Manager

Handles meshflow execution with fallback strategies.
"""

import logging
from pathlib import Path
from typing import Dict, Any

try:
    from meshflow.core import MESHWorkflow
    MESHFLOW_AVAILABLE = True
except ImportError:
    MESHFLOW_AVAILABLE = False
    MESHWorkflow = None


class MESHFlowManager:
    """
    Manages meshflow execution for MESH preprocessing.

    Handles:
    - Full meshflow workflow execution
    - Fallback to DDB-only mode when full workflow fails
    - Parameter file generation via meshflow's render_configs
    """

    def __init__(
        self,
        forcing_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger = None
    ):
        """
        Initialize meshflow manager.

        Args:
            forcing_dir: Directory for MESH files
            config: Meshflow configuration dictionary
            logger: Optional logger instance
        """
        self.forcing_dir = forcing_dir
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def is_available() -> bool:
        """Check if meshflow is available."""
        return MESHFLOW_AVAILABLE

    def run(self, prepare_forcing_callback=None, postprocess_callback=None) -> None:
        """
        Run meshflow to generate MESH input files.

        Args:
            prepare_forcing_callback: Callback for direct forcing preparation
            postprocess_callback: Callback for post-processing output
        """
        if not MESHFLOW_AVAILABLE:
            from symfluence.core.exceptions import ModelExecutionError
            raise ModelExecutionError(
                "meshflow is not available. Install with: "
                "pip install git+https://github.com/CH-Earth/meshflow.git@main"
            )

        self._check_required_files()
        self._clean_output_files()

        try:
            import meshflow
            self.logger.info(f"Using meshflow version: {getattr(meshflow, '__version__', 'unknown')}")

            self.logger.info("Initializing MESHWorkflow with config")
            workflow = MESHWorkflow(**self.config)

            try:
                self.logger.info("Running full meshflow workflow")
                workflow.run(save_path=str(self.forcing_dir))
                workflow.save(output_dir=str(self.forcing_dir))
                self.logger.info("Full meshflow workflow completed successfully")

            except Exception as run_error:
                self.logger.warning(f"Full meshflow workflow failed ({run_error}), using fallback")
                self._run_fallback(workflow, prepare_forcing_callback)

            # Post-process
            if postprocess_callback:
                postprocess_callback()

            self.logger.info("meshflow preprocessing completed successfully")

        except Exception as e:
            self.logger.error(f"meshflow preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            from symfluence.core.exceptions import ModelExecutionError
            raise ModelExecutionError(f"meshflow preprocessing failed: {e}")

    def _check_required_files(self) -> None:
        """Check that required input files exist."""
        from symfluence.core.exceptions import ConfigurationError

        required_files = [self.config.get('riv'), self.config.get('cat')]
        missing_files = [f for f in required_files if f and not Path(f).exists()]

        if missing_files:
            raise ConfigurationError(
                f"MESH preprocessing requires these files: {missing_files}. "
                "Run geospatial preprocessing first."
            )

    def _clean_output_files(self) -> None:
        """Clean existing output files."""
        output_files = [
            self.forcing_dir / "MESH_forcing.nc",
            self.forcing_dir / "MESH_drainage_database.nc",
        ]
        for f in output_files:
            if f.exists():
                f.unlink()

    def _run_fallback(self, workflow, prepare_forcing_callback) -> None:
        """Run meshflow in fallback DDB-only mode."""
        self.logger.info("Running meshflow for drainage database only")
        workflow.init()
        workflow.init_ddb()

        ddb_path = self.forcing_dir / "MESH_drainage_database.nc"
        workflow.ddb.to_netcdf(ddb_path)
        self.logger.info(f"Created drainage database: {ddb_path}")

        # Try to generate parameter files
        self._generate_parameter_files(workflow)

        # Prepare forcing separately
        if prepare_forcing_callback:
            self.logger.info("Preparing forcing data directly")
            prepare_forcing_callback()

    def _generate_parameter_files(self, workflow) -> None:
        """Generate parameter files via meshflow's render_configs."""
        class_dict = self._try_init_class(workflow)
        hydro_dict = self._try_init_hydrology(workflow)
        options_dict = self._try_init_options(workflow)

        if class_dict and hydro_dict and options_dict:
            self._render_and_save_configs(workflow, class_dict, hydro_dict, options_dict)
        else:
            self.logger.warning("Could not generate all required dicts for render_configs")

    def _try_init_class(self, workflow):
        """Try to initialize CLASS parameters."""
        try:
            self.logger.info("Attempting to generate CLASS parameters via meshflow")
            class_dict = workflow.init_class(return_dict=True)
            self.logger.debug(f"CLASS dict keys: {list(class_dict.keys()) if class_dict else 'None'}")
            return class_dict
        except Exception as e:
            self.logger.warning(f"meshflow CLASS init failed: {e}")
            return None

    def _try_init_hydrology(self, workflow):
        """Try to initialize hydrology parameters."""
        try:
            self.logger.info("Attempting to generate hydrology parameters via meshflow")
            hydro_dict = workflow.init_hydrology(return_dict=True)
            self.logger.debug(f"Hydrology dict keys: {list(hydro_dict.keys()) if hydro_dict else 'None'}")
            return hydro_dict
        except Exception as e:
            self.logger.warning(f"meshflow hydrology init failed: {e}")
            return None

    def _try_init_options(self, workflow):
        """Try to initialize run options."""
        try:
            self.logger.info("Attempting to generate run options via meshflow")
            options_dict = workflow.init_options(return_dict=True)
            self.logger.debug(f"Options dict keys: {list(options_dict.keys()) if options_dict else 'None'}")
            return options_dict
        except Exception as e:
            self.logger.warning(f"meshflow options init failed: {e}")
            return None

    def _render_and_save_configs(self, workflow, class_dict, hydro_dict, options_dict) -> None:
        """Render and save configuration files."""
        try:
            self.logger.info("Rendering meshflow configs to text")

            process_details = {
                'routing': ['r2n', 'r1n', 'pwr', 'flz'],
                'hydrology': [],
            }

            workflow.render_configs(
                class_dicts=class_dict,
                hydrology_dicts=hydro_dict,
                options_dict=options_dict,
                process_details=process_details
            )

            if hasattr(workflow, 'class_text') and workflow.class_text:
                class_path = self.forcing_dir / "MESH_parameters_CLASS.ini"
                with open(class_path, 'w') as f:
                    f.write(workflow.class_text)
                self.logger.info(f"Created CLASS parameters via meshflow: {class_path}")

            if hasattr(workflow, 'hydrology_text') and workflow.hydrology_text:
                hydro_path = self.forcing_dir / "MESH_parameters_hydrology.ini"
                with open(hydro_path, 'w') as f:
                    f.write(workflow.hydrology_text)
                self.logger.info(f"Created hydrology parameters via meshflow: {hydro_path}")

            if hasattr(workflow, 'options_text') and workflow.options_text:
                run_path = self.forcing_dir / "MESH_input_run_options.ini"
                with open(run_path, 'w') as f:
                    f.write(workflow.options_text)
                self.logger.info(f"Created run options via meshflow: {run_path}")

        except Exception as e:
            self.logger.warning(f"meshflow render_configs failed: {e}")
