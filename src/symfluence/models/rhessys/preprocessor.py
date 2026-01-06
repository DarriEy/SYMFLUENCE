"""
RHESSys Model Preprocessor
"""
import logging
from pathlib import Path
from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("RHESSys")
class RHESSysPreprocessor(BaseModelPreProcessor):
    """
    Prepares inputs for a RHESSys model run.
    """

    def __init__(self, config, logger_instance):
        super().__init__(config, logger_instance)
        try:
            self.vmfire_enabled = self.config.model.rhessys.use_vmfire
        except AttributeError:
            self.vmfire_enabled = False

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "RHESSys"

    def run_preprocessing(self) -> bool:
        """
        Run the complete RHESSys preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting RHESSys preprocessing...")

            # Get the paths - use default if config not available
            try:
                forcing_dir = self.config.model.rhessys.input.forcing_dir
            except AttributeError:
                forcing_dir = "SUMMA_input"  # Default fallback
            
            self.input_dir = self.project_dir / forcing_dir
            self.input_dir.mkdir(parents=True, exist_ok=True)

            if self.vmfire_enabled:
                self._run_vmfire()

            self._generate_climate_files()
            self._generate_worldfile()
            self._generate_flow_table()

            logger.info("RHESSys preprocessing complete.")
            return True
        except Exception as e:
            logger.error(f"RHESSys preprocessing failed: {e}")
            return False

    def preprocess(self, **kwargs):
        """
        Main preprocessing routine for RHESSys.
        - Generates climate files
        - Generates worldfiles and flow tables (potentially using VMFire)
        """
        logger.info("Starting RHESSys preprocessing...")

        # Get the paths
        self.input_dir = self.project_dir / self.config.model.rhessys.input.forcing_dir
        self.input_dir.mkdir(parents=True, exist_ok=True)

        if self.vmfire_enabled:
            self._run_vmfire()

        self._generate_climate_files()
        self._generate_worldfile()
        self._generate_flow_table()

        logger.info("RHESSys preprocessing complete.")
        return self.input_dir

    def _run_vmfire(self):
        """
        Run the VMFire preprocessor to generate RHESSys inputs.
        """
        logger.info("VMFire is enabled. Running VMFire...")
        # Placeholder for VMFire logic
        pass

    def _generate_climate_files(self):
        """
        Generate RHESSys-compatible climate input files from forcing data.
        """
        logger.info("Generating climate files...")
        # Placeholder for climate file generation
        pass

    def _generate_worldfile(self):
        """
        Generate the RHESSys worldfile from templates and domain data.
        """
        logger.info("Generating worldfile...")
        # Placeholder for worldfile generation
        pass

    def _generate_flow_table(self):
        """
        Generate the RHESSys flow table.
        """
        logger.info("Generating flow table...")
        # Placeholder for flow table generation
        pass
