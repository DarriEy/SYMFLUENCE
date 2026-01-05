"""
RHESSys Model Runner
"""
import logging
from pathlib import Path
from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner("RHESSys")
class RHESSysRunner(BaseModelRunner):
    """
    Runs the RHESSys model.
    """

    def __init__(self, config, project_dir, experiment_id):
        super().__init__(config, project_dir, experiment_id)
        self.model_name = "RHESSys"

    def run(self, **kwargs):
        """
        Execute the RHESSys model.
        """
        logger.info(f"Running {self.model_name} for experiment {self.experiment_id}...")

        # Setup paths
        self.output_dir = self.project_dir / self.config.model.rhessys.output.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        rhessys_exe = self._get_executable_path(
            self.config.model.rhessys.installation.install_path,
            self.config.model.rhessys.installation.exe_name
        )

        # Build command
        cmd = self._build_command()

        # Execute command
        self._run_executable(cmd, self.output_dir)

        logger.info(f"{self.model_name} run complete.")
        return self.output_dir

    def postprocess(self, **kwargs):
        """
        Postprocess RHESSys output files into a standard format.
        """
        logger.info(f"Postprocessing {self.model_name} outputs...")
        # Placeholder for postprocessing logic
        pass

    def _build_command(self):
        """
        Construct the command to run RHESSys.
        """
        logger.info("Building RHESSys command...")
        # Placeholder for command building logic
        return ["echo", "RHESSys command placeholder"]

