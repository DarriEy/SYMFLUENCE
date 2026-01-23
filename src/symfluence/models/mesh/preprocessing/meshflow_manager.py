"""
MESH Meshflow Manager

Handles meshflow execution for MESH preprocessing.
Meshflow is the single required pathway - no fallbacks.
"""

import logging
import traceback
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

    Meshflow is the required preprocessing pathway. If meshflow fails,
    preprocessing fails - there are no fallback strategies.
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

        Raises:
            ModelExecutionError: If meshflow is not available or fails.
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

            self.logger.info("Running meshflow workflow")
            workflow.run(save_path=str(self.forcing_dir))
            workflow.save(output_dir=str(self.forcing_dir))
            self.logger.info("Meshflow workflow completed successfully")

            # Post-process
            if postprocess_callback:
                postprocess_callback()

            self.logger.info("Meshflow preprocessing completed successfully")

        except Exception as e:
            self.logger.error(f"Meshflow preprocessing failed: {e}")
            self.logger.debug(traceback.format_exc())
            from symfluence.core.exceptions import ModelExecutionError
            raise ModelExecutionError(f"Meshflow preprocessing failed: {e}")

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
