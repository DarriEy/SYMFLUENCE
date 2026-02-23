"""
IGNACIO Model Runner for SYMFLUENCE

Executes the IGNACIO fire spread model using the ignacio Python package.
"""

import logging
from pathlib import Path
from typing import Optional

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner('IGNACIO')
class IGNACIORunner(BaseModelRunner):
    """
    Runs the IGNACIO fire spread model.

    IGNACIO is a Python package implementing the Canadian FBP System
    for fire spread simulation. This runner invokes the ignacio package
    either via its Python API or CLI.

    Handles:
    - Loading IGNACIO configuration
    - Running fire spread simulation
    - Collecting and organizing outputs
    """


    MODEL_NAME = "IGNACIO"
    def __init__(self, config, logger_instance=None, reporting_manager=None):
        """
        Initialize the IGNACIO runner.

        Args:
            config: SymfluenceConfig object with IGNACIO settings
            logger_instance: Optional logger for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger_instance or logger, reporting_manager)

        # IGNACIO-specific paths
        self.ignacio_input_dir = self.project_dir / "IGNACIO_input"
        self.ignacio_config_path = self.ignacio_input_dir / "ignacio_config.yaml"

    def run_ignacio(self, **kwargs) -> Optional[Path]:
        """
        Execute the IGNACIO fire spread simulation.

        Attempts to run IGNACIO via its Python API first, falling back
        to CLI if the API is unavailable.

        Returns:
            Path to output directory on success, None on failure
        """
        self.logger.info(f"Running IGNACIO for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "IGNACIO fire simulation",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = (
                self.project_dir / "simulations" /
                self.config.domain.experiment_id / "IGNACIO"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Verify config file exists
            if not self.ignacio_config_path.exists():
                raise FileNotFoundError(
                    f"IGNACIO config not found: {self.ignacio_config_path}\n"
                    f"Run preprocessing first: symfluence workflow step preprocessing"
                )

            # Try Python API first
            success = self._run_via_api()

            if not success:
                # Fall back to CLI
                self.logger.info("Falling back to IGNACIO CLI...")
                success = self._run_via_cli()

            if success:
                self.logger.info(f"IGNACIO simulation completed. Output: {self.output_dir}")
                return self.output_dir
            else:
                raise ModelExecutionError("IGNACIO simulation failed")

    def _run_via_api(self) -> bool:
        """
        Run IGNACIO via Python API.

        Returns:
            True if successful, False if API not available or failed
        """
        try:
            from ignacio.config import load_config, validate_paths
            from ignacio.simulation import run_simulation

            self.logger.info(f"Loading IGNACIO config from {self.ignacio_config_path}")
            config = load_config(self.ignacio_config_path)

            # Update output directory to SYMFLUENCE location
            config.project.output_dir = str(self.output_dir)

            # Validate input files
            warnings = validate_paths(config)
            for warning in warnings:
                self.logger.warning(f"IGNACIO: {warning}")

            # Run simulation
            self.logger.info(f"Starting IGNACIO simulation: {config.project.name}")
            results = run_simulation(config)

            # Log results summary
            self.logger.info("IGNACIO simulation complete:")
            self.logger.info(f"  - Fires simulated: {results.n_fires}")
            self.logger.info(f"  - Total area burned: {results.total_area_ha:.2f} ha")

            return True

        except ImportError as e:
            self.logger.warning(f"IGNACIO Python package not available: {e}")
            self.logger.warning("Install with: symfluence binary install ignacio")
            return False

        except Exception as e:
            self.logger.error(f"IGNACIO API execution failed: {e}")
            return False

    def _run_via_cli(self) -> bool:
        """
        Run IGNACIO via command line interface.

        Returns:
            True if successful, False otherwise
        """
        import shutil
        import subprocess

        # Check if ignacio CLI is available
        ignacio_cli = shutil.which('ignacio')
        if ignacio_cli is None:
            self.logger.error(
                "IGNACIO CLI not found. Install with: symfluence binary install ignacio"
            )
            return False

        try:
            cmd = [
                ignacio_cli,
                'run',
                str(self.ignacio_config_path),
                '--output', str(self.output_dir),
            ]

            self.logger.info(f"Running IGNACIO CLI: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.stdout:
                self.logger.debug(f"IGNACIO stdout: {result.stdout[-2000:]}")
            if result.stderr:
                self.logger.debug(f"IGNACIO stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                self.logger.error(f"IGNACIO CLI failed with code {result.returncode}")
                self.logger.error(f"stderr: {result.stderr[-1000:] if result.stderr else 'none'}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.logger.error("IGNACIO simulation timed out after 1 hour")
            return False

        except Exception as e:
            self.logger.error(f"IGNACIO CLI execution failed: {e}")
            return False

    def run(self, **kwargs) -> Optional[Path]:
        """Execute IGNACIO fire spread simulation."""
        return self.run_ignacio(**kwargs)

    def _should_create_output_dir(self) -> bool:
        """Don't create output dir in __init__, we do it in run_ignacio."""
        return False
