"""
RHESSys Model Runner

Executes the RHESSys model using prepared input files.
"""
import logging
import subprocess
import os
from pathlib import Path

import pandas as pd

from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner("RHESSys")
class RHESSysRunner(BaseModelRunner):
    """
    Runs the RHESSys model.

    Handles:
    - Executable path resolution
    - Command line argument construction
    - Library path setup for WMFire
    - Output directory management
    """

    def __init__(self, config, logger_instance, reporting_manager=None):
        """
        Initialize the RHESSys runner.

        Sets up paths to RHESSys input directories and checks for optional
        WMFire wildfire spread module configuration.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                RHESSys settings, domain paths, and simulation parameters.
            logger_instance: Logger instance for status messages and debugging.
            reporting_manager: Optional reporting manager for experiment tracking.
        """
        super().__init__(config, logger_instance, reporting_manager=reporting_manager)
        # Check for WMFire support (handles both wmfire and legacy vmfire config names)
        self.wmfire_enabled = self._check_wmfire_enabled()

        # Setup paths
        self.rhessys_input_dir = self.project_dir / "RHESSys_input"
        self.worldfiles_dir = self.rhessys_input_dir / "worldfiles"
        self.tecfiles_dir = self.rhessys_input_dir / "tecfiles"
        self.climate_dir = self.rhessys_input_dir / "clim"
        self.defs_dir = self.rhessys_input_dir / "defs"

    def _check_wmfire_enabled(self) -> bool:
        """Check if WMFire fire spread is enabled (supports both new and legacy config names)."""
        try:
            # Try new naming first
            if hasattr(self.config.model.rhessys, 'use_wmfire'):
                return self.config.model.rhessys.use_wmfire
            # Fall back to legacy vmfire naming
            if hasattr(self.config.model.rhessys, 'use_vmfire'):
                return self.config.model.rhessys.use_vmfire
        except AttributeError:
            pass
        return False

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "RHESSys"

    def _get_rhessys_executable(self) -> Path:
        """
        Get the RHESSys executable path.

        Uses the standardized get_model_executable method with candidates
        to search multiple potential locations for the RHESSys binary.

        Returns:
            Path: Path to RHESSys executable.

        Raises:
            FileNotFoundError: If executable not found in any location.
        """
        return self.get_model_executable(
            install_path_key='RHESSYS_INSTALL_PATH',
            default_install_subpath='installs/rhessys',
            default_exe_name='rhessys',
            typed_exe_accessor=lambda: (
                self.config.model.rhessys.installation.exe_name
                if self.config.model and self.config.model.rhessys and self.config.model.rhessys.installation
                else None
            ),
            candidates=['bin', ''],  # Search bin subdirectory first, then root
            must_exist=True
        )

    def run(self, **kwargs):
        """
        Execute the RHESSys model.

        Returns:
            Path to output directory
        """
        logger.info(f"Running RHESSys for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "RHESSys model execution",
            logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = self.project_dir / "simulations" / self.config.domain.experiment_id / "RHESSys"
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            rhessys_exe = self._get_rhessys_executable()

            # Build command
            cmd = self._build_command(rhessys_exe)

            # Set library path for WMFire
            env = os.environ.copy()

            # Add both rhessys/lib and wmfire/lib to library path
            lib_paths = []
            rhessys_bin_dir = rhessys_exe.parent
            rhessys_lib_dir = rhessys_bin_dir.parent / "lib"
            wmfire_lib_dir = Path(self.config.system.data_dir) / "installs" / "wmfire" / "lib"

            for lib_dir in [rhessys_bin_dir, rhessys_lib_dir, wmfire_lib_dir]:
                if lib_dir.exists():
                    lib_paths.append(str(lib_dir))

            if lib_paths:
                lib_path_str = ":".join(lib_paths)
                import sys
                if sys.platform == "darwin":
                    env["DYLD_LIBRARY_PATH"] = f"{lib_path_str}:{env.get('DYLD_LIBRARY_PATH', '')}"
                else:
                    env["LD_LIBRARY_PATH"] = f"{lib_path_str}:{env.get('LD_LIBRARY_PATH', '')}"

            logger.info(f"Executing command: {' '.join(cmd)}")

            # Run the model
            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            # Log output
            if result.stdout:
                logger.debug(f"RHESSys stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"RHESSys stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"RHESSys execution returned code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}")
                raise ModelExecutionError(f"RHESSys execution failed with return code {result.returncode}")

            logger.info("RHESSys execution completed successfully")

            # Verify output was produced
            self._verify_output()

            return self.output_dir

    def _verify_output(self):
        """Verify that RHESSys produced valid output files with data."""
        basin_daily = self.output_dir / "rhessys_basin.daily"

        if not basin_daily.exists():
            raise RuntimeError(f"RHESSys did not produce basin daily output: {basin_daily}")

        # Check that file has more than just header
        with open(basin_daily, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise RuntimeError(
                f"RHESSys basin.daily output has no data rows (only {len(lines)} lines). "
                "Check worldfile, TEC file, and climate data configuration."
            )

        logger.info(f"Verified RHESSys output: {len(lines)-1} days of data in basin.daily")

    def _build_command(self, rhessys_exe: Path):
        """
        Construct the command to run RHESSys.

        Args:
            rhessys_exe: Path to RHESSys executable

        Returns:
            List of command arguments
        """
        cmd = [str(rhessys_exe)]

        # World file
        world_file = self.worldfiles_dir / f"{self.config.domain.name}.world"
        if world_file.exists():
            cmd.extend(["-w", str(world_file)])
        else:
            logger.warning(f"Worldfile not found: {world_file}")

        # TEC file
        tec_file = self.tecfiles_dir / f"{self.config.domain.name}.tec"
        if tec_file.exists():
            cmd.extend(["-t", str(tec_file)])
        else:
            logger.warning(f"TEC file not found: {tec_file}")

        # Output prefix
        output_prefix = self.output_dir / "rhessys"
        cmd.extend(["-pre", str(output_prefix)])

        # Start and end dates (required by RHESSys)
        start_str = self._get_config_value(
            lambda: self.config.domain.time_start
        )
        end_str = self._get_config_value(
            lambda: self.config.domain.time_end
        )
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        # Format: -st year month day hour -ed year month day hour
        cmd.extend(["-st", str(start_date.year), str(start_date.month),
                    str(start_date.day), "1"])
        cmd.extend(["-ed", str(end_date.year), str(end_date.month),
                    str(end_date.day), "1"])

        # Note: Default files are specified in worldfile header, no -d flag needed

        # Output flags - enable legacy basin output (required for streamflow)
        # -b enables basin daily output (streamflow, ET, etc.)
        cmd.extend(["-b"])

        # Fire spread if WMFire is enabled
        if self.wmfire_enabled:
            fire_dir = self.rhessys_input_dir / "fire"
            patch_grid = fire_dir / "patch_grid.txt"
            dem_grid = fire_dir / "dem_grid.txt"
            if patch_grid.exists() and dem_grid.exists():
                cmd.extend(["-firespread", "30", str(patch_grid), str(dem_grid)])
                logger.info("WMFire fire spread enabled with patch and DEM grids")
            else:
                logger.warning("WMFire enabled but fire grid files not found, skipping -firespread flag")

        # Routing if available
        routing_file = self.rhessys_input_dir / "routing" / f"{self.config.domain.name}.routing"
        if routing_file.exists():
            cmd.extend(["-r", str(routing_file)])

        return cmd
