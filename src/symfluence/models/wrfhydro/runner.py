"""
WRF-Hydro Model Runner

Executes the WRF-Hydro model using prepared input files.
"""
import logging
import subprocess
import os
from pathlib import Path
from typing import Optional

from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner("WRFHYDRO")
class WRFHydroRunner(BaseModelRunner):
    """
    Runs the WRF-Hydro model.

    Handles:
    - Executable path resolution (wrf_hydro.exe or wrf_hydro_NoahMP.exe)
    - Namelist file setup (HRLDAS + hydro)
    - Model execution with MPI support
    - Output verification (CHRTOUT, LDASOUT NetCDF files)
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the WRF-Hydro runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Setup paths
        self.wrfhydro_input_dir = self.project_dir / "WRFHydro_input"
        self.settings_dir = self.wrfhydro_input_dir / "settings"
        self.forcing_dir = self.wrfhydro_input_dir / "forcing"
        self.routing_dir = self.wrfhydro_input_dir / "routing"

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "WRFHYDRO"

    def _get_wrfhydro_executable(self) -> Path:
        """
        Get the WRF-Hydro executable path.

        Tries wrf_hydro.exe first, then wrf_hydro_NoahMP.exe.

        Returns:
            Path: Path to WRF-Hydro executable.

        Raises:
            FileNotFoundError: If executable not found.
        """
        return self.get_model_executable(
            install_path_key='WRFHYDRO_INSTALL_PATH',
            default_install_subpath='installs/wrfhydro',
            default_exe_name='wrf_hydro.exe',
            typed_exe_accessor=lambda: (
                self.config.model.wrfhydro.exe
                if self.config.model and self.config.model.wrfhydro
                else None
            ),
            candidates=['bin', 'Run', ''],
            must_exist=True
        )

    def _get_namelist_file(self) -> Path:
        """Get path to the HRLDAS namelist file."""
        namelist_name = self._get_config_value(
            lambda: self.config.model.wrfhydro.namelist_file,
            default='namelist.hrldas'
        )
        return self.settings_dir / namelist_name

    def _get_hydro_namelist_file(self) -> Path:
        """Get path to the hydro namelist file."""
        hydro_name = self._get_config_value(
            lambda: self.config.model.wrfhydro.hydro_namelist,
            default='hydro.namelist'
        )
        return self.settings_dir / hydro_name

    def _get_timeout(self) -> int:
        """Get execution timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.wrfhydro.timeout,
            default=7200
        )

    def run(self, **kwargs) -> Optional[Path]:
        """
        Execute the WRF-Hydro model.

        Args:
            **kwargs: Additional arguments (unused)

        Returns:
            Path to output directory on success

        Raises:
            ModelExecutionError: If model execution fails
        """
        logger.info(f"Running WRF-Hydro for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "WRF-Hydro model execution",
            logger,
            error_type=ModelExecutionError
        ):
            # Setup output directory
            self.output_dir = (
                self.project_dir / "simulations" /
                self.config.domain.experiment_id / "WRFHYDRO"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            wrfhydro_exe = self._get_wrfhydro_executable()
            logger.info(f"Using WRF-Hydro executable: {wrfhydro_exe}")

            # Verify namelists exist
            namelist_file = self._get_namelist_file()
            hydro_namelist = self._get_hydro_namelist_file()

            if not namelist_file.exists():
                raise ModelExecutionError(
                    f"HRLDAS namelist not found: {namelist_file}"
                )
            if not hydro_namelist.exists():
                raise ModelExecutionError(
                    f"Hydro namelist not found: {hydro_namelist}"
                )

            # Copy namelists to working directory (WRF-Hydro expects them in cwd)
            import shutil
            shutil.copy2(namelist_file, self.output_dir / namelist_file.name)
            shutil.copy2(hydro_namelist, self.output_dir / hydro_namelist.name)

            # Copy domain/routing files if they exist
            for nc_file in ['wrfinput_d01.nc', 'Fulldom_hires.nc', 'Route_Link.nc',
                            'soil_properties.nc', 'GWBUCKPARM.nc']:
                src_file = self.settings_dir / nc_file
                if src_file.exists():
                    shutil.copy2(src_file, self.output_dir / nc_file)

            # Build command — WRF-Hydro is MPI-compiled, must use mpirun
            mpi_procs = self._get_config_value(
                lambda: self.config.compute.mpi_processes,
                default=1
            )
            cmd = ['mpirun', '-np', str(mpi_procs), str(wrfhydro_exe)]

            logger.info(f"Executing command: {' '.join(cmd)}")

            # Set environment
            env = os.environ.copy()

            # Run the model
            timeout = self._get_timeout()
            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Log output
            if result.stdout:
                logger.debug(f"WRF-Hydro stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"WRF-Hydro stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"WRF-Hydro execution returned code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}")
                raise ModelExecutionError(
                    f"WRF-Hydro execution failed with return code {result.returncode}"
                )

            logger.info("WRF-Hydro execution completed successfully")

            # Verify output was produced
            self._verify_output()

            return self.output_dir

    def _verify_output(self) -> None:
        """
        Verify that WRF-Hydro produced valid output files.

        Expects CHRTOUT and/or LDASOUT NetCDF files.

        Raises:
            RuntimeError: If expected output files are missing or empty
        """
        # Look for CHRTOUT files (channel output)
        chrtout_files = list(self.output_dir.glob("*CHRTOUT*"))

        # Look for LDASOUT files (land surface output)
        ldasout_files = list(self.output_dir.glob("*LDASOUT*"))

        output_files = chrtout_files + ldasout_files

        if not output_files:
            raise RuntimeError(
                f"WRF-Hydro did not produce expected CHRTOUT or LDASOUT files "
                f"in {self.output_dir}"
            )

        # Verify files have content
        for output_file in output_files:
            if output_file.stat().st_size == 0:
                raise RuntimeError(f"WRF-Hydro output file is empty: {output_file}")

        logger.info(
            f"Verified WRF-Hydro output: {len(chrtout_files)} CHRTOUT, "
            f"{len(ldasout_files)} LDASOUT file(s) produced"
        )
