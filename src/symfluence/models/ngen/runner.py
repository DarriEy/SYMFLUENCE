"""
NGen Model Runner.

Manages the execution of the NOAA NextGen Framework (ngen).
Refactored to use the Unified Model Execution Framework.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelRunner
from symfluence.models.execution import ModelExecutor
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler


@ModelRegistry.register_runner('NGEN', method_name='run_ngen')
class NgenRunner(BaseModelRunner, ModelExecutor):
    """
    Runner for NextGen Framework simulations.

    Handles execution of ngen with proper paths and error handling.
    Uses the Unified Model Execution Framework for subprocess execution.
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up NGEN-specific paths."""
        self.ngen_setup_dir = self.project_dir / "settings" / "NGEN"

        # Use standardized executable resolution from BaseModelRunner
        # Note: NGEN install path is relative to parent of data_dir (../installs/ngen/build)
        self.ngen_exe = self.get_model_executable(
            install_path_key='NGEN_INSTALL_PATH',
            default_install_subpath='installs/ngen/build',
            exe_name_key=None,  # NGEN exe name is just 'ngen'
            default_exe_name='ngen',
            relative_to='code_dir'
        )

    def _get_model_name(self) -> str:
        """Return model name for NextGen."""
        return "NGEN"

    def _should_create_output_dir(self) -> bool:
        """NGEN creates directories on-demand in run_ngen."""
        return False

    def run_ngen(self, experiment_id: str = None):
        """
        Execute NextGen model simulation.

        Args:
            experiment_id: Optional experiment identifier. If None, uses config value.

        Runs ngen with the prepared catchment, nexus, forcing, and configuration files.
        """
        self.logger.debug("Starting NextGen model run")

        with symfluence_error_handler(
            "NextGen model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Get experiment info
            if experiment_id is None:
                if self.config:
                    experiment_id = self.config.domain.experiment_id
                else:
                    experiment_id = self.config_dict.get('EXPERIMENT_ID', 'default_run')
            output_dir = self.get_experiment_output_dir(experiment_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Setup paths for ngen execution
            if self.config:
                domain_name = self.config.domain.name
            else:
                domain_name = self.config_dict.get('DOMAIN_NAME')
            use_geojson = getattr(self, "_use_geojson_catchments", False)
            if use_geojson:
                catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.geojson"
            else:
                catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.gpkg"
            fallback_catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.geojson"
            nexus_file = self.ngen_setup_dir / "nexus.geojson"
            realization_file = self.ngen_setup_dir / "realization_config.json"

            # Verify required files exist
            self.verify_required_files(
                [catchment_file, nexus_file, realization_file],
                context="NextGen model execution"
            )

            # Build ngen command
            ngen_cmd = [
                str(self.ngen_exe),
                str(catchment_file),
                "all",
                str(nexus_file),
                "all",
                str(realization_file)
            ]

            self.logger.debug(f"Running command: {' '.join(ngen_cmd)}")

            # Run ngen
            log_file = output_dir / "ngen_log.txt"
            try:
                # Setup environment with library paths
                env = os.environ.copy()

                self.execute_model_subprocess(
                    ngen_cmd,
                    log_file,
                    cwd=self.ngen_exe.parent,  # Run from ngen build directory (needed for relative library paths)
                    env=env,  # Use modified environment with library paths
                    success_message="NextGen model run completed successfully"
                )

                # Move outputs from build directory to output directory
                self._move_ngen_outputs(self.ngen_exe.parent, output_dir)

                return True

            except subprocess.CalledProcessError as e:
                if not use_geojson and fallback_catchment_file.exists():
                    try:
                        log_text = log_file.read_text(errors='ignore')
                    except Exception:
                        log_text = ""
                    sqlite_error = "SQLite3 support required to read GeoPackage files"
                    if sqlite_error in log_text:
                        self.logger.warning(
                            "NGEN lacks GeoPackage support; retrying with GeoJSON catchments"
                        )
                        ngen_cmd[1] = str(fallback_catchment_file)
                        try:
                            self.execute_model_subprocess(
                                ngen_cmd,
                                log_file,
                                cwd=self.ngen_exe.parent,
                                env=env,
                                success_message="NextGen model run completed successfully (GeoJSON fallback)"
                            )
                            self._use_geojson_catchments = True
                            self._move_ngen_outputs(self.ngen_exe.parent, output_dir)
                            return True
                        except subprocess.CalledProcessError as retry_error:
                            self.logger.error(
                                f"NextGen model run failed with error code {retry_error.returncode}"
                            )
                            self.logger.error(f"Check log file: {log_file}")
                            return False

                self.logger.error(f"NextGen model run failed with error code {e.returncode}")
                self.logger.error(f"Check log file: {log_file}")
                return False
    
    def _move_ngen_outputs(self, build_dir: Path, output_dir: Path):
        """
        Move ngen output files from build directory to output directory.
        
        ngen writes outputs to its working directory, so we need to move them
        to the proper experiment output directory.
        
        Args:
            build_dir: ngen build directory where outputs are written
            output_dir: Target output directory for this experiment
        """
        import shutil
        
        # Common ngen output patterns
        output_patterns = [
            'cat-*.csv',      # Catchment outputs
            'nex-*.csv',      # Nexus outputs  
            '*.parquet',      # Parquet outputs
            'cfe_output_*.txt',  # CFE specific outputs
            'noah_output_*.txt', # Noah specific outputs
        ]
        
        moved_files = []
        for pattern in output_patterns:
            for file in build_dir.glob(pattern):
                dest = output_dir / file.name
                shutil.move(str(file), str(dest))
                moved_files.append(file.name)
        
        if moved_files:
            self.logger.debug(f"Moved {len(moved_files)} output files to {output_dir}")
            for f in moved_files[:10]:  # Log first 10
                self.logger.debug(f"  - {f}")
            if len(moved_files) > 10:
                self.logger.debug(f"  ... and {len(moved_files) - 10} more")
        else:
            existing_outputs = []
            for pattern in output_patterns:
                existing_outputs.extend(output_dir.glob(pattern))
            if not existing_outputs:
                self.logger.warning(
                    f"No output files found in {build_dir} or {output_dir}. Check if model ran correctly."
                )
