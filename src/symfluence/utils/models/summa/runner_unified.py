"""
SUMMA Runner - Refactored using Unified Model Execution Framework.

This module demonstrates how to implement a model runner using the new
unified framework, reducing code by ~50% while maintaining full functionality.

Migration Notes:
    Old: ~700 lines with duplicated subprocess/SLURM logic
    New: ~200 lines with framework methods

The key changes:
    1. Inherits from UnifiedModelRunner instead of BaseModelRunner
    2. Uses execute_subprocess() instead of manual subprocess.run()
    3. Uses submit_slurm_job() instead of inline SLURM handling
    4. Uses SpatialOrchestrator for routing instead of inline conversion

Author: SYMFLUENCE Development Team
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr

from ..registry import ModelRegistry
from ..templates import UnifiedModelRunner, ModelRunResult
from ..execution import ExecutionResult, SlurmJobConfig, ExecutionMode
from symfluence.utils.exceptions import ModelExecutionError


@ModelRegistry.register_runner('SUMMA_UNIFIED', method_name='run_summa_unified')
class SummaRunnerUnified(UnifiedModelRunner):
    """
    SUMMA runner using the unified model execution framework.

    This implementation reduces the original ~700 lines to ~200 lines
    by leveraging the framework components:
        - ModelExecutor: handles subprocess and SLURM execution
        - SpatialOrchestrator: handles routing integration
        - ModelConfigSchema: validates configuration

    Attributes:
        settings_path: Path to SUMMA settings directory
        file_manager: Path to SUMMA file manager file
    """

    def _get_model_name(self) -> str:
        """Return model name for SUMMA."""
        return "SUMMA"

    def _setup_model_specific_paths(self) -> None:
        """Set up SUMMA-specific paths."""
        self.settings_path = self.get_config_path(
            'SETTINGS_SUMMA_PATH',
            'settings/SUMMA/'
        )
        self.file_manager = self.settings_path / self.config_dict.get(
            'SETTINGS_SUMMA_FILEMANAGER',
            'fileManager.txt'
        )

        # Legacy alias for backward compatibility
        self.setup_path_aliases({'root_path': 'data_dir'})

    def _should_create_output_dir(self) -> bool:
        """SUMMA creates output dirs on-demand."""
        return False

    def _build_command(self) -> List[str]:
        """Build SUMMA execution command."""
        return [
            str(self.model_exe),
            '-m', str(self.file_manager)
        ]

    def _get_environment(self) -> Dict[str, str]:
        """Get environment variables for SUMMA."""
        import os
        return {
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
        }

    def _validate_model_specific(self) -> List[str]:
        """Validate SUMMA-specific configuration."""
        errors = []

        # Check file manager exists
        if hasattr(self, 'file_manager') and not self.file_manager.exists():
            errors.append(f"File manager not found: {self.file_manager}")

        # Check executable exists
        if self.model_exe and not self.model_exe.exists():
            errors.append(f"SUMMA executable not found: {self.model_exe}")

        return errors

    def _get_slurm_config(self) -> Optional[SlurmJobConfig]:
        """Get SLURM configuration for parallel SUMMA."""
        use_parallel = self.config_dict.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA', False)

        if not use_parallel:
            return None

        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        log_path = self.get_config_path(
            'EXPERIMENT_LOG_SUMMA',
            f"simulations/{experiment_id}/SUMMA/SUMMA_logs/"
        )

        return SlurmJobConfig(
            job_name=f"SUMMA-{self.domain_name}",
            time_limit="03:00:00",
            memory="4G",
            cpus_per_task=1,
            output_pattern=str(log_path / "summa_%A_%a.out"),
            error_pattern=str(log_path / "summa_%A_%a.err"),
        )

    def _pre_execution(self) -> bool:
        """Set up for SUMMA execution."""
        experiment_id = self.config_dict.get('EXPERIMENT_ID')

        # Create output directories
        self.output_dir = self.get_config_path(
            'EXPERIMENT_OUTPUT_SUMMA',
            f"simulations/{experiment_id}/SUMMA/"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.get_config_path(
            'EXPERIMENT_LOG_SUMMA',
            f"simulations/{experiment_id}/SUMMA/SUMMA_logs/"
        )
        log_path.mkdir(parents=True, exist_ok=True)

        # Backup settings if requested
        if self.config_dict.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            self.backup_settings(self.settings_path)

        return True

    def _post_execution(self, result: ExecutionResult) -> ModelRunResult:
        """Process SUMMA outputs."""
        run_result = ModelRunResult(
            success=result.success,
            output_path=self.output_dir,
            error=result.error_message,
            metadata={
                'duration_seconds': result.duration_seconds,
                'job_id': result.job_id,
            }
        )

        if not result.success:
            return run_result

        # Check if we need to convert lumped output for distributed routing
        domain_method = self.config_dict.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = self.config_dict.get('ROUTING_DELINEATION', 'lumped')

        if domain_method == 'lumped' and routing_delineation == 'river_network':
            self.logger.info("Converting lumped output for distributed routing")
            self._convert_lumped_for_routing()

        return run_result

    # =========================================================================
    # SUMMA-Specific Methods
    # =========================================================================

    def run_summa(self) -> Optional[Path]:
        """
        Run SUMMA - main entry point.

        Delegates to appropriate execution method based on configuration.
        """
        use_parallel = self.config_dict.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA', False)

        if use_parallel:
            return self.run_parallel_summa()
        else:
            return self.run()

    def run_parallel_summa(self) -> Optional[Path]:
        """
        Run SUMMA in parallel using SLURM arrays.

        This method replaces the original 150+ line run_summa_parallel()
        by delegating to the framework's parallel execution support.
        """
        self.logger.info("Starting parallel SUMMA run with SLURM")

        # Get GRU count from shapefile
        total_grus = self._count_grus()
        self.logger.info(f"Total GRUs: {total_grus}")

        # Calculate optimal parallelization
        grus_per_job = self.estimate_optimal_grus_per_job(total_grus)
        self.logger.info(f"GRUs per job: {grus_per_job}")

        # Pre-execution setup
        if not self._pre_execution():
            return None

        # Create and submit SLURM script
        script_content = self.create_gru_parallel_script(
            model_exe=self.model_exe,
            file_manager=self.file_manager,
            log_dir=self.get_log_path(),
            total_grus=total_grus,
            grus_per_job=grus_per_job,
            job_name=f"SUMMA-{self.domain_name}",
        )

        script_path = self.project_dir / 'run_summa_parallel.sh'
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        # Backup settings if requested
        if self.config_dict.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            self.backup_settings(self.settings_path)

        # Submit and optionally wait
        result = self.submit_slurm_job(
            script_path=script_path,
            wait=self.config_dict.get('MONITOR_SLURM_JOB', True),
            max_wait_time=3600
        )

        if result.success:
            return self._merge_parallel_outputs()
        else:
            self.logger.error(f"Parallel SUMMA failed: {result.error_message}")
            return None

    def _count_grus(self) -> int:
        """Count total GRUs from catchment shapefile."""
        subbasins_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = (
                f"{self.domain_name}_HRUs_"
                f"{self.config_dict.get('DOMAIN_DISCRETIZATION')}.shp"
            )

        shapefile = self.project_dir / "shapefiles" / "catchment" / subbasins_name

        try:
            gdf = gpd.read_file(shapefile)
            gru_col = self.config_dict.get('CATCHMENT_SHP_GRUID', 'GRU_ID')
            return len(gdf[gru_col].unique())
        except Exception as e:
            self.logger.error(f"Error counting GRUs: {e}")
            raise

    def _merge_parallel_outputs(self) -> Optional[Path]:
        """
        Merge parallel SUMMA outputs into unified files.

        Creates:
            - {experiment_id}_timestep.nc
            - {experiment_id}_day.nc
        """
        self.logger.info("Merging parallel SUMMA outputs")

        experiment_id = self.config_dict.get('EXPERIMENT_ID')

        try:
            # Process timestep and daily files
            for pattern, suffix in [
                (f"{experiment_id}_*_timestep.nc", "timestep"),
                (f"{experiment_id}_*_day.nc", "day")
            ]:
                output_file = self.output_dir / f"{experiment_id}_{suffix}.nc"
                self._merge_files(pattern, output_file)

            self.logger.info("SUMMA output merging completed")
            return self.output_dir

        except Exception as e:
            self.logger.error(f"Error merging outputs: {e}")
            return None

    def _merge_files(self, pattern: str, output_file: Path) -> None:
        """Merge files matching pattern into output file."""
        input_files = sorted(self.output_dir.glob(pattern))

        if not input_files:
            self.logger.warning(f"No files matching: {pattern}")
            return

        merged_ds = None
        reference_date = pd.Timestamp('1990-01-01')

        for src_file in input_files:
            try:
                ds = xr.open_dataset(src_file)

                # Convert time to seconds since reference
                time_values = pd.to_datetime(ds.time.values)
                seconds_since_ref = (time_values - reference_date).total_seconds()
                ds = ds.assign_coords(time=seconds_since_ref)
                ds.time.attrs = {
                    'units': 'seconds since 1990-1-1 0:0:0.0 -0:00',
                    'calendar': 'standard',
                }

                if merged_ds is None:
                    merged_ds = ds
                else:
                    merged_ds = xr.merge([merged_ds, ds])

                ds.close()

            except Exception as e:
                self.logger.warning(f"Error processing {src_file}: {e}")

        if merged_ds is not None:
            encoding = {'time': {'dtype': 'double', '_FillValue': None}}
            for var in merged_ds.data_vars:
                encoding[var] = {'_FillValue': None}

            merged_ds.to_netcdf(
                output_file,
                encoding=encoding,
                unlimited_dims=['time'],
                format='NETCDF4'
            )
            merged_ds.close()
            self.logger.info(f"Created: {output_file}")

    def _convert_lumped_for_routing(self) -> None:
        """Convert lumped SUMMA output for distributed routing."""
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        timestep_file = self.output_dir / f"{experiment_id}_timestep.nc"

        if not timestep_file.exists():
            self.logger.warning(f"Timestep file not found: {timestep_file}")
            return

        # Use SpatialOrchestrator's conversion
        routing_config = self.spatial_config.routing
        self.convert_to_routing_format(
            timestep_file,
            routing_config=routing_config
        )

    def run_summa_point(self) -> Optional[Path]:
        """
        Run SUMMA in point simulation mode.

        Executes SUMMA for multiple point simulations based on file manager lists.
        """
        self.logger.info("Starting SUMMA point simulations")

        fm_ic_list_path = self.settings_path / 'list_fileManager_IC.txt'
        fm_list_path = self.settings_path / 'list_fileManager.txt'

        # Verify files exist
        self.verify_required_files(
            [fm_ic_list_path, fm_list_path],
            "SUMMA point simulations"
        )

        # Read file manager lists
        with open(fm_ic_list_path) as f:
            fm_ic_list = [line.strip() for line in f if line.strip()]
        with open(fm_list_path) as f:
            fm_list = [line.strip() for line in f if line.strip()]

        # Create output directory
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        output_path = self.project_dir / 'simulations' / experiment_id / 'SUMMA_point'
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each site
        for i, (ic_fm, main_fm) in enumerate(zip(fm_ic_list, fm_list)):
            site_name = Path(ic_fm).stem.split('_')[1]
            self.logger.info(f"Processing site {i+1}/{len(fm_list)}: {site_name}")

            site_output = output_path / site_name
            site_output.mkdir(parents=True, exist_ok=True)
            log_path = site_output / "logs"
            log_path.mkdir(parents=True, exist_ok=True)

            # Run IC simulation
            ic_result = self.execute_subprocess(
                command=[str(self.model_exe), '-m', ic_fm, '-r', 'e'],
                log_file=log_path / f"{site_name}_IC.log",
                check=False
            )

            if not ic_result.success:
                self.logger.error(f"IC simulation failed for {site_name}")
                continue

            # Copy restart file
            restart_files = list(site_output.glob("*restart*"))
            if restart_files:
                import shutil
                newest = max(restart_files, key=lambda p: p.stat().st_mtime)
                shutil.copy(newest, Path(ic_fm).parent / "warm_state.nc")

            # Run main simulation
            main_result = self.execute_subprocess(
                command=[str(self.model_exe), '-m', main_fm],
                log_file=log_path / f"{site_name}_main.log",
                check=False
            )

            if main_result.success:
                self.logger.info(f"Completed site: {site_name}")
            else:
                self.logger.error(f"Main simulation failed for {site_name}")

        self.logger.info(f"Completed {len(fm_list)} point simulations")
        return output_path
