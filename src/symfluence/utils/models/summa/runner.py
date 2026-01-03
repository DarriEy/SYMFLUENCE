"""
SUMMA Runner Module

This module contains the SummaRunner class for executing the SUMMA
(Structure for Unifying Multiple Modeling Alternatives) model.

The SummaRunner handles model execution in various modes:
- Serial execution for single-threaded runs
- Parallel execution using SLURM job arrays
- Point simulation mode for multiple point-based simulations

Refactored to use the Unified Model Execution Framework:
- ModelExecutor: For subprocess and SLURM execution
- SpatialOrchestrator: For routing integration

Author: SYMFLUENCE Development Team
"""

# Standard library imports
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Third-party imports
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import netCDF4 as nc4
import tempfile

# Local imports
from ..registry import ModelRegistry
from ..base import BaseModelRunner
from ..execution import ModelExecutor, SpatialOrchestrator, ExecutionResult, SlurmJobConfig
from symfluence.utils.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)
from symfluence.utils.data.utilities.netcdf_utils import create_minimal_encoding


@ModelRegistry.register_runner('SUMMA', method_name='run_summa')
class SummaRunner(BaseModelRunner, ModelExecutor, SpatialOrchestrator):
    """
    A class to run the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    This class handles the execution of the SUMMA model, including setting up paths,
    running the model, and managing log files.

    Now uses the Unified Model Execution Framework for:
    - SLURM job submission and monitoring (via ModelExecutor)
    - Routing integration (via SpatialOrchestrator)

    Attributes:
        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """
    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # SummaRunner uses 'root_path' alias for backwards compatibility
        self.setup_path_aliases({'root_path': 'data_dir'})

    def _get_model_name(self) -> str:
        """Return model name for SUMMA."""
        return "SUMMA"

    def _should_create_output_dir(self) -> bool:
        """SUMMA creates output dirs on-demand in run methods."""
        return False

    def run_summa(self):
        """
        Run the SUMMA model.

        This method selects the appropriate run mode (parallel, serial, or point)
        based on configuration settings and executes the SUMMA model accordingly.

        Raises:
            ModelExecutionError: If model execution fails
        """
        with symfluence_error_handler(
            "SUMMA model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Phase 3: Use typed config when available for clearer intent
            if self.config:
                use_parallel = self.config.model.summa.use_parallel if self.config.model.summa else False
            else:
                use_parallel = self.config_dict.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA', False)

            if use_parallel:
                self.run_summa_parallel()
            else:
                self.run_summa_serial()

    def run_summa_point(self):
        """
        Run SUMMA in point simulation mode.

        This method executes SUMMA for multiple point simulations, based on the
        file manager list created during preprocessing. It handles both the
        initial condition runs and the main simulation runs.
        """
        self.logger.info("Starting SUMMA point simulations")

        # Set up paths
        summa_path = self.get_install_path('SUMMA_INSTALL_PATH', 'installs/summa/bin/')
        summa_exe = self.config_dict.get('SUMMA_EXE')
        setting_path = self.get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA_point/')

        # Run all sites from the file manager lists
        fm_ic_list_path = setting_path / 'list_fileManager_IC.txt'
        fm_list_path = setting_path / 'list_fileManager.txt'

        # Check if file manager lists exist
        if not fm_ic_list_path.exists() or not fm_list_path.exists():
            self.logger.error(f"File manager lists not found at {setting_path}")
            raise FileNotFoundError(f"Required file manager lists not found at {setting_path}")

        # Read file manager lists
        with open(fm_ic_list_path, 'r') as f:
            fm_ic_list = [line.strip() for line in f if line.strip()]

        with open(fm_list_path, 'r') as f:
            fm_list = [line.strip() for line in f if line.strip()]

        if len(fm_ic_list) != len(fm_list):
            self.logger.warning(f"Mismatch in file manager list lengths: {len(fm_ic_list)} IC files vs {len(fm_list)} main files")

        # Create output directory
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        main_output_path = self.project_dir / 'simulations' / experiment_id / 'SUMMA_point'
        main_output_path.mkdir(parents=True, exist_ok=True)

        # Process each site
        for i, (ic_fm, main_fm) in enumerate(zip(fm_ic_list, fm_list)):
            site_name = os.path.basename(ic_fm).split('_')[1]  # Extract site name from file manager name
            self.logger.info(f"Processing site {i+1}/{len(fm_list)}: {site_name}")

            # Create site-specific output directory
            site_output_path = main_output_path / site_name
            site_output_path.mkdir(parents=True, exist_ok=True)

            # Create log directory
            log_path = site_output_path / "logs"
            log_path.mkdir(parents=True, exist_ok=True)

            # Run initial conditions (IC) simulation
            self.logger.info(f"Running initial conditions simulation for {site_name}")
            ic_command = [str(summa_path / summa_exe), '-m', ic_fm, '-r', 'e']

            try:
                ic_result = self.execute_subprocess(
                    command=ic_command,
                    log_file=log_path / f"{site_name}_IC.log",
                    check=False,
                    success_message=f"IC simulation completed for {site_name}"
                )

                if not ic_result.success:
                    self.logger.error(f"IC simulation failed for {site_name}")
                    continue

                # Find the restart file (newest file with 'restart' in name)
                site_setting_path = Path(os.path.dirname(ic_fm))
                site_output_files = list(site_output_path.glob("*restart*"))

                if not site_output_files:
                    self.logger.error(f"No restart file found for {site_name}")
                    continue

                # Sort by modification time and get the most recent
                restart_file = sorted(site_output_files, key=os.path.getmtime)[-1]

                # Copy to warm_state.nc in settings directory
                shutil.copy(restart_file, site_setting_path / "warm_state.nc")
                self.logger.info(f"Copied restart file to warm state for {site_name}")

                # Run main simulation
                self.logger.info(f"Running main simulation for {site_name}")
                main_command = [str(summa_path / summa_exe), '-m', main_fm]

                main_result = self.execute_subprocess(
                    command=main_command,
                    log_file=log_path / f"{site_name}_main.log",
                    check=False,
                    success_message=f"Main simulation completed for {site_name}"
                )

                if main_result.success:
                    self.logger.info(f"Completed simulation for {site_name}")
                else:
                    self.logger.error(f"Main simulation failed for {site_name}")

            except subprocess.CalledProcessError as e:
                self.logger.error(f"SUMMA run failed for {site_name} with error code {e.returncode}")
                self.logger.error(f"Command that failed: {e.cmd}")
            except Exception as e:
                self.logger.error(f"Error processing site {site_name}: {str(e)}")

        self.logger.info(f"Completed all SUMMA point simulations ({len(fm_list)} sites)")
        return main_output_path

    def run_summa_parallel(self):
        """
        Run SUMMA in parallel using SLURM array jobs.

        This method uses the ModelExecutor framework for SLURM job management,
        replacing ~100 lines of inline SLURM code with framework methods.
        """
        self.logger.info("Starting parallel SUMMA run with SLURM")

        # Check SLURM availability using framework method
        if not self.is_slurm_available():
            self.logger.error("SLURM 'sbatch' command not found. Is SLURM installed?")
            raise RuntimeError("SLURM 'sbatch' command not found")

        # Set up paths and filenames
        summa_path = self.get_install_path('SETTINGS_SUMMA_PARALLEL_PATH', 'installs/summa/bin/')
        summa_exe = self.config_dict.get('SETTINGS_SUMMA_PARALLEL_EXE')
        settings_path = self.get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config_dict.get('SETTINGS_SUMMA_FILEMANAGER')

        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        summa_log_path = self.get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_out_path = self.get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Create output and log directories
        summa_log_path.mkdir(parents=True, exist_ok=True)
        summa_out_path.mkdir(parents=True, exist_ok=True)

        # Get total GRU count from catchment shapefile
        total_grus = self._count_grus_from_shapefile()

        # Use framework method for optimal GRU estimation
        grus_per_job = self.estimate_optimal_grus_per_job(total_grus)
        self.logger.info(f"Optimal GRUs per job: {grus_per_job} for {total_grus} total GRUs")

        # Use framework method to create SLURM script
        script_content = self.create_gru_parallel_script(
            model_exe=summa_path / summa_exe,
            file_manager=settings_path / filemanager,
            log_dir=summa_log_path,
            total_grus=total_grus,
            grus_per_job=grus_per_job,
            job_name=f"SUMMA-{self.domain_name}",
        )

        # Write SLURM script
        script_path = self.project_dir / 'run_summa_parallel.sh'
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        # Backup settings if required
        if self.config_dict.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            self.backup_settings(settings_path)

        # Use framework method to submit and monitor SLURM job
        monitor_job = self.config_dict.get('MONITOR_SLURM_JOB', True)
        result = self.submit_slurm_job(
            script_path=script_path,
            wait=monitor_job,
            poll_interval=60,
            max_wait_time=3600
        )

        if not result.success:
            raise RuntimeError(f"SLURM job failed: {result.error_message}")

        self.logger.info("SUMMA parallel run completed")
        return self.merge_parallel_outputs()

    def _count_grus_from_shapefile(self) -> int:
        """Count total GRUs from catchment shapefile."""
        subbasins_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.domain_name}_HRUs_{self.config_dict.get('DOMAIN_DISCRETIZATION')}.shp"
        subbasins_shapefile = self.project_dir / "shapefiles" / "catchment" / subbasins_name

        try:
            gdf = gpd.read_file(subbasins_shapefile)
            total_grus = len(gdf[self.config_dict.get('CATCHMENT_SHP_GRUID')].unique())
            self.logger.info(f"Counted {total_grus} unique GRUs from: {subbasins_shapefile}")
            return total_grus
        except Exception as e:
            self.logger.error(f"Error counting GRUs: {e}")
            raise RuntimeError(f"Failed to count GRUs from {subbasins_shapefile}: {e}")

    def _estimate_grus_per_job(self, total_grus: int) -> int:
        """
        Estimate the optimal number of GRUs per job based on total GRU count.

        This function balances computational efficiency with queue management by:
        - Keeping the number of parallel jobs reasonable (not too many small jobs)
        - Ensuring each job has enough work to be worthwhile
        - Adapting to different domain sizes

        Args:
            total_grus (int): Total number of GRUs in the domain

        Returns:
            int: Optimal number of GRUs to process per job
        """
        # Define optimization parameters
        min_jobs = 10          # Minimum number of jobs to split into
        max_jobs = 500         # Maximum number of jobs to prevent queue flooding
        min_grus_per_job = 1   # Minimum GRUs per job
        ideal_grus_per_job = 50  # Ideal number of GRUs per job for efficiency

        # For very small domains, process all GRUs in fewer jobs
        if total_grus <= min_jobs:
            return 1

        # For small to medium domains, aim for the ideal GRUs per job
        if total_grus <= ideal_grus_per_job * min_jobs:
            return max(min_grus_per_job, total_grus // min_jobs)

        # For larger domains, balance between ideal number and not exceeding max jobs
        ideal_jobs = total_grus // ideal_grus_per_job

        if ideal_jobs <= max_jobs:
            # We can use the ideal number
            grus_per_job = ideal_grus_per_job
        else:
            # Need to increase GRUs per job to stay under max_jobs limit
            grus_per_job = -(-total_grus // max_jobs)  # Ceiling division

        # Additional consideration for very large domains
        # If we have more than 10,000 GRUs, we might want to increase GRUs per job
        # to reduce overhead and improve efficiency
        if total_grus > 10000:
            # Scale up based on domain size
            scale_factor = min(3.0, total_grus / 10000)
            grus_per_job = int(grus_per_job * scale_factor)

        # Ensure we don't exceed total GRUs
        grus_per_job = min(grus_per_job, total_grus)

        self.logger.debug(f"GRU estimation details: total_grus={total_grus}, "
                        f"ideal_jobs={ideal_jobs}, grus_per_job={grus_per_job}")

        return grus_per_job

    def _create_slurm_script(self, summa_path: Path, summa_exe: str, settings_path: Path,
                            filemanager: str, summa_log_path: Path, summa_out_path: Path,
                            total_grus: int, grus_per_job: int, n_array_jobs: int) -> str:
        """
        Create a SLURM batch script for running SUMMA in parallel.

        Args:
            summa_path (Path): Path to SUMMA executable directory
            summa_exe (str): Name of SUMMA executable
            settings_path (Path): Path to SUMMA settings directory
            filemanager (str): Name of SUMMA file manager
            summa_log_path (Path): Path for SUMMA log files
            summa_out_path (Path): Path for SUMMA output files
            total_grus (int): Total number of GRUs to process
            grus_per_job (int): Number of GRUs to process per job
            n_array_jobs (int): Number of array jobs (0-based maximum index)

        Returns:
            str: Content of the SLURM batch script
        """

        # Create the script
        script = f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00'
#SBATCH --mem=4G
#SBATCH --job-name=Summa-{self.config_dict.get('DOMAIN_NAME')}
#SBATCH --output={summa_log_path}/summa_%A_%a.out
#SBATCH --error={summa_log_path}/summa_%A_%a.err
#SBATCH --array=0-{n_array_jobs}

# Print job info for debugging
echo "Starting SUMMA parallel job at $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"

# Create required directories
mkdir -p {summa_out_path}
mkdir -p {summa_log_path}

# Calculate GRU range for this job
gru_start=$(( ({grus_per_job} * $SLURM_ARRAY_TASK_ID) + 1 ))
gru_end=$(( gru_start + {grus_per_job} - 1 ))

# Ensure we don't exceed total GRUs
if [ $gru_end -gt {total_grus} ]; then
    gru_end={total_grus}
fi

echo "Processing GRUs $gru_start to $gru_end"

# Check if SUMMA executable exists
if [ ! -f "{summa_path}/{summa_exe}" ]; then
    echo "ERROR: SUMMA executable not found at {summa_path}/{summa_exe}"
    exit 1
fi

# Check if filemanager exists
if [ ! -f "{settings_path}/{filemanager}" ]; then
    echo "ERROR: File manager not found at {settings_path}/{filemanager}"
    exit 1
fi

# Process each GRU in the range
for gru in $(seq $gru_start $gru_end); do
    echo "Starting GRU $gru"

    # Run SUMMA
    {summa_path}/{summa_exe} -g $gru 1 -m {settings_path}/{filemanager}

    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "SUMMA failed for GRU $gru with exit code $exit_code"
        exit 1
    fi

    echo "Completed GRU $gru"
done

echo "Completed all GRUs for this job at $(date)"
"""
        return script

    def run_summa_serial(self):
        """
        Run the SUMMA model.

        This method sets up the necessary paths, executes the SUMMA model,
        and handles any errors that occur during the run.
        """
        self.logger.info("Starting SUMMA run")

        # Set up paths and filenames
        summa_path = self.get_install_path('SUMMA_INSTALL_PATH', 'installs/summa/bin/')
        summa_exe = self.config_dict.get('SUMMA_EXE')
        settings_path = self.get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config_dict.get('SETTINGS_SUMMA_FILEMANAGER')

        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        summa_log_path = self.get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_log_name = "summa_log.txt"

        summa_out_path = self.get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Backup settings if required
        if self.config_dict.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            self.backup_settings(settings_path, backup_subdir="run_settings")

        # Run SUMMA
        os.makedirs(summa_log_path, exist_ok=True)
        summa_binary = str(summa_path / summa_exe)
        filemanager_path = str(settings_path / filemanager)

        # Get current environment and ensure LD_LIBRARY_PATH is set
        env = os.environ.copy()
        ld_library_path = env.get('LD_LIBRARY_PATH', '')
        self.logger.info(f"Running SUMMA with LD_LIBRARY_PATH: {ld_library_path}")
        self.logger.info(f"SUMMA binary: {summa_binary}")
        self.logger.info(f"File manager: {filemanager_path}")

        try:
            # Use standardized subprocess execution from ModelExecutor
            result = self.execute_subprocess(
                command=[summa_binary, '-m', filemanager_path],
                log_file=summa_log_path / summa_log_name,
                env=env,
                check=False,
                success_message="SUMMA run completed successfully"
            )

            if not result.success:
                raise subprocess.CalledProcessError(
                    result.return_code, [summa_binary, '-m', filemanager_path]
                )

            # Check if we need to convert lumped output for distributed routing
            domain_method = self.config_dict.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = self.config_dict.get('ROUTING_DELINEATION', 'lumped')

            if domain_method == 'lumped' and routing_delineation == 'river_network':
                self.logger.info("Converting lumped SUMMA output for distributed routing")
                self._convert_lumped_to_distributed_routing()

            return summa_out_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA run failed with error: {e}")
            self.logger.error(f"LD_LIBRARY_PATH was: {ld_library_path}")
            self.logger.error(f"Binary path was: {summa_binary}")
            raise

    def _convert_lumped_to_distributed_routing(self):
        """
        Convert lumped SUMMA output to format suitable for distributed routing.
        
        This handles the case where a lumped model run needs to be mapped to 
        distributed river segments for routing.
        """
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        summa_output_dir = self.project_dir / "simulations" / experiment_id / "SUMMA"
        mizuroute_settings_dir = self.project_dir / "settings" / "mizuRoute"
        summa_timestep_file = summa_output_dir / f"{experiment_id}_timestep.nc"
        
        if not summa_timestep_file.exists():
            self.logger.warning(f"SUMMA timestep file not found for conversion: {summa_timestep_file}")
            return

        topology_file = mizuroute_settings_dir / self.config_dict.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
        if not topology_file.exists():
            self.logger.warning(f"Topology file not found for conversion: {topology_file}")
            return

        # We assume HRU ID 1 for lumped case, but could read from topology
        hru_id = 1 
        
        routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
        
        try:
            summa_output = xr.open_dataset(summa_timestep_file, decode_times=False)
            
            mizuForcing = xr.Dataset()
            original_time = summa_output['time']
            mizuForcing['time'] = xr.DataArray(original_time.values, dims=('time',), attrs=dict(original_time.attrs))
            if 'units' in mizuForcing['time'].attrs:
                mizuForcing['time'].attrs['units'] = mizuForcing['time'].attrs['units'].replace('T', ' ')
            
            mizuForcing['gru'] = xr.DataArray([hru_id], dims=('gru',), attrs={'long_name': 'Index of GRU', 'units': '-'})
            mizuForcing['gruId'] = xr.DataArray([hru_id], dims=('gru',), attrs={'long_name': 'ID of grouped response unit', 'units': '-'})
            mizuForcing.attrs.update(summa_output.attrs)
            
            source_var = None
            for var in [routing_var, 'averageRoutedRunoff', 'basin__TotalRunoff']:
                if var in summa_output:
                    source_var = var
                    break
            
            if source_var:
                lumped_runoff = summa_output[source_var].values
                # Handle extra dimensions if present
                if len(lumped_runoff.shape) == 2:
                    lumped_runoff = lumped_runoff[:, 0]
                    
                mizuForcing[routing_var] = xr.DataArray(lumped_runoff[:, np.newaxis], dims=('time', 'gru'), attrs={'units': 'm/s'})
                summa_output.close()
                
                # Write to temp file then move
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=summa_output_dir) as tmp_file:
                    temp_path = tmp_file.name
                
                mizuForcing.to_netcdf(temp_path, format='NETCDF4')
                mizuForcing.close()
                shutil.move(temp_path, summa_timestep_file)
                self.logger.info(f"Converted {summa_timestep_file} for distributed routing")
            else:
                self.logger.warning("Could not find runoff variable for conversion")
                summa_output.close()
                
        except Exception as e:
            self.logger.error(f"Error converting lumped output: {e}")
            # Don't fail the run, just log error
            if 'summa_output' in locals():
                summa_output.close()


    def merge_parallel_outputs(self):
        """
        Merge parallel SUMMA outputs into two MizuRoute-readable files:
        one for timestep data and one for daily data.
        This function is called after parallel SUMMA execution completes.
        Preserves all variables from the original SUMMA output.
        """
        self.logger.info("Starting to merge parallel SUMMA outputs")

        # Get experiment settings
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        summa_out_path = self.get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")
        mizu_in_path = self.get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")
        mizu_in_path.mkdir(parents=True, exist_ok=True)

        try:
            # Define output files
            timestep_output = mizu_in_path / f"{experiment_id}_timestep.nc"
            daily_output = mizu_in_path / f"{experiment_id}_day.nc"

            # Source file patterns
            timestep_pattern = f"{experiment_id}_*_timestep.nc"
            daily_pattern = f"{experiment_id}_*_day.nc"

            def process_and_merge_files(file_pattern, output_file):
                self.logger.info(f"Processing files matching {file_pattern}")
                input_files = list(summa_out_path.glob(file_pattern))
                input_files.sort()

                if not input_files:
                    self.logger.warning(f"No files found matching pattern: {file_pattern}")
                    return

                merged_ds = None
                for src_file in input_files:
                    try:
                        ds = xr.open_dataset(src_file)

                        # Convert time to seconds since reference date
                        reference_date = pd.Timestamp('1990-01-01')
                        time_values = pd.to_datetime(ds.time.values)
                        seconds_since_ref = (time_values - reference_date).total_seconds()

                        # Replace the time coordinate with seconds since reference
                        ds = ds.assign_coords(time=seconds_since_ref)

                        # Set time attributes
                        ds.time.attrs = {
                            'units': 'seconds since 1990-1-1 0:0:0.0 -0:00',
                            'calendar': 'standard',
                            'long_name': 'time since time reference (instant)'
                        }

                        # Merge with existing data
                        if merged_ds is None:
                            merged_ds = ds
                        else:
                            merged_ds = xr.merge([merged_ds, ds])

                        ds.close()

                    except Exception as e:
                        self.logger.error(f"Error processing file {src_file}: {str(e)}")
                        continue

                # Save merged data
                if merged_ds is not None:
                    # Use standardized minimal encoding (no compression, no fill values)
                    encoding = create_minimal_encoding(merged_ds)

                    # Preserve the original attributes
                    if 'summaVersion' in merged_ds.attrs:
                        global_attrs = merged_ds.attrs
                    else:
                        global_attrs = {
                            'summaVersion': '',
                            'buildTime': '',
                            'gitBranch': '',
                            'gitHash': '',
                        }

                    # Update merged dataset attributes
                    merged_ds.attrs.update(global_attrs)

                    # Save to netCDF
                    merged_ds.to_netcdf(
                        output_file,
                        encoding=encoding,
                        unlimited_dims=['time'],
                        format='NETCDF4'
                    )
                    self.logger.info(f"Successfully created merged file: {output_file}")
                    merged_ds.close()

            # Process both timestep and daily files
            process_and_merge_files(timestep_pattern, timestep_output)
            process_and_merge_files(daily_pattern, daily_output)

            self.logger.info("SUMMA output merging completed successfully")
            return mizu_in_path

        except Exception as e:
            self.logger.error(f"Error merging SUMMA outputs: {str(e)}")
            raise
