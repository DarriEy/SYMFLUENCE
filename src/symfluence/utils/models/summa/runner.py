"""
SUMMA Runner Module

This module contains the SummaRunner class for executing the SUMMA
(Structure for Unifying Multiple Modeling Alternatives) model.

The SummaRunner handles model execution in various modes:
- Serial execution for single-threaded runs
- Parallel execution using SLURM job arrays
- Point simulation mode for multiple point-based simulations

Author: SYMFLUENCE Development Team
"""

# Standard library imports
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any

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
from symfluence.utils.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)


@ModelRegistry.register_runner('SUMMA', method_name='run_summa')
class SummaRunner(BaseModelRunner):
    """
    A class to run the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    This class handles the execution of the SUMMA model, including setting up paths,
    running the model, and managing log files.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        # Call base class
        super().__init__(config, logger)

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
            if self.typed_config:
                use_parallel = self.typed_config.model.summa.use_parallel if self.typed_config.model.summa else False
            else:
                use_parallel = self.config.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA', False)

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
        summa_exe = self.config.get('SUMMA_EXE')
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
        experiment_id = self.config.get('EXPERIMENT_ID')
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
            ic_command = f"{str(summa_path / summa_exe)} -m {ic_fm} -r e"

            try:
                with open(log_path / f"{site_name}_IC.log", 'w') as log_file:
                    subprocess.run(ic_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)

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
                main_command = f"{str(summa_path / summa_exe)} -m {main_fm}"

                with open(log_path / f"{site_name}_main.log", 'w') as log_file:
                    subprocess.run(main_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)

                self.logger.info(f"Completed simulation for {site_name}")

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
        This method handles GRU-based parallelization using SLURM's job array capability.
        """
        self.logger.info("Starting parallel SUMMA run with SLURM")

        # Set up paths and filenames
        summa_path = self.get_install_path('SETTINGS_SUMMA_PARALLEL_PATH', 'installs/summa/bin/')
        summa_exe = self.config.get('SETTINGS_SUMMA_PARALLEL_EXE')
        settings_path = self.get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config.get('SETTINGS_SUMMA_FILEMANAGER')

        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_log_path = self.get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_out_path = self.get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Create output and log directories if they don't exist
        summa_log_path.mkdir(parents=True, exist_ok=True)
        summa_out_path.mkdir(parents=True, exist_ok=True)

        # Get total GRU count from catchment shapefile
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config.get('DOMAIN_NAME')}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"
        subbasins_shapefile = self.project_dir / "shapefiles" / "catchment" / subbasins_name

        # Read shapefile and count unique GRU_IDs
        try:
            gdf = gpd.read_file(subbasins_shapefile)
            total_grus = len(gdf[self.config.get('CATCHMENT_SHP_GRUID')].unique())
            self.logger.info(f"Counted {total_grus} unique GRUs from shapefile: {subbasins_shapefile}")
        except Exception as e:
            self.logger.error(f"Error counting GRUs from shapefile: {str(e)}")
            raise RuntimeError(f"Failed to count GRUs from shapefile {subbasins_shapefile}: {str(e)}")

        # Logically estimate GRUs per job based on total GRU count
        grus_per_job = self._estimate_grus_per_job(total_grus)
        self.logger.info(f"Estimated optimal GRUs per job: {grus_per_job} for {total_grus} total GRUs")

        # Calculate number of array jobs needed (minimum 1)
        n_array_jobs = max(1, -(-total_grus // grus_per_job))  # Ceiling division

        self.logger.info(f"Will launch {n_array_jobs} parallel jobs with {grus_per_job} GRUs per job")

        # Create SLURM script
        slurm_script = self._create_slurm_script(
            summa_path=summa_path,
            summa_exe=summa_exe,
            settings_path=settings_path,
            filemanager=filemanager,
            summa_log_path=summa_log_path,
            summa_out_path=summa_out_path,
            total_grus=total_grus,
            grus_per_job=grus_per_job,
            n_array_jobs=n_array_jobs - 1  # SLURM arrays are 0-based
        )

        # Write SLURM script
        script_path = self.project_dir / 'run_summa_parallel.sh'
        with open(script_path, 'w') as f:
            f.write(slurm_script)

        # Make script executable
        import os
        os.chmod(script_path, 0o755)

        # Submit job
        try:
            import subprocess
            import shutil

            # Check if sbatch exists in the path
            if not shutil.which("sbatch"):
                self.logger.error("SLURM 'sbatch' command not found. Is SLURM installed on this system?")
                raise RuntimeError("SLURM 'sbatch' command not found")

            # Log the full command being executed
            cmd = f"sbatch {script_path}"
            self.logger.info(f"Executing command: {cmd}")

            # Run the command
            process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            job_id = process.stdout.strip().split()[-1]
            self.logger.info(f"Submitted SLURM array job with ID: {job_id}")

            # Backup settings if required
            if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
                backup_path = summa_out_path / "run_settings"
                self._backup_settings(settings_path, backup_path)

            # Check if we should monitor the job
            monitor_job = self.config.get('MONITOR_SLURM_JOB', True)
            if monitor_job:
                import time

                self.logger.info(f"Monitoring SLURM job {job_id}")

                # Wait for SLURM job to complete
                wait_time = 0
                max_wait_time = 3600  # 1 hour
                check_interval = 60  # 1 minute

                while wait_time < max_wait_time:
                    try:
                        result = subprocess.run(f"squeue -j {job_id}", shell=True, capture_output=True, text=True)

                        # If result only contains header, job is no longer in queue
                        if result.stdout.count('\n') <= 1:
                            self.logger.info(f"Job {job_id} no longer in queue, checking status")

                            # Check if job completed successfully
                            sacct_cmd = f"sacct -j {job_id} -o State -n | head -1"
                            state_result = subprocess.run(sacct_cmd, shell=True, capture_output=True, text=True)
                            state = state_result.stdout.strip()

                            if "COMPLETED" in state:
                                self.logger.info(f"Job {job_id} completed successfully")
                                break
                            elif "FAILED" in state or "CANCELLED" in state or "TIMEOUT" in state:
                                self.logger.error(f"Job {job_id} ended with status: {state}")
                                raise RuntimeError(f"SLURM job {job_id} failed with status: {state}")
                            else:
                                self.logger.warning(f"Job {job_id} has unknown status: {state}")
                                break
                        else:
                            pending_count = result.stdout.count("PENDING")
                            running_count = result.stdout.count("RUNNING")
                            self.logger.info(f"Job {job_id} status: {running_count} running, {pending_count} pending")
                    except subprocess.SubprocessError as e:
                        self.logger.warning(f"Error checking job status: {str(e)}")

                    # Wait before checking again
                    time.sleep(check_interval)
                    wait_time += check_interval

                if wait_time >= max_wait_time:
                    self.logger.warning(f"Maximum wait time exceeded for job {job_id}. Continuing without waiting for completion.")

            self.logger.info("SUMMA parallel run completed or continuing in background")
            return self.merge_parallel_outputs()

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing sbatch command: {str(e)}")
            self.logger.error(f"Command output: {e.stdout}")
            self.logger.error(f"Command error: {e.stderr}")
            raise RuntimeError(f"Failed to submit SLURM job. Error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in parallel SUMMA workflow: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

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
#SBATCH --job-name=Summa-{self.config.get('DOMAIN_NAME')}
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
        summa_exe = self.config.get('SUMMA_EXE')
        settings_path = self.get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config.get('SETTINGS_SUMMA_FILEMANAGER')

        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_log_path = self.get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_log_name = "summa_log.txt"

        summa_out_path = self.get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Backup settings if required
        if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
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
            with open(summa_log_path / summa_log_name, 'w') as log_file:
                # Use shell=False and pass command as list for better reliability
                subprocess.run([summa_binary, '-m', filemanager_path], check=True, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            self.logger.info("SUMMA run completed successfully")

            # Check if we need to convert lumped output for distributed routing
            domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')

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
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_output_dir = self.project_dir / "simulations" / experiment_id / "SUMMA"
        mizuroute_settings_dir = self.project_dir / "settings" / "mizuRoute"
        summa_timestep_file = summa_output_dir / f"{experiment_id}_timestep.nc"
        
        if not summa_timestep_file.exists():
            self.logger.warning(f"SUMMA timestep file not found for conversion: {summa_timestep_file}")
            return

        topology_file = mizuroute_settings_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
        if not topology_file.exists():
            self.logger.warning(f"Topology file not found for conversion: {topology_file}")
            return

        # We assume HRU ID 1 for lumped case, but could read from topology
        hru_id = 1 
        
        routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
        
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
        experiment_id = self.config.get('EXPERIMENT_ID')
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
                    # Create encoding dict for all variables
                    encoding = {
                        'time': {
                            'dtype': 'double',
                            '_FillValue': None
                        }
                    }

                    # Add encoding for all other variables
                    for var in merged_ds.data_vars:
                        encoding[var] = {'_FillValue': None}

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
