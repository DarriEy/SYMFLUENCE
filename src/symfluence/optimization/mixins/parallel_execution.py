"""
Parallel Execution Mixin

Provides parallel processing infrastructure for model optimization.
Handles MPI and multiprocessing-based parallel evaluation of solutions.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pickle
import subprocess # Added to fix NameError
from datetime import datetime

logger = logging.getLogger(__name__)


class ParallelExecutionMixin:
    """
    Mixin class providing parallel processing infrastructure for optimizers.

    Requires the following attributes on the class using this mixin:
    - self.config: Dict[str, Any]
    - self.logger: logging.Logger
    - self.project_dir: Path

    Provides:
    - Parallel directory setup and management
    - Task distribution across processes
    - Batch execution with process pools
    - MPI-based execution support
    """

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_processes(self) -> int:
        """Get number of processes to use for parallel execution."""
        return max(1, self.config.get('MPI_PROCESSES', 1))

    @property
    def use_parallel(self) -> bool:
        """Check if parallel execution is enabled."""
        return self.num_processes > 1

    @property
    def max_workers(self) -> int:
        """Get maximum number of worker processes."""
        return min(self.num_processes, mp.cpu_count())

    @property
    def is_mpi_run(self) -> bool:
        """Check if running under MPI."""
        return "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ

    # =========================================================================
    # Directory setup
    # =========================================================================

    def setup_parallel_processing(
        self,
        base_dir: Path,
        model_name: str,
        experiment_id: str
    ) -> Dict[int, Dict[str, Path]]:
        """
        Setup parallel processing directories for each process.

        Creates process-specific directories to avoid file conflicts during
        parallel model evaluations.

        Args:
            base_dir: Base directory for parallel processing
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier

        Returns:
            Dictionary mapping process IDs to their directory paths
        """
        parallel_dirs = {}

        for proc_id in range(self.num_processes):
            proc_dir = base_dir / f'process_{proc_id}'
            sim_dir = proc_dir / 'simulations' / experiment_id / model_name
            settings_dir = proc_dir / 'settings' / model_name
            output_dir = proc_dir / 'output'

            # Create directories
            for d in [sim_dir, settings_dir, output_dir]:
                d.mkdir(parents=True, exist_ok=True)

            parallel_dirs[proc_id] = {
                'root': proc_dir,
                'sim_dir': sim_dir,
                'settings_dir': settings_dir,
                'output_dir': output_dir,
            }

            self.logger.debug(f"Created parallel directories for process {proc_id}")

        return parallel_dirs

    def copy_base_settings(
        self,
        source_settings_dir: Path,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str
    ) -> None:
        """
        Copy base settings to each parallel process directory.

        Args:
            source_settings_dir: Source settings directory
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model
        """
        for proc_id, dirs in parallel_dirs.items():
            dest_dir = dirs['settings_dir']

            if source_settings_dir.exists():
                # Copy settings files
                for item in source_settings_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dest_dir / item.name)
                    elif item.is_dir():
                        dest_subdir = dest_dir / item.name
                        if dest_subdir.exists():
                            shutil.rmtree(dest_subdir)
                        shutil.copytree(item, dest_subdir)

                self.logger.debug(
                    f"Copied settings from {source_settings_dir} to process {proc_id}"
                )

    def update_file_managers(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str,
        experiment_id: str,
        file_manager_name: str = 'fileManager.txt'
    ) -> None:
        """
        Update file manager paths in process-specific directories.

        Updates settingsPath, outputPath, outFilePrefix, and simulation times
        to point to process-specific directories and use calibration period.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier
            file_manager_name: Name of the file manager file (default: 'fileManager.txt')
        """
        # Get calibration period from config (use spinup start to calibration end)
        cal_period = self.config.get('CALIBRATION_PERIOD', '')
        spinup_period = self.config.get('SPINUP_PERIOD', '')

        cal_start = None
        cal_end = None

        if spinup_period:
            # Use spinup start as simulation start
            spinup_parts = [p.strip() for p in spinup_period.split(',')]
            if len(spinup_parts) >= 1:
                cal_start = spinup_parts[0]

        if cal_period:
            # Use calibration end as simulation end
            cal_parts = [p.strip() for p in cal_period.split(',')]
            if len(cal_parts) >= 2:
                cal_end = cal_parts[1]

        # Adjust end time to align with forcing timestep (3-hourly CERRA ends at 21:00, not 23:00)
        if cal_end:
            try:
                forcing_timestep = self.config.get('FORCING_TIME_STEP_SIZE', 3600)
                if forcing_timestep >= 3600:  # Hourly or coarser
                    end_dt = datetime.strptime(cal_end, '%Y-%m-%d')
                    forcing_hours = forcing_timestep / 3600
                    last_hour = int(24 - (24 % forcing_hours)) - forcing_hours
                    if last_hour < 0:
                        last_hour = 0
                    # Adjust to last valid timestep
                    cal_end = end_dt.strftime('%Y-%m-%d') + f' {int(last_hour):02d}:00'
            except Exception:
                pass  # Keep original if adjustment fails

        for proc_id, dirs in parallel_dirs.items():
            file_manager_path = dirs['settings_dir'] / file_manager_name

            if not file_manager_path.exists():
                self.logger.warning(
                    f"File manager not found for process {proc_id}: {file_manager_path}"
                )
                continue

            try:
                # Read existing file manager
                with open(file_manager_path, 'r') as f:
                    lines = f.readlines()

                # Update relevant paths and times
                updated_lines = []
                for line in lines:
                    if model_name.upper() == 'HYPE' and line.startswith('resultdir'):
                        # Update HYPE results directory
                        output_path = str(dirs['output_dir']).replace('\\', '/').rstrip('/') + '/'
                        updated_lines.append(f"resultdir\t{output_path}\n")
                    elif 'settingsPath' in line:
                        # Update to process-specific settings directory
                        settings_path = str(dirs['settings_dir']).replace('\\', '/')
                        updated_lines.append(f"settingsPath         '{settings_path}/'\n")
                    elif 'outputPath' in line:
                        # Update to process-specific simulation directory
                        output_path = str(dirs['sim_dir']).replace('\\', '/')
                        updated_lines.append(f"outputPath           '{output_path}/'\n")
                    elif 'outFilePrefix' in line:
                        # Update with process-specific prefix
                        prefix = f'proc_{proc_id:02d}_{experiment_id}'
                        updated_lines.append(f"outFilePrefix        '{prefix}'\n")
                    elif 'simStartTime' in line and cal_start:
                        # Update to calibration start time (including spinup)
                        updated_lines.append(f"simStartTime         '{cal_start}'\n")
                    elif 'simEndTime' in line and cal_end:
                        # Update to calibration end time (adjusted for forcing)
                        updated_lines.append(f"simEndTime           '{cal_end}'\n")
                    else:
                        updated_lines.append(line)

                # Write updated file manager
                with open(file_manager_path, 'w') as f:
                    f.writelines(updated_lines)

                if cal_start and cal_end:
                    self.logger.info(
                        f"Updated file manager for process {proc_id} with calibration period: {cal_start} to {cal_end}"
                    )
                else:
                    self.logger.debug(
                        f"Updated file manager for process {proc_id}: {file_manager_path}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Failed to update file manager for process {proc_id}: {e}"
                )

    def update_mizuroute_controls(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str,
        experiment_id: str,
        control_file_name: str = 'mizuroute.control'
    ) -> None:
        """
        Update mizuRoute control file paths in process-specific directories.

        Updates <input_dir>, <output_dir>, <ancil_dir>, <case_name>, and <fname_qsim>
        to point to process-specific directories instead of global directories.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier
            control_file_name: Name of the control file (default: 'mizuroute.control')
        """
        for proc_id, dirs in parallel_dirs.items():
            # mizuRoute settings are typically in a subdirectory
            mizu_settings_dir = dirs['settings_dir'].parent / 'mizuRoute'
            control_file_path = mizu_settings_dir / control_file_name

            if not control_file_path.exists():
                self.logger.debug(
                    f"mizuRoute control file not found for process {proc_id}: {control_file_path}"
                )
                continue

            try:
                # Read existing control file
                with open(control_file_path, 'r') as f:
                    lines = f.readlines()

                # Construct process-specific paths
                # dirs['sim_dir'] is .../process_N/simulations/run_1/SUMMA (or other model)
                # Input dir should point to this process's SUMMA output (same as sim_dir)
                proc_summa_dir = dirs['sim_dir']

                # Output dir should be sibling to SUMMA dir: .../process_N/simulations/run_1/mizuRoute
                proc_mizu_dir = proc_summa_dir.parent / 'mizuRoute'

                # Ancil dir should point to process-specific mizuRoute settings (topology, etc.)
                proc_ancil_dir = mizu_settings_dir

                # Ensure mizuRoute simulation directory exists
                proc_mizu_dir.mkdir(parents=True, exist_ok=True)

                # Normalize paths (forward slashes, trailing slash)
                def normalize_path(path):
                    return str(path).replace('\\', '/').rstrip('/') + '/'

                input_dir = normalize_path(proc_summa_dir)
                output_dir = normalize_path(proc_mizu_dir)
                ancil_dir = normalize_path(proc_ancil_dir)
                case_name = f'proc_{proc_id:02d}_{experiment_id}'
                
                # Set model-specific filename, variable name, and timestep for mizuRoute input
                # Default timestep is 3600s (hourly), HYPE uses 86400s (daily)
                dt_qsim = self.config.get('SETTINGS_MIZU_ROUTING_DT', '3600')
                if dt_qsim in ('default', None, ''):
                    dt_qsim = '3600'
                # Default sim times use 01:00 for hourly models; HYPE overrides to 00:00 for daily
                sim_start_time = '01:00'
                sim_end_time = '23:00'

                if model_name.upper() == 'SUMMA':
                    fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
                    vname_qsim = 'averageRoutedRunoff'
                elif model_name.upper() == 'FUSE':
                    fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
                    vname_qsim = 'q_routed'
                elif model_name.upper() == 'GR':
                    domain_name = self.config.get('DOMAIN_NAME')
                    fname_qsim = f"{domain_name}_{experiment_id}_runs_def.nc"
                    vname_qsim = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
                    if vname_qsim in ('default', None, ''):
                        vname_qsim = 'q_routed'
                elif model_name.upper() == 'HYPE':
                    fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
                    vname_qsim = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
                    if vname_qsim in ('default', None, ''):
                        vname_qsim = 'q_routed'
                    # HYPE outputs daily data - override dt_qsim to 86400 (daily)
                    # Also use 00:00 for sim_start/sim_end (HYPE timestamps are at midnight)
                    dt_qsim = '86400'
                    sim_start_time = '00:00'
                    sim_end_time = '00:00'
                else:
                    fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
                    vname_qsim = 'q_routed'  # Default for other models

                # Update relevant lines
                updated_lines = []
                for line in lines:
                    if '<ancil_dir>' in line:
                        if '!' in line:
                            comment = '!' + '!'.join(line.split('!')[1:])
                            updated_lines.append(f"<ancil_dir>             {ancil_dir}    {comment}")
                        else:
                            updated_lines.append(f"<ancil_dir>             {ancil_dir}    ! Folder that contains ancillary data\n")
                    elif '<input_dir>' in line:
                        if '!' in line:
                            comment = '!' + '!'.join(line.split('!')[1:])
                            updated_lines.append(f"<input_dir>             {input_dir}    {comment}")
                        else:
                            updated_lines.append(f"<input_dir>             {input_dir}    ! Folder that contains runoff data from SUMMA\n")
                    elif '<output_dir>' in line:
                        if '!' in line:
                            comment = '!' + '!'.join(line.split('!')[1:])
                            updated_lines.append(f"<output_dir>            {output_dir}    {comment}")
                        else:
                            updated_lines.append(f"<output_dir>            {output_dir}    ! Folder that will contain mizuRoute simulations\n")
                    elif '<case_name>' in line:
                        if '!' in line:
                            comment = '!' + '!'.join(line.split('!')[1:])
                            updated_lines.append(f"<case_name>             {case_name}    {comment}")
                        else:
                            updated_lines.append(f"<case_name>             {case_name}    ! Simulation case name\n")
                    elif '<fname_qsim>' in line:
                        if '!' in line:
                            comment = '!' + '!'.join(line.split('!')[1:])
                            updated_lines.append(f"<fname_qsim>            {fname_qsim}    {comment}")
                        else:
                            updated_lines.append(f"<fname_qsim>            {fname_qsim}    ! netCDF name for {model_name} runoff\n")
                    elif '<vname_qsim>' in line:
                        # Set model-specific variable name
                        updated_lines.append(f"<vname_qsim>            {vname_qsim}    ! Variable name for {model_name} runoff\n")
                    elif '<dt_qsim>' in line:
                        # Set model-specific timestep (HYPE=daily, others=hourly)
                        updated_lines.append(f"<dt_qsim>               {dt_qsim}    ! Time interval of input runoff in seconds\n")
                    elif '<sim_start>' in line:
                        # Extract date part and update time for model compatibility
                        # Parse existing line to get the date
                        import re
                        match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                        if match:
                            sim_date = match.group(1)
                            updated_lines.append(f"<sim_start>             {sim_date} {sim_start_time}    ! Time of simulation start\n")
                        else:
                            updated_lines.append(line)
                    elif '<sim_end>' in line:
                        # Extract date part and update time for model compatibility
                        import re
                        match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                        if match:
                            sim_date = match.group(1)
                            updated_lines.append(f"<sim_end>               {sim_date} {sim_end_time}    ! Time of simulation end\n")
                        else:
                            updated_lines.append(line)
                    else:
                        updated_lines.append(line)

                # Write updated control file
                with open(control_file_path, 'w') as f:
                    f.writelines(updated_lines)

                self.logger.debug(
                    f"Updated mizuRoute control file for process {proc_id}: {control_file_path}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to update mizuRoute control file for process {proc_id}: {e}"
                )

    def cleanup_parallel_processing(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]]
    ) -> None:
        """
        Cleanup parallel processing directories.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
        """
        for proc_id, dirs in parallel_dirs.items():
            root_dir = dirs.get('root')
            if root_dir and root_dir.exists():
                try:
                    shutil.rmtree(root_dir)
                    self.logger.debug(f"Cleaned up parallel directory for process {proc_id}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to cleanup parallel directory for process {proc_id}: {e}"
                    )

    # =========================================================================
    # Task distribution
    # =========================================================================

    def distribute_tasks(
        self,
        tasks: List[Dict[str, Any]],
        parallel_dirs: Optional[Dict[int, Dict[str, Path]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Distribute tasks across processes.

        Assigns each task to a process and updates the task with
        process-specific directory paths.

        Args:
            tasks: List of task dictionaries
            parallel_dirs: Optional process-specific directories

        Returns:
            List of tasks with process assignments
        """
        distributed_tasks = []

        for i, task in enumerate(tasks):
            proc_id = i % self.num_processes
            task_copy = task.copy()
            task_copy['proc_id'] = proc_id

            if parallel_dirs and proc_id in parallel_dirs:
                dirs = parallel_dirs[proc_id]
                task_copy['proc_settings_dir'] = str(dirs['settings_dir'])
                task_copy['proc_sim_dir'] = str(dirs['sim_dir'])
                task_copy['proc_output_dir'] = str(dirs['output_dir'])

            distributed_tasks.append(task_copy)

        return distributed_tasks

    # =========================================================================
    # Batch execution
    # =========================================================================

    def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks using MPI if parallel, otherwise sequentially.
        """
        if max_workers is None:
            max_workers = self.max_workers

        if self.use_parallel and len(tasks) > 1:
            # Parallel execution via MPI
            try:
                return self._execute_batch_mpi(tasks, worker_func, max_workers)
            except Exception as e:
                self.logger.error(f"MPI batch execution failed: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Return empty results with errors for all tasks
                return [{'individual_id': task.get('individual_id', i), 'score': None, 'error': str(e)}
                        for i, task in enumerate(tasks)]
        else:
            # Sequential execution for a single process or single task
            results = []
            for task in tasks:
                try:
                    result = worker_func(task)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Task failed in sequential execution: {e}")
                    results.append({'error': str(e), 'task': task})
            return results

    def execute_batch_ordered(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks and return results in the same order as input.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results in the same order as input tasks
        """
        if max_workers is None:
            max_workers = self.max_workers

        if max_workers == 1 or len(tasks) == 1:
            return [worker_func(task) for task in tasks]

        # Use ProcessPoolExecutor.map to preserve order
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(worker_func, tasks))

        return results

    # =========================================================================
    # Environment setup
    # =========================================================================

    def setup_worker_environment(self) -> Dict[str, str]:
        """
        Setup environment variables for worker processes.

        Returns:
            Dictionary of environment variables to set
        """
        return {
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'VECLIB_MAXIMUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'NETCDF_DISABLE_LOCKING': '1',
            'HDF5_USE_FILE_LOCKING': 'FALSE',
            'HDF5_DISABLE_VERSION_CHECK': '1',
        }

    def apply_worker_environment(self) -> None:
        """Apply worker environment variables to current process."""
        for key, value in self.setup_worker_environment().items():
            os.environ[key] = value

    def _create_mpi_worker_script(self, script_path: Path, tasks_file: Path, results_file: Path, worker_module: str, worker_function: str) -> None:
        """Create the MPI worker script file."""
        # Calculate the correct path to the src directory (absolute, not relative)
        # This file is in: src/symfluence/optimization/mixins/parallel_execution.py
        # Path(__file__).parent = src/symfluence/optimization/mixins
        # .parent.parent.parent.parent = src/ (the directory we want for PYTHONPATH)
        src_path = Path(__file__).parent.parent.parent.parent

        script_content = f'''#!/usr/bin/env python3
import sys
import pickle
import os
from pathlib import Path
from mpi4py import MPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# Silence noisy libraries
for noisy_logger in ['rasterio', 'fiona', 'boto3', 'botocore', 'matplotlib', 'urllib3', 's3transfer']:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Add symphluence src to path to ensure imports work

sys.path.insert(0, r"{str(src_path)}")

try:
    from {worker_module} import {worker_function}
except ImportError as e:
    logger.error(f"Failed to import worker function: {{e}}")
    logger.error(f"sys.path = {{sys.path}}")
    sys.exit(1)

def main():
    """MPI worker main function."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    tasks_file = Path(sys.argv[1])
    results_file = Path(sys.argv[2])
    
    if rank == 0:
        # Master process - load all tasks
        try:
            with open(tasks_file, 'rb') as f:
                all_tasks = pickle.load(f)
        except Exception as e:
            logger.error(f"Master failed to load tasks: {{e}}")
            all_tasks = []

        logger.info(f"Rank 0: Loaded {{len(all_tasks)}} tasks")

        # Distribute tasks by proc_id to avoid race conditions
        # Tasks with the same proc_id share directories, so they must run on the same rank
        from collections import defaultdict
        tasks_by_proc = defaultdict(list)
        for task in all_tasks:
            proc_id = task.get('proc_id', 0)
            # Assign to rank based on proc_id modulo size
            assigned_rank = proc_id % size
            tasks_by_proc[assigned_rank].append(task)

        logger.info(f"Rank 0: Distributed tasks by proc_id - {{{{r: len(tasks_by_proc[r]) for r in range(size)}}}}")
        all_results = []

        for worker_rank in range(size):
            worker_tasks = tasks_by_proc[worker_rank]

            if worker_rank == 0:
                my_tasks = worker_tasks
                logger.info(f"Rank 0: Processing {{len(my_tasks)}} tasks locally")
            else:
                logger.info(f"Rank 0: Sending {{len(worker_tasks)}} tasks to rank {{worker_rank}}")
                comm.send(worker_tasks, dest=worker_rank, tag=1)

        # Process rank 0 tasks
        for i, task in enumerate(my_tasks):
            try:
                worker_result = {worker_function}(task)
                all_results.append(worker_result)
            except Exception as e:
                logger.error(f"Rank 0: Task {{i}} failed: {{e}}")
                error_result = {{
                    'individual_id': task.get('individual_id', -1),
                    'params': task.get('params', {{}}),
                    'score': None,
                    'error': f'Rank 0 error: {{str(e)}}'
                }}
                all_results.append(error_result)

        # Collect results from workers
        for worker_rank in range(1, size):
            try:
                logger.info(f"Rank 0: Waiting for results from rank {{worker_rank}}")
                worker_results = comm.recv(source=worker_rank, tag=2)
                logger.info(f"Rank 0: Received {{len(worker_results)}} results from rank {{worker_rank}}")
                all_results.extend(worker_results)
            except Exception as e:
                logger.error(f"Error receiving from worker {{worker_rank}}: {{e}}")

        # Save results
        logger.info(f"Rank 0: Saving {{len(all_results)}} results to {{results_file}}")
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"Rank 0: Results saved successfully")

    else:
        # Worker process
        logger.info(f"Rank {{rank}}: Waiting for tasks from rank 0")
        try:
            my_tasks = comm.recv(source=0, tag=1)
            logger.info(f"Rank {{rank}}: Received {{len(my_tasks)}} tasks")

            my_results = []

            for i, task in enumerate(my_tasks):
                logger.info(f"Rank {{rank}}: Processing task {{i+1}}/{{len(my_tasks)}}")
                try:
                    worker_result = {worker_function}(task)
                    my_results.append(worker_result)
                except Exception as e:
                    logger.error(f"Rank {{rank}}: Task {{i}} failed: {{e}}")
                    error_result = {{
                        'individual_id': task.get('individual_id', -1),
                        'params': task.get('params', {{}}),
                        'score': None,
                        'error': f'Rank {{rank}} error: {{str(e)}}'
                    }}
                    my_results.append(error_result)

            logger.info(f"Rank {{rank}}: Sending {{len(my_results)}} results back to rank 0")
            comm.send(my_results, dest=0, tag=2)
            logger.info(f"Rank {{rank}}: Results sent successfully")

        except Exception as e:
            logger.error(f"Worker {{rank}} failed: {{e}}")

if __name__ == "__main__":
    main()
'''
        with open(script_path, 'w') as f:
            f.write(script_content)

    def _execute_batch_mpi(self, tasks: List[Dict[str, Any]], worker_func: Callable, max_workers: int) -> List[Dict[str, Any]]:
        """Execute a batch of tasks using mpirun."""
        import uuid
        import sys

        work_dir = self.project_dir / "temp_mpi"
        work_dir.mkdir(exist_ok=True)

        unique_id = uuid.uuid4().hex[:8]
        tasks_file = work_dir / f'mpi_tasks_{unique_id}.pkl'
        results_file = work_dir / f'mpi_results_{unique_id}.pkl'
        worker_script = work_dir / f'mpi_worker_{unique_id}.py'

        # Get worker module and function name from the callable
        if hasattr(worker_func, '__module__'):
            worker_module = worker_func.__module__
            worker_function = worker_func.__name__
        else:
            # Fallback to defaults
            worker_module = "symfluence.optimization.workers.summa_parallel_workers"
            worker_function = "_evaluate_parameters_worker_safe"

        cleanup_files = True  # Track whether to clean up files
        try:
            self.logger.debug(f"MPI batch: {len(tasks)} tasks, worker={worker_module}.{worker_function}")

            with open(tasks_file, 'wb') as f:
                pickle.dump(tasks, f)

            self._create_mpi_worker_script(worker_script, tasks_file, results_file, worker_module, worker_function)
            worker_script.chmod(0o755)

            num_processes = min(max_workers, self.num_processes, len(tasks))

            # Use the Python executable from sys.executable (should be the venv Python)
            # If sys.executable is not found or doesn't have mpi4py, try to detect the venv
            python_exe = sys.executable
            self.logger.debug(f"sys.executable: {python_exe}, exists: {Path(python_exe).exists()}")

            # Always prefer the venv Python if it exists
            venv_paths = [
                Path(__file__).parent.parent.parent.parent.parent / "venv" / "bin" / "python",
                Path(__file__).parent.parent.parent.parent.parent / "venv" / "bin" / "python3",
                Path(__file__).parent.parent.parent.parent.parent / "venv" / "bin" / "python3.11",
                Path.home() / "venv" / "bin" / "python",
                Path.home() / "venv" / "bin" / "python3",
                Path.home() / "venv" / "bin" / "python3.11",
            ]
            for venv_path in venv_paths:
                if venv_path.exists():
                    python_exe = str(venv_path)
                    self.logger.info(f"Using venv Python: {python_exe}")
                    break

            if not Path(python_exe).exists():
                self.logger.warning(f"Python executable not found: {python_exe}, using sys.executable")

            # Don't use -x for PYTHONPATH since venv Python already includes src in sys.path
            # Use -x for worker environment variables that control threading and HDF5
            mpi_cmd = ['mpirun', '-x', 'OMP_NUM_THREADS', '-x', 'HDF5_USE_FILE_LOCKING', '-x', 'MKL_NUM_THREADS',
                       '-n', str(num_processes), python_exe, str(worker_script), str(tasks_file), str(results_file)]

            self.logger.debug(f"MPI command: {' '.join(mpi_cmd)}")

            # Setup environment for MPI workers
            mpi_env = os.environ.copy()

            # Ensure PYTHONPATH includes the src directory (same path as in script)
            src_path = str(Path(__file__).parent.parent.parent.parent)
            current_pythonpath = mpi_env.get('PYTHONPATH', '')
            if current_pythonpath:
                mpi_env['PYTHONPATH'] = f"{src_path}:{current_pythonpath}"
            else:
                mpi_env['PYTHONPATH'] = src_path

            # Add worker environment variables for thread control and HDF5 locking
            worker_env = self.setup_worker_environment()
            mpi_env.update(worker_env)

            # Ensure OpenMPI passes environment variables to spawned processes
            # This is important for PYTHONPATH and other settings
            if 'OMPI_MCA_' not in mpi_env:
                mpi_env['OMPI_MCA_pls_rsh_agent'] = 'ssh'

            self.logger.debug(f"MPI environment - PYTHONPATH: {mpi_env.get('PYTHONPATH')}")
            self.logger.debug(f"MPI command: {' '.join(mpi_cmd)}")

            # Run MPI command
            result = subprocess.run(mpi_cmd, capture_output=True, text=True, env=mpi_env)

            # Always log MPI output for debugging
            self.logger.debug(f"MPI returncode: {result.returncode}")
            
            # Check for diagnostic markers in stdout/stderr and force print if found
            for output in [result.stdout, result.stderr]:
                if output:
                    if "[TWS]" in output or "[WORKER DIAG]" in output:
                        # Extract and log lines with markers
                        for line in output.split('\n'):
                            if "[TWS]" in line or "[WORKER DIAG]" in line:
                                print(f"  > {line}", flush=True)

            if result.stdout:
                self.logger.debug(f"MPI stdout: {result.stdout[:1000]}")
            if result.stderr:
                # Still log stderr if there's significant output, but maybe keep it at debug if it's just expected warnings
                self.logger.debug(f"MPI stderr: {result.stderr[:1000]}")

            if result.returncode != 0:
                self.logger.error(f"MPI execution failed (returncode={result.returncode})")
                self.logger.error(f"MPI stdout: {result.stdout[:2000] if result.stdout else 'empty'}")
                self.logger.error(f"MPI stderr: {result.stderr[:2000] if result.stderr else 'empty'}")
                cleanup_files = False  # Keep files for debugging
                raise RuntimeError(f"MPI execution failed with returncode {result.returncode}")

            if results_file.exists():
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
                self.logger.debug(f"MPI completed: {len(results)} results")
                return results
            else:
                self.logger.error(f"MPI results file not created: {results_file}")
                self.logger.error(f"MPI stdout: {result.stdout[:2000] if result.stdout else 'empty'}")
                self.logger.error(f"MPI stderr: {result.stderr[:2000] if result.stderr else 'empty'}")
                cleanup_files = False  # Keep files for debugging
                raise RuntimeError("MPI results file not created")

        finally:
            # Cleanup only if successful
            if cleanup_files:
                for file_path in [tasks_file, results_file, worker_script]:
                    if file_path.exists():
                        try:
                            file_path.unlink()
                        except OSError:
                            pass
