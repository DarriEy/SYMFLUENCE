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

        Updates settingsPath, outputPath, and outFilePrefix to point to
        process-specific directories instead of global directories.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier
            file_manager_name: Name of the file manager file (default: 'fileManager.txt')
        """
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

                # Update relevant paths
                updated_lines = []
                for line in lines:
                    if 'settingsPath' in line:
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
                    else:
                        updated_lines.append(line)

                # Write updated file manager
                with open(file_manager_path, 'w') as f:
                    f.writelines(updated_lines)

                self.logger.debug(
                    f"Updated file manager for process {proc_id}: {file_manager_path}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to update file manager for process {proc_id}: {e}"
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
            return self._execute_batch_mpi(tasks, worker_func, max_workers)
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

# Add symfluence src to path to ensure imports work
sys.path.append(r"{Path(__file__).parent.parent.parent.parent}") 

try:
    from {worker_module} import {worker_function}
except ImportError as e:
    logger.error(f"Failed to import worker function: {{e}}")
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

        # Distribute tasks evenly across ranks (v0.5.0 approach)
        tasks_per_rank = len(all_tasks) // size
        extra_tasks = len(all_tasks) % size
        all_results = []

        for worker_rank in range(size):
            start_idx = worker_rank * tasks_per_rank + min(worker_rank, extra_tasks)
            end_idx = start_idx + tasks_per_rank + (1 if worker_rank < extra_tasks else 0)

            if worker_rank == 0:
                my_tasks = all_tasks[start_idx:end_idx]
                logger.info(f"Rank 0: Processing {{len(my_tasks)}} tasks locally")
            else:
                worker_tasks = all_tasks[start_idx:end_idx]
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

        try:
            with open(tasks_file, 'wb') as f:
                pickle.dump(tasks, f)
            
            self._create_mpi_worker_script(worker_script, tasks_file, results_file, worker_module, worker_function)
            worker_script.chmod(0o755)
            
            num_processes = min(max_workers, self.num_processes, len(tasks))
            mpi_cmd = ['mpirun', '-n', str(num_processes), sys.executable, str(worker_script), str(tasks_file), str(results_file)]
            
            # Run MPI command
            result = subprocess.run(mpi_cmd, capture_output=True, text=True, env=os.environ.copy())

            if result.returncode != 0:
                self.logger.error(f"MPI execution failed. Stderr: {result.stderr}")
                raise RuntimeError("MPI execution failed")

            if results_file.exists():
                with open(results_file, 'rb') as f:
                    return pickle.load(f)
            else:
                raise RuntimeError("MPI results file not created")

        finally:
            # Cleanup
            for file_path in [tasks_file, results_file, worker_script]:
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except OSError:
                        pass
