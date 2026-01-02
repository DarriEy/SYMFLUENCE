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
        Execute a batch of tasks using a process pool.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes (default: self.max_workers)

        Returns:
            List of results from each task
        """
        if max_workers is None:
            max_workers = self.max_workers

        results = []

        if max_workers == 1 or len(tasks) == 1:
            # Sequential execution
            for task in tasks:
                try:
                    result = worker_func(task)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
                    results.append({'error': str(e), 'task': task})
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(worker_func, task): task for task in tasks}

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Task failed: {e}")
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
