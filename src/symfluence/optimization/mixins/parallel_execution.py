"""
Parallel Execution Mixin

Provides parallel processing infrastructure for model optimization.
Handles MPI and multiprocessing-based parallel evaluation of solutions.

This module has been refactored to delegate to specialized classes:
- DirectoryManager: Manages parallel processing directories
- ConfigurationUpdater: Updates model config files for each process
- TaskDistributor: Distributes tasks across processes
- WorkerEnvironmentConfig: Manages worker environment variables
- ExecutionStrategy: Abstract interface for execution strategies
  - SequentialExecutionStrategy: Sequential execution
  - ProcessPoolExecutionStrategy: Python multiprocessing
  - MPIExecutionStrategy: MPI-based distributed execution
"""

import os
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from .parallel import (
    DirectoryManager,
    ConfigurationUpdater,
    TaskDistributor,
    WorkerEnvironmentConfig,
    SequentialExecutionStrategy,
    ProcessPoolExecutionStrategy,
    MPIExecutionStrategy,
)

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
    # Lazy initialization of helper classes
    # =========================================================================

    @property
    def _directory_manager(self) -> DirectoryManager:
        """Get or create directory manager."""
        if not hasattr(self, '__directory_manager'):
            self.__directory_manager = DirectoryManager(self.logger)
        return self.__directory_manager

    @property
    def _config_updater(self) -> ConfigurationUpdater:
        """Get or create configuration updater."""
        if not hasattr(self, '__config_updater'):
            self.__config_updater = ConfigurationUpdater(self.config, self.logger)
        return self.__config_updater

    @property
    def _task_distributor(self) -> TaskDistributor:
        """Get or create task distributor."""
        if not hasattr(self, '__task_distributor'):
            self.__task_distributor = TaskDistributor(self.num_processes)
        return self.__task_distributor

    @property
    def _worker_env_config(self) -> WorkerEnvironmentConfig:
        """Get or create worker environment config."""
        if not hasattr(self, '__worker_env_config'):
            self.__worker_env_config = WorkerEnvironmentConfig()
        return self.__worker_env_config

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
    # Directory setup (delegates to DirectoryManager)
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
        return self._directory_manager.setup_parallel_directories(
            base_dir, model_name, experiment_id, self.num_processes
        )

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
        self._directory_manager.copy_base_settings(
            source_settings_dir, parallel_dirs, model_name
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
        self._directory_manager.cleanup(parallel_dirs)

    # =========================================================================
    # Configuration updates (delegates to ConfigurationUpdater)
    # =========================================================================

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
        self._config_updater.update_file_managers(
            parallel_dirs, model_name, experiment_id, file_manager_name
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
        self._config_updater.update_mizuroute_controls(
            parallel_dirs, model_name, experiment_id, control_file_name
        )

    # =========================================================================
    # Task distribution (delegates to TaskDistributor)
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
        return self._task_distributor.distribute(tasks, parallel_dirs)

    # =========================================================================
    # Batch execution (uses execution strategies)
    # =========================================================================

    def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks using MPI if parallel, otherwise sequentially.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results from task execution
        """
        if max_workers is None:
            max_workers = self.max_workers

        if self.use_parallel and len(tasks) > 1:
            # Parallel execution via MPI
            try:
                strategy = MPIExecutionStrategy(
                    self.project_dir, self.num_processes, self.logger
                )
                return strategy.execute(tasks, worker_func, max_workers)
            except Exception as e:
                self.logger.error(f"MPI batch execution failed: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Return empty results with errors for all tasks
                return [
                    {
                        'individual_id': task.get('individual_id', i),
                        'score': None,
                        'error': str(e)
                    }
                    for i, task in enumerate(tasks)
                ]
        else:
            # Sequential execution for a single process or single task
            strategy = SequentialExecutionStrategy(self.logger)
            return strategy.execute(tasks, worker_func, max_workers)

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

        strategy = ProcessPoolExecutionStrategy(self.logger)
        return strategy.execute(tasks, worker_func, max_workers)

    # =========================================================================
    # Environment setup (delegates to WorkerEnvironmentConfig)
    # =========================================================================

    def setup_worker_environment(self) -> Dict[str, str]:
        """
        Setup environment variables for worker processes.

        Returns:
            Dictionary of environment variables to set
        """
        return self._worker_env_config.get_environment()

    def apply_worker_environment(self) -> None:
        """Apply worker environment variables to current process."""
        self._worker_env_config.apply_to_current_process()

    # =========================================================================
    # Legacy method for backward compatibility
    # =========================================================================

    def _create_mpi_worker_script(
        self,
        script_path: Path,
        tasks_file: Path,
        results_file: Path,
        worker_module: str,
        worker_function: str
    ) -> None:
        """
        Create the MPI worker script file.

        Note: This method is kept for backward compatibility.
        New code should use MPIExecutionStrategy directly.
        """
        strategy = MPIExecutionStrategy(
            self.project_dir, self.num_processes, self.logger
        )
        strategy._create_worker_script(
            script_path, tasks_file, results_file, worker_module, worker_function
        )
