# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Task Distributor

Handles distribution of optimization tasks across parallel processes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskDistributor:
    """
    Distributes tasks across parallel processes.

    Assigns each task to a process ID and optionally injects
    process-specific directory paths into the task dictionary.
    """

    def __init__(self, num_processes: int):
        """
        Initialize task distributor.

        Args:
            num_processes: Number of parallel processes available
        """
        self.num_processes = max(1, num_processes)

    def distribute(
        self,
        tasks: List[Dict[str, Any]],
        parallel_dirs: Optional[Dict[int, Dict[str, Path]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Distribute tasks across processes.

        Assigns each task to a process using round-robin distribution
        and updates the task with process-specific directory paths.

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

    def group_by_process(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group tasks by their assigned process ID.

        Args:
            tasks: List of tasks with 'proc_id' assigned

        Returns:
            Dictionary mapping process IDs to their tasks
        """
        from collections import defaultdict
        tasks_by_proc = defaultdict(list)

        for task in tasks:
            proc_id = task.get('proc_id', 0)
            tasks_by_proc[proc_id].append(task)

        return dict(tasks_by_proc)
