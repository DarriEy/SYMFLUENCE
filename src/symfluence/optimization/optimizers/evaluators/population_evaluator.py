"""
Population Evaluator for Optimization

Handles batch evaluation of parameter populations with parallel execution.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import numpy as np

from .task_builder import TaskBuilder


class PopulationEvaluator:
    """
    Evaluates populations of solutions for optimization algorithms.

    Handles:
    - Single solution evaluation
    - Population batch evaluation (single objective)
    - Population batch evaluation (multi-objective)
    - Parallel execution coordination
    - Result extraction and error handling
    """

    DEFAULT_PENALTY_SCORE = -999.0

    def __init__(
        self,
        task_builder: TaskBuilder,
        worker: Any,
        execute_batch: Callable,
        use_parallel: bool,
        num_processes: int,
        model_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize population evaluator.

        Args:
            task_builder: Task builder instance
            worker: Worker instance for evaluations
            execute_batch: Batch execution function
            use_parallel: Whether to use parallel execution
            num_processes: Number of parallel processes
            model_name: Model name (e.g., 'SUMMA', 'FUSE')
            logger: Optional logger instance
        """
        self.task_builder = task_builder
        self.worker = worker
        self.execute_batch = execute_batch
        self.use_parallel = use_parallel
        self.num_processes = num_processes
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)

    def _resolve_worker_function(self) -> Callable:
        """
        Resolve module-level worker function for MPI compatibility.

        Returns:
            Worker function callable
        """
        worker_func = None

        try:
            worker_module_name = self.worker.__class__.__module__
            worker_module = sys.modules.get(worker_module_name)

            func_name = f"_evaluate_{self.model_name.lower()}_parameters_worker"
            if worker_module and hasattr(worker_module, func_name):
                worker_func = getattr(worker_module, func_name)
                self.logger.debug(
                    f"Resolved MPI worker function from loaded module: "
                    f"{worker_module_name}.{func_name}"
                )
        except Exception as e:
            self.logger.debug(f"Dynamic worker resolution failed: {e}")

        if worker_func is None:
            worker_func = self.worker.evaluate_worker_function

        return worker_func

    def _extract_scores(
        self,
        results: List[Dict],
        n_individuals: int
    ) -> np.ndarray:
        """
        Extract fitness scores from batch results.

        Args:
            results: List of result dictionaries
            n_individuals: Number of individuals in population

        Returns:
            Array of fitness scores
        """
        fitness = np.full(n_individuals, self.DEFAULT_PENALTY_SCORE)
        valid_count = 0

        for result in results:
            idx = result.get('individual_id', 0)
            score = result.get('score')
            error = result.get('error')

            if error:
                error_str = str(error)
                self.logger.debug(f"Task {idx} full error: {error_str}")
                self.logger.warning(
                    f"Task {idx} error: {error_str[:500] if len(error_str) > 500 else error_str}"
                )

            if score is not None and not np.isnan(score):
                fitness[idx] = score
                if score != self.DEFAULT_PENALTY_SCORE:
                    valid_count += 1
            else:
                self.logger.warning(f"Task {idx} returned score={score}")

        self.logger.debug(f"Batch results: {len(results)} returned, {valid_count} valid scores")
        return fitness

    def _extract_objectives(
        self,
        results: List[Dict],
        n_individuals: int,
        n_objectives: int
    ) -> np.ndarray:
        """
        Extract objective values from batch results.

        Args:
            results: List of result dictionaries
            n_individuals: Number of individuals in population
            n_objectives: Number of objectives

        Returns:
            Array of objective values (n_individuals x n_objectives)
        """
        objectives = np.full((n_individuals, n_objectives), self.DEFAULT_PENALTY_SCORE)
        valid_count = 0

        for result in results:
            idx = result.get('individual_id', 0)
            obj = result.get('objectives')
            error = result.get('error')

            if error:
                error_str = str(error)
                self.logger.debug(f"Task {idx} full error: {error_str}")
                self.logger.warning(
                    f"Task {idx} error: {error_str[:500] if len(error_str) > 500 else error_str}"
                )

            if obj and len(obj) == n_objectives:
                objectives[idx] = np.array(obj, dtype=float)
                if not np.any(np.isnan(objectives[idx])) and objectives[idx][0] != self.DEFAULT_PENALTY_SCORE:
                    valid_count += 1
            else:
                self.logger.warning(f"Task {idx} returned objectives={obj}")

        self.logger.debug(
            f"Batch objectives: {len(results)} returned, {valid_count} valid objective sets"
        )
        return objectives

    def evaluate_solution(
        self,
        normalized_params: np.ndarray,
        proc_id: int = 0
    ) -> float:
        """
        Evaluate a single normalized parameter set.

        Args:
            normalized_params: Normalized parameters [0, 1]
            proc_id: Process ID for parallel execution

        Returns:
            Fitness score
        """
        from symfluence.optimization.workers.base_worker import WorkerTask

        params = self.task_builder.param_manager.denormalize_parameters(normalized_params)
        
        # Use TaskBuilder to ensure all model-specific paths (like mizuroute_settings_dir)
        # are correctly included in the task.
        task_data = self.task_builder.build_task(
            individual_id=0,
            params=params,
            proc_id=proc_id,
            evaluation_id="single_eval"
        )
        
        task = WorkerTask.from_legacy_dict(task_data)

        result = self.worker.evaluate(task)
        return result.score if result.score is not None else self.DEFAULT_PENALTY_SCORE

    def evaluate_population(
        self,
        population: np.ndarray,
        iteration: int = 0,
        base_random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Evaluate a population of solutions (single objective).

        Args:
            population: Array of normalized parameter sets (n_individuals x n_params)
            iteration: Current iteration number
            base_random_seed: Base random seed

        Returns:
            Array of fitness scores
        """
        n_individuals = len(population)
        fitness = np.full(n_individuals, self.DEFAULT_PENALTY_SCORE)

        if self.use_parallel and n_individuals > 1:
            # Parallel evaluation
            tasks = self.task_builder.build_population_tasks(
                population,
                iteration=iteration,
                base_random_seed=base_random_seed
            )

            worker_func = self._resolve_worker_function()
            results = self.execute_batch(tasks, worker_func)
            fitness = self._extract_scores(results, n_individuals)
        else:
            # Sequential evaluation
            for i, params_normalized in enumerate(population):
                fitness[i] = self.evaluate_solution(params_normalized, proc_id=0)

        return fitness

    def evaluate_population_objectives(
        self,
        population: np.ndarray,
        objective_names: List[str],
        iteration: int = 0,
        base_random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Evaluate a population for multiple objectives.

        Args:
            population: Array of normalized parameter sets (n_individuals x n_params)
            objective_names: Ordered list of objective metric names
            iteration: Current iteration number
            base_random_seed: Base random seed

        Returns:
            Array of objective values (n_individuals x n_objectives)
        """
        n_individuals = len(population)
        n_objectives = len(objective_names)
        objectives = np.full((n_individuals, n_objectives), self.DEFAULT_PENALTY_SCORE)

        tasks = self.task_builder.build_population_tasks(
            population,
            iteration=iteration,
            multiobjective=True,
            objective_names=objective_names,
            base_random_seed=base_random_seed
        )

        worker_func = self._resolve_worker_function()
        results = self.execute_batch(tasks, worker_func)
        objectives = self._extract_objectives(results, n_individuals, n_objectives)

        return objectives

    def evaluate_trials(
        self,
        trials: List[np.ndarray],
        trial_indices: List[int],
        iteration: int,
        base_random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Evaluate trial solutions (e.g., for DE algorithm).

        Args:
            trials: List of trial parameter arrays (normalized)
            trial_indices: Corresponding indices in population
            iteration: Current iteration number
            base_random_seed: Base random seed

        Returns:
            Array of fitness scores indexed by trial_indices
        """
        pop_size = len(trials)
        trial_fitness = np.full(pop_size, self.DEFAULT_PENALTY_SCORE)

        if self.use_parallel and len(trials) > 1:
            tasks = self.task_builder.build_trial_tasks(
                trials,
                trial_indices,
                iteration,
                base_random_seed=base_random_seed
            )

            worker_func = self._resolve_worker_function()
            results = self.execute_batch(tasks, worker_func)

            for result in results:
                idx = result.get('individual_id', 0)
                score = result.get('score')
                error = result.get('error')

                if error:
                    error_str = str(error)
                    self.logger.debug(f"Task {idx} full error: {error_str}")
                    self.logger.warning(
                        f"Task {idx} error: {error_str[:500] if len(error_str) > 500 else error_str}"
                    )

                if score is not None and not np.isnan(score):
                    if idx in trial_indices:
                        trial_idx = trial_indices.index(idx)
                        trial_fitness[trial_idx] = score
        else:
            # Sequential evaluation
            for i, trial in enumerate(trials):
                proc_id = trial_indices[i] % self.num_processes
                trial_fitness[i] = self.evaluate_solution(trial, proc_id=proc_id)

        return trial_fitness
