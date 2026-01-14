"""
Dynamically Dimensioned Search (DDS) optimizer implementation.

DDS is an efficient single-solution optimization algorithm designed for
calibration problems with many parameters and limited function evaluations.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer

class DDSOptimizer(BaseOptimizer):
    """Dynamically Dimensioned Search (DDS) Optimizer for SYMFLUENCE.

    DDS is a greedy, single-solution search algorithm that adapts its search
    neighborhood based on the remaining function evaluation budget. Early in the
    search, it perturbs many dimensions to explore globally; later it focuses on
    fewer dimensions for local refinement.

    Algorithm Overview:
        DDS solves: maximize score (or minimize error)
        1. Start with initial parameter set
        2. For each iteration:
           a. Calculate probability of perturbing each dimension
              prob_select = 1 - log(iter) / log(max_iterations)
              (decreases from ~1.0 to ~1/num_params as iterations progress)
           b. For each dimension, perturb with probability prob_select
              perturbation ~ Normal(0, dds_r) where dds_r is step size
              Apply reflective boundary to keep in [0,1]
           c. Evaluate new solution
           d. Accept if better (greedy), else revert
        3. Track best solution found so far

    Why DDS Works Well for Calibration:
        - Trades global exploration for local refinement dynamically
        - Early iterations explore widely (high prob_select)
        - Later iterations refine promising region (low prob_select)
        - Single solution is memory-efficient for high-dimensional problems
        - Guaranteed to make progress (greedy acceptance)
        - No stochastic population, highly reproducible

    Key Features:
        - Efficient for high-dimensional problems with limited evaluations
        - Automatic adaptation of search intensity
        - Optional multi-start parallel execution for improved robustness
        - Greedy (hill-climbing) acceptance strategy
        - Reflective boundary handling for parameter bounds

    Configuration Parameters:
        DDS_R: Step size for perturbations (default: 0.2)
               Controls magnitude of random changes
               Typical range: 0.1-0.5
               Larger values = more aggressive exploration
        NUMBER_OF_ITERATIONS: Total budget of function evaluations
        MPI_PROCESSES: For multi-start parallel execution (> 1 uses parallel)

    Parallel Mode (Multi-Start DDS):
        When MPI_PROCESSES > 1:
        - Divides iterations equally among parallel processes
        - Each process runs independent DDS from different random starting point
        - Returns best solution found across all starts
        - Improves robustness for noisy objectives

    Workflow:
        1. Initialize single solution from initial parameters
        2. If parallel: spawn multiple DDS instances, each from different start
        3. If serial: run single DDS until max_iterations
        4. Return best parameters found

    References:
        Tolson, B.A. and C.A. Shoemaker (2007), Dynamically dimensioned search
        algorithm for computationally efficient watershed model calibration,
        Water Resources Research, 43, W01413.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize DDS optimizer.

        Args:
            config: Configuration dictionary with DDS_R and optimization parameters
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.dds_r = self._cfg(
            'DDS_R', default=0.2,
            typed=lambda: self._config.optimization.dds.r if self._config and self._config.optimization.dds else None
        )
        self.population: Optional[np.ndarray] = None
        self.population_scores: Optional[np.ndarray] = None

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "DDS"

    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run DDS algorithm with optional multi-start parallel support.

        Dispatches to either single or parallel DDS based on MPI_PROCESSES.

        Returns:
            Tuple of (best_params, best_score, iteration_history)
        """
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_dds(initial_params)

        if self.use_parallel and self.num_processes > 1:
            return self._run_multi_start_parallel_dds()
        else:
            return self._run_single_dds()

    def _initialize_dds(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize DDS with a single solution"""
        param_count = len(self.parameter_manager.all_param_names)
        self.population = np.random.random((1, param_count))
        self.population_scores = np.full(1, np.nan, dtype=float)

        assert self.population is not None
        assert self.population_scores is not None

        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.population[0] = np.clip(initial_normalized, 0, 1)

        self.population_scores[0] = self._evaluate_individual(self.population[0])

        if not np.isnan(self.population_scores[0]):
            self.best_score = self.population_scores[0]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[0])
        else:
            self.best_score = float('-inf')
            self.best_params = initial_params

        self._record_generation(0)

    def _run_single_dds(self) -> Tuple[Dict, float, List]:
        """Run single-instance DDS algorithm (main optimization loop).

        Core DDS algorithm that:
        1. Starts with current solution (from initialization)
        2. For each iteration:
           - Calculate adaptive probability: prob_select = 1 - log(iter)/log(max_iter)
           - Probabilistically select dimensions to perturb
           - Perturb selected dimensions using Gaussian noise
           - Apply reflective boundary to enforce [0,1] bounds
           - Evaluate new trial solution
           - Accept if score improves, else stay with current solution
           - Track best and current solutions separately
        3. Return best solution found

        Adaptive Probability (key innovation):
            - Iteration 1: prob_select ≈ 1.0 (perturb many dimensions, global search)
            - Mid iterations: prob_select ≈ 0.5 (balanced exploration/refinement)
            - Final iterations: prob_select ≈ 0.1-0.3 (refine few dimensions locally)
            This automatic adaptation requires no parameter tuning!

        Perturbation Strategy:
            - perturbation ~ Normal(0, dds_r)
            - Reflective boundary: if x < 0, use -x; if x > 1, use 2-x
            - Ensures parameters stay in valid [0,1] range

        Side Effects:
            - Updates self.population[0] if trial improves
            - Updates self.best_params and self.best_score if trial beats best
            - Appends iteration records to self.iteration_history

        Returns:
            Tuple of (best_params, best_score, iteration_history)
        """
        assert self.population is not None
        assert self.population_scores is not None
        assert self.best_score is not None

        current_solution = self.population[0].copy()
        current_score = self.population_scores[0]
        num_params = len(self.parameter_manager.all_param_names)

        for iteration in range(1, self.max_iterations + 1):
            prob_select = 1.0 - np.log(iteration) / np.log(self.max_iterations) if self.max_iterations > 1 else 0.5
            prob_select = max(prob_select, 1.0 / num_params)

            trial_solution = current_solution.copy()
            variables_to_perturb = np.random.random(num_params) < prob_select
            if not np.any(variables_to_perturb):
                variables_to_perturb[np.random.randint(0, num_params)] = True

            for i in range(num_params):
                if variables_to_perturb[i]:
                    perturbation = np.random.normal(0, self.dds_r)
                    trial_solution[i] = current_solution[i] + perturbation
                    if trial_solution[i] < 0: trial_solution[i] = -trial_solution[i]
                    elif trial_solution[i] > 1: trial_solution[i] = 2.0 - trial_solution[i]
                    trial_solution[i] = np.clip(trial_solution[i], 0, 1)

            trial_score = self._evaluate_individual(trial_solution)

            improvement = False
            if trial_score > current_score:
                current_solution = trial_solution.copy()
                current_score = trial_score
                improvement = True
                self.population[0] = current_solution.copy()
                self.population_scores[0] = current_score
                if trial_score > self.best_score:
                    self.best_score = trial_score
                    self.best_params = self.parameter_manager.denormalize_parameters(trial_solution)

            self._record_dds_generation(iteration, current_score, trial_score, improvement,
                                      np.sum(variables_to_perturb), prob_select)

        return self.best_params, self.best_score, self.iteration_history

    def _run_multi_start_parallel_dds(self) -> Tuple[Dict, float, List]:
        """Multi-start parallel DDS"""
        num_starts = self.num_processes
        iterations_per_start = max(1, self.max_iterations // num_starts)

        initial_params = self.parameter_manager.get_initial_parameters()
        base_normalized = self.parameter_manager.normalize_parameters(initial_params)
        param_count = len(self.parameter_manager.all_param_names)

        dds_tasks = []
        for start_id in range(num_starts):
            if start_id == 0: starting_solution = base_normalized.copy()
            else:
                noise = np.random.normal(0, 0.15, param_count)
                starting_solution = np.clip(base_normalized + noise, 0, 1)

            task = {
                'start_id': start_id,
                'starting_solution': starting_solution,
                'max_iterations': iterations_per_start,
                'dds_r': self.dds_r,
                'random_seed': start_id * 12345,
                'proc_id': start_id % len(self.parallel_dirs),
                'evaluation_id': f"multi_dds_{start_id:02d}"
            }
            dds_tasks.append(task)

        from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenProcessPool
        from symfluence.optimization.workers.summa_parallel_workers import _run_dds_instance_worker

        results = []
        try:
            with ProcessPoolExecutor(max_workers=min(len(dds_tasks), self.num_processes)) as executor:
                try:
                    future_to_task = {executor.submit(_run_dds_instance_worker, self._prepare_dds_worker_data(task)): task for task in dds_tasks}
                    for future in as_completed(future_to_task):
                        results.append(future.result())
                except BrokenProcessPool as e:
                    self.logger.warning(f"Process pool was broken during multi-start DDS: {str(e)}. Falling back to sequential execution.")
                    # Fallback to sequential execution
                    for task in dds_tasks:
                        try:
                            result = _run_dds_instance_worker(self._prepare_dds_worker_data(task))
                            results.append(result)
                        except Exception as task_error:
                            self.logger.error(f"Error in sequential fallback for task {task.get('start_id')}: {str(task_error)}")
                            results.append({'best_score': None, 'best_params': None})
        except Exception as e:
            self.logger.error(f"Critical error in multi-start parallel DDS: {str(e)}")
            # Fallback to sequential execution
            for task in dds_tasks:
                try:
                    result = _run_dds_instance_worker(self._prepare_dds_worker_data(task))
                    results.append(result)
                except Exception as task_error:
                    self.logger.error(f"Error in sequential fallback for task {task.get('start_id')}: {str(task_error)}")
                    results.append({'best_score': None, 'best_params': None})

        valid_results = [r for r in results if r.get('best_score') is not None]
        if not valid_results:
            self.logger.error("No valid results from multi-start parallel DDS")
            return self.best_params, self.best_score, self.iteration_history

        best_result = max(valid_results, key=lambda x: x['best_score'])

        self.best_score = best_result['best_score']
        self.best_params = best_result['best_params']
        return self.best_params, self.best_score, self.iteration_history

    def _prepare_dds_worker_data(self, task: Dict) -> Dict:
        """Prepare data for DDS instance worker"""
        proc_dirs = self.parallel_dirs[task['proc_id']] if hasattr(self, 'parallel_dirs') and task['proc_id'] < len(self.parallel_dirs) else {}
        # When parallel_dirs is empty (MPI_PROCESSES=1), use output_dir as the SUMMA output location
        # This ensures the metrics calculation looks for files where SUMMA actually writes them
        fallback_summa_dir = self.output_dir
        fallback_mizuroute_dir = self.output_dir / "mizuRoute"
        summa_fm_name = self._cfg('SETTINGS_SUMMA_FILEMANAGER', default='fileManager.txt')
        return {
            # Use config_dict for worker compatibility (workers expect dict.get())
            'dds_task': task, 'config': self.config_dict, 'target_metric': self.target_metric,
            'param_bounds': self.parameter_manager.param_bounds, 'all_param_names': self.parameter_manager.all_param_names,
            'summa_exe': str(self._get_summa_exe_path()),
            'summa_dir': str(proc_dirs.get('summa_dir', fallback_summa_dir)),
            'mizuroute_dir': str(proc_dirs.get('mizuroute_dir', fallback_mizuroute_dir)),
            'summa_settings_dir': str(proc_dirs.get('summa_settings_dir', self.optimization_settings_dir)),
            'mizuroute_settings_dir': str(proc_dirs.get('mizuroute_settings_dir', self.optimization_dir / "settings" / "mizuRoute")),
            'file_manager': str(proc_dirs.get('summa_settings_dir', self.optimization_settings_dir) / summa_fm_name)
        }

    def _record_generation(self, generation: int) -> None:
        assert self.population_scores is not None
        valid_scores = self.population_scores[~np.isnan(self.population_scores)]
        self.iteration_history.append({
            'generation': generation, 'algorithm': 'DDS', 'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params else None,
            'valid_individuals': len(valid_scores)
        })

    def _record_dds_generation(self, iteration: int, current_score: float, trial_score: float,
                              improvement: bool, num_perturbed: int, prob_select: float) -> None:
        self.iteration_history.append({
            'generation': iteration, 'algorithm': 'DDS', 'best_score': self.best_score,
            'current_score': current_score, 'trial_score': trial_score, 'improvement': improvement,
            'num_variables_perturbed': num_perturbed, 'selection_probability': prob_select
        })
