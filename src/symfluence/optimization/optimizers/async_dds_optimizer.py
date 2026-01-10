"""
Asynchronous parallel DDS optimizer using MPI batch framework.

Extends DDS for parallel execution across multiple processors,
enabling efficient calibration on HPC systems.
"""

import numpy as np
import logging
import time
import random
from typing import Dict, Any, List, Tuple, Optional
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer

class AsyncDDSOptimizer(BaseOptimizer):
    """
    Asynchronous Parallel DDS using existing MPI batch framework
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.dds_r = config.get('DDS_R', 0.2)
        self.pool_size = config.get('ASYNC_DDS_POOL_SIZE', min(20, self.num_processes * 2))
        self.batch_size = config.get('ASYNC_DDS_BATCH_SIZE', self.num_processes)
        self.total_target_evaluations = self.max_iterations * self.num_processes
        self.target_batches = self.total_target_evaluations // self.batch_size
        # Stagnation limit: stop if no improvement for this many consecutive batches
        # Default 25 is more appropriate for 4-8 parameter problems
        self.stagnation_limit = config.get('DDS_STAGNATION_LIMIT', 25)
        # Convergence threshold: consider converged if improvement < this
        self.convergence_threshold = config.get('DDS_CONVERGENCE_THRESHOLD', 1e-4)
        self.solution_pool = []; self.pool_scores = []; self.batch_history = []
        self.total_evaluations = 0; self.stagnation_counter = 0
    
    def get_algorithm_name(self) -> str:
        return "AsyncDDS_MPI"
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        self.logger.info("Starting Asynchronous Parallel DDS")
        self.logger.info(f"  Stagnation limit: {self.stagnation_limit} batches")
        self._initialize_solution_pool()
        batch_num = 0; start_time = time.time()
        while batch_num < self.target_batches and self.stagnation_counter < self.stagnation_limit:
            tasks = self._generate_batch_from_pool(batch_num)
            if not tasks: break
            results = self._run_parallel_evaluations(tasks) if self.use_parallel else self._run_sequential_batch(tasks)
            improvements = self._update_pool_with_batch_results(results, batch_num)
            self._record_batch_statistics(batch_num, results, improvements, 0.0)
            batch_num += 1
        return self._extract_final_results(batch_num, time.time() - start_time)
    
    def _run_sequential_batch(self, tasks):
        results = []
        for task in tasks:
            score = self._evaluate_individual(self.parameter_manager.normalize_parameters(task['params']))
            results.append({'individual_id': task['individual_id'], 'params': task['params'], 'score': score if score != float('-inf') else None})
        return results

    def _initialize_solution_pool(self) -> None:
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        initial_params = self.parameter_manager.get_initial_parameters()
        initial_solutions = []
        if initial_params: initial_solutions.append(np.clip(self.parameter_manager.normalize_parameters(initial_params), 0, 1))
        while len(initial_solutions) < self.pool_size: initial_solutions.append(np.random.random(param_count))
        
        tasks = [{'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(sol), 'proc_id': i % self.num_processes, 'evaluation_id': f"init_{i}"} for i, sol in enumerate(initial_solutions)]
        results = self._run_parallel_evaluations(tasks) if self.use_parallel else self._run_sequential_batch(tasks)
        
        for res in results:
            if res.get('score') is not None:
                self.solution_pool.append((self.parameter_manager.normalize_parameters(res['params']), res['score'], 0, 'init'))
                self.pool_scores.append(res['score'])
        
        if self.solution_pool:
            self._sort_pool()
            self.best_score = self.pool_scores[0]
            self.best_params = self.parameter_manager.denormalize_parameters(self.solution_pool[0][0])
            self.total_evaluations = len(self.solution_pool)

    def _generate_batch_from_pool(self, batch_num: int) -> List[Dict]:
        tasks = []
        for i in range(self.batch_size):
            parent = self._select_parent()
            trial = self._generate_dds_trial(parent, batch_num, i)
            tasks.append({'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(trial), 'proc_id': i % self.num_processes, 'evaluation_id': f"b{batch_num}_{i}"})
        return tasks

    def _select_parent(self):
        if not self.solution_pool: return None
        candidates = random.sample(range(len(self.solution_pool)), min(3, len(self.solution_pool)))
        return self.solution_pool[max(candidates, key=lambda i: self.pool_scores[i])][0].copy()

    def _generate_dds_trial(self, parent, batch_num, trial_num):
        param_count = len(self.parameter_manager.all_param_names)
        prob = 1.0 - np.log(self.total_evaluations + 1) / np.log(self.total_target_evaluations) if self.total_target_evaluations > 1 else 0.5
        prob = max(prob, 1.0 / param_count)
        trial = parent.copy(); variables = np.random.random(param_count) < prob
        if not variables.any(): variables[np.random.randint(param_count)] = True
        for i in range(param_count):
            if variables[i]:
                trial[i] = np.clip(parent[i] + np.random.normal(0, self.dds_r), 0, 1)
        return trial

    def _update_pool_with_batch_results(self, results, batch_num):
        improvements = 0
        for res in results:
            if res.get('score') is not None:
                self.total_evaluations += 1
                if len(self.solution_pool) < self.pool_size or res['score'] > min(self.pool_scores):
                    self.solution_pool.append((self.parameter_manager.normalize_parameters(res['params']), res['score'], batch_num, 'batch'))
                    self.pool_scores.append(res['score'])
                    improvements += 1
        self._sort_pool()
        if len(self.solution_pool) > self.pool_size:
            self.solution_pool = self.solution_pool[:self.pool_size]; self.pool_scores = self.pool_scores[:self.pool_size]
        if self.pool_scores and self.pool_scores[0] > self.best_score + self.convergence_threshold:
            # Significant improvement found - reset stagnation counter
            self.best_score = self.pool_scores[0]
            self.best_params = self.parameter_manager.denormalize_parameters(self.solution_pool[0][0])
            self.stagnation_counter = 0
        elif self.pool_scores and self.pool_scores[0] > self.best_score:
            # Minor improvement - update best but count toward stagnation
            self.best_score = self.pool_scores[0]
            self.best_params = self.parameter_manager.denormalize_parameters(self.solution_pool[0][0])
            self.stagnation_counter += 1
        else:
            self.stagnation_counter += 1
        return improvements

    def _sort_pool(self):
        combined = sorted(zip(self.solution_pool, self.pool_scores), key=lambda x: x[1], reverse=True)
        self.solution_pool = [x[0] for x in combined]; self.pool_scores = [x[1] for x in combined]

    def _record_batch_statistics(self, batch_num, results, improvements, duration):
        valid = [r['score'] for r in results if r['score'] is not None]
        self.batch_history.append({
            'generation': batch_num, 'algorithm': 'AsyncDDS', 'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params else None,
            'mean_score': np.mean(valid) if valid else None, 'valid_individuals': len(valid)
        })

    def _extract_final_results(self, final_batch, runtime):
        self.iteration_history = self.batch_history
        return self.best_params, self.best_score, self.iteration_history
