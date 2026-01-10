"""
Population-based parallel DDS optimizer.

Extends DDS with population-based parallelism for improved exploration
and faster convergence on multi-core systems.
"""

import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer

class PopulationDDSOptimizer(BaseOptimizer):
    """
    Population-based Parallel DDS Optimizer
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.dds_r = config.get('DDS_R', 0.2)
        self.population_size = self._determine_population_size()
        self.population = None; self.population_scores = None; self.current_generation = 0
    
    def get_algorithm_name(self) -> str:
        return "PopulationDDS"
    
    def _determine_population_size(self) -> int:
        config_pop_size = self.config.get('POPULATION_SIZE')
        if config_pop_size: return config_pop_size
        param_count = len(self.parameter_manager.all_param_names)
        return max(10, min(3 * param_count, 50))
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        self._initialize_population()
        while self.current_generation < self.max_iterations:
            tasks = self._generate_population_trials()
            results = self._run_parallel_evaluations(tasks)
            improvements = self._update_population_with_trials(results)
            self._record_generation_statistics(improvements, 0.0)
            self.current_generation += 1
        return self._extract_final_results()
    
    def _initialize_population(self) -> None:
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        self.population = np.random.random((self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)
        tasks = [{'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(self.population[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"p_init_{i}"} for i in range(self.population_size)]
        results = self._run_parallel_evaluations(tasks)
        for res in results: self.population_scores[res['individual_id']] = res['score'] if res['score'] is not None else float('-inf')
        self.best_score = np.nanmax(self.population_scores)
        self.best_params = self.parameter_manager.denormalize_parameters(self.population[np.nanargmax(self.population_scores)])

    def _generate_population_trials(self) -> List[Dict]:
        tasks = []
        for i in range(self.population_size):
            trial = self._generate_dds_trial(self.population[i], i)
            tasks.append({'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(trial), 'proc_id': i % self.num_processes, 'evaluation_id': f"p_gen_{self.current_generation}_{i}"})
        return tasks

    def _generate_dds_trial(self, parent, idx):
        param_count = len(self.parameter_manager.all_param_names)
        prob = 1.0 - np.log(self.current_generation * self.population_size + idx + 1) / np.log(self.max_iterations * self.population_size) if self.max_iterations > 1 else 0.5
        prob = max(prob, 1.0 / param_count)
        trial = parent.copy(); variables = np.random.random(param_count) < prob
        if not variables.any(): variables[np.random.randint(param_count)] = True
        for i in range(param_count):
            if variables[i]: trial[i] = np.clip(parent[i] + np.random.normal(0, self.dds_r), 0, 1)
        return trial

    def _update_population_with_trials(self, results):
        improvements = 0
        for res in results:
            idx = res['individual_id']
            if res['score'] is not None and res['score'] > self.population_scores[idx]:
                self.population[idx] = self.parameter_manager.normalize_parameters(res['params'])
                self.population_scores[idx] = res['score']
                improvements += 1
                if res['score'] > self.best_score:
                    self.best_score = res['score']; self.best_params = res['params'].copy()
        return improvements

    def _record_generation_statistics(self, improvements, duration):
        valid = self.population_scores[~np.isnan(self.population_scores)]
        self.iteration_history.append({
            'generation': self.current_generation, 'algorithm': 'PopulationDDS', 'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params else None,
            'mean_score': np.mean(valid) if len(valid) > 0 else None, 'valid_individuals': len(valid)
        })

    def _extract_final_results(self):
        return self.best_params, self.best_score, self.iteration_history
