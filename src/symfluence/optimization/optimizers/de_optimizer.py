import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer

class DEOptimizer(BaseOptimizer):
    """
    Differential Evolution (DE) Optimizer for SYMFLUENCE
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.population_size = self._determine_population_size()
        self.F = config.get('DE_SCALING_FACTOR', 0.5)
        self.CR = config.get('DE_CROSSOVER_RATE', 0.9)
        self.population = None
        self.population_scores = None
    
    def get_algorithm_name(self) -> str:
        return "DE"
    
    def _determine_population_size(self) -> int:
        config_pop_size = self.config.get('POPULATION_SIZE')
        if config_pop_size: return config_pop_size
        total_params = len(self.parameter_manager.all_param_names)
        return max(15, min(4 * total_params, 50))
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_population(initial_params)
        return self._run_de_algorithm()
    
    def _initialize_population(self, initial_params: Dict[str, np.ndarray]) -> None:
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        self.population = np.random.random((self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)
        
        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.population[0] = np.clip(initial_normalized, 0, 1)
        
        self._evaluate_population()
        best_idx = np.nanargmax(self.population_scores)
        if not np.isnan(self.population_scores[best_idx]):
            self.best_score = self.population_scores[best_idx]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[best_idx])
        
        self._record_generation(0)
    
    def _run_de_algorithm(self) -> Tuple[Dict, float, List]:
        for generation in range(1, self.max_iterations + 1):
            trial_population = self._create_trial_population()
            trial_scores = self._evaluate_trial_population(trial_population)
            
            improvements = 0
            for i in range(self.population_size):
                if not np.isnan(trial_scores[i]) and trial_scores[i] > self.population_scores[i]:
                    self.population[i] = trial_population[i].copy()
                    self.population_scores[i] = trial_scores[i]
                    improvements += 1
                    if trial_scores[i] > self.best_score:
                        self.best_score = trial_scores[i]
                        self.best_params = self.parameter_manager.denormalize_parameters(trial_population[i])
            
            self._record_generation(generation)
        return self.best_params, self.best_score, self.iteration_history
    
    def _create_trial_population(self) -> np.ndarray:
        trial_population = np.zeros_like(self.population)
        param_count = len(self.parameter_manager.all_param_names)
        for i in range(self.population_size):
            candidates = list(range(self.population_size)); candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = np.clip(self.population[r1] + self.F * (self.population[r2] - self.population[r3]), 0, 1)
            trial = self.population[i].copy()
            j_rand = np.random.randint(param_count)
            for j in range(param_count):
                if np.random.random() < self.CR or j == j_rand: trial[j] = mutant[j]
            trial_population[i] = trial
        return trial_population
    
    def _evaluate_population(self) -> None:
        if self.use_parallel: self._evaluate_population_parallel()
        else: self._evaluate_population_sequential()
    
    def _evaluate_trial_population(self, trial_population: np.ndarray) -> np.ndarray:
        if self.use_parallel: return self._evaluate_trial_population_parallel(trial_population)
        return np.array([self._evaluate_individual(ind) for ind in trial_population])
    
    def _evaluate_population_sequential(self) -> None:
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                self.population_scores[i] = self._evaluate_individual(self.population[i])
    
    def _evaluate_population_parallel(self) -> None:
        tasks = []
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                tasks.append({'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(self.population[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"pop_{i}"})
        if tasks:
            for res in self._run_parallel_evaluations(tasks):
                self.population_scores[res['individual_id']] = res['score'] if res['score'] is not None else float('-inf')

    def _evaluate_trial_population_parallel(self, trial_population: np.ndarray) -> np.ndarray:
        tasks = [{'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(trial_population[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"trial_{i}"} for i in range(self.population_size)]
        results = self._run_parallel_evaluations(tasks)
        scores = np.full(self.population_size, np.nan)
        for res in results: scores[res['individual_id']] = res['score'] if res['score'] is not None else float('-inf')
        return scores
    
    def _record_generation(self, generation: int) -> None:
        valid_scores = self.population_scores[~np.isnan(self.population_scores)]
        self.iteration_history.append({
            'generation': generation, 'algorithm': 'DE', 'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params else None,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores)
        })
