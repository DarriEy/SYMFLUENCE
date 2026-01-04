import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer

class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) Optimizer for SYMFLUENCE
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.swarm_size = config.get('SWRMSIZE', 20)
        self.c1 = config.get('PSO_COGNITIVE_PARAM', 1.5)
        self.c2 = config.get('PSO_SOCIAL_PARAM', 1.5)
        self.w_initial = config.get('PSO_INERTIA_WEIGHT', 0.7)
        self.w_reduction_rate = config.get('PSO_INERTIA_REDUCTION_RATE', 0.99)
        self.swarm_positions = None
        self.swarm_velocities = None
        self.swarm_scores = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self.current_inertia = self.w_initial
    
    def get_algorithm_name(self) -> str:
        return "PSO"
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_swarm(initial_params)
        return self._run_pso_algorithm()
    
    def _initialize_swarm(self, initial_params: Dict[str, np.ndarray]) -> None:
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        self.swarm_positions = np.random.random((self.swarm_size, param_count))
        self.swarm_scores = np.full(self.swarm_size, np.nan)
        if initial_params:
            self.swarm_positions[0] = np.clip(self.parameter_manager.normalize_parameters(initial_params), 0, 1)
        self.swarm_velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, param_count))
        self.personal_best_positions = self.swarm_positions.copy()
        self.personal_best_scores = np.full(self.swarm_size, float('-inf'))
        self._evaluate_swarm()
        self._update_personal_bests()
        self._update_global_best()
        self._record_generation(0)
    
    def _run_pso_algorithm(self) -> Tuple[Dict, float, List]:
        schedule = str(self.config.get("INERTIA_SCHEDULE", "MULTIPLICATIVE")).upper()
        decay = float(self.config.get("INERTIA_DECAY_RATE", 0.99))
        for iteration in range(1, self.max_iterations + 1):
            if schedule == "MULTIPLICATIVE": self.current_inertia *= decay
            self._update_velocities()
            self._update_positions()
            self._evaluate_swarm()
            self._update_personal_bests()
            if self._update_global_best():
                pass # New global best found
            self._record_generation(iteration)
        self.best_score = self.global_best_score
        self.best_params = self.parameter_manager.denormalize_parameters(self.global_best_position)
        return self.best_params, self.best_score, self.iteration_history

    def _update_velocities(self) -> None:
        param_count = len(self.parameter_manager.all_param_names)
        for i in range(self.swarm_size):
            r1, r2 = np.random.random(param_count), np.random.random(param_count)
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.swarm_positions[i])
            social = self.c2 * r2 * (self.global_best_position - self.swarm_positions[i])
            self.swarm_velocities[i] = np.clip(self.current_inertia * self.swarm_velocities[i] + cognitive + social, -0.2, 0.2)
    
    def _update_positions(self) -> None:
        self.swarm_positions += self.swarm_velocities
        for i in range(self.swarm_size):
            for j in range(len(self.parameter_manager.all_param_names)):
                if self.swarm_positions[i, j] < 0:
                    self.swarm_positions[i, j] = -self.swarm_positions[i, j]
                    self.swarm_velocities[i, j] = -self.swarm_velocities[i, j]
                elif self.swarm_positions[i, j] > 1:
                    self.swarm_positions[i, j] = 2.0 - self.swarm_positions[i, j]
                    self.swarm_velocities[i, j] = -self.swarm_velocities[i, j]
        self.swarm_positions = np.clip(self.swarm_positions, 0, 1)
    
    def _evaluate_swarm(self) -> None:
        if self.use_parallel: self._evaluate_swarm_parallel()
        else: self._evaluate_swarm_sequential()
    
    def _evaluate_swarm_sequential(self) -> None:
        for i in range(self.swarm_size):
            self.swarm_scores[i] = self._evaluate_individual(self.swarm_positions[i])
    
    def _evaluate_swarm_parallel(self) -> None:
        tasks = [{'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(self.swarm_positions[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"pso_{i}"} for i in range(self.swarm_size)]
        results = self._run_parallel_evaluations(tasks)
        for res in results: self.swarm_scores[res['individual_id']] = res['score'] if res['score'] is not None else float('-inf')
    
    def _update_personal_bests(self) -> int:
        improvements = 0
        for i in range(self.swarm_size):
            if not np.isnan(self.swarm_scores[i]) and self.swarm_scores[i] > self.personal_best_scores[i]:
                self.personal_best_scores[i] = self.swarm_scores[i]
                self.personal_best_positions[i] = self.swarm_positions[i].copy()
                improvements += 1
        return improvements
    
    def _update_global_best(self) -> bool:
        valid_scores = self.personal_best_scores[~np.isnan(self.personal_best_scores)]
        if len(valid_scores) == 0: return False
        best_idx = np.nanargmax(self.personal_best_scores)
        if self.personal_best_scores[best_idx] > self.global_best_score:
            self.global_best_score = self.personal_best_scores[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].copy()
            return True
        return False

    def _record_generation(self, generation: int) -> None:
        valid_scores = self.swarm_scores[~np.isnan(self.swarm_scores)]
        self.iteration_history.append({
            'generation': generation, 'algorithm': 'PSO', 'best_score': self.global_best_score,
            'best_params': self.parameter_manager.denormalize_parameters(self.global_best_position) if self.global_best_position is not None else None,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores)
        })
