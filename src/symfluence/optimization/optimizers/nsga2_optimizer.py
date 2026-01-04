import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer
from symfluence.optimization.calibration_targets import (
    CalibrationTarget, StreamflowTarget, SnowTarget, GroundwaterTarget, ETTarget, SoilMoistureTarget, TWSTarget
)

class NSGA2Optimizer(BaseOptimizer):
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II) for SYMFLUENCE
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.population_size = self._determine_population_size()
        self.crossover_rate = config.get('NSGA2_CROSSOVER_RATE', 0.9)
        self.mutation_rate = config.get('NSGA2_MUTATION_RATE', 0.1)
        self.eta_c = config.get('NSGA2_ETA_C', 20)
        self.eta_m = config.get('NSGA2_ETA_M', 20)
        self.multi_target_mode = config.get('NSGA2_MULTI_TARGET', False)
        
        if self.multi_target_mode: self._setup_multi_target_objectives()
        else:
            self.objectives = ['NSE', 'KGE']; self.objective_names = ['NSE', 'KGE']
            self.num_objectives = 2; self.primary_target = self.calibration_target
            self.secondary_target = None; self.primary_metric = 'NSE'; self.secondary_metric = 'KGE'
        
        self.population = None; self.population_objectives = None
        self.population_ranks = None; self.population_crowding_distances = None
        self.pareto_front = None; self.best_score = None; self.best_params = None

    def _setup_multi_target_objectives(self) -> None:
        primary_target_type = self.config.get('NSGA2_PRIMARY_TARGET', self.config.get('OPTIMIZATION_TARGET', 'streamflow'))
        secondary_target_type = self.config.get('NSGA2_SECONDARY_TARGET', 'gw_depth')
        self.primary_metric = self.config.get('NSGA2_PRIMARY_METRIC', 'KGE')
        self.secondary_metric = self.config.get('NSGA2_SECONDARY_METRIC', 'KGE')
        self.primary_target = self._create_calibration_target_by_type(primary_target_type)
        self.secondary_target = self._create_calibration_target_by_type(secondary_target_type)
        self.calibration_target = self.primary_target
        self.objectives = [f"{primary_target_type}_{self.primary_metric}", f"{secondary_target_type}_{self.secondary_metric}"]
        self.objective_names = [f"{primary_target_type.upper()}_{self.primary_metric}", f"{secondary_target_type.upper()}_{self.secondary_metric}"]
        self.num_objectives = 2

    def _create_calibration_target_by_type(self, target_type: str):
        target_type = target_type.lower()
        if target_type in ['streamflow', 'flow', 'discharge']: return StreamflowTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['swe', 'sca', 'snow_depth', 'snow']: return SnowTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['gw_depth', 'gw_grace', 'groundwater', 'gw']: return GroundwaterTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['et', 'latent_heat', 'evapotranspiration']: return ETTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['sm_point', 'sm_smap', 'sm_esa', 'soil_moisture', 'sm']: return SoilMoistureTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['tws', 'grace', 'grace_tws', 'total_storage']: return TWSTarget(self.config, self.project_dir, self.logger)
        else: raise ValueError(f"Unknown target type: {target_type}")

    def get_algorithm_name(self) -> str:
        return "NSGA2"
    
    def _determine_population_size(self) -> int:
        config_pop_size = self.config.get('POPULATION_SIZE')
        if config_pop_size: return config_pop_size
        total_params = len(self.parameter_manager.all_param_names)
        return max(50, min(8 * total_params, 100))
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_population(initial_params)
        return self._run_nsga2_algorithm()

    def _initialize_population(self, initial_params: Dict[str, np.ndarray]) -> None:
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        self.population = np.random.random((self.population_size, param_count))
        self.population_objectives = np.full((self.population_size, self.num_objectives), np.nan)
        if initial_params:
            self.population[0] = np.clip(self.parameter_manager.normalize_parameters(initial_params), 0, 1)
        self._evaluate_population_multiobjective()
        self._perform_nsga2_selection()
        self._update_representative_solution()
        self._record_generation(0)
    
    def _run_nsga2_algorithm(self) -> Tuple[Dict, float, List]:
        for generation in range(1, self.max_iterations + 1):
            offspring = self._generate_offspring()
            offspring_objectives = self._evaluate_offspring(offspring)
            combined_population = np.vstack([self.population, offspring])
            combined_objectives = np.vstack([self.population_objectives, offspring_objectives])
            selected_indices = self._environmental_selection(combined_population, combined_objectives)
            self.population = combined_population[selected_indices]
            self.population_objectives = combined_objectives[selected_indices]
            self._perform_nsga2_selection()
            self._update_representative_solution()
            self._record_generation(generation)
        return self.best_params, self.best_score, self.iteration_history
    
    def _evaluate_population_multiobjective(self) -> None:
        if self.use_parallel: self._evaluate_population_parallel_multiobjective()
        else:
            for i in range(self.population_size):
                if np.any(np.isnan(self.population_objectives[i])):
                    self.population_objectives[i] = self._evaluate_individual_multiobjective(self.population[i])
    
    def _evaluate_individual_multiobjective(self, normalized_params: np.ndarray) -> np.ndarray:
        try:
            params = self.parameter_manager.denormalize_parameters(normalized_params)
            if not self._apply_parameters(params): return np.array([-1.0, -1.0])
            if not self.model_executor.run_models(self.summa_sim_dir, self.mizuroute_sim_dir, self.optimization_settings_dir): return np.array([-1.0, -1.0])
            if self.multi_target_mode:
                obj1 = self._extract_specific_metric(
                    self.primary_target.calculate_metrics(
                        self.summa_sim_dir,
                        mizuroute_dir=self.mizuroute_sim_dir
                    ),
                    self.primary_metric
                )
                obj2 = self._extract_specific_metric(
                    self.secondary_target.calculate_metrics(
                        self.summa_sim_dir,
                        mizuroute_dir=self.mizuroute_sim_dir
                    ),
                    self.secondary_metric
                )
            else:
                metrics = self.calibration_target.calculate_metrics(
                    self.summa_sim_dir,
                    mizuroute_dir=self.mizuroute_sim_dir
                )
                obj1 = self._extract_specific_metric(metrics, 'NSE'); obj2 = self._extract_specific_metric(metrics, 'KGE')
            return np.array([obj1 or -1.0, obj2 or -1.0])
        except Exception: return np.array([-1.0, -1.0])

    def _extract_specific_metric(self, metrics: Dict[str, float], metric_name: str) -> Optional[float]:
        if metrics is None: return None
        if metric_name in metrics: return metrics[metric_name]
        calib_key = f"Calib_{metric_name}"
        if calib_key in metrics: return metrics[calib_key]
        for key, value in metrics.items():
            if key.endswith(f"_{metric_name}"): return value
        return None

    def _evaluate_population_parallel_multiobjective(self) -> None:
        tasks = []
        for i in range(self.population_size):
            if np.any(np.isnan(self.population_objectives[i])):
                tasks.append({'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(self.population[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"nsga2_pop_{i}", 'multiobjective': True})
        if tasks:
            results = self._run_parallel_evaluations(tasks)
            for res in results:
                self.population_objectives[res['individual_id']] = np.array(res.get('objectives') or [-1.0, -1.0])

    def _evaluate_offspring(self, offspring: np.ndarray) -> np.ndarray:
        offspring_objectives = np.full((len(offspring), self.num_objectives), np.nan)
        if self.use_parallel:
            tasks = [{'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(offspring[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"nsga2_off_{i}", 'multiobjective': True} for i in range(len(offspring))]
            results = self._run_parallel_evaluations(tasks)
            for res in results: offspring_objectives[res['individual_id']] = np.array(res.get('objectives') or [-1.0, -1.0])
        else:
            for i in range(len(offspring)): offspring_objectives[i] = self._evaluate_individual_multiobjective(offspring[i])
        return offspring_objectives

    def _generate_offspring(self) -> np.ndarray:
        offspring = np.zeros_like(self.population)
        for i in range(0, self.population_size, 2):
            p1, p2 = self.population[self._tournament_selection()], self.population[self._tournament_selection()]
            if np.random.random() < self.crossover_rate: c1, c2 = self._sbx_crossover(p1, p2)
            else: c1, c2 = p1.copy(), p2.copy()
            offspring[i] = self._polynomial_mutation(c1)
            if i + 1 < self.population_size: offspring[i + 1] = self._polynomial_mutation(c2)
        return offspring
    
    def _tournament_selection(self) -> int:
        candidates = np.random.choice(self.population_size, 2, replace=False)
        best_idx = candidates[0]
        for candidate in candidates[1:]:
            if (self.population_ranks[candidate] < self.population_ranks[best_idx] or
                (self.population_ranks[candidate] == self.population_ranks[best_idx] and
                 self.population_crowding_distances[candidate] > self.population_crowding_distances[best_idx])):
                best_idx = candidate
        return best_idx
    
    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            if np.random.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-14:
                u = np.random.random()
                beta = (2 * u)**(1/(self.eta_c+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(self.eta_c+1))
                c1[i], c2[i] = 0.5*((1+beta)*p1[i]+(1-beta)*p2[i]), 0.5*((1-beta)*p1[i]+(1+beta)*p2[i])
        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                u = np.random.random()
                delta = (2*u)**(1/(self.eta_m+1))-1 if u < 0.5 else 1-(2*(1-u))**(1/(self.eta_m+1))
                mutated[i] = individual[i] + delta
        return np.clip(mutated, 0, 1)
    
    def _environmental_selection(self, combined_population: np.ndarray, combined_objectives: np.ndarray) -> np.ndarray:
        fronts = self._non_dominated_sorting(combined_objectives)
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= self.population_size: selected_indices.extend(front)
            else:
                remaining = self.population_size - len(selected_indices)
                distances = self._calculate_crowding_distance(combined_objectives[front])
                front_sorted = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                selected_indices.extend([x[0] for x in front_sorted[:remaining]])
                break
        return np.array(selected_indices)
    
    def _perform_nsga2_selection(self) -> None:
        fronts = self._non_dominated_sorting(self.population_objectives)
        self.population_ranks = np.zeros(self.population_size)
        for rank, front in enumerate(fronts): self.population_ranks[front] = rank
        self.population_crowding_distances = np.zeros(self.population_size)
        for front in fronts:
            if len(front) > 0: self.population_crowding_distances[front] = self._calculate_crowding_distance(self.population_objectives[front])
    
    def _non_dominated_sorting(self, objectives: np.ndarray) -> List[List[int]]:
        n = len(objectives); domination_counts = np.zeros(n); dominated_solutions = [[] for _ in range(n)]; fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(objectives[i] >= objectives[j]) and np.any(objectives[i] > objectives[j]): dominated_solutions[i].append(j)
                    elif np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]): domination_counts[i] += 1
            if domination_counts[i] == 0: fronts[0].append(i)
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0: next_front.append(j)
            current_front += 1; fronts.append(next_front)
        return fronts[:-1]
        
    def _calculate_crowding_distance(self, front_fvals: np.ndarray) -> np.ndarray:
        if front_fvals is None or len(front_fvals) == 0: return np.array([])
        N, M = front_fvals.shape; distances = np.zeros(N, dtype=float)
        if N <= 2: distances[:] = np.inf; return distances
        for j in range(M):
            vals = front_fvals[:, j].astype(float)
            if np.isnan(vals).any(): vals[np.isnan(vals)] = np.nanmedian(vals) or 0.0
            order = np.argsort(vals); sorted_vals = vals[order]
            distances[order[0]] = np.inf; distances[order[-1]] = np.inf
            obj_range = sorted_vals[-1] - sorted_vals[0]
            if obj_range > 0: distances[order[1:-1]] += (sorted_vals[2:] - sorted_vals[:-2]) / obj_range
        return distances

    def _update_representative_solution(self) -> None:
        front_1_indices = np.where(self.population_ranks == 0)[0]
        if len(front_1_indices) > 0:
            objs = self.population_objectives[front_1_indices]
            composite_scores = np.sum((objs + 1) / 2, axis=1)
            best_idx = front_1_indices[np.argmax(composite_scores)]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[best_idx])
            self.best_score = np.max(composite_scores)
        else:
            self.best_params = None; self.best_score = -2.0
        
    def _record_generation(self, generation: int) -> None:
        valid_obj1 = self.population_objectives[:, 0][~np.isnan(self.population_objectives[:, 0])]
        valid_obj2 = self.population_objectives[:, 1][~np.isnan(self.population_objectives[:, 1])]
        self.iteration_history.append({
            'generation': generation, 'algorithm': 'NSGA-II', 'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params else None,
            'mean_nse': np.mean(valid_obj1) if len(valid_obj1) > 0 else None,
            'mean_kge': np.mean(valid_obj2) if len(valid_obj2) > 0 else None,
            'valid_individuals': len(valid_obj1)
        })
