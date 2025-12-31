import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from symfluence.utils.optimization.optimizers.base_optimizer import BaseOptimizer

class DDSOptimizer(BaseOptimizer):
    """
    Dynamically Dimensioned Search (DDS) Optimizer for SYMFLUENCE
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.dds_r = config.get('DDS_R', 0.2)
        self.population = None
        self.population_scores = None
    
    def get_algorithm_name(self) -> str:
        return "DDS"
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run DDS algorithm with optional multi-start parallel support"""
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
        """Run single-instance DDS algorithm"""
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
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from symfluence.utils.optimization.worker_scripts import _run_dds_instance_worker
        
        results = []
        with ProcessPoolExecutor(max_workers=min(len(dds_tasks), self.num_processes)) as executor:
            future_to_task = {executor.submit(_run_dds_instance_worker, self._prepare_dds_worker_data(task)): task for task in dds_tasks}
            for future in as_completed(future_to_task):
                results.append(future.result())
        
        valid_results = [r for r in results if r.get('best_score') is not None]
        best_result = max(valid_results, key=lambda x: x['best_score'])
        
        self.best_score = best_result['best_score']
        self.best_params = best_result['best_params']
        return self.best_params, self.best_score, self.iteration_history
    
    def _prepare_dds_worker_data(self, task: Dict) -> Dict:
        """Prepare data for DDS instance worker"""
        proc_dirs = self.parallel_dirs[task['proc_id']] if hasattr(self, 'parallel_dirs') and task['proc_id'] < len(self.parallel_dirs) else {}
        return {
            'dds_task': task, 'config': self.config, 'target_metric': self.target_metric,
            'param_bounds': self.parameter_manager.param_bounds, 'all_param_names': self.parameter_manager.all_param_names,
            'summa_exe': str(self._get_summa_exe_path()),
            'summa_dir': str(proc_dirs.get('summa_dir', self.summa_sim_dir)),
            'mizuroute_dir': str(proc_dirs.get('mizuroute_dir', self.mizuroute_sim_dir)),
            'summa_settings_dir': str(proc_dirs.get('summa_settings_dir', self.optimization_settings_dir)),
            'mizuroute_settings_dir': str(proc_dirs.get('mizuroute_settings_dir', self.optimization_dir / "settings" / "mizuRoute")),
            'file_manager': str(proc_dirs.get('summa_settings_dir', self.optimization_settings_dir) / 'fileManager.txt')
        }

    def _record_generation(self, generation: int) -> None:
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
