#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DDS Optimization for SUMMA Workers

This module contains the DDS (Dynamically Dimensioned Search) algorithm
implementation for running within worker processes.
"""

import logging
import os
import traceback
from pathlib import Path
from typing import Dict

import numpy as np

from .worker_safety import _evaluate_parameters_worker_safe


def _export_worker_profile_data(worker_id: int = None):
    """Export profiling data from worker process to file.

    This function should be called at the end of worker tasks to ensure
    profile data is captured even if atexit handlers don't run reliably
    (e.g., with ProcessPoolExecutor on macOS).
    """
    try:
        from symfluence.core.profiling import get_profile_directory, get_profiler

        profiler = get_profiler()
        if not profiler.enabled or len(profiler._operations) == 0:
            return

        profile_dir = get_profile_directory()
        if not profile_dir:
            return

        profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename with PID and optional worker ID
        pid = os.getpid()
        if worker_id is not None:
            profile_file = profile_dir / f"worker_profile_{pid}_{worker_id}.json"
        else:
            profile_file = profile_dir / f"worker_profile_{pid}.json"

        profiler.export_to_file(str(profile_file))
    except (ValueError, KeyError, AttributeError):
        # Silently fail - don't want profiling to break workers
        pass


def _run_dds_instance_worker(worker_data: Dict) -> Dict:
    """
    Worker function that runs a complete DDS instance
    This runs in a separate process
    """
    try:
        dds_task = worker_data['dds_task']
        start_id = dds_task['start_id']
        max_iterations = dds_task['max_iterations']
        dds_r = dds_task['dds_r']
        starting_solution = dds_task['starting_solution']

        # Set up process-specific random seed
        np.random.seed(dds_task['random_seed'])

        # Set up logger
        log_level = worker_data.get('config', {}).get('LOG_LEVEL', 'INFO').upper()
        logger = logging.getLogger(f'dds_worker_{start_id}')
        if not logger.handlers:
            logger.setLevel(getattr(logging, log_level))
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[DDS-{start_id:02d}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.info(f"Starting DDS instance {start_id} with {max_iterations} iterations")

        # Initialize DDS state
        current_solution = starting_solution.copy()
        param_count = len(current_solution)

        # Evaluate initial solution
        current_score = _evaluate_single_solution_worker(current_solution, worker_data, logger)

        best_solution = current_solution.copy()
        best_score = current_score
        best_params = _denormalize_params_worker(best_solution, worker_data)

        history = []
        total_evaluations = 1

        # Record initial state
        history.append({
            'generation': 0,
            'best_score': best_score,
            'current_score': current_score,
            'best_params': best_params
        })

        # DDS main loop
        for iteration in range(1, max_iterations + 1):
            # Calculate selection probability
            prob_select = 1.0 - np.log(iteration) / np.log(max_iterations)
            prob_select = max(prob_select, 1.0 / param_count)

            # Create trial solution
            trial_solution = current_solution.copy()

            # Select variables to perturb
            variables_to_perturb = np.random.random(param_count) < prob_select
            if not np.any(variables_to_perturb):
                random_idx = np.random.randint(0, param_count)
                variables_to_perturb[random_idx] = True

            # Apply perturbations
            for i in range(param_count):
                if variables_to_perturb[i]:
                    perturbation = np.random.normal(0, dds_r)
                    trial_solution[i] = current_solution[i] + perturbation

                    # Reflect at bounds
                    if trial_solution[i] < 0:
                        trial_solution[i] = -trial_solution[i]
                    elif trial_solution[i] > 1:
                        trial_solution[i] = 2.0 - trial_solution[i]

                    trial_solution[i] = np.clip(trial_solution[i], 0, 1)

            # Evaluate trial solution
            trial_score = _evaluate_single_solution_worker(trial_solution, worker_data, logger)
            total_evaluations += 1

            # Selection (greedy)
            improvement = False
            if trial_score > current_score:
                current_solution = trial_solution.copy()
                current_score = trial_score
                improvement = True

                if trial_score > best_score:
                    best_solution = trial_solution.copy()
                    best_score = trial_score
                    best_params = _denormalize_params_worker(best_solution, worker_data)
                    logger.info(f"Iter {iteration}: NEW BEST! Score={best_score:.6f}")

            # Record iteration
            history.append({
                'generation': iteration,
                'best_score': best_score,
                'current_score': current_score,
                'trial_score': trial_score,
                'improvement': improvement,
                'num_variables_perturbed': np.sum(variables_to_perturb),
                'best_params': best_params
            })

        logger.info(f"DDS instance {start_id} completed: Best={best_score:.6f}, Evaluations={total_evaluations}")

        # Export profiling data before returning
        _export_worker_profile_data(worker_id=start_id)

        return {
            'start_id': start_id,
            'best_score': best_score,
            'best_params': best_params,
            'best_solution': best_solution,
            'history': history,
            'total_evaluations': total_evaluations,
            'final_current_score': current_score
        }

    except (ValueError, KeyError, AttributeError) as e:
        # Still export profiling data even on error
        try:
            _export_worker_profile_data(worker_id=worker_data['dds_task']['start_id'])
        except (ValueError, KeyError, AttributeError):
            pass
        return {
            'start_id': worker_data['dds_task']['start_id'],
            'best_score': None,
            'error': f'DDS worker exception: {str(e)}\n{traceback.format_exc()}'
        }


def _evaluate_single_solution_worker(solution: np.ndarray, worker_data: Dict, logger) -> float:
    """Evaluate a single solution in the worker process"""
    try:
        # Denormalize parameters
        params = _denormalize_params_worker(solution, worker_data)

        # Create evaluation task
        task_data = {
            'individual_id': 0,
            'params': params,
            'config': worker_data['config'],
            'target_metric': worker_data['target_metric'],
            'calibration_variable': worker_data['calibration_variable'],
            'domain_name': worker_data['domain_name'],
            'project_dir': worker_data['project_dir'],
            'original_depths': worker_data['original_depths'],
            'summa_exe': worker_data['summa_exe'],
            'file_manager': worker_data['file_manager'],
            'summa_dir': worker_data['summa_dir'],
            'mizuroute_dir': worker_data['mizuroute_dir'],
            'summa_settings_dir': worker_data['summa_settings_dir'],
            'mizuroute_settings_dir': worker_data['mizuroute_settings_dir'],
            'proc_id': 0
        }

        # Use existing worker function (skip profile export since DDS handles it at end)
        result = _evaluate_parameters_worker_safe(task_data, skip_profile_export=True)

        score = result.get('score')
        if score is None:
            logger.warning(f"Evaluation failed: {result.get('error', 'Unknown error')}")
            return float('-inf')

        return score

    except (ValueError, KeyError, AttributeError) as e:
        logger.error(f"Error evaluating solution: {str(e)}")
        return float('-inf')


def _denormalize_params_worker(normalized_solution: np.ndarray, worker_data: Dict) -> Dict:
    """Denormalize parameters in worker process"""
    param_bounds = worker_data['param_bounds']
    param_names = worker_data['all_param_names']

    params = {}
    for i, param_name in enumerate(param_names):
        if param_name in param_bounds:
            bounds = param_bounds[param_name]
            value = bounds['min'] + normalized_solution[i] * (bounds['max'] - bounds['min'])
            params[param_name] = np.array([value])

    return params
