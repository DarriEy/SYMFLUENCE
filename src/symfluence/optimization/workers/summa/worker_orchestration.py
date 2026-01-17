#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Worker Orchestration for SUMMA Workers

This module contains the core evaluation pipeline orchestration
for SUMMA parameter optimization.
"""

import time
import logging
import traceback
from pathlib import Path
from typing import Dict

import numpy as np

from .parameter_application import _apply_parameters_worker
from .model_execution import _run_summa_worker, _run_mizuroute_worker
from .netcdf_utilities import _convert_lumped_to_distributed_worker
from .metrics_calculation import _calculate_metrics_with_target, _calculate_multitarget_objectives
from .error_logging import log_worker_failure


def _evaluate_parameters_worker(task_data: Dict) -> Dict:
    """Enhanced worker with inline metrics calculation and runtime tracking"""
    # Start timing the core evaluation
    eval_start_time = time.time()

    debug_info = {
        'stage': 'initialization',
        'files_checked': [],
        'commands_run': [],
        'errors': [],
        'iteration': task_data.get('iteration', 0),
        'individual_id': task_data.get('individual_id', 0)
    }

    try:
        # Extract task info
        individual_id = task_data['individual_id']
        params = task_data['params']
        proc_id = task_data['proc_id']

        debug_info['individual_id'] = individual_id
        debug_info['proc_id'] = proc_id

        # Setup process logger only if LOG_LEVEL is DEBUG
        config = task_data.get('config', {})
        enable_worker_logging = config.get('LOG_LEVEL', 'INFO').upper() == 'DEBUG'

        if enable_worker_logging:
            logger = logging.getLogger(f'worker_{proc_id}_{individual_id}')
            if not logger.handlers:
                logger.setLevel(logging.DEBUG)
                handler = logging.StreamHandler()
                formatter = logging.Formatter(f'[P{proc_id:02d}-I{individual_id:03d}] %(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        else:
            # Use a minimal logger that only logs errors
            logger = logging.getLogger(f'worker_{proc_id}_{individual_id}')
            if not logger.handlers:
                logger.setLevel(logging.ERROR)
                handler = logging.StreamHandler()
                formatter = logging.Formatter(f'[P{proc_id:02d}-I{individual_id:03d}] %(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)

        logger.info(f"Starting evaluation of individual {individual_id}")

        # Check multi-objective flag
        is_multiobjective = task_data.get('multiobjective', False)
        logger.info(f"Multi-objective evaluation: {is_multiobjective}")

        # DETERMINE ROUTING NEEDS EARLY
        # Check if any target requires routing (streamflow needs routing in distributed domains)
        config = task_data['config']
        needs_routing = False

        # Collect all target types that need to be evaluated
        targets_to_check = []
        if task_data.get('multi_target_mode', False):
            # Multi-target mode: check both primary and secondary targets
            targets_to_check.append(task_data.get('primary_target_type', 'streamflow'))
            targets_to_check.append(task_data.get('secondary_target_type', ''))
        else:
            # Single-target mode: use calibration_variable
            targets_to_check.append(task_data.get('calibration_variable', 'streamflow'))

        # Check if any target is streamflow (requires routing in non-lumped domains)
        streamflow_aliases = ['streamflow', 'flow', 'discharge']
        has_streamflow_target = any(t.lower() in streamflow_aliases for t in targets_to_check if t)

        if has_streamflow_target:
            domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')

            if domain_method not in ['point', 'lumped'] or (domain_method == 'lumped' and routing_delineation == 'river_network'):
                needs_routing = True

        logger.info(f"Needs routing: {needs_routing} (targets: {targets_to_check})")

        # Convert paths
        debug_info['stage'] = 'path_setup'
        summa_exe = Path(task_data['summa_exe']).resolve()
        file_manager = Path(task_data['file_manager']).resolve()
        summa_dir = Path(task_data['summa_dir']).resolve()
        mizuroute_dir = Path(task_data['mizuroute_dir']).resolve()
        summa_settings_dir = Path(task_data['summa_settings_dir']).resolve()

        # Verify paths
        debug_info['stage'] = 'path_verification'
        critical_paths = {
            'SUMMA executable': summa_exe,
            'File manager': file_manager,
            'SUMMA directory': summa_dir,
            'Settings directory': summa_settings_dir
        }

        for name, path in critical_paths.items():
            debug_info['files_checked'].append(f"{name}: {path}")
            if not path.exists():
                error_msg = f'{name} not found: {path}'
                logger.error(error_msg)
                eval_runtime = time.time() - eval_start_time
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'objectives': None if is_multiobjective else None,
                    'error': error_msg,
                    'debug_info': debug_info,
                    'runtime': eval_runtime
                }

        # Apply parameters
        debug_info['stage'] = 'parameter_application'
        logger.info("Applying parameters")
        if not _apply_parameters_worker(params, task_data, summa_settings_dir, logger, debug_info):
            error_msg = 'Failed to apply parameters'
            logger.error(error_msg)
            eval_runtime = time.time() - eval_start_time
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'objectives': None if is_multiobjective else None,
                'error': error_msg,
                'debug_info': debug_info,
                'runtime': eval_runtime
            }

        # Run SUMMA
        debug_info['stage'] = 'summa_execution'
        logger.info("Running SUMMA")
        summa_start = time.time()
        if not _run_summa_worker(summa_exe, file_manager, summa_dir, logger, debug_info, summa_settings_dir):
            error_msg = 'SUMMA simulation failed'
            logger.error(error_msg)
            eval_runtime = time.time() - eval_start_time

            # Log failure artifacts if PARAMS_KEEP_TRIALS is enabled
            log_worker_failure(
                iteration=debug_info.get('iteration', 0),
                params=params,
                debug_info=debug_info,
                settings_dir=summa_settings_dir,
                summa_dir=summa_dir,
                error_message=error_msg,
                proc_id=proc_id,
                individual_id=individual_id,
                config=config
            )

            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'objectives': None if is_multiobjective else None,
                'error': error_msg,
                'debug_info': debug_info,
                'runtime': eval_runtime
            }
        summa_runtime = time.time() - summa_start

        # Handle mizuRoute routing
        mizuroute_runtime = 0.0

        if needs_routing:
            # Check if we need lumped-to-distributed conversion
            debug_info['stage'] = 'lumped_to_distributed_conversion'
            config = task_data['config']
            domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = config.get('ROUTING_DELINEATION', 'river_network')

            # ONLY convert if domain is lumped but routing expects a network
            if domain_method in ['lumped', 'point'] and routing_delineation == 'river_network':
                logger.info("Converting lumped SUMMA output to distributed format")
                if not _convert_lumped_to_distributed_worker(task_data, summa_dir, logger, debug_info):
                    error_msg = 'Lumped-to-distributed conversion failed'
                    logger.error(error_msg)
                    eval_runtime = time.time() - eval_start_time
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'objectives': None if is_multiobjective else None,
                        'error': error_msg,
                        'debug_info': debug_info,
                        'runtime': eval_runtime
                    }

            debug_info['stage'] = 'mizuroute_execution'
            logger.info("Running mizuRoute")
            mizu_start = time.time()
            if not _run_mizuroute_worker(task_data, mizuroute_dir, logger, debug_info, summa_dir):
                error_msg = 'mizuRoute simulation failed'
                logger.error(error_msg)
                eval_runtime = time.time() - eval_start_time

                # Log failure artifacts if PARAMS_KEEP_TRIALS is enabled
                log_worker_failure(
                    iteration=debug_info.get('iteration', 0),
                    params=params,
                    debug_info=debug_info,
                    settings_dir=summa_settings_dir,
                    summa_dir=summa_dir,
                    error_message=error_msg,
                    proc_id=proc_id,
                    individual_id=individual_id,
                    config=config
                )

                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'objectives': None if is_multiobjective else None,
                    'error': error_msg,
                    'debug_info': debug_info,
                    'runtime': eval_runtime
                }
            mizuroute_runtime = time.time() - mizu_start

        # Calculate metrics using INLINE method to avoid import issues
        debug_info['stage'] = 'metrics_calculation'
        metrics_start = time.time()

        if is_multiobjective:
            logger.info("Starting multi-objective metrics calculation")

            try:
                # NEW: Support multi-target optimization (e.g. streamflow + TWS)
                if task_data.get('multi_target_mode', False):
                    logger.info("Using multi-target objective calculation")
                    project_dir = task_data.get('project_dir', '.')
                    objectives = _calculate_multitarget_objectives(
                        task_data, str(summa_dir), str(mizuroute_dir),
                        task_data['config'], project_dir, logger
                    )

                    # Log extracted objectives
                    objective_names = task_data.get('objective_names', ['OBJ1', 'OBJ2'])
                    for i, val in enumerate(objectives):
                        name = objective_names[i] if i < len(objective_names) else f"OBJ{i+1}"
                        logger.info(f"Extracted {name}: {val}")

                    # Set primary score for backward compatibility
                    target_metric = task_data.get('target_metric', 'OBJ1')
                    score = objectives[0] if objectives else -1e6
                    metrics = {'objectives': objectives} # Minimal metrics dict for compatibility
                else:
                    # Fallback to single-target multi-metric calculation (e.g. NSE + KGE for streamflow)
                    # Use proper calibration target instead of inline streamflow-only calculation
                    metrics = _calculate_metrics_with_target(
                        summa_dir,
                        mizuroute_dir if needs_routing else None,
                        task_data['config'],
                        logger,
                        project_dir=task_data.get('project_dir')
                    )

                    if not metrics:
                        error_msg = 'Metrics calculation failed'
                        logger.error(error_msg)
                        eval_runtime = time.time() - eval_start_time
                        return {
                            'individual_id': individual_id,
                            'params': params,
                            'score': None,
                            'objectives': None,
                            'error': error_msg,
                            'debug_info': debug_info,
                            'runtime': eval_runtime
                        }

                    logger.info(f"Inline metrics calculated: {list(metrics.keys())}")

                    # Dynamically get objectives based on the list passed from the main process
                    objective_names = task_data.get('objective_names', ['NSE', 'KGE'])  # Fallback to old behavior
                    logger.info(f"Extracting objectives: {objective_names}")

                    objectives = []
                    for obj_name in objective_names:
                        # Look for both 'Calib_OBJ' and 'OBJ' to be safe
                        value = metrics.get(obj_name) or metrics.get(f'Calib_{obj_name}')

                        logger.info(f"Extracted {obj_name}: {value}")

                        # Handle None/NaN values with a penalty
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            logger.warning(f"{obj_name} is None/NaN, setting to a penalty value.")
                            # Use a large penalty for minimization metrics, and a large negative penalty for maximization metrics
                            if obj_name.upper() in ['RMSE', 'MAE', 'PBIAS']:
                                value = 1e6
                            else:
                                value = -1e6

                        objectives.append(float(value))

                    # Set the single 'score' based on the primary target_metric for backward compatibility or other modules
                    target_metric = task_data.get('target_metric', 'KGE')
                    score = -1e6  # Default penalty
                    if target_metric in objective_names:
                        target_idx = objective_names.index(target_metric)
                        score = objectives[target_idx]
                    else:  # Fallback to KGE if available
                        if 'KGE' in objective_names:
                            score = objectives[objective_names.index('KGE')]

                logger.info(f"Final objectives: {objectives}")

                metrics_runtime = time.time() - metrics_start
                eval_runtime = time.time() - eval_start_time

                logger.info(f"Multi-objective completed. Objectives: {objectives}, Score ({target_metric}): {score:.6f}")
                logger.info(f"Runtime breakdown: Total={eval_runtime:.1f}s, SUMMA={summa_runtime:.1f}s, mizuRoute={mizuroute_runtime:.1f}s, Metrics={metrics_runtime:.1f}s")

                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': score,
                    'objectives': objectives,
                    'error': None,
                    'debug_info': debug_info,
                    'runtime': eval_runtime,
                    'runtime_breakdown': {
                        'total': eval_runtime,
                        'summa': summa_runtime,
                        'mizuroute': mizuroute_runtime,
                        'metrics': metrics_runtime
                    }
                }

            except Exception as e:
                error_msg = f"Exception in inline multi-objective calculation: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                eval_runtime = time.time() - eval_start_time
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'objectives': None,
                    'error': error_msg,
                    'debug_info': debug_info,
                    'runtime': eval_runtime
                }

        else:
            # Single-objective evaluation
            logger.info("Single-objective evaluation using calibration target")

            try:
                metrics = _calculate_metrics_with_target(
                    summa_dir,
                    mizuroute_dir if needs_routing else None,
                    task_data['config'],
                    logger,
                    project_dir=task_data.get('project_dir')
                )

                if not metrics:
                    error_msg = 'Metrics calculation failed'
                    logger.error(error_msg)
                    eval_runtime = time.time() - eval_start_time
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'error': error_msg,
                        'debug_info': debug_info,
                        'runtime': eval_runtime
                    }

                # Extract target metric
                target_metric = task_data['target_metric']
                score = metrics.get(target_metric) or metrics.get(f'Calib_{target_metric}')

                if score is None:
                    # Try to find any metric with the target name
                    for key, value in metrics.items():
                        if target_metric in key:
                            score = value
                            break

                if score is None:
                    logger.error(f"Could not extract {target_metric} from metrics. Available metrics: {list(metrics.keys())}")
                    eval_runtime = time.time() - eval_start_time
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'error': f'Could not extract {target_metric}',
                        'debug_info': debug_info,
                        'runtime': eval_runtime
                    }

                # Apply negation for minimize metrics
                if target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
                    score = -score

                metrics_runtime = time.time() - metrics_start
                eval_runtime = time.time() - eval_start_time

                logger.info(f"Single-objective completed. {target_metric}: {score:.6f}")
                logger.info(f"Runtime breakdown: Total={eval_runtime:.1f}s, SUMMA={summa_runtime:.1f}s, mizuRoute={mizuroute_runtime:.1f}s, Metrics={metrics_runtime:.1f}s")

                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': score,
                    'error': None,
                    'debug_info': debug_info,
                    'runtime': eval_runtime,
                    'runtime_breakdown': {
                        'total': eval_runtime,
                        'summa': summa_runtime,
                        'mizuroute': mizuroute_runtime,
                        'metrics': metrics_runtime
                    }
                }

            except Exception as e:
                error_msg = f"Exception in single-objective calculation: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                eval_runtime = time.time() - eval_start_time
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'error': error_msg,
                    'debug_info': debug_info,
                    'runtime': eval_runtime
                }

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f'Worker exception at stage {debug_info.get("stage", "unknown")}: {str(e)}'
        debug_info['errors'].append(f"{error_msg}\nTraceback:\n{error_trace}")

        is_multiobjective = task_data.get('multiobjective', False)
        eval_runtime = time.time() - eval_start_time

        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'objectives': None if is_multiobjective else None,
            'error': error_msg,
            'debug_info': debug_info,
            'full_traceback': error_trace,
            'runtime': eval_runtime
        }
