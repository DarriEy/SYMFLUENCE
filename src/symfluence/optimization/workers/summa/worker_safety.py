#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Worker Safety for SUMMA Workers

This module contains error handling, retry logic, and signal handling
for safe worker execution in parallel processes.
"""

import os
import sys
import time
import random
import gc
import signal
import traceback
from pathlib import Path
from typing import Dict

import numpy as np

from .worker_orchestration import _evaluate_parameters_worker


def _export_worker_profile_data():
    """Export profiling data from worker process to file.

    Called in the finally block to ensure profile data is captured
    even if the worker exits abnormally.
    """
    try:
        from symfluence.core.profiling import get_profiler, get_profile_directory

        profiler = get_profiler()
        if not profiler.enabled or len(profiler._operations) == 0:
            return

        profile_dir = get_profile_directory()
        if not profile_dir:
            return

        profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename with PID
        pid = os.getpid()
        profile_file = profile_dir / f"worker_profile_{pid}.json"

        profiler.export_to_file(str(profile_file))
    except (OSError, IOError):
        # Silently fail - don't want profiling to break workers
        pass


def _evaluate_parameters_worker_safe(task_data: Dict, skip_profile_export: bool = False) -> Dict:
    """Safe wrapper for parameter evaluation with error handling and retries

    Args:
        task_data: Dictionary containing task parameters and configuration
        skip_profile_export: If True, skip exporting profile data (used when
            called from DDS worker which handles its own export)
    """
    worker_seed = task_data.get('random_seed')
    if worker_seed is not None:
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # Set up signal handler for clean termination
    def signal_handler(signum, frame):
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Set process-specific environment for isolation
    process_id = os.getpid()

    # Force single-threaded execution and disable problematic file locking
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NETCDF_DISABLE_LOCKING': '1',
        'HDF5_USE_FILE_LOCKING': 'FALSE',
        'HDF5_DISABLE_VERSION_CHECK': '1',
    })

    # Add small random delay to stagger file system access
    initial_delay = random.uniform(0.1, 0.8)
    time.sleep(initial_delay)

    try:
        # Force garbage collection at start
        gc.collect()

        # Enhanced logging setup for debugging
        proc_id = task_data.get('proc_id', 0)
        individual_id = task_data.get('individual_id', -1)

        # Try the evaluation with basic retry for stale file handle errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    retry_delay = 2.0 * (attempt + 1) + random.uniform(0, 1)
                    time.sleep(retry_delay)
                    gc.collect()

                # Call the worker function
                result = _evaluate_parameters_worker(task_data)

                # Force cleanup
                gc.collect()
                return result

            except (OSError, IOError, TimeoutError) as e:
                error_str = str(e).lower()
                error_trace = traceback.format_exc()

                # Check for stale file handle or similar filesystem errors
                if any(term in error_str for term in ['stale file handle', 'errno 116', 'input/output error', 'errno 5']):
                    if attempt < max_retries - 1:  # Not the last attempt
                        continue

                # For other errors or final attempt, return the error with full traceback
                return {
                    'individual_id': individual_id,
                    'params': task_data.get('params', {}),
                    'score': None,
                    'error': f'Worker exception (attempt {attempt + 1}): {str(e)}\nTraceback:\n{error_trace}',
                    'proc_id': proc_id,
                    'debug_info': {
                        'attempt': attempt + 1,
                        'max_retries': max_retries,
                        'process_id': process_id
                    }
                }

        # If we get here, all retries failed
        return {
            'individual_id': individual_id,
            'params': task_data.get('params', {}),
            'score': None,
            'error': f'Worker failed after {max_retries} attempts',
            'proc_id': proc_id
        }

    except (OSError, IOError, TimeoutError) as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'error': f'Critical worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }

    finally:
        # Export profiling data before cleanup (unless skipped for batch operations)
        if not skip_profile_export:
            _export_worker_profile_data()
        # Final cleanup
        gc.collect()
