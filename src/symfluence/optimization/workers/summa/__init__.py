#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SUMMA Worker Modules

This package contains modularized worker functions for SUMMA optimization.
The original summa_parallel_workers.py has been split into focused modules:

- worker_safety: Error handling, retry logic, signal handling
- worker_orchestration: Core evaluation pipeline orchestration
- netcdf_utilities: NetCDF time fixes and format conversion
- metrics_calculation: Multi-target calibration metrics
- parameter_application: Parameter file writing
- model_execution: SUMMA/mizuRoute execution
- dds_optimization: DDS algorithm for workers
"""

from .dds_optimization import (
    _denormalize_params_worker,
    _evaluate_single_solution_worker,
    _run_dds_instance_worker,
)
from .error_logging import (
    ErrorLogger,
    get_worker_error_logger,
    init_worker_error_logger,
    log_worker_failure,
)
from .metrics_calculation import (
    _calculate_metrics_inline_worker,
    _calculate_metrics_with_target,
    _calculate_multitarget_objectives,
    _get_catchment_area_worker,
    resample_to_timestep,
)
from .model_execution import (
    _needs_mizuroute_routing_worker,
    _run_mizuroute_worker,
    _run_summa_worker,
)
from .netcdf_utilities import (
    _convert_lumped_to_distributed_worker,
    fix_summa_time_precision,
    fix_summa_time_precision_inplace,
)
from .parameter_application import (
    _apply_parameters_worker,
    _generate_trial_params_worker,
    _update_mizuroute_params_worker,
    _update_soil_depths_worker,
)
from .worker_orchestration import (
    _evaluate_parameters_worker,
)
from .worker_safety import (
    _evaluate_parameters_worker_safe,
)

__all__ = [
    # NetCDF utilities
    'fix_summa_time_precision',
    'fix_summa_time_precision_inplace',
    '_convert_lumped_to_distributed_worker',
    # Parameter application
    '_apply_parameters_worker',
    '_update_soil_depths_worker',
    '_update_mizuroute_params_worker',
    '_generate_trial_params_worker',
    # Metrics calculation
    '_get_catchment_area_worker',
    '_calculate_metrics_with_target',
    '_calculate_metrics_inline_worker',
    '_calculate_multitarget_objectives',
    'resample_to_timestep',
    # Model execution
    '_run_summa_worker',
    '_run_mizuroute_worker',
    '_needs_mizuroute_routing_worker',
    # Worker orchestration
    '_evaluate_parameters_worker',
    # DDS optimization
    '_run_dds_instance_worker',
    '_evaluate_single_solution_worker',
    '_denormalize_params_worker',
    # Worker safety
    '_evaluate_parameters_worker_safe',
    # Error logging
    'ErrorLogger',
    'log_worker_failure',
    'init_worker_error_logger',
    'get_worker_error_logger',
]
