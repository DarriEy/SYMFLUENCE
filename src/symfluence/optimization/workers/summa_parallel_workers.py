#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SUMMA Parallel Workers - Backward Compatibility Facade

This module maintains backward compatibility with existing imports.
All implementations have been refactored into the summa/ subpackage
for better maintainability:

- summa/worker_safety.py: Error handling, retry logic, signal handling
- summa/worker_orchestration.py: Core evaluation pipeline orchestration
- summa/netcdf_utilities.py: NetCDF time fixes and format conversion
- summa/metrics_calculation.py: Multi-target calibration metrics
- summa/parameter_application.py: Parameter file writing
- summa/model_execution.py: SUMMA/mizuRoute execution
- summa/dds_optimization.py: DDS algorithm for workers

All public functions are re-exported here for backward compatibility.
"""

# Re-export all public functions for backward compatibility
from .summa import (
    # NetCDF utilities
    fix_summa_time_precision,
    fix_summa_time_precision_inplace,
    _convert_lumped_to_distributed_worker,
    # Parameter application
    _apply_parameters_worker,
    _update_soil_depths_worker,
    _update_mizuroute_params_worker,
    _generate_trial_params_worker,
    # Metrics calculation
    _get_catchment_area_worker,
    _calculate_metrics_with_target,
    _calculate_metrics_inline_worker,
    _calculate_multitarget_objectives,
    resample_to_timestep,
    # Model execution
    _run_summa_worker,
    _run_mizuroute_worker,
    _needs_mizuroute_routing_worker,
    # Worker orchestration
    _evaluate_parameters_worker,
    # DDS optimization
    _run_dds_instance_worker,
    _evaluate_single_solution_worker,
    _denormalize_params_worker,
    # Worker safety
    _evaluate_parameters_worker_safe,
)

__all__ = [
    # Utility functions
    'resample_to_timestep',
    'fix_summa_time_precision',
    'fix_summa_time_precision_inplace',
    # Worker functions (public API)
    '_evaluate_parameters_worker_safe',
    '_evaluate_parameters_worker',
    '_run_dds_instance_worker',
    # Internal functions (exposed for advanced usage)
    '_convert_lumped_to_distributed_worker',
    '_apply_parameters_worker',
    '_update_soil_depths_worker',
    '_update_mizuroute_params_worker',
    '_generate_trial_params_worker',
    '_get_catchment_area_worker',
    '_calculate_metrics_with_target',
    '_calculate_metrics_inline_worker',
    '_calculate_multitarget_objectives',
    '_run_summa_worker',
    '_run_mizuroute_worker',
    '_needs_mizuroute_routing_worker',
    '_evaluate_single_solution_worker',
    '_denormalize_params_worker',
]
