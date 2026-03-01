# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Calibration Targets

Calibration target classes that handle data loading, processing, and metric
calculation for specific variables during the optimization/calibration process.

Each calibration target is responsible for:
- Loading observed data for a specific variable
- Extracting simulated data from model outputs
- Calculating objective metrics for calibration

Base calibration targets (aliases from evaluation.evaluators):
- CalibrationTarget: Base class for all calibration targets
- ETTarget: Evapotranspiration calibration target
- StreamflowTarget: Streamflow calibration target (generic/SUMMA)
- SoilMoistureTarget: Soil moisture calibration target
- SnowTarget: Snow calibration target
- GroundwaterTarget: Groundwater calibration target
- TWSTarget: Terrestrial water storage calibration target
- MultivariateTarget: Multivariate calibration combining multiple variables

Model-specific calibration targets:
- SUMMAStreamflowTarget: SUMMA model streamflow calibration
- SUMMASnowTarget: SUMMA model snow calibration
- SUMMAETTarget: SUMMA model ET calibration
- GRStreamflowTarget: GR4J/GR6J model streamflow calibration
- HYPEStreamflowTarget: HYPE model streamflow calibration
- RHESSysStreamflowTarget: RHESSys model streamflow calibration
- NgenStreamflowTarget: NextGen model streamflow calibration
- FUSEStreamflowTarget: FUSE model streamflow calibration
- FUSESnowTarget: FUSE model snow calibration

Factory function for registry-based target creation:
- create_calibration_target(): Creates targets using registry with fallback
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from .base import (
    CalibrationTarget,
    ETTarget,
    GroundwaterTarget,
    MultivariateTarget,
    SnowTarget,
    SoilMoistureTarget,
    StreamflowTarget,
    TWSTarget,
)
from .gr_calibration_targets import GRStreamflowTarget
from .hype_calibration_targets import HYPEStreamflowTarget
from .ngen_calibration_targets import NgenStreamflowTarget
from .rhessys_calibration_targets import RHESSysStreamflowTarget

# Model-specific calibration targets
# Note: FUSE targets are NOT imported here to avoid circular dependencies during migration.
# They are available via:
# 1. Direct import: from symfluence.optimization.calibration_targets.fuse_calibration_targets import ...
# 2. Registry pattern: OptimizerRegistry.get_calibration_target('FUSE', 'streamflow')
# 3. Factory function: create_calibration_target() (uses registry + fallback)
from .summa_calibration_targets import (
    SUMMAETTarget,
    SUMMASnowTarget,
    SUMMAStreamflowTarget,
)

# FUSE targets: Use create_calibration_target() or import directly from .fuse_calibration_targets


# =========================================================================
# Default target mapping (model-agnostic targets)
# =========================================================================

_DEFAULT_TARGETS: Dict[str, Type[CalibrationTarget]] = {
    'streamflow': StreamflowTarget,
    'et': ETTarget,
    'evapotranspiration': ETTarget,
    'snow': SnowTarget,
    'swe': SnowTarget,
    'sca': SnowTarget,
    'groundwater': GroundwaterTarget,
    'gw': GroundwaterTarget,
    'soil_moisture': SoilMoistureTarget,
    'sm': SoilMoistureTarget,
    'sm_point': SoilMoistureTarget,
    'sm_smap': SoilMoistureTarget,
    'sm_ismn': SoilMoistureTarget,
    'sm_esa': SoilMoistureTarget,
    'tws': TWSTarget,
    'stor_grace': TWSTarget,
    'stor_mb': TWSTarget,
    'multivariate': MultivariateTarget,
}

# Model-specific target overrides
# Note: FUSE removed from this dict - use registry or factory function instead
_MODEL_SPECIFIC_TARGETS: Dict[str, Dict[str, Type[CalibrationTarget]]] = {
    'SUMMA': {
        'streamflow': SUMMAStreamflowTarget,
        'snow': SUMMASnowTarget,
        'et': SUMMAETTarget,
    },
    # 'FUSE': removed to avoid circular import - use OptimizerRegistry.get_calibration_target('FUSE', 'streamflow') instead
    'NGEN': {
        'streamflow': NgenStreamflowTarget,
    },
    'GR': {
        'streamflow': GRStreamflowTarget,
    },
    'HYPE': {
        'streamflow': HYPEStreamflowTarget,
    },
    'RHESSYS': {
        'streamflow': RHESSysStreamflowTarget,
    },
}


def create_calibration_target(
    model_name: str,
    target_type: str,
    config: Dict[str, Any],
    project_dir: Path,
    logger: logging.Logger
) -> CalibrationTarget:
    """
    Factory function to create calibration targets using registry with fallback.

    This function provides a centralized way to create calibration targets:
    1. First checks OptimizerRegistry for registered model-specific targets
    2. Falls back to model-specific target mappings
    3. Falls back to default (model-agnostic) targets

    Args:
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'NGEN')
        target_type: Type of calibration target (e.g., 'streamflow', 'snow', 'et')
        config: Configuration dictionary
        project_dir: Path to project directory
        logger: Logger instance

    Returns:
        Instantiated calibration target

    Raises:
        ValueError: If no suitable target class is found

    Example:
        >>> target = create_calibration_target(
        ...     model_name='SUMMA',
        ...     target_type='streamflow',
        ...     config=config,
        ...     project_dir=project_dir,
        ...     logger=logger
        ... )
    """
    from symfluence.core.registries import R

    model_key = model_name.upper()
    target_key = target_type.lower()

    # 1. Try registry first (for dynamically registered targets)
    target_cls = R.calibration_targets.get(f"{model_key}_{target_key.upper()}")

    # 2. Try model-specific mapping
    if target_cls is None and model_key in _MODEL_SPECIFIC_TARGETS:
        target_cls = _MODEL_SPECIFIC_TARGETS[model_key].get(target_key)

    # 3. Fall back to default targets
    if target_cls is None:
        target_cls = _DEFAULT_TARGETS.get(target_key)

    if target_cls is None:
        available = list(_DEFAULT_TARGETS.keys())
        raise ValueError(
            f"No calibration target found for model='{model_name}', type='{target_type}'. "
            f"Available target types: {available}"
        )

    logger.debug(f"Creating calibration target: {target_cls.__name__} for {model_name}/{target_type}")
    return target_cls(config, project_dir, logger)


def get_available_target_types(model_name: Optional[str] = None) -> list:
    """
    Get available calibration target types.

    Args:
        model_name: Optional model name to get model-specific targets

    Returns:
        List of available target type names
    """
    targets = set(_DEFAULT_TARGETS.keys())

    if model_name:
        model_key = model_name.upper()
        if model_key in _MODEL_SPECIFIC_TARGETS:
            targets.update(_MODEL_SPECIFIC_TARGETS[model_key].keys())

    return sorted(targets)


__all__ = [
    # Base targets
    'CalibrationTarget',
    'ETTarget',
    'StreamflowTarget',
    'SoilMoistureTarget',
    'SnowTarget',
    'GroundwaterTarget',
    'TWSTarget',
    'MultivariateTarget',
    # SUMMA targets
    'SUMMAStreamflowTarget',
    'SUMMASnowTarget',
    'SUMMAETTarget',
    # Other model-specific targets
    'GRStreamflowTarget',
    'HYPEStreamflowTarget',
    'RHESSysStreamflowTarget',
    'NgenStreamflowTarget',
    # Note: FUSE targets removed from __all__ to avoid circular import
    # Available via: from symfluence.optimization.calibration_targets.fuse_calibration_targets import FUSEStreamflowTarget
    # Factory functions
    'create_calibration_target',
    'get_available_target_types',
]
