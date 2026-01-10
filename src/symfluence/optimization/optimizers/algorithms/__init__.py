#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimization Algorithms Package

This package contains optimization algorithms implemented using the Strategy pattern.
Each algorithm can be used interchangeably through the common OptimizationAlgorithm interface.

Usage:
    from symfluence.optimization.optimizers.algorithms import get_algorithm

    algorithm = get_algorithm('dds', config, logger)
    result = algorithm.optimize(...)
"""

from typing import Dict, Any, Type
import logging

from .base_algorithm import OptimizationAlgorithm
from .dds import DDSAlgorithm
from .pso import PSOAlgorithm
from .de import DEAlgorithm
from .sce_ua import SCEUAAlgorithm
from .async_dds import AsyncDDSAlgorithm
from .nsga2 import NSGA2Algorithm
from .adam import AdamAlgorithm
from .lbfgs import LBFGSAlgorithm


# Algorithm registry mapping names to classes
ALGORITHM_REGISTRY: Dict[str, Type[OptimizationAlgorithm]] = {
    'dds': DDSAlgorithm,
    'pso': PSOAlgorithm,
    'de': DEAlgorithm,
    'sce-ua': SCEUAAlgorithm,
    'sce_ua': SCEUAAlgorithm,  # Alternative name
    'sceua': SCEUAAlgorithm,   # Alternative name
    'async_dds': AsyncDDSAlgorithm,
    'asyncdds': AsyncDDSAlgorithm,  # Alternative name
    'nsga2': NSGA2Algorithm,
    'nsga-ii': NSGA2Algorithm,  # Alternative name
    'adam': AdamAlgorithm,
    'lbfgs': LBFGSAlgorithm,
    'l-bfgs': LBFGSAlgorithm,  # Alternative name
}


def get_algorithm(
    name: str,
    config: Dict[str, Any],
    logger: logging.Logger
) -> OptimizationAlgorithm:
    """
    Get an optimization algorithm instance by name.

    Args:
        name: Algorithm name (case-insensitive). Supported values:
              'dds', 'pso', 'de', 'sce-ua', 'async_dds', 'nsga2'
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Instantiated algorithm

    Raises:
        ValueError: If algorithm name is not recognized
    """
    name_lower = name.lower().replace('-', '_').replace(' ', '_')

    if name_lower not in ALGORITHM_REGISTRY:
        available = list(set(ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available algorithms: {sorted(available)}"
        )

    algorithm_class = ALGORITHM_REGISTRY[name_lower]
    return algorithm_class(config, logger)


def list_algorithms() -> list:
    """
    List all available algorithm names.

    Returns:
        Sorted list of primary algorithm names
    """
    # Return only primary names (not aliases)
    primary_names = ['dds', 'pso', 'de', 'sce-ua', 'async_dds', 'nsga2', 'adam', 'lbfgs']
    return sorted(primary_names)


__all__ = [
    # Base class
    'OptimizationAlgorithm',
    # Algorithm classes
    'DDSAlgorithm',
    'PSOAlgorithm',
    'DEAlgorithm',
    'SCEUAAlgorithm',
    'AsyncDDSAlgorithm',
    'NSGA2Algorithm',
    'AdamAlgorithm',
    'LBFGSAlgorithm',
    # Factory functions
    'get_algorithm',
    'list_algorithms',
    # Registry
    'ALGORITHM_REGISTRY',
]
