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
from .cmaes import CMAESAlgorithm
from .dream import DREAMAlgorithm
from .glue import GLUEAlgorithm
from .basin_hopping import BasinHoppingAlgorithm
from .nelder_mead import NelderMeadAlgorithm
from .ga import GAAlgorithm
from .bayesian_optimization import BayesianOptimizationAlgorithm
from .moead import MOEADAlgorithm
from .simulated_annealing import SimulatedAnnealingAlgorithm
from .abc import ABCAlgorithm


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
    'cmaes': CMAESAlgorithm,
    'cma-es': CMAESAlgorithm,  # Alternative name
    'dream': DREAMAlgorithm,
    'glue': GLUEAlgorithm,
    'basin_hopping': BasinHoppingAlgorithm,
    'basinhopping': BasinHoppingAlgorithm,  # Alternative name
    'bh': BasinHoppingAlgorithm,  # Short alias
    'nelder_mead': NelderMeadAlgorithm,
    'neldermead': NelderMeadAlgorithm,  # Alternative name
    'nelder-mead': NelderMeadAlgorithm,  # Alternative name
    'nm': NelderMeadAlgorithm,  # Short alias
    'simplex': NelderMeadAlgorithm,  # Alternative name
    'ga': GAAlgorithm,
    'bayesian_opt': BayesianOptimizationAlgorithm,
    'bayesian': BayesianOptimizationAlgorithm,  # Alternative name
    'bo': BayesianOptimizationAlgorithm,  # Short alias
    'moead': MOEADAlgorithm,
    'moea_d': MOEADAlgorithm,  # Alternative name
    'moea-d': MOEADAlgorithm,  # Alternative name
    'simulated_annealing': SimulatedAnnealingAlgorithm,
    'sa': SimulatedAnnealingAlgorithm,  # Short alias
    'annealing': SimulatedAnnealingAlgorithm,  # Alternative name
    'abc': ABCAlgorithm,
    'abc_smc': ABCAlgorithm,  # Alternative name
    'approximate_bayesian': ABCAlgorithm,  # Alternative name
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
    primary_names = ['dds', 'pso', 'de', 'sce-ua', 'async_dds', 'nsga2', 'adam', 'lbfgs', 'cmaes', 'dream', 'glue', 'basin_hopping', 'nelder_mead', 'ga', 'bayesian_opt', 'moead', 'simulated_annealing', 'abc']
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
    'CMAESAlgorithm',
    'DREAMAlgorithm',
    'GLUEAlgorithm',
    'BasinHoppingAlgorithm',
    'NelderMeadAlgorithm',
    'GAAlgorithm',
    'BayesianOptimizationAlgorithm',
    'MOEADAlgorithm',
    'SimulatedAnnealingAlgorithm',
    'ABCAlgorithm',
    # Factory functions
    'get_algorithm',
    'list_algorithms',
    # Registry
    'ALGORITHM_REGISTRY',
]
