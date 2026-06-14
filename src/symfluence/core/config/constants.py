# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical validation constants for SYMFLUENCE configuration.

Single source of truth for valid algorithm names, metric names, and domain
definition methods. Consumed by Pydantic validators (root.py) and manager
validation methods so the same lists are never duplicated.
"""
from __future__ import annotations

VALID_OPTIMIZATION_ALGORITHMS = frozenset({
    'PSO', 'DE', 'DDS', 'ASYNC-DDS', 'ASYNCDDS', 'ASYNC_DDS',
    'SCE-UA', 'SCEUA', 'NSGA-II', 'NSGA2',
    'ADAM', 'LBFGS', 'CMA-ES', 'CMAES', 'DREAM', 'GLUE',
    'BASIN-HOPPING', 'BASINHOPPING', 'BH',
    'NELDER-MEAD', 'NELDERMEAD', 'NM', 'SIMPLEX', 'GA',
    'BAYESIAN-OPT', 'BAYESIAN_OPT', 'BAYESIAN', 'BO',
    'MOEAD', 'MOEA-D', 'MOEA_D',
    'SIMULATED-ANNEALING', 'SIMULATED_ANNEALING', 'SA', 'ANNEALING',
    'ABC', 'ABC-SMC', 'ABC_SMC', 'APPROXIMATE-BAYESIAN',
})

VALID_OPTIMIZATION_METRICS = frozenset({
    'KGE', 'KGEP', 'NSE', 'RMSE', 'MAE', 'PBIAS', 'R2', 'CORRELATION',
    'COMPOSITE',
})

VALID_DOMAIN_METHODS_CANONICAL = frozenset({
    'point', 'lumped', 'semidistributed', 'distributed',
})

VALID_DOMAIN_METHODS_WITH_LEGACY = frozenset({
    'point', 'lumped', 'semidistributed', 'distributed',
    'discretized', 'distribute', 'subset', 'delineate',
    # Spelling variants — several paper configs write these
    'semi_distributed', 'semi-distributed',
})

# Canonical attribute-acquisition profile names. The data layer
# (data/acquisition/attribute_profiles.py) defines the name -> dataset mapping
# and asserts its keys match this set; the config validator reads this constant
# so it need not import the data layer (preserves core -> data layering).
VALID_ATTRIBUTE_PROFILES = frozenset({
    'core', 'camels_spat', 'full',
})
