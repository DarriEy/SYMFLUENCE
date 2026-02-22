"""Canonical validation constants for SYMFLUENCE configuration.

Single source of truth for valid algorithm names, metric names, and domain
definition methods. Consumed by Pydantic validators (root.py) and manager
validation methods so the same lists are never duplicated.
"""

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
})

VALID_DOMAIN_METHODS_CANONICAL = frozenset({
    'point', 'lumped', 'semidistributed', 'distributed',
})

VALID_DOMAIN_METHODS_WITH_LEGACY = frozenset({
    'point', 'lumped', 'semidistributed', 'distributed',
    'discretized', 'distribute', 'subset', 'delineate',
})
