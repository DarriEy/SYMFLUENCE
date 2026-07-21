# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
from __future__ import annotations

"""
Reproducibility of the population optimizers (NSGA-II, MOEA/D, GA).

These drew selection/crossover/mutation randomness from the process-global
``np.random``, which is seeded once and then shared with everything else in the
process. Anything that perturbs that state between generations — most commonly a
model that crashes on some parameter sets and not others, changing how many
draws happen — desynchronised the run. Two runs of the same config on the same
machine produced KGE 0.843 vs 0.728. Each algorithm now owns a per-run
Generator, so its result is invariant to the global state.
"""

import logging

import numpy as np
import pytest

from symfluence.optimization.optimizers.algorithms.ga import GAAlgorithm
from symfluence.optimization.optimizers.algorithms.moead import MOEADAlgorithm
from symfluence.optimization.optimizers.algorithms.nsga2 import NSGA2Algorithm

CONFIG = {
    "OPTIMIZATION_METRIC": "KGE",
    "NUMBER_OF_ITERATIONS": 12,
    "POPULATION_SIZE": 16,
    "RANDOM_SEED": 42,
}


def _multi_objective(pop, names=None, iteration=0):
    """Analytic two-objective surface — deterministic in the parameters."""
    p = np.atleast_2d(pop)
    return np.column_stack([
        1.0 - np.mean((p - 0.3) ** 2, axis=1),
        1.0 - np.mean((p - 0.7) ** 2, axis=1),
    ])


def _single_objective(pop, iteration=0):
    return _multi_objective(pop)[:, 0]


def _noop(*args, **kwargs):
    return None


def _denorm(x):
    return {f"p{i}": float(v) for i, v in enumerate(np.atleast_1d(x))}


def _run(algo_cls, *, perturb_global):
    # Perturb the process-global RNG by a caller-chosen amount before the run.
    # A correctly-isolated algorithm ignores it; the old global-RNG code did not.
    np.random.seed(perturb_global)
    for _ in range(perturb_global % 23):
        np.random.random()

    algo = algo_cls(CONFIG, logging.getLogger("repro-test"))
    result = algo.optimize(
        n_params=6,
        evaluate_solution=lambda x, i: float(_single_objective(x.reshape(1, -1))[0]),
        evaluate_population=_single_objective,
        denormalize_params=_denorm,
        record_iteration=_noop,
        update_best=_noop,
        log_progress=_noop,
        evaluate_population_objectives=_multi_objective,
        objective_names=["KGE", "NSE"],
        multiobjective=True,
    )
    return result.get("best_score", result.get("best_fitness"))


@pytest.mark.parametrize("algo_cls", [NSGA2Algorithm, MOEADAlgorithm, GAAlgorithm])
def test_result_is_invariant_to_global_rng_state(algo_cls):
    a = _run(algo_cls, perturb_global=999)
    b = _run(algo_cls, perturb_global=3)

    assert a == pytest.approx(b, abs=1e-12), (
        f"{algo_cls.__name__} result changed with the global RNG state "
        f"({a} vs {b}); its randomness is not isolated"
    )


@pytest.mark.parametrize("algo_cls", [NSGA2Algorithm, MOEADAlgorithm, GAAlgorithm])
def test_same_seed_reproduces(algo_cls):
    assert _run(algo_cls, perturb_global=1) == pytest.approx(
        _run(algo_cls, perturb_global=1), abs=1e-12)
