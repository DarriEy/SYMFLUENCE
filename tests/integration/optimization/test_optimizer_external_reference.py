# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""External reference checks for SYMFLUENCE optimization algorithms.

Each SYMFLUENCE optimizer is cross-validated against an independent, widely-used
implementation of the same algorithm family on test functions with known optima:

    SYMFLUENCE            External reference
    ---------            ------------------
    L-BFGS               scipy.optimize  L-BFGS-B
    Nelder-Mead          scipy.optimize  Nelder-Mead
    Adam                 torch.optim.Adam
    Differential Evol.   scipy.optimize  differential_evolution
    Simulated Annealing  scipy.optimize  dual_annealing
    Basin-Hopping        scipy.optimize  basinhopping
    SCE-UA               spotpy.algorithms.sceua

SYMFLUENCE maximizes a fitness ``f``; the reference libraries minimize ``g = -f``.
The invariant checked is agreement on the *located optimum*: both independent
implementations must converge to the same point, and to the known analytic
optimum. This is a behavioural correctness check — e.g. the L-BFGS maximization
sign regression (secant history never populating, silent decay to gradient
ascent) surfaces here as a far-from-optimum solution or disagreement.

External libraries are optional; each test skips if its reference is unavailable.
"""
from __future__ import annotations

import contextlib
import io
import logging

import numpy as np
import pytest

scipy_optimize = pytest.importorskip("scipy.optimize")


@pytest.fixture
def logger():
    lg = logging.getLogger("test_optimizer_external_reference")
    lg.setLevel(logging.ERROR)
    return lg


# ---------------------------------------------------------------------------
# Test objectives — fitness to MAXIMIZE on the normalized box [0, 1]^n.
# Each returns (fitness_fn, analytic_optimum).
# ---------------------------------------------------------------------------

def sphere_interior(n):
    """Isotropic bowl with the optimum in the interior of the box."""
    opt = np.linspace(0.2, 0.8, n)
    return (lambda x: -np.sum((x - opt) ** 2)), opt


def ill_conditioned(n=3):
    """Anisotropic bowl, condition number 1000 — stresses step scaling."""
    curv = np.array([1.0, 30.0, 1000.0])[:n]
    opt = np.full(n, 0.6)
    return (lambda x: -np.sum(curv * (x - opt) ** 2)), opt


def rosenbrock(n):
    """Rosenbrock on [0,1]^n mapped to [-2,2]^n; optimum (1,..) -> x_norm 0.75."""
    opt = np.full(n, 0.75)

    def f(x):
        xd = x * 4 - 2
        return -sum(100.0 * (xd[i + 1] - xd[i] ** 2) ** 2 + (1 - xd[i]) ** 2
                    for i in range(len(x) - 1))

    return f, opt


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------

def _run_symfluence(cls, cfg, n, fitness, logger, seed=0, **extra):
    """Run a SYMFLUENCE algorithm and return (best_solution, best_score).

    Stochastic algorithms draw from the global numpy RNG, so we seed it for
    reproducibility.
    """
    np.random.seed(seed)
    algo = cls(cfg, logger)
    result = algo.optimize(
        n_params=n,
        evaluate_solution=lambda x, i=0: fitness(x),
        evaluate_population=lambda p, i=0: np.array([fitness(x) for x in p]),
        denormalize_params=lambda x: {f"p{i}": v for i, v in enumerate(x)},
        record_iteration=lambda *a, **k: None,
        update_best=lambda *a, **k: None,
        log_progress=lambda *a, **k: None,
        **extra,
    )
    return np.asarray(result["best_solution"]), result["best_score"]


def _assert_agrees(name, x_sym, x_ref, optimum, tol):
    """Both optimizers must agree with each other and with the analytic optimum."""
    assert np.linalg.norm(x_sym - x_ref) < tol, (
        f"{name}: SYMFLUENCE {np.round(x_sym, 4)} disagrees with reference "
        f"{np.round(x_ref, 4)}"
    )
    assert np.linalg.norm(x_sym - optimum) < tol, (
        f"{name}: SYMFLUENCE {np.round(x_sym, 4)} did not reach optimum "
        f"{np.round(optimum, 4)}"
    )


# ---------------------------------------------------------------------------
# L-BFGS  vs  scipy L-BFGS-B
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("problem,n", [
    ("sphere", 3), ("ill_conditioned", 3), ("rosenbrock", 2),
])
def test_lbfgs_vs_scipy(problem, n, logger):
    from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

    fitness, optimum = {
        "sphere": sphere_interior(n),
        "ill_conditioned": ill_conditioned(n),
        "rosenbrock": rosenbrock(n),
    }[problem]

    cfg = {"GRADIENT_EPSILON": 1e-7, "GRADIENT_CLIP_VALUE": 0.0,
           "LBFGS_STEPS": 200, "LBFGS_LR": 1.0, "LBFGS_HISTORY_SIZE": 10}
    x_sym, _ = _run_symfluence(LBFGSAlgorithm, cfg, n, fitness, logger,
                               gradient_mode="finite_difference")

    # scipy from several starts; take the best (parity with SYMFLUENCE's FD run)
    rng = np.random.default_rng(0)
    best = None
    for x0 in [np.full(n, 0.5)] + [rng.random(n) for _ in range(4)]:
        r = scipy_optimize.minimize(lambda x: -fitness(x), x0,
                                    method="L-BFGS-B", bounds=[(0, 1)] * n)
        if best is None or -r.fun > best[1]:
            best = (r.x, -r.fun)

    _assert_agrees(f"LBFGS/{problem}", x_sym, best[0], optimum, tol=1e-2)


# ---------------------------------------------------------------------------
# Nelder-Mead  vs  scipy Nelder-Mead
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("problem,n", [
    ("sphere", 3), ("ill_conditioned", 3), ("rosenbrock", 2),
])
def test_nelder_mead_vs_scipy(problem, n, logger):
    from symfluence.core.calibration.optimizers.algorithms.nelder_mead import NelderMeadAlgorithm

    fitness, optimum = {
        "sphere": sphere_interior(n),
        "ill_conditioned": ill_conditioned(n),
        "rosenbrock": rosenbrock(n),
    }[problem]

    cfg = {"NUMBER_OF_ITERATIONS": 400, "NM_SIMPLEX_SIZE": 0.1}
    x_sym, _ = _run_symfluence(NelderMeadAlgorithm, cfg, n, fitness, logger)

    r = scipy_optimize.minimize(
        lambda x: -fitness(x), np.full(n, 0.5), method="Nelder-Mead",
        bounds=[(0, 1)] * n,
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
    )
    _assert_agrees(f"NelderMead/{problem}", x_sym, r.x, optimum, tol=2e-2)


# ---------------------------------------------------------------------------
# Adam  vs  torch.optim.Adam  (identical algorithm, independent implementation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("problem,n", [("sphere", 3), ("ill_conditioned", 3)])
def test_adam_vs_torch(problem, n, logger):
    torch = pytest.importorskip("torch")
    from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

    if problem == "sphere":
        fitness, optimum = sphere_interior(n)
        curv = np.ones(n)
    else:
        fitness, optimum = ill_conditioned(n)
        curv = np.array([1.0, 30.0, 1000.0])[:n]

    steps, lr = 1500, 0.02
    cfg = {"NUMBER_OF_ITERATIONS": steps, "ADAM_LR": lr,
           "GRADIENT_EPSILON": 1e-6, "GRADIENT_CLIP_VALUE": 0.0}
    x_sym, _ = _run_symfluence(AdamAlgorithm, cfg, n, fitness, logger,
                               gradient_mode="finite_difference")

    # torch Adam minimizing the identical quadratic loss from the same start
    x = torch.full((n,), 0.5, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    t_opt = torch.tensor(optimum)
    t_curv = torch.tensor(curv)
    for _ in range(steps):
        opt.zero_grad()
        (t_curv * (x - t_opt) ** 2).sum().backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)

    _assert_agrees(f"Adam/{problem}", x_sym, x.detach().numpy(), optimum, tol=3e-2)


# ---------------------------------------------------------------------------
# Differential Evolution  vs  scipy differential_evolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("problem,n", [("sphere", 3), ("rosenbrock", 2)])
def test_de_vs_scipy(problem, n, logger):
    from symfluence.core.calibration.optimizers.algorithms.de import DEAlgorithm

    fitness, optimum = (sphere_interior(n) if problem == "sphere"
                        else rosenbrock(n))

    cfg = {"NUMBER_OF_ITERATIONS": 120, "POPULATION_SIZE": 25,
           "DE_SCALING_FACTOR": 0.7, "DE_CROSSOVER_RATE": 0.9}
    x_sym, _ = _run_symfluence(DEAlgorithm, cfg, n, fitness, logger, seed=1)

    r = scipy_optimize.differential_evolution(
        lambda x: -fitness(x), [(0, 1)] * n, seed=1, maxiter=300, tol=1e-10)
    _assert_agrees(f"DE/{problem}", x_sym, r.x, optimum, tol=3e-2)


# ---------------------------------------------------------------------------
# Simulated Annealing  vs  scipy dual_annealing
# ---------------------------------------------------------------------------

def test_simulated_annealing_vs_scipy(logger):
    from symfluence.core.calibration.optimizers.algorithms.simulated_annealing import (
        SimulatedAnnealingAlgorithm,
    )

    n = 3
    fitness, optimum = sphere_interior(n)

    cfg = {"NUMBER_OF_ITERATIONS": 2000, "SA_STEP_SIZE": 0.15}
    x_sym, _ = _run_symfluence(SimulatedAnnealingAlgorithm, cfg, n, fitness,
                               logger, seed=2)

    r = scipy_optimize.dual_annealing(
        lambda x: -fitness(x), [(0, 1)] * n, seed=2, maxiter=1000)
    _assert_agrees("SimulatedAnnealing/sphere", x_sym, r.x, optimum, tol=5e-2)


# ---------------------------------------------------------------------------
# Basin-Hopping  vs  scipy basinhopping
# ---------------------------------------------------------------------------

def test_basin_hopping_vs_scipy(logger):
    from symfluence.core.calibration.optimizers.algorithms.basin_hopping import (
        BasinHoppingAlgorithm,
    )

    n = 3
    fitness, optimum = sphere_interior(n)

    cfg = {"NUMBER_OF_ITERATIONS": 60, "BH_STEP_SIZE": 0.3, "BH_LOCAL_STEPS": 30}
    x_sym, _ = _run_symfluence(BasinHoppingAlgorithm, cfg, n, fitness, logger, seed=3)

    r = scipy_optimize.basinhopping(
        lambda x: -fitness(x), np.full(n, 0.5), niter=100, seed=3,
        minimizer_kwargs={"method": "L-BFGS-B", "bounds": [(0, 1)] * n})
    _assert_agrees("BasinHopping/sphere", x_sym, r.x, optimum, tol=5e-2)


# ---------------------------------------------------------------------------
# SCE-UA  vs  spotpy.algorithms.sceua  (hydrology-native reference)
# ---------------------------------------------------------------------------

def test_sce_ua_vs_spotpy(logger):
    spotpy = pytest.importorskip("spotpy")
    from symfluence.core.calibration.optimizers.algorithms.sce_ua import SCEUAAlgorithm

    n = 3
    fitness, optimum = sphere_interior(n)

    cfg = {"NUMBER_OF_ITERATIONS": 50, "NUMBER_OF_COMPLEXES": 4}
    x_sym, _ = _run_symfluence(SCEUAAlgorithm, cfg, n, fitness, logger, seed=0)

    class _Setup:
        def __init__(self):
            self._params = [spotpy.parameter.Uniform(f"p{i}", 0.0, 1.0)
                            for i in range(n)]

        def parameters(self):
            return spotpy.parameter.generate(self._params)

        def simulation(self, vector):
            return list(np.asarray(vector, dtype=float))

        def evaluation(self):
            return list(optimum)

        def objectivefunction(self, simulation, evaluation):
            return float(np.sum((np.array(simulation) - np.array(evaluation)) ** 2))

    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        sampler = spotpy.algorithms.sceua(_Setup(), dbname="ram", dbformat="ram",
                                          save_sim=False)
        sampler.sample(2000, ngs=4)
    data = sampler.getdata()
    best = int(np.argmin(data["like1"]))
    x_ref = np.array([data[f"parp{i}"][best] for i in range(n)])

    _assert_agrees("SCE-UA/sphere", x_sym, x_ref, optimum, tol=5e-2)
