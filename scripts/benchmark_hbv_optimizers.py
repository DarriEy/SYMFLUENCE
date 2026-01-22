#!/usr/bin/env python3
"""
HBV Calibration Benchmark - Optimizer Comparison

Compares multiple optimization algorithms on HBV model calibration:
- Derivative-free: DDS, PSO, DE, SCE-UA
- Gradient-based: Adam (using JAX autodiff)

HBV is ideal for this comparison because:
1. Fast JAX implementation (~0.01s per simulation)
2. Native gradient support via autodiff
3. Well-defined parameter bounds
"""

import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
import xarray as xr

# JAX imports
import jax
import jax.numpy as jnp
from jax import grad, jit


@dataclass
class CalibrationResult:
    """Store results from a single calibration run."""
    algorithm: str
    best_params: Dict[str, float]
    best_score: float
    convergence_history: List[float]
    n_evaluations: int
    runtime_seconds: float
    final_kge: float
    final_nse: float


class HBVModel:
    """Simple HBV model implementation in JAX for fast calibration."""

    # Parameter bounds
    PARAM_BOUNDS = {
        'FC': (50.0, 700.0),      # Field capacity (mm)
        'LP': (0.3, 1.0),          # Limit for potential ET
        'BETA': (1.0, 6.0),        # Shape coefficient
        'K0': (0.05, 0.9),         # Fast reservoir coefficient
        'K1': (0.01, 0.5),         # Slow reservoir coefficient
        'K2': (0.001, 0.2),        # Baseflow coefficient
        'UZL': (0.0, 100.0),       # Upper zone threshold
        'PERC': (0.0, 8.0),        # Percolation rate
        'MAXBAS': (1.0, 7.0),      # Routing parameter
    }

    PARAM_NAMES = list(PARAM_BOUNDS.keys())

    def __init__(self, forcing_path: Path, warmup_days: int = 365):
        """Initialize HBV model with forcing data."""
        ds = xr.open_dataset(forcing_path)

        # Load forcing
        self.precip = jnp.array(ds['precip'].values.flatten())
        self.temp = jnp.array(ds['temp'].values.flatten())
        self.pet = jnp.array(ds['pet'].values.flatten())
        self.q_obs = jnp.array(ds['q_obs'].values.flatten())
        self.n_timesteps = len(self.precip)
        ds.close()

        self.warmup_days = warmup_days
        self.n_evals = 0

        # JIT compile the simulation
        self._simulate_jit = jit(self._simulate_single)

        # Create gradient function
        self._loss_fn = lambda p: -self._kge_from_params(p)
        self._grad_fn = jit(grad(self._loss_fn))

    def _simulate_single(self, params: jnp.ndarray) -> jnp.ndarray:
        """Run HBV simulation with JAX (single parameter set)."""
        # Unpack parameters
        FC, LP, BETA, K0, K1, K2, UZL, PERC, MAXBAS = params

        # Initialize states
        SM = FC * 0.5   # Soil moisture
        SUZ = 0.0       # Upper zone storage
        SLZ = 0.0       # Lower zone storage

        # Snow parameters (simplified - no snow for now)
        TT = 0.0        # Threshold temperature
        CFMAX = 3.0     # Degree-day factor

        q_sim = jnp.zeros(self.n_timesteps)

        def step(carry, inputs):
            SM, SUZ, SLZ = carry
            P, T, PET = inputs

            # Snow routine (simplified)
            rain = jnp.where(T > TT, P, 0.0)
            snow = jnp.where(T <= TT, P, 0.0)
            melt = jnp.minimum(snow * 0.1, CFMAX * jnp.maximum(0.0, T - TT))

            # Effective precipitation
            P_eff = rain + melt

            # Soil moisture routine
            recharge = P_eff * (SM / FC) ** BETA
            SM_new = SM + P_eff - recharge

            # Limit SM to FC
            excess = jnp.maximum(0.0, SM_new - FC)
            SM_new = jnp.minimum(SM_new, FC)
            recharge = recharge + excess

            # Actual ET
            ET = PET * jnp.minimum(SM_new / (FC * LP), 1.0)
            SM_new = jnp.maximum(0.0, SM_new - ET)

            # Upper zone
            SUZ_new = SUZ + recharge

            # Percolation to lower zone
            perc = jnp.minimum(PERC, SUZ_new)
            SUZ_new = SUZ_new - perc
            SLZ_new = SLZ + perc

            # Runoff components
            Q0 = K0 * jnp.maximum(0.0, SUZ_new - UZL)  # Fast response
            Q1 = K1 * SUZ_new                           # Interflow
            Q2 = K2 * SLZ_new                           # Baseflow

            SUZ_new = SUZ_new - Q0 - Q1
            SLZ_new = SLZ_new - Q2

            # Total runoff
            Q = Q0 + Q1 + Q2

            return (SM_new, jnp.maximum(0.0, SUZ_new), jnp.maximum(0.0, SLZ_new)), Q

        # Run simulation
        inputs = (self.precip, self.temp, self.pet)
        _, q_sim = jax.lax.scan(step, (SM, SUZ, SLZ), inputs)

        return q_sim

    def simulate(self, params_dict: Dict[str, float]) -> np.ndarray:
        """Run simulation with parameter dictionary."""
        self.n_evals += 1
        params_arr = jnp.array([params_dict[name] for name in self.PARAM_NAMES])
        q_sim = self._simulate_jit(params_arr)
        return np.array(q_sim)

    def _kge_from_params(self, params: jnp.ndarray) -> float:
        """Calculate KGE from parameter array (for gradient computation)."""
        q_sim = self._simulate_single(params)

        # Evaluation period (skip warmup)
        qs = q_sim[self.warmup_days:]
        qo = self.q_obs[self.warmup_days:]

        # KGE calculation
        r = jnp.corrcoef(qo, qs)[0, 1]
        alpha = jnp.std(qs) / jnp.std(qo)
        beta = jnp.mean(qs) / jnp.mean(qo)

        kge = 1 - jnp.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return kge

    def calc_kge(self, q_sim: np.ndarray) -> float:
        """Calculate KGE."""
        qs = q_sim[self.warmup_days:]
        qo = np.array(self.q_obs[self.warmup_days:])

        valid = ~np.isnan(qo) & ~np.isnan(qs)
        qs, qo = qs[valid], qo[valid]

        r = np.corrcoef(qo, qs)[0, 1]
        alpha = np.std(qs) / np.std(qo) if np.std(qo) > 0 else 0
        beta = np.mean(qs) / np.mean(qo) if np.mean(qo) > 0 else 0

        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    def calc_nse(self, q_sim: np.ndarray) -> float:
        """Calculate NSE."""
        qs = q_sim[self.warmup_days:]
        qo = np.array(self.q_obs[self.warmup_days:])

        valid = ~np.isnan(qo) & ~np.isnan(qs)
        qs, qo = qs[valid], qo[valid]

        return 1 - np.sum((qo - qs)**2) / np.sum((qo - np.mean(qo))**2)

    def normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1]."""
        normalized = []
        for name in self.PARAM_NAMES:
            lo, hi = self.PARAM_BOUNDS[name]
            val = params.get(name, (lo + hi) / 2)
            normalized.append((val - lo) / (hi - lo))
        return np.array(normalized)

    def denormalize_params(self, normalized: np.ndarray) -> Dict[str, float]:
        """Denormalize parameters from [0, 1]."""
        params = {}
        for i, name in enumerate(self.PARAM_NAMES):
            lo, hi = self.PARAM_BOUNDS[name]
            params[name] = lo + np.clip(normalized[i], 0, 1) * (hi - lo)
        return params

    def objective(self, normalized: np.ndarray) -> float:
        """Objective function (minimize negative KGE)."""
        params = self.denormalize_params(normalized)
        q_sim = self.simulate(params)
        return -self.calc_kge(q_sim)

    def gradient(self, normalized: np.ndarray) -> np.ndarray:
        """Compute gradient of objective using JAX autodiff."""
        params = self.denormalize_params(normalized)
        params_arr = jnp.array([params[name] for name in self.PARAM_NAMES])

        # Get gradient in parameter space
        grad_params = self._grad_fn(params_arr)

        # Transform gradient to normalized space
        grad_norm = []
        for i, name in enumerate(self.PARAM_NAMES):
            lo, hi = self.PARAM_BOUNDS[name]
            grad_norm.append(float(grad_params[i]) * (hi - lo))

        return np.array(grad_norm)


# ============================================================================
# OPTIMIZATION ALGORITHMS
# ============================================================================

def run_dds(objective: Callable, n_params: int, max_iter: int = 500, r: float = 0.2) -> Tuple[np.ndarray, List[float]]:
    """Dynamically Dimensioned Search."""
    x_best = np.random.uniform(0, 1, n_params)
    f_best = objective(x_best)
    history = [f_best]

    for i in range(1, max_iter):
        p = 1 - np.log(i) / np.log(max_iter)

        perturb = np.random.uniform(0, 1, n_params) < p
        if not np.any(perturb):
            perturb[np.random.randint(n_params)] = True

        x_new = x_best.copy()
        for j in range(n_params):
            if perturb[j]:
                x_new[j] = x_best[j] + r * np.random.normal()
                if x_new[j] < 0:
                    x_new[j] = abs(x_new[j])
                if x_new[j] > 1:
                    x_new[j] = 2 - x_new[j]
                x_new[j] = np.clip(x_new[j], 0, 1)

        f_new = objective(x_new)
        if f_new < f_best:
            x_best, f_best = x_new, f_new

        history.append(f_best)

    return x_best, history


def run_pso(objective: Callable, n_params: int, max_iter: int = 100, n_particles: int = 20) -> Tuple[np.ndarray, List[float]]:
    """Particle Swarm Optimization."""
    w, c1, c2 = 0.7, 1.5, 1.5

    pos = np.random.uniform(0, 1, (n_particles, n_params))
    vel = np.random.uniform(-0.1, 0.1, (n_particles, n_params))

    fitness = np.array([objective(p) for p in pos])
    pbest_pos, pbest_fit = pos.copy(), fitness.copy()

    gbest_idx = np.argmin(fitness)
    gbest_pos, gbest_fit = pos[gbest_idx].copy(), fitness[gbest_idx]

    history = [gbest_fit]

    for _ in range(max_iter):
        r1, r2 = np.random.random((2, n_particles, n_params))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, 0, 1)

        fitness = np.array([objective(p) for p in pos])

        improved = fitness < pbest_fit
        pbest_pos[improved], pbest_fit[improved] = pos[improved], fitness[improved]

        if np.min(fitness) < gbest_fit:
            gbest_idx = np.argmin(fitness)
            gbest_pos, gbest_fit = pos[gbest_idx].copy(), fitness[gbest_idx]

        history.append(gbest_fit)

    return gbest_pos, history


def run_de(objective: Callable, n_params: int, max_iter: int = 100, n_pop: int = 20) -> Tuple[np.ndarray, List[float]]:
    """Differential Evolution."""
    F, CR = 0.8, 0.9

    pop = np.random.uniform(0, 1, (n_pop, n_params))
    fitness = np.array([objective(p) for p in pop])

    best_idx = np.argmin(fitness)
    best, best_fit = pop[best_idx].copy(), fitness[best_idx]

    history = [best_fit]

    for _ in range(max_iter):
        for i in range(n_pop):
            candidates = [j for j in range(n_pop) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)

            mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), 0, 1)

            cross = np.random.random(n_params) < CR
            if not np.any(cross):
                cross[np.random.randint(n_params)] = True
            trial = np.where(cross, mutant, pop[i])

            trial_fit = objective(trial)
            if trial_fit < fitness[i]:
                pop[i], fitness[i] = trial, trial_fit
                if trial_fit < best_fit:
                    best, best_fit = trial.copy(), trial_fit

        history.append(best_fit)

    return best, history


def run_adam(objective: Callable, gradient: Callable, n_params: int,
             max_iter: int = 200, lr: float = 0.02) -> Tuple[np.ndarray, List[float]]:
    """Adam optimizer with gradient."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    x = np.random.uniform(0.2, 0.8, n_params)
    m, v = np.zeros(n_params), np.zeros(n_params)

    f_best = objective(x)
    x_best = x.copy()
    history = [f_best]

    for t in range(1, max_iter + 1):
        g = gradient(x)

        # Clip gradient for stability
        g = np.clip(g, -10, 10)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        x = np.clip(x, 0, 1)

        f = objective(x)
        if f < f_best:
            f_best, x_best = f, x.copy()

        history.append(f_best)

    return x_best, history


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(forcing_path: Path, output_dir: Path, max_evals: int = 500):
    """Run full benchmark comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HBV CALIBRATION BENCHMARK - OPTIMIZER COMPARISON")
    print("=" * 70)
    print(f"Forcing: {forcing_path}")
    print(f"Output: {output_dir}")
    print(f"Max evaluations: {max_evals}")
    print()

    # Initialize model
    print("Loading HBV model...")
    model = HBVModel(forcing_path)
    n_params = len(model.PARAM_NAMES)
    print(f"Parameters: {n_params} ({model.PARAM_NAMES})")
    print(f"Timesteps: {model.n_timesteps}")
    print()

    # Test simulation speed
    print("Testing simulation speed...")
    model.n_evals = 0
    start = time.time()
    for _ in range(10):
        _ = model.simulate(model.denormalize_params(np.random.uniform(0, 1, n_params)))
    avg_time = (time.time() - start) / 10
    print(f"Average simulation time: {avg_time*1000:.1f} ms")
    print()

    results: List[CalibrationResult] = []

    # Define algorithms
    algorithms = [
        ('DDS', lambda: run_dds(model.objective, n_params, max_iter=max_evals)),
        ('PSO', lambda: run_pso(model.objective, n_params, max_iter=max_evals//20, n_particles=20)),
        ('DE', lambda: run_de(model.objective, n_params, max_iter=max_evals//20, n_pop=20)),
        ('Adam', lambda: run_adam(model.objective, model.gradient, n_params, max_iter=max_evals//2, lr=0.02)),
    ]

    # Run each algorithm
    for name, run_fn in algorithms:
        print(f"Running {name}...")
        model.n_evals = 0

        start = time.time()
        best_norm, history = run_fn()
        runtime = time.time() - start

        best_params = model.denormalize_params(best_norm)
        q_sim = model.simulate(best_params)
        final_kge = model.calc_kge(q_sim)
        final_nse = model.calc_nse(q_sim)

        result = CalibrationResult(
            algorithm=name,
            best_params=best_params,
            best_score=-history[-1],  # Convert to KGE
            convergence_history=[-h for h in history],
            n_evaluations=model.n_evals,
            runtime_seconds=runtime,
            final_kge=final_kge,
            final_nse=final_nse
        )
        results.append(result)

        print(f"  KGE: {final_kge:.4f}, NSE: {final_nse:.4f}")
        print(f"  Evaluations: {model.n_evals}, Time: {runtime:.1f}s")
        print()

    # Save results (convert numpy/jax types to native Python)
    results_data = [{
        'algorithm': r.algorithm,
        'final_kge': float(r.final_kge),
        'final_nse': float(r.final_nse),
        'n_evaluations': int(r.n_evaluations),
        'runtime_seconds': float(r.runtime_seconds),
        'best_params': {k: float(v) for k, v in r.best_params.items()}
    } for r in results]

    with open(output_dir / 'hbv_benchmark_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    # Generate plots
    generate_plots(results, model, output_dir)

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<10} {'KGE':>8} {'NSE':>8} {'Evals':>8} {'Time':>10} {'Evals/s':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: -x.final_kge):
        evals_per_sec = r.n_evaluations / r.runtime_seconds if r.runtime_seconds > 0 else 0
        print(f"{r.algorithm:<10} {r.final_kge:>8.4f} {r.final_nse:>8.4f} {r.n_evaluations:>8} {r.runtime_seconds:>9.1f}s {evals_per_sec:>10.1f}")

    return results


def generate_plots(results: List[CalibrationResult], model: HBVModel, output_dir: Path):
    """Generate comparison plots."""

    fig = plt.figure(figsize=(16, 12))

    # 1. Convergence curves
    ax1 = fig.add_subplot(2, 2, 1)
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for r, c in zip(results, colors):
        ax1.plot(r.convergence_history, color=c,
                label=f'{r.algorithm} (KGE={r.final_kge:.3f})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('KGE')
    ax1.set_title('Convergence Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Performance bar chart
    ax2 = fig.add_subplot(2, 2, 2)
    algorithms = [r.algorithm for r in results]
    kge_vals = [r.final_kge for r in results]
    nse_vals = [r.final_nse for r in results]

    x = np.arange(len(algorithms))
    width = 0.35
    ax2.bar(x - width/2, kge_vals, width, label='KGE', color='steelblue')
    ax2.bar(x + width/2, nse_vals, width, label='NSE', color='darkorange')
    ax2.set_ylabel('Score')
    ax2.set_title('Final Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Efficiency comparison
    ax3 = fig.add_subplot(2, 2, 3)
    runtimes = [r.runtime_seconds for r in results]
    evals = [r.n_evaluations for r in results]

    ax3.bar(algorithms, runtimes, color=colors)
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('Computational Cost')
    ax3.grid(True, alpha=0.3, axis='y')

    for i, (rt, ev) in enumerate(zip(runtimes, evals)):
        ax3.text(i, rt + 0.5, f'{ev} evals', ha='center', fontsize=9)

    # 4. Best hydrograph
    ax4 = fig.add_subplot(2, 2, 4)

    best_result = max(results, key=lambda r: r.final_kge)
    q_sim = model.simulate(best_result.best_params)
    q_obs = np.array(model.q_obs)

    # Plot first year after warmup
    start = model.warmup_days
    end = start + 365
    t = np.arange(end - start)

    ax4.plot(t, q_obs[start:end], 'b-', alpha=0.7, label='Observed', linewidth=1)
    ax4.plot(t, q_sim[start:end], 'r-', alpha=0.7,
            label=f'Simulated ({best_result.algorithm})', linewidth=1)
    ax4.set_xlabel('Day of Year')
    ax4.set_ylabel('Discharge (mm/day)')
    ax4.set_title(f'Hydrograph - Best Result (KGE={best_result.final_kge:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'hbv_optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'hbv_optimizer_comparison.png'}")

    # Parameter comparison plot
    fig2, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for i, name in enumerate(model.PARAM_NAMES):
        ax = axes[i]
        values = [r.best_params[name] for r in results]
        lo, hi = model.PARAM_BOUNDS[name]

        ax.bar(algorithms, values, color=colors)
        ax.axhline(y=(lo + hi) / 2, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel(name)
        ax.set_title(f'{name} [{lo:.1f}, {hi:.1f}]')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'hbv_parameter_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'hbv_parameter_comparison.png'}")

    plt.close('all')


def main():
    """Main entry point."""
    forcing_path = Path('/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Bow_at_Banff_lumped_casr/forcing/FUSE_input/Bow_at_Banff_lumped_casr_input.nc')
    output_dir = Path('/Users/darrieythorsson/compHydro/code/SYMFLUENCE/results/hbv_benchmark')

    _results = run_benchmark(forcing_path, output_dir, max_evals=500)


if __name__ == '__main__':
    main()
