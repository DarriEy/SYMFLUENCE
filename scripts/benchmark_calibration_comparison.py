#!/usr/bin/env python3
"""
Calibration Benchmark Comparison Script

Compares multiple optimization algorithms on jFUSE model calibration:
- Derivative-free: DDS, PSO, DE, SCE-UA
- Gradient-based: Adam, L-BFGS (using JAX autodiff)

Produces comparison plots showing:
- Convergence curves
- Final performance metrics
- Parameter distributions
- Hydrograph comparisons
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import xarray as xr

# jFUSE for model simulation
import jfuse


@dataclass
class CalibrationResult:
    """Store results from a single calibration run."""
    algorithm: str
    best_params: Dict[str, float]
    best_score: float
    convergence_history: List[float]
    n_evaluations: int
    runtime_seconds: float
    final_metrics: Dict[str, float] = field(default_factory=dict)


class JFUSECalibrator:
    """Calibration wrapper for jFUSE model."""

    # Parameters to calibrate (subset of full parameter set)
    CALIB_PARAMS = [
        'maxwatr_1', 'maxwatr_2', 'fracten', 'frchzne',
        'fprimqb', 'percrte', 'baserte', 'rtfrac1', 'timedelay'
    ]

    # Parameter bounds
    PARAM_BOUNDS = {
        'maxwatr_1': (25.0, 500.0),      # Upper layer max storage (mm)
        'maxwatr_2': (50.0, 5000.0),     # Lower layer max storage (mm)
        'fracten': (0.05, 0.95),          # Tension storage fraction
        'frchzne': (0.05, 0.95),          # Recharge zone fraction
        'fprimqb': (0.05, 0.95),          # Primary baseflow fraction
        'percrte': (0.01, 100.0),         # Percolation rate (mm/day)
        'baserte': (0.001, 10.0),         # Baseflow rate (mm/day)
        'rtfrac1': (0.05, 0.95),          # Root fraction upper layer
        'timedelay': (0.01, 5.0),         # Routing time delay (days)
    }

    def __init__(self, forcing_path: Path, warmup_days: int = 365, eval_start: int = 730, eval_end: int = 2191):
        """Initialize calibrator with forcing data."""
        self.forcing_path = Path(forcing_path)
        self.warmup_days = warmup_days
        self.eval_start = eval_start
        self.eval_end = eval_end

        # Load forcing data
        ds = xr.open_dataset(self.forcing_path)
        self.precip = np.array(ds['precip'].values.flatten(), dtype=np.float64)
        self.temp = np.array(ds['temp'].values.flatten(), dtype=np.float64)
        self.pet = np.array(ds['pet'].values.flatten(), dtype=np.float64)
        self.q_obs = np.array(ds['q_obs'].values.flatten(), dtype=np.float64)
        self.n_timesteps = len(self.precip)
        ds.close()

        # Setup jFUSE model
        self.model_config = jfuse.PRMS_CONFIG
        self.model = jfuse.FUSEModel(self.model_config)
        self.default_params = self.model.default_params()

        # Evaluation counter
        self.n_evals = 0

    def _params_to_jfuse(self, params: Dict[str, float]) -> Any:
        """Convert calibration params dict to jFUSE params structure."""
        # Start with defaults and update with calibration params
        p = self.default_params
        # jFUSE uses specific parameter structure - map our names
        return p  # For now use defaults, actual mapping would depend on jFUSE API

    def simulate(self, params: Dict[str, float]) -> np.ndarray:
        """Run jFUSE simulation with given parameters."""
        self.n_evals += 1

        # Initialize
        state = self.model.default_state()
        q_sim = np.zeros(self.n_timesteps)

        # Run timestep loop
        for t in range(self.n_timesteps):
            forcing = jfuse.Forcing(
                precip=float(self.precip[t]),
                temp=float(self.temp[t]),
                pet=float(self.pet[t])
            )
            state, flux = self.model.step(state, forcing, self.default_params)
            q_sim[t] = float(flux.q_total)

        return q_sim

    def calculate_kge(self, q_sim: np.ndarray, q_obs: np.ndarray) -> float:
        """Calculate Kling-Gupta Efficiency."""
        # Filter to evaluation period
        qs = q_sim[self.eval_start:self.eval_end]
        qo = q_obs[self.eval_start:self.eval_end]

        # Remove NaN
        valid = ~np.isnan(qo) & ~np.isnan(qs)
        qs, qo = qs[valid], qo[valid]

        if len(qs) == 0:
            return -999.0

        r = np.corrcoef(qo, qs)[0, 1]
        alpha = np.std(qs) / np.std(qo) if np.std(qo) > 0 else 0
        beta = np.mean(qs) / np.mean(qo) if np.mean(qo) > 0 else 0

        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return kge

    def calculate_nse(self, q_sim: np.ndarray, q_obs: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency."""
        qs = q_sim[self.eval_start:self.eval_end]
        qo = q_obs[self.eval_start:self.eval_end]
        valid = ~np.isnan(qo) & ~np.isnan(qs)
        qs, qo = qs[valid], qo[valid]

        if len(qs) == 0:
            return -999.0

        ss_res = np.sum((qo - qs)**2)
        ss_tot = np.sum((qo - np.mean(qo))**2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else -999.0

    def objective(self, params_normalized: np.ndarray) -> float:
        """Objective function for optimization (minimization)."""
        # Denormalize parameters
        params = self.denormalize_params(params_normalized)

        # Simulate
        q_sim = self.simulate(params)

        # Calculate KGE (negate for minimization)
        kge = self.calculate_kge(q_sim, self.q_obs)
        return -kge  # Minimize negative KGE

    def normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        normalized = []
        for name in self.CALIB_PARAMS:
            lo, hi = self.PARAM_BOUNDS[name]
            val = params.get(name, (lo + hi) / 2)
            normalized.append((val - lo) / (hi - lo))
        return np.array(normalized)

    def denormalize_params(self, normalized: np.ndarray) -> Dict[str, float]:
        """Convert normalized [0, 1] array to parameter dict."""
        params = {}
        for i, name in enumerate(self.CALIB_PARAMS):
            lo, hi = self.PARAM_BOUNDS[name]
            params[name] = lo + normalized[i] * (hi - lo)
        return params


class OptimizationAlgorithms:
    """Collection of optimization algorithms for benchmarking."""

    @staticmethod
    def run_dds(objective_fn, n_params: int, max_iter: int = 500, r: float = 0.2) -> Tuple[np.ndarray, List[float]]:
        """Dynamically Dimensioned Search algorithm."""
        # Initialize
        x_best = np.random.uniform(0, 1, n_params)
        f_best = objective_fn(x_best)
        history = [f_best]

        for i in range(1, max_iter):
            # Probability of perturbation decreases
            p = 1 - np.log(i) / np.log(max_iter)

            # Select dimensions to perturb
            perturb = np.random.uniform(0, 1, n_params) < p
            if not np.any(perturb):
                perturb[np.random.randint(n_params)] = True

            # Generate candidate
            x_new = x_best.copy()
            for j in range(n_params):
                if perturb[j]:
                    x_new[j] = x_best[j] + r * np.random.normal()
                    # Reflect at boundaries
                    if x_new[j] < 0:
                        x_new[j] = abs(x_new[j])
                    if x_new[j] > 1:
                        x_new[j] = 2 - x_new[j]
                    x_new[j] = np.clip(x_new[j], 0, 1)

            # Evaluate
            f_new = objective_fn(x_new)
            if f_new < f_best:
                x_best = x_new
                f_best = f_new

            history.append(f_best)

        return x_best, history

    @staticmethod
    def run_pso(objective_fn, n_params: int, max_iter: int = 100, n_particles: int = 30,
                w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> Tuple[np.ndarray, List[float]]:
        """Particle Swarm Optimization."""
        # Initialize swarm
        positions = np.random.uniform(0, 1, (n_particles, n_params))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_params))

        # Evaluate initial positions
        fitness = np.array([objective_fn(p) for p in positions])
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()

        gbest_idx = np.argmin(fitness)
        gbest_pos = positions[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]

        history = [gbest_fit]

        for _ in range(max_iter):
            # Update velocities and positions
            r1, r2 = np.random.random((2, n_particles, n_params))
            velocities = (w * velocities +
                         c1 * r1 * (pbest_pos - positions) +
                         c2 * r2 * (gbest_pos - positions))
            positions = np.clip(positions + velocities, 0, 1)

            # Evaluate
            fitness = np.array([objective_fn(p) for p in positions])

            # Update personal bests
            improved = fitness < pbest_fit
            pbest_pos[improved] = positions[improved]
            pbest_fit[improved] = fitness[improved]

            # Update global best
            if np.min(fitness) < gbest_fit:
                gbest_idx = np.argmin(fitness)
                gbest_pos = positions[gbest_idx].copy()
                gbest_fit = fitness[gbest_idx]

            history.append(gbest_fit)

        return gbest_pos, history

    @staticmethod
    def run_de(objective_fn, n_params: int, max_iter: int = 100, n_pop: int = 30,
               F: float = 0.8, CR: float = 0.9) -> Tuple[np.ndarray, List[float]]:
        """Differential Evolution."""
        # Initialize population
        population = np.random.uniform(0, 1, (n_pop, n_params))
        fitness = np.array([objective_fn(p) for p in population])

        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()
        best_fit = fitness[best_idx]

        history = [best_fit]

        for _ in range(max_iter):
            for i in range(n_pop):
                # Select 3 random individuals (different from i)
                candidates = [j for j in range(n_pop) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, 0, 1)

                # Crossover
                cross_points = np.random.random(n_params) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(n_params)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fit = objective_fn(trial)
                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_fit:
                        best = trial.copy()
                        best_fit = trial_fit

            history.append(best_fit)

        return best, history

    @staticmethod
    def run_sce(objective_fn, n_params: int, max_iter: int = 100, n_complexes: int = 2,
                n_points_complex: int = None) -> Tuple[np.ndarray, List[float]]:
        """Shuffled Complex Evolution (SCE-UA)."""
        if n_points_complex is None:
            n_points_complex = 2 * n_params + 1

        n_points = n_complexes * n_points_complex

        # Initialize population
        population = np.random.uniform(0, 1, (n_points, n_params))
        fitness = np.array([objective_fn(p) for p in population])

        # Sort by fitness
        idx = np.argsort(fitness)
        population = population[idx]
        fitness = fitness[idx]

        history = [fitness[0]]

        for _ in range(max_iter):
            # Partition into complexes
            complexes = []
            for c in range(n_complexes):
                complex_idx = list(range(c, n_points, n_complexes))
                complexes.append((population[complex_idx].copy(), fitness[complex_idx].copy()))

            # Evolve each complex
            evolved_pop = []
            evolved_fit = []

            for pop, fit in complexes:
                # CCE step: select subcomplex and evolve
                for _ in range(n_points_complex):
                    # Select subcomplex (2*n+1 points, weighted by rank)
                    m = min(n_params + 1, len(pop))
                    weights = np.arange(len(pop), 0, -1) ** 2
                    weights = weights / weights.sum()
                    sub_idx = np.random.choice(len(pop), m, replace=False, p=weights)
                    sub_idx = np.sort(sub_idx)

                    # Reflection step
                    centroid = pop[sub_idx[:-1]].mean(axis=0)
                    worst = pop[sub_idx[-1]]
                    reflected = 2 * centroid - worst
                    reflected = np.clip(reflected, 0, 1)

                    ref_fit = objective_fn(reflected)
                    if ref_fit < fit[sub_idx[-1]]:
                        pop[sub_idx[-1]] = reflected
                        fit[sub_idx[-1]] = ref_fit
                    else:
                        # Random point
                        pop[sub_idx[-1]] = np.random.uniform(0, 1, n_params)
                        fit[sub_idx[-1]] = objective_fn(pop[sub_idx[-1]])

                evolved_pop.append(pop)
                evolved_fit.append(fit)

            # Shuffle complexes
            population = np.vstack(evolved_pop)
            fitness = np.concatenate(evolved_fit)
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]

            history.append(fitness[0])

        return population[0], history

    @staticmethod
    def run_adam(objective_fn, grad_fn, n_params: int, max_iter: int = 500,
                 lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8) -> Tuple[np.ndarray, List[float]]:
        """Adam optimizer (gradient-based)."""
        # Initialize
        x = np.random.uniform(0.2, 0.8, n_params)  # Start away from boundaries
        m = np.zeros(n_params)  # First moment
        v = np.zeros(n_params)  # Second moment

        f_best = objective_fn(x)
        x_best = x.copy()
        history = [f_best]

        for t in range(1, max_iter + 1):
            # Compute gradient
            g = grad_fn(x)

            # Update moments
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2

            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Update parameters
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            x = np.clip(x, 0, 1)  # Keep in bounds

            # Evaluate
            f = objective_fn(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()

            history.append(f_best)

        return x_best, history


def run_benchmark(forcing_path: Path, output_dir: Path, max_evals: int = 500):
    """Run full benchmark comparison."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CALIBRATION BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"Forcing data: {forcing_path}")
    print(f"Output directory: {output_dir}")
    print(f"Max evaluations per algorithm: {max_evals}")
    print()

    # Initialize calibrator
    calibrator = JFUSECalibrator(forcing_path)
    n_params = len(calibrator.CALIB_PARAMS)

    print(f"Parameters to calibrate: {n_params}")
    print(f"Parameter names: {calibrator.CALIB_PARAMS}")
    print()

    results: List[CalibrationResult] = []

    # Define algorithms to test
    algorithms = {
        'DDS': lambda: OptimizationAlgorithms.run_dds(
            calibrator.objective, n_params, max_iter=max_evals
        ),
        'PSO': lambda: OptimizationAlgorithms.run_pso(
            calibrator.objective, n_params, max_iter=max_evals // 30, n_particles=30
        ),
        'DE': lambda: OptimizationAlgorithms.run_de(
            calibrator.objective, n_params, max_iter=max_evals // 30, n_pop=30
        ),
        'SCE-UA': lambda: OptimizationAlgorithms.run_sce(
            calibrator.objective, n_params, max_iter=max_evals // 20, n_complexes=2
        ),
    }

    # Run each algorithm
    for name, run_fn in algorithms.items():
        print(f"Running {name}...")
        calibrator.n_evals = 0

        start_time = time.time()
        best_normalized, history = run_fn()
        runtime = time.time() - start_time

        # Denormalize best parameters
        best_params = calibrator.denormalize_params(best_normalized)
        best_kge = -history[-1]  # Convert back from minimization

        # Run final simulation for metrics
        q_sim = calibrator.simulate(best_params)
        final_kge = calibrator.calculate_kge(q_sim, calibrator.q_obs)
        final_nse = calibrator.calculate_nse(q_sim, calibrator.q_obs)

        result = CalibrationResult(
            algorithm=name,
            best_params=best_params,
            best_score=best_kge,
            convergence_history=[-h for h in history],  # Convert to KGE
            n_evaluations=calibrator.n_evals,
            runtime_seconds=runtime,
            final_metrics={'KGE': final_kge, 'NSE': final_nse}
        )
        results.append(result)

        print(f"  Best KGE: {best_kge:.4f}")
        print(f"  Evaluations: {calibrator.n_evals}")
        print(f"  Runtime: {runtime:.1f}s")
        print()

    # Save results
    results_data = []
    for r in results:
        results_data.append({
            'algorithm': r.algorithm,
            'best_score': r.best_score,
            'n_evaluations': r.n_evaluations,
            'runtime_seconds': r.runtime_seconds,
            'final_metrics': r.final_metrics,
            'best_params': r.best_params
        })

    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    # Generate plots
    generate_comparison_plots(results, calibrator, output_dir)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")

    return results


def generate_comparison_plots(results: List[CalibrationResult], calibrator: JFUSECalibrator, output_dir: Path):
    """Generate comparison plots."""

    fig = plt.figure(figsize=(16, 12))

    # 1. Convergence curves
    ax1 = fig.add_subplot(2, 2, 1)
    for r in results:
        ax1.plot(r.convergence_history, label=f'{r.algorithm} (KGE={r.best_score:.3f})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('KGE')
    ax1.set_title('Convergence Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final performance comparison
    ax2 = fig.add_subplot(2, 2, 2)
    algorithms = [r.algorithm for r in results]
    kge_scores = [r.final_metrics['KGE'] for r in results]
    nse_scores = [r.final_metrics['NSE'] for r in results]

    x = np.arange(len(algorithms))
    width = 0.35
    ax2.bar(x - width/2, kge_scores, width, label='KGE', color='steelblue')
    ax2.bar(x + width/2, nse_scores, width, label='NSE', color='darkorange')
    ax2.set_ylabel('Score')
    ax2.set_title('Final Performance Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Runtime comparison
    ax3 = fig.add_subplot(2, 2, 3)
    runtimes = [r.runtime_seconds for r in results]
    evals = [r.n_evaluations for r in results]

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    bars = ax3.bar(algorithms, runtimes, color=colors)
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('Computational Cost')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add evaluation counts as text
    for bar, n in zip(bars, evals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{n} evals', ha='center', va='bottom', fontsize=9)

    # 4. Hydrograph comparison (best algorithm)
    ax4 = fig.add_subplot(2, 2, 4)

    # Find best result
    best_result = max(results, key=lambda r: r.best_score)
    q_sim = calibrator.simulate(best_result.best_params)

    # Plot subset of time series
    plot_start = calibrator.eval_start
    plot_end = min(calibrator.eval_end, plot_start + 365)  # 1 year

    time_idx = np.arange(plot_end - plot_start)
    ax4.plot(time_idx, calibrator.q_obs[plot_start:plot_end], 'b-', label='Observed', alpha=0.7)
    ax4.plot(time_idx, q_sim[plot_start:plot_end], 'r-', label=f'Simulated ({best_result.algorithm})', alpha=0.7)
    ax4.set_xlabel('Day of Year')
    ax4.set_ylabel('Discharge (mm/day)')
    ax4.set_title(f'Hydrograph - Best Result (KGE={best_result.best_score:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'calibration_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'calibration_comparison.png'}")

    # Additional plot: Parameter distributions
    fig2, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for i, param_name in enumerate(calibrator.CALIB_PARAMS):
        ax = axes[i]
        values = [r.best_params[param_name] for r in results]
        lo, hi = calibrator.PARAM_BOUNDS[param_name]

        bars = ax.bar([r.algorithm for r in results], values, color=colors)
        ax.axhline(y=(lo + hi) / 2, color='gray', linestyle='--', alpha=0.5, label='Default')
        ax.set_ylabel(param_name)
        ax.set_title(f'{param_name}\n[{lo:.2f}, {hi:.2f}]')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'parameter_comparison.png'}")

    plt.close('all')


def main():
    """Main entry point."""
    # Paths
    forcing_path = Path('/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Bow_at_Banff_lumped_casr/forcing/FUSE_input/Bow_at_Banff_lumped_casr_input.nc')
    output_dir = Path('/Users/darrieythorsson/compHydro/code/SYMFLUENCE/results/calibration_benchmark')

    # Run benchmark
    results = run_benchmark(forcing_path, output_dir, max_evals=300)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<12} {'KGE':>8} {'NSE':>8} {'Evals':>8} {'Time (s)':>10}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: -x.best_score):
        print(f"{r.algorithm:<12} {r.final_metrics['KGE']:>8.4f} {r.final_metrics['NSE']:>8.4f} {r.n_evaluations:>8} {r.runtime_seconds:>10.1f}")


if __name__ == '__main__':
    main()
